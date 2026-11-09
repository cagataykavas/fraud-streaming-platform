from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from case_triage import FraudSignal, ReviewPolicy, _utc, triage_priority

SCHEMA_VERSION = "1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,127}$")


class DurableQueueError(RuntimeError):
    """The durable review queue cannot safely complete an operation."""


class QueueEmpty(DurableQueueError):
    """No unleased or reclaimable case is available."""


class LeaseConflict(DurableQueueError):
    """The caller does not hold the active lease for a case."""


@dataclass(frozen=True)
class DurableReviewPolicy:
    review: ReviewPolicy = field(default_factory=ReviewPolicy)
    default_lease: timedelta = timedelta(minutes=10)
    max_lease: timedelta = timedelta(hours=1)
    busy_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.review, ReviewPolicy):
            raise TypeError("review must be a ReviewPolicy")
        for name in ("default_lease", "max_lease"):
            value = getattr(self, name)
            if not isinstance(value, timedelta):
                raise TypeError(f"{name} must be a timedelta")
            if value <= timedelta(0):
                raise ValueError(f"{name} must be positive")
        if self.default_lease > self.max_lease:
            raise ValueError("default_lease cannot exceed max_lease")
        if isinstance(self.busy_timeout_seconds, bool) or not isinstance(
            self.busy_timeout_seconds, (int, float)
        ):
            raise TypeError("busy_timeout_seconds must be numeric")
        if not 0 < self.busy_timeout_seconds <= 60:
            raise ValueError("busy_timeout_seconds must be in (0, 60]")


@dataclass(frozen=True)
class ReviewLease:
    signal: FraudSignal
    worker_id: str
    token: str = field(repr=False)
    leased_at: datetime
    expires_at: datetime
    version: int


@dataclass(frozen=True)
class DurableQueueSnapshot:
    active_cases: int
    queued_cases: int
    leased_cases: int
    completed_cases: int
    expired_leases: int
    overdue_cases: int
    capacity: int
    utilization: float
    oldest_age_seconds: float
    earliest_deadline: datetime | None


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} must be a bounded identifier")
    return value


def _timestamp(value: datetime) -> str:
    return _utc(value, field="timestamp").isoformat(timespec="microseconds")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:  # pragma: no cover - DB invariant
        raise DurableQueueError("database contains a naive timestamp")
    return parsed.astimezone(timezone.utc)


def _signal_payload(signal: FraudSignal) -> dict[str, object]:
    triage_priority(signal)
    return {
        "case_id": signal.case_id,
        "risk_score": float(signal.risk_score),
        "confidence": float(signal.confidence),
        "customer_value": float(signal.customer_value),
        "amount": float(signal.amount),
        "evidence_count": signal.evidence_count,
    }


def _signal_digest(signal: FraudSignal) -> str:
    encoded = json.dumps(
        _signal_payload(signal),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _signal_from_row(row: sqlite3.Row) -> FraudSignal:
    return FraudSignal(
        case_id=row["case_id"],
        risk_score=row["risk_score"],
        confidence=row["confidence"],
        customer_value=row["customer_value"],
        amount=row["amount"],
        evidence_count=row["evidence_count"],
    )


class SQLiteReviewQueue:
    """Transactional review queue with renewable single-owner analyst leases.

    SQLite is used as a local durability and concurrency reference. Every write
    begins with ``BEGIN IMMEDIATE`` so capacity checks, claims, renewals, and
    completions are serialized across processes sharing one database file.
    """

    def __init__(
        self,
        path: Path,
        policy: DurableReviewPolicy | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self.policy = policy or DurableReviewPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._prepare_private_file()
        self._initialize_schema()

    def _prepare_private_file(self) -> None:
        parent = self.path.parent
        try:
            parent_stat = parent.lstat()
        except OSError as error:
            raise DurableQueueError("database parent is not accessible") from error
        if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
            raise DurableQueueError("database parent must be a real directory")
        if parent_stat.st_uid != os.getuid() or parent_stat.st_mode & 0o077:
            raise DurableQueueError("database parent must be owner-only")

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            descriptor = None
        except OSError as error:
            raise DurableQueueError("database file cannot be created safely") from error
        if descriptor is not None:
            os.close(descriptor)

        try:
            file_stat = self.path.lstat()
        except OSError as error:
            raise DurableQueueError("database file is not accessible") from error
        if (
            stat.S_ISLNK(file_stat.st_mode)
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.getuid()
            or file_stat.st_nlink != 1
            or file_stat.st_mode & 0o077
        ):
            raise DurableQueueError("database must be an owner-only, single-link regular file")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            isolation_level=None,
            timeout=self.policy.busy_timeout_seconds,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.policy.busy_timeout_seconds * 1000)}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
        finally:
            connection.close()
        with self._transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS queue_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_cases (
                    case_id TEXT PRIMARY KEY,
                    signal_sha256 TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    customer_value REAL NOT NULL,
                    amount REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    priority REAL NOT NULL,
                    enqueued_at TEXT NOT NULL,
                    deadline_at TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('queued', 'leased', 'completed')),
                    lease_owner TEXT,
                    lease_token_sha256 TEXT,
                    lease_expires_at TEXT,
                    completed_at TEXT,
                    completion_id TEXT UNIQUE,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    CHECK (length(signal_sha256) = 64),
                    CHECK (risk_score BETWEEN 0 AND 1),
                    CHECK (confidence BETWEEN 0 AND 1),
                    CHECK (customer_value >= 0),
                    CHECK (amount >= 0),
                    CHECK (evidence_count >= 0)
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS review_cases_claim
                ON review_cases(state, deadline_at, priority DESC, enqueued_at, case_id)
                """
            )
            row = connection.execute(
                "SELECT value FROM queue_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO queue_metadata(key, value) VALUES ('schema_version', ?)",
                    (SCHEMA_VERSION,),
                )
            elif row["value"] != SCHEMA_VERSION:
                raise DurableQueueError("unsupported durable queue schema version")

    def enqueue(
        self,
        signal: FraudSignal,
        *,
        enqueued_at: datetime | None = None,
    ) -> bool:
        priority = triage_priority(signal)
        accepted_at = _utc(self._clock(), field="clock")
        queued_at = accepted_at if enqueued_at is None else _utc(enqueued_at, field="enqueued_at")
        if queued_at > accepted_at and queued_at - accepted_at > self.policy.review.max_future_skew:
            raise ValueError("enqueued_at exceeds the allowed future clock skew")
        try:
            deadline = queued_at + self.policy.review.review_sla
        except OverflowError as error:
            raise ValueError("review_sla produces an invalid deadline") from error
        signal_sha256 = _signal_digest(signal)
        payload = _signal_payload(signal)

        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT signal_sha256 FROM review_cases WHERE case_id = ?",
                (signal.case_id,),
            ).fetchone()
            if existing is not None:
                if hmac.compare_digest(existing["signal_sha256"], signal_sha256):
                    return False
                raise DurableQueueError("case_id already identifies different evidence")
            active = connection.execute(
                "SELECT COUNT(*) AS count FROM review_cases WHERE state != 'completed'"
            ).fetchone()["count"]
            if active >= self.policy.review.max_cases:
                raise OverflowError("review queue capacity is exhausted")
            connection.execute(
                """
                INSERT INTO review_cases(
                    case_id, signal_sha256, risk_score, confidence, customer_value,
                    amount, evidence_count, priority, enqueued_at, deadline_at,
                    state, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 1)
                """,
                (
                    payload["case_id"],
                    signal_sha256,
                    payload["risk_score"],
                    payload["confidence"],
                    payload["customer_value"],
                    payload["amount"],
                    payload["evidence_count"],
                    priority,
                    _timestamp(queued_at),
                    _timestamp(deadline),
                ),
            )
        return True

    def claim_next(
        self,
        worker_id: str,
        *,
        now: datetime | None = None,
        lease_duration: timedelta | None = None,
    ) -> ReviewLease:
        worker = _identifier(worker_id, "worker_id")
        observed_at = _utc(self._clock() if now is None else now, field="now")
        duration = self.policy.default_lease if lease_duration is None else lease_duration
        self._validate_lease_duration(duration)
        try:
            expires_at = observed_at + duration
        except OverflowError as error:
            raise ValueError("lease duration produces an invalid expiry") from error
        now_text = _timestamp(observed_at)

        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM review_cases
                WHERE state = 'queued'
                   OR (state = 'leased' AND lease_expires_at <= ?)
                ORDER BY
                    CASE WHEN deadline_at <= ? THEN 0 ELSE 1 END ASC,
                    CASE WHEN deadline_at <= ? THEN deadline_at END ASC,
                    CASE WHEN deadline_at <= ? THEN priority END DESC,
                    CASE WHEN deadline_at > ? THEN priority END DESC,
                    CASE WHEN deadline_at > ? THEN enqueued_at END ASC,
                    case_id ASC
                LIMIT 1
                """,
                (now_text, now_text, now_text, now_text, now_text, now_text),
            ).fetchone()
            if row is None:
                raise QueueEmpty("no review case is currently claimable")
            token = secrets.token_urlsafe(32)
            next_version = row["version"] + 1
            updated = connection.execute(
                """
                UPDATE review_cases
                SET state = 'leased', lease_owner = ?, lease_token_sha256 = ?,
                    lease_expires_at = ?, version = ?
                WHERE case_id = ? AND version = ?
                """,
                (
                    worker,
                    _token_digest(token),
                    _timestamp(expires_at),
                    next_version,
                    row["case_id"],
                    row["version"],
                ),
            )
            if updated.rowcount != 1:  # pragma: no cover - serialized transaction invariant
                raise DurableQueueError("case claim lost its transaction lease")
            return ReviewLease(
                signal=_signal_from_row(row),
                worker_id=worker,
                token=token,
                leased_at=observed_at,
                expires_at=expires_at,
                version=next_version,
            )

    def renew(
        self,
        lease: ReviewLease,
        *,
        now: datetime | None = None,
        lease_duration: timedelta | None = None,
    ) -> ReviewLease:
        if not isinstance(lease, ReviewLease):
            raise TypeError("lease must be a ReviewLease")
        observed_at = _utc(self._clock() if now is None else now, field="now")
        duration = self.policy.default_lease if lease_duration is None else lease_duration
        self._validate_lease_duration(duration)

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?",
                (lease.signal.case_id,),
            ).fetchone()
            self._require_active_lease(row, lease.worker_id, lease.token, observed_at)
            current_expiry = _parse_timestamp(row["lease_expires_at"])
            try:
                expires_at = max(current_expiry, observed_at) + duration
            except OverflowError as error:
                raise ValueError("lease duration produces an invalid expiry") from error
            next_version = row["version"] + 1
            connection.execute(
                """
                UPDATE review_cases SET lease_expires_at = ?, version = ?
                WHERE case_id = ? AND version = ?
                """,
                (
                    _timestamp(expires_at),
                    next_version,
                    row["case_id"],
                    row["version"],
                ),
            )
            return ReviewLease(
                signal=_signal_from_row(row),
                worker_id=lease.worker_id,
                token=lease.token,
                leased_at=lease.leased_at,
                expires_at=expires_at,
                version=next_version,
            )

    def complete(
        self,
        lease: ReviewLease,
        *,
        completion_id: str,
        completed_at: datetime | None = None,
    ) -> bool:
        if not isinstance(lease, ReviewLease):
            raise TypeError("lease must be a ReviewLease")
        completion = _identifier(completion_id, "completion_id")
        observed_at = _utc(
            self._clock() if completed_at is None else completed_at,
            field="completed_at",
        )
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?",
                (lease.signal.case_id,),
            ).fetchone()
            if row is None:
                raise LeaseConflict("case does not exist")
            if row["state"] == "completed":
                if row["completion_id"] == completion:
                    return False
                raise LeaseConflict("case was completed with a different completion_id")
            self._require_active_lease(row, lease.worker_id, lease.token, observed_at)
            try:
                connection.execute(
                    """
                    UPDATE review_cases
                    SET state = 'completed', lease_token_sha256 = NULL,
                        lease_expires_at = NULL, completed_at = ?, completion_id = ?,
                        version = version + 1
                    WHERE case_id = ? AND version = ?
                    """,
                    (
                        _timestamp(observed_at),
                        completion,
                        row["case_id"],
                        row["version"],
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise LeaseConflict("completion_id already belongs to another case") from error
        return True

    def snapshot(self, *, now: datetime | None = None) -> DurableQueueSnapshot:
        observed_at = _utc(self._clock() if now is None else now, field="now")
        now_text = _timestamp(observed_at)
        connection = self._connect()
        try:
            counts = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN state = 'queued' THEN 1 ELSE 0 END) AS queued,
                    SUM(CASE WHEN state = 'leased' THEN 1 ELSE 0 END) AS leased,
                    SUM(CASE WHEN state = 'completed' THEN 1 ELSE 0 END) AS completed,
                    SUM(CASE WHEN state != 'completed' AND deadline_at <= ? THEN 1 ELSE 0 END) AS overdue,
                    SUM(CASE WHEN state = 'leased' AND lease_expires_at <= ? THEN 1 ELSE 0 END) AS expired,
                    MIN(CASE WHEN state != 'completed' THEN enqueued_at END) AS oldest,
                    MIN(CASE WHEN state != 'completed' THEN deadline_at END) AS earliest
                FROM review_cases
                """,
                (now_text, now_text),
            ).fetchone()
        finally:
            connection.close()
        queued = counts["queued"] or 0
        leased = counts["leased"] or 0
        active = queued + leased
        oldest = _parse_timestamp(counts["oldest"]) if counts["oldest"] else None
        earliest = _parse_timestamp(counts["earliest"]) if counts["earliest"] else None
        oldest_age = max(0.0, (observed_at - oldest).total_seconds()) if oldest else 0.0
        return DurableQueueSnapshot(
            active_cases=active,
            queued_cases=queued,
            leased_cases=leased,
            completed_cases=counts["completed"] or 0,
            expired_leases=counts["expired"] or 0,
            overdue_cases=counts["overdue"] or 0,
            capacity=self.policy.review.max_cases,
            utilization=round(active / self.policy.review.max_cases, 6),
            oldest_age_seconds=oldest_age,
            earliest_deadline=earliest,
        )

    def _validate_lease_duration(self, value: timedelta) -> None:
        if not isinstance(value, timedelta):
            raise TypeError("lease_duration must be a timedelta")
        if value <= timedelta(0) or value > self.policy.max_lease:
            raise ValueError("lease_duration must be positive and within max_lease")

    @staticmethod
    def _require_active_lease(
        row: sqlite3.Row | None,
        worker_id: str,
        token: str,
        observed_at: datetime,
    ) -> None:
        if row is None or row["state"] != "leased":
            raise LeaseConflict("case has no active lease")
        if row["lease_owner"] != worker_id or not hmac.compare_digest(
            row["lease_token_sha256"], _token_digest(token)
        ):
            raise LeaseConflict("lease owner or token does not match")
        if _parse_timestamp(row["lease_expires_at"]) <= observed_at:
            raise LeaseConflict("lease has expired")


__all__ = [
    "DurableQueueError",
    "DurableQueueSnapshot",
    "DurableReviewPolicy",
    "LeaseConflict",
    "QueueEmpty",
    "ReviewLease",
    "SQLiteReviewQueue",
]
