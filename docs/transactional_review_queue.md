# Transactional fraud-review leases

The in-memory `ReviewQueue` defines bounded admission and starvation-safe ordering, but removing a
case from one process is not sufficient ownership evidence. A crash can lose the case, two replicas
can assign it twice, and a retried completion can create conflicting review outcomes.

`durable_review_queue.SQLiteReviewQueue` preserves that scheduling contract behind a transactional
local database:

- enqueue is durable and capacity is checked in the same write transaction;
- an exact replay of the same case evidence is idempotent, while the same `case_id` with different
  evidence fails closed;
- every claim is a `BEGIN IMMEDIATE` transaction, so concurrent processes sharing the file cannot
  lease the same case;
- pre-SLA cases retain materiality ordering, while overdue cases use earliest deadline first;
- a claim returns a random bearer token while only its SHA-256 digest is stored;
- leases have a bounded duration, one owner, explicit renewal, and deterministic expiry reclaim;
- completion requires the active owner and token, and a globally unique `completion_id` makes an
  exact client retry idempotent; and
- pressure snapshots expose counts, utilization, overdue work and expired leases without case or
  customer identifiers.

## Example

The database and its immediate parent must be owned by the service user and have no group or world
permissions because the reference schema retains synthetic case evidence.

```python
from datetime import UTC, datetime
from pathlib import Path

from case_triage import FraudSignal
from durable_review_queue import SQLiteReviewQueue

queue = SQLiteReviewQueue(Path("/run/fraud-review/review.sqlite3"))
queue.enqueue(
    FraudSignal(
        case_id="case-2026-0001",
        risk_score=0.91,
        confidence=0.82,
        customer_value=4200.0,
        amount=850.0,
        evidence_count=4,
    )
)

lease = queue.claim_next("analyst-7", now=datetime.now(UTC))
# Review work occurs outside the database transaction.
queue.complete(
    lease,
    completion_id="review-command-7d36c4",
    completed_at=datetime.now(UTC),
)
```

The token in `ReviewLease` is a secret capability. Do not log it or place it in URLs. A worker that
needs more time must renew before expiry; an expired token cannot complete a case after another
worker has reclaimed it.

## Transaction boundary

SQLite write transactions are serialized with `BEGIN IMMEDIATE`. Claim selection and lease creation
therefore commit together. Capacity checks and inserts also commit together, so concurrent producers
cannot overrun `ReviewPolicy.max_cases`. WAL mode permits readers during writes, `synchronous=FULL`
prioritizes durability, and a bounded busy timeout prevents indefinite lock waits.

Completed rows remain in the database as idempotency evidence and do not consume active capacity.
Reusing a completed `case_id` is intentionally rejected: upstream producers should issue a new case
identity for materially new evidence.

## Trust boundary and limitations

This is a dependency-free local persistence and multi-process concurrency reference, not a
multi-node queue. All workers must share one filesystem with correct SQLite locking semantics. NFS
and other filesystems with weak advisory locking or durability guarantees are outside the supported
boundary. The implementation trusts the host clock, does not authenticate worker identities, does
not encrypt retained case evidence, and does not append a separate immutable review history.

Production deployment should move the same state machine to PostgreSQL with row-level locking or a
durable workflow engine, authenticate workers with workload identity, encrypt sensitive fields, and
emit an append-only decision audit. Lease expiry limits abandoned ownership; it cannot guarantee
that an analyst finishes within the review SLA or prevent side effects performed outside the queue.
