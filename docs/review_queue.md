# Starvation-safe fraud review queue

`case_triage.ReviewQueue` is an in-memory reference implementation for the
admission and ordering boundary in front of a human fraud-review team.

## Contract

- Signal inputs are finite and bounded where the domain has a natural bound:
  risk and confidence are in `[0, 1]`, monetary values are non-negative, and
  case identifiers are non-empty and at most 128 characters.
- The queue rejects a duplicate active `case_id`; it never creates two review
  tasks for the same case.
- `ReviewPolicy.max_cases` makes overload visible with `OverflowError` instead
  of allowing unbounded process memory growth.
- Before `review_sla` expires, the weighted risk/materiality score determines
  ordering. After expiry, the earliest deadline wins. Equal deadlines use the
  weighted priority and then deterministic identifiers.
- Time evidence must be timezone-aware. A bounded future-skew allowance
  rejects timestamps that could postpone a case indefinitely.
- `snapshot()` exposes current capacity use, overdue count, oldest age and the
  earliest deadline without publishing case or customer identifiers.

Push and pop are normally `O(log n)`. Lazy heap entries are periodically rebuilt
in `O(n)` so repeated dequeue/re-enqueue churn cannot bypass the configured
memory bound. Snapshot creation is also intentionally `O(n)` because it computes
exact pressure over the active queue; call it on a monitoring cadence, not in
the hot path.

## Operational boundary

This component is process-local and uses application wall-clock evidence. It
does not persist cases, lease work to analysts, authenticate callers, or
coordinate replicas. A production deployment should store the same contract
in a transactional queue, use a trusted clock, expose rejection/overdue
counters, and add lease expiry plus idempotent review completion. The queue
orders work; it does not claim that an analyst will complete work within the
configured SLA.
