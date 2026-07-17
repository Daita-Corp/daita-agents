# ADR 0005: State, events, and crash consistency

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- Persist authoritative current-state records plus an append-only canonical
  runtime event log; do not require full event sourcing.
- Commit a lifecycle mutation and its corresponding event atomically whenever
  they describe one transition.
- Publish events to subscribers only after durable commit. Telemetry consumes
  committed events and is never required for a state commit.
- Persist the operation/trigger, context manifest/turn, normalized model
  response, task, lease, evidence/observation, readiness decision, and terminal
  state at the checkpoints defined by the architecture plan.
- Treat remote model calls as at-least-once at the network boundary. Protect
  side effects separately with tasks, fencing, idempotency, and manual recovery.
- Keep raw provider payload retention separate from normalized resume state.

## Consequences

Repository contracts remain lifecycle-specific rather than becoming one
unbounded `StateStore`. Failure-injection tests are required at each checkpoint.

## Phase 2 implementation binding

- SQLite assigns each committed runtime event a positive, contiguous sequence
  scoped to its agent inside the same `BEGIN IMMEDIATE` transaction that
  persists lifecycle state. The existing operation transaction remains the
  sole writer; consumers receive no append API or duplicate outbox.
- Migration backfill uses legacy insertion (`rowid`) order, not timestamps,
  and the resulting log rejects update and delete operations. Durable cursors
  are agent-bound and readers reject cross-agent or nonexistent positions.
- In-process subscription signals are post-commit wake hints only. Bounded
  replay is authoritative, with a clear-then-reread wait boundary and polling
  to recover from missed, failed, cross-store, and future cross-process hints.
- Local blobs separate immutable content identity from logical provenance.
  Fsynced content becomes visible before a checksummed, fsynced logical
  manifest is atomically published as the commit point; retries stabilize an
  already-visible result before reporting success.
