# ADR 0004: Sole executor boundary and recovery semantics

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- Only the operation runtime may invoke `Executor.execute(...)`.
- Every capability action is schema/scope/identity validated and persisted as
  a task before executor I/O.
- The runtime owns governance, approvals, dependencies, idempotency, timeouts,
  fenced leases, evidence acceptance, and terminal task commits.
- Terminal tasks never rerun. Expired replay-safe reads may be reclaimed; a
  stale lease holder may not commit.
- Unknown side-effect outcomes become `manual_recovery_required`; they are not
  guessed successful or automatically retried.
- Approval channels mutate approval state only. The owning runtime resumes the
  same operation and skips completed work.
- Catalog-controlled `discover()` and `inspect()` calls are bounded source
  workflows, not capability execution. All query/read/write/compare actions
  use persisted tasks and executors.

## Consequences

Loops, domain controllers, skills, monitors, providers, adapters, and the agent
facade cannot call executors directly. Architecture scans and tests enforce
the single invocation site.
