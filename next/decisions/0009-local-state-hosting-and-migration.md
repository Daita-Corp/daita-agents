# ADR 0009: Local state, hosting, and migration

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- Before cutover, v2 defaults to `~/.daita-next/`; tests and embedding callers
  may supply an explicit root. V2 refuses to open v1 state implicitly.
- Each agent has a stable directory. `state.db` is authoritative mutable state;
  `agent.toml` is a bootstrap manifest; top-level config stores installation
  defaults and pointers only.
- One local `AgentHost` owns the per-agent writer lock, migrations, scheduler,
  recovery, trigger inbox, and event streams.
- Embedded mutations acquire the same lock. If a host owns it, callers route
  through the host API when configured or fail clearly with `host_active`.
- Default to fresh v2 state because v1 and v2 schemas/lifecycles differ. Phase 9
  must implement and test the fresh-start behavior.
- A one-time, backup-first, explicitly invoked migration may be added only if
  concrete retention requirements are documented. Permanent dual readers,
  lazy migration, and normal-runtime dual writes are forbidden.

## Consequences

SQLite uses WAL, foreign keys, busy timeout, ordered migrations, compatibility
checks, and backup-before-migrate. Phase 10 must still obtain explicit approval
before any state or source cutover.
