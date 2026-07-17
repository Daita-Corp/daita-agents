# ADR 0010: Public API, CLI ownership, and secrets

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- The target local API centers on async `Agent.create(...)`, `Agent.open(...)`,
  source attach/detach, `run(...)`, `stream(...)`, inspection, approval,
  rejection, cancellation, and resume.
- `Agent` remains a thin facade over owning services; it does not contain the
  loop, source clients, catalog discovery, memory writes, policy, monitor
  execution, or provider translation.
- V1 exports are dispositions to assess, not signatures to reproduce through a
  fallback. Breaking changes must be explicit in the parity matrix and release
  documentation.
- `next/` provides a thin development CLI by Phase 6 against `AgentHost`
  contracts. Production integration of external `daita-cli` and `daita-client`
  is a Phase 9 gate; neither package may become the local runtime owner.
- Persist secret references only. Resolution supports an injected in-process
  provider, an OS keychain where available, then environment variables.
- No encrypted local secret fallback is added without a separate security
  design review. Secret values never enter config, SQLite, events, artifacts,
  diagnostics, or model context.

## Consequences

The parity matrix is the authoritative migration surface. Host/SDK behavior is
stable even if command parsing later moves to an external package.
