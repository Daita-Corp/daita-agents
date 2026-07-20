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
- `next/` provides a thin CLI against `AgentHost` contracts. ADR 0016
  supersedes the original external-integration clause: `daita-agents` owns the
  Daita 2.0 console entry point, while `daita-cli` and `daita-client` remain
  excluded legacy packages. Neither package may become the local runtime owner.
- Persist secret references only. Resolution supports an injected in-process
  provider, an OS keychain where available, then environment variables.
- No encrypted local secret fallback is added without a separate security
  design review. Secret values never enter config, SQLite, events, artifacts,
  diagnostics, or model context.

## Consequences

The parity matrix is the authoritative migration surface. Host/SDK behavior is
stable across internal CLI organization; distribution and entry-point
ownership are fixed by ADR 0016.
