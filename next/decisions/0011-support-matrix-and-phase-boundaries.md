# ADR 0011: Support matrix and phase boundaries

- **Status:** Accepted
- **Date:** 2026-07-16

## Decision

- Test and support CPython 3.11 and 3.12 for the replacement candidate. Declare
  `requires-python = ">=3.11"`; newer versions are not claimed until tested.
- Keep the minimal core dependency set small. Every source/provider-specific
  SDK belongs to a matching optional extra and follows lazy imports.
- The architecture MVP requires mock plus OpenAI, SQLite, sandboxed CSV/JSON,
  the fake durable side effect, memory/skills, host/monitor, and restart proof.
- PostgreSQL and one controlled real write are replacement-cutover blockers.
- Retain OpenAI, Anthropic, Gemini, Grok, Ollama, and supported
  OpenAI-compatible endpoints for the proposed 2.0 matrix. Each must pass the
  Phase 8 conformance suite or receive an explicit documented disposition
  before the Phase 9 gate; no fallback may hide a failure.
- The Phase 2 fake side effect mutates only a test-owned durable marker to prove
  approval/recovery. It does not satisfy the Phase 7 controlled real-write gate.
- Google Drive, object stores, additional databases/warehouses, cloud inventory,
  rich documents, arbitrary event triggers, vector acceleration, multi-agent
  delegation, and managed-cloud infrastructure are post-MVP unless the parity
  matrix records a required cutover disposition.
- Phase 0 through Phase 9 produce a candidate. Phase 10 is a separate,
  destructive cutover requiring explicit human approval.

## Consequences

Architecture-MVP and replacement-candidate gates are tracked separately.
Nothing in this ADR authorizes removal or modification of root `daita/`.
