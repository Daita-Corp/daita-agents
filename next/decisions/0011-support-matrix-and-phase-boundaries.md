# ADR 0011: Support matrix and phase boundaries

- **Status:** Accepted
- **Date:** 2026-07-16
- **Superseded in part by:** ADR 0017 for replacement-readiness timing after
  Phase 9

## Decision

- Test and support CPython 3.11 and 3.12 for the replacement candidate. Declare
  `requires-python = ">=3.11"`; newer versions are not claimed until tested.
- Keep the minimal core dependency set small. Every source/provider-specific
  SDK belongs to a matching optional extra and follows lazy imports.
- The architecture MVP requires mock plus OpenAI, SQLite, sandboxed CSV/JSON,
  the fake durable side effect, memory/skills, host/monitor, and restart proof.
- PostgreSQL and one controlled real write are replacement-cutover blockers.
- Phase 8 PostgreSQL query support is deliberately conservative: discovery
  advertises attached base tables only, omits views and foreign tables, and
  omits a table when any column uses a custom/extension or unsupported native
  type. Queries require schema-qualified catalog identities, execute each base
  relation as `ONLY`, and use a transaction-local `pg_catalog` search path.
  The attached database and login role remain an explicit trust boundary for
  row-security policies and other server-owned semantics. PostgreSQL reads are
  non-idempotent and non-replay-safe until durable callable/operator/policy/type
  provenance exists; an ambiguous interrupted attempt requires manual recovery.
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
- Phase 0 through Phase 9 produce the hardened component candidate. ADR 0017
  adds a mandatory Phase 9.5 joined-product gate before the candidate is
  replacement-ready. Phase 10 remains a separate, destructive cutover
  requiring explicit human approval.
- Fake-driver conformance does not satisfy the PostgreSQL cutover blocker by
  itself. The actual asyncpg/base-table, codec, cancellation, wrapper-size, and
  trusted-role contract must pass the credential-gated Phase 9 live suite.

## Consequences

Architecture-MVP and replacement-candidate gates are tracked separately.
Nothing in this ADR authorizes removal or modification of root `daita/`.
