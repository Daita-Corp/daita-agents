# Breaking changes from Daita 1.x

Daita 2.0 is a replacement architecture, not an in-place internal refactor.
The parity matrix records every retained, replaced, deferred, or proposed-
removed public surface. The main user-visible changes are:

- `Agent.create(...)` and `Agent.open(...)` are asynchronous and operate on a
  durable local identity. `Agent.run(...)`, streaming, inspection, approvals,
  cancellation, resume, source attach, and source detach are thin operations
  over the same persistent runtime.
- Database work uses `Agent.attach(SQLiteSource(...))` or
  `Agent.attach(PostgreSQLSource(...))` and the ordinary agent loop. There is no
  `DbAgent`, `from_db()` subclass path, or alternate database runtime.
- Capabilities are explicit declarations. Model-visible tools never invoke
  executors directly, and the v1 plugin base-class hierarchy is not retained.
- Caller-configured capability-provider manifests compose additively with
  built-ins and bind Agent Home by exact ordered fingerprints. They must be
  supplied identically on reopen. Resource-adapter, backend-provider, and
  observer extension categories remain post-MVP rather than silently loading.
- Canonical model messages and operation state survive provider changes;
  provider-native request/response formats stop at adapters.
- Natural memory and skill learning use durable redaction-safe proposals and
  explicit version owners. Skills proposed by an ordinary operation do not
  self-activate; a reviewed `accept_skill_change(...)` call is required.
- Configuration is immutable and service-owned. Future-operation budgets and
  the default governance profile have one authoritative `state.db` binding;
  retry policy belongs to the model router and its immutable route identity;
  source credentials remain secret references.
- V2 uses a fresh state root and never reads v1 state implicitly.
- `daita-agents` now owns both the Daita 2.0 Python API and the installed
  `daita` command. The separately distributed `daita-cli` and `daita-client`
  remain legacy Daita 1.x products and are unsupported in a Daita 2.0
  environment.
- Local setup is host-native: create the agent, persist a non-secret model
  route with `model set`, attach sources, then run model-free `serve`. Chat and
  committed-event follow use JSON Lines; inspection and mutations use bounded
  strict JSON over the private Unix socket.
- Uninstall legacy `daita-cli` before installing Daita 2.0. Both distributions
  otherwise claim an executable named `daita`, and the active script can depend
  on installation order. Remove `daita-client` too unless the environment is
  deliberately retained for the v1 hosted API.
- Managed-cloud CLI commands and a remote SDK are not silently forwarded to
  the v1 HTTP contracts. They are deferred until a stable Daita 2.0 cloud API
  exists; the intended future command namespace is `daita cloud ...`.
- Global tracing singletons are replaced by audience-specific committed-event
  projections and an optional failure-isolated exporter observer. A concrete
  OTLP transport adapter is deferred.
- Unsupported v1 integrations do not fall back to v1 implementations. See the
  [support matrix](SUPPORT_MATRIX.md) for explicit deferrals.

The Phase 9 hardening gate and mandatory Phase 9.5 replacement-readiness gate
pass, making the candidate eligible for human Phase 10 review. Phase 10 may
approve final removal names and cutover mechanics, but it has not been
authorized or started. This document does not itself remove v1 or authorize
cutover.
