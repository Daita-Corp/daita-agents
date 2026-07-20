# Security model

## Authority boundaries

- The model proposes semantics; it has no direct executor, source client,
  secret, approval, lease, or persistence authority.
- The operation runtime is the sole executor invocation boundary. Tasks and
  validation facts persist before I/O; evidence is usable only after validation
  and a fenced durable commit. Adapters already revalidate scope immediately
  before I/O. P9.5-Q02 proves that exact validator-owned source/resource/
  revision/sensitivity facts survive in every read task and canonical evidence
  record and drive governance/provider routing.
- Catalog identities and revisions define resource scope. Stale or incomplete
  safety-critical structure is refreshed or rejected.
- Provider routing checks capability, context size, sensitivity, and explicit
  destination policy before any provider I/O. Persisted route revisions contain
  only provider-neutral configuration and `SecretReference` values.

## Untrusted inputs

Catalog names/descriptions, database values, rows, filenames, file content,
connector errors, model output, and extension metadata are untrusted data.
They are bounded and structurally encoded; they cannot supply system
instructions. SQLite and PostgreSQL reads are parsed and catalog-scoped before
connector I/O. Local-directory access is descriptor-rooted and rejects path,
symlink, hard-link, and protected-state escapes.

Configured capability-provider extensions are explicit caller-supplied code,
not discovered packages. Their declarations are validated atomically against
built-ins, bounded to 64 manifests, and persisted as an ordered immutable
version/declaration/manifest fingerprint set. Missing or drifted configuration
fails before provider or executor I/O. Extension tools still execute only as
persisted tasks through the operation runtime; monitor projection excludes
configured tools because no monitor-specific scope authority is declared.

Natural learning enters the same proposal, provenance, safety, revision, and
atomic-store owners as precise learning APIs. Direct `remember that` statements
are accepted only from completed successful user operations. Resource-alias
statements require one accepted current read binding; evidence-backed facts
remain visible proposals tied to exact accepted applicable evidence. Raw-row,
PII, credential, policy/security mutation, and executable candidates retain
only a hash and bounded rejection reason. Learned skill versions remain inert
until an explicit audited acceptance activates them.

## Secrets and diagnostics

Persist only a `SecretReference`. Resolution uses an injected provider,
supported OS keychain, or environment variable at the adapter boundary. Secret
values must not enter configuration, SQLite, events, blobs, model context,
errors, logs, CLI output, or telemetry. Provider and connector exceptions are
normalized and detached from raw tracebacks and server diagnostics.

Canonical audit state may be more detailed than model or public views. Public
event/inspection projections omit capability arguments, evidence payloads,
raw rows, connector text, credentials, and provider-native payloads. Optional
telemetry consumes committed events and can fail without affecting a runtime
commit.

## Local host

The host uses a private local Unix socket with ownership and permission checks,
bounded framing, admission idempotency, and one per-agent writer lock. The CLI
persists only route metadata and `SecretReference` URIs through `AgentHost`;
model show/status never resolve or print secret values. Event follow advances
only from committed durable cursors, and interactive output is a bounded public
projection rather than canonical payload state. The
attached PostgreSQL database and login role remain a deliberate trust boundary;
use a least-privilege role and treat row-security/server policy as external
configuration. Phase 9 verified the real SELECT-only adapter boundary; Phase
9.5 reran the affected real OpenAI/default-data-domain/SQLite path across cold
reopen with exact accepted read authority. The unchanged least-privilege real
PostgreSQL and other retained-provider Phase 9 rows remain cited evidence.
