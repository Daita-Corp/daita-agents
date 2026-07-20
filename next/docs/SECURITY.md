# Security model

## Authority boundaries

- The model proposes semantics; it has no direct executor, source client,
  secret, approval, lease, or persistence authority.
- The operation runtime is the sole executor invocation boundary. Tasks and
  validation facts persist before I/O; evidence is usable only after validation
  and a fenced durable commit.
- Catalog identities and revisions define resource scope. Stale or incomplete
  safety-critical structure is refreshed or rejected.
- Provider routing checks capability, context size, sensitivity, and explicit
  destination policy before any provider I/O.

## Untrusted inputs

Catalog names/descriptions, database values, rows, filenames, file content,
connector errors, model output, and extension metadata are untrusted data.
They are bounded and structurally encoded; they cannot supply system
instructions. SQLite and PostgreSQL reads are parsed and catalog-scoped before
connector I/O. Local-directory access is descriptor-rooted and rejects path,
symlink, hard-link, and protected-state escapes.

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
bounded framing, admission idempotency, and one per-agent writer lock. The
attached PostgreSQL database and login role remain a deliberate trust boundary;
use a least-privilege role and treat row-security/server policy as external
configuration that Phase 9 live acceptance must verify.
