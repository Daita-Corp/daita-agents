# Remote MCP tools and actions

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

How admitted MCP bindings are read and called.

## Bindings, calls and actions

Remote MCP support uses one bounded server-neutral Streamable HTTP client.
Every binding records an exact endpoint, negotiated identity, admitted
tool allowlist, schema digests, local access/effect/eligibility and completion
admission, sensitivity ceilings, and secret
reference. Agent open reconstructs immutable declarations without network I/O.
The first exact call initializes the client and rechecks the binding revision,
remote identity, schemas, and authentication.

Remote metadata and results are untrusted. MCP tools cannot gain write access
from annotations, names, descriptions, or schemas. Revocation is binding-local
and takes effect immediately. A stale, changed, revoked, unavailable, or
authentication-failed binding yields one bounded tool error without switching
servers or retrying the remote call.

MCP selections default to reads; explicit action selections default to interactive
approval. Locally admitted unattended actions use the shared `mcp.tool_call` grant:
exact binding/revision/tool, fixed top-level JSON values and bounded distinct scalar
variable names. Fix nested values in full; arbitrary variable JSON cannot enforce
nested recipient or target restrictions. Tool and binding outbound ceilings apply
to the full request classification. No remote annotation grants authority.

The existing MCP executor preflights without dispatch, then sends exactly one
`tools/call` after the runtime's reservation. Valid plain-text results without an
output schema establish server-reported invocation evidence only. Partial errors,
unusable output, response loss or explicit task acceptance produce uncertainty;
positive local non-dispatch can establish non-application. Known asynchronous-only
completion cannot execute or enter an unattended proposal. No task polling, custom
remote receipts, status/idempotency extension, per-server action adapter or replay
exists. Current MCP output/capability identities are `mcp.tool.result` / `mcp.tool`.
