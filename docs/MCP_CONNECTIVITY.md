# Remote MCP read connectivity

Daita can admit independently configured remote Model Context Protocol (MCP)
servers as read-only tools in the existing direct model/tool loop. The feature
is deliberately narrow and server neutral: a binding records one exact HTTPS
endpoint, one negotiated server identity, an explicit read-tool allowlist, and
canonical schema digests. A server's names, descriptions, annotations, and
results are untrusted data and never create authorization.

## Supported surface

- Remote Streamable HTTP only. Plain HTTP is accepted only for loopback hosts.
- MCP protocol versions `2025-11-25` and `2025-06-18`.
- No authentication or a static bearer token resolved from an existing
  `SecretReference` at request time.
- Bounded JSON object input schemas using the documented accepted subset. An
  omitted dialect defaults to JSON Schema 2020-12; the exact explicit root
  declaration `https://json-schema.org/draft/2020-12/schema` is also accepted,
  retained in the admission/drift digest, and removed from the model-facing
  projection. Other or nested `$schema` declarations, unsupported keywords,
  and `$ref` are rejected.
- Text content and optional structured JSON-object results only.
- At most 32 independently keyed bindings per agent, 256 inspected tools per
  server, 128 admitted tools per binding, and 384 active admitted MCP tools in
  aggregate per agent. Stale and revoked bindings do not consume the active-tool
  allowance. Four discovery pages, 1 MiB per binding aggregate, 8 MiB of binding
  aggregates per agent, and fixed request, response, nesting, and result limits
  remain independently enforced.
- One run catalog admits at most 512 applicable tools and 2 MiB of canonical
  catalog material. Its canonical toolbox manifest is separately bounded to
  six entries, 8 KiB, and 2,000 estimated tokens.
- Pinned tools are selected exactly, never by priority: at most 32 pinned
  definitions and 96 KiB of pinned definition material may enter the initial
  provider surface. A step may contain at most 16 loaded tools, 96 KiB of
  loaded definition material, and 50 definitions or 128 KiB overall.
- Discover on-demand tools with `toolbox_search` using a natural-language
  `query` and optional bounded `limit`. Search spans applicable toolboxes;
  access modes and operational effects are enforced internally, not selected
  as search filters. Load exact names with `toolbox_load`, then invoke them
  normally on the next model step. Known exact names can be loaded directly
  without searching. Every applicable tool remains present exactly once in
  the frozen run catalog; a verified load receipt replaces the prior loaded
  set. Search and load grant no authority and perform no remote tool calls.

Stdio, OAuth, dynamic client registration, sampling, roots, prompts,
resources, subscriptions, server-initiated requests, binary content, arbitrary
schema dialects, and remote writes are not supported. There is no automatic
server or tool discovery at model request time and no server-specific default.

## Inspect and attach

MCP administration is an explicit operator action through the public Python
API, CLI, or TUI; it is never a model tool. First inspect an endpoint.
Inspection makes no persistent change and grants no execution authority:

```python
from daita import MCPAuthentication, MCPToolSelection

inspection = await agent.inspect_mcp_server(
    endpoint="https://mcp.example.com/mcp",
    authentication=MCPAuthentication.no_auth(),
)

for tool in inspection.tools:
    print(tool.remote_name, tool.supported, tool.unsupported_reason)
```

Attach only exact tools that an operator has independently established are
read only. The local server label, alias, description, summary, use guidance,
keywords, exposure, and priority are trusted admission data; a remote server
name or description is not copied into the trusted search corpus.

```python
status = await agent.attach_mcp_server(
    endpoint=inspection.endpoint,
    local_label="Reference service",
    authentication=MCPAuthentication.no_auth(),
    selections=(
        MCPToolSelection(
            remote_name="lookup",
            local_alias="reference_lookup",
            description="Look up a reference record by its exact identifier.",
            summary="Look up one admitted reference record.",
            when_to_use="Use for an exact identifier lookup in the reference service.",
            keywords=("reference", "identifier", "lookup"),
        ),
    ),
)

print(status.binding.binding_id, status.reopen_required)
```

An attached or refreshed binding becomes a static capability only on the next
controlled `Agent.open`. Open reconstructs declarations entirely from the
accepted aggregate without network I/O or secret resolution; the remote client
is created lazily at the first exact call. This preserves one immutable
registry for each open agent. Local tool names include a binding-derived
namespace, so identical remote names on different servers cannot collide.

For bearer authentication, persist only a reference to an environment or
keychain secret:

```python
from daita import MCPAuthentication
from daita.security import SecretReference

authentication = MCPAuthentication.bearer(
    SecretReference.environment("EXAMPLE_MCP_TOKEN")
)
```

The token value and MCP session identifier are never stored in the binding or
state database. The reference is resolved again immediately before every
network request.

The CLI provides the same bounded lifecycle surface. Each command takes the
local agent name first:

```text
daita mcp inspect <agent> <endpoint> [--bearer-env NAME]
daita mcp attach <agent> <endpoint> --tool <remote> <alias> <description> <result-sensitivity>
daita mcp status <agent> [binding-id]
daita mcp refresh <agent> <binding-id>
daita mcp revoke <agent> <binding-id> --yes
```

In the TUI, `/mcp` opens the server-oriented MCP manager. It groups independently
keyed bindings with the same trusted local label and endpoint for presentation, so
older one-tool bindings appear as one server without being merged or rewritten.
The primary statuses are `Accepted (validated at call)`, `Restart required`,
`Needs refresh`, and `Revoked`; accepted does not claim a network check happened
at open. Internal binding IDs and protocol details are not part of the normal
management flow.

Choose **Add server** (or run `/mcp add`) for the guided no-auth path:

1. enter and inspect one Streamable HTTP endpoint;
2. review supported and unsupported tools with exact schema-rejection reasons;
3. multi-select only tools independently verified to be read only;
4. review Daita-generated provider-safe aliases, code-owned descriptions, and
   the default `internal` result sensitivity;
5. confirm the read-only attestation; and
6. optionally perform a controlled agent-runtime restart to activate the new
   immutable registry.

All tools selected in one guided admission are stored in one server binding, so
refresh and revocation apply to that reviewed tool set. The manager uses
descriptive server/tool pickers for refresh and revocation rather than asking
the operator to copy a binding ID. `/mcp status`, `/mcp inspect`, `/mcp attach`,
`/mcp refresh`, and `/mcp revoke` remain available as power-user commands.
If activation was deferred, the manager exposes **Restart now** while any
current binding revision is absent from the open runtime.
Use the CLI or Python API when a bearer-secret reference or non-default
sensitivity is needed.

Inspection and attachment report the bounded code-owned reason when a schema
is unsupported, such as an unsupported dialect or `$ref`; a generic rejection
does not hide the exact admission constraint.

## Status, drift, and revocation

Use the explicit administration methods:

```python
statuses = await agent.list_mcp_servers()
refreshed = await agent.refresh_mcp_server(binding_id)
revoked = await agent.revoke_mcp_server(binding_id)
```

Before every admitted tool call, Daita reloads the exact binding revision,
re-resolves authentication, re-inspects server identity and accepted schema
digests, and enforces the configured outbound sensitivity ceiling. A stale,
changed, revoked, unavailable, or authentication-failed binding returns one
bounded structured tool error. It does not fall back to another server or
retry the remote call.

Refresh records a new active or stale revision and requires another controlled
open before execution. Revocation is binding-local and immediately removes
that revision's authority while sibling bindings remain usable. Agent close
waits for in-flight binding work and closes only clients that were actually
initialized; a never-used binding owns no transport resource.

All successful remote results receive code-owned provenance containing the
binding and revision, exact remote tool identity, schema digests, call
identity, observation time, and sensitivity classification. The shared
capability-runtime result bound applies to successes and to every typed or
unexpected error before anything is appended to the transcript.
