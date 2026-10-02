# Remote MCP tools and external actions

Daita can use approved remote Model Context Protocol (MCP) tools to read from
other systems or take specific external actions. An operator first inspects a
server and admits its exact tools, access, and effects. A server's own names,
descriptions, annotations, and results cannot grant permission.

Each binding retains one exact endpoint, accepted protocol and tool capability
facts, local tool permissions, and schema digests. Calls recheck those facts
before remote I/O. Optional `serverInfo` is a display and drift hint.

## Supported surface

- Remote Streamable HTTP only. Plain HTTP is accepted only for loopback hosts.
- MCP protocol versions `2026-07-28`, `2025-11-25`, and `2025-06-18`.
  The SDK adapter probes `2026-07-28` explicitly and selects the legacy
  handshake only for `METHOD_NOT_FOUND`, an explicit admitted 2025 version, or
  initial discovery HTTP 400 containing a valid implementation-defined legacy
  JSON-RPC error, excluding the SDK's request-timeout code. Recognized modern
  errors, malformed bodies, authentication, network, deadlines and redirects
  never trigger a downgrade. Existing bindings
  execute with their accepted protocol pinned.
- No authentication, static bearer credentials from a `SecretReference`, or a
  host-owned personal connection. OAuth acquisition and refresh belong to the host.
- Bounded Draft 7 and JSON Schema 2020-12 object contracts. Omitted dialects
  default to 2020-12. Raw schemas and digests retain the dialect and annotations;
  model schemas remove annotations and expand bounded local JSON Pointer references.
  Nullable types and `anyOf`, `oneOf`, and `allOf` retain their assertion semantics.
  Network/file references, recursion and unsupported assertions are rejected.
  Validation uses the SDK's `jsonschema` dependency in an isolated, cancellable
  worker: 64 KiB schemas, depth 12, 1,024 expansion nodes, 32 references, at most
  eight composition branches, and a five-second validation deadline. Values are
  separately bounded. No defaults are inserted. `format` remains an annotation.
- Model admission checks every configured route candidate. Reviewed OpenAI
  non-strict and Anthropic schemas preserve these assertions. Other and custom
  adapters retain the previous portable primitive/object/array subset. Unsupported
  model projections fail admission rather than dropping constraints. Server output
  validation is independent of model parameter projection.
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

Stdio, framework-owned OAuth, dynamic client registration, sampling, roots, prompts,
resources, subscriptions, server-initiated requests, binary content, arbitrary
schema dialects, shell/infrastructure/arbitrary execution and asynchronous
completion are not supported. There is no automatic
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

Attach only exact tools whose access, effects and completion behavior an operator
has independently verified. Ordinary selections default to read access with no
operational effect. Local labels and discovery hints remain untrusted presentation;
they never change execution authority. Remote annotations cannot establish
read-only behavior, action permission or replay safety.

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

assert status.active_in_runtime
await agent.run("Look up the requested reference.")
```

Attachment and successful refresh activate in the same open Agent. The host
stages one new immutable catalog under its existing system admission lease,
then publishes it to runtime, routine and graph admission, and contract readers
between runs. In-flight runs keep their frozen catalog; changed binding revisions
invalidate old grants. Failed staging preserves the prior admission. Open still
reconstructs accepted declarations without network I/O. SDK clients are lazy,
unchanged bindings keep their owner, and replaced owners drain before management
returns. Binding namespaces prevent collisions across servers.

`MCPToolSelection("lookup")` is sufficient after independently verifying read
behavior. Alias, description and discovery hints are optional overrides. Generated
aliases are bounded and deterministic; re-admission preserves existing aliases.
Remote descriptions are untrusted presentation, including when used as defaults.
Imported descriptions are trimmed to the local description bound; default discovery
summaries and guidance are fitted to their own smaller bounds. Empty remote prose
uses a code-owned fallback. These presentation changes do not alter wire schemas
or execution permissions.

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

## Client factory API

Ordinary callers omit `mcp_client_factory` from `Agent.create` and `Agent.open`.
The agent uses the official SDK through its sole built-in factory. Importing
Daita, constructing the factory, and opening an agent home do not load the MCP
SDK, resolve credentials, or contact a server. Client construction loads the
SDK; the first inspection or call starts the connection.

The supported injection contract lives in `daita.adapters.mcp`:

- `MCPClientFactory.create(*, endpoint, authentication, secrets) -> MCPClient`
  creates a new independent client without network or credential I/O.
- `MCPClient.inspect(*, observed_at)` returns `MCPServerInspection` with freshly
  fetched contracts. It must not cache `tools/list` responses.
- `MCPClient.call_tool(remote_name, arguments)` returns `MCPToolResult`, sends at
  most one `tools/call`, and never retries or automatically continues a result.
- `MCPClient.close()` drains owned work and closes resources. It is terminal and
  idempotent. Operations have finite bounds, propagate cancellation, and report
  bounded failures as `MCPError` without credentials or unsafe remote error text.

The agent retains the injected factory and owns the clients it returns. Temporary
inspection clients close after success or failure. An activated binding owns one
client, constructed at its first exact call and reused for later drift checks and
calls until revocation or agent shutdown. A client rejected during credential
binding is closed before inspection or dispatch. Injection does not replace
Daita's binding admission, caller checks, approval, or receipt enforcement.

The built-in factory can be configured explicitly through the same interface:

```python
from daita import Agent
from daita.adapters.mcp import MCPClientFactory, SDKMCPClientFactory

factory: MCPClientFactory = SDKMCPClientFactory(timeout_seconds=10)
agent = await Agent.open("my-agent", mcp_client_factory=factory)
```

The `SDKMCPClientFactory` constructor's `http_transport` option accepts
`httpx2.AsyncBaseTransport` for SDK transport configuration and disposable tests;
it is outside the generic factory contract. Direct SDK client construction and
its internal owner/session objects are implementation details. The former
`StreamableHTTPMCPClient` and its factory
are removed. Migrate their imports to `SDKMCPClientFactory` and obtain clients via
`create`; migrate injected `httpx` fixtures to `httpx2`.

`Agent` and `MCPClientFactory` remain compatible; setup options are additive. Removing the
former concrete client imports is an intentional API break in `1.2.0`; those
callers must migrate to the factory interface. This interface clarification
requires no additional version bump: the planned package remains `1.2.0` with
`mcp==2.2.0` and the single pending home revision 3. Internal external-schema
validation is now asynchronous; extensions using `CapabilityRegistry` directly
must await `validate_arguments_async`. Native synchronous validation is unchanged.
An injected client may implement optional `MCPProtocolPinnedClient.bind_protocol`
to enforce the accepted protocol before network I/O; inspection drift checks
still apply to clients that implement only `MCPClient`.

### Hosted personal connections

`Agent.create` and `Agent.open` accept an `mcp_connection_provider`. The host
owns this provider and passes an authenticated `caller_principal_id` to each
foreground `run`, `learn`, candidate acceptance, MCP management, and hosted
artifact/history read. Calls without an actor retain the agent-owner default
for existing agent-owned work; they cannot access a personal connection owned
by another principal. The hosted dispatcher must pass its authenticated actor
for personal connections. Model instructions and tool arguments cannot set it.

An injected personal client implements the optional, typed
`MCPPersonalConnectionClient` protocol. Its synchronous
`bind_personal_connection(provider, principal_id)` hook binds the exact provider
and verified principal once before its first inspection or call. No-auth and
static-bearer clients need only the base `MCPClient` protocol. Binding the provider
does not resolve a token or grant access; rights and credentials are checked again
at their existing projection, dispatch, and request boundaries.

For a personal binding, use
`MCPAuthentication.personal_connection(connection_id, owner_principal_id,
resource_uri, required_scopes)`. The binding stores these exact non-secret claims
and its owner. The provider's `check_access` verifies connection ownership,
resource, scopes, and revocation at projection and call time. Its
`access_token` returns a short-lived token for the same claims on each HTTP
request. The resource URI and MCP endpoint must share one HTTPS origin;
redirects and responses that echo a credential are rejected before SDK parsing.
Provider failures return one of `needs_authorization`, `needs_scope_upgrade`,
`connection_revoked`, or `account_unavailable`; provider error text is discarded.
The host must enforce the same actor on its saved
conversation and token-vault APIs. The framework has no hosted connection setup
or callback flow.

Existing no-auth and static-bearer tools remain agent-scoped for execution;
binding management uses the binding owner's principal.

### What hosted integration must consume

The host must consume the eventual `daita-agents==1.2.0` artifact with its exact
`mcp==2.2.0` dependency, both private JSON Schema worker modules, and the matching
agent-home revision-3 migration and release-contract snapshot. Revision-1 and
revision-2 homes must pass the normal staged upgrade before opening; released revision-2 binding IDs,
grants, receipts and data remain intact. No revision 4 is introduced.
An unreleased development home carrying an earlier revision-3 checksum is not
silently rewritten; restore its pre-upgrade backup and apply the amended migration.

Use the supported `MCPClientFactory` contract for injection, or omit it to use the
SDK factory. Supply the personal connection provider and authenticated caller,
including exact connection, resource and scopes on every token request. Keep the
ordinary scope, approval and receipt paths. Attach and refresh now publish the
catalog in the same Agent, so callers should use the returned active status instead
of restarting. The host still owns OAuth acquisition and token refresh.

This local work does not publish a package, change a hosted pin or establish
connector release readiness. The host's own token-vault, caller-authentication and
deployment integration require separate verification.

Personal tools are omitted from another caller's catalog, including toolbox
search and connector metadata. Management lists omit bindings owned by another
caller; exact operations refuse them. A machine scope must freeze both the
personal binding ID, its exact tool origin digest, and its owner principal
before it can project or call the tool. A token is never written to the
binding, home database, transcript, artifact, or effect receipt.

The CLI provides the same bounded lifecycle surface. Each command takes the
local agent name first:

```text
daita mcp inspect <agent> <endpoint> [--bearer-env NAME]
daita mcp attach <agent> <endpoint> --tool <remote> [--tool <another-remote>]
# Existing four-value --tool syntax remains accepted for explicit presentation.
daita mcp status <agent> [binding-id]
daita mcp refresh <agent> <binding-id>
daita mcp revoke <agent> <binding-id> --yes
```

In the TUI, `/mcp` opens the server-oriented MCP manager. It groups independently
keyed bindings with the same trusted local label and endpoint for presentation, so
older one-tool bindings appear as one server without being merged or rewritten.
The primary statuses are `Accepted (validated at call)`, `Activation pending`,
`Needs refresh`, and `Revoked`; accepted does not claim a network check happened
at open. Internal binding IDs and protocol details are not part of the normal
management flow.

Choose **Add server** (or run `/mcp add`) for guided setup:

1. enter one Streamable HTTP endpoint and choose no auth, an environment/keychain
   reference, or masked bearer entry saved to the local keychain; inspect it;
2. review supported and unsupported tools with exact schema-rejection reasons;
3. select exact tools; default selections admit reads only;
4. use **Configure selected tool permissions** to review each alias, description,
   access mode, operational effect, unattended eligibility, result/outbound
   sensitivity and known completion semantics; choose the server outbound ceiling;
5. confirm the exact local permissions; the tools activate before the next run.

All tools selected in one guided admission are stored in one server binding, so
refresh and revocation apply to that reviewed tool set. The manager uses
descriptive server/tool pickers for refresh and revocation rather than asking
the operator to copy a binding ID. `/mcp status`, `/mcp inspect`, `/mcp attach`,
`/mcp refresh`, and `/mcp revoke` remain available as power-user commands.
CLI `--bearer-env NAME` and `--bearer-ref env:NAME|keychain:ACCOUNT` use existing
references. `--bearer-prompt` masks entry and saves an agent-owned credential when
attaching; inspection-only credentials are deleted afterward. The TUI shares the
same Agent admission and credential APIs. Cancelled setup removes its unused
credential. Revocation retains referenced credentials; deleting an agent removes
its owned MCP credentials and preserves external/shared references. The CLI `--tool` and
text `/mcp attach` commands retain explicit read-only admission. Action admission
is available through the Python API and the guided TUI permission controls.

Inspection and attachment report the bounded code-owned reason when a schema
is unsupported, such as an unsupported dialect or nonlocal `$ref`; a generic rejection
does not hide the exact admission constraint.

## Status, drift, and revocation

Use the explicit administration methods:

```python
statuses = await agent.list_mcp_servers()
refreshed = await agent.refresh_mcp_server(binding_id)
revoked = await agent.revoke_mcp_server(binding_id)
```

Before every admitted tool call, Daita reloads the exact binding revision,
re-resolves authentication, re-inspects the accepted protocol, capabilities,
and schema digests, and enforces the configured outbound sensitivity ceiling. A stale,
changed, revoked, unavailable, or authentication-failed binding returns one
bounded structured tool error. It does not fall back to another server or
retry the remote call.

Refresh records and activates a new checked revision between runs. Drift yields
a stale binding and requires explicit re-admission of the changed contract. Revocation is binding-local and immediately removes
that revision's authority while sibling bindings remain usable. Agent close
waits for in-flight binding work and closes each used SDK client in its own
serial owner task; a never-used binding owns no transport resource.

All successful remote results receive code-owned provenance containing the
binding and revision, exact remote tool identity, schema digests, call
identity, observation time, and sensitivity classification. The shared
capability-runtime result bound applies to successes and to every typed or
unexpected error before anything is appended to the transcript.

Same-conversation follow-ups retain bounded previews of successful MCP reads,
including after agent restart. The preview carries the original provenance and
marks truncated text or omitted structured data. Small results can fit in full.
Historical results are untrusted observations; they do not activate a tool or
authorize another call. Actions are excluded. Daita omits a read preview if its
original result lineage or exact locally admitted binding revision cannot be
verified. Existing conversation byte, message, and model-input bounds still apply.
Historical identity is verified from admitted binding and tool contracts;
connector discovery metadata remains presentation only.

## Explicit external actions

An action has separately admitted data access and operational effects. For example,
a notification uses `AccessMode.NONE` with `OperationalEffect.EXTERNAL_ACTION`.
Remote data mutations use `AccessMode.WRITE` and `OperationalEffect.MUTATE_DATA`.
Only these effect families are supported; calling an arbitrary execution tool an
external action does not make it supported.

```python
from daita import MCPToolSelection
from daita.capabilities import AccessMode, AutomationEligibility, OperationalEffect
from daita.llm.models import ModelSensitivity

selection = MCPToolSelection(
    remote_name="notify",
    local_alias="notify",
    description="Send a notification through the reviewed service.",
    access_mode=AccessMode.NONE,
    operational_effect=OperationalEffect.EXTERNAL_ACTION,
    automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
    result_sensitivity=ModelSensitivity.INTERNAL,
    maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
)
status = await agent.attach_mcp_server(
    endpoint="https://mcp.example.com/mcp",
    selections=(selection,),
    maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
)
```

Action selections default to `INTERACTIVE_ONLY` when eligibility is omitted.
Setting `AUTOMATION_DIRECT` permits a routine or eligible graph job proposal;
it grants no standing authority by itself. Each foreground invocation still
requires exact approval through the ordinary runtime. The public typed
admission API is a local operator control and must not be delegated to
model-written content.

Both tool and binding outbound ceilings apply to the **full model request
classification**, including retained history, research and connector metadata.
A restricted request cannot call an internal-only action. Adjust local admission
only when the actual service may receive that classification.

Every action uses the same `mcp.tool_call` grant constraints:

```python
constraints = {
    "binding_id": status.binding.binding_id,
    "binding_revision": status.binding.revision,
    "remote_tool_name": "notify",
    "fixed_arguments": {"destination": "reviewed-room"},
    "variable_argument_names": ["content"],
}
```

The enclosing `RequestedCapabilityGrant` supplies the exact capability ID and
`max_calls_per_occurrence` (1 to 256, further bounded by the run). The constraints
kind is `mcp.tool_call`; it is assigned by the domain policy, not sent to the server.
Approval shows the fixed values and variable names. Every fixed value must be
present and match exactly; extra names are rejected. The complete call must pass
the admitted input schema. This release permits only declared scalar variables
(string, integer, number or boolean). Fix an entire nested object or array to
restrict recipients or targets inside it. Variable arbitrary JSON cannot enforce
nested restrictions. Credentials belong in binding-owned secret references, never
in grants or tool arguments.

## Invocation evidence and uncertainty

Preflight reloads current admission and checks identity, schemas, arguments and
authority without invoking the action. The runtime rechecks after approval and
atomically reserves an effect receipt and call allowance before dispatch. The
existing executor sends exactly one `tools/call`. Canonically equivalent arguments
within the same run/occurrence cannot dispatch again, even with a new model call ID.
Reservations remain consumed after failures. Advertised idempotency never enables
replay, connector substitution or a second attempt to obtain a cleaner response.

A valid normal response, including plain text with **no output schema**, establishes
`SUCCEEDED / SERVER_REPORTED`: successful invocation only. It does not verify a
downstream business result. Text such as “success” has no special evidentiary meaning.
When an output schema is admitted, structured results must satisfy it. All ordinary
content, provenance, sensitivity and result-size checks precede success finalization.
The shared receipt kind `mcp.tool_call` records bounded identity/revision, argument
fingerprints and response/error classification; it omits credentials and private
argument/result bodies.

Lost responses, `isError=true` (which may follow partial application), malformed or
oversized output, timeout and cancellation after possible dispatch yield uncertainty.
Positive local non-dispatch can establish `NOT_APPLIED / LOCAL_NOT_DISPATCHED`.
Missing/invalid observations use runtime uncertainty. Failed receipt persistence
leaves a blocking reservation; reopening recovers it as uncertain.

Tools known to require asynchronous completion cannot enter an unattended proposal.
`MCPCompletionSemantics.ASYNCHRONOUS_ONLY` records a locally known limitation and
cannot execute through this release. Protocol `execution.taskSupport="required"`
also prevents direct execution; optional task support uses ordinary direct calls.
Unexpected task acceptance or other nonfinal results become uncertain
server-reported acceptance. Daita never polls or retrieves task results.
These distinctions follow the [MCP task protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks).

Required actions need authenticated successful receipts and validated tool results.
MCP invocation evidence cannot satisfy an `ADAPTER_VERIFIED` requirement. Optional
no-action paths are explicit; uncertainty still prevents successful completion.
Research and valid partial artifacts remain available when a required action fails.

Unresolved effects block potentially duplicating work across restart, future slots,
run-now, resume, changed arguments, revision and new effectful clones. Exact human
recovery records a receipt-linked decision without invoking anything. See
[effect receipts and recovery](EFFECT_RECEIPTS.md) and
[scheduled assignments](SCHEDULED_ROUTINES.md). Graph jobs admit only an exact,
foreground-approved synchronous MCP action as their initial effectful task; an
uncertain receipt blocks that task and its descendants. This implementation and
its deterministic fake-I/O acceptance are not production release approval.

The [offline assignment and recovery walkthrough](../examples/03_offline_assignments_and_recovery.py)
uses the production MCP client with an in-memory HTTP transport. It demonstrates
research, artifacts, a fixed-destination standing action, response loss and human
recovery without a live service. Live model and service validation remains opt-in.
