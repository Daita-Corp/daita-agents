# Runtime: composition, loop, context and capabilities

> Contributor architecture notes, moved out of `AGENTS.md`. They describe the
> current design in detail. Code and tests win when they disagree; fix the note.
> The invariants in `AGENTS.md` are the hard rules. Everything else here can
> change through *Proposing architecture changes* in `AGENTS.md`.

How one agent is composed and how a run moves from a message to an answer.

## Run model

Daita is a persistent, read-first data agent with a narrowly scoped,
explicitly enabled native relational update/upsert capability, initially backed by PostgreSQL,
and locally admitted MCP external actions. It uses one direct loop:

```text
user message -> model -> zero or more tool calls -> ordered tool results
             -> model -> answer
```

The exact current-run transcript is the loop state. Tool failures are ordinary
model-visible results, so the model can correct a call on the next step. Normal
model text completes the run. There is no verifier, repair, synthesis, or
resumable-session pass.

Persisted tool calls keep the model's run-local handles. Code that reads call
arguments uses the executed record (`CapabilityRegistry.executed_tool_pairs`,
public `Agent.tool_evidence`), never per-reader re-resolution;
`tests/architecture` enforces this.

Outer step, wall-time, token, and estimated-cost limits bound the loop. Every
requested tool call receives exactly one result in call order, including when
independent reads execute concurrently or another call fails.

Budget exhaustion is terminal failure, with exact usage and partial evidence
retained. There is no post-limit model request or separate final-context path.
Returned usage is checked before tool dispatch and before accepting completion.
Canonical requests carry the remaining allowance; adapters own provider-native
input counting, reviewed price admission, and supported wire output controls.
Actual consumption is never clipped to an authorization or reservation ceiling.
Unknown machine-run usage consumes at least its reservation; it is never recorded
as measured zero. Exhausted cumulative budgets prevent subsequent reservation.

Daita supports catalog-backed SQLite, PostgreSQL and folder reads, foreground
local computer file access through the live session folder source with a
bounded-workspace option for typed callers, and explicitly admitted
server-neutral remote MCP reads and external actions. SQL is validated against
the current catalog before source I/O. Local file reads are descriptor-contained
and revision-bound. MCP calls
revalidate the exact binding revision, remote identity, and schemas.

Agent identity, source registrations, current catalog snapshots, declared
catalog links, exact run transcripts, terminal results, jobs, routines,
deliveries, permissions, and receipts are stored in one SQLite database inside
the agent home.

## Composition and public API

`daita.agent.Agent` is a thin public facade. It validates caller inputs and
delegates to `EmbeddedAgent`; it does not implement model progression, catalog
truth, capability execution, or persistence.

`daita.hosting.embedded.EmbeddedAgent` is the composition root. It admits the
agent home, holds the process-level writer lock, composes the
`RunAdmissionCoordinator` with immutable `AgentConfig.host_capacity` and retains
the broad mutation lock, constructs the
catalog, registry, domains, context builder, runtime, loop, artifact store, and
supervisors, and closes them in drain-safe order. Composition belongs here rather
than in the loop or a dependency-injection framework. The default
PostgreSQL read backend owns optional lazy bounded native pools and closes them
after drain, before credentials and state. Every read/export retains a fresh
read-only transaction and full structural revalidation; reuse grants no authority.
Pool configuration and resolved credentials are runtime-only; retired or uncertain
connections are never reused or replayed.

One open agent home has one writer (invariant 1 in `AGENTS.md`). Normally the
Daita host composes the agent and holds that lock; the TUI, the CLI and
`daita.connect()` are its clients. A script's `Agent.open()` or a
`daita --no-host` command composes the agent in-process instead, and is refused
with `HostActiveError` while the host holds it.

## Model loop and context

`daita.loop.driver.AgentLoop` is responsible only for:

- starting and completing one run transcript;
- calling the model;
- appending normalized assistant and tool messages;
- invoking the injected `ToolRuntime`;
- enforcing outer budgets; and
- returning one terminal `LoopExit`.

Each host-owned loop invocation consumes one immutable, single-use `RunSession`.
The session owns its cancellation/deadline state, immutable run options and exactly
one `RunSessionWriter`; the writer alone starts, appends to and terminates that run's
transcript with exact ordering and conversation-predecessor binding. The coordinator
owns bounded execution admission, keyed conversation ordering, provider admission
and shutdown drain. Compatibility defaults are five executions, two provider
requests, strict foreground reserves of one, one graph task, two PostgreSQL reads
per source and four SQLite-pressure permits. Host capacity is runtime configuration,
not execution authority. SQLite sources and MCP bindings remain capacity one;
analysis retains one computation lane and an independent parser lane. A
combined `data_query` holds each distinct input source's permit once, acquired
in sorted source-ID order, so inputs sharing a capacity-one source cannot
deadlock. Keyed permits do not replace the mutation lock.

The loop depends on the small `ModelProvider`, `ContextBuilder`, `ToolRuntime`, and
transcript-store protocols through those session boundaries. Provider payloads,
catalog operations, SQL validation, source I/O, policy, and feature lifecycle state
stay outside the loop.

`daita.context.AgentContextBuilder` creates each model request from
the current transcript, current catalog, projected tool definitions, and model
profile. It keeps complete tool exchanges together and labels catalog, tool,
file, memory, skill, and data content as untrusted. Untrusted content cannot
become an instruction or grant authority.

Preparation freezes core instructions, admitted metadata, prior continuity, and
safe artifact destinations. Each step adds code-owned procedure guidance only for
its authenticated callable tool projection and reports the remaining run allowance.
Optional discovery and prior continuity are fitted against the configured input
window, with a separate conservative footprint allowance derived from the effective
cumulative run limit. Fixed instructions, current input, and pinned schemas form
the mandatory baseline; they do not consume the optional-addition allowance. This
presentation target cannot truncate the exact current
transcript, lower sensitivity, grant authority, or replace provider token admission.
The same projection reports remaining steps and an advisory two-request budget
forecast using measured input growth and output allowances. Forecasts cannot reserve
credit, alter hard admission, or guarantee completion with unknown future results.

Foreground `RunInput.source_scope_ids=()` admits all currently readable sources
as candidates at preparation; a nonempty tuple narrows them to exact caller IDs.
One explicit effective scope freezes source and resource candidates for that run.
Revocation narrows it; later attachment or permission expansion cannot widen it.
An empty machine ceiling or resolved scope means no sources. Files-only scopes the
run to exactly the session folder source, which excludes catalog and MCP tools. There is no active source, conversation source, or
implicit source argument injection.

Clarification is ordinary foreground model behavior within the direct loop, not a
run state, admission layer, persisted selection, or special terminal result. Current
catalog outcomes, the connector directory, toolbox metadata, and ordinary tool
results provide domain-owned evidence. When a request depends on one materially
ambiguous resource, connector or account, recipient, or destination, the model asks
one concise question unless the user selected a choice or requested the set or a
comparison. Multiple search results alone are not ambiguity. Exact scope, argument,
permission, approval, and receipt validation remain with the existing runtime and
domains; approval never resolves ambiguity. Machine runs instead carry exact frozen
bindings and fail normally when those bindings cannot be revalidated.

Completed-run sensitivity is a conservative floor retained with conversation
continuity, including compressed assistant answers after source detach. Requests
also include the classifications of rendered connector metadata, memory, skills,
semantics, and current results. Model-written advisory content inherits the full
request classification. Local advisory imports default to restricted; explicit
local classification uses typed owner APIs or the owned Markdown sensitivity
label. These labels affect information handling, never execution authority.
Scheduled reasoning starts from its approved self-contained instruction and
exact retained skills, without unrelated conversation history or mutable memory.

The catalog is authoritative for current source and resource identity,
schemas, facets, relationships, and freshness. A current validated tool result
is authoritative only for the values it returned. Model text, prior assistant
claims, preferences, and procedures cannot establish structural or source
facts.

Machine-originated runs carry one immutable `ExecutionScope` in their
`RunStartEnvelope`. The scope binds the agent and principal, grant, job or
routine identity, allowed sources, resources, connector bindings,
capabilities, access modes, operational effects, sensitivity ceiling, model
routes, per-run budgets, and distribution-plan digest. Scheduled instructions
are foreground-authorized content; job-event instructions are code-owned.
Untrusted payloads and model text cannot enlarge the scope. `contract_bindings`
retains exact capability, MCP-origin, resource-structure and model-configuration
digests. The composition supplies one bound current-contract reader to routine
admission, runtime checks and code-owned graph-task construction; it cannot execute
work. Every current scope states `JOB_EVENT`, `SCHEDULED_ROUTINE` or `GRAPH_TASK`;
only the revision-1 migration decoder may infer an omitted legacy kind. Revalidation
compares retained references, never accepts replacement current contracts implicitly.
Local hints and refresh timestamps are presentation/freshness facts, not execution
authority.

## Capabilities and execution

`daita.capabilities.CapabilityRegistry` holds immutable capability, tool-view,
executor, and domain identities. It projects tool schemas and validates model
arguments and executor output. A tool is a model-facing view of a capability,
not another execution path.

`toolbox_search` accepts a natural-language `query`, optional bounded
`limit`, and an opaque continuation `cursor` over the run's applicable catalog.
Catalog and toolbox search rank lexical matches first and include labeled
unmatched fallbacks; bounded pages retain access to every scoped candidate.
The framework context includes an ephemeral bounded connector directory built
from current catalog, MCP binding, and skill metadata, alongside one compact toolbox
manifest. The manifest derives prepared-candidate access modes and operational effects
from immutable capability metadata; compact group summaries do not imply missing
connector permissions. Local discovery
hints are untrusted presentation and never change execution authority.
Toolbox grouping, access modes, and
operational effects are metadata, not model-selected search filters. Improve
discovery vocabulary in existing `ToolPresentation` records without changing
capability execution contracts. `toolbox_load` accepts exact on-demand names
directly; search is unnecessary when those names are known. `toolbox_inspect`
reads an exact prepared contract without changing the callable set or executing
a capability. These controls grant no authority and bypass no admission checks.
Search includes the existing domain-owned automation contract for grant-requiring
tools so scheduling can inspect constraints without activating execution schemas.
Its `requires_automation_grant` flag refers to scheduled execution. Search,
load receipts and exact inspection share one registered contract projection,
with exact revision digests. Load retains compact incomplete references and grant
metadata; explicit inspection supplies input/output schemas. Inspected contracts
remain in the ordinary transcript after callable-set replacement. Foreground actions
use ordinary approval at invocation.
Byte pressure omits whole contracts with `automation_contract_omitted` before
removing candidates. Load receipts omit their remaining grant metadata as a whole
when byte/depth bounds require it, preserving exact inspection references. `toolbox_inspect` retrieves a complete
contract or explicitly partial path/page/string fragments bound to its digest.
Inspection intersects frozen candidates with current local domain applicability;
it never refreshes remote state or replaces execution-time validation. The default
50-tool surface reserves 32 pinned, 15 on-demand and three control slots.

`daita.capability_runtime.CapabilityRuntime` is the sole production boundary
between model tool calls and execution. For each call it:

1. asks every statically composed domain to project applicable tools;
2. rejects a call that was not projected;
3. resolves the exact tool view, capability, domain, and executor identities;
4. validates arguments against the declared schema;
5. asks the domain to bind arguments and revalidate current admission;
6. performs the fixed operational-effect preflight, approval, recheck, and
   mutation-lock path when required;
7. executes exactly once;
8. asks the domain to finalize capability-specific semantics;
9. commits any artifact through the artifact-store boundary;
10. validates output, sensitivity, provenance, and bounds; and
11. normalizes one structured result in original call order.

`EmbeddedAgent` statically composes these domains: `DataCapabilityDomain`,
`ArtifactCapabilityDomain`, `AnalysisCapabilityDomain`, `MemoryCapabilityDomain`,
`SkillCapabilityDomain`, `SemanticCapabilityDomain`, `JobCapabilityDomain`,
`GraphTaskCapabilityDomain`, `GraphAdmissionCapabilityDomain`,
`DataProfileCapabilityDomain`, `RoutineCapabilityDomain`,
`DistributionCapabilityDomain`, and the admitted `MCPCapabilityDomain`. The
composition in `hosting/embedded.py` is the authoritative list. Applicability and current-state checks remain in the concrete
domain.

`CapabilityRuntime.execute_internal` is the typed code-owned path used by the
job and routine supervisors. It resolves the same immutable registry contract
and applies the ordinary validation, execution, artifact, sensitivity,
provenance, result-bound, and observation rules. It is not a recursive model
call or a second runtime.

External native data effects and admitted external actions declare an
`EffectReceiptPolicy`. Automation proposals also require an
`AutomationGrantPolicy`; the runtime validates both requested and domain-normalized
constraints. Effect-free and local management capabilities cannot use these
external-effect policies. Unattended effects require concrete native/MCP admission
and exact standing grants. Generic routine authority and outcome conformance alone
do not enable them; eligibility metadata alone is not production authority.

The runtime reserves a unique operation and call identity in SQLite before
external dispatch, validates the resulting observation and ordinary output, and
persists terminal evidence before returning an authenticated receipt reference.
Reservations are never automatically refunded or replayed. Native commit evidence
is adapter-verified; server invocation evidence is server-reported. Unusable
server output, missing evidence, and ambiguous failures become uncertain. Startup
recovers leftover started receipts before effects are admitted.

Unresolved receipts block new foreground external effects and their originating
routine. `Agent.inspect_effect`, bounded `Agent.list_effects`, and the human-only
`Agent.resolve_effect` expose evidence and exact foreground-approved recovery.
One immutable resolution is retained separately from the original observation;
resolution performs no retry and grants no connector permission.
Foreground context reads this same durable blocking check at preparation and
retains bounded receipt IDs/counts as internal operational metadata, without
receipt payloads. Current row values do not resolve prior operation uncertainty.
The frozen context informs reporting; execution still rechecks the store.

Do not call source clients or executors directly from `AgentLoop`, `Agent`, a
tool view, or model-authored text. Do not infer access, effects, or automation
eligibility from a tool name when capability metadata defines them.

## Memory, skills, semantics, approval and observation

Conversation continuity is a bounded projection of completed runs, not a
session runtime or resumable loop.

`MEMORY.md`, `USER.md`, and retained `SKILL.md` content are bounded advisory
text. They are not catalog truth, evidence, policy, authorization, executors,
or plugins. Semantic annotations are also advisory and must remain grounded in
current resource identity and evidence. Learning uses the ordinary foreground
loop and explicit capabilities; there is no background learning agent.

Operational effects use the fixed governance branch immediately before the
effect in `CapabilityRuntime`. Approval is once-only and bound to exact frozen
arguments; that binding is the invariant. Today approval is also in-process: the
host calls an `ApprovalHandler` and waits, with no pending-approval record or
resume API. A design that persists pending approvals must keep the exact-argument
binding and the once-only decision.

Observation is one best-effort callback. It cannot direct execution. Today it is
not persisted and is not a telemetry, tracing or replay source; durable truth
stays in transcripts, receipts and the other stores.
