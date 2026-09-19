# Repository guide

This file describes the current Daita architecture and the rules for changing
it safely.

## Repository scope

`src/daita/` is the sole production package and Python import namespace. Put
tests in `tests/`, user-facing documentation in `docs/`, and runnable examples
in `examples/`. Run commands from the repository root so Python uses the
configured `src` layout. Do not create another root-level `daita/` package or a
parallel replacement source tree.

Use this order of authority when repository material disagrees:

1. the current task requirements;
2. production code under `src/daita/` and executable tests under `tests/`;
3. `README.md` and current user-facing documents under `docs/`.

Preserve unrelated working-tree changes. Historical code and documents can
explain intent, but they do not define current behavior.

## Architecture status: current revision 2

The production code, current schema, tests and all ordinary statements in this file
describe **current agent-home revision 2**. Revision 2 is the sole runnable/current
format. Revision-1 records and decoders exist only inside the immutable revision-2
migration and its fixtures; they are not runtime compatibility paths. There is no
feature flag, fallback or dual-runtime interval selecting legacy or draft execution.

The current revision-2 graph has these non-negotiable boundaries:

- the adaptive task graph is owned inside `jobs`; `JobOwner` owns commands and one
  graph-aware `JobSupervisor` owns graph-local selection, claims and recovery;
- there is still one reentrant `AgentLoop` semantic implementation and one
  `CapabilityRuntime`; each model task attempt has a fresh isolated `RunSession`,
  exact transcript and one `RunSessionWriter`;
- `RunAdmissionCoordinator` owns host workload admission, lifecycle drain and keyed
  provider/source/MCP/SQLite/effect permits; it does not own graph transitions,
  model routing or capability semantics;
- root authority, outcome, distribution, deadlines and total budgets are immutable;
  planner-created task scope is a validated subset and model text never grants
  authority;
- graph V1 is structurally effect-free. Effectful graph work is a separately gated
  later phase; the existing receipt boundary remains the only effect truth;
- graph transactions are short indexed SQLite CAS operations. Provider, source, MCP,
  artifact and effect I/O never occurs beneath a graph transaction or broad host
  lock;
- normalized tasks, edges, attempts, accepted results, controls, mutations, budget
  ledgers and bounded audit events are current graph state/evidence. Events are never
  event-sourced authority;
- routines remain the separate time-triggered occurrence system; a routine is not a
  graph task and no routine-to-graph bridge is part of graph V1;
- revision 2 is one atomic conversion/cutover. Current runtime codecs contain only
  the graph shape afterward; revision-1 decoders remain migration-only; the old
  single-job, connected-executor and autonomous-follow-up paths are deleted at the
  specified gates rather than retained as a selectable compatibility engine.

## Current production architecture (revision 2)

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

Daita supports catalog-backed SQLite and PostgreSQL reads, foreground local
computer file access with a bounded-workspace option for typed callers, and
explicitly admitted server-neutral remote MCP reads and external actions. SQL is
validated against the current catalog before source I/O. Local file reads are
descriptor-contained and revision-bound. MCP calls
revalidate the exact binding revision, remote identity, and schemas.

Agent identity, source registrations, current catalog snapshots, exact run
transcripts, terminal results, jobs, routines, deliveries, permissions, and
receipts are stored in one SQLite database inside the agent home.

## Directory layout

```text
src/daita/
  agent.py                    # public Agent facade
  hosting/embedded.py         # composition, agent-home admission, locks
  hosting/resident.py         # headless host for the same composition
  loop/                       # direct transcript progression and run records
  llm/                        # canonical model records, routing, adapters
  capabilities.py             # declarations, registry, schema validation
  capability_runtime.py       # common model-to-execution boundary
  domains/                    # statically composed capability domains
  context.py                  # bounded classified framework request construction
  scope.py                    # effective source/resource scope intersection
  domains/data/               # data validation, SQL, files, artifacts
  catalog/                    # normalized source and resource truth
  adapters/                   # source admission, discovery, and bounded I/O
  artifacts/                  # artifact records, renderers, storage, delivery
  jobs/                       # durable job records, lifecycle, supervisor
  routines/                   # scheduled routines and occurrence supervisor
  distribution/              # outcomes, destinations, deliveries, inbox view
  memory/                     # bounded advisory memory
  skills/                     # bounded retained Markdown procedures
  storage/sqlite.py           # durable state operation boundary
  storage/sqlite_schema.py    # exact current physical schema
  storage/sqlite_codecs/      # strict current-record serializers
  storage/home_migrations/    # sole append-only agent-home revision registry
  security/                   # secret references and lazy resolution
  config.py                   # immutable runtime and model configuration
  workspace.py                # runtime-only local file access intent and known folders
  cli.py                      # CLI over the public embedded API
  tui/                        # source-free navigation, approvals and human controls
tests/                        # deterministic and opt-in live tests
examples/                     # offline examples
docs/                         # user-facing guides
pyproject.toml                # package, dependencies, entry point, tools
```

Add a module only when a current behavior needs a clear responsibility. Fix a
broken contract in its existing component instead of adding a parallel
abstraction.

## Composition and public API

`daita.agent.Agent` is a thin public facade. It validates caller inputs and
delegates to `EmbeddedAgent`; it does not implement model progression, catalog
truth, capability execution, or persistence.

`daita.hosting.embedded.EmbeddedAgent` is the composition root. It admits the
agent home, holds the process-level writer lock, composes the capacity-one
`RunAdmissionCoordinator` and retains the broad mutation lock, constructs the
catalog, registry, domains, context builder, runtime, loop, artifact store, and
supervisors, and closes them in drain-safe order. Composition belongs here rather
than in the loop or a dependency-injection framework.

One open agent home has one writer. A foreground TUI or CLI process and the
resident host must hand off that lock; they cannot open the same home
concurrently.

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
transcript with exact ordering and conversation-predecessor binding. The capacity-one
coordinator preserves revision-1 serial behavior while owning execution admission,
keyed conversation ordering, provider admission and shutdown drain. Conservative
source/resource, MCP, SQLite-pressure and effect permit interfaces do not yet enable
cross-run I/O concurrency or replace the mutation lock.

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
An empty machine ceiling or resolved scope means no sources. Files-only excludes
both catalog and MCP tools. There is no active source, conversation source, or
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

The statically composed domains are `DataCapabilityDomain`,
`MemoryCapabilityDomain`, `SkillCapabilityDomain`,
`SemanticCapabilityDomain`, `ArtifactCapabilityDomain`,
`RoutineCapabilityDomain`, `DistributionCapabilityDomain`, and the admitted
MCP domain. `LearningCandidateGuard` supplies the bounded transient learning
selection and mutation-success state shared by relevant domains. Applicability
and current-state checks remain in the concrete domain.

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
do not enable them; implementation acceptance is not production release approval.

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

## Data, catalog, and adapters

`daita.catalog` holds normalized structural truth and implements catalog
search, inspection, and traversal. Data code consumes catalog contracts rather
than building another schema graph or querying source clients for planning
facts.

`daita.adapters` implements source admission, containment, discovery,
freshness checks, and I/O. SQLite paths must be absolute and bounded. Local
workspace access uses only admitted relative logical paths and rejects
symlink, traversal, secret-file, and special-file access. PostgreSQL and MCP
credentials remain secret references and are resolved only at the integration
boundary.

SQL validation belongs in `daita.domains.data.sql`; connector guardrails still
apply during execution. Do not duplicate either mechanism in a generic policy
layer.

Data capabilities are reads except for explicitly admitted `data_update_rows` and
`data_upsert_rows`, initially backed by PostgreSQL. Their read-only counterparts are
`data_preview_update_rows` and `data_preview_upsert_rows`. Execution requires an
authenticated current-run preview and matching intent. Update retains its exact
selection, target-count, drift, rollback and receipt safeguards.

`RelationalWriteScope` binds exact structural revision, explicit update/upsert
operations, keys, insert/update columns, admitted identity generation and row limits.
Update permission never implies insertion authority. Upsert uses one bounded uniform
scalar batch, supported non-null unique-key equality, explicit omitted/null semantics,
and narrowly admitted identity generation. Its transaction acquires EXCLUSIVE on the
exact table before the authoritative scan, rebuilds the preview, rejects drift,
inserts missing rows, updates changed rows, skips unchanged rows, and verifies exact
counts summing to the input count. Mismatches roll back the entire batch. Sequence
allocations may leave gaps after rollback; receipts describe table-row effects.

The data domain normalizes native standing grants and permits at most one native
write capability per routine, with one invocation per occurrence and exact ceilings.
An unchanged batch consumes its reservation and produces verified zero-mutation
evidence. Full request sensitivity must fit the current target classification.
Research lineage remains model-derived claims, distinct from transaction facts.
Receipt reservation/finalization stays in CapabilityRuntime. No arbitrary SQL,
insert-only tool, delete, DDL, chunking, automatic retry or replay is supported.
Native implementation acceptance is not production release approval.

## Local computer files and artifacts

The local Files surface is separate from cataloged sources. Plain local CLI/TUI
composition uses computer access: its frozen working directory is the default,
while absolute and `~/` paths can address other OS-permitted locations.
`file_search` supports one root or a bounded multi-root set with one shared
budget, qualified matches, and per-root coverage. Typed `LocalWorkspace` callers
remain contained by default unless they explicitly select computer access. Both
modes use the same resolver, descriptor-contained backend, declarations, runtime,
and edit pipeline. Hosted and machine-originated runs have no ambient local-file
authority. Private Daita state is rejected when targeted and pruned from broader
searches.

`file_search`, `file_read`, and `file_query` return bounded, revision-bound
results. `file_query` uses a private one-call DuckDB worker over an exact input
manifest and exposes only the relation `data` to validated SQL.

An authenticated current-run `file_read` binding can feed
`artifact_edit_text`. That capability commits a complete replacement artifact
without changing the local file. `artifact_save_local` requires exact approval,
revalidates the unchanged bound file, and atomically publishes the artifact.
Drift requires a fresh read and edit.

The artifact store is the sole storage boundary for committed artifact bytes
and manifests. `artifact_list` returns bounded safe metadata only for the
current conversation. `artifact_read` returns a bounded preview for an exact
known artifact ID owned by the current agent home. `artifact_convert` supports
only a verified Daita-generated XLSX `Data` snapshot converted to CSV and
records the parent artifact. There is no public agent-wide inventory, hidden
current-file pointer, raw model path/byte interface, or alternate artifact
execution path.

`artifact_create_tabular` creates one bounded model-authored CSV, XLSX, or
HTML table from exact earlier successful tool-call IDs in the current run. The
artifact domain authenticates each result against the immutable registry and
persisted transcript, rejects failed, stale, reordered, cross-run, or tampered
lineage, inherits the highest result sensitivity, and preserves current
relational resource revisions where present. The artifact remains explicitly
derived analysis rather than exact or complete source data. Exact complete
relational export remains the separate `data_export_tabular` capability.

## Remote MCP tools and actions

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

## Durable graph jobs

`GraphJob`, `JobGraph`, normalized tasks, dependencies, attempts, accepted results,
controls, mutations, bounded events and budget ledgers are the sole current durable
job records. `JobOwner` owns admission and bounded lifecycle commands. One
graph-aware `JobSupervisor` owns graph-local selection, claims, fencing, recovery,
fairness and finalization; all capability work still executes through the one
`CapabilityRuntime`, and every model task uses the one `AgentLoop` implementation.

`start_data_profile` admits the static effect-free profile graph. `start_graph_job`
admits one code-resolved graph-eligible initial task and internal finalizer when a
model route and finite cost ceiling are configured. Planner-created work is a
validated immutable-authority subset. Unmappable migrated work is
`needs_attention`; no legacy executor can run it.

Agent identity is the job authorization boundary. The originating
conversation and run are immutable provenance, not access gates. Bounded list,
inspect, result-read, and cancellation operations can address any job owned by
the agent. Cross-agent lookup fails without exposing metadata. Work pauses
when no `EmbeddedAgent` host is open.

There is no current connected-executor mode, embedded single-job attempt/result
state, autonomous-follow-up driver or selectable legacy supervisor. Finalizers
authenticate accepted task results and publish through the existing delivery
boundary. Revision-1 terminal follow-up/delivery provenance is preserved only by
migration-owned conversion.

## Scheduled routines and deliveries

Routine mutation tools return compact identity/revision/state, schedule and reservation
receipts; full contracts remain in inspection and enforcement. Stopped model runs
retain completed tool evidence and do not roll back committed routines. Terminal
notices and CLI summaries project that evidence without creating another durable
outcome or synthesizing a model answer.

`ScheduledRoutine` and `RoutineOccurrence` are the only scheduled-work records.
`RoutineOwner` admits exact foreground-authorized, self-contained instructions
and implements bounded create, list, inspect, update, pause, resume, run-now,
disable, expiry, budget, and lifecycle transitions. `RoutineSupervisor`
computes due slots, conditionally claims occurrences, fences stale work,
reserves one run, invokes `AgentLoop`, and atomically finalizes the occurrence
and its logical deliveries.

Routines support exact one-time, anchored interval, and IANA-timezone calendar
schedules with explicit daylight-saving gap/overlap behavior and bounded
`skip` or `latest_only` misfire handling. A routine freezes its exact source,
resource, MCP binding, capability, model-route, sensitivity, outcome,
distribution, budget, expiry, and optional retained skill-content contracts.
Raw prompt text never determines whether a time slot is due.

Scheduled execution permits statically declared `automation_direct` capabilities.
Native effects additionally require exact data-owned grants and runtime receipts;
effect-free operations retain read/none access. It can
create only these artifacts:

- `artifact.create_document`;
- `data.export_tabular`;
- `artifact.snapshot_result`.

`artifact.snapshot_result` produces bounded canonical `application/json` from
an exact earlier successful result in the same run. It performs no source I/O
or format projection.

Scheduled runs can perform one explicitly granted native update/upsert and admitted
MCP actions within their exact per-occurrence grants. They
cannot start or cancel jobs, manage routines, call
unadmitted remote actions, publish local files, deliver through external distribution destinations, run shell
commands, or submit workflows or execution graphs. An exact resource-revision
precheck may complete an unchanged occurrence without a model call.

`OutcomeContract` validates terminal conclusions and required artifacts.
`DistributionPlan` freezes ordered target bindings. `Delivery` is the sole
durable representation of a logical delivery; the Inbox is a bounded product
view over deliveries. The only distribution destination is the originating
conversation inbox. Producer finalization validates committed artifacts and
the current destination, constructs immutable references, and commits the
producer outcome and unique deliveries atomically before any UI wake.

The resident host keeps the same `EmbeddedAgent` composition open for one
agent home. It does not add an API server, IPC framework, daemon manager,
multi-host queue, execution runtime, state store, or competing writer.

## Memory, skills, semantics, approval, and observation

Conversation continuity is a bounded projection of completed runs, not a
session runtime or resumable loop.

`MEMORY.md`, `USER.md`, and retained `SKILL.md` content are bounded advisory
text. They are not catalog truth, evidence, policy, authorization, executors,
or plugins. Semantic annotations are also advisory and must remain grounded in
current resource identity and evidence. Learning uses the ordinary foreground
loop and explicit capabilities. Candidate review is disabled by default. When
explicitly requested, it uses one tool-free model request outside `AgentLoop`
and places proposals in an inactive inbox. `/memory accept <id>` handles exactly
one candidate through a fresh foreground run; there is no bulk acceptance or
background learning agent.

Operational effects use the fixed governance branch immediately before the
effect in `CapabilityRuntime`. Approval is once-only, in-process, and bound to
exact frozen arguments. It does not create pending state or a resume API.

Observation is one best-effort callback. It cannot direct execution and does
not create durable events, telemetry, tracing, or replay state.

## Persistence and production upgrades

`daita.storage.sqlite.SQLiteStateStore` is the sole current SQLite operation
boundary. `daita.storage.home_migrations` is the sole persistence-compatibility
authority for the complete agent home. One monotonic home revision covers the
database, persisted records, model configuration, memory, user profile, skills,
artifacts, and other durable files that must change together. It is independent
of the package version and Git tag.

Production home revisions 1 and 2 are frozen. The registry is ordered and append-only;
released migration IDs, checksums, implementations, target schemas, historical
decoders, and golden fixtures never change. `CURRENT_HOME_REVISION` derives from
the last registry entry. A format change appends one owner-local home migration
and declares every affected relative path. Fresh homes are created directly at
the current complete format rather than replaying history.

Current runtime serializers accept only the current shape and contain no
per-record version discriminators or compatibility branches. Historical parsing
belongs only to the immutable migration that consumes that shape. The revision-1
bridge admits only the exact three observed preproduction shapes; it is not a
general legacy framework.

Open owns the existing agent-home writer lock across inspection, crash recovery,
upgrade, full-home validation, and runtime composition. Upgrades preflight disk
space, stage and back up all affected files, use SQLite's backup API, apply the
known suffix in order, validate the complete staged home, and publish `state.db`
last. Hash-bound journals recover staging, partial commit, and rollback states.
Publication failure restores the prior home, and one rollback bundle is retained.
Newer, unsupported, reordered, checksum-edited, or damaged homes fail closed
without a state rewrite.

Every supported production revision has one immutable whole-home golden fixture.
Tests cover every supported-to-current path, durable-data preservation, failure
and crash boundaries, recovery, rollback, and downgrade refusal. The inclusive
support window is an explicit registry policy; changing it is a release decision.
See `docs/LOCAL_STATE_UPGRADES.md` for the operational contract.

Release compatibility is maintained between tagged releases, not arbitrary
development commits. `release/agent-home-contract.json` is the committed
semantic snapshot of the current physical SQLite schema, strict stored-record
field contracts, owned home-file layouts, and migration registry. Before a pull
request or tag, refresh and verify it with:

```bash
python scripts/check_home_release_contract.py write
python scripts/check_home_release_contract.py check
```

CI compares the candidate with the latest earlier tag containing a snapshot. A
durable-contract change requires `CURRENT_HOME_REVISION` to exceed that tagged
revision and requires the matching migration and golden fixture. A release with
no persistence change retains the current home revision. Every managed GitHub
release publishes its exact snapshot even when it matches the prior release.

One not-yet-released next migration may be amended across development commits;
update its candidate checksum, fixture, and snapshot together. Compatibility
between homes created by unreleased commits is not promised. Once the first Git
tag containing that revision is created, its migration ID, checksum,
implementation, target schema, historical decoder, and golden fixture are
immutable. The snapshot is release evidence, not a second runtime compatibility
authority. A semantic durable-format change that the generator cannot infer
still requires an explicit new home revision.

Source read authority exists only in `source_read_scopes`. Native relational write
authority exists only in `relational_write_scopes`. Connection JSON never
owns either permission. Reconstruction fails closed, refresh preserves exact
scopes, and detach revokes both scope families atomically.

In current revision 2, all state mutation must be atomic and cancellation-safe. Do
not add event sourcing, replay projections, another state abstraction or a second
writer around SQLite. Only the normalized graph records, bounded task checkpoints/
comments and audit event cursor defined by the current schema are permitted; those
records are not an event-sourced replay system and remain behind `SQLiteStateStore`.

## Models and providers

Canonical messages, tool calls and results, usage, requests, responses, and
errors live in `daita.llm.models` and `daita.llm.errors`. Provider-native
payloads end inside provider adapters. `daita.llm.routing` handles retry and
fallback decisions from normalized failures; `AgentLoop` does not retry a
whole run or inspect provider-specific failures.

`daita.llm.provider_definitions` is the sole static source for built-in provider
identity, display metadata, authentication and endpoint modes, lazy construction,
request-policy facts, and unreviewed-profile capability defaults. The factory,
embedded host, and TUI derive their views from those definitions; do not add a
parallel provider list or vendor dispatch branch. Reviewed model limits and
prices remain in `profiles.py` and `pricing.py`.

One immutable `ModelCallPolicy` in `AgentConfig` and `ModelRequest` governs
configured and conforming injected providers, both delivery modes, foreground,
routines, graph tasks, validation and candidate review. Defaults are 180 seconds
per logical request, 120 per attempt, 60 to first substantive progress, 30 idle,
15 counting, 5 connect, 120 read, 30 write, 5 pool and 5 cleanup. Every logical
request intersects its caller/run deadline before setup; retries retain that
logical deadline and receive a fresh, narrower attempt deadline. Counting has
its own phase cap (`call_policy.input_count_timeout_seconds`, at most 60).
Monotonic deadlines are runtime-only; policy durations and both retry ceilings
are serialized and included in frozen machine model-contract digests.
Counting transport failures retain their canonical cause and proven zero generation
usage; invalid count data and unsupported counting remain permanent admission
failures. External cancellation remains cancellation.

Bounded OpenAI, Anthropic, and Gemini API requests use provider-owned input
counting over the prepared generation input, including tools and retained
provider content. Count and generation calls share the run deadline and SDK
ownership. Token counts are admission estimates; returned usage is incurred
consumption. Byte bounds must not substitute for billed-token counts. Routes
without complete counting retain usage-based progression and supported output
caps, and cannot admit estimated-cost ceilings. A failed count never submits
generation. Preserve known zero usage when counting is cancelled while retaining
normal cancellation/deadline behavior; dispatched generation can remain unmeasured.

The router owns model retries. API SDK retries are disabled on owned clients and
on borrowed-client request views without changing or closing the caller's client.
Every retried attempt receives the remaining logical-request allowance. Unknown
failed-attempt consumption stops budgeted routing; it cannot fund a retry or
fallback. A provider-reported charge alone does not establish advance price
coverage. Subscription output bounds remain advisory where the external surface
does not support a wire limit; returned usage still controls loop progression.

Retry accounting covers active attempts, backoff, cancellation, and stream cleanup.
Visible stream progress, including terminal completion, closes retry/fallback
eligibility. Completed usage remains authoritative during shutdown. Valid HTTP
retry delays are honored without shortening; waits exceeding 60 seconds or the
remaining deadline stop recovery. Local exponential backoff uses bounded jitter.
`RetryPolicy` defaults to `max_attempts_per_candidate=2` and
`max_total_attempts=3`. Both limits are one for one-shot validation/review.
Every configured route uses `ModelRouter`, including a one-candidate route.
Injected providers retain caller ownership and must honor the canonical policy;
they do not gain an implicit router. Tests exercise both compositions.

Provider lifecycle follows explicit ownership. Providers constructed from an
agent's persisted model route are closed by `EmbeddedAgent` after runs and
supervisors drain. Providers injected by a caller remain caller-owned and may
be reused across agent instances. Each provider closes only SDK clients it
created; injected SDK clients remain borrowed. Temporary validation and
candidate-review providers are closed by the component that creates them.

Owners stop new work and drain active calls before closing a provider. Close
callers share one absolute cleanup deadline and a retained once-only outcome;
repeated calls cannot retry native cleanup or renew grace. Native SDK scopes
enter, iterate and exit in one owned task. A bounded supervisor retains native
work that fails to retire, poisons its owner, discards late output and rejects
replacement work. Only native I/O, never tools or persistence, can remain there.
Transport bytes, decoded activity, substantive progress, canonical emission and
terminal completion are separate facts. Empty introductions, snapshots without
growth and keepalives do not reset progress. Opaque nonstreaming/CLI calls report
progress as unobservable and obey fixed deadlines. Terminal usage survives
cleanup failure; successful completion is published only after native release. Adapters scope
request-stream cleanup to completion, failure, cancellation, or early exit.
Canonical stream wrappers finalize in the iteration context and propagate
closure to their delegate; releasing a request stream never closes a borrowed
provider or SDK client. Verify underlying transport release with actual-SDK
tests rather than assuming that closing a public SDK generator releases it.

To add a provider:

1. implement `ManagedModelProvider` under `src/daita/llm/providers/`, delegating
   common attempt and once-only close behavior to `llm._lifecycle`; use a
   provider-named package with `adapter.py`, `messages.py`, and `stream.py` when
   the protocol has substantial translation or stream grammar, while a small
   specialization of an existing adapter remains one module;
2. keep client ownership and native calls in the package's orchestration
   adapter, message translation in its message module, and substantial stream
   grammar in its decoder; the package `__init__.py` exports only the supported
   provider classes so the public provider import stays stable;
3. reuse `providers._fields` for identical native-field validation and import a
   new SDK only at client construction, with normalized pipx repair guidance;
4. add one immutable `ProviderDefinition`, including its explicit lazy
   construction callable, in `llm/provider_definitions.py`; factory, host, and
   TUI provider choices require no provider-specific edit;
5. add a bounded SDK dependency only when the protocol needs one; and
6. add focused definition, lazy-import, translation, lifecycle, accounting,
   routing, and architecture tests.

Unknown configured names continue through the explicit OpenAI-compatible path
and require a base URL; they never inherit built-in authentication, endpoint, or
profile privileges. Subscription subprocess mechanics live in
`providers.subscription_cli.process`, the canonical envelope in
`providers.subscription_cli.envelope`, and vendor flags, inspection, and
decoding in the package's Claude or Grok adapters. Preserve dispatch-time
revalidation even when it uses the same definition fact as lazy route preflight:
those checks protect different boundaries.

Do not add provider branches to `AgentLoop`.

## Architectural constraints

Current revision 2 has one `AgentLoop`, one `CapabilityRuntime`, one capability
registry, one catalog, one artifact store, one graph-aware jobs supervisor, one
routines supervisor, one SQLite state boundary, and one writer per agent home. The
graph coordinator is inside `jobs`; it does not add a second execution runtime. Do
not add a second
AgentLoop implementation, capability runtime, state store, transcript path, jobs
supervisor, agent-home writer, generic workflow engine, graph engine outside `jobs`,
dynamic executor/plugin registry, event bus, completion router, parallel recovery
service, policy DSL or generic scheduler. `RunSession` is isolated invocation state,
not a second loop or resumable conversation runtime. `RunAdmissionCoordinator`
coordinates host capacity, not graph truth. Audit events cannot drive replay.

Keep feature responsibilities in the existing concrete components. Avoid
middleware frameworks, lifecycle-hook systems, dynamic extension scanning,
background learning agents, vector stores, telemetry stores, and compatibility
frameworks for unreleased state.

## Development setup

Use a dedicated environment:

```bash
cd /path/to/daita-agents
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

`pyproject.toml` `[project].version` is the sole authored Daita release
identity. Runtime version displays read installed `daita-agents` distribution
metadata. Because editable metadata is an installation snapshot, rerun the
editable install command after changing that value or checking out a commit
with another value, before importing Daita or running tests. The agent-home
revision remains independent and changes only through an appended durable-format
migration.

Python 3.11 and 3.12 are supported.

## Tests and checks

Tests live under `tests/`, organized by their current production owner. Use
`docs/TESTING.md` for the directory map, marker meanings, live requirements and
examples. Put multi-owner public journeys in `tests/acceptance/`, external-service
cases in `tests/live/`, and extended offline soaks in
`tests/slow/`. Default pytest runs exclude live tests and extended soaks.

Before adding or changing a test:

1. Identify the current contract, its production owner, the defect to detect,
   and the observable failure.
2. Search that owner's suite and the relevant public acceptance cases for existing
   coverage. Extend an existing case or parameter family when it protects the
   same boundary; do not add another test merely for a new task or milestone.
3. Exercise the real component responsible for the contract. Mock only
   dependencies outside the behavior being asserted. Prefer real local state or
   adapters, or installed SDKs with injected transport when their behavior is
   the subject.
4. State expected behavior independently of the production implementation. A
   scripted model can verify framework execution and response forwarding; it
   cannot establish model reasoning, tool-choice quality or grounded answers.
5. Cover distinct boundaries and failure states without repeating the same
   scenario at every layer. Preserve authorization, cancellation, exact usage,
   rollback, receipt uncertainty, restart and no-replay distinctions.
6. Use a behavior-based filename and descriptive test name. Do not introduce
   phase, stage, milestone or obsolete MVP labels into collection names.

Each retained test must protect a distinct current contract or materially
different failure scenario. Remove a redundant test only after identifying its
retained coverage; remove an obsolete or tautological test with a recorded
reason. Replace a weak oracle before removing the only coverage of necessary
behavior. Never remove unique coverage to achieve a count target or hide a
failing or flaky test.

Do not assert documentation wording, comments, exact private variable names or
source-statement counts as proxies for behavior. Keep narrowly scoped structural
tests for real import/public-surface boundaries and exact schema/dependency
contracts where those are requirements.

Use explicit `tests.<owner>...` or `tests.support...` imports. Test modules and
support code must not import other `test_*.py` modules or `conftest.py`. Keep
reusable ordinary helpers in support modules and fixtures in the narrowest
applicable `conftest.py`. Do not introduce global auto-approval, permission
bypasses, synthetic tool-loading behavior or mutable shared agent state through
autouse fixtures. Shared test abstractions follow the same three-current-call-site
rule as other repository abstractions.

Use the existing `unit`, `contract`, `integration`, and `acceptance` markers
according to what the test actually exercises. Use `requires_llm`, `requires_db`,
`requires_network`, and `slow` for actual execution requirements. Preserve
explicit live authorization and per-run limits. Importing or collecting test or
support modules must not contact external services or resolve credentials.

`asyncio_mode = "auto"` is configured; do not add `@pytest.mark.asyncio`. Use
deterministic clocks/events and bounded waits where appropriate. Keep resource
cleanup explicit and fixtures narrowly scoped. Do not replace tests of actual
timeout or transport-release behavior with mocks of the mechanism being verified.

Run from the repository root:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/loop -v
.venv/bin/python -m pytest tests/architecture
.venv/bin/python -m black --check src tests
.venv/bin/python -m ruff check --select I .
.venv/bin/python -m mypy src/daita tests
```

Use focused red/green validation for changed contracts, then run the full
deterministic suite for cross-cutting changes and test-suite reorganizations. For
moves, splits or deletions, reconcile original and final collected node IDs,
parameter cases and markers; report intentional removals and their reasons. Do
not count a skipped, deselected or fake-only test as evidence of a live
integration.

Live model, remote service and external database execution requires explicit
authorization and credentials. Use the selection override and resource-specific
gates documented in `docs/TESTING.md`. Diagnose deterministic failures offline
before using external resources.

## Default production dependencies

The complete customer installation is:

```text
pipx install daita-agents
daita
```

`openai`, `anthropic`, `google-genai`, `asyncpg`, `sqlglot`, `httpx`,
`keyring`, `textual`, `rich`, `XlsxWriter`, and exact `duckdb==1.5.5` are
default production dependencies. `dev` is the only optional dependency group.

Default installation does not permit eager imports. Provider SDKs, `asyncpg`,
`sqlglot`, `httpx`, `keyring`, `textual`, `rich`, and DuckDB remain imported lazily
at the boundary that first needs them. XlsxWriter is imported only by the XLSX
renderer. Importing `daita` or `daita.cli`, and running headless commands, must
not load those integrations early.

A missing or damaged production dependency raises a normalized `ImportError`
that directs the user to:

```text
pipx reinstall daita-agents
```

Do not advertise an extras-based repair. Use `if TYPE_CHECKING:` for type-only
imports.

## Change discipline

Before adding a helper, abstraction, module, base class, builder, registry, or
shared utility:

1. identify the existing component responsible for the behavior;
2. state the broken or painful contract;
3. choose the smallest complete change;
4. confirm that it does not introduce a parallel responsibility; and
5. name the focused tests that detect drift.

Prefer a complete current behavior over placeholders for hypothetical features.
Add a shared abstraction only when at least three current call sites need it
and it removes more complexity than it adds. Avoid churn-only renames and broad
consistency edits.

For reliability failures, trace the problem to the responsible contract,
lifecycle, state, or trust boundary. Replace the incorrect mechanism there and
remove the obsolete path. Do not hide a broken design behind retries or
special cases.

When adding a source or data capability:

1. extend the existing adapter, catalog, and concrete domain;
2. declare stable `Capability`, `Executor`, and optional `ToolView` identities;
3. compose them in `EmbeddedAgent` through `CapabilityRegistry`;
4. keep discovery in the catalog and source I/O in the adapter;
5. validate current facts before I/O and retain connector guardrails;
6. return bounded schema-validated `ToolOutput`; and
7. add focused contract tests and one public end-to-end test.

Do not commit changes unless the task explicitly requests a commit.

## Key files

| File | Responsibility |
| --- | --- |
| `src/daita/__init__.py` | focused public exports |
| `src/daita/agent.py` | public persistent-agent facade |
| `src/daita/hosting/embedded.py` | composition and agent-home admission |
| `src/daita/loop/driver.py` | direct model/tool progression |
| `src/daita/loop/models.py` | run, transcript, limits, and exit records |
| `src/daita/capabilities.py` | declarations and registry |
| `src/daita/capability_runtime.py` | common execution mechanics |
| `src/daita/context.py` | model-request construction |
| `src/daita/domains/data/sql/` | catalog-scoped SQL validation |
| `src/daita/domains/mcp.py` | MCP projection and call-time rechecks |
| `src/daita/jobs/` | durable job records and supervision |
| `src/daita/routines/` | scheduled routine records and supervision |
| `src/daita/distribution/` | outcomes, destinations, and deliveries |
| `src/daita/artifacts/store.py` | committed artifact storage boundary |
| `src/daita/storage/sqlite.py` | durable state operations |
| `src/daita/storage/sqlite_schema.py` | current physical schema |
| `src/daita/storage/sqlite_codecs/` | strict current-record serializers |
| `src/daita/storage/home_migrations/` | whole-home revision registry and immutable transitions |
| `src/daita/hosting/home_upgrade.py` | staged upgrade, validation, rollback, and crash recovery |
| `src/daita/llm/routing.py` | normalized provider routing |
| `tests/architecture/test_boundaries.py` | architecture and public-surface boundaries |

## Product control surfaces

CLI and TUI routine inspection reuse the routine domain's current projections.
Approval summaries derive only from the exact validated request and retain its
complete bounded details. An explicit `confirmation_handler` reviews direct routine
create/update controls before mutation; typed Python callers without that callback
remain the authorizing owner. CLI create/update supplies this callback. Approval cannot grant missing connector permission.

Native approval documents contain the exact execution `arguments`, catalog-backed
`target`, and bounded `preview` review facts. The data domain reuses its existing
preflight preview; rendering performs no I/O. Execution arguments and intent digests
remain separate from review labels and samples. The runtime still approves and
rechecks one exact plan before reserving an effect. A positive count alone does not
prove the selected business entity; there is no second target authorization record.

Model-authored routine creation requires an explicit `run_immediately` boolean.
Model-authored updates accept omission or false and reject true; typed owner defaults
remain false. Immediate creation and later run-now use the existing owner/supervisor.

The source permission editor authors one exact table scope through existing
preview/apply APIs. Catalog-backed column/key choices are presentation, not a
second validator or proof of live database readiness. Apply confirms the complete
before/after state, including required read additions; it performs no source write.

Receipt list, inspect and human recovery controls call the existing Agent APIs.
Recovery never calls the model or an external executor. The original observation
and separate immutable resolution stay owned by SQLite. Product host status must
distinguish saved assignments, queued/running occurrences and the lifetime of the
currently open TUI or headless command; no host means no execution progress.
