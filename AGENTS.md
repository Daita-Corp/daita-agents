# AGENTS.md

Guide for AI assistants and humans working in this repository.

## Active architecture and scope

The active architecture is the small transcript-driven Daita MVP in
`src/daita/`, which is the sole production package and final Python import
namespace. The superseded root-level `daita/` implementation and isolated
replacement directory have been removed; do not recreate either as a parallel
owner.

When working in this repository:

- make production changes only in `src/daita/`;
- make tests under `tests/`, documentation under `docs/`, and examples under
  `examples/`;
- run commands from the repository root so Python imports the package through
  the configured `src` layout;
- do not restore deleted operation, workflow, event, memory, monitor, skill,
  extension, or telemetry implementations merely because they remain in Git
  history or stale bytecode caches; and
- preserve unrelated working-tree changes. Current source, tests, README, and
  focused design documents take precedence over superseded project ledgers.

Use this order of authority:

1. explicitly accepted task requirements;
2. current code under `src/daita/` and executable tests under `tests/`;
3. `README.md` and current user-facing documentation under `docs/`;
4. legacy code and historical documents, as reference only.

## What the product is

The product is a persistent, read-first data agent with explicitly enabled,
structured PostgreSQL updates, built around one direct loop:

```text
user message -> model -> zero or more tool calls -> ordered tool results
             -> model -> answer
```

The exact current-run transcript is the loop state. Tool failures are ordinary
model-visible tool results, allowing the model to correct a call on the next
step. Normal text from the model completes the run; there is no second
readiness, repair, verification, or synthesis pass.

The loop has only outer step, wall-time, token, and estimated-cost limits. The
MVP supports catalog-backed reads from SQLite and PostgreSQL, bounded reads
from one explicitly admitted local workspace, plus explicitly admitted
server-neutral remote MCP read tools. SQL access is validated against the
current catalog before source I/O; workspace file access is
descriptor-contained and revision-bound; each MCP call is revalidated against
its exact binding revision and remote identity.

Agent identity, source registrations, current catalog snapshots, exact run
transcripts, and terminal run results are persisted in a small SQLite state
database inside one agent home.

## Directory layout

```text
src/daita/
  agent.py             # focused public Agent facade
  hosting/embedded.py  # local composition, agent-home admission, locks
  loop/                # direct transcript progression and loop records
  llm/                 # canonical model records, routing, provider adapters
  capabilities.py      # tool declarations, registry, schema validation
  capability_runtime.py # sole domain-neutral model-to-execution boundary
  domains/             # static capability owners and learning guard state
  domains/data/        # data context, current validation, SQL, artifacts
  domains/mcp.py       # admitted remote MCP projection and call-time rechecks
  catalog/             # normalized current source/resource truth
  adapters/            # bounded source admission, discovery, and read I/O
  storage/sqlite.py    # sole durable state operation/admission boundary
  storage/sqlite_schema.py     # exact current physical schema
  storage/sqlite_codecs/       # explicit persisted-record family codecs
  storage/sqlite_migrations/   # development baseline and migration engine
  security/            # secret references and lazy secret resolution
  config.py            # immutable runtime/model configuration records
  workspace.py         # runtime-only local workspace admission record
  cli.py               # thin local CLI over the public embedded API
tests/                 # deterministic product tests
examples/              # offline examples
docs/                  # user-facing documentation only
pyproject.toml         # package dependencies, dev extra, entry point, and tools
```

Do not recreate a directory just because the old architecture had one. Add a
module only when a working vertical slice needs a clear owner.

Keep development-stage status, implementation ledgers, and North Star phase
closure in the authoritative North Star architecture document rather than
adding internal development records under `docs/`.

## Pre-production state policy

Until the user explicitly declares the first production release after the
North Star work is complete, persisted development state has no backward-
compatibility guarantee. During this pre-production period:

- keep exactly one current physical schema and one current shape per persisted
  record family;
- keep a codec discriminator at version `1` where the current codec uses one,
  but do not add historical decoders or multi-version branches;
- change the current schema, codecs, and single development baseline in place;
- do not add migrations, pre-ledger bridges, compatibility fixtures, aliases,
  or fallbacks for shapes produced only by unreleased code; and
- treat development agent homes as disposable when the current state shape
  changes.

The existing checksummed journal and staged-copy migration engine remain the
sole future production upgrade mechanism. At the explicitly approved first
production release, freeze the then-current schema and codec-v1 shapes as the
first immutable baseline. Only durable changes after that freeze add immutable
owner-local migrations or additional supported codec versions.

## Ownership and dependency direction

### Public API and composition

`daita.agent.Agent` is a thin public facade. It validates user-facing inputs
and delegates to `EmbeddedAgent`; it does not own planning, model semantics,
catalog truth, tool execution, or persistence.

`daita.hosting.embedded.EmbeddedAgent` is the composition root. It owns agent
home admission, the process-level writer lock, in-process run/mutation locks,
construction of the catalog, capabilities, context builder, tool runtime, and
loop, and orderly close. Composition belongs here rather than in the loop or a
new dependency-injection framework.

### One direct loop

`daita.loop.driver.AgentLoop` owns only:

- starting and finishing one run transcript;
- model calls;
- appending normalized assistant and tool messages;
- calling the injected `ToolRuntime`;
- enforcing the outer budgets; and
- returning one terminal `LoopExit`.

It depends on the small `ModelProvider`, `ContextBuilder`, `ToolRuntime`, and
`TranscriptStore` protocols. Keep provider wire formats, catalog operations,
SQL validation, source I/O, policy, and feature-specific lifecycle state out of
the loop.

Every model-requested tool call must receive exactly one result. Results must
be returned in the original call order even when independent reads execute
concurrently. One failed call must not suppress other calls.

### Context and trust

`daita.domains.data.context.DataContextBuilder` owns construction of the model
request from the current transcript, current catalog context, projected tool
definitions, and model profile. It labels catalog and tool content as
untrusted data, keeps complete tool exchanges together, and must never turn
catalog text or data values into instructions.

The catalog is authoritative for current source/resource identity, schemas,
facets, relationships, and freshness. Current validated tool results are
authoritative for the values they actually returned. Model text, historical
assistant claims, preferences, and procedural guidance cannot establish either
kind of fact.

### Capabilities and execution

`daita.capabilities.CapabilityRegistry` owns immutable capability, tool-view,
executor, and domain-owner identity and exact resolution. It projects tool
schemas and validates arguments and executor output. Tools are a presentation
over capabilities, not an alternate execution path.

`daita.capability_runtime.CapabilityRuntime` is the sole production
model-to-execution boundary. It owns only cross-domain execution mechanics:
ordered bounded batches, exact registry resolution, common schema validation,
fixed effect governance, exact executor dispatch, artifact commit, result
bounds, sensitivity/provenance enforcement, error normalization, and
observation. Its order is:

1. ask each statically composed domain to project applicable tools;
2. reject a call that was not projected;
3. resolve tool view, capability, and domain owner by exact registry identity;
4. validate model arguments against the declared schema;
5. ask the owner to bind arguments and revalidate current admission;
6. resolve the exact registered executor;
7. apply the fixed side-effect preflight, approval, recheck, and lock branch;
8. execute exactly once;
9. ask the owner to finalize capability-specific result semantics;
10. validate output, artifact, classification, provenance, and bounds; and
11. return exactly one structured result per requested call in original order.

Static owners are `DataCapabilityDomain` for catalog, query, update, and local
file behavior; `MemoryCapabilityDomain`; `SkillCapabilityDomain`;
`SemanticCapabilityDomain`; and `ArtifactCapabilityDomain`.
`LearningCandidateGuard` owns the bounded transient learning selection and
mutation-success state shared by those owners. Applicability and current-state
validation remain with these concrete owners, never the common runtime.

Do not let `AgentLoop`, `Agent`, a tool view, or model-authored text call source
clients or executors directly. Do not infer capability behavior from tool-name
strings when stable capability metadata owns it.

All currently projected data capabilities are non-side-effecting reads except
the explicitly enabled structured PostgreSQL update. One cardinality-independent
plan covers single-row and bulk selections and retains resource-scoped
readiness, current admission rechecks, an exact target-set preview and
fingerprint, once-only approval, transactional drift detection, exact affected
count, and an immutable receipt path. Arbitrary SQL mutation and every other
external data write are outside the MVP. A new data write beyond this exact
exception requires an explicit design for
validation, authorization, transactionality, idempotency, uncertain outcomes,
and recovery; approval alone is not such a design.

Artifact continuity is model-led through the existing capability/runtime path.
`artifact_list` exposes only bounded safe metadata for the current conversation;
it is not a public Python, CLI, TUI, or browser inventory. `artifact_read`
returns only bounded previews for an exact known artifact ID owned by the
current agent home, including an ID returned by `job_read_results`; it never
provides an agent-wide inventory. `artifact_convert` remains scoped to the
current conversation and currently supports only a verified Daita-generated
XLSX `Data` snapshot to CSV. Conversion reads committed artifact bytes at the
artifact-store boundary, inherits runtime-bound provenance and sensitivity,
records its parent artifact ID, and commits through the normal artifact policy.
Do not add prompt-intent classifiers, historical artifact-reference
projections, a current-file pointer, raw model paths/bytes, or a second
execution path to improve file reference continuity.

### Catalog and adapters

`daita.catalog` owns normalized structural truth and catalog search, inspection,
and traversal. Data-domain code consumes catalog contracts rather than building
a second schema graph or querying source clients for planning facts.

`daita.adapters` owns source-specific admission, containment, discovery,
freshness checks, and I/O, the descriptor-contained local-workspace backend,
plus the single bounded server-neutral Streamable HTTP MCP protocol client.
SQLite paths must be absolute and bounded; workspace access uses only admitted
workspace-relative logical paths and resists symlink/path escape. PostgreSQL
and MCP credentials remain secret references and integration SDK use stays
behind the execution boundary.

SQL validation belongs in `daita.domains.data.sql`; connector guardrails still
apply at execution. Do not duplicate either system in a generic policy layer.

### Persistence

`daita.storage.sqlite.SQLiteStateStore` persists only state used by the MVP:
identity, sources, current catalog snapshots, run transcripts and terminal
results, semantic annotations, learning review state, immutable database-write
receipts, explicit source read scopes, exact PostgreSQL update scopes, and
independently keyed MCP server binding aggregates. It is the sole owner of the
checksummed `state_migrations` journal, explicit persisted-record codecs, and
the exact current schema. Before the first production freeze, the journal
contains one mutable development baseline and no compatibility path for older
development state. After that freeze, migrations run only on a verified staged
copy under the existing agent-home writer boundary, validate their source and
target schemas, and atomically replace the active database only after complete
target validation. Put each post-freeze durable change in one owner-local
migration file; never edit a released ID/checksum or create migration ownership
in hosting, the loop, or a new runtime.

Source read access is owned only by `source_read_scopes`, and PostgreSQL update
access is owned only by `postgresql_update_scopes`, never by source connection
JSON. Public source registrations expose no permission compatibility
projection; connection reconstruction remains fail-closed, refresh preserves
exact scopes, and detach revokes both scope families atomically.

State mutation must remain atomic and cancellation-safe. Preserve the single
agent-home writer boundary. Do not add event sourcing, replay projections,
checkpoint reconstruction, or another state-store abstraction around SQLite.

### Models and providers

Canonical messages, tool calls/results, usage, requests, responses, and errors
live in `daita.llm.models` and `daita.llm.errors`. Provider-native payloads end
inside provider adapters. `daita.llm.routing` owns retry/fallback decisions from
normalized failures; the generic loop does not retry a whole run or understand
provider-specific failures.

## Architecture exclusions

### Accepted Phase B durable-job slice

North Star Phase B is accepted for implementation, but only through these
existing owners and one concrete job kind:

- one bounded, feature-owned `JobRun` aggregate may represent each independent
  durable job lifecycle, with attempts, claims, fencing, cancellation intent,
  receipts, external observations, validated result/artifact references, and
  terminal-observation state embedded in its current codec-v1 shape;
- one Stage B job owner may admit frozen jobs and own bounded list, inspect,
  result, cancel, and conditional lifecycle transitions;
- one bounded Daita job supervisor may run under the existing open
  `EmbeddedAgent` and single-writer boundary, claim independent jobs within
  exact global/per-agent/per-source limits, fence stale claims, and recover
  safely on reopen;
- `CapabilityRuntime` may expose one trusted typed internal request for that
  supervisor, reusing the ordinary registry resolution, validation,
  governance, execution, artifact, sensitivity, provenance, result-bound, and
  observation path without a model envelope or recursive runtime call; and
- the data domain may own only the first job-kind-specific
  `start_data_profile` capability and internal-only `data_profile` execution
  capability. The start capability freezes the exact non-secret job
  specification and persists the exact execution capability ID and immutable
  registry contract digest before returning its bounded receipt.

The agent identity is the job authorization boundary. Model and public
lifecycle reads and cancellation may address that agent's jobs from any of its
conversations; `JobRun.conversation_id` and `origin_run_id` are immutable origin
provenance, not access gates. `job_list` is a direct core capability even when
the agent owns no jobs. Once jobs exist, `job_inspect` and `job_read_results`
are direct core capabilities; `job_cancel` remains deferred, effect-governed,
and projected only while cancelable work exists. Known-ID result recovery uses
`job_read_results` directly, while `job_inspect` is reserved for lifecycle,
attempt, execution, and failure details. Cross-agent lookup fails without
exposing job metadata.

`Capability` metadata now separates data access from operational effect.
Starting or cancelling durable work is an operational effect even when the
job's data access is read-only; this distinction remains immutable registry
metadata and does not authorize a policy registry or DSL.

Daita execution is the default. Phase B connected-executor work is limited to
deterministic offline conformance for an exact explicitly selected profile,
including revocation/drift, uncertain start/cancel responses, reconciliation,
result bounds, and no fallback. Production exposes no placeholder external
profile and does not claim support for a real external executor until a later
separately accepted connector slice is certified.

Phase B by itself does not authorize a generic scheduler, arbitrary job
dispatcher, workflow or execution graph, universal task/work abstraction,
job-kind switch or handler registry, dynamic registration, plugin executor,
completion router, recovery service, event bus, resident daemon, client/server
split, competing writer, multi-host queue, resumable model state, or later-stage
scaffolding. The accepted Stage D1 slice below is the sole exception for its
exact scheduled-routine owner and resident single-writer host. Work still
pauses whenever no admitted `EmbeddedAgent` host is open; D1 may keep that same
host open in a dedicated headless process but may not create another execution
or state owner.

### Accepted Stage D1 general scheduled-read slice

North Star Stage D1 is accepted for implementation, but only as the first
general scheduled-read vertical slice. It extends the implemented Stage C
invocation, immutable scope, bounded run, terminal convergence, and
conversation-inbox contracts. It does not authorize D2 external channel
delivery, D3 recurring ingestion or any other scheduled effect, Stage E graphs,
or a generic scheduler framework.

The accepted D1 physical shape is:

- one feature-owned `ScheduledRoutine` codec-v1 aggregate and one
  `RoutineOccurrence` codec-v1 aggregate, persisted only in
  `scheduled_routines` and `routine_occurrences` under `SQLiteStateStore` and
  the single mutable development baseline;
- one `RoutineOwner` that admits exact foreground-authorized, self-contained
  instructions and owns bounded create, list, inspect, update, pause, resume,
  run-now, disable, expiry, budget, and lifecycle transitions;
- one bounded `RoutineSupervisor` under the existing `EmbeddedAgent` and
  single-writer boundary that computes due slots, conditionally claims one
  occurrence, fences stale claims, reserves one run, invokes the ordinary
  `AgentLoop`, and idempotently finalizes the occurrence and one existing
  conversation-inbox delivery;
- typed `once`, anchored `interval`, and calendar schedules with exact IANA
  timezone, explicit daylight-saving gap/overlap behavior, and bounded `skip`
  or `latest_only` misfire handling; raw prompt text is never evaluated to
  decide that a slot is due;
- one `scheduled_routine` run origin and schedule identity fields added to the
  existing `RunStartEnvelope` and `ExecutionScope`; the user-authorized
  instruction is projected as instruction content, never as a new user turn or
  as authority;
- one orthogonal registry-owned `AutomationEligibility` value. Its safe
  default is `interactive_only`; D1 may explicitly mark an existing
  effect-free capability `scheduled_direct` only at its static declaration and
  only after deterministic unattended tests. No tool name, access mode, MCP
  annotation, prompt, or remote schema may infer eligibility;
- foreground routine-management capabilities owned by one static
  `RoutineCapabilityDomain`. They use the existing effect-governance and exact
  approval branch under one fixed `manage_scheduled_routine` operational
  effect, are themselves `interactive_only`, and are never included in a
  scheduled run's capability ceiling;
- `always` reporting plus, for the second D1 proof, one exact internal
  data-domain resource-revision observation precheck. It executes through the
  existing trusted runtime request, compares a bounded canonical observation,
  and can finalize a no-change occurrence with zero model call. D1 does not add
  a condition DSL, precheck registry, or model-authored executable predicate;
- optional exact skill binding only after `SkillStore` can retain and reload
  the bounded pinned content by digest. A routine without that retained-content
  contract cannot claim a skill binding, and no second skill lifecycle or
  executable plugin path is allowed; and
- one headless resident-host entry point that keeps the same `EmbeddedAgent`
  composition open for an exact agent home. It may be supervised by the local
  operating environment, but D1 adds no remote API, IPC framework, generic
  daemon manager, multi-host queue, or competing writer. A foreground TUI/CLI
  process and the headless host must hand off the same writer admission rather
  than opening the home concurrently.

D1 scheduled execution is read-only: the routine's frozen scope permits only
explicit `scheduled_direct` capabilities with `OperationalEffect.NONE`, uses
the current conversation inbox as its only delivery destination, and cannot
start or cancel a Stage B job, create another routine, invoke an external
delivery adapter, write data, call an MCP write tool, or submit a graph. Routine
creation or revision may reference a prior successful run as evidence, but it
must persist one explicit self-contained instruction and exact current
resource/capability envelope; hidden chat references and a run-bound
`tool_ref` are invalid durable identity.

D1 adds no `Schedule`, `Trigger`, `Invocation`, `Grant`, `Claim`, `Budget`,
`Audit`, or `Disposition` table. Those logical values remain embedded in the
two routine-owned aggregates. It adds no generic cron command executor,
monitor type hierarchy, workflow, event bus, completion router, recovery
service, policy DSL, second model pass, second loop, second runtime, dynamic
handler registry, or compatibility path. The North Star Stage D1 physical
design ledger and implementation plan are normative for file ownership,
atomic transitions, sequencing, and exit tests.

Do not add or restore these mechanisms in order to implement an MVP feature:

- `Operation`, `Task`, `Workflow`, `Checkpoint`, `Lease`, or resume runtimes;
- `DbRuntime`, `RuntimeKernel`, a second executor boundary, or a second agent
  loop;
- readiness evaluators, repair workflows, verifier/synthesis passes, or
  persisted pending plans;
- event sourcing, durable event buses, projections, subscriptions, or replay;
- session runtimes, LLM-authored history summaries, or compression
  checkpoints;
- policy registries, policy DSLs, fact graphs, or generic governance engines;
- middleware, interceptors, lifecycle-hook frameworks, extension scanning, or
  plugin auto-install;
- background learning/review agents, memory provider registries, vector stores,
  skill activation state machines, or any monitor/scheduler outside the exact
  accepted Stage D1 routine owner;
- trace trees, telemetry stores/exporters, or versioned telemetry payloads; or
- interaction-protocol version frameworks, generic compatibility decoders,
  root-framework v1 fallbacks, or migration machinery outside the existing
  SQLite state-store owner.

Fix a broken contract at its existing owner. Do not work around it with a
parallel abstraction or a compatibility path to root v1.

## Planned conversations, memory, skills, approval, and observation

`docs/MVP_MEMORY_SKILLS_GOVERNANCE_OBSERVABILITY_PLAN_2026-07-21.md`
describes proposed additions to the trimmed MVP. It does not mean the features
already exist. Implement them only when the task places the corresponding
stage in scope, and preserve the direct architecture:

- conversation continuity is a bounded projection of completed prior runs,
  not a session runtime or resumable loop;
- memory is bounded advisory `MEMORY.md` and `USER.md` content, not structural
  truth, evidence, policy, or a retrieval system;
- skills are bounded user-authorized `SKILL.md` procedures with progressive
  disclosure, not plugins, executors, or catalog resources;
- learning is the ordinary foreground loop using explicit memory/skill tools,
  not a worker or second model pass;
- governance is a fixed branch immediately before a side effect at the
  existing `CapabilityRuntime` boundary, not a policy engine;
- approval is once-only, in-process, and bound to exact frozen arguments; it
  does not create pending state or resume APIs; and
- observation is one best-effort callback that cannot direct execution and is
  not a durable event or telemetry subsystem.

Extend `CapabilityRegistry`, `CapabilityRuntime`, the appropriate static domain,
`DataContextBuilder`, `EmbeddedAgent`, and `SQLiteStateStore` only for the
concerns they already own. Do not wrap them in a replacement workflow runtime.

## Development setup

Use an environment dedicated to this package:

```bash
cd /path/to/daita-agents
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Python 3.11 and 3.12 are the supported versions.

## Running tests

Run commands from the repository root:

```bash
# Current deterministic suite
pytest

# One focused file
pytest tests/test_loop.py -v

# Exclude live model and external database tests as the suite grows
pytest tests/ -m "not requires_llm and not requires_db"

# Formatting and static checks
python -m black --check src tests
python -m mypy src/daita tests
```

`asyncio_mode = "auto"` is configured in `pyproject.toml`; do not add
`@pytest.mark.asyncio` to individual tests.

Test markers are `unit`, `contract`, `integration`, `acceptance`,
`requires_llm`, and `requires_db`. Live tests require explicit authorization
and credentials. Do not turn a deterministic failure into repeated paid model
runs; isolate and repair it offline first.

Use focused red/green tests while implementing a coherent vertical slice.
Before handoff, run the narrowest relevant suite and then the complete
deterministic suite when practical. Architecture changes should also run
`tests/test_architecture.py`.

## Critical: default production dependencies stay lazy

The customer installation is the complete application:

```text
pipx install daita-agents
daita
```

`openai`, `anthropic`, `google-genai`, `asyncpg`, `sqlglot`, `httpx`, `keyring`,
`textual`, `rich`, `XlsxWriter`, and exact `duckdb==1.5.5` are default
production dependencies.
`dev` is the only optional dependency group; do not restore provider, keychain,
database, parser, CLI, recommended, complete, aggregate, or other customer
extras.

Default installation does not authorize eager imports. Import provider SDKs,
`asyncpg`, `sqlglot`, `httpx`, `keyring`, `textual`, Rich, and DuckDB only inside the
provider/client or terminal-selection boundary that first needs them, and
XlsxWriter only inside the XLSX renderer boundary—never at module import time.
Importing `daita` or `daita.cli`, and running headless commands, must not load
those integration packages before their execution boundary is used; all are
still imported lazily.

If a default production dependency is missing or damaged, raise a normalized
`ImportError` that points to the application repair command:

```text
pipx reinstall daita-agents
```

Do not advertise an extras-based repair. Use `if TYPE_CHECKING:` for type-only
imports when needed.

## Change discipline

Before introducing a helper, abstraction, module, base class, builder,
registry, or shared utility:

1. identify the current owner of the behavior;
2. explain the broken or painful contract;
3. choose the smallest change that fixes it;
4. confirm the change does not introduce a parallel owner; and
5. name the focused tests that catch behavior drift.

Prefer one complete vertical slice over scaffolding for hypothetical later
features. Do not add a shared abstraction solely to reduce repetition unless
at least three current call sites need it and it removes more complexity than
it adds. Avoid churn-only renames and broad consistency edits.

For bugs and reliability failures, trace the issue to the responsible contract,
state owner, lifecycle, or trust boundary. Replace an incorrect mechanism at
that owner and remove the obsolete path it supersedes. Do not stack retries,
special cases, or defensive checks on a broken design.

## Adding a model provider

1. Implement the `ModelProvider` contract under
   `src/daita/llm/providers/`.
2. Keep its wire models and translation inside that adapter.
3. Lazy-import its SDK at first use and provide normalized pipx repair guidance.
4. Register construction in `src/daita/llm/factory.py`.
5. Add the bounded SDK version to default production dependencies when needed.
6. Add focused provider translation/error tests and routing tests where the
   normalized failure behavior changes.

Do not add provider branches to `AgentLoop`.

## Adding a source or data capability

1. Extend the existing adapter/catalog/data-domain owner before creating a new
   layer.
2. Declare a stable `Capability`, `Executor`, and optional `ToolView` in the
   owning domain.
3. Compose the declaration in `EmbeddedAgent` through `CapabilityRegistry`.
4. Keep structural discovery in the catalog path and source I/O in the
   adapter/backend.
5. Add concrete validation before executor I/O and retain connector-level
   guardrails.
6. Return bounded, schema-validated `ToolOutput`; let the owning domain
   normalize expected failures for `CapabilityRuntime` to render as structured
   model-visible results.
7. Add focused unit/contract coverage plus one public vertical-slice test.

Do not create a plugin base-class hierarchy, an alternate catalog, or a source-
specific agent loop.

## Key files

| File | Purpose |
| --- | --- |
| `src/daita/__init__.py` | focused public exports |
| `src/daita/agent.py` | public persistent-agent facade |
| `src/daita/hosting/embedded.py` | composition and agent-home ownership |
| `src/daita/loop/driver.py` | sole direct model/tool progression loop |
| `src/daita/loop/models.py` | run, transcript, limits, and exit records |
| `src/daita/capabilities.py` | capability declarations and registry |
| `src/daita/capability_runtime.py` | sole common model-to-execution runtime |
| `src/daita/workspace.py` | explicit runtime-only local workspace admission |
| `src/daita/adapters/local_workspace.py` | descriptor-contained bounded workspace search/read backend |
| `src/daita/adapters/mcp.py` | bounded server-neutral Streamable HTTP MCP protocol client and records |
| `src/daita/domains/mcp.py` | static MCP capability owner and call-time binding rechecks |
| `src/daita/domains/learning.py` | transient learning mutation guard |
| `src/daita/domains/data/context.py` | current model request construction |
| `src/daita/domains/data/controller.py` | data projection and current-state validation |
| `src/daita/domains/data/export_capabilities.py` | artifact capability owner and executors |
| `src/daita/memory/capabilities.py` | memory capability owner and executor |
| `src/daita/skills/capabilities.py` | skill capability owner and executors |
| `src/daita/semantics.py` | semantic capability owner and records |
| `src/daita/domains/data/sql/` | catalog-scoped SQL validation |
| `src/daita/catalog/service.py` | normalized catalog lifecycle |
| `src/daita/storage/sqlite.py` | sole durable state operation/admission boundary |
| `src/daita/storage/sqlite_schema.py` | exact physical schemas and validators |
| `src/daita/storage/sqlite_codecs/` | explicit durable record-family codecs |
| `src/daita/storage/sqlite_migrations/` | development baseline and staged-copy migration engine |
| `src/daita/llm/routing.py` | normalized model retry/fallback ownership |
| `tests/test_architecture.py` | prohibited-system and public-surface checks |

## Things to avoid

- Do not create a second root-level `daita/` package or another replacement
  source tree beside `src/daita/`.
- Do not restore deleted architecture to satisfy obsolete tests or documents.
- Do not add top-level integration SDK imports or unplanned dependencies.
- Do not split supported production integrations back into customer extras.
- Do not let untrusted catalog, file, row, memory, or skill text become runtime
  instructions or authorization.
- Do not weaken absolute-path, symlink, containment, SQL-scope, secret, output-
  bound, or single-writer protections.
- Do not add `@pytest.mark.asyncio`; it is configured globally.
- Do not skip formatting and relevant deterministic tests before handoff.
