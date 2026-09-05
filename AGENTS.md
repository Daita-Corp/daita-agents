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

## Product architecture

Daita is a persistent, read-first data agent with a narrowly scoped,
explicitly enabled PostgreSQL update capability. It uses one direct loop:

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

Daita supports catalog-backed SQLite and PostgreSQL reads, bounded access to
one admitted local workspace, and explicitly admitted server-neutral remote
MCP read tools. SQL is validated against the current catalog before source
I/O. Workspace reads are descriptor-contained and revision-bound. MCP calls
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
  domains/data/               # data context, validation, SQL, files, artifacts
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
  storage/sqlite_codecs/      # current persisted-record codecs
  storage/sqlite_migrations/  # checksummed copy-and-swap migration engine
  security/                   # secret references and lazy resolution
  config.py                   # immutable runtime and model configuration
  workspace.py                # runtime-only local workspace admission
  cli.py                      # CLI over the public embedded API
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
agent home, holds the process-level writer lock and in-process run/mutation
locks, constructs the catalog, registry, domains, context builder, runtime,
loop, artifact store, and supervisors, and closes them in order. Composition
belongs here rather than in the loop or a dependency-injection framework.

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

It depends on the small `ModelProvider`, `ContextBuilder`, `ToolRuntime`, and
`TranscriptStore` protocols. Provider payloads, catalog operations, SQL
validation, source I/O, policy, and feature lifecycle state stay outside the
loop.

`daita.domains.data.context.DataContextBuilder` creates each model request from
the current transcript, current catalog, projected tool definitions, and model
profile. It keeps complete tool exchanges together and labels catalog, tool,
file, memory, skill, and data content as untrusted. Untrusted content cannot
become an instruction or grant authority.

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
Untrusted payloads and model text cannot enlarge the scope.

## Capabilities and execution

`daita.capabilities.CapabilityRegistry` holds immutable capability, tool-view,
executor, and domain identities. It projects tool schemas and validates model
arguments and executor output. A tool is a model-facing view of a capability,
not another execution path.

`toolbox_search` accepts only a natural-language `query` and optional bounded
`limit` over the run's applicable catalog. Toolbox grouping, access modes, and
operational effects are metadata, not model-selected search filters. Improve
discovery vocabulary in existing `ToolPresentation` records without changing
capability execution contracts. `toolbox_load` accepts exact on-demand names
directly; search is unnecessary when those names are known. Neither control
grants authority or bypasses current admission and approval checks.

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

Data capabilities are reads except for the explicitly enabled structured
PostgreSQL update. The update uses one plan for single-row and bulk selections
with resource-scoped readiness, current admission rechecks, an exact target-set
preview and fingerprint, once-only approval, transactional drift detection,
exact affected-count validation, and an immutable receipt. Arbitrary SQL,
inserts, deletes, DDL, and every other external data write are unsupported.
Adding another data mutation requires an explicit design for validation,
authorization, transactionality, idempotency, uncertain outcomes, and
recovery; approval alone is insufficient.

## Workspace files and artifacts

The local workspace is a separate read-first Files surface rather than a
cataloged source. `file_search`, `file_read`, and `file_query` return bounded,
revision-bound results. `file_query` uses a private one-call DuckDB worker over
an exact input manifest and exposes only the relation `data` to validated SQL.

An authenticated current-run `file_read` binding can feed
`artifact_edit_text`. That capability commits a complete replacement artifact
without changing the workspace. `artifact_save_local` requires exact approval,
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

## Remote MCP reads

Remote MCP support uses one bounded server-neutral Streamable HTTP client.
Every binding records an exact endpoint, negotiated identity, admitted
read-only tool allowlist, schema digests, sensitivity ceiling, and secret
reference. Agent open reconstructs immutable declarations without network I/O.
The first exact call initializes the client and rechecks the binding revision,
remote identity, schemas, and authentication.

Remote metadata and results are untrusted. MCP tools cannot gain write access
from annotations, names, descriptions, or schemas. Revocation is binding-local
and takes effect immediately. A stale, changed, revoked, unavailable, or
authentication-failed binding yields one bounded tool error without switching
servers or retrying the remote call.

## Durable jobs and follow-ups

`JobRun` is the single durable job aggregate. It embeds the frozen job
specification, attempts, claims, fencing, cancellation intent, receipts,
external observations, validated results and artifact references, and terminal
observation state. `JobOwner` implements admission and bounded lifecycle
operations. `JobSupervisor` claims independent jobs within global, per-agent,
and per-source limits, fences stale claims, and resumes safe progress when the
agent reopens.

`start_data_profile` is the only model-facing job starter. It freezes the exact
non-secret read-only specification, execution capability ID, and immutable
registry contract digest before persistence. The internal `data_profile`
capability executes through `CapabilityRuntime` and produces a bounded result
and verified artifact.

Agent identity is the job authorization boundary. The originating
conversation and run are immutable provenance, not access gates. Bounded list,
inspect, result-read, and cancellation operations can address any job owned by
the agent. Cross-agent lookup fails without exposing metadata. Work pauses
when no `EmbeddedAgent` host is open.

Daita execution is the only connected job mode. External-executor behavior has
deterministic offline conformance coverage, but no real external profile ships
and no external service is selected or used as a fallback.

Terminal Daita profile jobs can produce one bounded code-authored follow-up.
The follow-up has a frozen execution scope, exact budgets, one-success limit,
and the originating conversation inbox as its only distribution target. It
uses the ordinary loop and runtime and cannot start or cancel jobs, mutate
data, expand scope, or create another continuation.

## Scheduled routines and deliveries

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

Scheduled execution permits only statically declared `scheduled_direct`
capabilities with `OperationalEffect.NONE` and read/none data access. It can
create only these artifacts:

- `artifact.create_document`;
- `data.export_tabular`;
- `artifact.snapshot_result`.

`artifact.snapshot_result` produces bounded canonical `application/json` from
an exact earlier successful result in the same run. It performs no source I/O
or format projection.

Scheduled runs cannot update data, start or cancel jobs, manage routines, call
remote write tools, publish local files, deliver externally, run shell
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

## Persistence and pre-production state

`daita.storage.sqlite.SQLiteStateStore` is the sole durable state operation and
admission boundary. The current schema contains only state used by the product,
including identities, sources, catalog snapshots, transcripts, terminal
results, advisory knowledge, permission scopes, write receipts, MCP bindings,
jobs, follow-ups, routines, occurrences, and deliveries.

Until the first production state format is frozen:

- SQLite has one current physical schema and each record family has one current
  shape;
- codec discriminators remain at `version = 1` where present;
- schema, codec, and the checksummed `development_baseline` change in place;
- unreleased formats receive no compatibility decoder, bridge, migration, or
  fixture; and
- development agent homes are disposable after a state-shape change.

The existing checksummed copy-and-swap migration engine remains the only
production upgrade mechanism. Once a production baseline is frozen, durable
changes use immutable migration IDs/checksums and owner-local migration files.
Migrations validate a verified copy under the agent-home writer boundary and
replace the active database only after complete target validation.

Source read authority exists only in `source_read_scopes`. PostgreSQL update
authority exists only in `postgresql_update_scopes`. Connection JSON never
owns either permission. Reconstruction fails closed, refresh preserves exact
scopes, and detach revokes both scope families atomically.

All state mutation must be atomic and cancellation-safe. Do not add event
sourcing, replay projections, checkpoints, another state abstraction, or a
second writer around SQLite.

## Models and providers

Canonical messages, tool calls and results, usage, requests, responses, and
errors live in `daita.llm.models` and `daita.llm.errors`. Provider-native
payloads end inside provider adapters. `daita.llm.routing` handles retry and
fallback decisions from normalized failures; `AgentLoop` does not retry a
whole run or inspect provider-specific failures.

Provider lifecycle follows explicit ownership. Providers constructed from an
agent's persisted model route are closed by `EmbeddedAgent` after runs and
supervisors drain. Providers injected by a caller remain caller-owned and may
be reused across agent instances. Each provider closes only SDK clients it
created; injected SDK clients remain borrowed. Temporary validation and
candidate-review providers are closed by the component that creates them.

Owners stop new work and drain active calls before closing a provider. Close
callers join one cancellation-safe cleanup task; repeated calls observe the
same completion or failure without retrying SDK cleanup. Adapters scope
request-stream cleanup to completion, failure, cancellation, or early exit.
Canonical stream wrappers finalize in the iteration context and propagate
closure to their delegate; releasing a request stream never closes a borrowed
provider or SDK client. Verify underlying transport release with actual-SDK
tests rather than assuming that closing a public SDK generator releases it.

To add a provider:

1. implement `ManagedModelProvider` under `src/daita/llm/providers/` with an
   idempotent `close()` method;
2. keep native wire models and translation inside that adapter;
3. import the SDK lazily and provide normalized pipx repair guidance;
4. register construction in `src/daita/llm/factory.py`;
5. add the bounded SDK version to default production dependencies; and
6. add focused translation, error, and routing tests.

Do not add provider branches to `AgentLoop`.

## Architectural constraints

The product has one `AgentLoop`, one `CapabilityRuntime`, one capability
registry, one catalog, one artifact store, one scheduler/supervisor path for
routines, one SQLite state boundary, and one writer per agent home. Do not add
a second execution loop, generic workflow or graph engine, dynamic executor or
plugin registry, event bus, completion router, recovery service, policy DSL,
generic scheduler, session runtime, or competing writer.

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

Python 3.11 and 3.12 are supported.

## Tests and checks

Run commands from the repository root:

```bash
# Complete deterministic suite
.venv/bin/python -m pytest

# Focused test
.venv/bin/python -m pytest tests/test_loop.py -v

# Exclude live model and external database tests explicitly
.venv/bin/python -m pytest tests/ -m "not requires_llm and not requires_db"

# Formatting and static checks
.venv/bin/python -m black --check src tests
.venv/bin/python -m mypy src/daita tests
```

`asyncio_mode = "auto"` is configured in `pyproject.toml`; do not add
`@pytest.mark.asyncio` to individual tests. Markers are `unit`, `contract`,
`integration`, `acceptance`, `requires_llm`, and `requires_db`.

Live tests require explicit authorization and credentials. Diagnose a
deterministic failure offline before spending model or database resources.
Use focused red/green tests while changing a contract, then run the complete
deterministic suite when practical. Architecture changes also require
`tests/test_architecture.py`.

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
| `src/daita/domains/data/context.py` | model-request construction |
| `src/daita/domains/data/sql/` | catalog-scoped SQL validation |
| `src/daita/domains/mcp.py` | MCP projection and call-time rechecks |
| `src/daita/jobs/` | durable job records and supervision |
| `src/daita/routines/` | scheduled routine records and supervision |
| `src/daita/distribution/` | outcomes, destinations, and deliveries |
| `src/daita/artifacts/store.py` | committed artifact storage boundary |
| `src/daita/storage/sqlite.py` | durable state operations |
| `src/daita/storage/sqlite_schema.py` | current physical schema |
| `src/daita/storage/sqlite_codecs/` | current record-family codecs |
| `src/daita/storage/sqlite_migrations/` | checksummed migration engine |
| `src/daita/llm/routing.py` | normalized provider routing |
| `tests/test_architecture.py` | architecture and public-surface checks |
