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
3. `README.md` and current focused design documents under `docs/`;
4. legacy code and historical documents, as reference only.

## What the product is

The product is a persistent, read-first data agent with one narrowly bounded,
explicitly enabled PostgreSQL one-row update, built around one direct loop:

```text
user message -> model -> zero or more tool calls -> ordered tool results
             -> model -> answer
```

The exact current-run transcript is the loop state. Tool failures are ordinary
model-visible tool results, allowing the model to correct a call on the next
step. Normal text from the model completes the run; there is no second
readiness, repair, verification, or synthesis pass.

The loop has only outer step, wall-time, token, and estimated-cost limits. The
MVP supports catalog-backed reads from SQLite, PostgreSQL, CSV, and JSON. SQL
and file access are validated against the current catalog before source I/O.

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
  domains/data/        # context building, tool runtime, SQL validation
  catalog/             # normalized current source/resource truth
  adapters/            # bounded source admission, discovery, and read I/O
  storage/sqlite.py    # sole durable state operation/admission boundary
  storage/sqlite_schema.py     # exact current and supported historical schemas
  storage/sqlite_codecs/       # explicit persisted-record family codecs
  storage/sqlite_migrations/   # immutable journal, baseline, and preledger bridge
  security/            # secret references and lazy secret resolution
  config.py            # immutable runtime/model configuration records
  cli.py               # thin local CLI over the public embedded API
tests/                 # deterministic product tests
examples/              # offline examples
docs/                  # focused design documents
pyproject.toml         # package dependencies, dev extra, entry point, and tools
```

Do not recreate a directory just because the old architecture had one. Add a
module only when a working vertical slice needs a clear owner.

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

`daita.capabilities.CapabilityRegistry` owns declared capability, executor,
and model-facing tool-view identity. It projects tool schemas and validates
arguments and executor output. Tools are a presentation over capabilities, not
an alternate execution path.

`daita.domains.data.controller.DataToolRuntime` is the current tool execution
boundary. Its order is:

1. determine which registered tools apply to the attached sources;
2. reject a call that was not projected;
3. resolve tool view and capability identity;
4. validate arguments against the declared schema;
5. perform capability-specific validation against current catalog facts;
6. resolve the exact registered executor;
7. execute once;
8. validate the output contract; and
9. return one structured success or error result.

Do not let `AgentLoop`, `Agent`, a tool view, or model-authored text call source
clients or executors directly. Do not infer capability behavior from tool-name
strings when stable capability metadata owns it.

All currently projected data capabilities are non-side-effecting reads except
the explicitly enabled PostgreSQL one-row update. That bounded write retains
its resource-scoped readiness, current admission rechecks, preview fingerprint,
exact once-only approval, one-row limit, and immutable receipt path. Arbitrary
SQL mutation and every other external data write are outside the MVP. A new
data write beyond this exact exception requires an explicit design for
validation, authorization, transactionality, idempotency, uncertain outcomes,
and recovery; approval alone is not such a design.

Artifact continuity is model-led through the existing capability/runtime path.
`artifact_list` exposes only bounded safe metadata for the current conversation;
it is not a public Python, CLI, TUI, or browser inventory. `artifact_read`
returns only bounded previews, and `artifact_convert` currently supports only a
verified Daita-generated XLSX `Data` snapshot to CSV. Conversion reads committed
artifact bytes at the artifact-store boundary, inherits runtime-bound
provenance and sensitivity, records its parent artifact ID, and commits through
the normal artifact policy. Do not add prompt-intent classifiers, historical
artifact-reference projections, a current-file pointer, raw model paths/bytes,
or a second execution path to improve file reference continuity.

### Catalog and adapters

`daita.catalog` owns normalized structural truth and catalog search, inspection,
and traversal. Data-domain code consumes catalog contracts rather than building
a second schema graph or querying source clients for planning facts.

`daita.adapters` owns source-specific admission, containment, discovery,
freshness checks, and I/O. SQLite and local-file paths must be absolute,
bounded, and resistant to symlink/path escape. PostgreSQL credentials remain
secret references and integration SDK use stays behind the execution boundary.

SQL validation belongs in `daita.domains.data.sql`; connector guardrails still
apply at execution. Do not duplicate either system in a generic policy layer.

### Persistence

`daita.storage.sqlite.SQLiteStateStore` persists only state used by the MVP:
identity, sources, current catalog snapshots, run transcripts and terminal
results, semantic annotations, learning review state, immutable database-write
receipts, and PostgreSQL write admission. It is the sole owner of the immutable
checksummed `state_migrations` journal, explicit persisted-record codecs, exact
current/historical schemas, and the bounded preledger bridge. Migrations run
atomically under the existing agent-home writer boundary and validate their
source and target schemas. Put each durable change in one owner-local migration
file; never edit an existing ID/checksum or create migration ownership in
hosting, the loop, or a new runtime. The isolated preledger unit is the only
code allowed to read the historical numeric marker, and only until its
documented minimum-release removal gate is met.

PostgreSQL write admission is owned only by
`postgresql_write_admissions`, never by source connection JSON. Public source
registrations expose `write_access` as a computed compatibility projection;
connection reconstruction remains fail-closed, refresh preserves admission,
and detach revokes it atomically.

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
  skill activation state machines, or monitor schedulers;
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
  existing `DataToolRuntime` boundary, not a policy engine;
- approval is once-only, in-process, and bound to exact frozen arguments; it
  does not create pending state or resume APIs; and
- observation is one best-effort callback that cannot direct execution and is
  not a durable event or telemetry subsystem.

Extend `CapabilityRegistry`, `DataToolRuntime`, `DataContextBuilder`,
`EmbeddedAgent`, and `SQLiteStateStore` only for the concerns they already own.
Do not wrap them in a replacement workflow runtime.

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

`openai`, `anthropic`, `google-genai`, `asyncpg`, `sqlglot`, `keyring`,
`prompt-toolkit`, `rich`, and `XlsxWriter` are default production dependencies.
`dev` is the only optional dependency group; do not restore provider, keychain,
database, parser, CLI, recommended, complete, aggregate, or other customer
extras.

Default installation does not authorize eager imports. Import provider SDKs,
`asyncpg`, `sqlglot`, `keyring`, `prompt_toolkit`, and Rich only inside the
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
6. Return bounded, schema-validated `ToolOutput`; convert expected failures to
   structured model-visible results at `DataToolRuntime`.
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
| `src/daita/domains/data/context.py` | current model request construction |
| `src/daita/domains/data/controller.py` | tool projection and execution boundary |
| `src/daita/domains/data/sql.py` | catalog-scoped SQL validation |
| `src/daita/catalog/service.py` | normalized catalog lifecycle |
| `src/daita/storage/sqlite.py` | sole durable state operation/admission boundary |
| `src/daita/storage/sqlite_schema.py` | exact physical schemas and validators |
| `src/daita/storage/sqlite_codecs/` | explicit durable record-family codecs |
| `src/daita/storage/sqlite_migrations/` | checksummed journal, baseline, and bounded bridge |
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
