# Repository guide

This file explains how Daita works and how to change it. Read it before working
in this repository.

It separates two kinds of guidance:

- **Invariants** are hard rules that protect users' data, authority and money. Do
  not break them. If a task appears to require it, stop and ask.
- **Everything else** describes the current design. It is the default, not a
  law. When a task needs a different design, follow *Proposing architecture
  changes* instead of working around the current one.

Detailed subsystem notes live in [`docs/architecture/`](docs/architecture/README.md).

## Repository scope

`src/daita/` is the sole production package and Python import namespace. Put
tests in `tests/`, user-facing documentation in `docs/`, contributor
architecture notes in `docs/architecture/`, and runnable examples in
`examples/`. Run commands from the repository root so Python uses the
configured `src` layout. Do not create another root-level `daita/` package or a
parallel replacement source tree.

Use this order of authority when repository material disagrees:

1. the current task requirements;
2. production code under `src/daita/` and executable tests under `tests/`;
3. this file, `docs/architecture/`, `README.md` and the user-facing documents
   under `docs/`.

When a document disagrees with the code, trust the code and fix the document
in the same change. Preserve unrelated working-tree changes. Historical code and
documents can explain intent, but they do not define current behavior.

## Extensibility boundary

This repository is a broadly reusable open-source framework. Consumers should be
able to supply their own infrastructure through narrow, documented interfaces.
Keep agent semantics, validation, durable record formats, shared transformations,
and the default local implementation here. Deployment-specific storage adapters,
tenant mapping, provisioning, credential policy, distributed lease services and
operational rollout belong to the consuming application, not this repository.
Never include private deployment identifiers or internal infrastructure docs.

Prefer improving injection in the existing owners over adding a parallel
runtime or copying feature logic. `SQLStateStore`, its
`SQLDatabase`/`SQLConnection` protocols and `ArtifactByteStorage` are supported
component extension points. `Agent.create/open` compose local homes;
`Agent.from_storage` composes the same runtime over caller-admitted state,
artifact bytes and `AdvisoryStorage` documents. Configuration persistence,
provisioning, upgrades and execution leases stay with the caller. Physical
document persistence must preserve the existing memory/skill owners'
validation, bounds and canonical bytes.

## What Daita is

Daita is a persistent, read-first data agent. Each agent has an **agent home**
(`~/.daita/agents/<name>/`) holding one SQLite database plus memory, skills,
artifacts and model configuration. The agent connects to data sources through a
catalog, answers questions through a single model loop, and can run scheduled
**routines** and durable **jobs** in the background. Writes to external systems
(PostgreSQL update/upsert, admitted MCP actions) are narrowly scoped, explicitly
enabled, approved and receipted.

One run looks like this:

```text
user message -> model -> zero or more tool calls -> ordered tool results
             -> model -> answer
```

The run's transcript is its state. Tool failures are ordinary results the model
can correct on its next step. Normal model text completes the run. Step,
wall-time, token and estimated-cost limits bound every run.

Runs come from two places:

- **Foreground runs**: a person asks through the TUI, CLI or Python API.
  Effects need interactive approval.
- **Machine runs**: routine occurrences, job events and graph tasks. Each builds
  its `ExecutionScope` when the run starts and asks a person, through the host,
  before every effect.

## System map

| Component | Responsibility | Where |
| --- | --- | --- |
| `Agent` | Thin public facade; validates inputs and delegates | `agent.py` |
| `EmbeddedAgent` | Composition root: admits the home, holds the writer lock, composes and closes everything | `hosting/embedded.py` |
| Agent home | Directory admission, manifest, writer lock | `hosting/agent_home.py` |
| `RunAdmissionCoordinator` | Execution admission, keyed resource permits, shutdown drain | `hosting/execution_governor.py` |
| `AgentLoop` | One run: call model, append messages, invoke tools, enforce budgets | `loop/driver.py` |
| `AgentContextBuilder` | Builds each model request; labels untrusted content | `context.py` |
| `CapabilityRegistry` | Capability, tool-view and executor declarations; schema projection and validation | `capabilities.py` |
| `CapabilityRuntime` | The one boundary from a tool call to execution | `capability_runtime.py` |
| Capability domains | Applicability, binding and semantics per area (data, artifacts, analysis, memory, skills, semantics, jobs, routines, distribution, MCP) | `domains/`, and each area's `capabilities.py` |
| Catalog | Source and resource truth: schemas, relationships, declared links, freshness | `catalog/` |
| Adapters / source types | Source admission, discovery and bounded I/O | `adapters/`, `adapters/registry.py` |
| Artifacts | Committed artifact bytes, manifests, renderers, delivery | `artifacts/` |
| Jobs | Durable graph jobs: `JobOwner`, `JobSupervisor` | `jobs/` |
| Routines | Scheduled routines: `RoutineOwner`, `RoutineSupervisor` | `routines/` |
| Distribution | Outcomes, destinations, deliveries | `distribution/`, `work_board.py` |
| Models | Canonical records, routing and retries, provider adapters | `llm/` |
| State | Shared SQL operations, schema, strict codecs, home migrations | `storage/` |
| Secrets | Secret references and lazy resolution | `security/` |
| Product surfaces | CLI and Textual TUI, clients of the host | `cli.py`, `tui/` |
| Host | One background process per user that opens every agent and serves clients over an owner-only Unix socket | `host/` |
| Resident host (deprecated) | `daita host --agent <name>`: now a deprecation notice plus `daita host serve` | `hosting/resident.py` |

## Invariants

These protect users. Each comes with the reason it exists.

1. **One writer per agent home.** Whatever process composes an agent holds the
   home's writer lock, and there is never a second writer around its SQLite
   state. *Why: concurrent writers corrupt state and break crash recovery.*
2. **Every tool call executes through `CapabilityRuntime`.** A tool is a view of
   a capability, not another execution path. `AgentLoop`, `Agent`, tool views and
   model text never call source clients or executors directly. *Why: projection,
   validation, admission, approval, receipts and output checks all live there.*
3. **Untrusted content never grants authority.** Catalog metadata, file contents,
   tool and MCP results, memory, skills, semantic notes and model text are
   labeled untrusted. Authority comes only from code-owned capability metadata,
   current permissions, approvals and grants. Never infer access, effects or
   automation eligibility from a tool name or description. *Why: prompt
   injection.*
4. **The catalog is structural truth, and SQL is validated against it before any
   source I/O.** Connector guardrails still apply during execution. *Why: model
   text cannot be trusted to name real resources or safe queries.*
5. **External effects are approved at the time, receipted and never replayed.**
   Every external effect is approved by a person at the time, bound to its exact
   arguments, receipted before dispatch, and never retried or replayed. An
   uncertain outcome stays uncertain until a person resolves it. A run that
   cannot ask a person cannot perform effects. *Why: duplicate or unexplained
   writes to a user's systems are the worst failure Daita can cause.*
6. **A run's scope is fixed when the run starts.** Model text and tool results
   cannot widen it during the run. *Why: unattended work must stay within what
   the person set up, and a prompt injection must not reach new sources.*
7. **Accounting is honest.** Every requested tool call gets exactly one result,
   in call order. Budget exhaustion is terminal. Actual usage is never clipped to
   a reservation, and unknown usage is never recorded as zero. *Why: users pay
   for tokens.*
8. **Secrets stay references.** Credentials are `SecretReference`s resolved only
   at the integration boundary. They never appear in connection JSON,
   transcripts, logs or error text. *Why: credential leakage.*
9. **Durable format changes are versioned, and released ones are immutable.**
   Any change to stored records or home files needs a home migration, a golden
   fixture and a refreshed release contract. Released migrations, checksums,
   decoders and fixtures never change. Newer, damaged or unknown homes fail
   closed without rewriting state. *Why: upgrades must never lose user data.*
10. **State mutation is atomic and cancellation-safe.** *Why: interrupted
    processes must leave a consistent home.*
11. **Optional and heavy dependencies are imported lazily.** Importing `daita` or
    `daita.cli`, or running headless commands, must not load provider SDKs,
    database drivers, the TUI or DuckDB. *Why: fast startup and a working CLI
    when an integration is broken.*
12. **Untrusted text is sanitized before it reaches a terminal.** Use
    `sanitize_terminal_text`/`safe_display` from `tui/sanitization.py`; it does not
    import Textual, so the CLI uses it too. *Why:
    terminal escape-sequence injection.*
13. **The framework stays deployment-agnostic** (see *Extensibility boundary*).
    *Why: this is an open-source framework with many consumers.*

## Current design choices

These describe how Daita is built today. They are deliberate, but they are not
invariants. If a task needs to change one, follow *Proposing architecture
changes*.

- **One of each core owner:** one `AgentLoop` implementation, one
  `CapabilityRuntime`, one capability registry, one catalog, one artifact store,
  one graph-aware jobs supervisor, one routines supervisor and one SQLite state
  boundary. Features extend these owners rather than adding parallel ones.
- **Statically composed capability domains** in `EmbeddedAgent`; there is no
  dynamic plugin or executor registry.
- **A direct loop.** There is no verifier, repair or synthesis pass, and no
  resumable session. Clarification is ordinary model behavior, not a run state.
- **Approval and observation through the host.** Inside a process, approval is
  an `ApprovalHandler` callback the loop awaits and observation a best-effort
  callback. The host turns both into *interactions* and an event stream: an
  approval or question is a pending interaction any connected client can answer
  (the first answer wins; expiry interrupts and never denies), and a desktop
  notification goes out when no client watches that agent. Scheduled and job
  runs ask through the same interactions; the card names the routine or job
  that asked. Interactions live in host memory, not in the agent home.
- **One host per user owns every agent.** `daita host serve` opens every agent
  with background work, holds each writer lock (the single writer is the host),
  runs schedulers and runs, and keeps going when every window is closed. The
  TUI, the CLI and `daita.connect()` are clients that reach it over an
  owner-only Unix socket under `~/.daita/run/`; there is no TCP listener. Clients
  auto-spawn the host when it is not running, and `daita host install` registers
  it as a login service. Library mode (`Agent.open()`, `daita --no-host`) still
  composes the agent in-process, and is refused with `HostActiveError` while
  the host holds that agent.
- **Routines and graph jobs are separate systems**, with no routine-to-graph
  bridge.
- **A routine is a title, a prompt and a schedule**, optionally limited to named
  sources, with one `RoutineIntent` shape for the tool, CLI, Python API and
  host. Creating or changing one shows one confirm card. Everything else is a
  fixed code default: delivery to the conversation's Inbox, missed slots
  skipped, the agent's current model and normal run limits, Needs you after 3
  consecutive failures, a one-time routine ending after it runs. A routine
  freezes no access, grants, budgets or expiry; each run asks before writes.
- **The only distribution destination** is the originating conversation's inbox.
- **SQLite behind `SQLStateStore`**, with no event sourcing or replay
  projections.
- **Advisory learning.** Memory, user profile, skills and semantic notes are
  bounded advisory text, created through the ordinary foreground loop. There is
  no background learning agent.
- **No general-purpose frameworks** for workflows, policy, middleware, lifecycle
  hooks, vectors or telemetry. Add a concrete mechanism to the owner that needs
  it first.

## Subsystem guide

Each area has detailed notes in `docs/architecture/`. Read the relevant file
before changing that area.

| Area | Summary | Notes |
| --- | --- | --- |
| Runtime | Composition, the loop, context building, budgets, source scope, capability execution, approval and observation | [`runtime.md`](docs/architecture/runtime.md) |
| Data and files | Source types and operations, catalog and declared links, SQL validation, combined queries, relational writes, folders, artifacts | [`data-and-files.md`](docs/architecture/data-and-files.md) |
| MCP | Admitted bindings, call-time rechecks, actions and grants | [`mcp.md`](docs/architecture/mcp.md) |
| Jobs and routines | Graph job boundaries, supervision, routine schedules, scheduled-run limits, outcomes and deliveries | [`jobs-and-routines.md`](docs/architecture/jobs-and-routines.md) |
| Persistence | State boundary, home revisions, migrations, upgrades, release contract | [`persistence.md`](docs/architecture/persistence.md) |
| Models | Canonical records, routing and retries, call policy, provider lifecycle | [`models.md`](docs/architecture/models.md) |
| Product surfaces | CLI and TUI rules over the public API | [`product-surfaces.md`](docs/architecture/product-surfaces.md) |

User-facing guides in `docs/` (for example `CONTEXT_AND_SCOPE.md`,
`DURABLE_JOBS.md`, `EFFECT_RECEIPTS.md`, `SCHEDULED_ROUTINES.md`,
`LOCAL_STATE_UPGRADES.md`, `MODEL_PROVIDERS.md`) describe the same areas from the
user's side.

## Recipes

### Add a capability or tool
1. Pick the owning domain (see *System map*); add the capability there rather
   than creating a new domain.
2. Declare a `Capability` in that domain's declarations with a stable `id`,
   input and output schemas, `access_mode`, `operational_effect` and
   `automation_eligibility`. Add an `EffectReceiptPolicy` for external effects
   and an `AutomationGrantPolicy` if machine runs may perform the effect.
3. Implement its `Executor`, plus an optional `ToolView` for the model-facing
   shape.
4. In the domain, project when the tool applies, bind arguments and revalidate
   current facts before I/O. Leave approval and receipts to `CapabilityRuntime`.
5. Make sure it is composed through `CapabilityRegistry` in
   `hosting/embedded.py`.
6. Return bounded, schema-validated output.
7. Test it: a focused contract test in the owner's suite, the
   `tests/architecture` boundaries, and one public end-to-end test.

### Make a capability usable by routines
Nothing to do: every capability is usable by scheduled and job runs unless it is
on the exclusion list in `hosting/embedded.py` (local computer files, and
creating, changing or running routines and jobs). An effectful capability still
needs an `EffectReceiptPolicy`; the run asks a person before performing it.
Check the scheduled-run limits in `docs/architecture/jobs-and-routines.md`.

### Add a source type
1. Register a `SourceType`: built-in in `adapters/registry.py`, or passed at
   composition through `AgentConfig.source_types`. Never add adapter-id dispatch
   outside the type's own modules.
2. Run the shared contract checks in `tests/support/source_contract.py`:
   `assert_source_type_contract`, plus `assert_tabular_read_contract` when it
   declares `TABULAR_READ`, and `assert_live_source_contract` for a live,
   uncataloged source. A source without SQL needs only discovery and a
   `TABULAR_READ` backend to join combined queries.
3. Keep discovery in the catalog and source I/O in the adapter. Extend the
   existing adapter, catalog and concrete domain.
4. Declare stable `Capability`, `Executor` and optional `ToolView` identities and
   compose them through `CapabilityRegistry` in `EmbeddedAgent`.
5. Validate current facts before I/O, keep connector guardrails, and return
   bounded schema-validated `ToolOutput`.
6. Add focused contract tests and one public end-to-end test.

### Add a model provider
1. Implement `ManagedModelProvider` under `src/daita/llm/providers/`, delegating
   common attempt and close behavior to `llm._lifecycle`. Use a provider package
   (`adapter.py`, `messages.py`, `stream.py`) when translation or streaming is
   substantial; a small specialization stays one module.
2. Reuse `providers._fields` for native-field validation, and import a new SDK
   only at client construction, with normalized pipx repair guidance.
3. Add one `ProviderDefinition`, including its lazy construction callable, in
   `llm/provider_definitions.py`. The factory, host and TUI need no
   provider-specific edits.
4. Add a bounded SDK dependency only when the protocol needs one.
5. Add definition, lazy-import, translation, lifecycle, accounting, routing and
   architecture tests.

Never add provider branches to `AgentLoop`. Details are in
`docs/architecture/models.md`.

### Change a stored record or home file
1. Change the schema in `storage/sqlite_schema.py` and the matching strict codec
   in `storage/sqlite_codecs/`.
2. If the next home revision is still unreleased, amend its migration in
   `storage/home_migrations/`. Otherwise append a new revision and register it in
   `registry.py`.
3. Regenerate the golden fixture with the matching
   `scripts/create_revision_<n>_golden_fixture.py`.
4. Refresh and check the release contract:
   ```bash
   python scripts/check_home_release_contract.py write
   python scripts/check_home_release_contract.py check
   ```
5. Test the upgrade from every supported revision, data preservation, and the
   failure, recovery and rollback paths.

The full release policy is in `docs/architecture/persistence.md`.

### Add a CLI command
Add the parser in `build_parser()` and its branch in the command dispatch in
`cli.py`. Use only the public `Agent` API, and format output through
`cli_text.py`. Add tests in `tests/cli/`.

### Add a host route
1. Add the member to the `Agent` facade first; the host adds no behaviour of its
   own.
2. Declare one `Route` in the area's module under `host/routes/` (one module per
   area: `sources.py`, `jobs.py`, `routines.py`, …). Handlers are the shared
   decode-call-encode endpoint; set `work=True` when the result can change the
   work board and `reopen=True` when the agent must reopen idle afterwards.
3. Add the member to `REMOTE_AGENT_SURFACE` in `host/protocol.py` and mirror it
   on `RemoteAgent` in `host/client.py`. Every parameter and return type must
   round-trip through `host/codec.py`; add a codec case for a new record type.
4. Run `tests/host`: the surface guard, the codec round-trip and the route
   reachability tests fail until all three agree.

### Add a TUI command or screen
1. Add the slash command to the right group in `COMMAND_GROUPS` in
   `tui/commands.py`.
2. Handle it in `PresentationController.dispatch_command` (`tui/controller.py`),
   returning a `CommandOutcome`.
3. Open screens from `DaitaApp._open_command_screen` (`tui/app.py`).
4. Use the public `Agent` API through the controller, and sanitize every
   displayed value (invariant 12).
5. Add tests in `tests/tui/` with Textual's `App.run_test`.

## Development setup

Use a dedicated environment:

```bash
cd /path/to/daita-agents
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Python 3.11 and 3.12 are supported.

`pyproject.toml` `[project].version` is the sole authored Daita release
identity. Runtime version displays read installed `daita-agents` distribution
metadata. Because editable metadata is an installation snapshot, rerun the
editable install command after changing that value or checking out a commit
with another value, before importing Daita or running tests. The agent-home
revision is independent of the package version.

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

## Dependencies

The complete customer installation is:

```text
pipx install daita-agents
daita
```

Default production dependencies are `openai`, `anthropic`, `google-genai`,
`asyncpg`, `sqlglot`, `keyring`, `rich`, `textual`, `XlsxWriter`, `httpx`,
`httpx2`, exact `mcp==2.2.0` and exact `duckdb==1.5.5`; `pyproject.toml` is the
authoritative list. `dev` supplies test and quality tools; `analysis` supplies
the pinned scientific runtime for native macOS ARM64 computation. Managed
installation includes `analysis`; a source checkout uses
`pip install -e '.[dev,analysis]'` to qualify that runtime.

All of these are imported lazily at the boundary that first needs them
(invariant 11): provider SDKs at client construction, the MCP SDK and `httpx2`
when an MCP client is created, XlsxWriter only in the XLSX renderer. Use
`if TYPE_CHECKING:` for type-only imports.

A missing or damaged production dependency raises a normalized `ImportError`
that directs the user to:

```text
pipx reinstall daita-agents
```

Do not advertise an extras-based repair. Add a dependency only when the
standard library and existing dependencies cannot do the job; keep it bounded
and lazily imported.

## Change discipline

Before adding a helper, abstraction, module, base class, builder, registry, or
shared utility:

1. identify the existing component responsible for the behavior;
2. state the broken or painful contract;
3. choose the smallest complete change;
4. confirm that it does not introduce a parallel responsibility; and
5. name the focused tests that detect drift.

Prefer a complete current behavior over placeholders for hypothetical features.
As a rule of thumb, add a shared abstraction when at least three current call
sites need it and it removes more complexity than it adds. Avoid churn-only
renames and broad consistency edits.

For reliability failures, trace the problem to the responsible contract,
lifecycle, state, or trust boundary. Replace the incorrect mechanism there and
remove the obsolete path. Do not hide a broken design behind retries or
special cases.

When you change behavior that a document in this file or `docs/architecture/`
describes, update that document in the same change.

Do not commit changes unless the task explicitly requests a commit.

## Proposing architecture changes

Some tasks need something the current design does not have: a new kind of
owner, a background service, persisted approvals, a new storage mechanism, a
general mechanism replacing several specific ones. That is allowed. Do it in the
open rather than bending the existing design to fit:

1. Write a short note in `docs/adr/NNNN-short-title.md` covering: the problem,
   the options considered, the choice, which invariants it touches and how they
   stay intact, the persistence and migration impact, and how it will be tested.
2. Ask for review before implementing.
3. Once accepted, implement it and update this file and `docs/architecture/` in
   the same change, so the next contributor sees the new design as current.

Changing an invariant follows the same path, and the note must say what now
protects the user in its place.

## Glossary

- **Agent home**: one agent's directory under the state root (`~/.daita` by
  default), holding `state.db`, memory, skills, artifacts and model
  configuration. Its format is versioned by the *home revision*.
- **Admitted / admission**: passed the code-owned checks that allow something to
  be used now: a source registration, a model profile, an MCP binding, a run
  under capacity. Admission is checked again at use.
- **Authority**: permission to read or change something. It comes only from
  code-owned capability metadata, current permissions, approvals and grants.
- **Bounded**: has an explicit size, count or time limit enforced in code.
- **Code-owned**: decided by Daita's code, never by model output or external
  content.
- **Effect (operational effect)**: a change outside the run's own records, such as a
  database write or an MCP action, declared through `OperationalEffect`.
- **Exact**: bound to specific identities, revisions or digests and rechecked
  before use. If something drifted, the operation fails rather than adapting.
- **Foreground run / machine run**: a run a person started, versus a machine
  run: a routine occurrence, job run or sub-run, which asks a person before
  effects.
- **Frozen**: fixed when a run starts, or when a person confirmed a routine, and
  unchanged afterwards.
- **Grant**: a standing, exact permission that let machine runs perform one
  specific effect without interactive approval. Being replaced by approval at
  the time; machine runs no longer use grants, and the machinery is removed in
  a later phase.
- **Receipt**: the durable record reserved before an external effect and
  finalized with its evidence; the only truth about whether the effect happened.
- **Sensitivity**: an information-handling classification. It never confers
  execution authority.
- **Source scope**: the sources and resources a run may read, frozen when the run
  is prepared.
- **Untrusted content**: anything not authored by Daita's code: model text,
  catalog metadata, file contents, tool and MCP results, memory, skills.

## Key files

| File | Responsibility |
| --- | --- |
| `src/daita/__init__.py` | focused public exports |
| `src/daita/agent.py` | public persistent-agent facade |
| `src/daita/hosting/embedded.py` | composition and agent-home admission |
| `src/daita/hosting/agent_home.py` | agent-home directories and the writer lock |
| `src/daita/hosting/execution_governor.py` | execution admission and resource permits |
| `src/daita/hosting/resident.py` | deprecated one-agent resident process; warns and is removed in a later release |
| `src/daita/host/server.py` | the host process: singleton lock, socket, app, graceful shutdown |
| `src/daita/host/registry.py` | opens and closes agents inside the host, idle policy |
| `src/daita/host/runs.py` | run submission, cancellation and completion |
| `src/daita/host/interactions.py` | approvals and questions a client answers |
| `src/daita/host/events.py` | per-agent event stream with bounded replay |
| `src/daita/host/notify.py` | desktop notifications when no client watches |
| `src/daita/host/service.py` | launchd and systemd login-service files |
| `src/daita/host/settings.py` | `~/.daita/host.toml` user settings |
| `src/daita/host/client.py` | `RemoteAgent` and `daita.connect()` |
| `src/daita/host/supervise.py` | start, stop and status of the host process |
| `src/daita/host/paths.py` | root, socket, lock and log paths |
| `src/daita/host/protocol.py` | protocol version, `REMOTE_AGENT_SURFACE`, error codes |
| `src/daita/host/codec.py` | wire encoding of every surface type |
| `src/daita/host/routes/` | one thin decode-call-encode route module per area |
| `src/daita/loop/driver.py` | direct model/tool progression |
| `src/daita/loop/models.py` | run, transcript, limits, and exit records |
| `src/daita/capabilities.py` | declarations and registry |
| `src/daita/capability_runtime.py` | common execution mechanics |
| `src/daita/context.py` | model-request construction |
| `src/daita/scope.py` | effective source and resource scope |
| `src/daita/adapters/registry.py` | source type registry and built-in source types |
| `src/daita/adapters/folder.py` | folder source type: attached and session folders |
| `src/daita/domains/data/sql/` | catalog-scoped SQL validation |
| `src/daita/domains/analysis/` | local Python analysis |
| `src/daita/domains/mcp.py` | MCP projection and call-time rechecks |
| `src/daita/jobs/` | durable job records and supervision |
| `src/daita/routines/` | scheduled routine records and supervision |
| `src/daita/distribution/` | outcomes, destinations, and deliveries |
| `src/daita/work_board.py` | work board projection over jobs, routines and deliveries |
| `src/daita/artifacts/store.py` | committed artifact storage boundary |
| `src/daita/semantics.py` | semantic annotations |
| `src/daita/memory/`, `src/daita/skills/` | advisory memory, user profile and retained skills |
| `src/daita/config.py` | immutable runtime and model configuration (`AgentConfig`) |
| `src/daita/workspace.py` | local file access intent (`LocalWorkspace`) |
| `src/daita/security/` | secret references and lazy resolution |
| `src/daita/observation.py` | observation events and observer protocol |
| `src/daita/storage/sql.py` | shared state operations |
| `src/daita/storage/sqlite.py` | default local database admission |
| `src/daita/storage/sqlite_schema.py` | current physical schema |
| `src/daita/storage/sqlite_codecs/` | strict current-record serializers |
| `src/daita/storage/home_migrations/` | whole-home revision registry and immutable transitions |
| `src/daita/hosting/home_upgrade.py` | staged upgrade, validation, rollback, and crash recovery |
| `src/daita/llm/routing.py` | normalized provider routing |
| `src/daita/llm/provider_definitions.py` | built-in provider definitions |
| `src/daita/cli.py`, `src/daita/cli_text.py` | CLI parser, dispatch and output |
| `src/daita/tui/` | Textual TUI: app, controller, commands, screens, widgets |
| `tests/architecture/test_boundaries.py` | architecture and public-surface boundaries |
