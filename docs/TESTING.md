# Testing Daita

Run tests from the repository root. The suite is organized by the production
component that owns each contract, not by an old milestone or by a separate
unit/integration directory hierarchy.

## Layout

- `tests/agent`, `loop`, `context`, `capabilities`, and `llm` cover the public
  facade and direct model/tool execution path.
- `tests/catalog`, `data`, `workspace`, and `mcp` cover source truth, bounded
  I/O, writes, files, and remote-tool contracts.
- `tests/artifacts`, `jobs`, `routines`, and `distribution` cover durable work
  and outputs.
- `tests/memory`, `skills`, `semantics`, `learning`, `storage`, `security`, and
  `hosting` cover advisory state, persistence, permissions, and lifecycle.
- `tests/cli` and `tests/tui` cover their own projections and controls.
- `tests/acceptance` contains selected multi-owner public journeys.
- `tests/architecture` contains only structural ownership, import, and public
  boundary contracts.
- `tests/support` contains shared non-test fakes and harnesses. Owner-local
  helpers stay beside their suite. Static service definitions remain in
  `tests/fixtures`.
- `tests/live` contains real model, external database, or remote-service cases.
  `tests/slow` contains extended deterministic local soaks. Both are excluded
  from default discovery.
- `tests/diagnostics` and `tests/packaging` contain explicit module entry points.

## Offline throughput components

The bounded public-composition campaign uses controlled model responses, three
local SQLite fixtures and the real native Python worker when its pinned runtime
is available. It checks values, artifact hashes, conversation order, graph
completion and native cleanup independently of the scripted answer. Scripted
models establish execution behavior; they provide no model-quality evidence.

```bash
DAITA_RUN_THROUGHPUT_COMPONENT=1 \
DAITA_THROUGHPUT_CAMPAIGN=baseline \
DAITA_THROUGHPUT_REPORT_DIR=/private/tmp/daita-throughput-evidence \
  .venv/bin/python -m pytest tests/slow/hosting \
  -o addopts="--tb=short -q --strict-markers" -s
```

Choose `final` for the integrated campaign. `DAITA_THROUGHPUT_SAMPLES` accepts
30–100 repetitions (default 30). Compare the same fixture digest, configuration,
host and workload; reports retain raw latency samples, p50/p95, a fixed-seed
bootstrap interval, task throughput, model requests and process peak RSS. Missing
model token/cost usage remains unknown. Reports must be outside the repository.
Unperformed real-database and model checks remain explicitly labeled and require
their existing live gates. An unavailable native runtime is unperformed evidence.
Use at most one `additional` campaign when uncertainty overlaps; keep conservative
defaults if that sampling remains inconclusive. Keep campaign reports private;
published performance claims should state the tested configuration, correctness,
usage, cleanup and uncertainty.

`Agent.admission_diagnostics()` returns immutable, runtime-only counters aggregated
by permit kind and workload class, plus applied host capacities. It retains no
per-request samples or permit identities. Queue, lease holding and release cleanup
times are separate cumulative measurements; lease holding time includes the work
performed while the lease is held. Set `AgentConfig.admission_observations=True`
to include `admission.updated` in the existing best-effort observer with payload-free
host identifiers. The default observer stream remains unchanged. Observer failure cannot
change admission or execution. Diagnostics survive close in memory and disappear
with the composition; no telemetry or agent-home format change is introduced.

## Local Python live acceptance

`tests/analysis` covers the production native execution slice. The required
opt-in acceptance suite is `tests/live/analysis`: real-model API, CLI and mounted
TUI journeys plus real-worker faults, resource accounting, active revocation,
crash/reopen, data fidelity, output formats and managed installation. Model turns
may be scripted for deterministic faults; native execution and teardown are real.

Run on macOS ARM64 with `.[dev,analysis]` installed. The live model defaults to
reviewed `openai:gpt-5.6-terra`, reads configured repository `.env` credentials
without displaying them, and permits at most $0.50 per paid run and $5.00 aggregate
reserved model cost per invocation. Set `DAITA_ANALYSIS_LIVE_MODEL_ID`,
`DAITA_ANALYSIS_LIVE_MAX_COST_USD`, `DAITA_ANALYSIS_LIVE_SUITE_MAX_COST_USD` or
`DAITA_ANALYSIS_LIVE_LLM_API_KEY` explicitly when needed. All ceilings must be
finite and positive. `DAITA_ANALYSIS_LIVE_DEADLINE_SECONDS` defaults to 900.

Build the exact candidate wheel and supply the reviewed uv archive from
`release/managed-installer.json`; the installer test verifies its checksum:

```bash
.venv/bin/python -m build --wheel --no-isolation --outdir /private/tmp/daita-analysis-wheel
export DAITA_ANALYSIS_LIVE_WHEEL=/private/tmp/daita-analysis-wheel/daita_agents-VERSION-py3-none-any.whl
export DAITA_ANALYSIS_LIVE_UV_ARCHIVE=/private/tmp/daita-analysis-uv-aarch64.tar.gz
DAITA_RUN_LIVE_ANALYSIS=1 .venv/bin/python -m pytest tests/live/analysis \
  -o addopts="--tb=short -q --strict-markers" \
  --junitxml=/private/tmp/daita-analysis-live.xml
```

Replace `VERSION` with the actual wheel version authored in `pyproject.toml`. No candidate is
published. The real installer downloads the pinned CPython and scientific
dependencies into disposable binary generations; no cell installs packages.
Missing configuration fails dedicated acceptance rather than skipping it.
Ordinary offline CI does not opt into these live tests.

Every call outcome writes JSON to `DAITA_ANALYSIS_LIVE_REPORT_DIR` (default
`/private/tmp/daita-analysis-live-evidence`), including failures, usage, captured
child evidence and actual cleanup. Prior results are archived in `history/`;
JUnit is retained independently. Inspect both run status and measurement
completeness. A mocked cleanup, absent worker measurement or model-written
analytical claim is not successful live evidence.

`tests/storage/contracts` is the portable durable-backend suite. The framework
runs it against SQLite and a disposable in-memory artifact-byte dependency:

```bash
.venv/bin/python -m pytest tests/storage/contracts
.venv/bin/python scripts/check_home_release_contract.py check
```

Downstream qualification can reuse the same cases with an external pytest plugin.
Indirectly parameterize `state_store_factory` with the name of a fixture returning
an async context-manager factory; each call must open a new handle to the same
disposable state. Indirectly parameterize `artifact_byte_storage` with the name of
a byte-storage fixture exposing the test dependency's `objects`, `publications`,
`reads` and `deleted` observations. `advisory_storage` accepts the name of a fixture
implementing the document transaction contract. Its cases cover memory/skill
semantics and public agent admission, tool execution, cancellation and reopening.
Defaults use SQLite and in-memory byte/document dependencies.
Tests never select an infrastructure provider from ambient credentials.

Keep concrete driver, transport, isolation, provisioning and failure tests with
the adapter. Missing required resources must fail that adapter's qualification,
not silently skip it. The framework's in-memory byte cases establish lifecycle
behavior, not external transport readiness. See [storage contract](STORAGE_CONTRACT.md)
and [artifact storage](ARTIFACT_STORAGE.md).

Use explicit imports such as `from tests.support.paths import REPO_ROOT`. Test
and support modules must not import a `test_*.py` module or `conftest.py`.
Fixtures belong in the narrowest applicable `conftest.py`; ordinary reusable
constructors and models belong in support modules.

## Default and focused checks

```bash
# Default deterministic suite. Configuration excludes tests/live and tests/slow.
.venv/bin/python -m pytest

# Focused owner suite.
.venv/bin/python -m pytest tests/llm -v

# Explicit deterministic CI selection.
.venv/bin/python -m pytest tests/ \
  -m "not requires_llm and not requires_db and not requires_network and not slow"

# Architecture boundaries.
.venv/bin/python -m pytest tests/architecture

# Static checks.
.venv/bin/python -m black --check src tests
.venv/bin/python -m ruff check --select I .
.venv/bin/python -m mypy src/daita tests
```

The default suite deliberately includes offline SDK transport tests, external
executor conformance, benchmark-support qualification, and live-harness
qualification. Those tests use injected local boundaries and do not establish
that a real external integration passed.

## Markers and external selection

### Count, recovery, progress and diagnostic qualification

The count/recovery implementation has six deterministic owner contracts: execution
limits and current local applicability share the registered contract; rejected
calls carry bounded recovery/disposition evidence; catalog identity discovery
preserves authorized selection and revisions; `data_count_rows` uses ordinary
SQL validation and source adapters; repeated permanent failures produce bounded
run-local guidance; opt-in observation retains per-call and provider-attempt
diagnostics. The agent-home format remains revision 4. None of these checks alone
qualifies real-model behavior or an external database.

Run the focused deterministic cases and the ordinary release checks:

```bash
.venv/bin/python -m pytest tests/capabilities/test_execution_recovery.py \
  tests/catalog/test_schema_slices.py tests/data/test_row_counts.py \
  tests/loop/test_sessions.py tests/agent/test_observation.py \
  tests/llm/test_routing.py tests/acceptance/test_count_recovery.py
.venv/bin/python -m pytest
.venv/bin/python -m black --check src tests
.venv/bin/python -m ruff check --select I .
.venv/bin/python -m mypy src/daita tests
.venv/bin/python scripts/check_home_release_contract.py check
```

`tests/support/count_recovery.py` independently seeds two disposable SQLite
databases, Juniper and Meridian, with 8 and 38 base tables and 257 total columns.
The manifest defines counts, totals and winner without disclosing them in model
prompts. Fixtures include empty tables, shared table names, quoted identifiers,
field-dependent control values and distractor views. Real SQLite owner cases
check exact per-resource counts, deduplicated totals, revision bindings, rejected
batches, and at least 60% canonical-byte reduction for identity discovery versus
the same full catalog selection. Full-schema discovery remains the default.
Each count call admits at most four resources, derived from capability policy;
views require explicit selection. Each batch has one source/statement snapshot.
Separate batches and sources do not establish a common snapshot or complete
physical-database coverage. `observed_at` marks execution start and `duration_ms`
covers execution and validation.

Identity tool results for one explicit source also supply
`selection.count_selection_ref`. Start a framework-managed count with
`data_count_rows({"selection_ref": "<returned reference>"})`; continue with
`data_count_rows({"cursor": "<returned next_cursor>"})` until `next_cursor` is
null. The original `source_id`/`resource_ids` form remains available. Do not mix
these forms. `include_views` can be selected on the first call only and stays
fixed throughout the continuation.

The data domain resolves references from authenticated, earlier successful
catalog/count results in the exact persisted current-run transcript. Each call
selects at most the policy's four resources, uses ordinary SQL validation and
adapter admission, and performs one read statement. It rechecks the whole fixed
selection against current permissions, frozen run scope and structural/source
revisions, including resources counted in earlier pages. Unknown, forged,
cross-run or stale references fail before source dispatch. Native structural
drift still fails at the adapter boundary. A failed page issues no cursor and
advances no progress; the framework performs no automatic replay.

`total_rows` remains the current batch total. `progress.total_rows` is the
code-computed cumulative total for that cursor chain, with exact selected,
counted and remaining resource counts and authenticated evidence call IDs.
Repeated reads of a cursor produce a new observation of the same next batch
without double-counting earlier pages. Branches are independent and their
totals must not be added together. Progress preserves the sensitivity floor of
its catalog and count lineage. Count-page provenance lists the exact batch;
prior evidence remains linked and unchanged in the transcript.

`complete_admitted_base_tables` requires both complete cursor coverage and a
complete source identity enumeration. Completing a truncated, queried or
explicitly selected catalog slice reports `complete_catalog_selection` with
`catalog_enumeration_complete=false`; it cannot establish a whole-source total.
Selections retain the existing maximum of 50 catalog resources and default
view exclusion. Separate batch snapshots remain explicit. Cursors are neither
durable jobs nor resumable runs and cannot extend token, step, time or tool-call
budgets. With sequential cursors alone, the 38-table source requires ten count
turns; discovery and the final answer normally bring the journey to twelve
model requests. That exceeds the existing eight-request efficiency target.
The target has not been relaxed, and prior live results do not establish this
interface's token consumption, latency or model adoption.

The row-count owner tests exercise the full 46-table reference/cursor journey,
scorer-independent totals and coverage, repeat/changed reads, failed-page
correction, tampered and unknown references, cross-run rejection, catalog and
native drift, revocation, partial discovery, and terminal step/token limits.
These scripted provider cases verify framework behavior, not live-model
reasoning or efficiency. Existing paid campaign evidence predates the cursor
interface and does not qualify it.

For targeted cursor adoption checks, set
`DAITA_COUNT_RECOVERY_REQUIRE_CURSORS=1` alongside the ordinary authorization and
budget gates. Reports include `cursor_usage`: requested and successful selection
starts/continuations, completed selections, legacy count calls and normalized
errors. A legacy-only answer cannot pass this adoption gate. L2 field controls
and L6 deliberately stale native reads are exempt, and their results do not
establish cursor adoption. A targeted subset or a single repetition remains
exploratory evidence, not the full qualification campaign.

Budget-pressure answers are scored against the independently measured table
subtotal and coverage, including ordinary lower-bound wording and Markdown
tables. A claimed winner requires sufficient measured evidence: its lower bound
must exceed every other source's complete total. Scoring must not require a
particular word such as "remaining" when "30 remain" or "counted so far" says
the same thing. Original failed reports are retained and offline scorer reviews
are labeled separately; corrected scoring does not change recorded usage.

API and Codex campaigns share the read-evaluation owner's provider-failure
classification. Token/cost budget termination is distinct from provider failure
and cannot trigger a two-provider-failure stop. Actual usage is still charged
without clipping; unknown subscription usage retains its reservation. A case
blocked before provider creation is unperformed evidence, never a successful
live run.

The dedicated live module uses the public agent, production runtime and reviewed
provider factory. Collection performs no credential resolution or service I/O:

```bash
.venv/bin/python -m pytest tests/live/acceptance/test_count_recovery.py \
  --collect-only -o addopts="--tb=short -q --strict-markers"
```

With the default three repetitions and one API profile, collection contains 81
cases: 27 API/SQLite, 27 API/PostgreSQL and 27 actual Codex/SQLite cases. Their
authorization gates are independent. A primary campaign has nine natural L1
comparisons (three paraphrases), then three each of L2 filtered field controls,
L3 unavailable durable recovery, L4 repeated-failure correction, L5 restricted
scope, L6 real structural drift and L7 four-step budget pressure. L1/L2/L5/L7
are unscripted real-model journeys. L3/L4/L6 record a synthetic model prefix that
invokes the real runtime and then yields to a real model in the same transcript;
those turns are excluded from unforced efficiency evidence. The ordinary
SQLite adapter suite separately verifies controlled timeout and cleanup.

Only after explicit authorization, supply private credentials to the process
and a fresh absolute report directory outside the repository. Do not load `.env`
or enable gates during collection. API credentials use the existing
`DAITA_STAGE_B_BENCHMARK_<PROVIDER>_API_KEY` or
`DAITA_STAGE_B_BENCHMARK_LLM_API_KEY` helper. Reviewed tool-capable API profiles
are selected by `DAITA_STAGE_B_BENCHMARK_MODEL_IDS` (default
`openai:gpt-5.6-terra`). For an authorized primary API/SQLite campaign:

```bash
DAITA_RUN_LIVE_COUNT_RECOVERY=1 \
DAITA_COUNT_RECOVERY_REPEATS=3 \
DAITA_STAGE_B_BENCHMARK_MAX_COST_USD=0.50 \
DAITA_COUNT_RECOVERY_TOTAL_COST_USD=15.00 \
DAITA_COUNT_RECOVERY_REPORT_DIR=/private/tmp/daita-count-recovery-candidate \
.venv/bin/python -m pytest tests/live/acceptance/test_count_recovery.py \
  -k api_sqlite -o addopts="--tb=short -q --strict-markers" \
  --junitxml=/private/tmp/daita-count-recovery-candidate.xml -o junit_family=xunit1
```

Both API dollar budgets must be explicitly supplied and positive. Runs reserve
the full per-run ceiling before dispatch; actual reported consumption is charged,
and missing cost retains the reservation. Serial execution shares aggregate
reservations and stops after two consecutive provider failures. `xdist` is
rejected. Normal runs allow 24 steps, 100,000 cumulative tokens and 300 seconds,
with a 1,536-token output allowance; L7 allows four steps. L3/L4 disable graph
registration through existing host configuration while a test-only decorator
narrows each canonical paid request to its remaining allowance. It cannot enlarge
authorization. Repetitions default to three and accept 1–5; a smaller exploratory
campaign does not establish the documented 27-run adoption campaign. Add another
model family only under a separately authorized budget after the primary is stable.

`DAITA_COUNT_RECOVERY_OUTPUT_TOKENS` optionally selects a 256–8,192 token
response allowance (default 1,536), still bounded by the reviewed profile and
remaining paid/run admission. The 46-table journey can require more than 1,536
output tokens to emit parallel calls containing full resource IDs. A 4,096-token
campaign is a distinct recorded setting; retain output-limited failures and do
not compare its improvement ratio to a baseline with another allowance. This
setting does not change the 50,000/100,000 token efficiency gates or per-run
dollar ceiling. Count results retain exact source IDs and catalog source labels
to support correct association of parallel batches.

PostgreSQL additionally requires `DAITA_RUN_COUNT_RECOVERY_POSTGRES=1`, the API
gate/budgets, `DAITA_COUNT_RECOVERY_POSTGRES_PORT`, and explicit
`DAITA_COUNT_RECOVERY_POSTGRES_ADMIN_PASSWORD` and
`DAITA_COUNT_RECOVERY_POSTGRES_READ_PASSWORD`. Select `-k api_postgresql` and a
fresh report directory. The fixture only targets `127.0.0.1` and the precreated
disposable database `daita_count_recovery_fixture`, with administrator `postgres`
and a precreated nonsuperuser `daita_count_reader`. Setup seeds two isolated
schemas, grants read-only access, verifies seed counts and removes those schemas
on completion or setup failure. This cohort proves the real PostgreSQL adapter
over two logical sources in one disposable database; it does not claim a common
snapshot across separate physical databases. Never point it at customer data.

The same explicit PostgreSQL fixture gate/credentials can separately run
native execution without a model call:

```bash
.venv/bin/python -m pytest tests/live/data/test_row_counts.py \
  -o addopts="--tb=short -q --strict-markers"
```

It checks all 46 independently seeded table counts, exact revisions and totals,
quoted/empty tables, batch bounds, structural drift and cleanup. Its private
report explicitly states `native_postgresql_only` and `llm_qualified=false`.
Passing it proves native adapter behavior, never PostgreSQL/LLM qualification.

Actual subscription transport qualification requires
`DAITA_RUN_COUNT_RECOVERY_CODEX=1`, `DAITA_COUNT_RECOVERY_CODEX_MODEL_ID` with an
explicit reviewed tool-capable `codex:` profile,
`DAITA_COUNT_RECOVERY_CODEX_CREDENTIAL`, and a positive
`DAITA_COUNT_RECOVERY_CODEX_TOTAL_TOKENS`. Select `-k codex_sqlite` and a fresh
report directory. The harness reserves 100,000 tokens per next run, retaining
that allowance for unknown consumption; a 27-run campaign needs up to 2,700,000
tokens. This uses the actual Codex provider, never an API substitute. Dollar
pricing is unavailable; the adapter's native preflight input/output limitations
remain explicit. Outer step/token/time limits still apply.

Private JSON reports retain fixture and candidate commit/diff identities,
model/transport/settings, scope, calls/results, source-dispatch observations,
independent numerical/coverage scores, usage completeness, timing, bytes,
seeded turns, injected faults, terminal reason and cleanup. The JSONL file named
by `diagnostics_path` retains logical model calls, measured provider attempts
where available, reasoning/cache subsets, tool-result bytes and remaining
allowances. Missing attempt/cache/cost measurements are unavailable, never
measured zero. Diagnostics are best effort and contain no prompts, result
payloads, credentials or opaque reasoning. JSON and JSONL files have mode 0600;
the report directory requires 0700. Setup failures retain a report and JUnit
`count_recovery_evidence` references its path. `campaign-summary.json` separates
cohorts and unperformed gates. Private evidence is not a public CI artifact.

Qualification requires every selected case's independent functional oracle and
complete live usage. L1 additionally requires median cumulative tokens at most
50,000, maximum at most 100,000, median logical requests at most eight and at
least 60% catalog-byte reduction. L4 requires correction or a truthful blocker
within two real model responses. Scope, transcript ordering, no-replay and hard
budgets remain mandatory. The conservative answer scorer accepts source-labelled
prose/tables and comma-formatted numbers; ambiguous parsing is unqualified. Its
coverage scorer authenticates count results or simple unfiltered single-resource
`COUNT(*)` queries. Other valid SQL aggregate shapes require scorer extension
before they can qualify, rather than being inferred correct from model text.

Compare baseline/candidate reports only with matching fixture digest, model
profile/output allowance, permissions, transport and run settings, retaining
all raw attempts. Historical incident numbers and API/subscription comparisons
cannot establish an improvement ratio. No gate or target may be relaxed merely
to turn a failure green. Skips, mocks, missing cases/measurements, exhausted
campaign budgets, setup/provider failures and incomplete runs never count as
successful live qualification. The implemented harness and deterministic checks
are separate from live-qualified API, PostgreSQL and Codex evidence.

`unit`, `contract`, `integration`, and `acceptance` describe the test level.
Execution requirements are separate:

- `requires_llm` performs an actual model call.
- `requires_db` contacts an actual external database.
- `requires_network` contacts another real external service; injected or local
  loopback transports do not need this marker.
- `slow` is an extended deterministic stress or soak case.

Collect live cases without authorizing execution:

```bash
.venv/bin/python -m pytest tests/live --collect-only \
  -o addopts="--tb=short -q --strict-markers"
```

Selecting `tests/live` requires the `-o addopts=...` override because the
normal configuration ignores that directory. Collection alone does not grant
credentials or authorize calls. Actual execution also requires the exact
module-specific `DAITA_RUN_*` gate, credentials, and any documented cost or
resource limits. For example, only after explicit authorization:

```bash
.venv/bin/python -m pytest tests/live/data/test_model_write_acceptance.py \
  -o addopts="--tb=short -q --strict-markers" -v
```

Keep generated live evidence private, including JSON reports, JUnit output,
logs, summaries, and disposable agent homes. They can contain prompts,
transcripts, account identifiers, and service results even after credential
redaction. Save them outside the repository, as in the examples below, or under
the ignored `live-evidence/` or `test-results/` directories. Do not commit or
publish these files as public CI artifacts. Test harnesses and synthetic fixtures
remain part of the repository.

Functional live suites use the ordinary outer safety envelope of 24 model
requests, 100,000 total tokens, and 300 seconds, together with each module's
explicit cost cap. These limits prevent runaway execution without turning an
efficiency target into the functional oracle. Record token use as benchmark
evidence. Test exact token admission and exhaustion behavior with deterministic
providers in the owning loop and LLM suites.

The live MCP action and scheduled execution suites default to a `$0.50`
estimated cost cap per agent run in both evaluation profiles. Configure it with
`DAITA_PHASE_F_LIVE_MAX_COST_USD`; it must remain finite and positive. These
suites use real model providers with simulated MCP servers.

### Live analytical and operational acceptance

`tests/live/data_operations` exercises Daita's data and analytics workflows using the
existing SQLite/PostgreSQL adapters, MCP SDK fixture, native macOS Python worker,
scheduled routines and approved semantic annotations. With one model and one
repetition it collects 58 cases:

| Area | Checks and independent evidence |
| --- | --- |
| Analytical quality | 50 fixed cases: paid/refund totals, join multiplication, local-day boundaries and counts across ten datasets. Python fixture facts provide the numerical oracle; current successful SQL evidence is required. Separate cases require account clarification and honest partial-file coverage. |
| Python and automation | Real foreground Python creates an independently scored JSON dataset and PNG chart, with child read evidence, measured resource use, cleanup and retained computation evidence. Public routine proposals execute scoped SQL and Python reports, deliver them to the inbox, and require a PNG artifact for Python. |
| Existing connectivity | SQLite reads, the existing disposable PostgreSQL fixture with a separate raw-row oracle, and two overlapping MCP accounts through the production SDK. Exact account selection and schema drift must prevent dispatch under a changed binding. |
| Retained business meaning | The same definition supplied in the prompt provides a control. One approved definition must survive reopen and produce the same correct result in three fresh conversations, leave an unrelated query correct, and become unusable after schema drift. |

The 50 numerical cases cover five question families with varied fixture amounts;
they are not 50 unrelated business problems. MCP transport and service responses
are fixtures; their cases establish model/tool behavior, not readiness of a real
remote service. The PostgreSQL case contacts the developer-operated
[existing read fixture](../tests/fixtures/postgresql/README.md), reads it without
mutation, and assumes it remains unchanged during evaluation. No infrastructure
is provisioned by this suite.

Run on the existing macOS ARM64 native analysis environment. Configure reviewed
tool-capable API models through `DAITA_STAGE_B_BENCHMARK_MODEL_IDS` (comma-separated,
one to six; default `openai:gpt-5.6-terra`). Export
`DAITA_STAGE_B_BENCHMARK_LLM_API_KEY` or provider-specific
`DAITA_STAGE_B_BENCHMARK_OPENAI_API_KEY`, `...ANTHROPIC_API_KEY`,
`...GEMINI_API_KEY` or `...GROK_API_KEY` privately. The harness reuses the existing
provider factory and does not load `.env` automatically. Set
`DAITA_FIXTURE_POSTGRES_PASSWORD` and, when needed,
`DAITA_FIXTURE_POSTGRES_PORT` (default `55432`) for the PostgreSQL case.
That case compares cold connections with an opt-in one-connection pool over two
fresh model runs per variant. Both answers use the same independent Decimal
oracle; the pooled variant must reuse one connection, complete each checkout,
and retain structural/query/cleanup diagnostics in its private report.

`DAITA_DATAOPS_REPEATS` accepts one to five independent repetitions (default one).
The default estimated-cost ceiling is $0.50 per run, configured through the
existing `DAITA_STAGE_B_BENCHMARK_MAX_COST_USD`, and $10.00 aggregate, configured
through `DAITA_DATAOPS_TOTAL_COST_USD`. Both must be finite and positive. Runs
reserve before dispatch, charge measured consumption, and retain the reservation
when usage is incomplete. The aggregate cap may prevent completion of the entire
matrix; those failures do not count as successful qualification. Use a serial
pytest invocation so this session budget is shared across all cases and models.

Collect or qualify the harness without live calls:

```bash
.venv/bin/python -m pytest tests/live/data_operations --collect-only \
  -o addopts="--tb=short -q --strict-markers"
.venv/bin/python -m pytest tests/acceptance/test_data_operations_harness.py \
  tests/mcp/test_read_harness.py
```

After explicit authorization and configuration of the existing resources:

```bash
DAITA_RUN_LIVE_DATA_OPERATIONS=1 \
DAITA_DATAOPS_REPORT_DIR=/private/tmp/daita-dataops-evidence \
.venv/bin/python -m pytest tests/live/data_operations \
  -o addopts="--tb=short -q --strict-markers" \
  --junitxml=/private/tmp/daita-dataops-live.xml -o junit_family=xunit1
```

Evidence is private JSON outside the repository. Without an explicit report
directory, one unique `/private/tmp/daita-dataops-...` directory is created per
invocation. JUnit properties identify each case report. Reports retain exact run
transcripts and model requests, result scores, fixture hashes, token/cost usage,
timings, scheduled run results and applicable Python computation evidence.
Inspect JUnit alongside JSON: setup failures can occur before a run is captured.
The authorization gate remains off during ordinary CI and collection.

Scheduled Python requires the same native macOS runtime as foreground Python.
The retained definition cases test advisory semantics within Daita; they do not
qualify an executable metric compiler. Warehouse integrations and Linux support
are outside this suite's scope.

### Real-model artifact lifecycle

`tests/live/artifacts/test_lifecycle.py` exercises three real `Agent.run` interactions:
creating a TXT artifact and a retained control, reading and saving the first after
reopening, and verifying its absence through model tools after owner deletion and
another reopen. It checks exact bytes, hashes, filesystem storage, preserved
transcripts, retained exported copies, the unaffected control, and removal of the
deleted registry row. Deletion uses the public typed owner API because model tools
do not grant deletion authority. Failure injection and crash timing remain in
`tests/artifacts/test_deletion.py`.

Run only after explicit live-model authorization:

```bash
DAITA_RUN_LIVE_ARTIFACT_LIFECYCLE=1 \
DAITA_ARTIFACT_LIVE_MODEL_ID=openai:gpt-5.6-terra \
DAITA_ARTIFACT_LIVE_MAX_COST_USD=0.15 \
.venv/bin/python -m pytest tests/live/artifacts/test_lifecycle.py \
  -o addopts="--tb=short -q --strict-markers" \
  --junitxml=/private/tmp/daita-artifact-lifecycle.xml
```

Set `DAITA_ARTIFACT_LIVE_LLM_API_KEY` or the selected provider's ordinary environment
key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `XAI_API_KEY`). The
default is one reviewed OpenAI model, with a $0.15 estimated-cost ceiling per run
and at most $0.45 across the three interactions. Temporary exports stay inside the
test's private directory. Collection does not contact a model or resolve credentials.

### Real remote MCP reads

`tests/live/mcp/test_interoperability.py` uses the production SDK transport against
the exact `MCP_HOST` endpoint. It has two separately authorized cases:

- `DAITA_RUN_LIVE_MCP=1` enables SDK inspection and one remote read, without an LLM.
- Both `DAITA_RUN_LIVE_MCP=1` and `DAITA_RUN_LIVE_MCP_LLM=1` enable two real OpenAI
  runs: inspect and attach, load/call in the same Agent, close/reopen the home,
  load/call again, verify exact arguments/provenance and grounded final answers.
  It uses `OPENAI_API_KEY`, defaults to `openai:gpt-5.6-terra`, and caps estimated
  model cost at `$0.50` per run (`$1.00` total by default). Override with
  `DAITA_LIVE_MCP_MODEL_ID` (a reviewed OpenAI tool model) and finite positive
  `DAITA_LIVE_MCP_MAX_COST_USD`.

Configure the exact read contract and an independently known result marker:

```bash
export MCP_HOST=https://mcp.firecrawl.dev/v2/mcp
export MCP_TOOL=firecrawl_scrape
export MCP_ARGUMENTS='{"url":"https://example.com","formats":["markdown"]}'
export MCP_EXPECT_TEXT='Example Domain'
# Set MCP_TOKEN privately to an existing Firecrawl API key for authenticated access.
# Omitting MCP_TOKEN selects no authentication; it does not start OAuth sign-in.
```

Firecrawl documents [API-key and limited keyless access](https://docs.firecrawl.dev/mcp-server/keyless)
at this endpoint. Its [OAuth endpoint](https://docs.firecrawl.dev/mcp-server)
requires an existing access token or a host-owned connection provider; this test
does not obtain or refresh one. An incompatible protocol or schema fails the
case before any paid model request; do not weaken admission or switch endpoints
after a failure. The example tool is subject to the server's current schemas,
availability, credit charges, and rate limits.

After authorizing the resource use and exporting credentials/configuration:

```bash
DAITA_RUN_LIVE_MCP=1 .venv/bin/python -m pytest \
  tests/live/mcp/test_interoperability.py \
  -o addopts="--tb=short -q --strict-markers" -m 'not requires_llm'

DAITA_RUN_LIVE_MCP=1 DAITA_RUN_LIVE_MCP_LLM=1 .venv/bin/python -m pytest \
  tests/live/mcp/test_interoperability.py \
  -o addopts="--tb=short -q --strict-markers" -m requires_llm \
  --junitxml=/private/tmp/daita-live-mcp.xml -o junit_family=xunit1
```

Pytest does not load `.env` automatically. Explicitly load its values into the
test process only after authorizing live execution. JUnit properties record
protocol, binding/schema identity, run ID, model requests, tokens, and estimated
cost. The disposable home retains the exact transcript for diagnosis.
These cases prove remote reads; external actions, approval, receipts, and OAuth
refresh require their own real-service qualification. The prior Context7-specific
smoke is now server-neutral; `DAITA_MCP_SMOKE_*` test settings are replaced by the
`MCP_*` settings above.

### Live MCP read workflow evaluations

`tests/live/mcp/test_read_workflows.py` uses disposable HTTP fixtures through the
official SDK. Dependent account/balance reads cover all three supported protocol
versions, 2/128-tool catalogs, normal responses, a 64 KiB text appendix, slow
responses and deliberate timeouts. Additional cases cover exact connector
selection with overlapping names, instructions embedded in untrusted results,
and schema drift between dependent calls. Random account IDs and verification
markers must come from tool results, including a marker at the end of the large
appendix. These cases exercise real model decisions and production execution;
their MCP servers are fixtures.

The slow fixture delays balance-response headers by two seconds and its streamed
body by one second. The timeout fixture sends a partial body, then waits sixteen
seconds against the production fifteen-second request deadline. It must yield
one bounded timeout, close the stream and answer without retrying or inventing
the balance. Keep these intentional faults separate from ordinary timeout rates.

`tests/live/mcp/test_remote_read_workflows.py` adds a real-service journey. A
natural request contains no tool name or JSON arguments. Three independent runs
measure a fresh owner, a reused owner and a restarted home, followed by one
question using the same conversation's evidence. Every remote read must inspect
current schemas and dispatch once; session reuse must avoid renegotiation. The
follow-up requests no new lookup and must use retained evidence.
Here, fresh/restarted refers to the SDK owner and session; the pass-through timing
transport can reuse underlying HTTP connections warmed during admission.

Authorize the matrix with `DAITA_RUN_LIVE_MCP_READS=1`. Remote execution also
requires `DAITA_RUN_LIVE_MCP=1` and the existing `MCP_HOST`, `MCP_TOOL`, optional
`MCP_TOKEN`, `MCP_EXPECT_TEXT` and `OPENAI_API_KEY` settings. For the Firecrawl
example, add:

```bash
export MCP_READ_PROMPT='Read https://example.com using the connected web service. Tell me its identifying title and briefly what it says about use in examples.'
export MCP_READ_ARGUMENT_MATCH='{"url":"https://example.com"}'
export MCP_EXPECT_TEXT='Example Domain'
```

The argument match is an independent subset oracle, not part of the prompt.
For other services, provide a matching natural request and result marker; set
`MCP_FOLLOWUP_PROMPT` when the default identifying-title question is unsuitable.

```bash
DAITA_RUN_LIVE_MCP_READS=1 DAITA_RUN_LIVE_MCP=1 \
DAITA_LIVE_MCP_READ_REPORT_DIR=/private/tmp/daita-mcp-read-evidence \
  .venv/bin/python -m pytest \
  tests/live/mcp/test_read_workflows.py \
  tests/live/mcp/test_remote_read_workflows.py \
  -o addopts="--tb=short -q --strict-markers" \
  --junitxml=/private/tmp/daita-mcp-read-evaluation.xml -o junit_family=xunit1
```

The full matrix contains 28 cases/31 Agent runs per repetition, with a `$0.50`
per-run ceiling (`DAITA_LIVE_MCP_READ_MAX_COST_USD`) and a `$3.00` aggregate ceiling
(`DAITA_LIVE_MCP_READ_TOTAL_COST_USD`) shared by these modules in one serial pytest
process. Run without distributed pytest workers. Each run reserves its whole
ceiling before model dispatch; complete usage releases unused allowance, while
missing usage retains the reservation. Budget exhaustion fails before that
Agent run dispatches a model request or tool call. Earlier binding admission
may already have inspected the server.
`DAITA_LIVE_MCP_READ_REPEATS` accepts 1–20 and defaults to 1. Repetition does not
increase the aggregate ceiling. Select cases and set an adequate explicit ceiling
before running a larger matrix; the default `$3.00` may stop it early. The model
is selected by `DAITA_LIVE_MCP_MODEL_ID`.

The serial read benchmark stops after two consecutive model-provider failures
by default. `DAITA_LIVE_MCP_READ_PROVIDER_FAILURE_LIMIT` accepts 1–5. A completed
Agent run resets this counter, including an honest answer about an MCP timeout;
tool errors and answer-oracle failures do not establish a provider failure.
The controller also stops when the remaining budget cannot reserve another run.
It completes ordinary teardown and saves the current case before exiting pytest
with status 1. Remaining selected cases are unexecuted, not passes or additional
failures. Failed attempts and incomplete-usage reservations remain in evidence.

JUnit links to JSON reports, retained on assertion failures as well as passes.
They include exact transcripts, canonical messages submitted to each model request,
protocol/schema identities, attachment time,
per-run wall/model/MCP time, first model progress, request/method counts, response
bytes, tool error codes, timeout counts, token usage and cost. Transport timing
includes response headers, the first streamed byte and complete consumption.
The wall-time residual includes framework work and scheduling; it is **not** a
measurement of schema-validation CPU time, and concurrent timings may overlap.
Reports omit HTTP headers, raw request bodies and raw transport exception text;
known environment credentials are redacted from saved evidence.

The remote follow-up checks both the retained read evidence in the submitted
request and the model's answer. Request capture distinguishes lost context from
a model declining to use available evidence; a scripted model alone cannot
qualify answer grounding.

OpenAI attempt diagnostics retain allowlisted SDK error types, upstream error
codes/types and the failure phase. Unknown error codes retain only a bounded
digest. Error messages, parameters and raw response bodies are excluded; observed
HTTP status and request-ID digests survive errors without response metadata.
These fields distinguish SDK-raised stream errors from decoded error events,
failed responses, missing terminal responses and progress deadlines. They do not
enable retries or establish complete usage after an interrupted request.
Quota rejections use the existing `rate_limit_error` category; the retained
upstream code/type distinguishes exhausted credits or spending limits from
request pacing. OpenAI's `credit_balance_exhausted` requires restoring the API
organization's credits before continuing the live sample. Retrying cannot
restore credits; see the [OpenAI error guide](https://developers.openai.com/api/docs/guides/error-codes).

Treat these measurements as samples, not latency guarantees or a connector
release gate. Compare repeated runs with the same model and service conditions;
keep failures and distinguish test-oracle errors, model choices, framework
failures and remote-service failures. The deterministic
`tests/mcp/test_read_harness.py` qualifies accounting, streamed timing, cleanup,
failure reporting and credential redaction without spending credentials.

#### Larger bounded latency sample

The following selects twenty runs each for normal dependent reads, larger results
and slow responses, five intentional timeout runs, and seven real-service cases
(21 reads plus seven follow-ups). It runs 72 cases/93 Agent runs in one serial
process with a shared `$10.00` estimated model-spend ceiling. It requires the
explicit live gates and configured credentials above; service credits are
separate from model spend. A whole-process deadline supplements the per-run and
request deadlines.

```bash
DAITA_RUN_LIVE_MCP_READS=1 DAITA_RUN_LIVE_MCP=1 \
DAITA_LIVE_MCP_READ_REPEATS=20 DAITA_LIVE_MCP_READ_TOTAL_COST_USD=10.00 \
DAITA_LIVE_MCP_READ_REPORT_DIR=/private/tmp/daita-mcp-latency-evidence \
  .venv/bin/python - <<'PY'
import subprocess
import sys

fixture = "tests/live/mcp/test_read_workflows.py::test_live_dependent_reads_discover_tools_and_bind_returned_account"
remote = "tests/live/mcp/test_remote_read_workflows.py::test_real_remote_natural_reads_reuse_session_restart_and_answer_follow_up"
cases = []
for repetition in range(20):
    for workload in ("normal", "large_result", "slow"):
        cases.append(f"{fixture}[{repetition}-2-2026-07-28-{workload}]")
    if repetition < 5:
        cases.append(f"{fixture}[{repetition}-2-2026-07-28-timeout]")
    if repetition < 7:
        cases.append(f"{remote}[{repetition}]")
result = subprocess.run(
    [sys.executable, "-m", "pytest", *cases,
     "-o", "addopts=--tb=short -q --strict-markers",
     "--junitxml=/private/tmp/daita-mcp-latency.xml", "-o", "junit_family=xunit1"],
    timeout=1800,
)
sys.exit(result.returncode)
PY
```

Summarize raw reports without model or credential I/O:

```bash
.venv/bin/python -m tests.support.mcp_read_harness \
  --report-dir /private/tmp/daita-mcp-latency-evidence \
  --output /private/tmp/daita-mcp-latency-summary.json
```

Use a fresh raw-report directory per batch and save the summary outside it.
The summary reports sample counts, median p50 and nearest-rank p95
(`ceil(0.95 * n)`, without interpolation), maximum latency, timeout rates, errors,
model requests and known estimated cost. Every attempt remains in the all-attempt
distribution; a separate distribution includes only passed cases with completed,
error-free runs. Cohorts separate workload, service and session phase. The remote
`all_reads` cohort additionally pools cold, warm and restarted reads; aggregate
cost counts every run once. Terminal reasons distinguish provider failures from
MCP timeouts. For fewer than twenty samples, this p95 is the maximum and offers
little tail confidence.
Twenty samples still provide an initial tail measurement rather than a production
service-level guarantee.

The fully offline job concurrency soak retains its explicit enable flag:

```bash
# Set DAITA_RUN_STAGE_B_CONCURRENCY_SOAK=1 only when choosing the soak.
.venv/bin/python -m pytest tests/slow \
  -o addopts="--tb=short -q --strict-markers"
```

Do not load `.env`, enable a live gate, or spend an external resource merely to
make validation green. Diagnose deterministic failures at their offline owner.

## Creating or reviewing a test

Identify the current contract, production owner, defect, scenario, and
observable failure before adding a case. Search the owner suite and relevant
acceptance journeys for existing coverage. Extend a genuine parameter family
when the boundary and oracle are the same; keep authorization, cancellation,
budget equality, rollback, receipt uncertainty, persistence/restart, and
exactly-once/no-replay scenarios distinct.

Exercise the real owner and mock only dependencies outside the behavior under
test. Scripted model output proves deterministic execution and forwarding, not
language understanding or tool-selection quality. Expected results must come
from an independent contract, source datum, or failure condition. Do not test
documentation prose, comments, statement counts, private local names, or a
mock's canned answer as product behavior.

When moving, merging, rewriting, or deleting tests, reconcile original and
final node IDs, parameter cases, markers, skips, and fixture behavior. Record a
reason for every deletion and name the retained coverage for every duplicate.
Never add a skip, weaken an assertion, or delete unique coverage to hide a
failure.

## Explicit entry points

The resident, diagnostic, and packaging entry points are modules so their
imports resolve from the repository root:

```bash
.venv/bin/python -m tests.diagnostics.live_stream_boundaries --help
.venv/bin/python -m tests.packaging.pipx_lifecycle_smoke --help
.venv/bin/python -m tests.packaging.managed_installer_lifecycle_smoke --help
```

The two packaging lifecycle modules require a candidate wheel for actual use.
A local `--help` check validates relocation only; CI's clean lifecycle jobs
remain the installed-wheel evidence.

PostgreSQL pool boundaries are covered offline by
`tests/data/postgresql/test_pool.py`, existing query/export tests and host lifecycle
cases. The injected native-pool fixture proves reuse, fresh transactions, source
and host bounds, admission retirement, cancellation, reset failure and safe close.
It does not establish real PostgreSQL transport performance. Use
`Agent.postgresql_read_diagnostics()` for aggregate checkout, structure, query and
cleanup metrics; connection counts distinguish creation from per-read transactions.

Paired real-model handoff cases are in
`tests/live/analysis/test_public_journeys.py` (exact export/bind versus capture) and
`tests/live/data_operations/test_workflows.py` (synchronous versus batched Python
reads). They retain the existing explicit `DAITA_RUN_LIVE_ANALYSIS=1` and
`DAITA_RUN_LIVE_DATA_OPERATIONS=1` authorization, credentials, per-run/aggregate
cost bounds and private reports. Numerical correctness is scored against fixture
oracles independently of latency, request count and usage. These cases remain
unperformed until those gates are supplied; scripted component models do not
qualify reasoning quality.
