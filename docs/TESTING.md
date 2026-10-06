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

`tests/storage/contracts` is the portable durable-backend suite. It runs SQLite
by default. With Docker and OpenSSL available, qualify both implementations and
PostgreSQL isolation/failure behavior with:

```bash
.venv/bin/python -m pytest tests/storage/contracts tests/storage/postgres --postgres
.venv/bin/python scripts/check_home_release_contract.py check
```

This command creates and removes a disposable PostgreSQL container, with TLS and
restricted login roles. It never accepts an ambient database URL. Missing fixture
prerequisites fail the command. The default suite excludes the PostgreSQL cases;
report them separately. See [storage contract](STORAGE_CONTRACT.md) and
[PostgreSQL state](POSTGRES_STATE.md) for remaining remote-home acceptance gates.

The same contracts include the artifact lifecycle with disposable injected object
storage. This qualifies registry/byte coordination on both state databases;
`tests/artifacts/test_s3_bytes.py` checks the S3 request and bounded-read contract.
These fixtures do not contact AWS or qualify a real S3 SDK, bucket, IAM policy,
version retention or deployed handover. See [artifact storage](ARTIFACT_STORAGE.md).

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
