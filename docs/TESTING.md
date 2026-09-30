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

### Real remote MCP reads

`tests/live/mcp/test_interoperability.py` uses the production SDK transport against
the exact `MCP_HOST` endpoint. It has two separately authorized cases:

- `DAITA_RUN_LIVE_MCP=1` enables SDK inspection and one remote read, without an LLM.
- Both `DAITA_RUN_LIVE_MCP=1` and `DAITA_RUN_LIVE_MCP_LLM=1` enable one real OpenAI
  agent run: inspect, admit one read tool, close/reopen the home, load and invoke
  the admitted tool, verify binding provenance, and check a grounded final answer.
  It uses `OPENAI_API_KEY`, defaults to `openai:gpt-5.6-terra`, and caps estimated
  model cost at `$0.50`. Override with `DAITA_LIVE_MCP_MODEL_ID` (a reviewed OpenAI
  tool model) and finite positive `DAITA_LIVE_MCP_MAX_COST_USD`.

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
case before model construction; do not weaken admission or switch endpoints
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
