# Daita v2 Replacement Status

This file is the persistent execution ledger for the isolated replacement
project. Update it before and after every material task.

## Current position

- **Active phase:** Phase 1 — loop laboratory
- **Active task:** P1-02 — prove text-only completion with a deterministic mock,
  static context, in-memory committed state/events, and the generic loop
- **Last completed task:** P1-01 — implemented and verified strict immutable
  canonical operation, loop, and provider-neutral model records
- **Current checkpoint:** Phase 0 gate commit
  `720adc8ac8c80f450fd9924349aa675f2c40cfe9`
  (`chore(v2-phase-0): complete phase 0 gate`)
- **Architecture-plan fingerprint:** ignored local source
  `docs/DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md`, SHA-256
  `403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935`
- **Exact next action:** commit the passing P1-01 checkpoint, then write one
  failing text-only trajectory proving the trigger/operation is committed
  before the model call, one turn and readiness decision precede terminal
  success, the exact final text is preserved, and no task/evidence exists

## Mandatory architecture re-read

After the Phase 0 gate commit and before any Phase 1 production edit, the full
text of plan Sections 6 and 15 was re-read on 2026-07-16. The implementation
sequence below therefore starts from these renewed constraints:

- the model owns semantic choice and the generic loop owns progression;
- the operation runtime is the only executor invocation boundary;
- a proposal is untrusted until validated and persisted;
- evidence and observations are unusable before durable acceptance;
- canonical records are provider-neutral and independently resumable; and
- no real catalog, memory, monitor, production provider, or broad extension
  SDK belongs in Phase 1.

## Current architectural decisions

- Build the replacement only under `next/`, using `next/src/daita/` as the
  final import namespace.
- Treat root `daita/` as a read-only behavioral oracle. V2 production code may
  never import or execute it.
- Preserve one provider-neutral generic loop and one operation-runtime
  executor boundary.
- Keep authoritative current state plus an append-only committed event log;
  do not require full event sourcing.
- Use a distinct pre-cutover local state root (`~/.daita-next/` by default) and
  never open v1 state implicitly.
- Prefer a documented fresh-state v2 start unless Phase 9 evidence establishes
  a concrete migration requirement.
- Phase 0 through Phase 9 are authorized. Phase 10, root-package replacement,
  pushing, publishing, PR creation, and release work are excluded.

The binding rationale and consequences are recorded in `next/decisions/`.

## Ordered Phase 0 tasks

| ID | Status | Inputs | Expected output | Tests/evidence | Dependencies |
| --- | --- | --- | --- | --- | --- |
| P0-01 | complete | `AGENTS.md`; full architecture plan; root source/tests/config; Git state | Recorded baseline, constraints, package/public API inventory, and clean-worktree assessment | `git status --short --branch`; safe-suite collection; plan hash | none |
| P0-02 | complete | Plan Sections 6, 15, 16, 19, 20, and 23; P0-01 findings | Numbered ADRs for loop ownership, terminology, execution boundary, persistence, learning, extensions, state root/migration, public API/CLI/secrets/retention, provider/cutover scope | 14 accepted ADRs; deterministic count/status/required-decision assertions passed | P0-01 |
| P0-03 | complete | Import-firewall requirements; Python support; isolated packaging constraints | Minimal installable `next/` project, README, final-name package stub, import-firewall and architecture tests | 6 isolated firewall tests pass; clean-copy sdist/wheel build and fresh-venv import pass; Black passes | P0-02 |
| P0-04 | complete | Mandatory preservation inventory; root exports, extras, README, tests | Complete parity/disposition matrix with v1 reference, v2 owner/test, phase class, and disposition; explicit MVP versus cutover gates | 109 matrix rows validated; 51 mandatory behaviors, 61/23/13 exports, 44 extras, and all 164 test files covered exactly once | P0-02 |
| P0-05 | complete | Stable v1 behavioral oracles and neutral fixtures | Serialized golden fixtures for terminology, provider-neutral transcripts, task/evidence lifecycle, readiness failures, and public surface snapshots | V1 capture reproducibility passes; 151 focused v1 oracle tests and 19 isolated v2 tests pass | P0-04 |
| P0-06 | complete | P0-03 through P0-05 artifacts | Phase 0 regression and architecture evidence, including exact environment/results and known baseline failures | 2,498 root safe tests; 201 focused runtime tests; 19 v2 tests on each of Python 3.11/3.12 at P0-06; Black/compile/mypy/pyright; isolated and root distribution builds | P0-03, P0-04, P0-05 |
| P0-07 | complete | Passing P0-06 results | Final STATUS/PARITY/QUALITY/ADR evidence and coherent Phase 0 gate commit | clean scoped diff; no v1 import/fallback scan; all hooks passed; commit `720adc8` | P0-06 |

## Ordered Phase 1 tasks

| ID | Status | Inputs | Expected output | Tests/evidence | Dependencies |
| --- | --- | --- | --- | --- | --- |
| P1-01 | complete | Re-read Sections 6 and 15; Phase 1 work/gate; ADRs 0002–0005 and 0008; v1 neutral fixtures | Minimal immutable canonical loop/model records and `ModelProvider` protocol, with no provider wire format or executor reference | 25 focused tests; strict JSON mutation isolation; provider-neutral message/tool records; trigger/operation/turn/action/observation/readiness/exit invariants; exact Decimal usage; mypy/pyright clean | Phase 0 gate |
| P1-02 | active | P1-01 records | Text-only vertical slice with deterministic scripted mock model, static context, in-memory operation/turn/event state, direct generic loop, and readiness commit | Text-only completion trajectory; operation/turn/model-response/readiness/terminal event order; optional session | P1-01 |
| P1-03 | pending | P1-02 loop; fake capability contract | Minimal capability registry/tool projection, fake read executor, in-memory task/evidence state, and operation runtime submission path | Proposal → persisted task → sole executor boundary → accepted evidence → durable observation → next turn → final answer | P1-02 |
| P1-04 | pending | P1-03 action path | Structured invalid-action observations, bounded repair turns, normalized failure fingerprints, and no-progress termination | Invalid proposal repairs once; repeated identical failure stops early; changed action may progress | P1-03 |
| P1-05 | pending | P1-02 through P1-04 | Cancellation checks plus turn, action, repair, identical-retry, wall-time, task-timeout, token, observation, and estimated-cost budgets | Every budget exits honestly with inspectable state/events; cancellation persists interruption state | P1-04 |
| P1-06 | pending | Complete loop laboratory | Deterministic scripted acceptance trajectories and architecture assertions covering every Phase 1 gate | No domain/provider branch; only operation runtime contains executor invocation; full suites on Python 3.11/3.12; static/build scans | P1-05 |
| P1-07 | pending | Passing P1-06 results | Final Phase 1 ledger/evidence and coherent gate commit | scoped diff; root oracle unchanged; pre-commit; `chore(v2-phase-1): complete phase 1 gate` | P1-06 |

## Files/components being changed or planned

- Phase 0 artifacts are complete at `720adc8`.
- P1-01 completed production owners: the narrow shared JSON-value utility
  `next/src/daita/_json.py`, `next/src/daita/loop/models.py`,
  `next/src/daita/operations/models.py`, `next/src/daita/llm/models.py`, and
  `next/src/daita/llm/protocols.py`.
- P1-01 completed tests: focused modules under `next/tests/unit/loop/`,
  `next/tests/unit/operations/`, and `next/tests/unit/llm/`, plus direct tests
  for strict, recursively immutable canonical JSON values. The shared utility
  already has more than three record consumers and avoids parallel mutable
  payload rules.
- P1-02 planned owners: `next/src/daita/llm/providers/mock.py`,
  `next/src/daita/operations/runtime.py`, and `next/src/daita/loop/driver.py`;
  static context/readiness doubles remain test-owned until another real
  implementation requires a production subsystem.
- Later Phase 1 owners are created only when their vertical slice becomes the
  active task; the target tree is not being scaffolded in advance.
- Root `daita/`: **not being changed**

## Tests last run

Environment: repository `.venv`, Python 3.11.15, pytest 9.1.1.

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest --collect-only -q tests/ -m 'not requires_llm and not requires_db'` | PASS — 2,719 collected; 221 deselected; 2,498 selected; no collection errors (read-only P0-01 audit) |
| `set -e;` count 14 numbered ADRs; count 14 accepted statuses; assert baseline, sole executor boundary, and Phase 10 exclusion text | PASS — exit 0 (P0-02 completeness smoke) |
| `PYTHONPATH=src:../.venv/lib/python3.11/site-packages ../.venv/bin/python -S -m pytest tests/architecture/test_import_firewall.py -q` (cwd `next/`) | PASS — 6 passed; `-S` prevents processing the root editable-install path |
| Clean-copy `python -m build --no-isolation`, install wheel with `--no-deps` into a fresh venv, then isolated import/package-content assertions | PASS — sdist and wheel built; installed `2.0.0a0` imported only from fresh `site-packages`; wheel contains only `daita/__init__.py` and distribution metadata |
| `.venv/bin/black --check next/src next/tests` | PASS — 2 files unchanged |
| `../.venv/bin/python scripts/build_test_disposition.py --check` then isolated `pytest tests/architecture -q` (cwd `next/`, `-S`) | PASS — 14 passed; generated inventory matches all 164 tracked test modules exactly |
| `.venv/bin/python next/scripts/capture_v1_oracles.py --check` | PASS — root v1 paths match the baseline and all four canonical JSON fixtures reproduce byte-for-byte |
| `.venv/bin/python -m pytest tests/unit/runtime/test_primitives.py tests/unit/agents/test_chat_runtime.py tests/unit/db/test_plan_validation.py tests/unit/db/test_agent_loop_completion_targets.py tests/unit/db/test_public_api.py -q` | PASS — 151 passed in 0.51s |
| `.venv/bin/python -m pytest tests/ -m "not requires_llm and not requires_db"` | PASS — 2,498 passed, 221 deselected in 10.84s |
| `.venv/bin/python -m pytest tests/unit/runtime/test_kernel.py tests/unit/db/test_agent_loop_completion_targets.py tests/unit/db/test_agent_loop_concurrency.py tests/unit/db/test_agent_loop_phase2.py tests/unit/db/test_governance_runtime.py -q` | PASS — 201 passed in 0.81s |
| Isolated `PYTHONPATH=src:../.venv/lib/python3.11/site-packages ../.venv/bin/python -S -m pytest tests/ -q` (cwd `next/`) | PASS — 19 passed on CPython 3.11.15 |
| Isolated `PYTHONPATH=src:/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/site-packages python3.12 -S -m pytest tests/ -q` (cwd `next/`) | PASS — 19 passed on CPython 3.12.7 |
| `.venv/bin/black --check next/src next/tests next/scripts`; `.venv/bin/python -m compileall -q next/src next/tests next/scripts` | PASS — 7 Python files unchanged; byte compilation succeeded |
| `../.venv/bin/python -m mypy src/daita tests scripts/build_test_disposition.py` (cwd `next/`) | PASS — no issues in 6 source files |
| `npx --yes pyright@1.1.411 --pythonpath ../.venv/bin/python src/daita tests scripts/build_test_disposition.py` (cwd `next/`) | PASS — 0 errors and 0 warnings |
| Final clean-copy v2 build, archive-content scan, and fresh Python 3.11/3.12 wheel installs | PASS — wheel 5 entries, sdist 14; both imports resolved to fresh `site-packages` at version `2.0.0a0`; no tests/scripts/decisions/nested `next/` packaged |
| Root distribution build with a physical `next/` alongside v1, followed by archive/content scan | PASS — root wheel 401 entries, sdist 442; neither included `next/`; root wheel retained v1 `1.0.0` with no v2 version string |
| Isolated final constitution test | PASS — 5 passed |
| Final isolated v2 suite with `-o addopts=''` on CPython 3.11.15 and 3.12.7 | PASS — 24 passed in 0.10s on 3.11; 24 passed in 0.12s on 3.12 |
| Final Black, compile, mypy, and pyright checks after adding the constitution test | PASS — 8 files unchanged; compilation succeeded; mypy found no issues in 7 files; pyright reported 0 errors and 0 warnings |
| Generated-inventory/v1-fixture checks, all isolated architecture tests, production-reference/symlink scans, and root-baseline diff | PASS — 19 architecture tests; every command exited zero; no prohibited reference, symlink, or root-oracle change |
| Scoped stage/diff review and configured pre-commit hooks | PASS — 33 staged files, all under `next/`; diff check clean; all hooks passed after the end-of-file fixer normalized 16 Markdown files |
| Initial P1-01 focused test run before production modules existed | EXPECTED RED — 4 collection errors for missing `_json`, `llm`, `loop`, and `operations` modules |
| Final P1-01 focused unit suite | PASS — 25 passed in 0.04s |
| Final P1-01 full isolated suite on CPython 3.11.15 and 3.12.7 | PASS — 49 passed in 0.19s on 3.11; 49 passed in 0.15s on 3.12 |
| P1-01 Black, compile, mypy, pyright, architecture, dependency-direction, and root-oracle checks | PASS — 20 Python files formatted; compile succeeded; mypy found no issues in 13 files; pyright 0 errors/warnings; 19 architecture tests; no operations→loop import; root unchanged |
| P1-01 clean-copy sdist/wheel build and content scan | PASS — build succeeded; wheel contained 13 entries including all 5 required contract modules and no tests or nested `next/` path |

Phase 0 and P1-01 are complete. P1-02 has not yet added its production slice.

## Known failures and baselines

- No failures were observed during safe-suite collection.
- P0-03 red/repair history: the first root-cwd `PYTHONPATH=next/src` smoke loaded
  v1 because the current directory precedes `PYTHONPATH`; the permanent test
  runs from `next/` and proves every loaded `daita` module is v2. The first
  firewall test run found an empty-`Path('.')` assertion bug, which was fixed.
- The first clean-copy build lacked the `wheel` command in the repository
  venv. Declared `wheel>=0.45.0` as a dev dependency, installed cached wheel
  0.47.0 locally, and reran successfully. No tracked root file changed.
- The first Black check found both new Python files unformatted; Black was run
  and the subsequent check passed.
- The first staged pre-commit run normalized the final newline in 16 new
  Markdown files. Only `next/` was restaged, and the complete hook set then
  passed.
- The root distribution build emitted pre-existing setuptools deprecation
  warnings for its license metadata and classifier. Both archives were valid,
  and the read-only root package is unchanged.
- No current test, static-analysis, isolation, install, or build failure is
  outstanding.
- P1-01 red/repair history: the initial test-first collection failed on the
  intentionally absent modules; duplicate `test_models.py` basenames then
  caused a pytest import collision and were renamed; the first static pass
  exposed formatting and JSON-value type-narrowing errors, which were repaired
  without ignores in production code.
- A pre-checkpoint dependency review found `operations.models` importing loop
  checkpoint types. Trigger, operation, proposal, and observation ownership was
  consolidated under operations; loop phase/state/turn/readiness/exit remained
  under loop; their atomic pairing is deferred to the P1-02 runtime owner.

## Credentials and external dependencies

- Live LLM and external database credentials have not been assumed or tested.
- PostgreSQL and production-provider live gates are later-phase requirements;
  unavailable credentials will be recorded without substituting mock results.
- Phase 0 is designed to complete with deterministic local tests only.
- The architecture plan is intentionally local and ignored by the root
  `.gitignore`; its fingerprint above anchors the governing version.

## Phase 0 gate record

- [x] Required numbered ADRs make subsystem ownership unambiguous.
- [x] `next/README.md`, `next/pyproject.toml`, `next/PARITY_MATRIX.md`, and
      `next/QUALITY_GATES.md` are staged only after the final scoped review.
- [x] `next/decisions/` contains all Phase 0 decisions.
- [x] Import-firewall test proves `daita.__file__` resolves under
      `next/src/daita/` and root v1 is neither imported nor executed.
- [x] Repository architecture scan rejects known v1 internal references.
- [x] V1 public-feature and mandatory-behavior disposition matrix is complete.
- [x] Neutral golden fixtures and test-disposition inventory are captured.
- [x] Root baseline and focused oracle results are recorded exactly.
- [x] `next/` builds, installs, and tests in isolation on Python 3.11.
- [x] Root distribution excludes `next/` before cutover.
- [x] No feature implementation or compatibility fallback has been added.
- [x] Passing Phase 0 evidence is committed as `720adc8`
      (`chore(v2-phase-0): complete phase 0 gate`).

## Remaining Phase 1 gate requirements

- [ ] Text-only response completes through the generic loop.
- [ ] Fake read follows the durable task/execution/evidence/observation order.
- [ ] Invalid action becomes a bounded observation and repair turn.
- [ ] Repeated identical failure stops early with an inspectable reason.
- [ ] Cancellation persists an inspectable state.
- [ ] Turn, action, repair, time, token, observation, and cost budgets terminate
      honestly.
- [ ] The loop contains no domain-specific or provider-specific branch.
- [ ] A repository assertion proves only the operation runtime invokes the
      executor.
- [ ] Root v1 oracle paths remain unchanged.
- [ ] Passing Phase 1 evidence is committed as
      `chore(v2-phase-1): complete phase 1 gate`.
