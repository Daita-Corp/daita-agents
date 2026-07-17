# Daita v2 Replacement Status

This file is the persistent execution ledger for the isolated replacement
project. Update it before and after every material task.

## Current position

- **Active phase:** Phase 0 — architecture constitution and v1 oracle
- **Active task:** P0-07 — perform the final scoped diff, architecture scan,
  commit-hook run, and Phase 0 gate commit
- **Last completed task:** P0-06 — completed the root/v2 regression, static,
  isolation, build, install, and package-content verification matrix
- **Current checkpoint:** branch `next` at v1 baseline
  `b87df31873d33fffbf50498f5dc4d8892115e8f8` (`Memory hardening`,
  2026-07-13); worktree was clean before `next/` creation
- **Architecture-plan fingerprint:** ignored local source
  `docs/DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md`, SHA-256
  `403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935`
- **Exact next action:** run the final Phase 0 constitution/reference tests,
  inspect and stage only `next/`, run pre-commit and the scoped-diff checks,
  then commit the passing checkpoint as
  `chore(v2-phase-0): complete phase 0 gate`

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
| P0-07 | active | Passing P0-06 results | Final STATUS/PARITY/QUALITY/ADR evidence and coherent Phase 0 gate commit | clean scoped diff; no v1 import/fallback scan; pre-commit/format checks; commit hooks | P0-06 |

## Files/components being changed

- `next/STATUS.md` (active ledger)
- `next/decisions/` (P0-02 complete)
- `next/pyproject.toml`, `next/README.md`, `next/src/daita/__init__.py`
  (P0-03 complete)
- `next/tests/architecture/test_import_firewall.py` (P0-03 complete)
- `next/PARITY_MATRIX.md`, `next/TEST_DISPOSITION.csv`,
  `next/scripts/build_test_disposition.py`, and P0-04 architecture tests
  (P0-04 complete)
- `next/scripts/capture_v1_oracles.py`, four
  `next/tests/fixtures/v1/*.json` fixtures, and the fixture contract test
  (P0-05 complete)
- `next/QUALITY_GATES.md` (P0-06 complete; P0-07 final evidence active)
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

No feature implementation exists yet. No Phase 0 gate is claimed until P0-07
is committed.

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

## Credentials and external dependencies

- Live LLM and external database credentials have not been assumed or tested.
- PostgreSQL and production-provider live gates are later-phase requirements;
  unavailable credentials will be recorded without substituting mock results.
- Phase 0 is designed to complete with deterministic local tests only.
- The architecture plan is intentionally local and ignored by the root
  `.gitignore`; its fingerprint above anchors the governing version.

## Remaining Phase 0 gate requirements

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
- [ ] Passing Phase 0 evidence is committed as
      `chore(v2-phase-0): complete phase 0 gate`.
