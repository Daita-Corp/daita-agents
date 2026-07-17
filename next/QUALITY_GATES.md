# Daita v2 quality-gate evidence

This is the authoritative record of commands actually run for the isolated
replacement. A mock result is never recorded as a live result. `PASS` means
the exact command shown exited zero in the stated environment; `NOT RUN` is
not a passing gate.

## Governing baseline

- Branch: `next`
- Root v1 oracle commit: `b87df31873d33fffbf50498f5dc4d8892115e8f8`
- Architecture plan SHA-256:
  `403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935`
- Phase scope: 0 through 9; Phase 10 is excluded.
- Root `daita/`, root `tests/`, and root `pyproject.toml` must remain identical
  to the oracle until a separately authorized v1 fix.

## Current local environment

Recorded 2026-07-16 in `America/Chicago`:

- macOS 26.4.1 (Darwin 25.4.0, arm64)
- Repository venv: CPython 3.11.15, pytest 9.1.1, setuptools 82.0.1,
  wheel 0.47.0
- Available secondary interpreter: CPython 3.12.7, pytest 8.4.1 at
  `/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/site-packages`
- Live LLM/database credentials: not assumed or used in Phase 0

## Phase 0 — architecture constitution and v1 oracle

Status: **P0-06 verification complete; P0-07 final scoped review in progress.**

### Executed passing evidence

| ID | Working directory | Exact command | Result |
| --- | --- | --- | --- |
| P0-Q01 | repository root | `.venv/bin/python -m pytest --collect-only -q tests/ -m 'not requires_llm and not requires_db'` | PASS — 2,719 collected, 221 deselected, 2,498 selected; no collection error |
| P0-Q02 | repository root | ADR count/status assertions for 14 numbered decisions plus required baseline/executor/Phase-10 text | PASS — exit 0 |
| P0-Q03 | `next/` | `PYTHONPATH=src:../.venv/lib/python3.11/site-packages ../.venv/bin/python -S -m pytest tests/architecture/test_import_firewall.py -q` | PASS — 6 passed |
| P0-Q04 | clean copy under `/private/tmp` | `.venv/bin/python -m build --no-isolation`; fresh venv `pip install --no-deps <wheel>`; isolated version/origin and wheel-content assertions | PASS — sdist/wheel built; v2 imported from fresh `site-packages`; wheel contained only `daita/__init__.py` and distribution metadata |
| P0-Q05 | repository root | `.venv/bin/black --check next/src next/tests` | PASS — 2 files unchanged at P0-03 checkpoint |
| P0-Q06 | `next/` | `../.venv/bin/python scripts/build_test_disposition.py --check` followed by isolated `pytest tests/architecture -q` | PASS — 14 passed; 164 tracked v1 test files covered exactly once |
| P0-Q07 | repository root | `.venv/bin/python next/scripts/capture_v1_oracles.py --check` | PASS — 4 JSON fixtures reproduce byte-for-byte and root oracle paths match baseline |
| P0-Q08 | repository root | `.venv/bin/python -m pytest tests/unit/runtime/test_primitives.py tests/unit/agents/test_chat_runtime.py tests/unit/db/test_plan_validation.py tests/unit/db/test_agent_loop_completion_targets.py tests/unit/db/test_public_api.py -q` | PASS — 151 passed in 0.51s |
| P0-Q09 | `next/` | `PYTHONPATH=src:../.venv/lib/python3.11/site-packages ../.venv/bin/python -S -m pytest tests/ -q` | PASS — 19 passed |
| P0-Q10 | repository root | `.venv/bin/black --check next/src next/tests next/scripts` then `git diff --quiet b87df31873d33fffbf50498f5dc4d8892115e8f8 -- daita tests pyproject.toml` | PASS — 7 Python files formatted; root oracle unchanged |
| P0-Q11 | repository root | `.venv/bin/python -m pytest tests/ -m "not requires_llm and not requires_db"` | PASS — 2,498 passed, 221 deselected in 10.84s |
| P0-Q12 | repository root | `.venv/bin/python -m pytest tests/unit/runtime/test_kernel.py tests/unit/db/test_agent_loop_completion_targets.py tests/unit/db/test_agent_loop_concurrency.py tests/unit/db/test_agent_loop_phase2.py tests/unit/db/test_governance_runtime.py -q` | PASS — 201 passed in 0.81s |
| P0-Q13 | `next/` | `PYTHONPATH=src:../.venv/lib/python3.11/site-packages ../.venv/bin/python -S -m pytest tests/ -q` | PASS — 19 passed on CPython 3.11.15 |
| P0-Q14 | `next/` | `PYTHONPATH=src:/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/site-packages python3.12 -S -m pytest tests/ -q` | PASS — 19 passed on CPython 3.12.7 |
| P0-Q15 | repository root | `.venv/bin/black --check next/src next/tests next/scripts` and `.venv/bin/python -m compileall -q next/src next/tests next/scripts` | PASS — Black reported 7 files unchanged; byte compilation exited zero |
| P0-Q16 | `next/` | `../.venv/bin/python -m mypy src/daita tests scripts/build_test_disposition.py` | PASS — no issues in 6 source files |
| P0-Q17 | `next/` | `npx --yes pyright@1.1.411 --pythonpath ../.venv/bin/python src/daita tests scripts/build_test_disposition.py` | PASS — 0 errors, 0 warnings, 0 information messages |
| P0-Q18 | clean copy `/private/tmp/daita-v2-phase0-final.b4SRsQ` | Build sdist/wheel with `python -m build --no-isolation`; inspect archives; install wheel into fresh CPython 3.11 and 3.12 venvs; assert version and import origin | PASS — wheel had 5 entries and sdist 14; neither contained tests, scripts, decisions, or a nested `next/`; both fresh installs imported `2.0.0a0` from their own `site-packages` |
| P0-Q19 | root copy `/private/tmp/daita-root-phase0.UL8asq` containing a physical `next/` tree | Build root sdist/wheel with `.venv/bin/python -m build --no-isolation`; inspect archive paths and root `daita/__init__.py` | PASS — wheel had 401 entries and sdist 442; neither contained `next/`; root wheel retained v1 version `1.0.0` and contained no v2 version string |
| P0-Q20 | `next/` | Isolated `pytest tests/architecture/test_phase0_constitution.py -q` after adding the final constitution assertions | PASS — 5 passed |
| P0-Q21 | `next/` | CPython 3.11 and 3.12 isolated `pytest -o addopts='' tests/ -q` runs after all Phase 0 tests were final | PASS — 24 passed in 0.10s on 3.11; 24 passed in 0.12s on 3.12 |
| P0-Q22 | `next/` | `../.venv/bin/black --check src tests scripts`; compile all Python; mypy; pyright 1.1.411 | PASS — 8 files unchanged; compilation succeeded; mypy found no issues in 7 files; pyright reported 0 errors and 0 warnings |
| P0-Q23 | repository root and `next/` | Recheck generated inventory and v1 fixtures; run all 19 architecture tests; scan production imports/fallbacks and symlinks; compare root oracle paths to baseline | PASS — every check exited zero; no prohibited production reference or symlink was found; root oracle remained unchanged |
| P0-Q24 | repository root | Stage only `next/`; run `git diff --cached --check`, scoped path review, and all configured pre-commit hooks | PASS — 33 staged files, every path under `next/`; no diff-check error; trailing-whitespace, end-of-file, merge-conflict, large-file, and Black hooks passed |

### Red/repair evidence

These failures were fixed and are not current gate failures:

| Attempt | Result | Resolution |
| --- | --- | --- |
| Root-cwd `PYTHONPATH=next/src .venv/bin/python -S -c 'import daita'` | FAIL — current directory resolved root v1 before `PYTHONPATH`; missing `pydantic` then exposed the wrong origin | Permanent subprocess firewall runs with cwd `next/`, disables site processing, and asserts every loaded `daita` module path |
| First import-firewall pytest run | FAIL — test indexed empty `Path('.').parts` | Made the root packaging assertion handle the implicit `.` search root; 6/6 then passed |
| First clean-copy build | FAIL — no `wheel` command in repository venv | Declared `wheel>=0.45.0`, installed cached wheel 0.47.0, reran build/install successfully |
| First Black check | FAIL — 2 new files needed formatting | Ran Black and reran the check successfully |
| First staged pre-commit run | FAIL — end-of-file fixer normalized 16 new Markdown files | Restaged only `next/` and reran every configured hook successfully |

The root-distribution build emitted setuptools deprecation warnings for its
existing license metadata and license classifier. They are pre-existing root
packaging warnings, did not affect either archive, and are outside the
read-only Phase 0 scope.

### Outstanding Phase 0 checks

- [x] Full root safe suite on Python 3.11.
- [x] Focused runtime kernel/loop regression set.
- [x] Complete v2 suite on Python 3.11 after all Phase 0 artifacts are final.
- [x] Complete v2 suite on available Python 3.12.
- [x] Black, compile, and available type checks.
- [x] V2 clean-copy sdist/wheel build, fresh install, and content smoke after
      all Phase 0 artifacts are final.
- [x] Root wheel build/content scan proving `next/` is excluded.
- [x] Final architecture/reference scans and clean scoped diff.
- [ ] Phase 0 gate commit with final ledger evidence.

## Phases 1 through 9

No later-phase gate has been run or claimed. Each phase receives its own
command/result section only after Phase 0 passes and Sections 6 and 15 are
re-read.

## Live and external gates

- Phase 0 requires no live provider or external database.
- Production-model, PostgreSQL, and other live checks are **NOT RUN** and will
  be recorded in their owning later phases.
- No credential presence is inferred from local environment files, and no
  mock result will satisfy a live gate.
