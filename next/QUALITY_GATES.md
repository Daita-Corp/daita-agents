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

Status: **COMPLETE at gate commit
`720adc8ac8c80f450fd9924349aa675f2c40cfe9`.**

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
- [x] Phase 0 gate commit with final ledger evidence (`720adc8`).

## Phase 1 — loop laboratory

Status: **Phase 1 gate complete in the exact gate commit containing this
ledger; Phase 2 is active and Phases 3–9 have not started.**

Plan Sections 6 and 15 were re-read in full after the Phase 0 commit and before
the first Phase 1 production edit. Phase 1 evidence will be appended only for
commands actually run.

### Executed P1-01 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P1-Q01 | `next/` | Run the new P1-01 unit tests before adding production modules | EXPECTED RED — 4 collection errors identified the absent `_json`, `llm`, `loop`, and `operations` owners |
| P1-Q02 | `next/` | Isolated focused `pytest -o addopts='' tests/unit -q` after contract and ownership review | PASS — 25 passed in 0.04s |
| P1-Q03 | `next/` | Complete isolated suite on CPython 3.11.15 and 3.12.7 | PASS — 49 passed in 0.19s on 3.11; 49 passed in 0.15s on 3.12 |
| P1-Q04 | `next/` | Black check, byte compilation, mypy, pyright 1.1.411, all architecture tests, operations dependency scan, and root-oracle diff | PASS — 20 files formatted; compile succeeded; mypy clean across 13 files; pyright 0 errors/warnings; 19 architecture tests; no operations→loop import; root unchanged |
| P1-Q05 | clean copy `/private/tmp/daita-v2-p1-01.VXrIhE` | Build sdist/wheel without isolation; inspect wheel for required modules and forbidden test/nested paths | PASS — 13-entry wheel contained all 5 required contract modules; no tests or nested `next/` path |

### P1-01 red/repair evidence

- Duplicate `test_models.py` leaf names initially collided under pytest's
  import mode; each test module now has a responsibility-specific unique name.
- The first static pass required Black formatting and exposed overly broad
  mapping annotations in the JSON boundary; the boundary and tests were typed
  precisely, then mypy and pyright passed without a production suppression.
- A dependency review rejected an operations→loop model import. Operation
  boundary records now belong to operations, loop progression records belong
  to loop, and P1-02 will pair them transactionally in the runtime owner.

### Executed P1-02 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P1-Q06 | `next/` | Run the new mock-provider and text-only acceptance tests before adding their production owners | EXPECTED RED — 2 collection errors for the absent provider package and loop/runtime modules |
| P1-Q07 | `next/` | Isolated focused mock, runtime-commit, and text-only acceptance tests | PASS — 7 passed in 0.03s |
| P1-Q08 | `next/` | Complete isolated suite on CPython 3.11.15 and 3.12.7 | PASS — 56 passed in 0.20s on 3.11; 56 passed in 0.24s on 3.12 |
| P1-Q09 | `next/` | Black check, byte compilation, mypy, pyright 1.1.411, architecture tests, and corrected root-oracle diff | PASS — 27 files formatted; compile succeeded; mypy clean across 25 files; pyright 0 errors/warnings; full suite included architecture tests; root unchanged |
| P1-Q10 | clean copy `/private/tmp/daita-v2-p1-02.19EFBE` | Build sdist/wheel without isolation; inspect wheel for the mock provider, driver, runtime, and forbidden test/nested paths | PASS — 17-entry wheel contained all 3 new runtime modules; no tests or nested `next/` path |

### P1-02 red/repair evidence

- The first implementation kept `Observation` at its pre-review import path;
  the focused collection error was repaired by importing the canonical record
  from its operations owner.
- The runtime initially mutated its private state object directly under a
  lock. It now applies every transition to a private working copy and publishes
  only after the state and corresponding events are complete; an injected
  event-commit failure leaves the committed snapshot exactly unchanged.
- A root-oracle command used a mistyped long checkpoint hash and Git returned
  `bad object`. The exact hash was read from Git, corrected in `STATUS.md`, and
  the oracle-path diff passed.

### Executed P1-03 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P1-Q11 | `next/` | Run the new capability unit and fake-read acceptance tests before adding the production capability owner | EXPECTED RED — 2 collection errors for the intentionally absent `daita.capabilities` module |
| P1-Q12 | `next/` | Final isolated capability, adversarial execution-boundary, context-tool-boundary, runtime-commit, text-only, and fake-read tests | PASS — 30 passed in 0.07s |
| P1-Q13 | `next/` | Complete isolated suite with `-S`, explicit v2 `PYTHONPATH`, and `-o addopts=''` on CPython 3.11.15 and 3.12.7 | PASS — 81 passed in 0.23s on each interpreter |
| P1-Q14 | `next/` | Black check, byte compilation, full mypy, and pyright 1.1.411 across source, tests, and inventory script | PASS — Black reported 32 files unchanged; compilation succeeded; mypy found no issues in 31 source files; pyright exited zero with no diagnostics (npm emitted only its nonfatal `unsafe-perm` warning) |
| P1-Q15 | repository root and `next/` | Independent final boundary review; sole `.execute()` AST/text checks; absolute/v1 import scan; symlink scan; root-oracle diff; `git diff --check`; scoped status review | PASS — no review blocker, prohibited reference, symlink, root change, diff error, or path outside `next/`; the sole executor call is in `operations/runtime.py` |
| P1-Q16 | clean copy `/private/tmp/daita-v2-p1-03.kp5giU` | Build sdist/wheel without isolation and inspect the wheel for P1-03 modules and forbidden test/nested paths | PASS — build succeeded; 18-entry wheel contains `capabilities.py`, the generic driver, and operation runtime; no tests or nested `next/` path |
| P1-Q17 | repository root | Stage the 11 reviewed P1-03 paths under `next/`; run `git diff --cached --check` and all configured hooks; create the scoped checkpoint | PASS — every staged path was under `next/`; trailing-whitespace, end-of-file, merge-conflict, large-file, and Black hooks passed; commit `e5258c0a0234e377b4581c744073609fe9e3d999` |

### P1-03 red/repair evidence

- The first combined path run passed 18 tests and failed only the two-read
  trajectory because its test executor asserted that the entire operation had
  no earlier evidence. The pre-I/O invariant is task-scoped; after correcting
  that assertion, all 19 focused tests passed.
- The first focused mypy pass found tuple-cardinality inference in the test
  transcript builder. Its canonical-message tuple was annotated explicitly;
  production code required no suppression.
- Final adversarial review found that same-name tool definitions were compared
  by name only, per-turn domain visibility was not rebound to the context
  request, a mutable executor identity could strand a RUNNING task after I/O,
  and provider call IDs were incorrectly operation-scoped. Exact declaration
  and domain-set equality, a terminal identity-failure path, and turn-scoped
  call identity repaired those contracts. Regression tests also inject
  task-start, evidence-terminal, and observation-event failures to prove the
  copy-on-write checkpoints publish no partial state.

### Executed P1-04 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P1-Q18 | `next/` | Run new record tests before adding `ActionRejection` and `LoopBudgets` | EXPECTED RED — 2 collection errors for the intentionally absent records |
| P1-Q19 | `next/` | Run new scripted action-repair trajectories before adding loop budgets/progression | EXPECTED RED — 2 failures because `AgentLoop.__init__` did not accept `budgets` |
| P1-Q20 | `next/` | Final focused loop/operation record, runtime repair/readiness, and scripted acceptance modules | PASS — 42 passed in 0.08s; covers exact/canonical fingerprints, bounded corrections, evidence resets, thresholds, complete multi-call skip transcripts, and atomic failure injection |
| P1-Q21 | `next/` | Complete isolated suite with `-S`, explicit v2 `PYTHONPATH`, and `-o addopts=''` on CPython 3.11.15 and 3.12.7 | PASS — 113 passed in 0.35s on 3.11; 113 passed in 0.37s on 3.12 |
| P1-Q22 | `next/` | Black check, byte compilation, full mypy, and pyright 1.1.411 across source, tests, and inventory script | PASS — Black reported 34 files unchanged; compilation succeeded; mypy found no issues in 33 source files; pyright reported 0 errors and 0 warnings |
| P1-Q23 | repository root and `next/` | Independent final boundary review; all architecture tests; sole `.execute()` scan; v1/symlink/root-oracle checks; `git diff --check`; scoped status review | PASS — no blocker, prohibited reference, symlink, root change, diff error, or path outside `next/`; 19 architecture tests; sole production executor invocation is `operations/runtime.py:719` |
| P1-Q24 | clean copy `/private/tmp/daita-v2-p1-04-final.ySk4bb` | Build sdist/wheel without isolation and inspect both archives for required runtime modules and forbidden test/nested paths | PASS — build succeeded; 18-entry wheel and 31-entry sdist contain the generic loop/operation runtime and no tests or nested `next/` path |
| P1-Q25 | repository root | Stage the 10 reviewed P1-04 paths under `next/`; run `git diff --cached --check` and all configured pre-commit hooks; create the scoped checkpoint | PASS — every staged path was under `next/`; no diff-check error; all configured hooks passed; commit `91d73764af34a3e04938b9c5cdde4a457d2af7e9` |

### P1-04 red/repair evidence

- Parallel runtime tests first landed after their production surface and were
  green; aligning the clarified minimal observation projection produced one
  temporary fingerprint-history expectation failure. The resolved contract
  treats fingerprints as a current no-progress epoch cleared by accepted
  evidence, while events retain the operation's audit history.
- The first complete regression run passed 95 tests and failed one P1-03
  defensive assertion because a refactored error message omitted the words
  "tool call." Restoring explicit boundary wording preserved the existing
  contract; the subsequent complete suite passed.
- Independent final review found that the first implementation accepted a
  caller hash containing the rejection code, left later calls in a rejected
  multi-call response without tool results, admitted unbounded readiness
  correction facts, and could terminalize no-progress without the current
  rejection commit. Runtime-owned canonical name/argument fingerprints,
  atomic taskless skip observations, bounded readiness records, and exact
  current-call event/observation grounding repaired those boundaries.
- Boundary review also made threshold semantics explicit: the identical
  failure limit counts the first failed attempt, no-progress wins when its
  threshold and the aggregate repair limit cross together, and the neutral
  `no_progress_action_failure_limit` reason remains truthful when the
  configured limit is one.
- The first amended mypy/pyright pass found two test-only payloads used as
  `FrozenJsonObject` without explicit narrowing. Adding `isinstance` checks
  repaired both diagnostics; production code had no type failure.
- The first final clean-copy invocation and one search used cwd-relative paths
  from the wrong directory. Both commands failed before changing tracked
  state and passed immediately with corrected explicit paths.
- The sandbox rejected the first scoped stage and pre-commit attempts because
  Git could not create `index.lock`. Approved reruns used the same ten
  `next/` paths and configured hooks; staging and all hooks then passed.

### Executed P1-05 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P1-Q26 | `next/` | Run expanded `LoopBudgets` model tests before adding comprehensive fields | EXPECTED RED — 10 failed and 8 passed; missing turn/action/observation/token/time/task/cost fields were identified |
| P1-Q27 | `next/` | Focused loop-model, budget acceptance, and runtime-budget tests after the first implementation | PASS — 33 passed; operation-bound limits, exact N/N+1 behavior, response usage, observation overrun, wall/model timeout, task timeout, executor-owned `TimeoutError`, budget facts, and atomic rollback are covered |
| P1-Q28 | `next/` | Focused cancellation acceptance and runtime-interruption tests | PASS — 7 passed; model/executor/post-evidence cancellation, repeated cancellation, wall-terminal-commit race, typed interruption, and atomic event failure are covered |
| P1-Q29 | `next/` | Run cancellation-suppressing provider/executor and task-persistence deadline regressions before repairing deadline ownership | EXPECTED RED — exactly 3 failures: late provider response completed, late executor result became evidence, and executor I/O ran after the wall deadline |
| P1-Q30 | `next/` | Run external caller-cancellation suppression regressions before adding explicit cancellation checks | EXPECTED RED — 2 failures because provider/executor adapters swallowed `CancelledError` and returned normally |
| P1-Q31 | `next/` | Final focused loop-model, budget, adversarial-deadline, cancellation, and runtime modules | PASS — 46 passed, including 28 deadline/budget/cancellation/runtime cases; independent re-review found no blocker |
| P1-Q32 | `next/` | Complete isolated suite with `-S` and explicit v2 `PYTHONPATH` on CPython 3.11.15 and 3.12.7 | PASS — 150 passed on each interpreter |
| P1-Q33 | repository root and `next/` | Black check, byte compilation, full mypy, and pyright 1.1.411 across source, tests, and inventory script | PASS — Black reported 39 files unchanged; compilation succeeded; mypy found no issues in 38 source files; pyright reported 0 errors and 0 warnings |
| P1-Q34 | repository root and `next/` | Independent final deadline review; architecture tests; sole `.execute()` scan; v1/symlink/root-oracle checks; `git diff --check`; scoped status review | PASS — no blocker, prohibited reference, symlink, root change, or diff error; 19 architecture tests; sole production executor invocation is `operations/runtime.py:830` |
| P1-Q35 | clean copy `/private/tmp/daita-v2-p1-05.ZIaT8S` | Build sdist/wheel without isolation and inspect both archives for required runtime modules and forbidden test/nested paths | PASS — build succeeded; 18-entry wheel and 31-entry sdist contain the generic loop/operation runtime and no tests or nested `next/` path |
| P1-Q36 | repository root | Stage the 12 reviewed P1-05 paths under `next/`; run `git diff --cached --check` and all configured pre-commit hooks; create the scoped checkpoint | PASS — every staged path was under `next/`; cached diff and all configured hooks passed; commit `5c87494af5c905b7254cd1ce67196daa9869f3f6` |

### P1-05 red/repair evidence

- The expanded budget-record run first failed ten cases on the intentionally
  absent fields. The first parallel budget acceptance run met concurrent
  production and exposed eight naming/payload expectation mismatches rather
  than semantic gaps; aligning the tests to the locked inspectable facts made
  all 33 model/budget cases pass.
- Initial raw-cancellation tests landed after their production surface and
  were already green. Follow-up races prove cancellation-resistant terminal
  commits, active-child intent, atomic rollback, and preservation of accepted
  evidence without falsely claiming an observation.
- Independent adversarial review found three root deadline defects. Both
  provider and executor adapters could suppress timeout cancellation and
  return usable late results, while task-persistence work could exhaust wall
  time before the runtime's sole executor call. Post-context expiry checks and
  a runtime-owned authoritative pre-I/O deadline recomputation now reject late
  results, omit false executor-start events, and preserve distinct wall/task
  terminal facts.
- Two additional expected-red cases showed an adapter could suppress the
  caller's raw `CancelledError`. Pending task-cancellation checks at both model
  and executor await boundaries restore cancellation precedence and durable
  interruption. The nested wall-failure handler also ensures cancellation
  during a terminal budget commit cannot strand a running operation.
- The first full mypy pass found two narrow production typing errors; explicit
  integer/duration loop variables and a narrowed final-text binding repaired
  them without suppressions. The first pyright command omitted the venv
  `--pythonpath` and reported only unresolved pytest imports; the established
  command passed with zero diagnostics. An initial collection-count path and
  archive filename guess likewise failed without changing tracked state and
  passed immediately when corrected.
- The sandbox rejected the first scoped stage and pre-commit invocations at
  Git's `index.lock`. Approved reruns used the same 12 `next/` paths; cached
  diff review and all configured hooks passed.

### P1-06 Phase 1 gate coverage

| Gate behavior | Deterministic proof |
| --- | --- |
| One loop handles sessionless/session user, schedule, monitor, and internal triggers | Parameterized `test_text_only_response_completes_from_committed_runtime_state`; reserved event rejection proves no partial state or trigger-ID reservation |
| Normal text completion uses committed state | Exact 8-event text-only transcript, committed model response/readiness, terminal operation, usage, and provider-neutral request assertions |
| Model-selected action follows task → executor → evidence → observation ordering | `test_fake_reads_follow_the_only_durable_executor_path_in_order` plus the complete repaired-action 24-event transcript and correlation assertions |
| Invalid action becomes a bounded observation and a changed action can repair | `test_invalid_action_is_observed_then_changed_action_repairs` and stable multi-call skip transcript cases |
| Repeated normalized failure stops without more model or executor I/O | `test_repeated_normalized_failure_stops_before_more_model_or_io` and exact threshold cases |
| Cancellation is inspectable at model, executor, post-evidence, repeated-cancel, and terminal-budget boundaries | All seven `test_loop_cancellation.py` trajectories, including cancellation-suppressing adapters |
| Turn/action/observation/token/cost/wall/task limits have truthful terminal facts | Eleven `test_loop_budgets.py` trajectories plus three adversarial timeout-suppression/deadline cases |
| One generic provider/domain-neutral loop and one executor invocation boundary | Six repository AST assertions in `test_phase1_loop_architecture.py`; import firewall also rejects literal dynamic v1/self imports |

### Executed P1-06 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P1-Q37 | `next/` | Focused new/changed architecture, text-only, repaired-action, runtime-commit, and execution-boundary modules with isolated v2 import path | PASS — 46 passed in 0.16s; complete event/correlation, all MVP triggers, reserved-event atomicity, dynamic import, one-loop, dependency-leaf, branchlessness, and sole-executor assertions pass |
| P1-Q38 | `next/` | Complete isolated suite with `-S`, explicit v2 `PYTHONPATH`, and disabled repository addopts on CPython 3.11.15 and 3.12.7 | PASS — final checkpoint-tree rerun: 160 passed in 0.57s on 3.11 and 160 passed in 0.52s on 3.12 |
| P1-Q39 | `next/` | Black check; byte compilation; full mypy with `MYPYPATH=src`; pyright 1.1.411 with repository venv | PASS — Black reported 40 files unchanged; compilation succeeded; mypy found no issues in 39 source files; pyright reported 0 errors/warnings |
| P1-Q40 | repository root and `next/` | Generated disposition and v1-oracle checks; isolated architecture suite; root-oracle diff from `b87df318`; symlink and `git diff --check` scans | PASS — 26 architecture tests; inventories/fixtures reproduce; no root change, v2 symlink, or diff error |
| P1-Q41 | clean copy `/private/tmp/daita-v2-p1-06.VGBQWx` | Build sdist/wheel without isolation; inspect required and forbidden archive paths; install wheel into fresh CPython 3.11/3.12 venvs and verify version/origin | PASS — 18-entry wheel and 31-entry sdist exclude tests/scripts/decisions/nested `next/`; both fresh installs import v2 `2.0.0a0` from their own `site-packages` |
| P1-Q42 | repository root | Stage the 9 reviewed P1-06 paths under `next/`; run cached diff and configured pre-commit hooks; create the scoped checkpoint | PASS — every path was under `next/`; cached diff and hooks passed; commit `eb57f9d76dc33c840410529672f5381f33b4423b` |

### P1-06 review/repair evidence

- Gate inventory found no missing production behavior and therefore added no
  second acceptance harness. Existing deterministic trajectories remain the
  behavioral owners; only their missing trigger/event/correlation assertions
  were strengthened.
- The sole executor-invocation repository scan moved from an operation unit
  module into the architecture suite, which now also proves exactly one
  `AgentLoop`, an implementation-free checkpoint-record leaf, no operation
  dependency on the driver, an explicit driver import allowlist, and no
  provider/domain identity branching.
- Section 15's operations-to-loop wording was reconciled with Sections 6–7,
  ADRs 0002/0003/0005, and the P1-01 ledger: the operation runtime consumes
  canonical checkpoint records to commit them but cannot import the executable
  driver or orchestration protocols. The source module documentation and
  architecture assertions make that distinction inspectable.
- The first focused Black check correctly rejected the newly added architecture
  module's formatting. Black reformatted that file, after which the focused
  check, complete static gate, and both interpreter suites passed.
- Independent checkpoint review caught that the first moved executor scan
  filtered by receiver name, the first identity-branch scan both missed type
  checks and overmatched generic `name` fields, and the ledger misstated the
  text-only transcript as 14 events. The executor and model-generation scans
  now conservatively cover every production call, the identity detector has
  synthetic forbidden/allowed regressions, and the corrected text count is 8.
  Its first allowed-contract regression was expected red because the detector
  still matched the inner `self._domain` expression; context-sensitive direct
  identity handling repaired that overreach before the final green reruns.

### Executed P1-07 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P1-Q43 | `next/` | Final complete isolated suite on CPython 3.11.15/3.12.7; Black; compilation; mypy; pyright 1.1.411 | PASS — 160 tests in 0.52s on 3.11 and 160 in 0.48s on 3.12; Black 40 files; mypy 39 files; pyright 0 errors/warnings |
| P1-Q44 | repository root and `next/` | Isolated architecture suite; disposition/v1-oracle reproduction; root-oracle diff; Phase 1 commit-range scope/diff scan; v2 symlink scan | PASS — 26 architecture tests; every committed Phase 1 path is under `next/`; no root change, fixture drift, symlink, or diff error |
| P1-Q45 | committed P1-06 archives under `/private/tmp/daita-v2-p1-gate.Y9lKPa` and `/private/tmp/daita-root-p1-gate.Bbhplk` | Build v2 from `HEAD:next` and root from `HEAD`; inspect archive counts, versions, and cross-inclusion | PASS — v2 wheel/sdist 18/31 at `2.0.0a0`; root wheel/sdist 401/442 at `1.0.0`; neither archive family contains the other tree |
| P1-Q46 | repository root | Stage only the final `STATUS.md` and `QUALITY_GATES.md`; run cached diff and every configured hook before the exact gate commit | PASS — exactly 2 paths, both under `next/`; cached diff, whitespace, EOF, merge-conflict, large-file, and applicable Black checks passed |

## Phases 2 through 9

Phase 2 status: **P2-02 canonical checkpoints and the authoritative in-memory
operation-store seam, P2-03 normalized SQLite repository, P2-04 blob/event
persistence, and P2-05d's cross-adapter fenced lifecycle are complete;
P2-05e's sole-runtime execution migration is active. The overall Phase 2 gate
has not been run or claimed.**

Plan Sections 6 and 15 were re-read in full after the Phase 1 gate and before
any Phase 2 production edit. Later phases remain unstarted.

### Executed P2-02 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P2-Q01a | `next/` | Isolated combined canonical event/checkpoint, in-memory operation-store contract, and Phase 2 persistence architecture tests before adding production modules | EXPECTED RED — collection stopped on exactly 3 `ModuleNotFoundError` results for the intentionally absent `daita.events` and `daita.operations.store` owners |
| P2-Q01b | `next/` | Focused runtime-store seam tests after the standalone canonical-record/store contract passed 28 tests | EXPECTED RED — 6 failures because the existing runtime did not accept `store=`; tests lock atomic create/CAS failure, conflict visibility, cross-runtime authority, and trigger uniqueness |
| P2-Q01c | `next/` | Structured typed-store-error attributes after the optimistic contract was otherwise green | EXPECTED RED — 4 failures identified absent stable operation/trigger/reason attributes; typed constructors repaired the contract and all 9 store tests then passed |
| P2-Q01d | `next/` | Standalone canonical event/checkpoint and optimistic in-memory store tests, excluding only compatibility re-exports before runtime extraction | PASS — 28 passed; this was the representative green slice reviewed before standardizing the runtime seam |
| P2-Q01e | `next/` | Focused canonical/store/runtime-commit and Phase 1/2 architecture integration after removing both runtime state maps | PASS — 53 passed; no split authority, hidden CAS retry, partial failed commit, duplicate trigger claim, duplicate record owner, SQL leakage, or executor-boundary drift |
| P2-Q01f | `next/` | Complete isolated suite on CPython 3.11.15 and 3.12.7 | PASS — 204 passed in 1.00s on 3.11 and 204 passed in 1.07s on 3.12 |
| P2-Q01g | repository root and `next/` | Black, byte compilation, mypy, pyright 1.1.411, all architecture tests, v1 fixture/disposition reproduction, v1-import/symlink/root-oracle scans | PASS — 49 files Black-clean; compile succeeded; mypy 48 files; pyright 0 errors/warnings; 34 architecture tests; every isolation/oracle scan clean |
| P2-Q01h | clean copy `/private/tmp/daita-v2-p2-02.zm30hd` | Build sdist/wheel; inspect archive content; install wheel without dependencies into fresh Python 3.11/3.12 environments; import canonical event/store modules | PASS — wheel/sdist 22/36 entries at `2.0.0a0`; no tests or nested `next/`; both fresh imports resolved to their own site-packages |
| P2-Q01i | `next/` | Adversarial snapshot-linkage and committed-lifecycle-history contracts before repair | EXPECTED RED — 6 failed and 17 passed; same-turn pointers, response-owned call IDs, symmetric task/evidence linkage, and immutable non-event history were not fully enforced |
| P2-Q01j | `next/` | Focused canonical checkpoint and optimistic-store contracts after structural repair | PASS — 23 passed; invalid linkage and prior-record rewrite/removal are rejected without publishing candidate state |
| P2-Q01k | `next/` | Cancellation after durable create and interruption lost-CAS convergence before repair | EXPECTED RED — 2 failed; one running orphan and one surfaced revision conflict reproduced the reviewer findings |
| P2-Q01l | `next/` | Focused cancellation-store contracts after central atomic-write/interruption repair on CPython 3.11.15/3.12.7 | PASS — 2 passed on each interpreter; durable create cannot remain running and interruption preserves a concurrently committed event while converging |
| P2-Q01m | `next/` | Downstream canonical model-call correlation and terminal-race cleanup contracts before repair | EXPECTED RED — 9 acceptance trajectories and 1 terminal-race case failed; canonical fields were absent and the driver's post-shield handler was unreachable |
| P2-Q01n | `next/` | Canonical aggregate ownership and event-ancestry regressions before repair | EXPECTED RED — focused cases accepted an erased event call correlation, an unowned model call, and child IDs without one explicit consistent turn/model/task ancestry |
| P2-Q01o | `next/` | Cancellation delivered before a blocked ordinary transition later lost CAS | EXPECTED RED — 1 integration failure surfaced `OperationStateError`, skipped loop interruption cleanup, and left the externally advanced operation running |
| P2-Q01p | `next/` | Final complete isolated suite on CPython 3.11.15 and 3.12.7 | PASS — 218 passed in 1.24s on 3.11 and 218 passed in 1.26s on 3.12 |
| P2-Q01q | repository root and `next/` | Final Black, compilation, mypy, pyright 1.1.411, architecture, disposition/v1-oracle, root-oracle, import, symlink, SQL-leakage, and diff gates | PASS — Black 50 files; compile clean; mypy 49 files; pyright 0 errors/warnings; 34 architecture tests; every isolation/oracle scan clean |
| P2-Q01r | clean copy `/private/tmp/daita-v2-p2-02-gate.31k4au` | Rebuild final sdist/wheel; inspect content; fresh isolated Python 3.11/3.12 installs and canonical-module imports | PASS — wheel/sdist 22/36 entries at `2.0.0a0`; no tests or nested `next/`; both imports resolved to their own site-packages |
| P2-Q01s | current P2-02 diff | Two independent final adversarial reviews after all repairs | PASS — no remaining architecture/cancellation blocker; exact ancestry, committed history, authoritative store state, cancellation precedence, deliberate interruption retry, and sole executor boundary verified |
| P2-Q01t | repository root | Stage only the 19 reviewed P2-02 paths under `next/`; cached diff; configured hooks; local checkpoint commit | PASS — every staged path was under `next/`; diff and all hooks passed; commit `b13e66abc5d645b685f7bbf840d2e8d9ea903f2f` |

Rows P2-Q01f through P2-Q01h predate the adversarial repairs and remain useful
interim evidence only. Rows P2-Q01p through P2-Q01s close the refreshed seam
gate.

### Executed P2-03 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P2-Q02a | `next/` | Run the test-only SQLite foundation contract before adding a production storage owner | EXPECTED RED — collection stopped on exactly one missing `daita.storage` module |
| P2-Q02b | `next/` | First fixed-schema SQLite foundation on CPython 3.11/3.12 | PASS — 12 tests per interpreter; v2 marker, WAL, foreign keys, busy timeout, FULL synchronization, migrations, backup, rollback, compatibility, and reopen passed |
| P2-Q02c | `next/` | Independent adversarial foundation/cancellation cases before repair | EXPECTED RED — 6 failures exposed transient backup acceptance, pre-lock cancellation escape, failed-fresh-init residue, transaction-control injection, raw-constructor bypass, and unusable existing-backup recovery |
| P2-Q02d | `next/` | Repaired foundation and cancellation suite | PASS — 23 tests on each interpreter; offloaded migration/open/inspect/close work reaches a definitive state before cancellation propagates |
| P2-Q02e | `next/` | Maximal normalized aggregate before repository reads/writes existed | EXPECTED RED — exactly 2 failures for absent `create`, `load`, and `load_by_trigger` methods |
| P2-Q02f | `next/` | Maximal normalized lifecycle and strict codec contracts | PASS — all current trigger/operation/loop/turn/model/readiness/task/evidence/observation/event fields round-trip independently across 11 tables, explicit positions, UTC time, exact Decimal, and versioned/tagged provider-neutral JSON; no opaque snapshot row exists |
| P2-Q02g | `next/` | Schema/codec adversarial review before repair | EXPECTED RED — 2 failures showed BLOB-to-text coercion and silent positional-gap reordering; strict storage-class and contiguous-position checks repaired both |
| P2-Q02h | `next/` | Cancellation and post-COMMIT acknowledgement cases before reconciliation | EXPECTED RED — 2 of 4 failed because a real durable commit followed by an injected SQLite error escaped as raw `OperationalError` |
| P2-Q02i | `next/` | Repaired write-cancellation and transaction contract | PASS — 11 cases; create/commit cannot outlive their public await, ambiguous commit acknowledgement is reconciled or typed unknown, legal mutable transitions append exact suffixes, stale/missing writes are typed, two connections have one CAS winner, duplicate claims serialize, and an event abort rolls back all 11 tables |
| P2-Q02j | `next/` | Complete focused persistence suite after transaction repair | PASS — 50 storage tests covering foundation, migration, schema, codecs, normalized repository, cancellation, reconciliation, CAS, rollback, and reopen |
| P2-Q02k | `next/` | Complete isolated suite on CPython 3.11.15 and 3.12.7 before final audit assertions | PASS — 268 passed in 1.30s on 3.11 and 268 passed in 1.40s on 3.12 |
| P2-Q02l | repository root and `next/` | Black, compilation, mypy, pyright 1.1.411, architecture, disposition/v1 oracle, root-oracle, and root safe-suite checks | PASS — Black 58 files; compilation clean; mypy 57 files; pyright 0 errors/warnings; 34 architecture tests; fixture/inventory/root scans clean; root suite 2,498 passed and 221 deselected |
| P2-Q02m | clean copies under `/private/tmp` | Build and inspect v2 plus root distributions; fresh isolated Python 3.11/3.12 v2 imports | PASS — v2 wheel/sdist 24/39 entries at `2.0.0a0` include `daita.storage.sqlite`; root wheel/sdist 401/442 at `1.0.0` exclude the physical `next/` tree; both fresh v2 imports resolve to their own site-packages |
| P2-Q02n | current P2-03 diff | Independent transaction-design and final architecture/scope reviews | EXPECTED RED then PASS — final review found one incorrect later-migration diagnostic; a dedicated migration-3 regression failed with migration 1, then passed after tracking the active migration. No commit/CAS/schema/codec blocker remains; two requested architecture assertions permanently lock the adapter import allowlist and forbid opaque/delete/replace/ignore/upsert SQL shortcuts |
| P2-Q02o | `next/` | Final post-review complete suites and static/architecture gate | PASS — 271 passed in 2.25s on CPython 3.11.15 and 271 passed in 2.26s on CPython 3.12.7; Black 58 files; compilation clean; mypy 57 files; pyright 0 errors/warnings; all 36 architecture tests pass |
| P2-Q02p | repository root | Stage exactly 12 reviewed P2-03 paths; cached diff; configured hooks; local checkpoint | PASS — every path was under `next/`; cached diff and whitespace/EOF/merge-conflict/large-file/Black hooks passed; commit `ee6763bb6e0b55b1f95ec4d9fcc1f40505fbe4d5` |

Rows P2-Q02k through P2-Q02m remain useful pre-review evidence; P2-Q02o is the
refreshed code/test-tree gate and P2-Q02p closes P2-03.

### Executed P2-04 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P2-Q03a | `next/` | Isolated canonical blob contract and architecture tests before a blob owner existed | EXPECTED RED — collection stopped on the absent `daita.storage.blobs`; the architecture run had exactly 2 missing-owner failures |
| P2-Q03b | `next/` | Local content-addressed adapter durability/cancellation contracts before `LocalBlobStore` existed | EXPECTED RED — both adapter suites stopped at collection on the intentionally absent concrete owner |
| P2-Q03c | `next/` | Focused blob contract, durability, cancellation, corruption, race, cleanup, and architecture suites after independent repair | PASS — 71 focused cases prove parent-directory fsync, partial-write handling, visible-result retry stabilization, strict recursive codecs, cancellation-poisoned readers, content sharing, lifecycle CAS, corruption/link rejection, bounded verification, grace cleanup, and no ownership leak |
| P2-Q03d | `next/` | `PYTHONPATH=src:... python -S -m pytest tests/ -q` on CPython 3.11.15/3.12.7 plus full Black/compile/mypy/pyright and architecture checks at the blob checkpoint | PASS — 344 tests per interpreter; Black/compile clean; mypy clean; pyright 0 errors/warnings; all 38 architecture tests pass; independent review found no remaining blob blocker; commit `320a9e0` |
| P2-Q03e | `next/` | Canonical committed-event record/protocol tests before the records existed | EXPECTED RED — collection stopped on exactly 1 import error for absent `CommittedEvent` and `EventCursor` |
| P2-Q03f | `next/` | Migration 3 and SQLite replay contracts before the projection/reader existed | EXPECTED RED — exactly 7 failures identified the absent per-agent sequence, migration history, and `read_after` implementation |
| P2-Q03g | `next/` | Focused event record, Migration 3, bounded replay, codec-corruption, and architecture tests after independent review | PASS — 42 cases prove rowid-order backfill, atomic migration rollback, positive unique append-only sequences, state/event/cursor atomicity, rollback/CAS gaplessness, pagination, typed cursor errors, reopen, and strict corruption normalization; independent review found no blocker |
| P2-Q03h | `next/` | `PYTHONPATH=src:... python -S -m pytest tests/contract/storage/test_sqlite_event_subscription.py -q` before subscription production code | EXPECTED RED — exactly 9 failures identified absent `subscribe`, polling/batch constants, and local wake seam |
| P2-Q03i | `next/` | Focused 10-case subscription suite plus combined event/storage/architecture tests, Black, mypy, pyright, and independent lifecycle review | PASS — 10 subscription and 52 combined cases; durable post-commit notification, rollback invisibility, failed/missed/cross-store wake recovery, double-read race closure, exact reconnect, multiple/slow subscribers, bounded paging, cleanup, and agent isolation pass; static checks are clean and review found no remaining blocker |
| P2-Q03j | `next/` | `PYTHONPATH=src:../.venv/lib/python3.11/site-packages ../.venv/bin/python -S -m pytest tests/ -q -p no:cacheprovider --junitxml=/private/tmp/daita-v2-p2-04-py311-final.xml`; equivalent CPython 3.12 command with the miniforge site-packages path | PASS — 370 tests, 0 failures/errors in 1.992s on CPython 3.11.15; 370 tests, 0 failures/errors in 2.056s on CPython 3.12.7 |
| P2-Q03k | `next/` | `black --check src tests scripts`; `compileall -q src tests scripts`; full mypy; pyright 1.1.411; isolated `pytest tests/architecture -q -p no:cacheprovider` | PASS — Black 67 files; compilation clean; mypy 66 files; pyright 0 errors/warnings; 38 architecture tests |
| P2-Q03l | repository root and `next/` | Root safe suite with `-m "not requires_llm and not requires_db"`; disposition/v1-oracle checks; root diff from `b87df318`; symlink and `git diff --check` scans | PASS — root collected 2,719 and passed all 2,498 selected with 221 deselected in 10.38s; inventories/fixtures reproduce; no root change, v2 symlink, or diff error |
| P2-Q03m | clean copy `/private/tmp/daita-v2-p2-04-final.ckXW37` | Build v2 and root distributions without isolation; inspect cross-inclusion; install the v2 wheel without dependencies into fresh CPython 3.11/3.12 environments; isolated imports of blob/event/SQLite owners | PASS — v2 wheel/sdist contain 26/41 entries at `2.0.0a0`; root wheel/sdist contain 401/442 entries at `1.0.0`; no tests/nested `next/` crossed archives; both fresh imports resolve to their own site-packages |
| P2-Q03n | repository root | Stage exactly the 11 reviewed P2-04 event/checkpoint paths; run cached diff and `.venv/bin/pre-commit run`; create the authorized local checkpoint | PASS — every path was under `next/`; diff and whitespace/EOF/conflict/large-file/Black hooks passed; commit `b04fcb11e2e6bbf38341648b9b171341ae1996e3` |
| P2-Q04a | plan, current v2, and root v1 oracle (read-only) | Inventory task models/checkpoints/store/SQLite/runtime/capability facts; inspect v1 kernel/store/governance/worker tests; run two independent v2 lease-boundary design audits | PASS — the existing operation runtime/store/SQLite transaction remain the only owners; exact records, narrow repository operations, fail-closed recovery rules, highest-value tests, and later-phase deferrals are locked before production edits |
| P2-Q04b | `next/` | Run the portable task/safety/lease/store/checkpoint contracts before their production records exist | EXPECTED RED — the model slice reported 56 failures with no collection error; the combined run stopped on exactly 2 missing-lease-module collection errors |
| P2-Q04c | `next/` | Focused P2-05b contract; independent bypass/chronology/correlation audit; complete isolated v2 suite; Black, compile, mypy, pyright, and architecture checks | PASS after repair — 91 focused and 463 complete tests; Black 71 files; compilation clean; mypy 70 files; pyright 0 errors/warnings; generic commit cannot advance active leases or accept their evidence, lease attempts cannot overlap, dependencies freeze at readiness, and immutable safety/dependency/lease/fence contracts remain backend-neutral |
| P2-Q04d | `next/` | Migration 4 plus exact task/dependency/lease projection, corruption, migration rollback, and reopen contracts | PASS after repair — 5 migration, 34 projection, and 7 ownership cases pass; all 509 v2 tests pass on CPython 3.11.15/3.12.7; all 45 architecture tests and static gates are clean; checkpoint `94584a5` |
| P2-Q04e | `next/` | Representative authoritative-clock lifecycle tests before concrete adapter methods, plus ambiguous-commit event-prefix reconciliation before its repair | EXPECTED RED — lifecycle behavior was absent from both adapters, and the focused reconciliation regression proved a later event prefix was not exact commit acknowledgement |
| P2-Q04f | `next/` | Final P2-05d dual-interpreter suite, architecture/static/preservation gates, and two independent release audits | PASS — 549 tests with 0 failures/errors in 4.333s/4.338s on CPython 3.11.15/3.12.7; 50 architecture tests; Black 76 files; compilation clean; mypy 23 production files; pyright 0 errors/warnings; diff/root scopes clean; both audits GO |

P2-Q03j through P2-Q03n close the refreshed P2-04 code/test-tree, preservation,
distribution, review, and checkpoint gate.

### Planned Phase 2 evidence sequence

These rows are prospective gates, not passing claims. Each is updated only
after its command and evidence actually exist.

| ID | Scope | Required evidence before PASS |
| --- | --- | --- |
| P2-Q01 | Representative persistence seam | PASS — test-first canonical/store seam, optimistic conflict/rollback/history/cancellation proofs, 218 cross-version tests, and refreshed static/architecture/isolation/build reviews |
| P2-Q02 | SQLite and migrations | PASS — marker/PRAGMA/migration/backup/compatibility gates, normalized lifecycle round-trips, strict corruption rejection, optimistic CAS/rollback, cancellation, reconciliation, dual-Python/static/build proof, and checkpoint `ee6763b` |
| P2-Q03 | Blobs and events | Durable put-by-content; hash/rename/orphan behavior; state/event same transaction; post-commit subscription, cursor replay, and commit/publish crash-gap coverage |
| P2-Q04 | Tasks, leases, and recovery | Claim races; fencing; expiry; replay-safe reclaim; terminal skip; manual recovery for unknown side effects; all seven crash/cancel checkpoints |
| P2-Q05 | Governance and fake side effect | Risk facts and decision-only approval mutation; no executor before approval or after denial; same-operation wake/resume; repeated resume changes the marker once |
| P2-Q06 | Agent, sessions, and embedded mode | Isolated create/open identity; authoritative DB/manifest match; shared writer lock; restart-safe transcripts; transient/sessionless isolation; thin-facade architecture |
| P2-Q07 | OpenAI adapter and live loop | Lazy optional import; fake Responses client contracts; provider call-ID continuation; normalized errors; provider cannot execute; explicit live model completes persisted fake loop |
| P2-Q08 | Phase gate | Full Python 3.11/3.12 suite; static/architecture/import/root-oracle/build scans; parity/ADR/ledger review; scoped hooks and exact Phase 2 gate commit |

The production-provider row remains NOT RUN until an actual API key and an
explicit test model are available. A skip or mock trajectory cannot close the
Phase 2 gate.

## Live and external gates

- Phase 0 requires no live provider or external database.
- Production-model, PostgreSQL, and other live checks are **NOT RUN** and will
  be recorded in their owning later phases.
- No credential presence is inferred from local environment files, and no
  mock result will satisfy a live gate.
