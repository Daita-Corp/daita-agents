# Daita v2 quality-gate evidence

This is the authoritative record of commands actually run for the isolated
replacement. A mock result is never recorded as a live result. `PASS` means
the exact command shown exited zero in the stated environment; `NOT RUN` is
not a passing gate.

## Governing baseline

- Branch: `next`
- Root v1 oracle commit: `b87df31873d33fffbf50498f5dc4d8892115e8f8`
- Architecture plan SHA-256:
  `e54f43dd0bfc0fa8478b496e7d2a89e53439d7fe9f5c8cf58f5c947f7682364b`
- Phase scope: 0 through 9.5 complete; the candidate is eligible for human
  Phase 10 review, but Phase 10 is excluded and has not started.
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

Phase 2 status: **P2-02 through P2-09 are complete. P2-10's consolidated gate
and P2-11's evidence/checkpoint close Phase 2. Phase 3 has not started.**

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
| P2-Q04g | `next/` | P2-05e expected-red post-claim wall deadline, lease-bound timeout, and unknown side-effect timeout regressions | EXPECTED RED — exactly 3 failures proved executor I/O could begin after the wall deadline, outlive its configured lease, and freeze an unknown side-effect outcome as terminal failure |
| P2-Q04h | `next/` | Independent adversarial clock/fence/recovery review and repaired focused regressions | PASS after repair — runtime/store clock skew, delayed fenced-start acknowledgement, system-uncertainty/caller-cancellation conflation, and stale unknown-outcome annotation were reproduced and closed; the final live-fenced RUNNING annotation is no-evidence/no-lease-change, expired/stale rejection is exact zero delta, and replay-safe stable-key versus unsafe manual-recovery classification remains viable |
| P2-Q04i | `next/` and repository root | Final P2-05e dual-interpreter suite, architecture/static/preservation gates, and three independent release audits | PASS — 572 tests with 0 failures/errors in 4.402s/4.515s on CPython 3.11.15/3.12.7; 51 architecture tests; Black 76 files; compilation clean; mypy 23 production files; pyright 0 errors/warnings; diff/root scopes clean; all audits GO |
| P2-Q04j | `next/` | P2-05f runtime recovery, blob-backed evidence, and Migration 5/projection contracts before production support | EXPECTED RED — 8 recovery failures were confined to absent `resume_task`; 5 blob-runtime failures were confined to absent artifact puts; 7 migration/projection failures were confined to absent `Evidence.blob_id`, schema, and codecs |
| P2-Q04k | `next/` | Repaired focused recovery/blob/SQLite cases plus pre-audit complete suite, architecture, and static checks | PASS — 603 complete and 52 architecture tests; Black 77 files; compilation clean; mypy 23 production files; pyright 0 errors/warnings; stable-key reclaim, manual recovery, terminal skip, exact blob-before-evidence ordering, orphan-on-gap behavior, and nullable Migration 5 projection pass |
| P2-Q04l | current P2-05f diff | Three independent P2-05g representative read, side-effect, and blob/persistence audits | PASS — all audits GO; the store remains the sole expiry classifier, the runtime remains the sole executor/recovery/evidence boundary, stale fences accept nothing, SQLite rollback/reopen is exact, and broader startup/approval behavior remains deferred to P2-06/P2-07 |
| P2-Q04m | `next/` | Final adversarial blob cancellation-suppression regression before repair | EXPECTED RED — exactly 1 failure proved a blob adapter could catch caller cancellation, return valid metadata, and allow evidence acceptance |
| P2-Q04n | `next/` | Repaired final dual-interpreter, architecture, and static gate | PASS — 604 tests with 0 failures/errors in 4.659s/4.762s on CPython 3.11.15/3.12.7; 52 architecture tests; Black 79 files; compilation clean; mypy 23 production files; pyright 0 errors/warnings |
| P2-Q04o | repository root, `next/`, and clean copies under `/private/tmp` | Root preservation, v1-oracle/disposition, symlink/diff, v2/root distribution, and fresh-install gates | PASS — root 2,498 passed and 221 deselected in 10.40s; root and fixture scopes clean; v2 wheel/sdist 27/42 at `2.0.0a0`; root wheel/sdist 401/442 at `1.0.0`; fresh CPython 3.11/3.12 v2 imports resolve to their own site-packages |
| P2-Q04p | final P2-05 diff | Independent runtime, test, and scope/distribution release audits after the cancellation-precedence repair | PASS — all audits GO; pending cancellation is reasserted before evidence acceptance, and no persistence, runtime-owner, executor-boundary, package-scope, or distribution blocker remains |
| P2-Q04q | repository root | Stage only final P2-05 ledgers; cached diff; configured hooks; local checkpoint | PASS — exactly 2 paths, both under `next/`; diff and whitespace/EOF/conflict/large-file/applicable Black hooks pass; the containing checkpoint closes P2-05 |
| P2-Q05a | plan Sections 6/8.5–8.7/15, ADRs, current v2, and root v1 oracle (read-only) | Inventory whole-loop restart ownership, exact persisted checkpoints, necessarily repeated versus avoidable I/O, startup query seam, seven crash boundaries, and later-phase deferrals; run three independent audits | PASS — `AgentLoop`, `OperationRuntime`, `OperationStore`, and `OperationSnapshot` remain the only owners; the first representative SQLite-reopen expected-red trajectory and the ordered P2-06 sequence are locked before production edits |
| P2-Q05b | `next/` | Run one real-SQLite reopen trajectory with a completed tool response and no materialized task before adding loop continuation | EXPECTED RED — exactly 1 failure at the intentionally absent `AgentLoop.resume`; no collection, migration, reopen, or fixture failure |
| P2-Q05c | `next/` | Implement and independently review only the representative same-operation continuation before broadening the dispatcher | PASS — reopening SQLite and calling `resume(operation_id)` consumes the committed tool response, materializes/executes/projects it once, and performs only the necessary follow-up model call; 16 focused cases pass on Python 3.11/3.12, all 605 v2 cases and 52 architecture cases pass on Python 3.11, static checks are clean, and both reviews report GO |
| P2-Q05d | `next/` | Add focused SQLite-reopen contracts for every checkpoint state owned by the P2-06d dispatcher before broad production changes | EXPECTED RED — 1 representative case passes and exactly 6 cases fail only on missing requestless-turn reuse, exact STARTED-call resend, pending-task resume, accepted-evidence projection, live-lease deferral, and readiness/terminal/exact-trigger redelivery |
| P2-Q05e | `next/` | Complete and independently audit the checkpoint dispatcher, including ordered mixed calls and post-commit budget decisions | PASS after repair — 16 restart cases and 621 complete cases on each interpreter pass; the durable-frontier regression attributes aggregate observation failure to the latest durable call/task, A/B/C recovery preserves order and identity, unsafe expiry outranks wall failure, live and terminal redelivery are exact zero-delta, static/architecture checks are clean, and all three audits report GO |
| P2-Q05f | `next/` | Add portable agent-scoped nonterminal-query, runtime-projection, and ordered startup-recovery contracts before production support | EXPECTED RED — nine focused behavioral/architecture cases fail only on absent `load_nonterminal`, `inspect_nonterminal`, and `recover_startup`; both stores must return exact ordered versions, SQLite must reconstruct one transactionally consistent snapshot across reopen/instances, and startup must continue past a live-lease WAITING exit through the same public resume path |
| P2-Q05g | `next/` | Complete and independently audit agent-scoped startup recovery over the existing store/runtime/loop owners | PASS after repair — 631 complete tests on each interpreter and 53 architecture tests pass; ordinary ties and fold-sensitive timestamps order identically across adapters, unknown TEXT/BLOB statuses fail closed with typed rollback, a mid-decode writer cannot tear SQLite revisions, a live lease is exact zero-delta while later work continues, and all three reviews report GO with no P2-07/P2-08 leakage |
| P2-Q05h | `next/` | Inject abrupt process loss across every Phase 2-06 durable boundary and prove exact same-operation recovery without adding another owner | PASS ON ARRIVAL after assertion hardening — the initial 9-case and expanded 11-case selections passed unchanged production; the final 13 new cases prove turn/request and unknown-model outcomes, response/task/claim/executor frontiers, safe versus unsafe expiry, blob-before-evidence orphaning, durable evidence/observation context, readiness, and terminal delivery; 31 restart cases, 644 complete tests on each interpreter, 53 architecture tests, Black 81 files, compilation, focused mypy, scoped pyright, and three independent GO reviews pass with no production or later-phase change |
| P2-Q05i | checkpoint `110596b`, repository root, and clean copies under `/private/tmp` | Final P2-06 dual-interpreter, static, architecture, root-oracle, distribution, and fresh-install gate | PASS — 644 v2 tests in 6.700s/7.178s on CPython 3.11.15/3.12.7; 53 architecture tests; Black 81 files; compilation, focused mypy, and scoped pyright clean; 2,498 root safe tests with 221 deselected; four v1 fixtures reproduce; root diff/symlinks clean; v2/root wheel/sdist counts remain 27/42 and 401/442; fresh Python 3.11/3.12 installs import v2 `2.0.0a0` loop/runtime/SQLite/blob owners from their own site-packages |
| P2-Q05j | final P2-06 ledger diff | Independent recovery/scope reviews, configured hooks, and local checkpoint | PASS — both closing reviews report GO; exactly `next/STATUS.md` and `next/QUALITY_GATES.md` comprise the final diff; whitespace, EOF, conflict, large-file, and applicable Black hooks pass; the containing local checkpoint closes P2-06 and advances only the ledger to P2-07 |
| P2-Q05k | `next/` P2-07 approval/governance vertical slice | Focused governance, operation/store, SQLite migration/projection, acceptance, and sole-executor architecture tests plus one targeted safety review | PASS after repair — 573 affected-scope tests pass in 5.36s; the review-found pre-governance crash bypass is closed by an atomic PENDING-to-READY governance commit, PENDING recovery re-enters governance, and a SQLite crash/reopen test proves zero pre-approval I/O; post-fix review GO |
| P2-Q06a | `next/` P2-08 embedded lifecycle slice | Public Agent acceptance, affected SQLite migration/store, and all architecture tests plus one targeted durability review | PASS after repair — 132 focused cases; cancellation-safe writer admission/bootstrap, v1/symlink/alias rejection, authoritative identity/session linkage, monotonic restart-safe transcripts, and the thin facade pass; review-found lock/bootstrap/session gaps are closed |
| P2-Q07a | `next/` P2-09 provider-neutral/OpenAI slice | Canonical provider-call identity, strict legacy/current SQLite codecs, normalized error persistence, fake Responses client, optional-import, and provider-ownership tests | PASS after repair — the final 44-case provider/model/SQLite/architecture selection passes; review-required encrypted reasoning-item persistence/replay and actual `generate()` missing-extra behavior are covered, and the generic-loop allowlist remains provider-neutral |
| P2-Q07b | `next/` | `set -a; source ../.env; set +a; DAITA_OPENAI_MODEL=gpt-4.1-mini ../.venv/bin/python -m pytest tests/acceptance/test_openai_live_persisted_loop.py -m requires_llm -q -s` | PASS — 1 live case, reconfirmed after encrypted reasoning replay was added; the actual Responses API completes the public SQLite-backed fake-read loop and close/reopen preserves canonical/provider call identity, model calls, task, evidence, observation, events, and terminal answer; no mock substituted for this row |
| P2-Q08a | `next/` | `PYTHONPATH=src:/Users/jendala/daita/daita-agents/.venv/lib/python3.11/site-packages /Users/jendala/daita/daita-agents/.venv/bin/python -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q -p no:cacheprovider --junitxml=/private/tmp/daita-v2-p2-gate-final-py311.xml`; `PYTHONPATH=src:/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/site-packages python3.12 -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q -p no:cacheprovider --junitxml=/private/tmp/daita-v2-p2-gate-final-py312.xml` | PASS — 727 selected and 1 deselected in 7.58s/8.33s on CPython 3.11.15/3.12.7; the final confirmation includes the repaired static-boundary regression |
| P2-Q08b | `next/` | `/Users/jendala/daita/daita-agents/.venv/bin/black --check src tests scripts`; `/Users/jendala/daita/daita-agents/.venv/bin/python -m compileall -q src tests scripts`; `MYPYPATH=src /Users/jendala/daita/daita-agents/.venv/bin/python -m mypy src/daita tests scripts/build_test_disposition.py`; `npx --yes pyright@1.1.411 --pythonpath /Users/jendala/daita/daita-agents/.venv/bin/python` | PASS after contract repair — the first consolidated mypy/pyright runs exposed 70/65 genuine diagnostics, including malformed provider status/schema/metadata handling and test protocol narrowing; the focused repair passed 176 affected tests, focused mypy/pyright, then final Black (98 files), compilation, mypy (97 files), and pyright (0 errors/warnings) |
| P2-Q08c | repository root and `next/` | `PYTHONPATH=src:/Users/jendala/daita/daita-agents/.venv/lib/python3.11/site-packages /Users/jendala/daita/daita-agents/.venv/bin/python -S -m pytest -o addopts='' tests/architecture -q -p no:cacheprovider --junitxml=/private/tmp/daita-v2-p2-gate-final-architecture.xml`; `/Users/jendala/daita/daita-agents/.venv/bin/python next/scripts/build_test_disposition.py --check`; `git diff --quiet b87df31873d33fffbf50498f5dc4d8892115e8f8 -- daita tests pyproject.toml`; `git diff --name-only b87df31873d33fffbf50498f5dc4d8892115e8f8`; `find next -type l -print`; ADR-status/plan-parity review; `git diff --check` | PASS — 59 architecture tests in 1.96s; root `daita/`, root tests, and root packaging are unchanged from `b87df318`; every Phase 2 path is under `next/`; v2 has no root import or symlink; all 14 numbered ADRs are Accepted; plan fingerprint is `403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935` |
| P2-Q08d | repository root | `/Users/jendala/daita/daita-agents/.venv/bin/python -m pytest tests/ -m 'not requires_llm and not requires_db' -q -p no:cacheprovider`; `PYTHONPATH=.:/Users/jendala/daita/daita-agents/.venv/lib/python3.11/site-packages /Users/jendala/daita/daita-agents/.venv/bin/python -S next/scripts/capture_v1_oracles.py --check` | PASS — root collected 2,719 and passed all 2,498 selected with 221 deselected; all established v1 fixtures reproduce from the delegated tree |
| P2-Q08e | clean-copy `project/` directories under `/private/tmp/daita-v2-p2-gate.lDr8Gf` and `/private/tmp/daita-root-p2-gate.eNvGUj` | `/Users/jendala/daita/daita-agents/.venv/bin/python -m build --no-isolation` in each project; archive version/content/cross-inclusion scan; fresh `python -m venv`, `python -m pip install --no-deps .../daita_agents-2.0.0a0-py3-none-any.whl`, and import smoke with CPython 3.11/3.12 | PASS — v2 wheel/sdist contain 35/51 entries at `2.0.0a0`; root wheel/sdist contain 401/442 at `1.0.0`; neither distribution crosses package trees; both v2 installs import from their own site-packages, including the OpenAI adapter without the optional SDK |
| P2-Q08f | final Phase 2 diff | Review plan parity, ledgers, ADRs, P2-Q07b, root-oracle dispositions, scope, cached diff, and configured hooks; create exactly `chore(v2-phase-2): complete phase 2 gate` | PASS — P2-Q07b's actual `gpt-4.1-mini` result is internally consistent and was preserved without a second paid run; exactly 13 changed paths are under `next/`; hooks pass; the containing detached-worktree commit is the sole P2-10/P2-11 checkpoint |

P2-Q08 has no root-oracle failure. An initial v1-fixture precheck inherited the
main checkout's editable `daita` and therefore reported that checkout's
`daita/__init__.py` instead of the delegated worktree; the isolated P2-Q08d
command corrected module origin and passed without changing an oracle. The
root distribution emitted only its pre-existing setuptools license/classifier
deprecation warnings. Pip disabled its unwritable user cache, and npm reported
its nonfatal `unsafe-perm` configuration warning; all commands still exited 0.

P2-Q03j through P2-Q03n close the refreshed P2-04 code/test-tree, preservation,
distribution, review, and checkpoint gate.

### Phase 2 evidence accounting

Every row is backed by the executed evidence above.

| ID | Scope | Required evidence before PASS |
| --- | --- | --- |
| P2-Q01 | Representative persistence seam | PASS — test-first canonical/store seam, optimistic conflict/rollback/history/cancellation proofs, 218 cross-version tests, and refreshed static/architecture/isolation/build reviews |
| P2-Q02 | SQLite and migrations | PASS — marker/PRAGMA/migration/backup/compatibility gates, normalized lifecycle round-trips, strict corruption rejection, optimistic CAS/rollback, cancellation, reconciliation, dual-Python/static/build proof, and checkpoint `ee6763b` |
| P2-Q03 | Blobs and events | PASS — P2-Q03a through P2-Q03n prove durable content-addressed blobs, orphan/corruption/cancellation behavior, transactional event state, monotonic cursors, reconnect/subscription, commit-gap replay, cross-version/static/root/distribution gates, and checkpoint scope |
| P2-Q04 | Tasks, leases, and recovery | PASS — P2-Q04a through P2-Q04q and P2-Q05h prove claim races, immutable fences, expiry, replay-safe reclaim, terminal skip, unsafe manual recovery, blob linkage, all seven crash boundaries, dual-interpreter/static/root/distribution gates, and reviewed checkpoint scope |
| P2-Q05 | Governance and fake side effect | PASS — P2-Q05k proves persisted risk/fingerprint governance, decision-only approval mutation, zero executor I/O before approval/after denial, same-operation resume, exact-once fake side effect, crash/reopen safety, and post-repair review GO |
| P2-Q06 | Agent, sessions, and embedded mode | PASS — 132 focused public-agent, affected storage, and architecture cases cover isolated create/open identity, cancellation-safe shared writer admission/bootstrap, no-alias/no-follow paths, authoritative DB/manifest/session linkage, monotonic restart-safe transcripts, sessionless/session isolation, and the thin-facade boundary |
| P2-Q07 | OpenAI adapter and live loop | PASS — lazy optional import, fake Responses client contracts, provider call-ID and encrypted-reasoning continuation, normalized errors, execution-ownership scan, and explicit live `gpt-4.1-mini` persisted fake loop all pass |
| P2-Q08 | Phase gate | PASS — P2-Q08a through P2-Q08f complete the Python 3.11/3.12, static, architecture/import, root-oracle, distribution/fresh-install, parity/ADR/ledger, hook, scope, and exact-commit gate |

The production-provider row is satisfied by P2-Q07b; its live evidence was
audited for internal consistency and not rerun at additional paid cost.

## Phase 3 — data vertical slice and catalog

Status: **PASS. Phase 3 is complete and Phase 4 may begin.**

The Phase 3 gate will be recorded through a lean evidence sequence:

| ID | Scope | Required evidence before PASS |
| --- | --- | --- |
| P3-Q01 | Scope and contract inventory | SQLite-only task/test map; deferred cloud/additional-database dispositions reconciled; catalog/data/adapter/runtime owners and expected-red slices locked |
| P3-Q02 | Catalog vertical | Resource/facet/relationship/revision/sync contracts; atomic SQLite catalog persistence; FTS search; bounded traversal; reopen/rollback/corruption proof |
| P3-Q03 | SQLite source vertical | Persisted source lifecycle; discovery/inspection; declared query capability/executor; source-scope and sole-executor architecture proof |
| P3-Q04 | SQL and data-domain vertical | Pre-I/O SQL/scope/parameter validation; bounded accepted evidence and projections; untrusted context; repair and evidence-grounded readiness |
| P3-Q05 | Public Journey A | `Agent.attach/run/inspect` discovers/searches/inspects SQLite catalog state, executes only validated SQL, completes from accepted evidence, and remains inspectable after reopen |
| P3-Q06 | Consolidated phase gate | Focused and complete deterministic suites on Python 3.11/3.12; architecture/import/static/root-oracle/distribution/fresh-install checks; parity/ledger review; scoped hooks and exact gate commit |

### Executed Phase 3 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P3-Q01a | plan, ADRs, current v2, and root v1 oracle (read-only) | Re-read Sections 6/15 and Phase 3 supporting sections; inventory catalog/SQL/source/projection/readiness owners and leaf tests; independently audit documentation and Phase 3 scope | PASS — no new product decision is required; six coherent tasks cover the SQLite-only vertical; exact catalog/data/runtime/adapter owners and focused contract slices are locked before production edits |
| P3-Q01b | `next/` | `../.venv/bin/python scripts/build_test_disposition.py --write`; isolated `pytest -o addopts='' tests/architecture/test_test_disposition.py -q -p no:cacheprovider` | PASS — 4 passed in 0.05s; deferred cloud, MySQL/MongoDB, Focus, telemetry, evaluation, and schema-synthesis paths no longer silently expand Phase 3 |
| P3-Q02a | `next/` | Focused catalog model/protocol/store/service/source selections | PASS — immutable identity/revision/facet/relationship/sync contracts, Migration 8, atomic snapshot replacement, FTS, traversal, isolation, rollback, reopen, and refresh semantics pass |
| P3-Q03a | `next/` | Focused adapter/discovery/query/declaration selections | PASS — read-only SQLite discovery/inspection/query, persisted source lifecycle, stable declarations, same-source refresh, and cancellation-safe connection ownership pass; invalid SQL reaches no connector I/O |
| P3-Q04a | `next/` | Focused SQL/result/controller/context and query-freshness selections | PASS — 35 original data-domain cases plus 8 freshness/provenance cases pass; missing/malformed/stale catalog facts fail closed before user SQL, and accepted evidence carries bounded revision provenance |
| P3-Q05a | `next/` | `tests/acceptance/test_sqlite_catalog_journey.py` and public reattach journey | PASS — catalog search/inspect, premature completion repair, mutation rejection, bounded query evidence/citation, transcript/reopen, declaration admission, and repeated attach all pass |
| P3-Q06a | `next/` | Complete isolated suite with `PYTHONPATH=src:<site-packages> <python> -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` on CPython 3.11.15 and 3.12.7 | PASS — 817 selected and 1 deselected in 9.93s/10.70s |
| P3-Q06b | `next/` | Black check; byte compilation; full mypy with `MYPYPATH=src`; pyright 1.1.411 | PASS — Black reports 132 files unchanged; compilation clean; mypy reports no issues in 131 files; pyright reports 0 errors/warnings |
| P3-Q06c | repository root and `next/` | Isolated architecture suite; disposition reproduction; root-oracle diff from `8545f85`; symlink and diff scans | PASS — 60 architecture tests; generated disposition is current; no root change, v2 symlink, or diff error |
| P3-Q06d | repository root | `.venv/bin/python -m pytest tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` | PASS — 2,498 passed and 221 deselected in 10.00s; the root oracle remains unchanged |
| P3-Q06e | clean copies under `/private/tmp/daita-v2-p3-gate.Y4lCqE` and `/private/tmp/daita-root-p3-gate.xQPYFd` | Build v2/root sdist and wheel without isolation; inspect cross-inclusion; install v2 wheel without dependencies into fresh CPython 3.11/3.12 environments; isolated import smoke | PASS — v2 wheel/sdist contain 53/73 entries at `2.0.0a0`; root remains 401/442 at `1.0.0`; no tests/scripts/nested `next/` cross the v2 distribution and no `next/` crosses root; both fresh imports resolve to their own `site-packages` without optional SDKs |
| P3-Q06f | final Phase 3 diff | Independent boundary review, four root-cause repairs, parity/ledger/scope review, configured hooks, and exact containing commit | PASS — repeated attach history, source freshness/provenance, adapter declaration admission, and cancellation-safe discovery were repaired with regressions; every changed repository path remains under `next/`; the containing commit is `chore(v2-phase-3): complete phase 3 gate` |

No live provider or external database run was required for Phase 3: the
production-model translation boundary was already proven in Phase 2, while
the Phase 3 gate is the deterministic local SQLite data journey.

## Phase 4 — non-database proof

Status: **PASS. Phase 4 is complete and Phase 5 may begin.**

The Phase 4 gate retained the lean phase-level cadence: focused tests inside
the vertical, followed by one broad regression/static/package gate and one
targeted security review.

### Executed Phase 4 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P4-Q01 | plan, ADRs, current v2, and root v1 file references (read-only) | Re-read Sections 6/15 and Phase 4 supporting sections; inventory sandbox, adapter, catalog facet, evidence, comparison, context, and Journey B owners | PASS — one explicit canonical root, descriptor-relative I/O, FILE+TABULAR facets, containment-only relationships, evidence-ID comparison, strict bounded CSV/JSON, and explicit deferrals were locked without a new runtime or catalog owner |
| P4-Q02 | `next/` | Focused local adapter/read, catalog facet, accepted-evidence, comparison, controller, and Journey B selections | PASS — the consolidated affected selection passed 120 tests; final typing repairs passed 25 focused tests without changing behavior |
| P4-Q03 | `next/` | `tests/acceptance/test_local_file_comparison_journey.py` plus the existing public Phase 3 acceptance regressions | PASS — Journey B passes in 2.57s with newest-file selection, file/SQLite reads, strict discrepancies, artifact provenance, citations, reopen, and transcript assertions; 19 prior public regressions remain green |
| P4-Q04 | `next/` | Complete isolated suite with `PYTHONPATH=src:<site-packages> <python> -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` on CPython 3.11.15 and 3.12.7 | PASS — 867 selected and 1 deselected in 14.39s/15.35s |
| P4-Q05 | `next/` | `black --check src tests scripts`; byte compilation; `MYPYPATH=src mypy src/daita tests scripts/build_test_disposition.py`; pyright 1.1.411 | PASS after typed-boundary repair — Black reports 144 files unchanged; compilation clean; mypy reports no issues in 143 files; pyright reports 0 errors/warnings |
| P4-Q06 | repository root and `next/` | Isolated architecture suite; generated test-disposition check; root-oracle diff from `99f5eb6`; symlink and diff scans | PASS — 65 architecture tests; disposition current; no root change, v2 symlink, or diff error; the sole production executor invocation remains in `operations/runtime.py` |
| P4-Q07 | repository root | `.venv/bin/python -m pytest tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` | PASS — 2,498 passed and 221 deselected in 10.37s; the root oracle remains unchanged |
| P4-Q08 | clean copies under `/private/tmp/daita-p4-packaging.NY0bRl` | Build v2/root sdist and wheel without isolation; inspect cross-inclusion; install the v2 wheel without dependencies into fresh CPython 3.11/3.12 environments; isolated import smoke | PASS — v2 wheel/sdist contain 57/77 entries at `2.0.0a0`; root remains 401/442 at `1.0.0`; no tests/scripts/nested `next/` cross the v2 distribution and no `next/` crosses root; both fresh environments import the public API and all 53 packaged modules from their own `site-packages` |
| P4-Q09 | final Phase 4 boundary review and repair selection | Review local-root admission, evidence provenance, comparison readiness, and architectural ownership; run affected local-file/readiness/Journey B tests on both interpreters | PASS after root-cause repair — descriptor-relative root admission closes the ancestor-substitution race; negated partial claims no longer satisfy disclosure; reconciliation language requires comparison evidence; 23 affected tests pass in 2.94s/3.42s on Python 3.11/3.12; no accepted-evidence or ownership blocker remains |
| P4-Q10 | final Phase 4 diff | Parity/ledger/scope review, configured hooks, and exact containing commit | PASS — every changed repository path is under `next/`; EXT-06 records the native CSV/JSON path; the containing commit is `chore(v2-phase-4): complete phase 4 gate` |

The first consolidated static pass exposed genuine Phase 4 annotations at the
new comparison and accepted-evidence boundaries. Explicit immutable-JSON
narrowing, narrow protocol dependencies, and typed test fixtures repaired the
diagnostics without production suppressions. The final targeted review then
found the three security/readiness issues recorded in P4-Q09; each was fixed
at its owning boundary with deterministic regressions rather than retries or
special-case execution paths.

No live provider or external database run was required for Phase 4. The gate
uses only sandboxed test-owned local files and SQLite state.

## Phase 5 — context, memory, skills, and learning

Status: **PASS. Phase 5 is complete and Phase 6 may begin.**

Phase 5 used focused tests for its coherent owner slices and one consolidated
cross-version/static/package gate.

### Executed Phase 5 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P5-Q01 | plan, ADRs, current v2, and root v1 references (read-only) | Re-read Sections 6/15, Phase 5, context/session/memory/learning/skill sections, Journey C, current owners, and retained v1 behavior references | PASS — context, model-profile, compression, memory, learning, skill, Migration 9, public API, and acceptance owners were locked without adding a parallel loop/runtime/catalog owner |
| P5-Q02 | `next/` | Focused context, model-profile, compression, memory, learning, skill, SQLite-state, public-composition, and Journey C selections | PASS — the consolidated affected selection passed 210 tests; final provenance/approval integration additions passed 2 tests; the active-skill pointer repair selection passed 6 tests |
| P5-Q03 | `next/` | Public learning, session-compression, skill-guidance, and skill-change acceptance journeys | PASS — exact correction changes a later grounded result; transcripts compress without losing required facts; skills guide but cannot govern; proposal acceptance, rollback history, and reopen remain inspectable |
| P5-Q04 | `next/` | Independent context/memory/skill/storage review plus exact Phase 5 architecture assertions | PASS after repair — discovery now leaves all active skill-index metadata pinned and activation swaps it atomically under CAS; derived FTS projection refresh is distinguished from append-only history; no remaining blocker |
| P5-Q05 | `next/` | Complete isolated suite with `PYTHONPATH=src:<site-packages> <python> -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` on CPython 3.11.15 and 3.12.7 | PASS — 1,014 selected and 1 deselected in 25.11s/26.47s |
| P5-Q06 | `next/` | One Black formatting pass; byte compilation; `MYPYPATH=src mypy src/daita tests scripts/build_test_disposition.py`; pyright 1.1.411 | PASS after explicit narrowing — 11 files formatted and 172 unchanged; compilation clean; mypy reports no issues in 182 files; pyright reports 0 errors/warnings |
| P5-Q07 | repository root and `next/` | Isolated architecture suite; generated test-disposition check; root-oracle diff from `e0bab24`; symlink, sole-executor, and diff scans | PASS — 69 architecture tests; disposition current; no root change, v2 symlink, or diff error; the sole production executor invocation remains in `operations/runtime.py` |
| P5-Q08 | repository root | `.venv/bin/python -m pytest tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` | PASS — 2,498 passed and 221 deselected in 11.43s; the frozen v1 oracle remains unchanged |
| P5-Q09 | clean copies under `/private/tmp/daita-p5-packaging.EY70gJ` | Build v2/root sdist and wheel without isolation; inspect cross-inclusion; install v2 wheel without dependencies into fresh CPython 3.11/3.12 environments; isolated import smoke | PASS — v2 wheel/sdist contain 73/96 entries at `2.0.0a0`; root remains 401/442 at `1.0.0`; archive boundaries are clean; both fresh environments import all 68 packaged modules from their own `site-packages` without optional SDKs |
| P5-Q10 | final Phase 5 diff | Parity/ledger/README/scope review, configured hooks, and exact containing commit | PASS — every changed repository path remains under `next/`; actual Phase 5 test paths replace stale planned paths; the containing commit is `chore(v2-phase-5): complete phase 5 gate` |

The first complete gate exposed two stale Phase 2 architecture-test
assumptions, not behavioral failures: the SQLite canonical-record allowlist
predated the Phase 5 records, and its append-only scanner treated replacement
of the derived memory FTS projection as history deletion. The guardrails now
name the new canonical modules and exact derived projection while continuing
to reject memory-version rewrites. The first consolidated static pass then
found explicit narrowing gaps at two production and seven test boundaries;
all were repaired without ignores or behavioral changes.

No live provider or external database run was required for Phase 5. The live
OpenAI boundary was already proven in Phase 2, and all Phase 5 state is local,
deterministic SQLite/file state.

## Phase 6 — local host and monitors

Status: **PASS. Phase 6 is complete and Phase 7 may begin.**

Phase 6 retained the lean phase-level cadence: focused tests during the four
host/monitor slices, followed by one consolidated cross-version, static,
architecture, root-oracle, packaging, and targeted safety-review gate.

### Executed Phase 6 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P6-Q01 | plan, ADRs, current v2, and retained v1 monitor/host references (read-only) | Re-read Sections 6/15, Phase 6, hosting/concurrency/monitor sections, Journey/CLI requirements, current owners, and retained v1 behaviors | PASS — foreground-only host ownership, ordinary monitor triggers/operations, one-shot scheduling, durable inbox/wakeups, and replaceable thin CLI boundaries were locked before production edits |
| P6-Q02 | `next/` | Focused monitor models/store/service/scheduler, SQLite Migration 10, host inbox/control, client/server/CLI, public monitor lifecycle, and Phase 6 architecture selections | PASS — final repaired selection passes 147 tests; strict framing/permissions, durable mutation-key binding, restart recovery, scheduler fencing, run-now reclaim, graceful shutdown, bootstrap, source/chat/inspect/approval/monitor routes, and no hidden embedded tasks are covered |
| P6-Q03 | `next/` | Targeted scheduler-fencing, host-shutdown, and local-transport safety review | PASS after root-cause repair — expired manual claims now reclaim the stable occurrence at fence/attempt +1; accepted socket handlers have bounded reads and settle before host/store shutdown; Migration 12 binds every control mutation key to one exact method/parameter hash; source attach and interruption are replay-safe after an admitted-request crash |
| P6-Q04 | `next/` | Complete isolated suite with `PYTHONPATH=src:<site-packages> <python> -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` on CPython 3.11.15 and 3.12.7 | PASS — 1,154 passed, 1 skipped, and 1 deselected in 47.53s/50.32s; the skip is only the real AF_UNIX bind forbidden by this sandbox |
| P6-Q05 | `next/` | One phase formatter pass plus one targeted post-review formatting pass; byte compilation; `MYPYPATH=src mypy src/daita tests scripts/build_test_disposition.py`; pyright 1.1.411 | PASS — final formatting completed; compilation clean; mypy reports no issues in 208 files; pyright reports 0 errors/warnings |
| P6-Q06 | `next/` | Isolated architecture suite; import/diff/symlink/sole-executor scans | PASS — 77 architecture tests; host/monitor packages add no alternate executor, provider, SQLite, or hidden-background-work owner; every socket mutation is admission-gated |
| P6-Q07 | repository root | `.venv/bin/python -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` | PASS — 2,498 passed and 221 deselected in 15.57s; the frozen v1 oracle remains unchanged |
| P6-Q08 | clean copy under `/private/tmp/daita-v2-p6-gate.Q1MbTG` | Build v2 sdist/wheel without isolation; inspect archive boundaries; install wheel without dependencies into fresh CPython 3.11/3.12 environments; import every packaged module and run installed `daita --help` | PASS — wheel/sdist contain 84/108 entries; no tests/scripts/nested `next/` are packaged; both environments import all 78 modules from `site-packages` and the console script starts without optional SDKs |
| P6-Q09 | unrestricted local execution | `PYTHONPATH=src ../.venv/bin/python -m pytest -o addopts='' -q -rA tests/unit/hosting/test_local_server.py::test_private_socket_lifecycle_and_health_when_sandbox_allows_bind` | PASS — 1 passed in 0.12s; the actual AF_UNIX server/client health request, private run directory, `0600` socket, ownership checks, cleanup, and host shutdown all complete successfully |

No live model or external database was required for Phase 6. The real socket
smoke used only a test-owned agent home and mock/no-model health request; it
required no credential and performed no external network I/O.

## Phase 7 — governance, write proof, and approval resume

Status: **PASS. Phase 7 is complete and Phase 8 may begin.**

Phase 7 used one durable-governance slice, one controlled SQLite-write slice,
one public Journey E slice, and one consolidated gate. The existing operation
runtime remains the sole executor, policy, lease, evidence, and recovery owner.

### Executed Phase 7 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P7-Q01 | plan, ADRs, current v2, and retained v1 references (read-only) | Re-read Sections 6/15, governance/data-safety/recovery sections, Phase 7, Journey E, and current runtime/catalog/SQLite/host owners | PASS — one persisted validator-fact contract, one semantic SQLite update recipe, the existing approval/resume path, and manual recovery for unknown outcomes were locked without another runtime or mutation-SQL surface |
| P7-Q02 | `next/` | Focused task/governance/store/migration/checkpoint, controlled-data, SQLite adapter/source, host projection, approval, loop, and restart selections | PASS — validator facts and prerequisite evidence survive restart; policy denial invokes no executor; 10 controlled-domain and 15 adapter/source tests pass; focused runtime/restart suites cover confirmed-no-effect, ambiguous execution/evidence/timeout, and manual recovery |
| P7-Q03 | `next/` | `tests/acceptance/test_sqlite_update_approval_journey.py` plus protected-agent-state public admission | PASS — validation and one-row impact precede approval; reopen resumes the same operation; preview/read tasks and evidence are not replayed; one write/result is committed; repeated resume is an exact no-op; approver, task, facts, and evidence IDs join durably; own state DB and hard links are rejected |
| P7-Q04 | `next/` | Complete isolated suite with `PYTHONPATH=src:<site-packages> <python> -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` on CPython 3.11.15 and 3.12.7 | PASS — 1,202 passed, 1 skipped, and 1 deselected in 40.37s/43.89s; the skip remains only the sandbox-forbidden real AF_UNIX bind |
| P7-Q05 | `next/` | One Black formatting pass and final check; byte compilation; `MYPYPATH=src mypy src/daita tests scripts/build_test_disposition.py`; pyright 1.1.411 | PASS after explicit narrowing — Black reports 214 files unchanged; compilation clean; mypy reports no issues in 213 files; pyright reports 0 errors/warnings |
| P7-Q06 | `next/` | Isolated architecture suite; generated disposition check; import/symlink/sole-executor/diff scans | PASS — 77 architecture tests and the generated disposition pass; the loop has no adapter branch, runtime remains the only executor caller, and the controlled backend is composition-owned |
| P7-Q07 | repository root | `.venv/bin/python -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` | PASS — 2,498 passed and 221 deselected in 10.40s; the frozen v1 oracle remains unchanged |
| P7-Q08 | clean copy under `/private/tmp/daita-v2-p7-gate.E3vASU` | Build v2 sdist/wheel without isolation; inspect boundaries; install wheel without dependencies in fresh CPython 3.11/3.12 environments; import every packaged module and run installed `daita --help` | PASS — wheel/sdist contain 85/109 entries; no tests/scripts/nested `next/` are packaged; both environments import all 80 modules from `site-packages` and start the CLI without optional SDKs |
| P7-Q09 | final controlled-write and recovery safety review | Test protected file identities, descriptor/connect swaps, virtual/shadow/view objects, same-version schema replacement, SQLite Unicode identifier semantics, affinity/no-op behavior, foreign-key cascades, rollback certainty, and commit-loss ambiguity | PASS after root-cause repair — every reviewed issue has an owning guard and deterministic regression; confirmed rollback fails/releases while commit ambiguity waits and expires into manual recovery |
| P7-Q10 | final Phase 7 diff | Parity/ledger/README/scope review, configured hooks, and exact containing commit | PASS — every changed repository path remains under `next/`; no arbitrary mutation SQL or public backend surface was added; the containing commit is `chore(v2-phase-7): complete phase 7 gate` |

No live model or external database was required for Phase 7. The controlled
source and durable marker are test-owned local SQLite files; production-model
translation was proven in Phase 2, and PostgreSQL belongs to Phase 8.

## Phase 8 — PostgreSQL and multi-provider parity

Status: **PASS. Phase 8 is complete and Phase 9 may begin.**

Phase 8 added PostgreSQL through the existing catalog/capability/runtime path,
five retained provider families through adapter-local wire translations, and
one policy-aware router over canonical model state. It introduced no second
loop, data runtime, catalog, executor boundary, or provider-owned transcript.

### Executed Phase 8 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P8-Q01 | plan, ADRs, current v2, and retained v1 references (read-only) | Re-read Sections 6/15, PostgreSQL/data/model/security sections, Phase 8, Journey F, and the Phase 9 live boundary | PASS — existing catalog, capability, operation-runtime, canonical-model, provider-adapter, and router seams were selected; stable SQLite identities and all isolation rules were retained |
| P8-Q02 | `next/` | Focused PostgreSQL adapter/query/source/catalog/controller/SQL/security and public Journey F selections | PASS — the final affected data selection passes 105 tests; exact schema-qualified base tables, supported built-in types/functions, `ONLY`, bounded projection, cancellation cleanup, detached diagnostics, secret references, case-distinct identifiers, and non-replay-safe execution facts are covered |
| P8-Q03 | `next/` | Shared six-provider conformance, provider-specific fake clients, registry, routing, continuation/reopen, security, and provider architecture selections | PASS — OpenAI, Anthropic, Gemini, Grok, Ollama, and generic OpenAI-compatible adapters normalize applicable text/tools/streaming/usage/errors/cancellation and lazy extras; routing proves capability/context/sensitivity admission, bounded fallback, canonical continuation, cost, telemetry, and missing-estimate fail-closed behavior |
| P8-Q04 | final PostgreSQL and model-security reviews | Independently review narrowed PostgreSQL authority and provider/router exception, fallback, sensitivity, continuation, and persistence boundaries | PASS after root-cause repair — PostgreSQL rejects views/foreign/custom-type and unsafe resolution paths before user SQL; provider exceptions are detached from raw tracebacks/causes/contexts, HTTP 408 normalizes to timeout, and a missing input estimate causes zero provider I/O |
| P8-Q05 | `next/` | Complete isolated suite on CPython 3.11.15 and 3.12.7 with `not requires_llm and not requires_db` | PASS — 1,459 passed and 2 skipped in 48.50s/51.17s; skips are the environment-gated real local-socket and live rows, not substituted mocks |
| P8-Q06 | `next/` | One formatting pass; final Black check; byte compilation; full mypy; pyright 1.1.411; isolated architecture suite; generated disposition check | PASS after explicit narrowing — Black reports all 242 files unchanged, compilation clean, mypy no issues in 241 files, pyright 0 errors/warnings, 84 architecture tests pass, and disposition generation is current |
| P8-Q07 | repository root | `.venv/bin/python -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider` | PASS — 2,498 passed and 221 deselected in 10.48s; the frozen v1 oracle remains unchanged |
| P8-Q08 | clean copy under `/private/tmp/daita-v2-p8-gate.Pqsut2` | Build sdist/wheel without isolation; inspect boundaries; install wheel without dependencies in fresh CPython 3.11/3.12 environments; import every packaged module and run installed `daita --help` | PASS — wheel/sdist contain 96/121 entries; no tests/scripts/nested `next`/cache/bytecode paths are packaged; both environments import all 91 modules from `site-packages` and start the CLI without optional SDKs |
| P8-Q09 | final Phase 8 diff | Parity/ledger/README/ADR/scope review, configured hooks, and exact containing commit | PASS — every changed repository path remains under `next/`; root v1 is untouched; the containing commit is `chore(v2-phase-8): complete phase 8 gate` |

Live Anthropic, Gemini, Grok, Ollama, generic-compatible, and real PostgreSQL
checks were **NOT RUN** in Phase 8. Their adapters are proven with protocol-
shaped fakes, but Phase 9 owns the required credential/service-backed live
acceptance and no mock result will satisfy those rows. The earlier actual
OpenAI persisted-loop evidence remains recorded under Phase 2.

## Live and external gates

- Phase 0 requires no live provider or external database.
- The Phase 2 production-model gate passed with an explicit model and actual
  OpenAI API call; the credential value was neither printed nor persisted.
- Phase 9 passed real OpenAI, Anthropic, Gemini, Grok, Ollama, explicit generic-
  compatible, and least-privileged PostgreSQL rows under P9-Q10 through P9-Q12.
  No mock result was substituted for any live row.

## Phase 9 — replacement candidate hardening

Status: **PASS. P9-01 through P9-08 are complete. The containing commit uses
`chore(v2-phase-9): complete phase 9 gate`, and work stops before Phase 10.**

This remains valid historical evidence for the hardening scope that existed at
the time. The later plan-to-source audit added a mandatory Phase 9.5 joined-
product gate, so Phase 9 PASS alone no longer establishes replacement
readiness or eligibility for Phase 10 authorization.

Broad formatting, typing, dual-Python, root-oracle, and clean-distribution
evidence was consolidated at P9-08. Completed live, fault, performance, and
security slice evidence below was consumed without an unnecessary duplicate
run. P9-03 is complete by explicit compatibility decision: `daita-cli` and
`daita-client` are retired from the Daita 2.0 support surface, so no sibling
mutation or external co-install run is part of this candidate gate.

### Executed Phase 9 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P9-Q01 | plan, ADRs, current v2, parity/disposition inventories, and retained v1 references (read-only) | Re-read Sections 6/15, Phase 9 work/gate, and replacement definition of done; require a real executable anchor for every public/cutover claim | PASS — existing planner/runtime/catalog/store owners remain singular; public/audit views are narrower than canonical state; integrations remain explicit and lazy |
| P9-Q02 | `next/` | `../.venv/bin/python -m pytest tests/architecture/test_parity_matrix.py tests/architecture/test_test_disposition.py tests/architecture/test_release_documentation.py tests/contract/config tests/contract/security tests/contract/test_errors.py tests/contract/extensions tests/contract/telemetry tests/contract/packaging tests/acceptance/test_public_agent_stream_and_detach.py -q` | PASS — 84 tests; every active parity claim has a real executable module anchor and every explicitly named function exists, all 164 v1 test modules have one audited disposition, and the public/config/error/secret/extension/telemetry/package surfaces pass |
| P9-Q03 | `next/` | `../.venv/bin/python -m pytest tests/contract/packaging/test_examples.py tests/architecture/test_release_documentation.py -q` plus direct offline execution of the nine retained example programs | PASS — 16 tests and all nine retained examples; production-shaped local-host help/dry-run passes; examples contain no v1 import or automatic approval |
| P9-Q04 | `next/` | `../.venv/bin/python -m pytest tests/contract/packaging/test_candidate_ci.py tests/contract/packaging/test_candidate_metadata.py tests/contract/packaging/test_minimal_import.py -q` | PASS — 10 packaging/CI contracts; exact extras, one entry-point owner, lazy minimal imports, source allowlist, and staged 3.11/3.12 CI declarations pass |
| P9-Q05 | `next/` | `../.venv/bin/python scripts/verify_candidate_lifecycle.py --python /opt/homebrew/bin/python3.11 --python /opt/homebrew/Caskroom/miniforge/base/bin/python3.12` | PASS — clean 105-entry wheel/132-entry sdist; dependency-free install, CLI help/init, offline operation run, stop, reopen/inspect, uninstall, import absence, and state retention pass on CPython 3.11.15 and 3.12.7 |
| P9-Q06 | `next/` | `../.venv/bin/python -m pytest tests/acceptance/test_real_process_kill_recovery.py tests/performance/test_phase9_candidate_baselines.py --junitxml=/private/tmp/daita-phase9-baselines-root.xml -q` | PASS — 5 tests in 1.81s; real child SIGKILL/reopen resumes one persisted model-call identity, and compact loop/token/observation/SQLite-WAL/blob/catalog/contention/monitor measurements remain beneath documented conservative ceilings |
| P9-Q07 | `next/` | `../.venv/bin/python -m pytest tests/acceptance/test_phase9_adversarial_security.py tests/contract/security/test_secret_provider.py -q` | PASS — 9 tests in 2.49s after root-cause repair; malicious catalog/file/row/connector content gains no authority, a unique resolved secret is absent from persisted/public/diagnostic/CLI surfaces, projections remain separated, and explicit empty secret resolution cannot regain implicit fallback through double composition |
| P9-Q08 | `next/` | `../.venv/bin/python -m pytest tests/contract/test_errors.py tests/unit/llm/test_provider_errors.py tests/unit/llm/test_routing.py tests/contract/models/test_provider_conformance.py -q` | PASS — 91 tests; broad subsystem failures fail closed with unknown retryability, normalized provider rate-limit/authentication failures are catchable through both provider-specific and public categories, and routing/conformance behavior is unchanged |
| P9-Q09 | `next/` | `../.venv/bin/python -m pytest tests/contract/config tests/unit/hosting/test_host.py tests/contract/hosting/test_sqlite_host_inbox.py tests/contract/storage/test_sqlite_task_migration.py -q` | PASS — 38 tests; migration 14 stores one immutable canonical budget/policy binding, omitted Agent/Host reopen loads it, explicit drift fails closed, v13 migration invents no defaults, and corruption/trigger guards hold |
| P9-Q10 | temporary test-owned Docker Ollama service plus `next/` | `DAITA_RUN_LIVE_LLM=1 DAITA_RUN_OLLAMA=1 OLLAMA_TEST_MODEL=qwen2.5:0.5b-instruct ../.venv/bin/python -m pytest tests/live/test_model_providers_live.py -q -k ollama --junitxml=/private/tmp/daita-p9-ollama-live.xml`, followed by container/image absence checks | PASS — 1 real adapter row; an initial `smollm:135m` run honestly failed because the model reached the 16-token limit, while the official instruction-tuned 0.5B model returned a canonical STOP response without changing the test or adapter; the test-owned container and newly pulled runtime image were removed |
| P9-Q11 | `next/`, with provider credentials inherited by their named environment variables and never printed | `DAITA_RUN_LIVE_LLM=1 ../.venv/bin/python -m pytest tests/live/test_model_providers_live.py -q -rs`; targeted Gemini rerun after adapter repair; focused provider regression and marker-off selections | PASS — real OpenAI Responses, Anthropic Messages, Grok/xAI, and explicit OpenAI-compatible Chat Completions rows passed initially; Gemini exposed nullable `Part.thought`, then passed after the shared nonstreaming/streaming decoder repair; 87 deterministic provider regressions passed; the marker-off selection reported 7 explicit skips rather than mock substitutions |
| P9-Q12 | temporary no-volume PostgreSQL 16 service plus `next/`, with the named `DAITA_LIVE_POSTGRES_*` variables supplied without logging values | `DAITA_RUN_LIVE_POSTGRES=1 ../.venv/bin/python -m pytest tests/live/test_postgresql_live.py -q -rs`; exact-role privilege and persisted-secret scans; exact-container absence check | PASS — 1 real least-privileged row cataloged/searched a test-owned two-row table and executed bounded `COUNT(*)` through the existing backend; the ordinary login had SELECT but no schema-create/INSERT/UPDATE/DELETE authority; its credential was absent from Agent Home; the container stopped and auto-removed |
| P9-Q13 | `next/` | `../.venv/bin/python -m pytest tests/architecture/test_parity_matrix.py tests/architecture/test_test_disposition.py tests/architecture/test_release_documentation.py tests/contract/packaging/test_candidate_metadata.py -q` | PASS — 23 focused contracts; live provider/PostgreSQL parity anchors resolve to real test functions, the specifically live v1 modules map to those live v2 owners, all 164 dispositions match the generator, source documentation links resolve, and the distribution README cannot link to source-only local files |
| P9-Q14 | `next/` | First clean dual-interpreter collection before the final deterministic run | REPAIRED — both interpreters exposed the same four release-candidate failures: a duplicate test-module basename and stale parity/disposition anchor, a missing canonical-config architecture allowance, and one pre-projection event assertion. Domain-specific test naming, regenerated dispositions, the canonical owner allowance, and the public projection assertion repaired all four; their focused regression passed 4/4 |
| P9-Q15 | `next/` | `PYTHONPATH=src:<interpreter-site-packages> <python> -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider --junitxml=<interpreter-result>` on CPython 3.11.15 and 3.12.7 | PASS — 1,555 passed, 1 skipped, and 8 live rows deselected on each interpreter in 54.60s/58.42s. The sole skip is the sandbox-forbidden AF_UNIX bind; real candidate socket lifecycle evidence already exists. The retired external packages have no co-install requirement or claimed result |
| P9-Q16 | `next/` | One Phase 9 formatting pass and final Black check; byte compilation; `MYPYPATH=src ../.venv/bin/python -m mypy src/daita tests scripts/build_test_disposition.py scripts/verify_candidate_lifecycle.py`; `npx --yes pyright@1.1.411 --pythonpath ../.venv/bin/python` | PASS after typed-boundary repair — all 281 Python files are formatted and compile; mypy reports no issues in 269 files; Pyright reports 0 errors/warnings. Public streams are explicitly async-closeable, extension decorator overloads are exact, and provider auth/rate-limit categories retain typed safe facts |
| P9-Q17 | `next/` | `../.venv/bin/python scripts/verify_candidate_lifecycle.py --python /opt/homebrew/bin/python3.11 --python /opt/homebrew/Caskroom/miniforge/base/bin/python3.12` | PASS — current 105-entry wheel and 132-entry sdist build reproducibly; dependency-free install/init/run/stop/reopen/uninstall succeeds on CPython 3.11.15/3.12.7, and state remains after uninstall |
| P9-Q18 | repository root | Root safe suite; `next/scripts/capture_v1_oracles.py --check`; baseline-scoped `git diff --quiet ... -- daita tests pyproject.toml`; and `git diff --check` | PASS — root v1 reports 2,498 passed and 221 deselected; frozen oracles match; root production/tests/metadata remain unchanged from the recorded baseline; the complete diff is whitespace-clean |
| P9-Q19 | `next/` | `../.venv/bin/python -m pytest tests/architecture/test_phase0_constitution.py tests/architecture/test_parity_matrix.py tests/architecture/test_release_documentation.py tests/contract/packaging/test_candidate_metadata.py -q` | PASS — 25 focused cases after the first run exposed three documentation wording/line-wrap assertion mismatches. ADRs 0001–0016 are accepted; one `daita` entry point belongs to `daita-agents`; no dependency/import or port-8123/secondary-runtime fallback exists; legacy-package retirement and actionable uninstall verification are explicit |
| P9-Q20 | repository root and `next/` | `../.venv/bin/python scripts/build_test_disposition.py --check`; `git diff --quiet b87df31873d33fffbf50498f5dc4d8892115e8f8 -- daita tests pyproject.toml`; `git diff --check`; final stale-claim and scope audit | PASS — all 164 v1 test dispositions match the generator; root production/tests/metadata remain frozen; the diff is whitespace-clean and entirely under `next/`; no pending external-integration claim or Phase 10 work remains |

P9-03 is complete by explicit product decision: `daita-agents` is the sole
Daita 2.0 local distribution and console owner, while the separate `daita-cli`
and `daita-client` packages are retired from the 2.0 support surface and their
sibling repositories remain unchanged. No external co-install/socket test was
run or is claimed. The staged candidate workflow under `next/.github/` remains
intentionally inert in the root repository until the excluded Phase 10
cutover. Every candidate, live, lifecycle, fault, performance, security,
focused retirement, and root row required by P9-08 has passed. The containing
gate commit uses the required Phase 9 message; Phase 10 remains excluded.

## Phase 9.5 — replacement-readiness closure

Status: **PASS.** P9.5-Q01 through P9.5-Q08 pass. Phase 9's exact PASS rows
above remain historical evidence and are cited where their boundaries are
unchanged. Passing this gate does not authorize Phase 10.

Focused red/green evidence will be recorded under the owning P9.5 task. Broad
formatting, static, dual-Python, root-oracle, build/install, security, and
affected live checks are deliberately consolidated at P9.5-Q08.

### Executed planning-only evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P9.5-PQ01 | repository root | Update the ignored governing plan, recompute `shasum -a 256 docs/DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md`, and bind the result in both ledgers and the import-firewall contract | PASS — the plan, `STATUS.md`, `QUALITY_GATES.md`, and `test_import_firewall.py` agree on `e54f43dd0bfc0fa8478b496e7d2a89e53439d7fe9f5c8cf58f5c947f7682364b` |
| P9.5-PQ02 | `next/` | `../.venv/bin/python -m pytest tests/architecture/test_import_firewall.py tests/architecture/test_phase0_constitution.py tests/architecture/test_release_documentation.py tests/architecture/test_parity_matrix.py -q` | PASS — 24 passed; the first run had one documentation-contract failure because revised parity wording omitted the exact `does not authorize Phase 10` phrase, which was restored before the passing rerun |

These rows validate only the Phase 9.5 plan, ADR/document inventory, plan
fingerprint, parity vocabulary, and local documentation links. They do not
satisfy any behavioral or gate row below.

### Planned Phase 9.5 evidence

| ID | Scope | Required evidence | Status |
| --- | --- | --- | --- |
| P9.5-Q01 | Contract rebaseline | Re-read the governing ownership/invariant sections; reconcile all six audited gaps with existing owners and deferrals; add representative expected-red black-box anchors that fail for the missing product contract rather than setup drift | PASS — mandatory sections and ADR 0017 re-read; owner/deferral map recorded in `STATUS.md`; `../.venv/bin/python -m pytest tests/acceptance/test_phase_9_5_product_contracts.py -q` produced the intended 6 failures: read facts schema 0, cold host unconfigured, default finding absent, natural memory absent, configured-extension argument absent, and real CLI `agent create` absent. The monitor anchor was corrected after its first run exposed a missing host-start/idempotency setup requirement; its isolated rerun then failed only because `finding_id` was `None` |
| P9.5-Q02 | Trusted read authority and evidence | SQL/file/comparison tasks durably preserve exact validator-owned source/resource/revision/freshness/sensitivity facts before I/O; governance/routing consume them; adapters revalidate at I/O; migrations invent no authority; rejected output retains no unsafe payload | PASS — the real SQLite acceptance anchor passes through close/reopen and detects task/evidence sensitivity tamper; data-domain, adapter, operation/runtime/router, storage-migration, config, and host compatibility selections pass 520/520; v15 persists exact single/multi-source freshness plus canonical mutually exclusive evidence disposition/projection/redaction metadata, while v4 legacy evidence reopens schema-zero; source/resource/revision drift, stale adapter scope, and disallowed provider sensitivity fail before I/O; rejected payload/artifact content is discarded. Touched files are Black-clean and compile; focused mypy reports no issues in 13 files and Pyright 1.1.411 reports 0 errors/warnings |
| P9.5-Q03 | Reconstructable model route | Versioned provider-neutral primary/fallback configuration, non-secret endpoints, retry/sensitivity/output policy, and secret references survive a new process; configured `Agent.open().run()` and model-free host start work; operation route revisions bind resume; drift/tamper/missing secrets fail closed | PASS — migration 16 stores append-only normalized candidates plus a CAS route head and nullable immutable operation bindings. `../.venv/bin/python -m pytest tests/contract/config/test_agent_configuration.py tests/contract/config/test_runtime_configuration.py tests/contract/config/test_model_route_configuration.py tests/contract/storage/test_sqlite_task_migration.py tests/contract/storage/test_sqlite_phase5_state.py tests/contract/storage/test_sqlite_approval_projection.py tests/contract/hosting/test_sqlite_host_inbox.py tests/acceptance/test_provider_continuity.py tests/acceptance/test_phase_9_5_product_contracts.py::test_configured_retained_model_reopens_without_caller_injection -q` passed 52/52. Nine route-specific cases cover exact cost/profile/order/grant/reference reopen, cold default-domain run and operation reinspection, router-owned retry/fallback, CAS replay/stale/nonterminal/success, v15 no-invention migration, tamper/unsafe-endpoint rejection, missing secret/extra zero provider I/O, and absence of a resolved sentinel secret from Agent Home. Fourteen touched route-boundary files are Black-clean and compile; focused mypy is clean across 11 source files; Pyright 1.1.411 reports 0 errors/warnings. |
| P9.5-Q04 | Default monitor semantics | The production default host binds confirmed definitions, enforces source/resource scope and restriction-only effective budgets/policy, rejects unsupported conditions, evaluates accepted current-operation evidence, and creates zero/one correctly linked finding across restart and replay without an injected positive projector | PASS — `.venv/bin/python -m pytest next/tests/unit/monitors next/tests/unit/domains/data next/tests/unit/catalog next/tests/unit/hosting next/tests/unit/operations next/tests/contract/monitors next/tests/contract/domains next/tests/contract/operations next/tests/contract/hosting next/tests/acceptance/test_monitor_lifecycle.py next/tests/acceptance/test_phase_9_5_monitor_operations.py next/tests/acceptance/test_phase_9_5_product_contracts.py::test_default_host_projects_always_monitor_finding_from_read_evidence -q --junitxml=/private/tmp/daita-p95-q04.xml` recorded 551 tests: 550 passed and one sandbox-only Unix-socket bind skip. Four default-host acceptance cases prove matched/unmatched typed thresholds, exactly linked zero/one findings, pre-catalog/pre-executor out-of-scope rejection, enforced one-turn budget, missed-run catch-up once, and two-restart deduplication with the production projector. Unsupported expression construction and authority-expanding budget/policy/template proposals reject before monitor persistence; approval-wake reclaim retains the same trigger/operation. Black and compilation are clean; focused mypy is clean across 12 source files; Pyright 1.1.411 reports 0 errors/warnings. |
| P9.5-Q05 | Natural learning and skill proposals | Ordinary correction/remember input changes a later grounded action through provenance/revision/sensitivity-safe memory; accepted evidence is required for fact learning; failed/blocked/sensitive candidates reject safely; ordinary skill changes remain inert proposals until accepted | PASS — `.venv/bin/python -m pytest next/tests/unit/test_learning.py next/tests/unit/memory next/tests/unit/skills next/tests/contract/storage/test_sqlite_phase5_state.py next/tests/acceptance/test_learning_journey.py next/tests/acceptance/test_skill_change_lifecycle.py next/tests/acceptance/test_phase_9_5_natural_learning.py next/tests/acceptance/test_phase_9_5_product_contracts.py::test_ordinary_remember_interaction_enters_learning_service -q -x --junitxml=/private/tmp/daita-p95-q05-focused.xml` recorded 91 passed with no skips. The natural alias journey changes a later grounded SQL parameter, survives close/reopen with exact user/operation/trigger provenance, supersedes compatibly, and becomes stale after catalog revision. Default-data fact proposals bind exact accepted current read evidence; absent evidence and PII reject with no retained candidate payload. Natural skill proposals survive reopen but remain absent from active skills until explicit audited acceptance; executable candidates redact and never create a version. Failed/blocked source and credential/policy/code safety contracts pass. Twenty touched learning-boundary files are Black-clean and compile; focused mypy is clean across eight source/test files; Pyright 1.1.411 reports 0 errors/warnings; the skill package dependency firewall remains green. |
| P9.5-Q06 | Additive extension composition | Explicit built-ins plus one configured extension coexist in normal Agent/host context and execute through the sole operation runtime; manifest identity/version/fingerprint drift and collisions fail atomically; unsupported manifest categories are reclassified | PASS — `.venv/bin/python -m pytest next/tests/unit/test_capabilities.py next/tests/unit/domains/data next/tests/unit/hosting next/tests/unit/storage next/tests/contract/extensions next/tests/contract/config next/tests/contract/storage/test_sqlite_task_migration.py next/tests/contract/storage/test_sqlite_phase5_state.py next/tests/contract/storage/test_sqlite_approval_projection.py next/tests/contract/hosting/test_sqlite_host_inbox.py next/tests/acceptance/test_public_agent.py next/tests/acceptance/test_phase_9_5_extension_composition.py next/tests/acceptance/test_phase_9_5_product_contracts.py::test_configured_extension_composes_with_builtin_data_capability next/tests/architecture/test_phase1_loop_architecture.py next/tests/architecture/test_phase2_task_execution_lifecycle_architecture.py next/tests/architecture/test_phase2_task_persistence_architecture.py next/tests/architecture/test_phase4_non_database_architecture.py next/tests/architecture/test_parity_matrix.py -q -x --junitxml=/private/tmp/daita-p95-q06-focused.xml` recorded 415 tests: 414 passed and one sandbox-only Unix-socket skip. Default built-ins and one explicit capability provider project together and execute as persisted tasks/evidence through the sole operation runtime; exact manifest-set binding survives Agent/host reopen. Missing, version/declaration drift, oversized sets, and built-in collisions fail before provider/executor I/O or partial Agent Home publication. Monitor operations exclude unscoped configured tools. Resource-adapter/backend-provider manifest kinds are explicit post-MVP rejections. Sixteen touched files are Black-clean and compile; focused mypy is clean across eight source files; Pyright 1.1.411 reports zero errors/warnings. |
| P9.5-Q07 | In-package API/CLI product journey | An installed candidate uses the real Unix socket for agent/model/source setup, model-free serve, interactive streamed chat, operation/approval/catalog/memory/skill/monitor inspection, natural monitor proposal/confirmation, and reconnecting committed-event follow; CLI code owns no runtime semantics | PASS — `.venv/bin/python -m pytest next/tests/unit/test_cli.py next/tests/unit/hosting/test_local_server.py next/tests/unit/hosting/test_host.py next/tests/unit/hosting/test_embedded_control.py next/tests/unit/monitors/test_service.py next/tests/contract/runtime/test_operation_store.py next/tests/architecture/test_phase2_persistence_architecture.py next/tests/architecture/test_phase6_host_architecture.py -q -rs --junitxml=/private/tmp/daita-p95-q07-focused.xml` recorded 117 tests: 116 passed and one sandbox-only Unix-socket skip. The acceptance journey then passed outside the sandbox from the source tree and with `DAITA_TEST_INSTALLED_CLI=/private/tmp/daita-p95-q07.uHu2p8/venv/bin/daita` after an isolated no-dependency editable install of candidate `2.0.0a0` on CPython 3.11.15. It exercises the real socket and installed console with agent/source/model first-run paths, model-free configured serve coverage, grounded SQLite interactive chat with committed events/evidence, operation and non-empty approval inspection, catalog/memory/skill/monitor reads, natural proposal plus confirmation, source detach, and a reconnecting cursor follower that observes a newly committed `monitor.pause`. Every mutation dispatches to AgentHost; the Phase 6 no-CLI-background-task assertion passes. Touched P9.5-07 files are Black-clean; focused mypy is clean across eight source files and Pyright 1.1.411 reports 0 errors/warnings. A clean non-editable wheel install remains part of P9.5-Q08. |
| P9.5-Q08 | Joined consolidated gate | A real retained provider and supported real source complete the default data-domain journey through close/reopen without reinjection; the default monitor, natural-learning, and additive-extension journeys pass; affected live, migration, restart/fault, security, deterministic dual-Python, static, architecture/import, frozen-root, and clean distribution gates pass in one consolidated run | PASS — P9.5-Q08a through P9.5-Q08h below pass. The candidate is eligible for separately authorized human Phase 10 review; no cutover, push, publication, PR, or release work occurred. |

### Executed P9.5-Q08 evidence

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| P9.5-Q08a | repository root | `PYTHONPATH=next/src .venv/bin/python -m pytest -o addopts='' next/tests/acceptance/test_phase_9_5_product_contracts.py next/tests/acceptance/test_phase_9_5_monitor_operations.py next/tests/acceptance/test_phase_9_5_natural_learning.py next/tests/acceptance/test_phase_9_5_extension_composition.py next/tests/acceptance/test_phase_9_5_cli_product_journey.py next/tests/contract/config/test_model_route_configuration.py next/tests/contract/extensions/test_extension_binding.py -q -rs -p no:cacheprovider --junitxml=/private/tmp/daita-p95-q08-focused.xml` | PASS — 25 passed and one expected sandbox-only Unix-socket bind skip. The exact monitor/natural-learning/extension subset separately passed 7/7. |
| P9.5-Q08b | repository root | Complete `next/tests` deterministic selection with `-S`, `-m 'not requires_llm and not requires_db'`, `-p no:cacheprovider`, and explicit `next/src` plus interpreter site-packages on `/Users/jendala/daita/daita-agents/.venv/bin/python` and `/opt/homebrew/Caskroom/miniforge/base/bin/python3.12` | PASS — the final post-documentation run reports 1,631 passed, two sandbox-only socket skips, and nine live deselections on both CPython 3.11.15 (79.92s) and 3.12.7 (89.59s). The first consolidated run found four real join defects; fixes bound CLI subprocess cwd, rejected provider/profile mismatch before home mutation, narrowed the monitor import exception, and made the v15 multi-source columns authoritative over the migration-13 scalar compatibility field. |
| P9.5-Q08c | repository root and `next/` | `.venv/bin/black next/src next/tests next/scripts`; `.venv/bin/black --check next/src next/tests next/scripts`; `.venv/bin/python -m compileall -q next/src next/tests next/scripts`; `MYPYPATH=next/src .venv/bin/python -m mypy next/src/daita next/tests next/scripts/build_test_disposition.py next/scripts/verify_candidate_lifecycle.py`; `npx --yes pyright@1.1.411 --pythonpath ../.venv/bin/python` from `next/`; `PYTHONPATH=next/src .venv/bin/python -m pytest -o addopts='' next/tests/architecture -q -p no:cacheprovider` | PASS after typed-caller repair — Black formatted four files and the final check reports 279 unchanged; compilation succeeded; mypy reports no issues in 278 files; Pyright 1.1.411 reports 0 errors/warnings; all 93 architecture/import-firewall cases pass. The first static pass exposed 48 mypy and 11 Pyright diagnostics in affected test doubles/narrowing, which were corrected without suppressions or a new abstraction. |
| P9.5-Q08d | repository root | Focused migration/operation-store/task-projection, loop restart/real-process-kill/cancellation, runtime recovery/lease/cancellation, SQLite/blob cancellation, adapter cancellation, and security selection recorded at `/private/tmp/daita-p95-q08-recovery-security.xml`; real SQLite attach/catalog/write-approval/local-file-comparison/trusted-authority selection recorded at `/private/tmp/daita-p95-q08-sqlite.xml` | PASS — 227 migration/recovery/fault/cancellation/lease/security cases and five real SQLite/local-file boundary cases. The cross-source regression proves exact multi-source authority round-trips through the existing SQLite owner. |
| P9.5-Q08e | `next/` and clean copies under `/tmp` | `../.venv/bin/python scripts/verify_candidate_lifecycle.py --phase-9-5-joined --python /opt/homebrew/bin/python3.11 --python /opt/homebrew/Caskroom/miniforge/base/bin/python3.12`; plus `PYTHONPATH=next/src .venv/bin/python -m pytest -o addopts='' next/tests/contract/packaging -q -p no:cacheprovider` | PASS — 26 packaging contracts; clean 106-entry wheel and 133-entry sdist at `2.0.0a0`; fresh CPython 3.11.15/3.12.7 installs include declared OpenAI/SQLite extras and use a loopback OpenAI-compatible provider plus real SQLite through agent create, retained model set, source add, model-free host, interactive chat, committed-event follow, evidence inspection, clean SIGINT stop, injection-free cold reopen/second run, uninstall, and retained state. The sandbox-only first attempt could not resolve PyPI; the authorized rerun exposed and repaired one interpreter-shared test database before both rows passed. |
| P9.5-Q08f | repository root outside the socket sandbox | `PYTHONPATH=next/src .venv/bin/python -m pytest -o addopts='' next/tests/acceptance/test_phase_9_5_cli_product_journey.py -q -s -p no:cacheprovider --junitxml=/private/tmp/daita-p95-q08-real-socket.xml` | PASS — one real Unix-socket/subprocess journey; combined with Q08e this proves source and clean-wheel console paths. |
| P9.5-Q08g | repository root with named configured environment variables, no values logged | `set -a; source .env; set +a; DAITA_RUN_LIVE_LLM=1 .venv/bin/python -m pytest next/tests/live/test_phase_9_5_retained_data_live.py -q -rs -s -p no:cacheprovider --junitxml=/private/tmp/daita-p95-q08-live-openai.xml` | PASS — one real OpenAI/default-data-domain/SQLite case in 8.938s creates accepted exact-authority evidence, closes, cold-opens with no model injection, and succeeds again. Unchanged Phase 9 P9-Q11 Anthropic/Gemini/Grok/Ollama/compatible rows and P9-Q12 real SELECT-only PostgreSQL row are cited rather than rerun. |
| P9.5-Q08h | repository root, `next/`, and clean root copy `/private/tmp/daita-p95-root-build.10pCWu` | `../.venv/bin/python scripts/build_test_disposition.py --check`; `PYTHONPATH=.:... .venv/bin/python -S next/scripts/capture_v1_oracles.py --check`; `git diff --quiet b87df31873d33fffbf50498f5dc4d8892115e8f8 -- daita tests pyproject.toml`; `git diff --check`; `.venv/bin/python -m pytest tests/ -m 'not requires_llm and not requires_db' -q --tb=short -p no:cacheprovider`; clean root `python -m build --no-isolation` plus archive/version/cross-inclusion scan | PASS — disposition and all four v1 fixtures reproduce; root production/tests/metadata remain frozen; diff is whitespace-clean; root reports 2,498 passed and 221 deselected; root `1.0.0` wheel/sdist remain 401/442 entries and contain no `next/`. The governing plan hash remains `e54f43dd0bfc0fa8478b496e7d2a89e53439d7fe9f5c8cf58f5c947f7682364b`. |

### Completed Phase 9.5 gate

- [x] P9.5-Q01 through P9.5-Q07 have truthful focused evidence.
- [x] The joined real provider/default data-domain/source/reopen journey passes;
      unchanged adapter-only Phase 9 live rows are cited rather than rerun.
- [x] P9.5-Q08 passes once, with exact commands, versions, skips, and unavailable
      credentials/services recorded here.
- [x] Support, parity, operations, security, migration, and breaking-change
      claims match executed behavior rather than the planned surface.
- [x] Root v1 remains frozen and no legacy-package fallback exists.
- [x] One reviewed `chore(v2-phase-9-5): complete phase 9.5 gate` commit contains
      the final ledger evidence. Phase 10 remains excluded before and after
      that commit unless separately authorized.

## Live LLM readiness — Wave 1

Status: **LLM-G00 PASS; LLM-G01 PASS; LLM-G02 FAIL.** The deterministic
harness and all non-live regression gates pass, but the first authorized real
OpenAI execution failed every row. No real-provider acceptance is claimed, no
repair or rerun occurred, no later wave is started, and Phase 10 remains
excluded.

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| W1-Q01 | `next/` | `../.venv/bin/python -m pytest tests/live/mvp -m 'not requires_llm' -q -rs` | PASS — 8 fixture/oracle/configuration/prompt/redaction self-tests. Manifest `wave1-commerce-v1` reproduces digest `sha256:a313d61d1e4dc0411b02e3d3314ba49b18dc238ea62621f1d77b9f5b4f01439c`; prompt corpus `wave1-prompts-v1` has three uncoached variants for each of four scenarios. |
| W1-Q02 | `next/`, named selectors and credential absent from the process | `../.venv/bin/python -m pytest tests/live/mvp -m requires_llm -q -rs` | BLOCKED / NOT RUN — 12/12 rows skipped before provider construction. The exact missing process settings were `DAITA_RUN_LIVE_LLM=1`, `DAITA_RUN_LIVE_MVP=1`, `DAITA_LIVE_MVP_PROVIDER=openai`, `DAITA_LIVE_MVP_MODEL=<explicit-model>`, and `OPENAI_API_KEY`. Zero remote calls, tokens, tool calls, provider latency, retries, or fallbacks were recorded. A repository-local dotenv contains a named credential, but the harness did not load it or infer a model. |
| W1-Q03 | `next/`, CPython 3.11.15 | `../.venv/bin/python -m pytest tests/ -m 'not requires_llm and not requires_db' -q -rs -p no:cacheprovider --junitxml=/private/tmp/daita-wave1-deterministic.xml` | PASS — 1,640 passed and two known sandbox-only Unix-socket tests skipped; 1,642 selected in 74.356s. No production code changed, so a second complete-interpreter run was not required by the Wave 1 contract. |
| W1-Q04 | `next/` | Black over `src tests scripts`; byte compilation; full mypy with `MYPYPATH=src`; Pyright 1.1.411; all architecture tests | PASS after one test-only unused-import repair — Black leaves 288 files unchanged; compilation succeeds; mypy reports no issues in 287 files; Pyright reports zero errors/warnings; 94 architecture tests pass. |
| W1-Q05 | repository root and `next/` | `scripts/build_test_disposition.py --check`; `next/scripts/capture_v1_oracles.py --check`; baseline-scoped root diff; `git diff --check`; root `pytest tests/ -m 'not requires_llm and not requires_db'` | PASS — generated disposition and four frozen-v1 fixtures reproduce; root production/tests/metadata match `b87df31873d33fffbf50498f5dc4d8892115e8f8`; the diff is whitespace-clean and confined to `next/`; root reports 2,498 passed and 221 deselected. |
| W1-Q06 | repository root, with user-approved synthetic-data transmission and credential loaded from `.env` without printing it | `set -a; source .env; set +a; DAITA_RUN_LIVE_LLM=1 DAITA_RUN_LIVE_MVP=1 DAITA_LIVE_MVP_PROVIDER=openai DAITA_LIVE_MVP_MODEL=gpt-4.1-mini .venv/bin/python -m pytest next/tests/live/mvp -m "requires_llm and acceptance" -q -rs -s -p no:cacheprovider --junitxml=/private/tmp/daita-live-mvp-wave1.xml` | FAIL — 12 failed, zero passed/skipped in 340.079s on CPython 3.11.15. All 74 real OpenAI calls completed without provider error. Retained operations record 273,205 input and 11,727 output tokens, 47 tasks, and 47 accepted evidence records; retries/fallbacks were zero. No failed row was rerun. |
| W1-Q07 | retained synthetic pytest homes and JUnit, read-only | Existing `assert_artifacts_redacted` plus a SQLite cross-session scan using the configured secret and generated sentinel values without printing either | PASS — the credential is absent from all retained Agent Homes and JUnit; all three isolation sentinels are absent from primary-session model requests. The first diagnostic command referenced `.env` from the wrong directory and performed no scan; the corrected `../.env` command passed. |

### First live row results

| Scenario / variant | Seconds | Calls / tasks | Input / output tokens | Terminal reason | Observed failure evidence |
| --- | ---: | ---: | ---: | --- | --- |
| LIVE-MVP-01 direct | 58.958 | 9 / 4 | 37,180 / 1,264 | `repair_budget_exhausted` | Repeated missing-column and invalid-action repairs; one query completed but the operation did not reach grounded readiness. |
| LIVE-MVP-01 conversational | 59.986 | 7 / 5 | 29,184 / 1,395 | `repair_budget_exhausted` | Wrong adapter/SQL rejections followed by five accepted query results and repeated `data.not_grounded`. |
| LIVE-MVP-01 ambiguous | 29.838 | 6 / 5 | 27,086 / 1,591 | `repair_budget_exhausted` | Inspections succeeded; invalid arguments and a nonexistent file resource prevented a completed answer. |
| LIVE-MVP-02 direct | 29.590 | 7 / 5 | 38,192 / 867 | `repair_budget_exhausted` | Five query results were accepted, but each final candidate remained `data.not_grounded`. |
| LIVE-MVP-02 conversational | 15.711 | 3 / 4 | 11,880 / 1,002 | `no_progress_action_failure_limit` | Four inspections succeeded; two repeated missing-column SQL actions triggered no-progress termination. |
| LIVE-MVP-02 ambiguous | 55.785 | 12 / 9 | 63,651 / 2,102 | `turn_budget_exhausted` | Nine queries succeeded amid wrong-adapter and missing-column repairs; no completed grounded answer emerged. |
| LIVE-MVP-03 direct | 17.666 | 5 / 0 | 9,113 / 597 | `repair_budget_exhausted` | File resource selection and arguments were invalid; no task executed. |
| LIVE-MVP-03 conversational | 27.406 | 7 / 5 | 31,275 / 1,068 | `repair_budget_exhausted` | Searches and two file reads succeeded, but invalid comparison/query arguments left the result ungrounded. |
| LIVE-MVP-03 ambiguous | 14.404 | 5 / 0 | 9,831 / 742 | `repair_budget_exhausted` | Repeated missing file-resource choices; no task executed. |
| LIVE-MVP-04 direct | 9.709 | 4 / 2 | 4,573 / 281 | `context_build_failed` | The isolation operation repaired wrong-adapter/missing-column actions, then its required current-operation context exceeded the 5,000-token profile. |
| LIVE-MVP-04 conversational | 12.062 | 5 / 6 | 6,630 / 520 | `context_build_failed` | The isolation operation completed; the primary operation accepted four query results, then failed building its next context. |
| LIVE-MVP-04 ambiguous | 8.860 | 4 / 2 | 4,610 / 298 | `context_build_failed` | Same isolation-path wrong-adapter/missing-column repairs followed by context construction failure. |

LLM-G00 now passes for the explicit OpenAI/`gpt-4.1-mini`/CPython 3.11.15
reference. LLM-G01 retains its deterministic PASS, with a newly observed
reporting gap: every failed test warned that `record_property` is incompatible
with xUnit2 and the assertions stopped before properties were recorded.
LLM-G02 fails. The completed read-only root-cause review supersedes the
provisional model-only classification. Direct product-contract causes include
newest-evidence context truncation (MVP-01 conversational) and false-positive
CTE validation (MVP-02 conversational); source/tool/catalog/repair defects
amplified most other model mistakes. Genuine model errors remain. All three
MVP-04 rows are primarily test-fixture failures caused by the nonviable
5,000-token profile and did not reach compression. Latent evaluator defects
would also reject some valid recovered paths. No provider-adapter execution or
infrastructure failure was observed. Exact row attribution, numeric context
overflow reconstruction, latent blockers, and the owner-aligned P0/P1 proof
plan are recorded in
`docs/LIVE_MVP_WAVE1_FAILURE_ANALYSIS_2026-07-20.md`. No production/test repair,
prompt change, budget change, or live rerun occurred during analysis.

### Wave 1 repair — consolidated pre-live gate

Status: **PRE-LIVE PASS; LLM-G01 PASS; LLM-G02 remains FAIL until the one
authorized repaired real-provider run passes all 12 rows.** The immutable first
live execution and failure analysis above are unchanged. No paid inference was
used during repair or this gate.

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| W1-RQ01 | `next/`, CPython 3.11.15 | The named repair-owner selection shown below, covering versioned capability/model/config/storage contracts, migration/corruption/reopen, observations/context/session, catalog/routing/SQL/comparison, providers/request policy, runtime repair/budgets, and `tests/live/mvp` with live rows deselected | PASS — 442 tests in 46.054s; JUnit `/private/tmp/daita-wave1-repair-focused-final.xml`. |
| W1-RQ02 | `next/`, CPython 3.11.15 | `PYTHONPATH=/Users/jendala/daita/daita-agents/next/src:/Users/jendala/daita/daita-agents/.venv/lib/python3.11/site-packages /Users/jendala/daita/daita-agents/.venv/bin/python -S -m pytest -o addopts='' tests/live/mvp -m 'not requires_llm' -q -rs -p no:cacheprovider --junitxml=/private/tmp/daita-wave1-harness-final-py311.xml -o junit_family=xunit1` | PASS — 32 passed and 12 live rows deselected in 1.15s. The fixture digest, three unchanged natural variants for each scenario, exact 12-row collection, full comparison kinds, exact runtime correlations, graph provenance, failure-path metrics, JSON/JUnit output, and redaction checks pass. |
| W1-RQ03 | `next/`, CPython 3.11.15 and 3.12.7 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/jendala/daita/daita-agents/next/src:/Users/jendala/daita/daita-agents/.venv/lib/python3.11/site-packages /Users/jendala/daita/daita-agents/.venv/bin/python -S -m pytest -o addopts='' tests/ -m 'not requires_llm and not requires_db' -q -rs --tb=short -p no:cacheprovider --junitxml=/private/tmp/daita-wave1-repair-final-py311.xml -o junit_family=xunit1`; repeat with `PYTHONPATH=/Users/jendala/daita/daita-agents/next/src:/opt/homebrew/Caskroom/miniforge/base/lib/python3.12/site-packages /opt/homebrew/Caskroom/miniforge/base/bin/python3.12` and JUnit `daita-wave1-repair-final-py312.xml` | PASS — 1,813 passed, two sandbox-only Unix-socket skips, and 21 live/database deselections on each interpreter in 116.77s/131.76s. |
| W1-RQ04 | `next/` | `../.venv/bin/black --check src tests scripts examples`; `../.venv/bin/python -m compileall -q src tests scripts examples`; `MYPYPATH=src ../.venv/bin/python -m mypy src/daita tests scripts/build_test_disposition.py scripts/verify_candidate_lifecycle.py`; `npx --yes pyright@1.1.411 --pythonpath ../.venv/bin/python` | PASS — Black leaves 303 files unchanged; compilation exits zero; mypy is clean over 291 files; Pyright reports 0 errors and 0 warnings. |
| W1-RQ05 | `next/`, isolated CPython 3.11.15 | `PYTHONPATH=/Users/jendala/daita/daita-agents/next/src:/Users/jendala/daita/daita-agents/.venv/lib/python3.11/site-packages /Users/jendala/daita/daita-agents/.venv/bin/python -S -m pytest -o addopts='' tests/architecture -q -p no:cacheprovider --junitxml=/private/tmp/daita-wave1-architecture-final.xml -o junit_family=xunit1` | PASS — 95 passed in 11.70s. |
| W1-RQ06 | repository root and `next/` | `../.venv/bin/python scripts/build_test_disposition.py --check`; isolated `next/scripts/capture_v1_oracles.py --check`; `git diff --quiet b87df31873d33fffbf50498f5dc4d8892115e8f8 -- daita tests pyproject.toml`; root safe pytest selection; `git diff --check` | PASS — generated disposition and all four frozen-v1 fixtures reproduce; root-v1 paths have no status/diff; root reports 2,498 passed and 221 deselected in 11.04s; diff check is clean. |
| W1-RQ07 | repository root | `PYTHONPATH=next/src .venv/bin/python -m pytest -o addopts='' next/tests/contract/packaging -q -p no:cacheprovider --junitxml=/private/tmp/daita-wave1-packaging-final.xml -o junit_family=xunit1` | PASS — 26 packaging contracts in 3.41s. |
| W1-RQ08 | `next/`, clean temporary copies | `../.venv/bin/python scripts/verify_candidate_lifecycle.py --python /opt/homebrew/bin/python3.11 --python /opt/homebrew/Caskroom/miniforge/base/bin/python3.12`; repeat with `--phase-9-5-joined` | PASS — clean 106-entry wheel/133-entry sdist; dependency-free and joined OpenAI/SQLite installed lifecycles succeed on Python 3.11.15/3.12.7, including local host inference, cold reopen without model injection, uninstall, and retained state. |
| W1-RQ09 | repository root after pre-live ledger update | `PYTHONPATH=next/src .venv/bin/python -m pytest -o addopts='' next/tests/architecture/test_release_documentation.py next/tests/architecture/test_parity_matrix.py -q -p no:cacheprovider`; `git diff --check` | PASS — 13 documentation/parity contracts in 0.21s and a clean diff check. |

The W1-RQ01 command was:

```text
PYTHONPATH=src ../.venv/bin/python -m pytest -o addopts='' tests/unit/test_capabilities.py tests/unit/context/test_models.py tests/unit/context/test_budgeting.py tests/unit/context/test_session_compression.py tests/unit/catalog/test_protocols.py tests/unit/catalog/test_sqlite_catalog_service.py tests/unit/storage/test_sqlite_catalog_store.py tests/unit/storage/test_sqlite_codec_corruption.py tests/contract/storage/test_sqlite_operation_store.py tests/contract/storage/test_sqlite_task_migration.py tests/contract/storage/test_sqlite_approval_projection.py tests/contract/config/test_runtime_configuration.py tests/unit/domains/data/test_phase4_capabilities.py tests/unit/domains/data/test_controller_context.py tests/unit/domains/data/test_phase4_controller.py tests/unit/domains/data/test_postgresql_controller.py tests/unit/domains/data/test_sql.py tests/unit/domains/data/test_sql_derived_relations.py tests/unit/domains/data/test_tabular_comparison_policy.py tests/unit/domains/data/test_tool_applicability_declarations.py tests/acceptance/test_local_file_comparison_journey.py tests/acceptance/test_session_compression_journey.py tests/unit/operations/test_runtime_budgets.py tests/unit/operations/test_runtime_repairs.py tests/unit/loop/test_context_tool_boundary.py tests/unit/llm/test_llm_models.py tests/unit/llm/test_openai_provider.py tests/unit/llm/test_openai_compatible_provider.py tests/unit/llm/test_anthropic_provider.py tests/unit/llm/test_gemini_provider.py tests/unit/llm/test_routing.py tests/live/mvp -m 'not requires_llm and not requires_db' -q -p no:cacheprovider --junitxml=/private/tmp/daita-wave1-repair-focused-final.xml -o junit_family=xunit1
```

Pre-live red/repair evidence is retained rather than rewritten: the first broad
repair run found the stale parity anchor and offline example request-policy
fixture (1,801 passed, two skipped, 21 deselected, two failed); focused repairs
passed. The first final dual-interpreter run then found one stale
threshold-as-retention-floor expectation in the approval-projection contract
(1,812 passed, two skipped, 21 deselected, one failed on each interpreter).
The corrected approval projection passed on both interpreters, and the complete
W1-RQ03 rerun above is green. No failed live row was executed or rerun during
these deterministic repairs.

### Wave 1 repair — one authorized live execution

Status: **LLM-G01 PASS; LLM-G02 FAIL.** All pre-live gates above were green,
so the unchanged 12-row corpus ran once. It produced 12 failures and zero
passes. The artifacts were preserved and reviewed read-only; no failed row was
rerun and no production/evaluator repair followed the result.

| ID | Working directory | Exact command/scope | Result |
| --- | --- | --- | --- |
| W1-RQ10 | `next/`, CPython 3.11.15, credential loaded from repository `.env` without printing it | `set -a; source /Users/jendala/daita/daita-agents/.env; set +a; DAITA_RUN_LIVE_LLM=1 DAITA_RUN_LIVE_MVP=1 DAITA_LIVE_MVP_PROVIDER=openai DAITA_LIVE_MVP_MODEL=gpt-4.1-mini DAITA_LIVE_MVP_JSON_SIDECAR=/private/tmp/daita-wave1-repair-live.XWPkkC/wave1.json PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/jendala/daita/daita-agents/next/src:/Users/jendala/daita/daita-agents/.venv/lib/python3.11/site-packages /Users/jendala/daita/daita-agents/.venv/bin/python -S -m pytest -o addopts='' tests/live/mvp/test_data_journeys_live.py tests/live/mvp/test_sessions_live.py -m 'requires_llm and acceptance' -q -rs -s -p no:cacheprovider --basetemp=/private/tmp/daita-wave1-repair-live.XWPkkC/homes --junitxml=/private/tmp/daita-wave1-repair-live.XWPkkC/wave1.xml -o junit_family=xunit1 -o log_file=/private/tmp/daita-wave1-repair-live.XWPkkC/wave1.log -o log_file_level=INFO` | FAIL — 12 failed, zero passed/skipped in 746.25s. All 105 real OpenAI `gpt-4.1-mini` calls completed; 87 tool calls, 51 tasks, 50 evidence records, 44 repairs, 497,195 input plus 19,204 output tokens, 420,935ms provider latency, zero retries, and zero fallbacks. This is the sole repaired live run. |
| W1-RQ11 | `/private/tmp/daita-wave1-repair-live.XWPkkC`, read-only retained sidecar/JUnit/log/homes | Harness session-finish configured-secret/session-sentinel scan; sidecar/JUnit structural review; per-operation SQLite and comparison-artifact review | PASS for artifact integrity/redaction — sidecar schema 1 has exactly 12 rows; JUnit has 12 failures/zero errors; the log records successful provider responses only; configured credentials and complete generated session sentinels are absent from the sidecar, JUnit, log, and retained homes. The artifacts remain private and unchanged. |
| W1-RQ12 | `next/`, after failure-ledger updates only | `PYTHONPATH=src ../.venv/bin/python -m pytest -o addopts='' tests/architecture/test_release_documentation.py tests/architecture/test_parity_matrix.py -q -p no:cacheprovider`; `git diff --check` | PASS — 13 documentation/parity contracts in 0.21s and a clean diff check. No behavioral or live test was rerun. |

Overall recorded metrics are 15 operations, 105 model calls, 87 tool calls,
51 actions/tasks, 50 accepted evidence records, 36 rejected actions, 44
repairs, 497,195 input tokens, 19,204 output tokens, 516,399 total tokens,
420,935ms provider latency, and 743.346086s summed row wall time. Selected
context was 432,240 tokens; omitted context, source/projection truncation,
cancelled tasks, retries, and fallbacks were all zero. Two harmless duplicate
reads were recorded. The normalized estimated-cost field was zero and is not
reported as an actual price estimate.

| Scenario / variant | Calls / tasks / evidence | Repairs / rejected | Input / output tokens | Wall / provider latency | Terminal and decisive failure |
| --- | ---: | ---: | ---: | ---: | --- |
| LIVE-MVP-01 direct | 2 / 1 / 0 | 1 / 1 | 3,399 / 453 | 9.609103s / 9,396ms | `action_processing_failed`; invented SQL was typed-rejected, then a model-corrupted resource ID became fatal generic `executor_failed` instead of repair evidence (mixed model/framework). |
| LIVE-MVP-01 conversational | 10 / 5 / 5 | 5 / 4 | 62,261 / 2,589 | 99.805195s / 52,147ms | `repair_budget_exhausted`; cited-answer repair followed accepted but semantically wrong `$195.23` query evidence (model). |
| LIVE-MVP-01 ambiguous | 8 / 4 / 4 | 3 / 3 | 40,166 / 1,566 | 46.102159s / 31,199ms | Completed, but used the forbidden archive and returned `0`/`NULL`; required current-resource inspection and hard `4`/`$162.00` oracle fail (model). |
| LIVE-MVP-02 direct | 10 / 7 / 7 | 2 / 2 | 56,599 / 1,833 | 109.622030s / 42,961ms | Completed `Europe`/`$50.00` without accepted graph traversal; required `$125.00` oracle fails (model). |
| LIVE-MVP-02 conversational | 10 / 5 / 5 | 4 / 4 | 62,611 / 2,856 | 102.489737s / 55,188ms | Completed from empty rows after wrong status semantics, without graph traversal; hard oracle fails (model). |
| LIVE-MVP-02 ambiguous | 8 / 3 / 3 | 5 / 5 | 43,565 / 852 | 56.099057s / 22,791ms | `repair_budget_exhausted`; repeated schema guessing produced no accepted query or traversal evidence (model; framework failed closed). |
| LIVE-MVP-03 direct | 6 / 3 / 3 | 2 / 1 | 21,852 / 1,082 | 27.998157s / 20,770ms | Exact five-discrepancy result, but zero accepted inspections meant “newest” was unproven (model provenance failure). |
| LIVE-MVP-03 conversational | 6 / 3 / 3 | 2 / 1 | 23,001 / 998 | 30.599051s / 22,012ms | Exact five-discrepancy result after typed key-policy repair, but zero freshness inspection (model provenance failure). |
| LIVE-MVP-03 ambiguous | 6 / 3 / 3 | 2 / 0 | 22,260 / 1,067 | 26.388884s / 19,515ms | Chose the older file and queried only IDs 1–3; comparison scope and freshness oracle fail (model). |
| LIVE-MVP-04 direct | 11 / 5 / 5 | 5 / 5 | 35,161 / 1,390 | 43.826481s / 29,615ms | Isolation succeeded; primary hit `no_progress_action_failure_limit` after invalid SQL (model; framework failed closed). |
| LIVE-MVP-04 conversational | 15 / 6 / 6 | 7 / 6 | 67,147 / 2,667 | 115.311781s / 63,753ms | Both operations completed, but primary cited `0`/`NULL` instead of `4`/`$162.00`; hard evidence oracle correctly failed (model). |
| LIVE-MVP-04 ambiguous | 13 / 6 / 6 | 6 / 4 | 59,173 / 1,851 | 75.494452s / 51,588ms | Isolation succeeded; primary omitted active-customer semantics, found `5`/`$174.99`, then exhausted citation repair (model). |

All three MVP-04 isolation sentinels remained outside primary-session model
requests. No MVP-04 row reached follow-up history, compression, or cold reopen,
so those live contracts remain unexercised. A latent evaluator risk was found
after MVP-03: discrepancy prose split across adjacent lines may be rejected by
the newline-local identity check. It did not cause these failures because all
three rows first lacked authoritative file freshness inspection. Requiring
current-table inspection may also be stronger than necessary but was likewise
not decisive. LLM-G02 therefore remains FAIL; later waves and Phase 10 are not
authorized by this record.

### Wave 1 — outcome-first MVP benchmark

Status: **LLM-G01 PASS; LLM-G02 FAIL.** By explicit user direction, the live
rows were reordered and narrowed to an MVP contract: exact business outcome,
read-only safety, and accepted evidence supporting the answer are hard layers.
Prescribed catalog/tool choreography, graph depth, duplicate reads, and other
pre-cutover robustness checks are diagnostic or separately marked
`live_precutover`.

| ID | Scope | Result |
| --- | --- | --- |
| W1-MB01 | `tests/live/mvp/test_evaluator_contracts.py` | PASS — 27 tests in 2.59s. |
| W1-MB02 | `tests/live/mvp -m 'not requires_llm'` | PASS — 37 passed and 15 deselected in 2.88s. |
| W1-MB03 | Exact `live_mvp` collection | PASS — 12 rows: LIVE-MVP-01 through 04 direct, then conversational, then answerable-ambiguous; three `live_precutover` rows excluded. |
| W1-MB04 | Complete deterministic suite on CPython 3.11.15 and 3.12.7 | PASS — 1,818 passed, two skipped, and 24 deselected in 137.10s / 151.40s. |
| W1-MB05 | Black, byte compilation, mypy, Pyright, architecture, and `git diff --check` | PASS — 303 files formatted; compilation clean; mypy clean over 289 files; Pyright 0 errors/warnings; 95 architecture tests pass. |
| W1-MB06 | One ordered OpenAI `gpt-4.1-mini` execution with exact `live_mvp` selector, CPython 3.11.15 | FAIL — 12 failed and three pre-cutover rows deselected in 1,341.85s. Artifacts: `/private/tmp/daita-wave1-mvp-benchmark.dzwuoC/{wave1.json,wave1.xml,wave1.log,homes}`. No row was rerun. |
| W1-MB07 | Failure-safe sidecar and prohibited-value scans | PASS — schema-1 sidecar contains all 12 rows; the configured credential is absent from retained homes/reports/logs; no complete generated session sentinel appears in JSON, JUnit, or log. |

Aggregate live metrics: 165 provider calls, 142 tool calls, 82 actions, 71
repairs, 60 rejected actions, 871,460 input plus 27,450 output tokens (898,910
total), 683,768ms provider latency, 1,335.236926s summed row wall time, six
duplicate reads, zero retries, zero fallbacks, zero omitted context tokens, and
zero observation/evidence truncations. OpenAI completed every provider request.

| Order | Row | Hard-layer result | Decisive observation |
| ---: | --- | --- | --- |
| 1 | LIVE-MVP-01 direct | outcome/safety/evidence fail | Five typed `data.sql.missing_column` rejections exhausted repair budget; no query evidence. |
| 2 | LIVE-MVP-02 direct | outcome fail; safety pass; evidence unavailable | Four missing-column rejections and repeated tool use exhausted the turn budget. |
| 3 | LIVE-MVP-03 direct | outcome fail; safety/evidence unavailable | Comparison bounds/type and read-statement rejections exhausted repair budget. |
| 4 | LIVE-MVP-04 direct | outcome/evidence fail; safety pass | Initial and follow-up answers were semantically wrong; post-reopen repair budget exhausted. |
| 5 | LIVE-MVP-01 conversational | outcome/evidence fail; safety pass | Completed with `0`/NULL rather than `4`/`$162.00`; cited evidence supports the wrong result. |
| 6 | LIVE-MVP-02 conversational | outcome/safety/evidence fail | Missing-column and adapter rejections exhausted repair budget without query evidence. |
| 7 | LIVE-MVP-03 conversational | outcome fail; safety/evidence pass | Exact five-discrepancy result; natural “present only in” wording exposed a remaining evaluator prose-parser false negative. |
| 8 | LIVE-MVP-04 conversational | outcome/safety/evidence fail | Initial/follow-up operations exhausted repair budget; post-reopen completed with the wrong `$0.00` result. |
| 9 | LIVE-MVP-01 answerable-ambiguous | outcome/evidence fail; safety pass | Completed with `3`/`$87.00` instead of `4`/`$162.00`. |
| 10 | LIVE-MVP-02 answerable-ambiguous | outcome fail; safety pass; evidence unavailable | Missing-column repairs and repeated tool use exhausted the turn budget. |
| 11 | LIVE-MVP-03 answerable-ambiguous | outcome fail; safety/evidence unavailable | Repeated incomplete response-contract repairs exhausted repair budget. |
| 12 | LIVE-MVP-04 answerable-ambiguous | outcome/evidence fail; safety pass | All operations completed, but initial, filtered follow-up, and post-reopen answers were incomplete or wrong. |

The removed topology/choreography assertions did not cause this red result:
every row independently failed the MVP outcome layer. Attribution is not
uniform. Row 7 is evaluator-owned; repeated schema guessing and incorrect
business results are model/trajectory failures with the framework failing
closed; comparison/read rejection behavior and the high repair volume require
separate offline review before assigning framework ownership. LLM-G02 remains
FAIL. No post-benchmark code change or paid rerun was performed.
