# Daita v2 Replacement Status

This file is the persistent execution ledger for the isolated replacement
project. Update it before and after every material task.

## Current position

- **Active phase:** Phase 2 — persistent local loop
- **Active task:** P2-03 — build the SQLite foundation and normalized operation
  lifecycle repository against the proven optimistic contract
- **Last completed task:** P2-02 — canonical checkpoints and authoritative
  async in-memory operation-store seam
- **Current checkpoint:** P2-02 store-seam commit
  `b13e66abc5d645b685f7bbf840d2e8d9ea903f2f`
  (`refactor(v2): add authoritative operation store`)
- **Architecture-plan fingerprint:** ignored local source
  `docs/DAITA_AUTONOMOUS_AGENT_V2_MVP_PLAN.md`, SHA-256
  `403ad8c3030a126375759b57af4ebe767c6066352b2db158488669a28cc3f935`
- **Exact next action:** write expected-red SQLite marker, PRAGMA, ordered
  checksummed migration, backup-before-migrate, compatibility, rollback, and
  reopen contracts before adding the concrete `storage/sqlite.py` owner

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

After the Phase 1 gate commit and before any Phase 2 production edit, Sections
6 and 15 were re-read in full again on 2026-07-16 together with Phase 2 and the
supporting persistence/hosting/provider sections. Phase 2 therefore begins
with these renewed constraints:

- persist current state and its lifecycle event atomically before publishing;
- keep one operation runtime as the executor, lease, approval, evidence, and
  recovery authority;
- resume the same operation from durable checkpoints without re-planning or
  rerunning terminal tasks;
- reject stale lease commits and fail unknown side-effect outcomes closed;
- keep SQLite/blob/hosting/provider implementations outside the generic loop;
- preserve the isolated v2 state root and lazy optional-provider imports; and
- do not pull catalog, memory, monitor, or later-phase domain work into the
  persistent-loop slice.

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
| P1-02 | complete | P1-01 records | Text-only vertical slice with deterministic scripted mock model, static context, copy-on-write in-memory operation/turn/event state, direct generic loop, and readiness commit | 7 focused tests; exact success event order; optional session; model call committed before I/O; stable provider failure with no whole-loop retry; injected commit failure publishes no partial state | P1-01 |
| P1-03 | complete | P1-02 loop; fake capability contract | Minimal capability registry/tool projection, fake read executor, in-memory task/evidence state, and operation runtime submission path | 30 focused tests; exact proposal/projection binding; durable PENDING/RUNNING/success checkpoints; accepted evidence before observation; one/two sequential reads; injected atomic failures; sole executor call site; commit `e5258c0` | P1-02 |
| P1-04 | complete | P1-03 action path | Structured invalid-action observations, bounded repair turns, normalized failure fingerprints, and no-progress termination | 42 focused and 113 complete tests; atomic ordered repair/skip boundaries; commit `91d7376` | P1-03 |
| P1-05 | complete | P1-02 through P1-04 | Cancellation checks plus turn, action, repair, identical-retry, wall-time, task-timeout, token, observation, and estimated-cost budgets | 46 focused and 150 complete tests; adversarial deadline/cancellation suppression; atomic interruption/budget commits; commit `5c87494` | P1-04 |
| P1-06 | complete | Complete loop laboratory | Deterministic scripted acceptance trajectories and architecture assertions covering every Phase 1 gate | 160 tests on Python 3.11/3.12; 26 architecture tests; static/isolation/build gates pass; commit `eb57f9d` | P1-05 |
| P1-07 | complete | Passing P1-06 results | Final Phase 1 ledger/evidence and coherent gate commit | 160 tests per interpreter; 26 architecture tests; committed-tree root/v2 builds isolated; exact gate commit contains this ledger | P1-06 |

## Ordered Phase 2 tasks

| ID | Status | Inputs | Expected output | Tests/evidence | Dependencies |
| --- | --- | --- | --- | --- | --- |
| P2-01 | complete | Re-read Sections 6 and 15; Phase 2 work/gate; Sections 8.1–8.8, 9.1–9.10, and 11.1–11.8; Phase 1 owners | Persistence, recovery, hosting, session, event, approval, and provider ownership inventory plus this ordered test-first plan | Read-only source/plan inventory; no production edit; smallest representative persistence seam selected | Phase 1 gate |
| P2-02 | complete | In-memory operation runtime commit seam; canonical operation/loop/model records | Narrow async optimistic `OperationStore` contract, in-memory implementation, canonical event/model-call/snapshot records, and one representative trigger/operation checkpoint migrated without changing loop semantics | 218 tests on each Python version; canonical linkage/history/cancellation/CAS adversarial regressions; architecture/static/isolation/build gates and independent review pass | P2-01 |
| P2-03 | active | Proven P2-02 seam; Section 11.2 lifecycle inventory | SQLite engine with v2 marker, WAL, foreign keys, busy timeout, correctness-first synchronous mode, checksummed ordered migrations, SQLite-API backup-before-migrate, compatibility rejection, normalized lifecycle tables, and transactional optimistic operation repository | Fresh/reopen/concurrent-CAS/rollback/migration/interruption/future-or-unknown-schema tests; every runtime lifecycle record round-trips independently | P2-02 |
| P2-04 | pending | SQLite transaction boundary; Section 8.4/11.5 contracts | Content-addressed blob store and durable committed-event log/subscription with per-agent monotonic cursors; event notification remains a post-commit wake hint | Temp/flush/hash/atomic-rename/orphan tests; rollback emits nothing; commit/publish gap replays; reconnect, slow subscriber, and cross-agent isolation tests | P2-03 |
| P2-05 | pending | Persisted tasks/evidence/events; Section 8.5 recovery rules | Ready/claimed/running/approval-waiting/cancelled/manual-recovery task states, persisted execution-safety and idempotency facts, durable fenced leases, and a split materialize/claim/execute/commit path preserving the sole executor boundary | Claim race, lease expiry/reclaim, stale-fence rejection with no evidence/event, terminal skip, unknown side-effect outcome, and task/evidence/event atomicity tests | P2-04 |
| P2-06 | pending | Durable checkpoints, tasks, evidence, observations, readiness, and leases | One loop with new-trigger and checkpoint-aware same-operation resume paths, plus startup recovery that reuses completed work and never rebuilds the plan from the original trigger | Crash/cancel injection at all seven Phase 2 boundaries; at-least-once started model call; terminal task skip; recovered response/evidence/observation/readiness paths do not repeat avoidable I/O | P2-05 |
| P2-07 | pending | Recovery runtime and persisted governance facts | Immutable risk/policy/task fingerprints, CAS approval records, and one test-owned durable fake side-effect marker with wait, decision-only approval mutation, wake, denial, same-operation resume, and exact-once idempotency | Pre-approval no-I/O, approval mutation no-I/O, exact-fingerprint resume, denial no-I/O, cancellation race with one winner, repeated/concurrent resume marker unchanged, stale holder blocked | P2-06 |
| P2-08 | pending | SQLite composition, recovery, approvals, committed events | Strict isolated agent home and bootstrap identity manifest, authoritative DB identity, durable sessions/transcripts, shared per-agent writer lock, thin `Agent.create/open/run/inspect/resume`, and embedded composition only | Create/reopen/path/mismatch/concurrency/lock/session isolation/restart tests; default never touches v1 state; facade contains no loop/executor/provider behavior | P2-07 |
| P2-09 | pending | Persisted embedded fake-capability loop; provider-neutral model contract | Separate canonical/provider call identity, normalized provider errors, lazy optional OpenAI Responses adapter, fake-client contract tests, and one live persisted fake-capability loop | Missing-extra/import isolation; response/tool continuation/error tests; architecture scan forbids provider execution; explicit credential/model live gate | P2-08 |
| P2-10 | pending | Complete Phase 2 vertical slice | Consolidated restart, failure-injection, approval, event, import, architecture, static, cross-version, root-oracle, and clean-build evidence with parity/quality/ADR review | Complete Phase 2 gate suite on Python 3.11/3.12 plus live production-provider evidence; no mock substitutes for live evidence | P2-09 |
| P2-11 | pending | Passing P2-10 results | Final Phase 2 ledger/evidence and coherent exact gate commit | Scoped diff/hooks; all paths under `next/`; exact `chore(v2-phase-2): complete phase 2 gate` commit | P2-10 |

### Ordered P2-03 internal tasks

| ID | Status | Smallest output | Required proof before advancing |
| --- | --- | --- | --- |
| P2-03a | active | Test-only SQLite foundation contract | Expected-red marker/PRAGMA/version/checksum/backup/compatibility/rollback/reopen cases with no production SQLite owner |
| P2-03b | pending | One concrete `SQLiteOperationStore` foundation in `storage/sqlite.py` | Foundation cases green; standard-library calls offloaded; no generic `StateStore`, loop import, or SQL leakage into runtime |
| P2-03c | pending | Normalized lifecycle schema plus canonical codecs | Every trigger/operation/loop/turn/model/task/evidence/readiness/observation/event field round-trips independently; no opaque-snapshot-only persistence |
| P2-03d | pending | Transactional optimistic operation repository | Reuse portable contract validation; create/load/by-trigger/CAS/conflict/rollback/reopen/shared-runtime conformance; failure leaves all tables/events unchanged |
| P2-03e | pending | Final P2-03 review and checkpoint | Dual-Python full/static/architecture/oracle/build gates, independent review, scoped hooks, local commit |

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
- P1-02 completed owners: `next/src/daita/llm/providers/mock.py`,
  `next/src/daita/operations/runtime.py`, and `next/src/daita/loop/driver.py`;
  static context/readiness doubles remain test-owned until another real
  implementation requires a production subsystem.
- P1-03 implemented owners: immutable declarations and tool projections in
  `next/src/daita/capabilities.py`, task/evidence records in
  `next/src/daita/operations/models.py`, the sole executor invocation and
  evidence-acceptance boundary in `runtime.py`, and sequential action
  progression in the existing generic driver. The fake controller, context
  builder, and executor remain test-owned.
- P1-04 extends those existing owners rather than adding a parallel repair
  subsystem: bounded rejection/readiness records live with canonical records;
  the generic driver owns repair progression; and the operation runtime owns
  exact-call binding, normalized fingerprints, taskless observations, atomic
  multi-call skips, no-progress facts, and terminal commits.
- P1-05 extends the same immutable budget, generic-driver, and operation-runtime
  owners. Budgets bind to the operation snapshot; the loop alone decides
  progression and terminal meaning; the runtime rechecks the authoritative
  wall deadline immediately before its sole executor call, rejects late
  timeout-suppressed results, and atomically records active-child cancellation
  intent before interruption becomes terminal.
- P1-06 dependency review resolved an ambiguity without ownership churn.
  Section 7 places checkpoint/progression contracts in `loop.models`; Sections
  6.3–6.4 and ADR 0005 require the operation runtime to commit those records;
  and the P1-01 ledger explicitly deferred their atomic pairing to that runtime
  owner. Section 15's ban on operations importing the agent loop therefore
  means the executable driver/orchestration protocols, not the immutable
  canonical checkpoint records. P1-Q04's broad “no operations→loop import” was
  a point-in-time P1-01 result for the rejected `operations.models` edge, not a
  permanent ban on `operations.runtime` consuming `loop.models`. Architecture
  tests now permit only that record import, forbid every operations-to-driver
  edge, and keep `loop.models` implementation-free. Lifecycle decisions remain
  loop-owned; durable commits remain operation-runtime-owned.
- P2-01 keeps `OperationRuntime` as the sole execution, approval, evidence,
  lease, recovery, and lifecycle-transition authority. Its current 1,600-line
  in-memory `_states` map and synchronous publication step are painful because
  they make restart correctness and transactional event pairing impossible.
  The smallest correction is a narrow optimistic operation repository at the
  existing copy-on-write commit seam, first proven on one checkpoint. This is
  not a parallel runtime or unbounded `StateStore`; SQLite remains an adapter,
  and later session/blob/event contracts stay narrow and lifecycle-owned.
- P2-02 first extracts only canonical records that a repository must consume
  without importing the concrete runtime implementation. Tests must catch
  behavior drift in Phase 1 event order, atomic rollback, provider-neutral
  records, sole executor invocation, and generic-loop dependency direction.
  The representative slice is reviewed green before the same seam is applied
  to all lifecycle transitions, as required by the refactoring discipline.
- P2-02 completed its representative review before standardization: 28
  canonical-record/store tests passed first, then the same load/copy/CAS seam
  replaced both runtime-owned state maps across all 17 transition commits.
  standalone tests proved the seam before all 17 transition commits migrated.
  `next/src/daita/events/models.py` owns canonical runtime events;
  `next/src/daita/operations/checkpoints.py` owns model-call and immutable
  operation aggregate records; and `next/src/daita/operations/store.py` owns
  the narrow protocol plus its lock-protected in-memory reference adapter.
- `OperationRuntime` remains the only lifecycle/executor owner. It now injects
  `OperationStore`, performs no retry after a revision loss, and translates a
  lost CAS into an inspectable operation-state conflict. Two runtime instances
  sharing one store see the same checkpoint and cannot claim one trigger twice.
  Commit results expose only the exact newly committed event suffix, but no
  subscriber is introduced before the P2-04 durable event-log call site.
- Canonical `RuntimeEvent.operation_id` is nullable because Section 8.4 also
  names operationless agent lifecycle events. An `OperationSnapshot` requires
  every contained event to match its operation and session exactly, binds turn
  pointers to model calls from the same turn, scopes task/observation/event tool
  calls to that turn's committed response, and enforces symmetric task,
  evidence, and evidence-backed-observation ownership. The in-memory store also
  preserves every committed lifecycle-record prefix, not only the event log.
  Readiness still lacks independent operation/turn identity and observations
  lack stable row IDs; P2-03 must add persistence envelopes/codecs before
  claiming their normalized SQLite round-trip rather than weakening semantic
  records now.
- Phase 2 persistence uses normalized lifecycle tables with explicit JSON
  record payloads where useful; it never stores one opaque operation snapshot
  as the only durable truth. Standard-library `sqlite3` is offloaded from the
  event loop, so minimal v2 gains no mandatory SQLite SDK dependency.
- Durable event consumers read from the committed log after a cursor. An
  in-process notification can reduce latency but is only a wake hint, which
  closes the commit-before-publish crash gap and preserves post-commit-only
  delivery.
- Phase 2 implements embedded single-writer execution only. `AgentHost`, local
  socket/API routing, inbox/scheduler/daemon/CLI lifecycle, background
  autonomy, session summaries, provider-token streaming, full routing and
  fallback policy, additional providers, and independent-operation concurrency
  are explicitly deferred to their owning later phases.
- Later owners are created only when their vertical slice becomes active; the
  target tree is not being scaffolded in advance.
- Root `daita/`: **not being changed**

P2-03 extends the existing `OperationStore` contract owner rather than adding
another persistence facade. The current pain is that its proven in-memory
adapter cannot survive restart or atomically normalize lifecycle records. The
smallest correction is one concrete standard-library SQLite adapter in the
plan-owned `storage/sqlite.py`, beginning with its connection/migration
boundary before adding codecs or repository writes. Portable checkpoint
validation remains owned by `operations/store.py`; the adapter may consume it
but may not duplicate runtime transition legality. Foundation tests catch
wrong databases, unsafe PRAGMAs, drifted or future schemas, missing backups,
partial migrations, event/state partial writes, and reopen drift. Repository
tests then reuse the existing optimistic conformance cases and add independent
normalized-row round-trips. This creates no unbounded `StateStore` and imports
no SQLite code into the generic loop or operation runtime.

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
| Initial P1-02 focused test run before the mock/runtime/driver modules existed | EXPECTED RED — 2 collection errors for the absent provider package and loop/runtime owners |
| Final P1-02 focused mock/runtime/text-loop tests | PASS — 7 passed in 0.03s |
| Final P1-02 full isolated suite on CPython 3.11.15 and 3.12.7 | PASS — 56 passed in 0.20s on 3.11; 56 passed in 0.24s on 3.12 |
| P1-02 Black, compile, mypy, pyright, architecture, and root-oracle checks | PASS — 27 Python files formatted; compile succeeded; mypy found no issues in 25 files; pyright 0 errors/warnings; architecture tests included in the full suite; root unchanged |
| P1-02 clean-copy sdist/wheel build and content scan | PASS — build succeeded; 17-entry wheel included the mock provider, generic driver, and operation runtime with no tests or nested `next/` path |
| Initial P1-03 focused run before the capability owner existed | EXPECTED RED — 2 collection errors for the intentionally absent `daita.capabilities` module |
| Final P1-03 focused capability/runtime/text/fake-read suite | PASS — 30 passed in 0.07s; includes forged proposal/projection binding, dynamic tool visibility, executor identity drift, response-scoped call IDs, rejected evidence, sequential one/two-read trajectories, atomic checkpoints, and the sole `.execute()` call-site assertion |
| Final P1-03 complete isolated v2 suite on CPython 3.11.15 and 3.12.7 | PASS — 81 passed in 0.23s on each interpreter |
| P1-03 Black, compile, mypy, and pyright checks | PASS — 32 files formatted; compilation succeeded; mypy found no issues in 31 source files; pyright exited zero with no diagnostics (only npm's nonfatal `unsafe-perm` warning) |
| P1-03 clean-copy sdist/wheel build and archive scan | PASS — built under `/private/tmp/daita-v2-p1-03.kp5giU`; 18-entry wheel contains the capability, generic loop, and operation-runtime modules and contains no tests or nested `next/` path |
| P1-03 architecture/scoped review | PASS — independent final review found no blocker; exact committed/domain tool binding, sequential task/evidence/observation ordering, atomic injected-failure states, sole executor call site, no v1 import/symlink/root-oracle change, and `git diff --check` all passed |
| Initial P1-04 record tests before repair contracts existed | EXPECTED RED — 2 collection errors for the intentionally absent `ActionRejection` and `LoopBudgets` records |
| Initial P1-04 scripted action trajectories before loop repair progression | EXPECTED RED — 2 failures because `AgentLoop` did not yet accept repair/no-progress budgets |
| Final P1-04 focused record/runtime/acceptance suite | PASS — 42 tests cover bounded records, action/readiness correction, changed-action recovery, normalized early stop, complete multi-call skip transcripts, exact call binding, thresholds, and atomic commits |
| Final complete isolated v2 suites on CPython 3.11.15 and 3.12.7 | PASS — 113 passed in 0.35s on 3.11; 113 passed in 0.37s on 3.12 |
| P1-04 Black, compile, mypy, and pyright checks | PASS — Black reported 34 files unchanged; compilation succeeded; mypy found no issues in 33 source files; pyright reported 0 errors/warnings |
| P1-04 independent review, architecture/root/scoped scans, and clean-copy build | PASS — no blocker or root/v1/symlink leak; 19 architecture tests; sole production executor call remains in `operations/runtime.py`; 18-entry wheel and 31-entry sdist exclude tests/nested `next/` paths |
| P1-04 scoped stage and configured pre-commit hooks | PASS — exactly 10 reviewed paths, all under `next/`; cached diff check clean; trailing-whitespace, end-of-file, merge-conflict, large-file, and Black hooks passed |
| Initial P1-05 loop-budget model run | EXPECTED RED — 10 failures and 8 passes identified the absent comprehensive budget fields and validation |
| Final P1-05 focused budget/cancellation/runtime suite | PASS — 46 passed; covers exact N/N+1 limits, response-usage commits, observation overrun, wall/task deadline precedence, cancellation at model/executor/post-evidence boundaries, repeated cancellation, atomic failures, and suppression-resistant deadlines |
| P1-05 adversarial deadline run before root-cause repair | EXPECTED RED — exactly 3 failures proved late provider/executor results were accepted and task-persistence time could cross the wall deadline before I/O |
| P1-05 external-cancellation suppression run before repair | EXPECTED RED — 2 failures proved provider/executor adapters could swallow the caller's cancellation |
| Final P1-05 complete isolated v2 suites on CPython 3.11.15 and 3.12.7 | PASS — 150 passed on each interpreter |
| P1-05 Black, compile, mypy, and pyright checks | PASS — Black reported 39 files unchanged; compilation succeeded; mypy found no issues in 38 source files; pyright 1.1.411 reported 0 errors/warnings |
| P1-05 independent review, architecture/root/scoped scans, and clean-copy build | PASS — no blocker or root/v1/symlink leak; 19 architecture tests; sole production executor call is `operations/runtime.py:830`; root oracle unchanged; 18-entry wheel and 31-entry sdist exclude tests/nested `next/` paths |
| P1-05 scoped stage and configured pre-commit hooks | PASS — exactly 12 reviewed paths, all under `next/`; cached diff check clean; trailing-whitespace, end-of-file, merge-conflict, large-file, and Black hooks passed |
| P1-06 focused gate-consolidation modules | PASS — 46 passed; complete repaired-action events/correlations, all accepted MVP triggers, reserved-event atomicity, dynamic import firewall, one loop, driver import allowlist, dependency leaf, branchlessness, and sole executor boundary |
| Final P1-06 complete isolated v2 suites on CPython 3.11.15 and 3.12.7 | PASS — final checkpoint-tree rerun: 160 passed in 0.57s on 3.11; 160 passed in 0.52s on 3.12 |
| P1-06 Black, compile, mypy, and pyright checks | PASS — Black reported 40 files unchanged; compilation succeeded; mypy found no issues in 39 source files; pyright 1.1.411 reported 0 errors/warnings |
| P1-06 architecture/root/isolation scans | PASS — 26 architecture tests; generated disposition and v1 fixtures reproduce; no root-oracle change, v2 symlink, or diff error |
| P1-06 clean-copy build and fresh-install smoke | PASS — `/private/tmp/daita-v2-p1-06.VGBQWx`; 18-entry wheel and 31-entry sdist exclude forbidden paths; fresh Python 3.11/3.12 installs import v2 `2.0.0a0` from their own `site-packages` |
| P1-06 scoped stage, hooks, and checkpoint | PASS — exactly 9 reviewed paths, all under `next/`; cached diff and configured hooks passed; commit `eb57f9d76dc33c840410529672f5381f33b4423b` |
| P1-07 final complete suites and static gate | PASS — 160 tests in 0.52s on Python 3.11 and 160 in 0.48s on Python 3.12; Black 40 files; mypy 39 files; pyright 0 errors/warnings |
| P1-07 architecture/oracle/range gate | PASS — 26 architecture tests; disposition and v1 fixtures reproduce; all committed Phase 1 paths are under `next/`; root oracle and symlink scans clean |
| P1-07 committed-tree distribution gate | PASS — v2 build `/private/tmp/daita-v2-p1-gate.Y9lKPa` produced 18/31-entry archives at `2.0.0a0`; root build `/private/tmp/daita-root-p1-gate.Bbhplk` produced 401/442-entry archives at `1.0.0`; neither crossed the isolation boundary |
| P1-07 scoped ledger stage and configured hooks | PASS — exactly `next/STATUS.md` and `next/QUALITY_GATES.md`; cached diff and all configured hooks passed before the exact gate commit |
| Initial P2-02 canonical-record/store/architecture run before production owners existed | EXPECTED RED — collection stopped on exactly 3 missing-module errors for `daita.events` and `daita.operations.store`; no production module existed or was edited |
| Initial P2-02 runtime-store seam run after the standalone store contract passed | EXPECTED RED — 6 tests failed because `OperationRuntime` did not yet accept an injected store; this locks the authoritative shared-store transition before the runtime refactor |
| P2-02 standalone canonical-record and optimistic-store contract | PASS — 28 passed after the initial missing-owner red; immutable records, atomic identity claims, exact event suffixes, stale/concurrent CAS, append-only event history, and rollback behavior are covered |
| P2-02 focused canonical/store/runtime/architecture integration | PASS — 53 passed; injected create and revision commits, failed-commit rollback, conflict visibility, shared-store authority, trigger uniqueness, canonical re-exports, sole runtime/executor ownership, and no SQL leakage are covered |
| P2-02 complete isolated suite on CPython 3.11.15 and 3.12.7 | PASS — 204 passed in 1.00s on 3.11 and 204 passed in 1.07s on 3.12 |
| P2-02 formatting, compilation, mypy, and pyright | PASS — Black clean across 49 files; compilation succeeded; mypy clean across 48 files after one test annotation repair; pyright 1.1.411 reported 0 errors/warnings |
| P2-02 architecture/oracle/isolation review | PASS — 34 architecture tests; disposition and v1 fixtures reproduce; no v1 import, root-oracle change, v2 symlink, or SQL/storage leakage in loop/runtime |
| P2-02 clean-copy build and fresh-install smoke | PASS — `/private/tmp/daita-v2-p2-02.zm30hd`; 22-entry wheel and 36-entry sdist contain canonical event/checkpoint/store modules and no tests/nested `next/`; fresh Python 3.11/3.12 installs import v2 `2.0.0a0` from their own site-packages |
| P2-02 adversarial structural/history contracts before repair | EXPECTED RED — 6 failed and 17 passed; cross-turn model pointers, response-owned call linkage, symmetric task/evidence ownership, and immutable committed lifecycle history were not yet enforced |
| P2-02 repaired structural/history contracts | PASS — 23 passed; invalid linkage and committed-record rewrite/removal are rejected without changing authoritative state or events |
| P2-02 cancellation-after-create and interruption-CAS contracts before repair | EXPECTED RED — 2 failed; cancellation could leave a durably created operation running and an interruption lost one revision race |
| P2-02 repaired focused cancellation-store contracts on CPython 3.11/3.12 | PASS — 2 passed on each interpreter; a durable create converges to interruption and interruption alone deliberately reloads after a lost CAS |
| P2-02 downstream model-call and terminal-cleanup contracts before repair | EXPECTED RED — 9 correlation trajectories and 1 terminal-race cleanup case failed; downstream readiness/action/task/evidence/observation events lacked canonical call IDs and the driver's conflict handler was unreachable |
| P2-02 aggregate model/event ancestry contracts before repair | EXPECTED RED — focused cases accepted an erased event call ID, an unowned model call, and child event IDs with missing or contradictory turn/model/task ancestry |
| P2-02 cancellation-versus-delayed-CAS contract before repair | EXPECTED RED — 1 integration case raised `OperationStateError` and left the externally advanced operation running instead of preserving cancellation and committing interruption |
| P2-02 final complete isolated suites | PASS — 218 passed in 1.24s on CPython 3.11.15 and 218 passed in 1.26s on CPython 3.12.7 |
| P2-02 final static/architecture/oracle gate | PASS — Black clean across 50 files; compilation succeeded; mypy clean across 49 files; pyright 0 errors/warnings; 34 architecture tests; disposition/v1 fixtures reproduce; root-oracle, import, symlink, SQL-leakage, and diff scans clean |
| P2-02 final distribution gate | PASS — clean copy `/private/tmp/daita-v2-p2-02-gate.31k4au` built 22-entry wheel and 36-entry sdist; fresh isolated Python 3.11/3.12 installs import v2 canonical event/checkpoint/store modules from their own site-packages |
| P2-02 independent final review | PASS — no architecture or cancellation blocker remains; store/runtime ownership, committed prefixes, exact event ancestry, cancellation precedence, interruption CAS convergence, and sole executor boundary verified |
| P2-02 scoped checkpoint | PASS — exactly 19 paths under `next/`; cached diff and configured whitespace/EOF/conflict/large-file/Black hooks passed; commit `b13e66abc5d645b685f7bbf840d2e8d9ea903f2f` |

Phase 0 and every Phase 1 task are complete. This ledger is committed by the
exact Phase 1 gate commit; Phase 2 begins only after its mandatory architecture
re-read and an updated ordered ledger.

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
- P2-02 is complete; the earlier 204-test/static/build rows remain interim
  evidence and are superseded by the final 218-test and rebuilt-artifact rows.
- P2-02 began test-first. Its combined event/checkpoint/store/architecture run
  stopped at collection on the intentionally absent canonical event and
  operation-store modules. This is expected-red evidence, not a gate failure;
  the repair must implement the narrow owners without weakening the tests.
- The first structured store-error attribute run was expected red: 4 tests
  found that typed not-found, duplicate, trigger-claim, and invalid-checkpoint
  exceptions lacked stable identity/reason attributes. Constructors now retain
  those facts, and all 9 store contracts pass.
- The first P2-02 static gate found one missing annotation in the architecture
  test's class-location accumulator. Adding the precise `dict[str, list[str]]`
  annotation repaired mypy; production code required no ignore or suppression.
- The adversarial history repair initially reused loop variable names across
  heterogeneous record collections; mypy reported 26 inference errors. Precise
  record-specific bindings and one three-call-site prefix helper repaired the
  type gate without a suppression or behavior change.
- Two fresh-install smoke commands were first launched with a repository or
  wrong relative-tool cwd, so they loaded v1 or could not locate the venv.
  Isolated absolute-path reruns imported v2 from both fresh site-packages.
- P1-01 red/repair history: the initial test-first collection failed on the
  intentionally absent modules; duplicate `test_models.py` basenames then
  caused a pytest import collision and were renamed; the first static pass
  exposed formatting and JSON-value type-narrowing errors, which were repaired
  without ignores in production code.
- A pre-checkpoint dependency review found `operations.models` importing loop
  checkpoint types. Trigger, operation, proposal, and observation ownership was
  consolidated under operations; loop phase/state/turn/readiness/exit remained
  under loop; their atomic pairing is deferred to the P1-02 runtime owner.
- P1-02 red/repair history: the first focused run identified the intentionally
  absent modules; the first implementation imported `Observation` from its old
  loop location and failed collection, then imported it from the operations
  owner. The runtime was changed to copy-on-write commits after reviewing
  mid-transition failure atomicity, and an injected event failure now proves
  no partial state/event publication.
- One root-oracle check used a mistyped long form of the P1-01 hash and Git
  rejected it as `bad object`. The checkpoint was read with `git rev-parse`,
  the ledger was corrected to `6b3eea11505799513a85ce8f45542feab2793f73`,
  and the root-oracle diff then passed.
- P1-03 red/repair history: the initial focused run identified the intentionally
  absent capability module. The first combined path run passed 18 tests and
  failed the two-read case because the test executor incorrectly required the
  whole operation to have no prior evidence; the assertion is now scoped to
  the task entering I/O, and 19/19 focused tests pass. The first mypy pass then
  found a tuple-cardinality inference in the test transcript builder; an
  explicit canonical-message tuple annotation repaired it without a
  production suppression.
- The final P1-03 trust-boundary review found and repaired four issues before
  checkpointing: same-name tool definitions now require exact registry
  equality; per-turn domain visibility is bound exactly to the model request;
  post-I/O executor identity drift terminalizes the task without evidence; and
  provider call IDs are scoped to their turn rather than the whole operation.
  Injected failures additionally prove task-start, evidence/task-success, and
  observation commits never publish partial state.
- P1-04 red/repair history: the record and scripted trajectory tests first
  identified the intentionally absent rejection/budget contracts and loop
  repair progression. A parallel runtime test initially passed after production
  landed, then exposed whether fingerprints were permanent history or a
  current no-progress epoch; accepted evidence now coherently clears the epoch
  while committed events retain audit history. The first full regression run
  found a changed helper error message no longer matching the P1-03 defensive
  assertion; restoring explicit "tool call" wording preserved that contract.
- Independent P1-04 review rejected caller-supplied/code-sensitive failure
  hashes, incomplete multi-call rejection transcripts, unbounded readiness
  facts, insufficient current-call no-progress grounding, and a misleading
  count-one terminal reason. The runtime now owns canonical name/argument
  hashes, atomically observes every skipped call, bounds correction facts,
  requires the current rejection event/observation, and uses the truthful
  `no_progress_action_failure_limit` reason. Rejection-first and
  success-then-rejection trajectories prove ordered provider-valid results.
- The first amended static pass found two test-only `.to_dict()` calls lacking
  explicit `FrozenJsonObject` narrowing. Adding the runtime type assertions
  made both mypy and pyright clean without production suppressions.
- P1-05 red/repair history: the initial loop-record run exposed the absent
  budget fields. Parallel budget tests first met concurrent production and
  reported eight API-label expectation mismatches; aligning them to the locked
  inspectable budget facts produced 33/33 passing model/budget tests. The first
  cancellation tests likewise landed after their production surface and were
  green, so later adversarial tests supplied the independent red evidence.
- Independent P1-05 review reproduced three root deadline failures: a provider
  and executor could suppress timeout cancellation and return a late result,
  and task persistence could consume the wall budget before executor I/O. The
  loop/runtime now reject post-timeout results, recompute operation remaining
  time immediately before execution, discard false executor-start state, and
  keep wall versus task terminal reasons distinct. A second red pair proved
  caller cancellation could also be swallowed; explicit pending-cancellation
  checks now preserve interruption precedence.
- The first P1-05 mypy pass exposed two production narrowing issues in reused
  budget loop variables and a nullable final-text closure; distinct variable
  names and an explicitly narrowed final-text binding repaired both without
  ignores. The first pyright invocation omitted the configured venv
  `--pythonpath` and therefore could not resolve pytest; the established
  command immediately passed with zero diagnostics. One collection count and
  the first archive-inspection command also used incorrect paths, failed before
  changing tracked state, and passed after correction.
- The sandbox rejected the first P1-05 stage and pre-commit invocations because
  Git could not create `index.lock`. Approved reruns preserved the exact 12-file
  `next/` scope; staging, cached diff review, and every configured hook passed.
- Two verification invocations used incorrect cwd-relative paths (the first
  clean-copy source and one `rg` scan). Both failed before changing tracked
  state and were immediately rerun with explicit correct paths.
- The first staging and pre-commit invocations could not create Git's
  `index.lock` inside the workspace sandbox. The same explicitly scoped
  `git add` and configured hook run were approved, then completed successfully.
- P1-06 gate review found no production behavior gap. It strengthened the
  existing deterministic trajectory owners and moved the sole-executor scan
  to the architecture suite rather than creating a duplicate acceptance
  harness. The first focused Black check rejected the new architecture file's
  formatting; Black reformatted it, and every focused/full/static gate passed.
- P1-06 also resolved the apparent operations-to-loop wording conflict from
  Section 15 without moving canonical records away from the plan's target
  ownership map. Only `operations.runtime` may consume the implementation-free
  `loop.models` checkpoint contracts; every operation import of the executable
  driver remains forbidden and regression-tested.
- Independent P1-06 checkpoint review caught and repaired a receiver-name
  loophole in the moved sole-executor scan, underinclusive/overbroad identity
  branch detection, and an 8-versus-14 event-count ledger typo. The first
  allowed-contract scanner regression was expected red, then passed after
  direct identity checks became context-sensitive. Final full/static gates were
  rerun on the repaired tree.

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

## Phase 1 gate record

- [x] Text-only response completes through the generic loop.
- [x] Fake read follows the durable task/execution/evidence/observation order.
- [x] Invalid action becomes a bounded observation and repair turn.
- [x] Repeated identical failure stops early with an inspectable reason.
- [x] Cancellation persists an inspectable state.
- [x] Turn, action, repair, time, token, observation, and cost budgets terminate
      honestly.
- [x] The loop contains no domain-specific or provider-specific branch.
- [x] A repository assertion proves only the operation runtime invokes the
      executor.
- [x] Root v1 oracle paths remain unchanged.
- [x] Passing Phase 1 evidence is committed by the gate commit containing this
      ledger with exact message `chore(v2-phase-1): complete phase 1 gate`.
