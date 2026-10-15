# Deterministic count comparison jobs: implementation guide

Status: implementation reference; the architecture below is not implemented or
live-qualified by this document. Creating this guide does not authorize
implementation, paid tests, credential resolution, commits, pushes, or a PR.
Follow the human's subsequent instructions for those actions.

Checkpoint baseline: `b44eeec34c0a8780b29eac8972aab52fd8c5b6b3`, branch
`tool-recovery-and-token-efficiency`. This guide was written after that
checkpoint and is intentionally separate from it.

Read the current root `AGENTS.md`, production owners, executable tests,
`docs/DURABLE_JOBS.md`, `docs/CONTEXT_AND_SCOPE.md`,
`docs/LOCAL_STATE_UPGRADES.md`, and `docs/TESTING.md` before editing. Symbol names
below locate the baseline; current code and tests take precedence if it has
changed. Preserve unrelated work and all prior qualification evidence.

## 1. Objective and boundaries

Execute fully specified durable count work in code through the existing jobs
system. The model interprets a request and proposes selections. Once a valid
plan is admitted, the framework owns execution, accounting, coverage, arithmetic,
and completion. Model text cannot establish a completed comparison.

The guarantee is **an evidence-backed outcome within the admitted bounds**. It
is not guaranteed database availability, a shared snapshot, or guaranteed model
interpretation/correction before admission.

Ordinary questions, quick comparisons, and exploratory reads keep the current
foreground path. They do not automatically create jobs. A job requires an
explicit admitted job request, including the typed caller entry point below.
Do not add a size heuristic that silently converts foreground work into a job.

### Required retention

- One `AgentLoop`, `CapabilityRuntime`, capability registry, catalog, graph-aware
  `JobSupervisor`, and SQL state boundary.
- Existing foreground `data_count_selection`, its framework-owned batching,
  reducer, inspection, and native per-source permits.
- Explicit `data_count_rows` calls, selection references, cursors, repeated
  legitimate reads, and full-schema catalog/SQL workflows.
- Exact current-run transcripts, one ordered result per requested model call,
  frozen authority, current binding validation, sensitivity, receipts, and hard
  budgets. Internal tasks must not manufacture model transcript exchanges.
- Provider transport retries remain separate from model correction. External
  writes and uncertain actions retain their existing no-replay rules.

### Exclusions

Do not add a second runtime, foreground task lifecycle, generic workflow engine,
read-plan DSL, plugin system, verifier, model repair pass, hidden continuation,
post-limit model request, source retry service, or diagnostic database. Do not
broaden native effects, general graph admission, provider routes, source limits,
or snapshot guarantees. No durable diagnostic schema change is included.

## 2. Why the baseline is insufficient

`Agent.run` accepts freeform text and returns `LoopExit`. In
`AgentLoop._finish_terminal_model_response`, normal foreground `STOP` means the
conversation turn completed; it does not prove the requested business task was
accomplished. Preserve this contract.

The baseline whole-selection count already plans bounded reads and calculates
totals in code. However, `GraphAdmissionBuilder.build` creates a `MODEL` worker
even after freezing an exact count plan. That worker must load a tool, reproduce
the plan reference, invoke it, and submit completion. These are unnecessary
model decisions for already specified work.

An authenticated count for one source also does not establish that all sources
intended for a comparison were covered. Authority is a ceiling; required
coverage is a separate obligation that admission must freeze.

The latest baseline evidence showed six successful targeted natural foreground
runs, but only one successful correction among three fault-seeded continuations.
The other two stopped before corrected reads. They remain failed model
admission/correction evidence. This architecture must not relabel them as passes.
Repeated-failure warning wording is a plausible influence, not a demonstrated
cause of the model's internal decision.

## 3. Admission interfaces

### 3.1 Narrow model view

Add `start_count_comparison` as a **ToolView of the existing**
`jobs.graph.start` capability and `jobs.graph.start.executor`, owned by
`GraphAdmissionCapabilityDomain`. Do not add another starter capability,
supervisor, or execution service.

The closed view schema is:

```json
{
  "type": "object",
  "properties": {
    "selection_refs": {
      "type": "array",
      "items": {"type": "string", "pattern": "^sel_[0-9a-f]{16}$"},
      "minItems": 2,
      "maxItems": 16,
      "uniqueItems": true
    },
    "deadline_seconds": {"type": "integer", "minimum": 60, "maximum": 300}
  },
  "required": ["selection_refs"],
  "additionalProperties": false
}
```

Omitted `deadline_seconds` means 300, bounded by current policy. Derive published
ceilings from the same policy/graph values used for execution; the example
numbers are the baseline values, not independent schema constants.

The view accepts no source/resource IDs, `plan_ref`, SQL, grants, arbitrary task
specifications, or model-authored outcome requirements. The selection list
declares the required comparison set; the data owner derives identities.

Keep the existing `start_graph_job` model view's generic call shape unchanged.
Give that view an explicit closed input schema rather than letting it inherit
the enlarged canonical schema. The canonical capability gains two disjoint
forms: the existing generic proposal and the count request above. Models never
receive their combined schema. Branch only after view and canonical validation,
using the validated canonical form, not a tool-name-derived permission.

Search, load receipts, inspection, definition/schema digests, descriptions, and
context must use the exact registered view. Publish the count bounds through
those existing projections from one code-owned limits projection; do not write
independent numeric promises in each surface. Admission returns the frozen
limits as part of its receipt.

### 3.2 Typed caller

Add the thin facade entry point:

```python
async def start_count_comparison(
    self,
    *,
    source_ids: tuple[str, ...],
    conversation_id: str | None = None,
    deadline_seconds: int = 300,
    caller_principal_id: str | None = None,
) -> GraphJob: ...
```

`Agent` validates caller inputs and delegates to `EmbeddedAgent`. An empty tuple
is invalid, never "all". Require 2 through the applicable maximum number of
distinct sources. Reuse current conversation/principal admission, catalog and
permission owners. Do not synthesize transcript references for typed callers.

Model references and typed source IDs resolve through separate authoring
adapters into the same immutable data-owned plan. The typed route does not
bypass current source permissions, source/resource binding, sensitivity,
distribution validation, or job admission governance.

### 3.3 Resolution and atomicity

For model requests, authenticate each reference against the exact persisted
current-run call and earlier successful catalog result. Reuse
`authenticate_count_selection`; do not create a mutable reference registry.
Require one selection per source, complete catalog enumeration, current
revisions, unique resources, and at least one eligible base table. Exclude views.
For typed requests, resolve the same bounded current admitted catalog sets.

Reject duplicate sources even if distinct references were supplied. Reject
partial/paged selections, stale/cross-run/forged references, unsupported adapters,
empty base-table selections, and resources outside current frozen authority.
An empty database with no base tables is outside this initial contract; a
genuinely zero-row table is supported and must count as measured zero.

Resolve and recheck outside graph transactions. `JobOwner.admit` remains the
atomic graph insertion/wake boundary. Recheck the exact prepared plan at ordinary
preflight/execution boundaries. Admission failure creates no job and performs no
source query; it may read local catalog/state. Never silently substitute targets.

The compact `graph.job_started` receipt includes job identity/state,
specification and plan digests, required-source count, planned-batch count, task
count, and frozen limits. It does not claim counts or completion.

### 3.4 Applicability and composition

Compose static count admission independently of model-worker route or estimated
model-cost configuration. It requires a job owner, admitted data backends, and
finite read/work/time/output limits. Generic model jobs keep their existing
route, token, and cost gates.

Expose the count view only to applicable foreground runs, respecting frozen
prepared candidates, files-only mode, scope, and current availability. It is not
a scheduled or graph-task starter. Starting a job must use the existing
submission-effect preflight/recheck/mutation path. Never wait for job completion
while holding that mutation lock or a graph transaction.

## 4. Immutable plan and bounds

Keep plan semantics in `domains/data/row_counts.py`; keep topology/admission in
`jobs/graph/admission.py`. Add a concrete immutable count-plan representation in
those existing owners, not a general planning interface.

```text
CountComparisonPlan:
  kind = "count_comparison"
  required_coverage = "complete_admitted_base_tables"
  include_views = false
  selections[]:
    source_id, source_revision
    exact resource_id/revision pairs
    complete_source_enumeration = true
  batches[]:
    batch_id, source_id, exact resource_ids
  limits:
    batch_width, maximum_batches
    job/read deadlines and output bounds
    work ceilings and finalizer reserve
    automatic_source_retries = 0
  plan_digest
```

Compute the semantic digest from canonical material excluding the digest itself
and generated task IDs. Bind request sensitivity, principal, deadlines, and
distribution through the existing immutable job/task specifications and their
digests. Source names/hints remain untrusted presentation, not identities.

Store the complete plan once in the job's `outcome_contract`; separately freeze
the exact batch-to-task mapping. Worker/finalizer contracts bind that plan digest
and their specific obligations. Repeated binding facts in task scope are trust
boundaries, not replacement plans.

| Bound | Baseline rule |
| --- | --- |
| Sources | 2–16, additionally constrained by total batch admission |
| Resources per selection | Existing catalog/count ceiling, 50; complete enumeration must actually fit |
| Resources per statement | `COUNT_EXECUTION_POLICY.inline_max_targets`, currently 4 |
| Batches `B` | At most `min(max_direct_parents, max_tasks - 1)` and any stricter existing work/byte limits; currently 16 |
| Topology | `B` internal workers, one finalizer, `B` success dependencies |
| Parallelism | Existing graph/host/source permits, at most current graph maximum 4 |
| Wall time | Requested deadline intersected with policy, at most 300 seconds |
| Batch time | Current bounded read allowance, intersected with task/job remaining time |
| Output | Existing 64 KiB task-result bound and all existing aggregate graph bounds |
| Read attempts | One per batch; no automatic source retry |
| Pure finalizer attempts | At most existing maximum, currently 3 |

Sixteen sources and fifty resources per source do not imply every combination
fits. Compute all batches before admission and reject an oversized plan as a
whole. Do not add a reduction tree, enlarge graph fan-in, merge catalog pages,
or promise support above these ceilings in this change.

Use root work ceiling `B + F`, worker cumulative ceiling 1, and finalizer
cumulative ceiling/control reserve `F` (currently 3). Reserve one work unit per
attempt, not the entire finalizer allowance. Check the complete envelope before
insertion. Unknown interrupted consumption consumes its reservation. Pure
recomputation cannot extend that allowance.

The synthetic Juniper/Meridian fixture has 8 and 38 base tables: 12 workers and
one finalizer. Expected totals are 261 and 1331, with Meridian winning.

## 5. Internal execution contracts

### 5.1 Count batch

Declare internal-only `data.count_batch` in the existing data domain. Reuse
`data.count_rows.executor` / `DataCountRowsExecutor`, its SQL preparation,
validation, adapters, timing, and `data.row_counts` output schema. No model view.

Closed code-owned input: `source_id`, bounded `resource_ids`,
`include_views=false`, and the frozen `selection_binding`. Define binding fields
explicitly from the existing count selection; reject extra/missing keys and
nonmatching source/resource revisions. The selected batch must be an exact
planned slice. Policy is effect-free read, bounded to the existing batch width.

Use `INTERNAL_CAPABILITY` tasks with the existing internal role. Keep the registry
rule rejecting internal execution of capabilities with model views. Do not add
an `allow_model_capability` or `bypass_validation` flag.

Each request carries a real machine `RunStartEnvelope`, `GRAPH_TASK`
`ExecutionScope`, exact `GraphTaskBinding`, principal, claim/fence, contract
digests, and sensitivity ceiling. Diagnostic `observation_origin` alone grants
no authority. Use no fabricated `RunSession`, provider, or transcript.

Extend existing task-scope construction for these internal kinds. Extend
`TaskAttemptStateGuard.revalidate` to accept optional canonical call arguments;
its current-authority snapshot must compare them with the frozen task's
capability/arguments and plan binding. Existing non-count calls remain unchanged.
The runtime guard wrapper can retain original canonical arguments for repeated
checks, without adding a new guard service or an internal request context type.

Preserve whole-selection permission/revision checks before every batch,
including after permit queueing. The task's frozen validation scope may include
its complete source selection, but exact call binding and batch policy must
limit actual SQL to the planned resources. Never interpret broader validation
scope as permission to change the read.

Use the existing source permit once, avoiding nested acquisition by both
supervisor and runtime. Revalidate immediately before I/O. Enforce the remaining
task/job deadline during permit waits and execution; a pre-dispatch deadline
check alone does not bound an in-flight query. Preserve adapter cancellation and
cleanup behavior. No source I/O occurs under a graph transaction or broad lock.

### 5.2 Zero-model scopes

The baseline `ExecutionScope` requires positive `per_run_max_tokens`. Permit zero
only for authenticated graph internal/finalizer tasks with all of:

- `scope_kind=GRAPH_TASK` and a matching graph-task binding;
- exact current internal task/capability contract and allowed role;
- empty eligible model routes, zero model cost, and zero tokens;
- read/none access, effect `NONE`, and no capability grants.

Keep positive budgets mandatory for model-driven and scheduled scopes. Role and
effect facts must be authenticated, not trusted from caller labels. Every model
admission path must reject a zero-model scope before provider construction.
Audit `value or default` patterns so zero never becomes a normal allowance.
Do not invent a one-token budget or fake route to satisfy the old validation.

### 5.3 Direct output acceptance

For successful batches, validate ordinary `ToolOutput`, then construct the
existing bounded `TaskResult` with:

```text
payload = validated output data
artifact_ids = ()
effect_receipt_ids = ()
provenance.authority = "authenticated_internal_capability_output"
provenance = plan/batch identity and task specification digest
verification = capability/executor identity, contract digest, output SHA-256
```

Use an attempt-derived result identity. On ambiguous SQL acknowledgment, read
back and reuse the exact stored result, including timestamp/digest; a stable ID
alone does not make newly regenerated content identical. Never repeat source I/O
to reconcile a storage response.

Extend `JobSupervisor._execute_graph_attempt` and existing result construction
only for the declared count kinds. Keep artifact requirements and recovery for
existing profile/generic finalizers intact. `complete_graph_attempt` remains the
atomic successful-output/task-state/budget settlement boundary. Only accepted
results contribute to coverage.

### 5.4 Finalizer

Declare internal-only `data.finalize_count_comparison`, effect-free and with no
source I/O or model route. The supervisor supplies the frozen plan, exact
required batch/task mapping, and accepted result envelopes from current graph
state. The data-owned pure reducer validates:

- exact required worker set, ownership, task/attempt/result identities and digests;
- exact capability/executor lineage and registered schema;
- source/resource revisions, batch membership, and absence of duplicate coverage;
- nonnegative integer counts (excluding booleans), batch subtotals and total coverage;
- the frozen comparison requirement and inherited sensitivity.

Reject missing, foreign, substituted, tampered, or partial evidence. The finalizer
does not call a model or query a source to repair it. Compute all totals, ties,
and winners in code. Share count validation/reduction logic with existing count
owners rather than copying arithmetic or creating a generic reducer framework.

Completed output kind is `data.count_comparison`; required fields include:

```json
{
  "plan_digest": "sha256:...",
  "status": "completed",
  "sources": [
    {"source_id": "source:A", "selected_resources": 8, "counted_resources": 8, "total_rows": 261},
    {"source_id": "source:B", "selected_resources": 38, "counted_resources": 38, "total_rows": 1331}
  ],
  "winner_source_ids": ["source:B"],
  "coverage": "complete_admitted_base_tables",
  "snapshot_scope": "separate_batch_and_source_snapshots",
  "accepted_results": [
    {"task_id": "task:...", "result_id": "result:...", "result_digest": "sha256:..."}
  ]
}
```

Also retain observation/timing facts and source revisions. Return all tied source
IDs on ties. No physical-database completeness or common-instant snapshot claim
is permitted. `total_rows` is an exact sum of the admitted batch observations,
whose separate snapshots remain explicit.

Persist the finalizer output and one `GraphJobDelivery` using existing
`finalize_graph_attempt` seal/CAS. Both have empty artifact references. No file
was requested, so create no artifact merely to satisfy the previous internal
path. Existing delivery and result types already permit empty artifact tuples;
extend the supervisor's count-specific finalization accordingly. A later export
request remains a separate workflow. The delivered numerical summary is
code-owned; optional model explanation cannot replace the structured facts.

## 6. Failure, progress, and recovery

Keep success-only task results, dependency edges, and the finalizer barrier.
A failed read is not a successfully completed "observation task". Do not alter
edge semantics to allow a finalizer to publish partial work as successful.

| Event | Required behavior |
| --- | --- |
| Invalid proposal | One bounded non-dispatched error; no graph/source query |
| Source/resource or contract drift | Existing `source_or_contract_drift` control/attention path; no refreshed authority |
| Permission loss | Existing `needs_authorization` control path; no scope substitution |
| Other batch query failure | Failed task/graph outcome, explicit no-source-retry policy |
| Deadline/work exhaustion | Existing terminal/control semantics; no extra allowance or later model pass |
| Successful batch, SQL acknowledgment lost | Read back exact accepted result; never query again to reconcile |
| Active read attempt without accepted result after restart/lease expiry | Fence without requeue; interrupted/failed outcome requiring fresh admission |
| User cancellation/shutdown | Respect cancellation/drain semantics; fence late publication, preserve accepted evidence, no automatic read replay |
| Unclaimed planned task after ordinary host close | Can run only if the original job remains runnable, current admission validates, and its original budgets/deadline permit it |
| Unaccepted pure finalizer after interruption | Recompute only from immutable accepted results, within original remaining work/deadline |
| Accepted final result/delivery, acknowledgment lost | Reuse exact committed records; do not regenerate or duplicate delivery |

Apply the no-read-replay rule consistently in startup recovery, stale lease
recovery, shutdown handling, and generic exception/retry classification. A
one-unit worker budget reinforces this rule but does not replace explicit policy.
Preserve retry/recovery behavior of unrelated model, profile, and effect tasks.
Never equate a claimed/started attempt with a proved source dispatch. An
unaccepted in-flight read may have occurred; its count remains unavailable.

The ordinary finalizer barrier must still require every required successful
result and no open controls. On incomplete work, no successful comparison result
or success delivery is published. Fresh reading requires a newly admitted plan;
existing task-replacement controls must not silently replace the frozen count
denominator or reset budgets.

### Partial progress

Add a count-specific `count_progress` field to the existing job inspection
projection. Reuse a pure data-owner reduction of the immutable plan, accepted
outputs, and retained attempt/control facts. Do not add a progress table, event
projection as authority, or a second persisted outcome.

For each source return source identity/revision, selected/countable resource
count, accepted counted resources, missing coverage, `observed_total_rows`, and
nullable `total_rows`. A fully covered source may have an exact total while
another source remains incomplete. For no accepted observations, the observed
sum is zero but `total_rows` stays null and counted resources are zero. It must
not be rendered as a measured zero-row database. No incomplete comparison has
an established winner.

Retain normalized failure codes, available dispatch disposition (including
unknown), timing and accepted evidence identities through existing attempts,
bounded checkpoints/controls and observation/provider boundaries. Do not promise
to recover values that were never durably accepted before a crash. Unaccepted
or rejected evidence cannot contribute to arithmetic or coverage.

`job_read_results` / `Agent.read_job_result` remain successful-final-result
surfaces. They must not silently return partial progress as a successful result.
Existing task-result readers continue to expose accepted batch evidence.

### Foreground recovery guidance

Keep specific view-validation errors with rejected fields, accepted fields,
non-dispatch disposition and the actual applicable correction path. The model
must issue a corrected call. Never strip arguments or replay automatically.

`RunSessionEvidence.observe_progress` remains bounded, run-local and warning-only.
It should describe repeated unchanged invalid calls as invocation failures,
without declaring the operation unavailable. Preserve unknown/transient/effect
distinctions, changed arguments, changed revisions and legitimate repeated reads.
Do not add a text-based completion detector or force a foreground continuation.

The repeated warning should add repetition facts to the existing recovery
guidance rather than inventing another blocker policy. Raw source messages and
arbitrary tool-output instructions remain untrusted. Do not copy them into a
system message. Keep safe code-owned projection and current tool availability.
This improves guidance; it does not guarantee model correction before admission.

## 7. Ownership and exact replacement map

Use existing modules. No new generic module, abstract base, registry, state
store, middleware layer, or scheduler is required.

| File / symbols | Change |
| --- | --- |
| `agent.py`, `hosting/embedded.py` | Thin typed entry point and existing-owner composition; separate static read admission from model-worker route/cost gates |
| `domains/data/controller.py` | Replace generic count MODEL-plan resolution with count-comparison resolution; reuse current catalog/permission/binding validation and count preparation |
| `domains/data/row_counts.py` | Concrete immutable plan, shared count validation, deterministic comparison/progress reduction |
| `domains/data/capabilities.py` | Internal count-batch alias over existing executor/output schema; internal pure finalizer declaration/executor |
| `jobs/graph/admission.py` | New narrow view on existing capability/executor, disjoint canonical validation, static graph builder and receipt |
| `jobs/graph/execution.py` | Explicit count internal contracts and internal task scope preparation |
| `jobs/graph/guard.py` | Exact canonical argument/plan checks against current authenticated task authority |
| `capabilities.py` | Narrow zero-model scope validation; preserve internal-only registry restriction and ToolView/canonical validation |
| `capability_runtime.py` | Existing internal execution and guard/permit/result boundaries; no count-specific bypass or new broker |
| `jobs/supervisor.py` | Count-kind output acceptance, no-read-replay recovery, pure finalization and atomic delivery |
| `jobs/projections.py`, `jobs/capabilities.py` | Bounded inspection progress, delegated to data-owned arithmetic |
| `storage/sql_graph.py`, `storage/sql.py` | Reuse current result, control, fence, budget and delivery transactions; extend only if an acceptance test identifies a missing atomic operation |

Remove the following only after their replacement tests pass:

1. `COUNT_PLAN_TOOL_NAME`, `data_count_plan`, its `plan_ref` schema branch and
   `DataCapabilityDomain.project` role-switching for that view.
2. `DataCapabilityDomain.prepare_call` expansion of model `plan_ref` into frozen
   read arguments. Retain foreground selection authentication and explicit
   canonical source/resource count arguments.
3. `GraphAdmissionBuilder.build` count `plan_ref` rewriting and count MODEL worker
   construction. Generic non-count model/effect tasks remain supported.
4. `_frozen_count_call_id` in `jobs/graph/capabilities.py`, automatic count-parent
   completion binding, and count-specific child-ID completion-error repairs.
   Preserve unrelated effect receipt binding and generic evidence authentication.
5. Count dependency `programmatic_parent` exceptions in
   `validate_graph_task_arguments` and associated runtime argument plumbing.
6. Model count-worker context instructions and count-specific
   `count_operation_incomplete` to `task_block` conversion.
7. Model-directed `task_block` argument generation in
   `count_operation_failure_details`; keep or replace any genuinely shared pure
   cause classification in the responsible owner.
8. `Capability.read_dependency_ids`, its digest/toolbox projections, and the
   `InitialTaskProposal` dependency/retained-selection propagation, if the final
   caller audit still confirms the baseline's sole declaration is count
   selection. Remove the obsolete resolver callback; compose the new concrete
   count-plan resolver through the same existing owners.

Retain `_programmatic_reader` count bindings, retained child observations,
`ProgrammaticReadBroker.observations`, and post-permit rechecks: the foreground
whole-selection operation still consumes them. Do not remove live foreground
functionality merely because it was also used by the retired worker.

Set foreground `data.count_selection` to no longer advertise generic graph MODEL
eligibility. Preserve its source/resource canonical form, selection-ref view and
foreground ceilings. New multi-source durable work uses the dedicated view on
the existing admission capability. Reject attempts to recreate the retired
count MODEL path with available-action guidance that actually fits the request.

This deliberately retires the development generic count MODEL-job path,
including its single-source form; it does not retire foreground single-source
counts. The new comparison view requires at least two sources. Do not fake a
second source or silently claim single-source durable compatibility. Audit and
document affected typed/documented consumers before cutover; if preserving a
single-source durable API is a required contract, resolve that scope explicitly
before deletion rather than adding an unreviewed compatibility execution path.

## 8. Persistence, upgrades, and existing jobs

Use existing `GraphJobSpecification`/`GraphTaskSpecification` mappings,
`TaskAttempt`, `TaskResult`, controls, budget ledgers and `GraphJobDelivery`.
No new tables, execution-kind enum, transcript shape or diagnostic store are
planned. Accepted output payloads/provenance remain bounded by existing records.

This is nevertheless a semantic durable-contract change: new internal task
kinds, zero-model scope semantics, output acceptance and recovery policy. It
requires the ordinary home-release process even if DDL is unchanged.

At the checkpoint, locally available release snapshots through `v1.2.2` have
home revision 3; current revision 4 is untagged in this checkout. Recheck release
status before editing migration history. If revision 4 remains unreleased,
amend `revision_0004.py`, its definition/checksum, revision-4 golden fixture and
`release/agent-home-contract.json` together. If it has been released, append the
next migration; never modify a released migration, decoder, checksum or fixture.

Compatibility requirements:

- Preserve terminal old jobs, transcripts, accepted results, artifacts,
  receipts and accounting as historical evidence.
- Never translate old frozen count `plan_ref` contracts into newly authorized
  internal work, replace root authority, or bless new contract digests implicitly.
- Old nonterminal count MODEL jobs must be rejected by current-contract/policy
  validation before any provider or source call, with attention requiring fresh
  admission. Include that pre-provider validation in owner execution preparation;
  do not retain a legacy count executor or spend model requests discovering a
  retired contract.
- Existing active attempts must be fenced and conservatively settled through
  migration/recovery owners. Never drop a reservation or infer measured zero.
- Arbitrary development revision-4 homes with an older checksum remain
  fail-closed. Do not add a generic same-revision compatibility bridge. Preserve
  their bytes and reports for review; do not delete or rewrite fixture homes.
- `Agent.from_storage` callers remain responsible for provisioning/upgrading
  supplied storage. Current admission still fails closed on retired contracts.

Run the migration/golden-home preservation matrix and home-release check. If
the generator cannot infer a semantic change, record it in the migration
definition and tests; unchanged column layouts do not excuse release coverage.

## 9. Acceptance matrix

Extend existing owner cases instead of duplicating the same invariant at each
layer. Use real local state/adapters and the runtime for framework tests.
Scripted providers test execution/forwarding, not model intelligence. Independent
fixture oracles, not production reducers, establish expected numbers/coverage.

| ID | Required observable acceptance | Existing suite / proposed extension |
| --- | --- | --- |
| C1 | Generic and count views expose only their own closed input schema; provider, search/load/inspection and digests agree | `tests/capabilities/test_toolbox_contracts.py` |
| C2 | Mixed IDs/refs, extra fields, duplicate sources, forged/cross-run refs, incomplete enumerations and empty selections create no job and query no source | `tests/data/test_row_counts.py`, `tests/jobs/graph/test_public_composition.py` |
| C3 | Typed and model admission resolve equivalent required sets; empty typed scope never widens to all; files-only/machine origins cannot acquire count-job authority | Public composition and existing scope families |
| C4 | Static count admission works with no model-worker route/cost configuration; generic MODEL work retains its original gates | Public composition/hosting contract tests |
| C5 | Plan, task fan-in, work/control reserves, deadlines and byte bounds validate before insertion; 16-source/50-resource ceilings do not bypass total batches | Data plan and graph model/property tests |
| C6 | Synthetic 8/38-table fixture produces 12 bounded reads, exact 261/1331 totals, complete admitted coverage and correct winner | Existing independent fixture in `tests/acceptance/test_count_recovery.py`; retarget public count-job case |
| C7 | Zero-row tables, ties, quoted identities and reordered completion are correct; reducer rejects malformed, duplicate, substituted, foreign and missing evidence | `tests/data/test_row_counts.py` and public graph families |
| C8 | Internal read arguments cannot change source/batch under a valid guard; revocation/drift during permit queueing prevents the affected I/O | Existing guard/scope and graph capacity cases |
| C9 | Zero-model scope is admitted only for authenticated internal graph slice; model/scheduled scopes reject zero and provider construction is forbidden | Capability/scope and model admission tests |
| C10 | Every successful batch/result acceptance is fenced and atomic with budget settlement; output results work with empty artifact tuples | Extend `tests/storage/contracts/test_graph_transactions.py` |
| C11 | Missing/failed batches block the unchanged success-only finalizer; no winner or success delivery is published | Existing finalizer barrier and public composition cases |
| C12 | Partial inspection separates no observations, measured zero, lower bounds and complete per-source totals | `tests/jobs/graph/test_operational_projections.py` |
| C13 | Restart/lease expiry/close/cancel never replay an active unaccepted read; accepted results survive; pure finalizer retries remain bounded | Extend relevant `tests/jobs/graph/test_static_profile_runtime.py` lifecycle parameter families |
| C14 | Lost acknowledgment before/after acceptance and delivery is reconciled by exact readback, with conserved budgets and one delivery | Existing response-loss storage/runtime cases, parameterized for output results |
| C15 | Late stale worker/finalizer cannot accept or publish; no source I/O or provider work spans a graph transaction/broad lock | Existing graph guard, transaction and runtime boundary tests |
| C16 | Existing profile/artifact jobs, ordinary model tasks, effect receipts, foreground count operations/cursors/full schemas retain behavior | Existing suites; preserve unique parameter cases |
| C17 | Released home upgrades preserve history; retired count jobs cause zero provider/source calls; new scope/result semantics round-trip and snapshot matches | `tests/storage/test_golden_homes.py`, `test_release_contract.py` and codec families |
| C18 | Exact ordered foreground transcript, budget failure, run-local warnings and per-call diagnostic boundaries remain intact | `tests/acceptance/test_count_recovery.py`, loop and observation suites |

Name new tests after observable behavior, for example
`test_public_count_comparison_executes_without_model_workers`, not project phase
numbers. Keep the ordinary MODEL case in the existing model-task test. Retarget
old count MODEL-worker cases to internal count execution; replace child-ID
repair tests with missing/tampered accepted-result rejection. Preserve unique
single-source foreground, scope, ordering, drift and no-replay coverage.

Explicit crash/fault positions: before source dispatch; after source response
but before result acceptance; after acceptance before acknowledgment; during
pure reduction; and after final result/delivery commit before acknowledgment.
Verify actual attempt/query/provider counts, not only a successful final string.

### Existing feature acceptance must remain true

The preceding contracts extend, rather than replace, baseline requirements:
actual run limits/current tools are projected consistently; recovery reports
constraints and execution disposition without unavailable actions; compact
catalog metadata retains full-schema/freshness/revision/scope coverage; native
counts retain validated SQL, exact observations, provenance and timing;
run-local repetition tracking preserves corrections and reads; diagnostics stay
within existing provider/observation boundaries. Do not weaken those tests to
make the new job path pass.

## 10. Implementation order and offline validation

Implement one coherent replacement with owner-level red/green checks. Do not
run successive paid campaigns against intermediate architectures.

1. Confirm baseline/current consumers and release status; collect retained test
   node IDs. Record any contract conflict, especially single-source durable
   consumers, before removing a path. Do not silently expand this guide's scope.
2. Add failing tests for closed admission, immutable required coverage, reducer
   evidence and limits. Implement data-owned plan resolution/reduction.
3. Add zero-model scope and exact internal argument-binding tests. Implement
   internal declarations over the existing count executor and guarded runtime.
4. Add static admission, bounded output acceptance and success-only finalization.
   Test transaction fencing, acknowledgment loss and deadline/accounting cases.
5. Add no-read-replay recovery and partial projections. Complete crash,
   cancellation, drift, revocation and restart tests.
6. Remove the retired count MODEL protocol and metadata from section 7. Reconcile
   original/final test node IDs and document intentional removals and retained
   coverage. Inspect the final diff for duplicate execution/validation logic.
7. Update user/API documentation, `docs/TESTING.md`, applicable `AGENTS.md`
   architecture statements, migration definition/fixtures and release snapshot.
8. Run focused owners and full deterministic/quality/release checks. Only then
   prepare separately authorized live qualification.

Representative focused command, from repository root:

```bash
.venv/bin/python -m pytest \
  tests/data/test_row_counts.py \
  tests/capabilities/test_toolbox_contracts.py \
  tests/jobs/graph \
  tests/storage/contracts/test_graph_transactions.py \
  tests/storage/test_golden_homes.py \
  tests/storage/test_release_contract.py \
  tests/acceptance/test_count_recovery.py
```

Complete required broader checks after the final source change:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m black --check src tests
.venv/bin/python -m ruff check --select I .
.venv/bin/python -m mypy src/daita tests
.venv/bin/python scripts/check_home_release_contract.py check
git diff --check
```

Refresh the release snapshot with the repository's `write` command only as part
of the reviewed persistence change, then check/compare it against the applicable
tagged snapshot. Use normal repository execution settings; do not change native
security/permission guards to compensate for a sandbox or launcher mistake.

## 11. Live qualification and spending gate

No live call is authorized by this guide. Collection/import must resolve no
credentials and contact no services. Do not load a repository `.env` or local
provider sign-in simply to discover available resources. Future execution needs
explicit authorization, an authorized credential source, and adequate resource
budgets. Preserve existing reservation, provider recording, private reports,
independent synthetic fixtures, numerical/coverage scoring and cleanup helpers.

Keep these evidence categories separate:

1. Unscripted real-model foreground journeys and real-model job admission.
2. Fault-seeded real-model correction **before** admission.
3. Fault-seeded framework execution **after** real-model admission. With no
   subsequent model call, this is not a real-model continuation test.
4. Deterministic graph execution, numerical coverage and lifecycle correctness.

Retarget the existing live durable count case to the new multi-source contract.
Keep original malformed-call continuations and historical failures separately;
the changed path must not make the same scenario appear retrospectively passed.
Extend existing live helpers/cases rather than adding a second campaign runner.

Report natural first-attempt adoption, correction requests/frequency, premature
stops before admission, and model-generated versus injected malformed calls.
After admission, assert zero count-worker/finalizer provider requests using
recorded calls and attempt identities. Report actual native reads, accepted
results, timings, usage completeness, charge/reservations and cleanup.

Required live gates include:

- Fresh full 27-case API/SQLite campaign for the final candidate, retaining
  natural paraphrases, field-dependent controls, unavailable durable recovery,
  repeated-failure correction, scope, drift/timeouts and budget pressure.
- Targeted real-model count-job admission with exact fixture totals/coverage;
  separately fault-seeded drift/lifecycle execution evidence.
- Separate PostgreSQL and Codex qualification under their existing gates and
  disposable fixtures. A SQLite/API pass does not qualify either cohort.
- Numerical/coverage and task completion scored separately from truthful
  incomplete reporting. Honest no-observation answers do not qualify execution.
- Complete measurements and all required cases. Skipped, mocked, missing,
  incomplete or unknown-measurement cases never count as live qualification.
- Existing campaign efficiency gates: median at most 50,000 tokens, maximum at
  most 100,000, median at most eight logical requests; retain finite per-run
  step/wall/output/cost ceilings and do not clip incurred usage.

At the existing USD 0.50 per-case cap, a fresh 27-case API cohort has a worst-case
ceiling of USD 13.50 before additional durable cases. Confirm remaining
authorization afresh; historical spending figures are not a new budget. Do not
start a campaign whose complete qualification cannot fit its authorized gates,
and do not silently treat a budget-stopped partial campaign as qualified.

## 12. Delivery and definition of done

The final implementation report must identify:

- The single admitted-plan/execution/reduction path and exact removed machinery.
- Preserved foreground/model/profile/effect behavior and deliberate API changes.
- Persistence/release decisions, upgrade results and old-job handling.
- Owner/full deterministic and quality results, including any environment limits.
- Live cohorts actually performed, original failures retained, actual usage/cost,
  and every unperformed qualification or prerequisite.
- Remaining bounds: required-set interpretation before admission, maximum
  selection/batch size, unsupported empty-table selections, and separate snapshots.

The coding agent must not call the change complete because scripted tests pass,
claim improved model correction from deterministic worker execution, or present
unperformed live cohorts as qualified. Report **implemented** and
**live-qualified** independently. Follow the human's current commit/push/PR
instructions; this reference document itself grants none.
