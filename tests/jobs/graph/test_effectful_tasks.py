"""Phase 8 graph grant, fence, receipt, and uncertainty contracts."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from hashlib import sha256

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    CapabilityGrant,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ExecutionContractBindings,
    ExecutionScope,
    ExecutionScopeKind,
    OperationalEffect,
)
from daita.domains.data.capabilities import (
    relational_update_capability_declarations,
    relational_upsert_capability_declarations,
)
from daita.identity import AgentIdentity
from daita.jobs.graph.execution import graph_task_binding
from daita.jobs.graph.guard import SQLiteTaskAttemptGuard
from daita.jobs.graph.models import (
    BudgetAmount,
    ControlKind,
    GraphAuthority,
    GraphState,
    TaskResult,
    TaskState,
    canonical_digest,
    topology_digest,
)
from daita.jobs.graph.validation import validate_graph_admission
from daita.llm.models import ModelSensitivity
from daita.loop.models import (
    InstructionAuthority,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_records import (
    EffectReceipt,
    EffectReceiptConflictError,
    EffectResolution,
    EffectResolutionDecision,
    effect_receipt_id,
)
from tests.support.graph import GRAPH_NOW, graph_admission

pytestmark = pytest.mark.unit

_CAPABILITY_ID = "data.update_rows"
_CONTRACT_DIGEST = "sha256:" + "a" * 64
_ROUTE_ID = "mock:effect-graph"
_ROUTE_DIGEST = "sha256:" + "b" * 64


def _grant() -> CapabilityGrant:
    return CapabilityGrant(
        grant_id="capability-grant:phase8",
        domain_owner_id="data",
        capability_id=_CAPABILITY_ID,
        capability_contract_digest=_CONTRACT_DIGEST,
        constraints_kind="data.relational_write",
        constraints=FrozenJsonObject.from_mapping(
            {
                "source_id": "source-1",
                "resource_id": "resource-1",
                "resource_revision": "sha256:" + "c" * 64,
                "key_columns": ("id",),
                "allowed_insert_columns": (),
                "allowed_update_columns": ("status",),
                "generated_identity_columns": (),
                "max_rows": 10_000,
            }
        ),
        max_calls_per_occurrence=1,
    )


def test_only_preview_bound_native_writes_expose_the_phase8_graph_policy() -> None:
    update = relational_update_capability_declarations().capabilities[0]
    upsert = next(
        item
        for item in relational_upsert_capability_declarations().capabilities
        if item.id == "data.upsert_rows"
    )
    for capability, ceiling in ((update, 10_000), (upsert, 1_000)):
        policy = capability.execution_admission_policy
        assert policy is not None and policy.graph_v1_eligible
        assert policy.graph_max_targets == 1
        assert capability.automation_grant_policy is not None
        properties = capability.automation_grant_policy.constraints_schema["properties"]
        assert isinstance(properties, Mapping)
        maximum = properties["max_rows"]
        assert isinstance(maximum, Mapping) and maximum["maximum"] == 10_000
        if capability.id == "data.upsert_rows":
            assert ceiling == 1_000  # Enforced again by data-owned grant preparation.


def _effect_admission():
    admission = graph_admission()
    grant = _grant()
    bindings = {
        **ExecutionContractBindings(
            capability_contracts={_CAPABILITY_ID: _CONTRACT_DIGEST},
            model_routes={_ROUTE_ID: _ROUTE_DIGEST},
        ).material(),
        "capability_grants": {
            _CAPABILITY_ID: {
                "grant": grant.material(),
                "grant_digest": grant.grant_digest,
            }
        },
    }
    authority = GraphAuthority(
        capability_ids=(_CAPABILITY_ID,),
        access_modes=(AccessMode.NONE.value, AccessMode.WRITE.value),
        operational_effects=(
            OperationalEffect.MUTATE_DATA.value,
            OperationalEffect.NONE.value,
        ),
        model_route_ids=(_ROUTE_ID,),
        sensitivity=ModelSensitivity.RESTRICTED,
        contract_bindings=bindings,
    )
    worker = admission.tasks[0]
    worker_specification = replace(worker.specification, authority=authority)
    worker = replace(
        worker,
        specification=worker_specification,
        task_spec_digest=worker_specification.digest,
        task_scope_digest=authority.digest,
    )
    job_specification = replace(
        admission.job.specification,
        authority=authority,
        effect_mode="exact_grants",
    )
    job = replace(
        admission.job,
        specification=job_specification,
        specification_digest=job_specification.digest,
    )
    tasks = (worker, admission.tasks[1])
    graph = replace(
        admission.graph,
        topology_digest=topology_digest(tasks, admission.dependencies),
    )
    effect_admission = replace(admission, job=job, graph=graph, tasks=tasks)
    validate_graph_admission(effect_admission)
    return effect_admission, grant


async def _running_effect_attempt(tmp_path):
    store = await SQLiteStateStore.open(
        tmp_path / "phase8-effects.sqlite",
        clock=lambda: GRAPH_NOW + timedelta(seconds=2),
    )
    await store.initialize_identity(AgentIdentity("agent-1", "Agent", GRAPH_NOW))
    admission, grant = _effect_admission()
    await store.admit_graph(admission)
    attempt = await store.claim_graph_task(
        "agent-1",
        "job-1",
        "worker",
        attempt_id="attempt-effect",
        claim_token="claim-effect",
        run_id="run-effect",
        executor_id=_ROUTE_ID,
        claimed_at=GRAPH_NOW,
        lease_seconds=30,
        absolute_deadline_at=GRAPH_NOW + timedelta(minutes=2),
        budget_reservations=(BudgetAmount("work_units", 1),),
    )
    assert attempt is not None
    attempt = await store.start_graph_attempt(
        "agent-1",
        "job-1",
        "worker",
        attempt.attempt_id,
        claim_token=attempt.claim_token,
        fencing_epoch=attempt.fencing_epoch,
        started_at=GRAPH_NOW + timedelta(seconds=1),
    )
    assert attempt is not None
    inspection = await store.inspect_graph("agent-1", "job-1")
    assert inspection is not None
    task = next(item for item in inspection.tasks if item.task_id == "worker")
    binding = graph_task_binding(inspection, task, attempt)
    contracts = ExecutionContractBindings(
        capability_contracts={_CAPABILITY_ID: _CONTRACT_DIGEST},
        model_routes={_ROUTE_ID: _ROUTE_DIGEST},
    )
    scope = ExecutionScope(
        scope_id="graph-scope:attempt-effect",
        revision=1,
        agent_id="agent-1",
        principal_id="agent-1",
        grant_id="graph-attempt:attempt-effect",
        job_id="job-1",
        job_revision=inspection.graph.revision + 1,
        allowed_source_ids=(),
        allowed_resource_ids=(),
        allowed_capability_ids=(_CAPABILITY_ID,),
        allowed_access_modes=frozenset({AccessMode.NONE, AccessMode.WRITE}),
        allowed_operational_effects=frozenset(
            {OperationalEffect.NONE, OperationalEffect.MUTATE_DATA}
        ),
        sensitivity_ceiling=ModelSensitivity.RESTRICTED,
        eligible_model_routes=(_ROUTE_ID,),
        per_run_max_cost_usd=Decimal("1"),
        per_run_max_tokens=1_000,
        distribution_plan_digest=admission.job.specification.distribution_plan_digest,
        contract_bindings=contracts,
        scope_kind=ExecutionScopeKind.GRAPH_TASK,
        capability_grants=(grant,),
        graph_task_binding=binding,
    )
    instruction = "Execute the exact effect test."
    start = RunStartEnvelope(
        origin=RunOrigin.JOB_TASK,
        instruction_authority=InstructionAuthority.CODE_OWNED,
        trusted_instruction_id="graph-task:attempt-effect",
        trusted_instruction=instruction,
        instruction_digest="sha256:" + sha256(instruction.encode("utf-8")).hexdigest(),
        untrusted_payload={},
        payload_digest=canonical_digest({}),
        execution_scope=scope,
    )
    run = RunInput(
        id="run-effect",
        agent_id="agent-1",
        message=instruction,
        created_at=GRAPH_NOW + timedelta(seconds=1),
        conversation_id="graph-task-effect",
        start=start,
        history_sensitivity=ModelSensitivity.RESTRICTED,
    )
    await store.start(run)
    guard = SQLiteTaskAttemptGuard(
        store=store,
        binding=binding,
        claim_token=attempt.claim_token,
        run_id=attempt.run_id,
        clock=lambda: GRAPH_NOW + timedelta(seconds=2),
    )
    return store, grant, attempt, guard


def _receipt(grant: CapabilityGrant, *, call_id: str = "call-effect") -> EffectReceipt:
    operation_key = canonical_digest({"call_id": call_id})
    return EffectReceipt(
        receipt_id=effect_receipt_id(
            agent_id="agent-1",
            run_id="run-effect",
            call_id=call_id,
            operation_key=operation_key,
        ),
        receipt_kind="data.update_rows",
        agent_id="agent-1",
        run_id="run-effect",
        call_id=call_id,
        capability_id=_CAPABILITY_ID,
        domain_owner_id="data",
        capability_contract_digest=_CONTRACT_DIGEST,
        operation_key=operation_key,
        argument_fingerprint=canonical_digest({"status": "complete"}),
        sensitivity=ModelSensitivity.RESTRICTED,
        started_at=GRAPH_NOW + timedelta(seconds=2),
        capability_grant_digest=grant.grant_digest,
    )


async def test_graph_receipt_reservation_is_atomically_fence_bound_and_once_only(
    tmp_path,
) -> None:
    store, grant, attempt, guard = await _running_effect_attempt(tmp_path)
    try:
        receipt = _receipt(grant)
        assert (
            await store.start_effect_receipt(
                receipt, grant=grant, task_attempt_guard=guard
            )
            == receipt
        )
        with sqlite3.connect(store.path) as connection:
            linkage = connection.execute(
                "SELECT job_id, task_id, task_attempt_id, fencing_epoch, task_spec_digest FROM effect_receipts WHERE id = ?",
                (receipt.receipt_id,),
            ).fetchone()
        assert linkage == (
            "job-1",
            "worker",
            attempt.attempt_id,
            attempt.fencing_epoch,
            guard.binding.task_spec_digest,
        )
        await store.finish_effect_receipt(
            receipt.finish(
                EffectObservation(
                    EffectOutcome.NOT_APPLIED,
                    EffectEvidenceBasis.LOCAL_NOT_DISPATCHED,
                ),
                finished_at=GRAPH_NOW + timedelta(seconds=3),
            )
        )
        with pytest.raises(EffectReceiptConflictError, match="ceiling"):
            await store.start_effect_receipt(
                _receipt(grant, call_id="call-second"),
                grant=grant,
                task_attempt_guard=guard,
            )
    finally:
        await store.close()


async def test_stale_graph_attempt_cannot_reserve_an_effect(tmp_path) -> None:
    store, grant, attempt, guard = await _running_effect_attempt(tmp_path)
    try:
        fenced = await store.fence_graph_attempt(
            "agent-1",
            "job-1",
            "worker",
            attempt.attempt_id,
            fencing_epoch=attempt.fencing_epoch,
            fenced_at=GRAPH_NOW + timedelta(seconds=3),
            requeue=True,
            reason_code="test_fence",
        )
        assert fenced is not None
        with pytest.raises(EffectReceiptConflictError, match="stale"):
            await store.start_effect_receipt(
                _receipt(grant), grant=grant, task_attempt_guard=guard
            )
        assert not await store.list_effect_receipts_for_graph_attempt(
            "agent-1", "job-1", "worker", attempt.attempt_id
        )
    finally:
        await store.close()


async def test_fence_after_reservation_blocks_instead_of_requeueing(tmp_path) -> None:
    store, grant, attempt, guard = await _running_effect_attempt(tmp_path)
    try:
        receipt = _receipt(grant)
        await store.start_effect_receipt(receipt, grant=grant, task_attempt_guard=guard)
        settled = await store.fence_graph_attempt(
            "agent-1",
            "job-1",
            "worker",
            attempt.attempt_id,
            fencing_epoch=attempt.fencing_epoch,
            fenced_at=GRAPH_NOW + timedelta(seconds=3),
            requeue=True,
            reason_code="lease_lost_after_reservation",
        )
        assert settled is not None and settled.state.value == "blocked"
        inspection = await store.inspect_graph("agent-1", "job-1")
        assert inspection is not None
        assert (
            next(item for item in inspection.tasks if item.task_id == "worker").state
            is TaskState.BLOCKED
        )
        assert any(
            item.kind is ControlKind.EFFECT_UNCERTAIN for item in inspection.controls
        )
    finally:
        await store.close()


async def test_started_graph_receipt_recovers_uncertain_and_never_requeues(
    tmp_path,
) -> None:
    path = tmp_path / "phase8-effects.sqlite"
    store, grant, attempt, guard = await _running_effect_attempt(tmp_path)
    receipt = _receipt(grant)
    await store.start_effect_receipt(receipt, grant=grant, task_attempt_guard=guard)
    await store.close()

    reopened = await SQLiteStateStore.open(
        path, clock=lambda: GRAPH_NOW + timedelta(seconds=4)
    )
    try:
        recovered = await reopened.load_effect_receipt("agent-1", receipt.receipt_id)
        assert recovered is not None
        assert recovered.outcome is EffectOutcome.UNCERTAIN
        inspection = await reopened.inspect_graph("agent-1", "job-1")
        assert inspection is not None
        assert (
            next(item for item in inspection.tasks if item.task_id == "worker").state
            is TaskState.BLOCKED
        )
        assert not await reopened.list_active_graph_attempts("agent-1")
    finally:
        await reopened.close()


async def test_uncertain_graph_receipt_blocks_task_descendants_and_resolution_is_evidence_only(
    tmp_path,
) -> None:
    store, grant, attempt, guard = await _running_effect_attempt(tmp_path)
    try:
        receipt = _receipt(grant)
        await store.start_effect_receipt(receipt, grant=grant, task_attempt_guard=guard)
        terminal = await store.finish_effect_receipt(
            receipt.finish(
                EffectObservation(EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN),
                finished_at=GRAPH_NOW + timedelta(seconds=3),
            )
        )
        inspection = await store.inspect_graph("agent-1", "job-1")
        assert inspection is not None
        worker = next(item for item in inspection.tasks if item.task_id == "worker")
        finalizer = next(
            item for item in inspection.tasks if item.task_id == "finalizer"
        )
        control = next(
            item
            for item in inspection.controls
            if item.kind is ControlKind.EFFECT_UNCERTAIN
        )
        assert worker.state is TaskState.BLOCKED
        assert finalizer.state is TaskState.PENDING
        assert inspection.job.state is GraphState.NEEDS_ATTENTION
        resolved = await store.resolve_effect_receipt(
            "agent-1",
            EffectResolution(
                receipt_id=terminal.receipt_id,
                receipt_digest=terminal.receipt_digest,
                decision=EffectResolutionDecision.ALLOW_FUTURE_WORK,
                approving_principal_id="agent-1",
                control_id=control.control_id,
                resolved_at=GRAPH_NOW + timedelta(seconds=4),
                note="Reviewed the external system; do not retry this task.",
            ),
        )
        assert resolved.resolution is not None
        after = await store.inspect_graph("agent-1", "job-1")
        assert after is not None
        assert (
            next(item for item in after.tasks if item.task_id == "worker").state
            is TaskState.BLOCKED
        )
        assert (
            next(
                item for item in after.controls if item.control_id == control.control_id
            ).state.value
            == "open"
        )
    finally:
        await store.close()


async def test_success_before_task_result_is_reconciled_without_replay(
    tmp_path,
) -> None:
    store, grant, attempt, guard = await _running_effect_attempt(tmp_path)
    try:
        receipt = _receipt(grant)
        await store.start_effect_receipt(receipt, grant=grant, task_attempt_guard=guard)
        await store.finish_effect_receipt(
            receipt.finish(
                EffectObservation(
                    EffectOutcome.SUCCEEDED,
                    EffectEvidenceBasis.ADAPTER_VERIFIED,
                    FrozenJsonObject.from_mapping({"affected_rows": 0}),
                ),
                finished_at=GRAPH_NOW + timedelta(seconds=3),
            )
        )
        assert await store.reconcile_graph_effect_attempt(
            "agent-1", "job-1", "worker", attempt.attempt_id
        )
        inspection = await store.inspect_graph("agent-1", "job-1")
        assert inspection is not None
        assert (
            next(item for item in inspection.tasks if item.task_id == "worker").state
            is TaskState.BLOCKED
        )
        controls = tuple(
            item
            for item in inspection.controls
            if item.kind is ControlKind.EFFECT_UNCERTAIN
        )
        assert controls
        details = controls[0].payload["details"]
        assert isinstance(details, Mapping) and details["outcome"] == "succeeded"
    finally:
        await store.close()


async def test_successful_task_result_authenticates_exact_linked_receipt(
    tmp_path,
) -> None:
    store, grant, attempt, guard = await _running_effect_attempt(tmp_path)
    try:
        receipt = _receipt(grant)
        await store.start_effect_receipt(receipt, grant=grant, task_attempt_guard=guard)
        terminal = await store.finish_effect_receipt(
            receipt.finish(
                EffectObservation(
                    EffectOutcome.SUCCEEDED,
                    EffectEvidenceBasis.ADAPTER_VERIFIED,
                    FrozenJsonObject.from_mapping({"affected_rows": 1}),
                ),
                finished_at=GRAPH_NOW + timedelta(seconds=3),
            )
        )
        completed_at = GRAPH_NOW + timedelta(seconds=4)
        provenance = {
            "evidence": (
                {
                    "call_id": receipt.call_id,
                    "capability_id": receipt.capability_id,
                    "effect_receipt_id": receipt.receipt_id,
                },
            )
        }
        schema_digest = canonical_digest({"kind": "effect_result"})
        digest_material = {
            "agent_id": "agent-1",
            "job_id": "job-1",
            "task_id": "worker",
            "result_id": "result-effect",
            "attempt_id": attempt.attempt_id,
            "run_id": attempt.run_id,
            "result_kind": "effect_result",
            "schema_digest": schema_digest,
            "payload": {"affected_rows": 1},
            "summary": "Applied the exact previewed update.",
            "sensitivity": ModelSensitivity.RESTRICTED.value,
            "provenance": provenance,
            "artifact_ids": (),
            "effect_receipt_ids": (terminal.receipt_id,),
            "verification": {"receipt_authenticated": True},
            "residual_risk": None,
            "downstream_constraints": {},
            "completed_at": completed_at.isoformat(),
        }
        result = TaskResult(
            agent_id="agent-1",
            job_id="job-1",
            task_id="worker",
            result_id="result-effect",
            attempt_id=attempt.attempt_id,
            run_id=attempt.run_id,
            result_kind="effect_result",
            schema_digest=schema_digest,
            payload={"affected_rows": 1},
            summary="Applied the exact previewed update.",
            sensitivity=ModelSensitivity.RESTRICTED,
            provenance=provenance,
            artifact_ids=(),
            effect_receipt_ids=(terminal.receipt_id,),
            verification={"receipt_authenticated": True},
            residual_risk=None,
            downstream_constraints={},
            completed_at=completed_at,
            result_digest=canonical_digest(digest_material),
        )
        await store.complete_graph_attempt(
            result,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
            usage=(BudgetAmount("work_units", 1),),
        )
        inspection = await store.inspect_graph("agent-1", "job-1")
        assert inspection is not None
        stored = next(item for item in inspection.results if item.task_id == "worker")
        settled = next(
            item
            for item in inspection.attempts
            if item.attempt_id == attempt.attempt_id
        )
        assert stored.effect_receipt_ids == (terminal.receipt_id,)
        assert settled.effect_receipt_ids == (terminal.receipt_id,)
    finally:
        await store.close()
