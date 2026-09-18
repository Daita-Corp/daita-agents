from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita.cli import _graph_blockers_mapping, build_parser
from daita.jobs.graph import execution, planning, reduction
from daita.jobs.graph.models import (
    AttemptState,
    BudgetAmount,
    ControlKind,
    ControlState,
    EdgeKind,
    GraphAuthority,
    GraphMutationRequest,
    GraphState,
    TaskControl,
    TaskDependency,
    TaskRole,
    TaskState,
    canonical_digest,
)
from daita.jobs.graph.reduction import reduce_attempt_failure
from daita.jobs.graph.validation import GraphValidationError
from daita.jobs.owner import GraphBlockerProjection, JobError, JobOwner
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sqlite import SQLiteStateStore
from tests.support.graph import GRAPH_NOW, graph_admission, task_result
from tests.support.model_graph_integration import (
    AGENT_ID,
    MODEL_ROUTE_ID,
    READ_CAPABILITY_ID,
    ModelGraphIntegration,
)

pytestmark = pytest.mark.integration


def _tool_response(
    call_id: str, name: str, arguments: dict[str, object]
) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
    )


async def _claim_and_start(
    store: SQLiteStateStore,
    *,
    agent_id: str,
    job_id: str,
    task_id: str,
    attempt_id: str,
    at,
    budget: int = 1,
):
    attempt = await store.claim_graph_task(
        agent_id,
        job_id,
        task_id,
        attempt_id=attempt_id,
        claim_token=f"{attempt_id}:claim",
        run_id=f"{attempt_id}:run",
        executor_id="test.executor",
        claimed_at=at,
        lease_seconds=30,
        absolute_deadline_at=at + timedelta(minutes=2),
        budget_reservations=(BudgetAmount("work_units", budget),),
    )
    assert attempt is not None
    started = await store.start_graph_attempt(
        agent_id,
        job_id,
        task_id,
        attempt_id,
        claim_token=f"{attempt_id}:claim",
        fencing_epoch=attempt.fencing_epoch,
        started_at=at + timedelta(milliseconds=1),
    )
    assert started is not None
    return started


def test_graph_private_helpers_are_stateless_and_have_no_execution_owners() -> None:
    forbidden_names = {
        "CapabilityRuntime",
        "JobOwner",
        "JobSupervisor",
        "SQLiteStateStore",
        "RunAdmissionCoordinator",
    }
    for module in (execution, planning, reduction):
        source = inspect.getsource(module)
        assert "asyncio.Lock" not in source
        assert "create_task(" not in source
        assert "execute(" not in source
        assert not forbidden_names & {
            name for name, value in vars(module).items() if inspect.isclass(value)
        }

    admission = graph_admission()
    running = replace(
        admission.tasks[0],
        state=TaskState.RUNNING,
        current_attempt_id="attempt-pure",
        attempt_count=1,
        fencing_epoch=1,
    )
    first = reduce_attempt_failure(
        running,
        failed_at=GRAPH_NOW,
        retryable=True,
        attempt_state=AttemptState.FAILED,
        maximum_attempts=3,
        deadline_at=GRAPH_NOW + timedelta(minutes=5),
    )
    second = reduce_attempt_failure(
        replace(running, attempt_count=2, failure_streak=1),
        failed_at=GRAPH_NOW,
        retryable=True,
        attempt_state=AttemptState.FAILED,
        maximum_attempts=3,
        deadline_at=GRAPH_NOW + timedelta(minutes=5),
    )
    circuit = reduce_attempt_failure(
        replace(running, attempt_count=3, failure_streak=2),
        failed_at=GRAPH_NOW,
        retryable=True,
        attempt_state=AttemptState.FAILED,
        maximum_attempts=3,
        deadline_at=GRAPH_NOW + timedelta(minutes=5),
    )
    assert first.not_before == GRAPH_NOW + timedelta(seconds=1)
    assert second.not_before == GRAPH_NOW + timedelta(seconds=5)
    assert circuit.circuit_open and circuit.task_state is TaskState.BLOCKED


async def test_model_graph_harness_propagates_independent_token_ceilings(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(
        tmp_path,
        loop_max_total_tokens=100_000,
    )
    try:
        loop = integration.supervisor._graph_model_loop
        assert loop is not None
        assert loop._limits.max_total_tokens == 100_000

        admission = integration.build(per_run_max_tokens=80_000)
        worker = next(task for task in admission.tasks if task.role is TaskRole.WORKER)
        assert worker.specification.expected_result_contract["per_run_max_tokens"] == (
            80_000
        )
        planner_template = admission.job.specification.planner_task_template
        assert isinstance(planner_template, Mapping)
        planner_specification = planner_template["specification"]
        assert isinstance(planner_specification, Mapping)
        planner_result = planner_specification["expected_result_contract"]
        assert isinstance(planner_result, Mapping)
        assert planner_result["per_run_max_tokens"] == 80_000
    finally:
        await integration.close()


def test_planner_role_and_cli_control_projection_are_narrow() -> None:
    assert execution.task_role_allows_capability(
        "planner", "jobs.graph.planner_list_tasks"
    )
    assert execution.task_role_allows_capability("planner", "catalog.search")
    assert execution.task_role_allows_capability("planner", "jobs.graph.task_complete")
    assert not execution.task_role_allows_capability("planner", READ_CAPABILITY_ID)
    assert not execution.task_role_allows_capability(
        "worker", "jobs.graph.planner_create_children"
    )

    parsed = build_parser().parse_args(
        [
            "jobs",
            "replace-task",
            "agent-name",
            "job-id",
            "task-id",
            "7",
            "--principal-id",
            "principal-id",
            "--idempotency-key",
            "replace-once",
            "--note",
            "bounded advisory",
        ]
    )
    assert parsed.jobs_command == "replace-task"
    assert parsed.expected_revision == 7
    projection = GraphBlockerProjection(
        job_id="job-id",
        graph_state=GraphState.BLOCKED,
        blockers=({"control_id": "control-id"},),
    )
    assert _graph_blockers_mapping(projection) == {
        "job_id": "job-id",
        "graph_state": "blocked",
        "blockers": ({"control_id": "control-id"},),
    }


async def test_unexpected_discovery_replans_with_immutable_replacement(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(tmp_path)
    admission = integration.build()
    worker = next(item for item in admission.tasks if item.role.value == "worker")
    resource_ids = worker.specification.authority.resource_ids
    script = (
        _tool_response(
            "discover-archive",
            "task_block",
            {
                "kind": "needs_replan",
                "message": "The admitted source needs one bounded replacement.",
                "details": {"evidence_references": ()},
            },
        ),
        _tool_response("list-current", "graph_list_tasks", {}),
        _tool_response(
            "replace-worker",
            "graph_supersede_unstarted",
            {
                "expected_revision": 1,
                "idempotency_key": "replace-discovered-work",
                "task_id": worker.task_id,
                "replacement": {
                    "client_key": "replacement",
                    "title": "Read the discovered bounded source",
                    "description": "Execute the replacement read within root scope.",
                    "result_kind": "test.graph.read",
                    "capability_id": READ_CAPABILITY_ID,
                    "arguments": {"resource_ids": resource_ids},
                    "parent_task_ids": (),
                    "input_result_ids": (),
                    "reduction": False,
                },
            },
        ),
        _tool_response(
            "complete-plan",
            "task_complete",
            {
                "result_kind": "graph.plan",
                "summary": "Created one immutable replacement.",
                "payload": {"replacement": True},
                "evidence_call_ids": ("replace-worker",),
                "artifact_ids": (),
                "residual_risk": None,
                "downstream_constraints": {},
            },
        ),
        _tool_response(
            "replacement-read",
            "graph_read",
            {"resource_ids": resource_ids},
        ),
        _tool_response(
            "complete-replacement",
            "task_complete",
            {
                "result_kind": "test.graph.read",
                "summary": "The replacement read completed.",
                "payload": {"value": 42},
                "evidence_call_ids": ("replacement-read",),
                "artifact_ids": (),
                "residual_risk": None,
                "downstream_constraints": {},
            },
        ),
    )
    provider = MockModelProvider(
        script,
        provider_id=MODEL_ROUTE_ID,
        complete_pricing=True,
    )
    assert integration.supervisor._graph_model_loop is not None
    integration.supervisor._graph_model_loop._model = provider
    integration.provider = provider
    try:
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)

        assert terminal.job.state is GraphState.SUCCEEDED
        original = next(
            item for item in terminal.tasks if item.task_id == worker.task_id
        )
        assert original.state is TaskState.SUPERSEDED
        assert original.superseded_by_task_id is not None
        replacement = next(
            item
            for item in terminal.tasks
            if item.task_id == original.superseded_by_task_id
        )
        assert replacement.supersedes_task_id == original.task_id
        assert replacement.state is TaskState.SUCCEEDED
        planners = tuple(
            item for item in terminal.tasks if item.role.value == "planner"
        )
        assert len(planners) == 1 and planners[0].state is TaskState.SUCCEEDED
        assert any(
            control.kind is ControlKind.NEEDS_REPLAN
            and control.state is ControlState.RESOLVED
            for control in terminal.controls
        )
        assert len(terminal.delivery_ids) == 1
        provider.assert_consumed()
    finally:
        await integration.close()


async def test_adversarial_mutations_reject_foreign_authority_stale_and_cycles(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open_draft_graph(
        tmp_path / "adversarial.sqlite", initialize=True
    )
    try:
        admission = graph_admission()
        await store.admit_graph(admission)
        worker = admission.tasks[0]
        zero_spec = replace(
            worker.specification,
            budgets=(BudgetAmount("work_units", 0),),
        )

        foreign = replace(
            worker,
            job_id="another-job",
            task_id="foreign-task",
            state=TaskState.READY,
            specification=zero_spec,
            task_spec_digest=zero_spec.digest,
        )
        foreign_result = await store.apply_graph_mutation(
            GraphMutationRequest(
                agent_id="agent-1",
                job_id="job-1",
                mutation_id="mutation-foreign",
                actor_kind="owner",
                actor_key="owner",
                idempotency_key="foreign",
                expected_revision=0,
                created_at=GRAPH_NOW + timedelta(seconds=1),
                tasks=(foreign,),
            )
        )
        assert foreign_result.failure_code == "ownership"

        expanded_authority = GraphAuthority(
            capability_ids=("capability.read", "capability.secret"),
            access_modes=("read",),
            operational_effects=("none",),
            sensitivity=worker.specification.authority.sensitivity,
            contract_bindings={
                "capability.read": "sha256:" + "1" * 64,
                "capability.secret": "sha256:" + "2" * 64,
            },
        )
        expanded_spec = replace(zero_spec, authority=expanded_authority)
        expanded = replace(
            worker,
            task_id="expanded-task",
            state=TaskState.READY,
            specification=expanded_spec,
            task_spec_digest=expanded_spec.digest,
            task_scope_digest=expanded_authority.digest,
        )
        expanded_result = await store.apply_graph_mutation(
            GraphMutationRequest(
                agent_id="agent-1",
                job_id="job-1",
                mutation_id="mutation-expanded",
                actor_kind="owner",
                actor_key="owner",
                idempotency_key="expanded",
                expected_revision=0,
                created_at=GRAPH_NOW + timedelta(seconds=2),
                tasks=(expanded,),
            )
        )
        assert expanded_result.failure_code == "authority_expansion"

        stale = replace(
            worker,
            task_id="stale-task",
            state=TaskState.READY,
            specification=zero_spec,
            task_spec_digest=zero_spec.digest,
        )
        stale_result = await store.apply_graph_mutation(
            GraphMutationRequest(
                agent_id="agent-1",
                job_id="job-1",
                mutation_id="mutation-stale",
                actor_kind="owner",
                actor_key="owner",
                idempotency_key="stale",
                expected_revision=9,
                created_at=GRAPH_NOW + timedelta(seconds=3),
                tasks=(stale,),
            )
        )
        assert stale_result.failure_code == "stale_graph_revision"

        first = replace(
            worker,
            task_id="cycle-a",
            state=TaskState.PENDING,
            specification=zero_spec,
            task_spec_digest=zero_spec.digest,
        )
        second = replace(first, task_id="cycle-b")
        cycle_edges = (
            TaskDependency(
                agent_id="agent-1",
                job_id="job-1",
                upstream_task_id="cycle-a",
                downstream_task_id="cycle-b",
                edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                created_at=GRAPH_NOW + timedelta(seconds=4),
                creator_key="owner",
            ),
            TaskDependency(
                agent_id="agent-1",
                job_id="job-1",
                upstream_task_id="cycle-b",
                downstream_task_id="cycle-a",
                edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                created_at=GRAPH_NOW + timedelta(seconds=4),
                creator_key="owner",
            ),
        )
        cycle_result = await store.apply_graph_mutation(
            GraphMutationRequest(
                agent_id="agent-1",
                job_id="job-1",
                mutation_id="mutation-cycle",
                actor_kind="owner",
                actor_key="owner",
                idempotency_key="cycle",
                expected_revision=0,
                created_at=GRAPH_NOW + timedelta(seconds=4),
                tasks=(first, second),
                dependencies=cycle_edges,
            )
        )
        assert cycle_result.failure_code == "graph_cycle"
    finally:
        await store.close()


async def test_finalizer_seal_releases_for_replan_and_mutation_replays(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(tmp_path)
    try:
        admission = integration.build()
        await integration.owner.admit_static_graph(admission)
        worker = next(item for item in admission.tasks if item.role.value == "worker")
        finalizer = next(
            item for item in admission.tasks if item.role.value == "finalizer"
        )
        worker_attempt = await _claim_and_start(
            integration.store,
            agent_id=AGENT_ID,
            job_id=admission.job.job_id,
            task_id=worker.task_id,
            attempt_id="attempt-worker",
            at=integration.clock(),
        )
        await integration.store.complete_graph_attempt(
            task_result(
                agent_id=AGENT_ID,
                job_id=admission.job.job_id,
                task_id=worker.task_id,
                attempt_id=worker_attempt.attempt_id,
                run_id=worker_attempt.run_id,
                completed_at=integration.clock() + timedelta(seconds=1),
            ),
            claim_token=worker_attempt.claim_token,
            fencing_epoch=worker_attempt.fencing_epoch,
            usage=(BudgetAmount("work_units", 1),),
        )
        final_attempt = await _claim_and_start(
            integration.store,
            agent_id=AGENT_ID,
            job_id=admission.job.job_id,
            task_id=finalizer.task_id,
            attempt_id="attempt-finalizer",
            at=integration.clock() + timedelta(seconds=2),
        )
        sealed = await integration.owner.inspect_graph(admission.job.job_id)
        assert sealed is not None
        assert sealed.graph.finalization_attempt_id == final_attempt.attempt_id

        proposed_spec = replace(worker.specification, created_by="owner")
        proposed = replace(
            worker,
            task_id="sealed-child",
            state=TaskState.READY,
            specification=proposed_spec,
            task_spec_digest=proposed_spec.digest,
            created_at=integration.clock() + timedelta(seconds=3),
            updated_at=integration.clock() + timedelta(seconds=3),
        )
        rejected = await integration.owner.mutate_graph(
            GraphMutationRequest(
                agent_id=AGENT_ID,
                job_id=admission.job.job_id,
                mutation_id="mutation-sealed",
                actor_kind="owner",
                actor_key="owner",
                idempotency_key="mutation-sealed",
                expected_revision=sealed.graph.revision,
                created_at=integration.clock() + timedelta(seconds=3),
                tasks=(proposed,),
            )
        )
        assert rejected.failure_code == "finalization_sealed"

        payload = {"message": "More bounded work is required.", "details": {}}
        control = TaskControl(
            agent_id=AGENT_ID,
            job_id=admission.job.job_id,
            task_id=finalizer.task_id,
            control_id="control-finalizer-replan",
            kind=ControlKind.NEEDS_REPLAN,
            state=ControlState.OPEN,
            requesting_attempt_id=final_attempt.attempt_id,
            payload=payload,
            created_at=integration.clock() + timedelta(seconds=4),
            payload_digest=canonical_digest(payload),
        )
        await integration.owner.open_graph_task_control(
            control,
            claim_token=final_attempt.claim_token,
            fencing_epoch=final_attempt.fencing_epoch,
        )
        replanning = await integration.owner.inspect_graph(admission.job.job_id)
        assert replanning is not None
        planner = next(
            item for item in replanning.tasks if item.role.value == "planner"
        )
        blocked_finalizer = next(
            item for item in replanning.tasks if item.task_id == finalizer.task_id
        )
        assert blocked_finalizer.state is TaskState.BLOCKED
        assert replanning.graph.finalization_attempt_id is None
        assert replanning.graph.revision == 1
        finalizer_reservations = (
            await integration.store.list_graph_attempt_reservations(
                AGENT_ID,
                admission.job.job_id,
                finalizer.task_id,
                final_attempt.attempt_id,
            )
        )
        assert [(item.settled, item.reserved) for item in finalizer_reservations] == [
            (1, 1)
        ]
        root_ledger = next(
            item
            for item in replanning.budget_ledgers
            if item.task_id is None and item.dimension == "work_units"
        )
        assert root_ledger.reserved == 0

        planner_attempt = await _claim_and_start(
            integration.store,
            agent_id=AGENT_ID,
            job_id=admission.job.job_id,
            task_id=planner.task_id,
            attempt_id="attempt-planner",
            at=integration.clock() + timedelta(seconds=5),
        )
        child_spec = replace(worker.specification, created_by=planner.task_id)
        child = replace(
            worker,
            task_id="planned-child",
            state=TaskState.READY,
            specification=child_spec,
            task_spec_digest=child_spec.digest,
            created_at=integration.clock() + timedelta(seconds=6),
            updated_at=integration.clock() + timedelta(seconds=6),
        )
        request = GraphMutationRequest(
            agent_id=AGENT_ID,
            job_id=admission.job.job_id,
            mutation_id="mutation-planned-child",
            actor_kind="planner_attempt",
            actor_key=planner.task_id,
            idempotency_key="planned-child",
            expected_revision=1,
            created_at=integration.clock() + timedelta(seconds=6),
            tasks=(child,),
            creator_task_id=planner.task_id,
            creator_attempt_id=planner_attempt.attempt_id,
            claim_token=planner_attempt.claim_token,
            fencing_epoch=planner_attempt.fencing_epoch,
        )
        first = await integration.owner.mutate_graph(request)
        replay = await integration.owner.mutate_graph(request)
        assert first == replay
        released = await integration.owner.inspect_graph(admission.job.job_id)
        assert released is not None
        assert sum(item.task_id == child.task_id for item in released.tasks) == 1
        current_finalizer = next(
            item for item in released.tasks if item.task_id == finalizer.task_id
        )
        assert current_finalizer.state is TaskState.READY
        assert released.graph.finalization_attempt_id is None
        assert (
            next(
                item
                for item in released.controls
                if item.control_id == control.control_id
            ).state
            is ControlState.RESOLVED
        )
        stale_child = replace(
            child,
            task_id="stale-fence-child",
            created_at=integration.clock() + timedelta(seconds=7),
            updated_at=integration.clock() + timedelta(seconds=7),
        )
        with pytest.raises(GraphValidationError, match="stale"):
            await integration.owner.mutate_graph(
                GraphMutationRequest(
                    agent_id=AGENT_ID,
                    job_id=admission.job.job_id,
                    mutation_id="mutation-stale-fence",
                    actor_kind="planner_attempt",
                    actor_key=planner.task_id,
                    idempotency_key="stale-fence",
                    expected_revision=released.graph.revision,
                    created_at=integration.clock() + timedelta(seconds=7),
                    tasks=(stale_child,),
                    creator_task_id=planner.task_id,
                    creator_attempt_id=planner_attempt.attempt_id,
                    claim_token=planner_attempt.claim_token,
                    fencing_epoch=planner_attempt.fencing_epoch + 1,
                )
            )
    finally:
        await integration.close()


async def test_retry_backoff_circuit_and_protocol_limit_are_persisted(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open_draft_graph(
        tmp_path / "retry.sqlite", initialize=True
    )
    try:
        await store.admit_graph(graph_admission())
        for ordinal, at in enumerate(
            (
                GRAPH_NOW,
                GRAPH_NOW + timedelta(seconds=3),
                GRAPH_NOW + timedelta(seconds=10),
            ),
            start=1,
        ):
            attempt = await _claim_and_start(
                store,
                agent_id="agent-1",
                job_id="job-1",
                task_id="worker",
                attempt_id=f"attempt-{ordinal}",
                at=at,
            )
            await store.fail_graph_attempt(
                "agent-1",
                "job-1",
                "worker",
                attempt.attempt_id,
                claim_token=attempt.claim_token,
                fencing_epoch=attempt.fencing_epoch,
                failed_at=at + timedelta(seconds=1),
                retryable=True,
                reason_code="compatible_failure",
            )
            inspection = await store.inspect_graph("agent-1", "job-1")
            assert inspection is not None
            worker = next(item for item in inspection.tasks if item.task_id == "worker")
            if ordinal == 1:
                assert worker.not_before == at + timedelta(seconds=2)
            elif ordinal == 2:
                assert worker.not_before == at + timedelta(seconds=6)

        attention = await store.inspect_graph("agent-1", "job-1")
        assert attention is not None
        worker = next(item for item in attention.tasks if item.task_id == "worker")
        assert worker.state is TaskState.BLOCKED and worker.failure_streak == 3
        assert attention.job.state is GraphState.NEEDS_ATTENTION
        assert any(
            item.kind is ControlKind.RETRY_CIRCUIT_OPEN
            and item.state is ControlState.OPEN
            for item in attention.controls
        )

        await store.admit_graph(graph_admission(job_id="job-protocol"))
        for ordinal, at in enumerate(
            (GRAPH_NOW, GRAPH_NOW + timedelta(seconds=3)), start=1
        ):
            attempt = await _claim_and_start(
                store,
                agent_id="agent-1",
                job_id="job-protocol",
                task_id="worker",
                attempt_id=f"protocol-{ordinal}",
                at=at,
            )
            await store.fail_graph_attempt(
                "agent-1",
                "job-protocol",
                "worker",
                attempt.attempt_id,
                claim_token=attempt.claim_token,
                fencing_epoch=attempt.fencing_epoch,
                failed_at=at + timedelta(seconds=1),
                retryable=True,
                reason_code="protocol_violation",
                attempt_state=AttemptState.PROTOCOL_VIOLATION,
            )
        protocol = await store.inspect_graph("agent-1", "job-protocol")
        assert protocol is not None
        assert protocol.job.state is GraphState.FAILED
        assert len(protocol.attempts) == 2
    finally:
        await store.close()


async def test_human_controls_require_principal_and_derive_only_policy_replacement(
    tmp_path: Path,
) -> None:
    now = [GRAPH_NOW]
    store = await SQLiteStateStore.open_draft_graph(
        tmp_path / "human.sqlite", initialize=True, clock=lambda: now[0]
    )
    owner = JobOwner(
        agent_id="agent-1",
        store=store,
        clock=lambda: now[0],
        id_factory=lambda prefix: f"{prefix}-human",
    )
    try:
        await owner.admit_static_graph(graph_admission())
        attempt = await _claim_and_start(
            store,
            agent_id="agent-1",
            job_id="job-1",
            task_id="worker",
            attempt_id="attempt-input",
            at=now[0],
        )
        payload = {
            "question": "Choose one bounded mode.",
            "response_schema": {
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["safe"]}},
                "required": ["mode"],
                "additionalProperties": False,
            },
            "choices": ("safe",),
            "why_blocked": "A typed choice is required.",
            "affected_downstream_task_ids": ("finalizer",),
            "sensitivity": "restricted",
            "evidence_references": (),
            "expires_at": (GRAPH_NOW + timedelta(hours=1)).isoformat(),
            "default_behavior": "remain_blocked",
        }
        control = TaskControl(
            agent_id="agent-1",
            job_id="job-1",
            task_id="worker",
            control_id="control-input",
            kind=ControlKind.NEEDS_INPUT,
            state=ControlState.OPEN,
            requesting_attempt_id=attempt.attempt_id,
            payload=payload,
            created_at=now[0] + timedelta(seconds=1),
            payload_digest=canonical_digest(payload),
        )
        await owner.open_graph_task_control(
            control,
            claim_token=attempt.claim_token,
            fencing_epoch=attempt.fencing_epoch,
        )
        blockers = await owner.graph_blockers("job-1")
        assert blockers is not None and len(blockers.blockers) == 1
        with pytest.raises(JobError, match="principal"):
            await owner.answer_graph_task_input(
                "job-1",
                "worker",
                control.control_id,
                principal_id="another-principal",
                answer={"mode": "safe"},
            )
        with pytest.raises(JobError, match="response schema"):
            await owner.answer_graph_task_input(
                "job-1",
                "worker",
                control.control_id,
                principal_id="agent-1",
                answer={"mode": "unsafe"},
            )
        now[0] += timedelta(seconds=2)
        answered = await owner.answer_graph_task_input(
            "job-1",
            "worker",
            control.control_id,
            principal_id="agent-1",
            answer={"mode": "safe"},
        )
        assert answered is not None and answered.state is ControlState.RESOLVED

        second = await _claim_and_start(
            store,
            agent_id="agent-1",
            job_id="job-1",
            task_id="worker",
            attempt_id="attempt-replacement",
            at=now[0] + timedelta(seconds=1),
        )
        replacement_payload = {
            "question": "Provide replacement guidance.",
            "response_schema": {"type": "object"},
        }
        replacement_control = TaskControl(
            agent_id="agent-1",
            job_id="job-1",
            task_id="worker",
            control_id="control-replacement",
            kind=ControlKind.NEEDS_INPUT,
            state=ControlState.OPEN,
            requesting_attempt_id=second.attempt_id,
            payload=replacement_payload,
            created_at=now[0] + timedelta(seconds=2),
            payload_digest=canonical_digest(replacement_payload),
        )
        await owner.open_graph_task_control(
            replacement_control,
            claim_token=second.claim_token,
            fencing_epoch=second.fencing_epoch,
        )
        before = await owner.inspect_graph("job-1")
        assert before is not None
        mutation = await owner.replace_graph_task_by_policy(
            "job-1",
            "worker",
            principal_id="agent-1",
            advisory_note="Use the already admitted safe mode.",
            idempotency_key="human-replacement",
            expected_revision=before.graph.revision,
        )
        replay = await owner.replace_graph_task_by_policy(
            "job-1",
            "worker",
            principal_id="agent-1",
            advisory_note="Use the already admitted safe mode.",
            idempotency_key="human-replacement",
            expected_revision=before.graph.revision,
        )
        assert mutation == replay
        assert mutation.failure_code is None
        replaced = await owner.inspect_graph("job-1")
        assert replaced is not None
        original = next(item for item in replaced.tasks if item.task_id == "worker")
        replacement = next(
            item
            for item in replaced.tasks
            if item.task_id == original.superseded_by_task_id
        )
        assert original.state is TaskState.SUPERSEDED
        assert replacement.specification.authority == original.specification.authority
        assert replacement.specification.expected_result_contract == (
            original.specification.expected_result_contract
        )
        assert (
            next(
                item
                for item in replaced.controls
                if item.control_id == replacement_control.control_id
            ).state
            is ControlState.RESOLVED
        )

        arbitrary = replace(
            replacement,
            task_id="human-arbitrary",
            supersedes_task_id=None,
            specification=replace(replacement.specification, created_by="agent-1"),
        )
        arbitrary = replace(
            arbitrary,
            task_spec_digest=arbitrary.specification.digest,
            task_scope_digest=arbitrary.specification.authority.digest,
        )
        denied = await owner.mutate_graph(
            GraphMutationRequest(
                agent_id="agent-1",
                job_id="job-1",
                mutation_id="human-arbitrary",
                actor_kind="human_policy",
                actor_key="agent-1",
                idempotency_key="human-arbitrary",
                expected_revision=replaced.graph.revision,
                created_at=now[0] + timedelta(seconds=3),
                tasks=(arbitrary,),
            )
        )
        assert denied.failure_code == "human_mutation_not_allowed"

        cancelled = await owner.cancel_graph_job("job-1", principal_id="agent-1")
        assert cancelled is not None and cancelled.state is GraphState.CANCELLED
        terminal = await owner.inspect_graph("job-1")
        assert terminal is not None
        assert all(
            control.state is not ControlState.OPEN for control in terminal.controls
        )
    finally:
        await store.close()
