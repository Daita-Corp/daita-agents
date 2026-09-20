"""Shared constructors for the isolated draft graph persistence tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from daita.distribution.models import (
    CONVERSATION_INBOX_DESTINATION_REVISION,
    ConversationInboxTarget,
    conversation_inbox_destination_id,
    distribution_plan_digest,
    target_fingerprint,
)
from daita.jobs.graph.models import (
    BudgetAmount,
    BudgetLimit,
    EdgeKind,
    GraphAdmission,
    GraphAuthority,
    GraphDesiredState,
    GraphJob,
    GraphJobSpecification,
    GraphState,
    GraphTask,
    GraphTaskSpecification,
    JobGraph,
    TaskDependency,
    TaskExecutionKind,
    TaskResult,
    TaskRole,
    TaskState,
    canonical_digest,
    topology_digest,
)
from daita.llm.models import ModelSensitivity

GRAPH_NOW = datetime(2026, 9, 16, 12, tzinfo=UTC)


def graph_admission(
    *,
    job_id: str = "job-1",
    agent_id: str = "agent-1",
    worker_state: TaskState = TaskState.READY,
) -> GraphAdmission:
    authority = GraphAuthority(
        capability_ids=("capability.read",),
        access_modes=("read",),
        operational_effects=("none",),
        sensitivity=ModelSensitivity.RESTRICTED,
        contract_bindings={"capability.read": "sha256:" + "1" * 64},
    )
    deadline = GRAPH_NOW + timedelta(hours=1)
    destination_id = conversation_inbox_destination_id("conversation-1")
    target = ConversationInboxTarget(
        conversation_id="conversation-1",
        destination_id=destination_id,
        destination_revision=CONVERSATION_INBOX_DESTINATION_REVISION,
        sensitivity_ceiling=ModelSensitivity.RESTRICTED,
        target_fingerprint=target_fingerprint(
            conversation_id="conversation-1",
            destination_id=destination_id,
            destination_revision=CONVERSATION_INBOX_DESTINATION_REVISION,
            sensitivity_ceiling=ModelSensitivity.RESTRICTED,
        ),
    )
    specification = GraphJobSpecification(
        principal_id=agent_id,
        objective="Complete the deterministic graph test.",
        outcome_contract={"kind": "test"},
        authority=authority,
        distribution_plan_digest=distribution_plan_digest(
            targets=(target,), required_target_count=1
        ),
        budgets=(BudgetLimit("work_units", 4, 1),),
        deadline_at=deadline,
        retry_policy={"max_attempts": 3},
        cancellation_policy={"preserve_evidence": True},
        finalizer_task_template={"kind": "promote"},
    )
    worker_specification = GraphTaskSpecification(
        title="Worker",
        description="Produce the accepted worker result.",
        expected_result_contract={"kind": "worker"},
        authority=authority,
        budgets=(BudgetAmount("work_units", 3),),
        max_steps=2,
        max_wall_time_seconds=120,
        created_by="owner",
    )
    finalizer_authority = GraphAuthority(
        sensitivity=ModelSensitivity.RESTRICTED,
        operational_effects=("none",),
    )
    finalizer_specification = GraphTaskSpecification(
        title="Finalizer",
        description="Promote the authenticated worker result.",
        expected_result_contract={"kind": "final"},
        authority=finalizer_authority,
        budgets=(BudgetAmount("work_units", 1),),
        max_steps=1,
        max_wall_time_seconds=120,
        created_by="owner",
    )
    worker = GraphTask(
        agent_id=agent_id,
        job_id=job_id,
        task_id="worker",
        state=worker_state,
        role=TaskRole.WORKER,
        execution_kind=TaskExecutionKind.MODEL,
        priority=10,
        not_before=None,
        current_attempt_id=None,
        task_revision=1,
        specification=worker_specification,
        task_spec_digest=worker_specification.digest,
        task_scope_digest=worker_specification.authority.digest,
        attempt_count=0,
        failure_streak=0,
        fencing_epoch=0,
        created_at=GRAPH_NOW,
        updated_at=GRAPH_NOW,
    )
    finalizer = GraphTask(
        agent_id=agent_id,
        job_id=job_id,
        task_id="finalizer",
        state=TaskState.PENDING,
        role=TaskRole.FINALIZER,
        execution_kind=TaskExecutionKind.INTERNAL_CAPABILITY,
        priority=1_000,
        not_before=None,
        current_attempt_id=None,
        task_revision=1,
        specification=finalizer_specification,
        task_spec_digest=finalizer_specification.digest,
        task_scope_digest=finalizer_specification.authority.digest,
        attempt_count=0,
        failure_streak=0,
        fencing_epoch=0,
        created_at=GRAPH_NOW,
        updated_at=GRAPH_NOW,
    )
    dependency = TaskDependency(
        agent_id=agent_id,
        job_id=job_id,
        upstream_task_id="worker",
        downstream_task_id="finalizer",
        edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
        created_at=GRAPH_NOW,
        creator_key="owner",
    )
    tasks = (worker, finalizer)
    dependencies = (dependency,)
    job = GraphJob(
        agent_id=agent_id,
        job_id=job_id,
        conversation_id="conversation-1",
        origin_run_id=f"{job_id}:origin-run",
        origin_call_id=f"{job_id}:origin-call",
        state=GraphState.QUEUED,
        desired_state=GraphDesiredState.RUN,
        created_at=GRAPH_NOW,
        updated_at=GRAPH_NOW,
        deadline_at=deadline,
        specification=specification,
        specification_digest=specification.digest,
        finalizer_task_id="finalizer",
    )
    graph = JobGraph(
        agent_id=agent_id,
        job_id=job_id,
        revision=0,
        task_count=len(tasks),
        edge_count=len(dependencies),
        mutation_count=0,
        active_attempt_count=0,
        next_ready_at=GRAPH_NOW if worker_state is TaskState.READY else None,
        finalization_attempt_id=None,
        finalization_started_revision=None,
        created_at=GRAPH_NOW,
        updated_at=GRAPH_NOW,
        topology_digest=topology_digest(tasks, dependencies),
    )
    return GraphAdmission(job=job, graph=graph, tasks=tasks, dependencies=dependencies)


def task_result(
    *,
    job_id: str = "job-1",
    agent_id: str = "agent-1",
    task_id: str = "worker",
    attempt_id: str = "attempt-1",
    run_id: str = "run-1",
    result_id: str = "result-1",
    completed_at: datetime = GRAPH_NOW + timedelta(seconds=2),
) -> TaskResult:
    schema_digest = canonical_digest({"kind": "test_result"})
    digest_material: dict[str, object] = {
        "agent_id": agent_id,
        "job_id": job_id,
        "task_id": task_id,
        "result_id": result_id,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "result_kind": "test_result",
        "schema_digest": schema_digest,
        "payload": {"answer": 42},
        "summary": "Accepted deterministic result.",
        "sensitivity": ModelSensitivity.RESTRICTED.value,
        "provenance": {"source": "test"},
        "artifact_ids": (),
        "effect_receipt_ids": (),
        "verification": {"accepted": True},
        "residual_risk": None,
        "downstream_constraints": {},
        "completed_at": completed_at.isoformat(),
    }
    return TaskResult(
        agent_id=agent_id,
        job_id=job_id,
        task_id=task_id,
        result_id=result_id,
        attempt_id=attempt_id,
        run_id=run_id,
        result_kind="test_result",
        schema_digest=schema_digest,
        payload={"answer": 42},
        summary="Accepted deterministic result.",
        sensitivity=ModelSensitivity.RESTRICTED,
        provenance={"source": "test"},
        artifact_ids=(),
        effect_receipt_ids=(),
        verification={"accepted": True},
        residual_risk=None,
        downstream_constraints={},
        completed_at=completed_at,
        result_digest=canonical_digest(digest_material),
    )


def with_topology(
    admission: GraphAdmission,
    *,
    tasks: tuple[GraphTask, ...],
    dependencies: tuple[TaskDependency, ...],
) -> GraphAdmission:
    graph = replace(
        admission.graph,
        task_count=len(tasks),
        edge_count=len(dependencies),
        topology_digest=topology_digest(tasks, dependencies),
    )
    return replace(admission, graph=graph, tasks=tasks, dependencies=dependencies)


__all__ = ["GRAPH_NOW", "graph_admission", "task_result", "with_topology"]
