from __future__ import annotations

from dataclasses import replace

import pytest

from daita.jobs.graph.models import (
    BudgetAmount,
    EdgeKind,
    GraphAuthority,
    GraphTask,
    GraphTaskSpecification,
    TaskDependency,
    TaskExecutionKind,
    TaskRole,
    TaskState,
    topology_digest,
)
from daita.jobs.graph.validation import (
    GraphValidationError,
    require_graph_transition,
    validate_graph_admission,
    validate_graph_topology,
)
from daita.llm.models import ModelSensitivity
from tests.support.graph import GRAPH_NOW, graph_admission, with_topology

pytestmark = pytest.mark.unit


def test_valid_admission_freezes_effect_free_authority_and_finalizer():
    admission = graph_admission()

    validate_graph_admission(admission)

    assert admission.job.specification.effect_mode == "disabled"
    assert admission.tasks[-1].role is TaskRole.FINALIZER
    assert admission.graph.topology_digest == topology_digest(
        admission.tasks, admission.dependencies
    )


def test_cycle_is_rejected_before_persistence():
    admission = graph_admission()
    reverse = TaskDependency(
        agent_id="agent-1",
        job_id="job-1",
        upstream_task_id="finalizer",
        downstream_task_id="worker",
        edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
        created_at=GRAPH_NOW,
        creator_key="owner",
    )
    invalid = with_topology(
        admission,
        tasks=admission.tasks,
        dependencies=(*admission.dependencies, reverse),
    )

    with pytest.raises(GraphValidationError, match="cycle"):
        validate_graph_admission(invalid)


def test_cross_job_task_and_authority_expansion_are_rejected():
    admission = graph_admission()
    worker = admission.tasks[0]
    foreign = replace(worker, job_id="job-2")
    invalid = with_topology(
        admission,
        tasks=(foreign, admission.tasks[1]),
        dependencies=admission.dependencies,
    )
    with pytest.raises(GraphValidationError, match="another graph"):
        validate_graph_admission(invalid)

    expanded_authority = GraphAuthority(
        source_ids=("ungranted-source",),
        sensitivity=ModelSensitivity.RESTRICTED,
    )
    expanded_specification = replace(worker.specification, authority=expanded_authority)
    expanded = replace(
        worker,
        specification=expanded_specification,
        task_spec_digest=expanded_specification.digest,
        task_scope_digest=expanded_authority.digest,
    )
    invalid = with_topology(
        admission,
        tasks=(expanded, admission.tasks[1]),
        dependencies=admission.dependencies,
    )
    with pytest.raises(GraphValidationError, match="authority"):
        validate_graph_admission(invalid)


def test_worker_budget_cannot_consume_control_reserve():
    admission = graph_admission()
    worker = admission.tasks[0]
    specification = replace(
        worker.specification, budgets=(BudgetAmount("work_units", 4),)
    )
    expanded = replace(
        worker,
        specification=specification,
        task_spec_digest=specification.digest,
    )
    invalid = with_topology(
        admission,
        tasks=(expanded, admission.tasks[1]),
        dependencies=admission.dependencies,
    )

    with pytest.raises(GraphValidationError, match="ordinary task ceilings"):
        validate_graph_admission(invalid)


def test_terminal_graph_transition_is_immutable():
    with pytest.raises(GraphValidationError, match="not allowed"):
        require_graph_transition(
            graph_admission().job.state.SUCCEEDED,
            graph_admission().job.state.ACTIVE,
        )


def test_depth_and_parent_bounds_are_enforced():
    admission = graph_admission()
    template = admission.tasks[0]
    tasks: list[GraphTask] = []
    dependencies: list[TaskDependency] = []
    for index in range(5):
        task_id = f"task-{index}"
        task = replace(
            template,
            task_id=task_id,
            state=TaskState.PENDING,
            role=TaskRole.WORKER,
            execution_kind=TaskExecutionKind.MODEL,
        )
        tasks.append(task)
        if index:
            dependencies.append(
                TaskDependency(
                    agent_id=task.agent_id,
                    job_id=task.job_id,
                    upstream_task_id=f"task-{index - 1}",
                    downstream_task_id=task_id,
                    edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                    created_at=GRAPH_NOW,
                    creator_key="owner",
                )
            )
    graph = replace(
        admission.graph,
        task_count=len(tasks),
        edge_count=len(dependencies),
        topology_digest=topology_digest(tuple(tasks), tuple(dependencies)),
    )
    with pytest.raises(GraphValidationError, match="depth"):
        validate_graph_topology(
            tuple(tasks),
            tuple(dependencies),
            graph=graph,
            max_tasks=5,
            max_edges=4,
            max_depth=4,
            max_direct_parents=2,
        )


def test_graph_v1_rejects_effectful_authority():
    with pytest.raises(ValueError, match="effect-free"):
        GraphAuthority(operational_effects=("external_write",))


def test_task_specification_requires_bounded_wall_time():
    authority = GraphAuthority()
    with pytest.raises(ValueError, match="wall time"):
        GraphTaskSpecification(
            title="Too long",
            description="Invalid task.",
            expected_result_contract={},
            authority=authority,
            budgets=(),
            max_steps=1,
            max_wall_time_seconds=301,
            created_by="owner",
        )
