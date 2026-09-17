from __future__ import annotations

import random
from dataclasses import replace

import pytest

from daita.jobs.graph.models import EdgeKind, TaskDependency, topology_digest
from daita.jobs.graph.validation import GraphValidationError, validate_graph_topology
from tests.support.graph import GRAPH_NOW, graph_admission

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("seed", range(64))
def test_seeded_dag_topologies_accept_only_acyclic_edge_sets(seed: int):
    randomizer = random.Random(seed)
    admission = graph_admission(job_id=f"job-{seed}")
    template = admission.tasks[0]
    task_count = randomizer.randint(2, 12)
    tasks = tuple(
        replace(template, task_id=f"task-{index}") for index in range(task_count)
    )
    dependencies = tuple(
        TaskDependency(
            agent_id=template.agent_id,
            job_id=template.job_id,
            upstream_task_id=f"task-{upstream}",
            downstream_task_id=f"task-{downstream}",
            edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
            created_at=GRAPH_NOW,
            creator_key=f"seed-{seed}",
        )
        for upstream in range(task_count)
        for downstream in range(upstream + 1, task_count)
        if randomizer.random() < 0.16
    )
    graph = replace(
        admission.graph,
        task_count=task_count,
        edge_count=len(dependencies),
        topology_digest=topology_digest(tasks, dependencies),
    )

    validate_graph_topology(
        tasks,
        dependencies,
        graph=graph,
        max_tasks=12,
        max_edges=66,
        max_depth=12,
        max_direct_parents=11,
    )

    if task_count > 1:
        cycle = (
            *dependencies,
            TaskDependency(
                agent_id=template.agent_id,
                job_id=template.job_id,
                upstream_task_id="task-0",
                downstream_task_id="task-1",
                edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                created_at=GRAPH_NOW,
                creator_key=f"seed-{seed}-forward",
            ),
            TaskDependency(
                agent_id=template.agent_id,
                job_id=template.job_id,
                upstream_task_id="task-1",
                downstream_task_id="task-0",
                edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                created_at=GRAPH_NOW,
                creator_key=f"seed-{seed}-reverse",
            ),
        )
        graph = replace(
            graph,
            edge_count=len(cycle),
            topology_digest=topology_digest(tasks, cycle),
        )
        with pytest.raises(GraphValidationError):
            validate_graph_topology(
                tasks,
                cycle,
                graph=graph,
                max_tasks=12,
                max_edges=68,
                max_depth=12,
                max_direct_parents=12,
            )
