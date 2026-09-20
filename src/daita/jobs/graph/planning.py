"""Pure construction helpers for policy-owned graph planning work."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from ...llm.models import ModelSensitivity
from .models import (
    BudgetAmount,
    GraphAuthority,
    GraphInspection,
    GraphTask,
    GraphTaskSpecification,
    TaskExecutionKind,
    TaskRole,
    TaskState,
)


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{label} is malformed")
    return value


def _authority(value: object) -> GraphAuthority:
    if not isinstance(value, Mapping):
        raise TypeError("planner template authority is malformed")
    sensitivity = value.get("sensitivity")
    if not isinstance(sensitivity, str):
        raise TypeError("planner template sensitivity is malformed")
    bindings = value.get("contract_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("planner template bindings are malformed")
    return GraphAuthority(
        source_ids=_text_tuple(value.get("source_ids", ()), "planner sources"),
        resource_ids=_text_tuple(value.get("resource_ids", ()), "planner resources"),
        connector_ids=_text_tuple(value.get("connector_ids", ()), "planner connectors"),
        capability_ids=_text_tuple(
            value.get("capability_ids", ()), "planner capabilities"
        ),
        access_modes=_text_tuple(value.get("access_modes", ()), "planner access"),
        operational_effects=_text_tuple(
            value.get("operational_effects", ()), "planner effects"
        ),
        model_route_ids=_text_tuple(value.get("model_route_ids", ()), "planner routes"),
        sensitivity=ModelSensitivity(sensitivity),
        contract_bindings=bindings,
    )


def planner_task_from_template(
    inspection: GraphInspection,
    *,
    task_id: str,
    created_at: datetime,
) -> GraphTask:
    """Instantiate one immutable replan task from the root-owned frozen template."""

    template = inspection.job.specification.planner_task_template
    if not isinstance(template, Mapping):
        raise TypeError("this graph has no admitted planner template")
    specification = template.get("specification")
    if not isinstance(specification, Mapping):
        raise TypeError("planner task template specification is malformed")
    authority = _authority(specification.get("authority"))
    raw_budgets = specification.get("budgets")
    if not isinstance(raw_budgets, tuple):
        raise TypeError("planner task template budgets are malformed")
    budgets: list[BudgetAmount] = []
    for raw in raw_budgets:
        if not isinstance(raw, Mapping):
            raise TypeError("planner task template budget is malformed")
        dimension = raw.get("dimension")
        amount = raw.get("amount")
        if not isinstance(dimension, str) or not isinstance(amount, int):
            raise TypeError("planner task template budget is malformed")
        budgets.append(BudgetAmount(dimension, amount))
    expected = specification.get("expected_result_contract")
    if not isinstance(expected, Mapping):
        raise TypeError("planner task result contract is malformed")
    title = specification.get("title")
    description = specification.get("description")
    max_steps = specification.get("max_steps")
    max_wall = specification.get("max_wall_time_seconds")
    created_by = specification.get("created_by")
    priority = template.get("priority")
    if (
        not isinstance(title, str)
        or not isinstance(description, str)
        or not isinstance(max_steps, int)
        or not isinstance(max_wall, int)
        or not isinstance(created_by, str)
        or not isinstance(priority, int)
    ):
        raise TypeError("planner task template is malformed")
    spec = GraphTaskSpecification(
        title=title,
        description=description,
        expected_result_contract=expected,
        authority=authority,
        budgets=tuple(budgets),
        max_steps=max_steps,
        max_wall_time_seconds=max_wall,
        created_by=created_by,
    )
    return GraphTask(
        agent_id=inspection.job.agent_id,
        job_id=inspection.job.job_id,
        task_id=task_id,
        state=TaskState.READY,
        role=TaskRole.PLANNER,
        execution_kind=TaskExecutionKind.MODEL,
        priority=priority,
        not_before=None,
        current_attempt_id=None,
        task_revision=1,
        specification=spec,
        task_spec_digest=spec.digest,
        task_scope_digest=authority.digest,
        attempt_count=0,
        failure_streak=0,
        fencing_epoch=0,
        created_at=created_at,
        updated_at=created_at,
    )


__all__ = ["planner_task_from_template"]
