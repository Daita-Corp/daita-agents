"""Shared helpers extracted from ``test_storage.py``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from daita.capabilities import AccessMode, OperationalEffect
from daita.llm.models import (
    ModelSensitivity,
)
from daita.routines.models import (
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    RoutineState,
    ScheduledRoutine,
    text_digest,
)
from tests.support.capability_runtime import frozen_execution_bindings
from tests.support.distribution import (
    inbox_distribution_plan,
    no_artifact_outcome_contract,
)

NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)


def routine_record(
    *,
    routine_id: str = "routine-1",
    agent_id: str = "agent-1",
    conversation_id: str = "conversation-1",
    next_due_at: datetime | None = NOW,
    state: RoutineState = RoutineState.ACTIVE,
    maximum_consecutive_failures: int = 3,
    consecutive_failures: int = 0,
) -> ScheduledRoutine:
    instruction = "Read the exact admitted resource and report its current value."
    return ScheduledRoutine(
        contract_bindings=frozen_execution_bindings(
            ("catalog.inspect", "data.query"), ("resource-1",), ("mock:routine",)
        ),
        routine_id=routine_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        owner_principal_id="principal-1",
        title="Current value report",
        authorized_instruction=instruction,
        instruction_digest=text_digest(instruction),
        schedule=IntervalSchedule(3_600, NOW),
        schedule_interpreter_revision=1,
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=ReportingMode.ALWAYS,
        precheck=None,
        last_acknowledged_precheck_observation=None,
        allowed_source_ids=("source-1",),
        allowed_connector_binding_ids=(),
        allowed_resource_ids=("resource-1",),
        allowed_capability_ids=("catalog.inspect", "data.query"),
        allowed_access_modes=frozenset({AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("mock:routine",),
        skill_bindings=(),
        outcome_contract=no_artifact_outcome_contract(),
        distribution_plan=inbox_distribution_plan(conversation_id),
        per_run_max_tokens=5_000,
        per_run_max_cost_usd=Decimal("0.05"),
        cumulative_max_tokens=50_000,
        cumulative_max_cost_usd=Decimal("0.50"),
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        reserved_tokens=0,
        reserved_cost_usd=Decimal("0"),
        charged_tokens=0,
        charged_cost_usd=Decimal("0"),
        attempt_count=0,
        occurrence_count=0,
        maximum_consecutive_failures=maximum_consecutive_failures,
        consecutive_failures=consecutive_failures,
        expires_at=NOW + timedelta(days=30),
        next_due_at=next_due_at,
        active_occurrence_id=None,
        last_occurrence_id=None,
        last_delivery_ids=(),
        promotion_evidence=None,
        state=state,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
    )
