"""Shared helpers extracted from ``test_storage.py``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from daita.capabilities import (
    AccessMode,
    ExecutionScope,
    ExecutionScopeKind,
    OperationalEffect,
)
from daita.llm.models import (
    ModelSensitivity,
)
from daita.routines.models import (
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineSlotKind,
    RoutineState,
    ScheduledRoutine,
    text_digest,
)
from daita.routines.schedule import occurrence_id, scheduled_slot_key
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


def execution_scope(occurrence: RoutineOccurrence) -> ExecutionScope:
    return ExecutionScope(
        contract_bindings=frozen_execution_bindings(
            ("catalog.inspect", "data.query"), ("resource-1",), ("mock:routine",)
        ),
        scope_id=f"scope:{occurrence.occurrence_id}",
        revision=1,
        agent_id=occurrence.agent_id,
        principal_id="principal-1",
        grant_id=f"routine:{occurrence.routine_id}:revision:{occurrence.routine_revision}",
        job_id=None,
        job_revision=None,
        allowed_source_ids=("source-1",),
        allowed_resource_ids=("resource-1",),
        allowed_capability_ids=("catalog.inspect", "data.query"),
        allowed_access_modes=frozenset({AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("mock:routine",),
        per_run_max_cost_usd=Decimal("0.05"),
        per_run_max_tokens=5_000,
        distribution_plan_digest=inbox_distribution_plan("conversation-1").plan_digest,
        routine_id=occurrence.routine_id,
        routine_revision=occurrence.routine_revision,
        occurrence_id=occurrence.occurrence_id,
        allowed_connector_binding_ids=(),
        scope_kind=ExecutionScopeKind.SCHEDULED_ROUTINE,
    )


def occurrence_record(
    *,
    routine: ScheduledRoutine | None = None,
    scheduled_for: datetime = NOW,
) -> RoutineOccurrence:
    routine = routine or routine_record()
    slot_key = scheduled_slot_key(
        routine.routine_id,
        routine.revision,
        scheduled_for,
    )
    return RoutineOccurrence(
        occurrence_id=occurrence_id(routine.routine_id, slot_key),
        agent_id=routine.agent_id,
        routine_id=routine.routine_id,
        routine_revision=routine.revision,
        slot_kind=RoutineSlotKind.SCHEDULED,
        slot_key=slot_key,
        scheduled_for=scheduled_for,
        claimed_at=NOW,
        claim_token="claim-1",
        lease_expires_at=NOW + timedelta(seconds=30),
        precheck_observation=None,
        execution_scope=None,
        execution_scope_digest=None,
        reserved_run_id=None,
        reserved_tokens=5_000,
        reserved_cost_usd=Decimal("0.05"),
        charged_tokens=0,
        charged_cost_usd=Decimal("0"),
        run_bound_at=None,
        run_terminal_at=None,
        conclusion_digest=None,
        terminal_run_id=None,
        delivery_ids=(),
        attempt_count=1,
        failure_code=None,
        retry_at=None,
        disposition=RoutineOccurrenceDisposition.CLAIMED,
        created_at=NOW,
        updated_at=NOW,
    )
