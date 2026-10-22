"""Scheduled-routine persistence for the SQL state store.

Every public function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns validation before admission and the transaction boundary.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256

from .._json import FrozenJsonObject, canonical_json
from ..capabilities import ExecutionScope
from ..distribution.models import (
    Delivery,
    DeliverySubjectKind,
    OutcomeArtifactReference,
    OutcomeConclusionKind,
    OutcomeState,
    conclusion_preview_projection,
    outcome_artifact_reference,
    validate_outcome_artifact_references,
)
from ..distribution.owner import construct_logical_delivery
from ..llm.errors import ProviderErrorCode
from ..llm.models import MessageRole, ToolResultBlock
from ..llm.pricing import STALL_USAGE_UNREPORTED, CostEstimateStatus
from ..loop.models import LoopExitKind, RunOrigin, Transcript
from ..routines.models import (
    ACTIVE_OCCURRENCE_DISPOSITIONS,
    MAX_ACTIVE_ROUTINES_PER_AGENT,
    MAX_ROUTINE_ATTEMPTS,
    MAX_SCHEDULED_ROUTINES_PER_AGENT,
    ROUTINE_CLAIM_LEASE_SECONDS,
    ResourceRevisionObservation,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineSlotKind,
    RoutineState,
    ScheduledRoutine,
)
from ..routines.schedule import (
    first_slot,
    manual_slot_key,
    next_slot,
    occurrence_id as routine_occurrence_id,
    scheduled_slot_key,
    select_due_slot,
)
from .sql_connection import SQLConnection
from .sql_deliveries import reserve_delivery_slot
from .sqlite_codecs import (
    decode_delivery,
    decode_loop_exit,
    decode_message,
    decode_receipt,
    decode_routine_occurrence,
    decode_run_input,
    decode_scheduled_routine,
    encode_delivery,
    encode_routine_occurrence,
    encode_scheduled_routine,
)
from .sqlite_records import (
    EffectOutcome,
    EffectUnresolvedError,
    effect_receipt_text as _effect_receipt_text,
)


def _insert_delivery(connection: SQLConnection, delivery: Delivery) -> None:
    if not reserve_delivery_slot(connection, delivery.agent_id):
        raise ValueError("delivery_retention_limit_exceeded")
    connection.execute(
        """INSERT INTO deliveries(
               agent_id, delivery_id, conversation_id, subject_kind, subject_id,
               logical_key, target_kind, target_fingerprint, state,
               created_at_us, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            delivery.agent_id,
            delivery.delivery_id,
            delivery.conversation_id,
            delivery.subject_kind.value,
            delivery.subject_id,
            delivery.logical_key,
            "conversation_inbox",
            delivery.target.target_fingerprint,
            delivery.visibility_state.value,
            _datetime_us(delivery.created_at),
            encode_delivery(delivery),
        ),
    )


def _datetime_us(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("routine query instant must be timezone-aware UTC")
    delta = value - datetime(1970, 1, 1, tzinfo=UTC)
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _load_routine_row(
    connection: SQLConnection,
    agent_id: str,
    routine_id: str,
) -> tuple[ScheduledRoutine, str] | None:
    row = connection.execute(
        "SELECT data FROM scheduled_routines WHERE agent_id = ? AND routine_id = ?",
        (agent_id, routine_id),
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], str):
        raise RuntimeError("stored scheduled routine payload is invalid")
    return (
        decode_scheduled_routine(
            row[0],
            agent_id=agent_id,
            routine_id=routine_id,
        ),
        row[0],
    )


def _replace_routine_row(
    connection: SQLConnection,
    current_data: str,
    routine: ScheduledRoutine,
) -> None:
    result = connection.execute(
        """UPDATE scheduled_routines
           SET conversation_id = ?, state = ?, next_due_at_us = ?, data = ?
           WHERE agent_id = ? AND routine_id = ? AND data = ?""",
        (
            routine.conversation_id,
            routine.state.value,
            _datetime_us(routine.next_due_at),
            encode_scheduled_routine(routine),
            routine.agent_id,
            routine.routine_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise RuntimeError("scheduled routine changed during its transition")


def _load_routine_occurrence_row(
    connection: SQLConnection,
    agent_id: str,
    occurrence_id: str,
) -> tuple[RoutineOccurrence, str] | None:
    row = connection.execute(
        "SELECT data FROM routine_occurrences WHERE agent_id = ? AND occurrence_id = ?",
        (agent_id, occurrence_id),
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], str):
        raise RuntimeError("stored routine occurrence payload is invalid")
    return (
        decode_routine_occurrence(
            row[0],
            agent_id=agent_id,
            occurrence_id=occurrence_id,
        ),
        row[0],
    )


def _replace_routine_occurrence_row(
    connection: SQLConnection,
    current_data: str,
    occurrence: RoutineOccurrence,
) -> None:
    result = connection.execute(
        """UPDATE routine_occurrences
           SET state = ?, lease_expires_at_us = ?, reserved_run_id = ?, data = ?
           WHERE agent_id = ? AND occurrence_id = ? AND data = ?""",
        (
            occurrence.disposition.value,
            _datetime_us(occurrence.lease_expires_at),
            occurrence.reserved_run_id,
            encode_routine_occurrence(occurrence),
            occurrence.agent_id,
            occurrence.occurrence_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise RuntimeError("routine occurrence changed during its transition")


def _insert_routine_occurrence(
    connection: SQLConnection,
    occurrence: RoutineOccurrence,
) -> None:
    connection.execute(
        """INSERT INTO routine_occurrences(
               agent_id, occurrence_id, routine_id, routine_revision, slot_key,
               state, lease_expires_at_us, reserved_run_id, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            occurrence.agent_id,
            occurrence.occurrence_id,
            occurrence.routine_id,
            occurrence.routine_revision,
            occurrence.slot_key,
            occurrence.disposition.value,
            _datetime_us(occurrence.lease_expires_at),
            occurrence.reserved_run_id,
            encode_routine_occurrence(occurrence),
        ),
    )


# Cost-estimate codes that mean "no price", never "unknown token usage".
_UNPRICED_COST_CODES = frozenset(
    {"subscription_billing", "pricing_schedule_unavailable", "pricing_rate_unavailable"}
)


def _cost_ceiling_widened(proposed: Decimal | None, current: Decimal | None) -> bool:
    """A removed dollar ceiling widens authority just as a larger one does."""
    if current is None:
        return False
    return proposed is None or proposed > current


def _require_effects_unblocked(
    connection: SQLConnection,
    agent_id: str,
    *,
    run_id: str | None = None,
    routine_id: str | None = None,
) -> None:
    _effect_receipt_text(agent_id, "effect agent")
    rows = connection.execute(
        "SELECT id, COUNT(*) OVER() FROM effect_receipts WHERE agent_id = ? AND unresolved = 1 AND (CAST(? AS TEXT) IS NULL OR routine_id = ? OR run_id = ?) ORDER BY id LIMIT 20",
        (agent_id, routine_id, routine_id, run_id),
    ).fetchall()
    if rows:
        raise EffectUnresolvedError(
            tuple(row[0] for row in rows), int(rows[0][1]) - len(rows)
        )


def admit_routine(
    connection: SQLConnection,
    normalized: ScheduledRoutine,
    encoded: str,
    *,
    clock: Callable[[], datetime],
) -> ScheduledRoutine:
    if normalized.capability_grants:
        _require_effects_unblocked(connection, normalized.agent_id)
    row = connection.execute(
        "SELECT 1 FROM scheduled_routines WHERE agent_id = ? AND routine_id = ?",
        (normalized.agent_id, normalized.routine_id),
    ).fetchone()
    if row is not None:
        raise ValueError("routine_identity_already_exists")
    count = connection.execute(
        "SELECT COUNT(*) FROM scheduled_routines WHERE agent_id = ?",
        (normalized.agent_id,),
    ).fetchone()
    assert count is not None
    if int(count[0]) >= MAX_SCHEDULED_ROUTINES_PER_AGENT:
        raise ValueError("routine_retention_limit_exceeded")
    active_count = connection.execute(
        "SELECT COUNT(*) FROM scheduled_routines WHERE agent_id = ? AND state = ?",
        (normalized.agent_id, RoutineState.ACTIVE.value),
    ).fetchone()
    assert active_count is not None
    if (
        normalized.state is RoutineState.ACTIVE
        and int(active_count[0]) >= MAX_ACTIVE_ROUTINES_PER_AGENT
    ):
        raise ValueError("routine_active_limit_exceeded")
    connection.execute(
        """INSERT INTO scheduled_routines(
               agent_id, routine_id, conversation_id, state,
               next_due_at_us, data
           ) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            normalized.agent_id,
            normalized.routine_id,
            normalized.conversation_id,
            normalized.state.value,
            _datetime_us(normalized.next_due_at),
            encoded,
        ),
    )
    if normalized.run_immediately:
        if normalized.state is not RoutineState.ACTIVE:
            raise ValueError("routine_immediate_requires_active_creation")
        claimed_at = clock()
        if claimed_at >= normalized.expires_at:
            raise ValueError("routine_expired")
        following = next_slot(
            normalized.schedule,
            after=claimed_at,
            expires_at=normalized.expires_at,
        )
        immediate = replace(normalized, next_due_at=following)
        _replace_routine_row(connection, encoded, immediate)
        immediate_encoded = encode_scheduled_routine(immediate)
        creation_identity = f"create:{normalized.routine_id}"
        _claim_routine_slot(
            connection,
            immediate,
            immediate_encoded,
            slot_kind=RoutineSlotKind.MANUAL,
            slot_key=manual_slot_key(
                normalized.routine_id, normalized.revision, creation_identity
            ),
            scheduled_for=claimed_at,
            claimed_at=claimed_at,
            claim_token=creation_identity,
        )
        claimed = _load_routine_row(
            connection, normalized.agent_id, normalized.routine_id
        )
        assert claimed is not None
        return claimed[0]
    return normalized


def list_routines(
    connection: SQLConnection,
    agent_id: str,
    *,
    states: frozenset[RoutineState],
    limit: int,
) -> tuple[ScheduledRoutine, ...]:
    rows = tuple(
        connection.execute(
            "SELECT routine_id, data FROM scheduled_routines WHERE agent_id = ?",
            (agent_id,),
        )
    )
    if len(rows) > MAX_SCHEDULED_ROUTINES_PER_AGENT:
        raise RuntimeError("stored routine count exceeds its fixed bound")
    decoded = tuple(
        decode_scheduled_routine(data, agent_id=agent_id, routine_id=routine_id)
        for routine_id, data in rows
    )
    selected = tuple(item for item in decoded if not states or item.state in states)
    return tuple(
        sorted(
            selected,
            key=lambda item: (item.updated_at, item.routine_id),
            reverse=True,
        )[:limit]
    )


def list_occurrences(
    connection: SQLConnection,
    agent_id: str,
    routine_id: str,
    *,
    limit: int,
) -> tuple[RoutineOccurrence, ...]:
    rows = tuple(
        connection.execute(
            """SELECT occurrence_id, data FROM routine_occurrences
               WHERE agent_id = ? AND routine_id = ?
               ORDER BY occurrence_id DESC LIMIT ?""",
            (agent_id, routine_id, limit),
        )
    )
    return tuple(
        decode_routine_occurrence(data, agent_id=agent_id, occurrence_id=occurrence_id)
        for occurrence_id, data in rows
    )


def list_active_occurrences(
    connection: SQLConnection,
    agent_id: str,
    *,
    limit: int,
) -> tuple[RoutineOccurrence, ...]:
    states = sorted(item.value for item in ACTIVE_OCCURRENCE_DISPOSITIONS)
    rows = tuple(
        connection.execute(
            "SELECT occurrence_id, data FROM routine_occurrences "
            f"WHERE agent_id = ? AND state IN ({', '.join('?' for _ in states)}) "
            "ORDER BY occurrence_id DESC LIMIT ?",
            (agent_id, *states, limit),
        )
    )
    return tuple(
        decode_routine_occurrence(data, agent_id=agent_id, occurrence_id=occurrence_id)
        for occurrence_id, data in rows
    )


def load_occurrences(
    connection: SQLConnection,
    agent_id: str,
    occurrence_ids: tuple[str, ...],
) -> tuple[RoutineOccurrence, ...]:
    if not occurrence_ids:
        return ()
    rows = tuple(
        connection.execute(
            "SELECT occurrence_id, data FROM routine_occurrences "
            "WHERE agent_id = ? AND occurrence_id IN "
            f"({', '.join('?' for _ in occurrence_ids)}) ORDER BY occurrence_id",
            (agent_id, *occurrence_ids),
        )
    )
    return tuple(
        decode_routine_occurrence(data, agent_id=agent_id, occurrence_id=occurrence_id)
        for occurrence_id, data in rows
    )


def revise_routine(
    connection: SQLConnection,
    routine: ScheduledRoutine,
    *,
    expected_revision: int,
) -> ScheduledRoutine | None:
    loaded = _load_routine_row(connection, routine.agent_id, routine.routine_id)
    if loaded is None:
        return None
    current, encoded = loaded
    if routine.capability_grants:
        _require_effects_unblocked(
            connection, routine.agent_id, routine_id=routine.routine_id
        )
        if (
            routine.capability_grants != current.capability_grants
            or routine.contract_bindings != current.contract_bindings
            or routine.allowed_source_ids != current.allowed_source_ids
            or routine.allowed_resource_ids != current.allowed_resource_ids
            or routine.allowed_connector_binding_ids
            != current.allowed_connector_binding_ids
            or routine.allowed_capability_ids != current.allowed_capability_ids
            or routine.sensitivity_ceiling != current.sensitivity_ceiling
            or routine.instruction_digest != current.instruction_digest
            or routine.schedule != current.schedule
            or routine.misfire_policy != current.misfire_policy
            or routine.skill_bindings != current.skill_bindings
            or routine.eligible_model_routes != current.eligible_model_routes
            or routine.outcome_contract != current.outcome_contract
            or routine.distribution_plan != current.distribution_plan
            or routine.expires_at > current.expires_at
            or routine.per_run_max_tokens > current.per_run_max_tokens
            or _cost_ceiling_widened(
                routine.per_run_max_cost_usd, current.per_run_max_cost_usd
            )
            or routine.cumulative_max_tokens > current.cumulative_max_tokens
            or _cost_ceiling_widened(
                routine.cumulative_max_cost_usd,
                current.cumulative_max_cost_usd,
            )
            or routine.cumulative_max_attempts > current.cumulative_max_attempts
            or routine.cumulative_max_occurrences > current.cumulative_max_occurrences
            or routine.maximum_consecutive_failures
            > current.maximum_consecutive_failures
        ):
            _require_effects_unblocked(connection, routine.agent_id)
    if current.revision != expected_revision:
        return None
    if current.active_occurrence_id is not None:
        raise ValueError("routine_has_active_occurrence")
    if (
        routine.revision != current.revision + 1
        or routine.agent_id != current.agent_id
        or routine.routine_id != current.routine_id
        or routine.conversation_id != current.conversation_id
        or routine.owner_principal_id != current.owner_principal_id
        or routine.created_at != current.created_at
        or routine.reserved_tokens != current.reserved_tokens
        or routine.reserved_cost_usd != current.reserved_cost_usd
        or routine.charged_tokens != current.charged_tokens
        or routine.charged_cost_usd != current.charged_cost_usd
        or routine.attempt_count != current.attempt_count
        or routine.occurrence_count != current.occurrence_count
    ):
        raise ValueError("routine revision changed immutable lifecycle state")
    if routine.state is RoutineState.ACTIVE:
        due = first_slot(
            routine.schedule,
            not_before=routine.updated_at,
            expires_at=routine.expires_at,
        )
        if due is None:
            raise ValueError("routine_schedule_expired")
    else:
        due = None
    normalized = replace(
        routine,
        next_due_at=due,
        active_occurrence_id=None,
    )
    _replace_routine_row(connection, encoded, normalized)
    return normalized


def transition_routine(
    connection: SQLConnection,
    agent_id: str,
    routine_id: str,
    *,
    expected_revision: int,
    state: RoutineState,
    transitioned_at: datetime,
) -> ScheduledRoutine | None:
    loaded = _load_routine_row(connection, agent_id, routine_id)
    if loaded is None:
        return None
    current, encoded = loaded
    if current.revision != expected_revision:
        return None
    if current.active_occurrence_id is not None:
        raise ValueError("routine_has_active_occurrence")
    if state is RoutineState.ACTIVE:
        if current.model_budget_exhausted:
            raise ValueError("routine_model_budget_exhausted")
        if current.occurrence_allowance_exhausted:
            raise ValueError("routine_occurrence_budget_exhausted")
        if current.capability_grants:
            _require_effects_unblocked(connection, agent_id, routine_id=routine_id)
        due = next_slot(
            current.schedule,
            after=transitioned_at,
            expires_at=current.expires_at,
        )
        if due is None:
            state_value = RoutineState.EXPIRED
            due = None
        else:
            state_value = RoutineState.ACTIVE
    else:
        state_value = state
        due = None
    updated = replace(
        current,
        state=state_value,
        next_due_at=due,
        revision=current.revision + 1,
        updated_at=transitioned_at,
    )
    _replace_routine_row(connection, encoded, updated)
    return updated


def next_deadline(connection: SQLConnection, agent_id: str) -> datetime | None:
    rows = tuple(
        connection.execute(
            """SELECT routine_id, data FROM scheduled_routines
               WHERE agent_id = ? AND state = ?
                 AND next_due_at_us IS NOT NULL
               ORDER BY next_due_at_us, routine_id""",
            (agent_id, RoutineState.ACTIVE.value),
        )
    )
    for routine_id, data in rows:
        routine = decode_scheduled_routine(
            data, agent_id=agent_id, routine_id=routine_id
        )
        if routine.active_occurrence_id is None:
            return routine.next_due_at
    return None


def claim_due_occurrence(
    connection: SQLConnection,
    agent_id: str,
    routine_id: str,
    *,
    expected_revision: int,
    expected_due_at: datetime,
    claimed_at: datetime,
    claim_token: str,
) -> RoutineOccurrence | None:
    loaded = _load_routine_row(connection, agent_id, routine_id)
    if loaded is None:
        return None
    current, encoded = loaded
    if (
        current.revision != expected_revision
        or current.state is not RoutineState.ACTIVE
        or current.next_due_at != expected_due_at
    ):
        return None
    if current.active_occurrence_id is not None:
        active = _load_routine_occurrence_row(
            connection,
            agent_id,
            current.active_occurrence_id,
        )
        if active is not None and active[0].slot_kind is RoutineSlotKind.SCHEDULED:
            return active[0]
        return None
    selection = select_due_slot(
        current.schedule,
        materialized_due_at=expected_due_at,
        now=claimed_at,
        expires_at=current.expires_at,
        misfire_policy=current.misfire_policy,
    )
    if selection.selected_at is None:
        skipped = replace(
            current,
            next_due_at=selection.next_due_at,
            updated_at=claimed_at,
            state=(
                RoutineState.ACTIVE
                if selection.next_due_at is not None
                else RoutineState.EXPIRED
            ),
        )
        _replace_routine_row(connection, encoded, skipped)
        return None
    return _claim_routine_slot(
        connection,
        current,
        encoded,
        slot_kind=RoutineSlotKind.SCHEDULED,
        slot_key=scheduled_slot_key(
            current.routine_id,
            current.revision,
            selection.selected_at,
        ),
        scheduled_for=selection.selected_at,
        claimed_at=claimed_at,
        claim_token=claim_token,
    )


def claim_manual_occurrence(
    connection: SQLConnection,
    agent_id: str,
    routine_id: str,
    *,
    expected_revision: int,
    slot: str,
    claimed_at: datetime,
    claim_token: str,
) -> RoutineOccurrence | None:
    existing = connection.execute(
        """SELECT occurrence_id, data FROM routine_occurrences
           WHERE agent_id = ? AND routine_id = ?
             AND routine_revision = ? AND slot_key = ?""",
        (agent_id, routine_id, expected_revision, slot),
    ).fetchone()
    if existing is not None:
        return decode_routine_occurrence(
            existing[1],
            agent_id=agent_id,
            occurrence_id=existing[0],
        )
    loaded = _load_routine_row(connection, agent_id, routine_id)
    if loaded is None:
        return None
    current, encoded = loaded
    if current.capability_grants:
        _require_effects_unblocked(connection, agent_id, routine_id=routine_id)
    if (
        current.revision != expected_revision
        or current.state is not RoutineState.ACTIVE
        or current.active_occurrence_id is not None
        or claimed_at >= current.expires_at
    ):
        return None
    return _claim_routine_slot(
        connection,
        current,
        encoded,
        slot_kind=RoutineSlotKind.MANUAL,
        slot_key=slot,
        scheduled_for=claimed_at,
        claimed_at=claimed_at,
        claim_token=claim_token,
    )


def _claim_routine_slot(
    connection: SQLConnection,
    current: ScheduledRoutine,
    encoded: str,
    *,
    slot_kind: RoutineSlotKind,
    slot_key: str,
    scheduled_for: datetime,
    claimed_at: datetime,
    claim_token: str,
) -> RoutineOccurrence:
    if current.capability_grants:
        _require_effects_unblocked(
            connection, current.agent_id, routine_id=current.routine_id
        )
    if current.occurrence_allowance_exhausted:
        raise ValueError("routine_occurrence_budget_exhausted")
    if current.model_budget_exhausted:
        raise ValueError("routine_model_budget_exhausted")
    identity = routine_occurrence_id(current.routine_id, slot_key)
    occurrence = RoutineOccurrence(
        occurrence_id=identity,
        agent_id=current.agent_id,
        routine_id=current.routine_id,
        routine_revision=current.revision,
        slot_kind=slot_kind,
        slot_key=slot_key,
        scheduled_for=scheduled_for,
        claimed_at=claimed_at,
        claim_token=claim_token,
        lease_expires_at=claimed_at + timedelta(seconds=ROUTINE_CLAIM_LEASE_SECONDS),
        precheck_observation=None,
        execution_scope=None,
        execution_scope_digest=None,
        reserved_run_id=None,
        reserved_tokens=current.per_run_max_tokens,
        reserved_cost_usd=current.per_run_max_cost_usd or Decimal("0"),
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
        created_at=claimed_at,
        updated_at=claimed_at,
    )
    _insert_routine_occurrence(connection, occurrence)
    updated = replace(
        current,
        active_occurrence_id=identity,
        reserved_tokens=current.reserved_tokens + current.per_run_max_tokens,
        reserved_cost_usd=(
            current.reserved_cost_usd + (current.per_run_max_cost_usd or Decimal("0"))
        ),
        attempt_count=current.attempt_count + 1,
        occurrence_count=current.occurrence_count + 1,
        updated_at=claimed_at,
    )
    _replace_routine_row(connection, encoded, updated)
    return occurrence


def bind_occurrence_run(
    connection: SQLConnection,
    agent_id: str,
    occurrence_id: str,
    *,
    claim_token: str,
    run_id: str,
    execution_scope: ExecutionScope,
    bound_at: datetime,
    precheck_observation: ResourceRevisionObservation | None,
) -> RoutineOccurrence | None:
    loaded = _load_routine_occurrence_row(connection, agent_id, occurrence_id)
    if loaded is None:
        return None
    current, encoded = loaded
    if current.reserved_run_id is not None:
        return (
            current
            if (
                current.reserved_run_id == run_id
                and current.claim_token == claim_token
                and current.execution_scope == execution_scope
            )
            else None
        )
    if (
        current.disposition is not RoutineOccurrenceDisposition.CLAIMED
        or current.claim_token != claim_token
        or current.lease_expires_at is None
        or current.lease_expires_at <= bound_at
        or execution_scope.agent_id != current.agent_id
        or execution_scope.routine_id != current.routine_id
        or execution_scope.routine_revision != current.routine_revision
        or execution_scope.occurrence_id != current.occurrence_id
    ):
        return None
    routine = _load_routine_row(connection, agent_id, current.routine_id)
    if routine is None or routine[0].active_occurrence_id != occurrence_id:
        return None
    if (
        routine[0].revision != execution_scope.routine_revision
        or routine[0].state is not RoutineState.ACTIVE
        or bound_at >= routine[0].expires_at
        or execution_scope.scope_id != f"scope:{occurrence_id}"
        or execution_scope.revision != 1
        or execution_scope.principal_id != routine[0].owner_principal_id
        or execution_scope.grant_id
        != f"routine:{current.routine_id}:revision:{current.routine_revision}"
        # What the run may read and call was fixed when it started; only the
        # record's routes, ceiling and budgets must still match.
        or routine[0].sensitivity_ceiling != execution_scope.sensitivity_ceiling
        or routine[0].eligible_model_routes != execution_scope.eligible_model_routes
        or routine[0].per_run_max_tokens != execution_scope.per_run_max_tokens
        or routine[0].per_run_max_cost_usd != execution_scope.per_run_max_cost_usd
        or routine[0].distribution_plan.plan_digest
        != execution_scope.distribution_plan_digest
    ):
        return None
    precheck = routine[0].precheck
    if (precheck is None) != (precheck_observation is None):
        return None
    if precheck is not None:
        assert precheck_observation is not None
        if (
            precheck.source_id != precheck_observation.source_id
            or precheck.resource_id != precheck_observation.resource_id
        ):
            return None
    updated = replace(
        current,
        precheck_observation=precheck_observation,
        execution_scope=execution_scope,
        execution_scope_digest=execution_scope.digest,
        reserved_run_id=run_id,
        run_bound_at=bound_at,
        disposition=RoutineOccurrenceDisposition.RUNNING,
        updated_at=bound_at,
    )
    _replace_routine_occurrence_row(connection, encoded, updated)
    return updated


def mark_occurrence_run_terminal(
    connection: SQLConnection,
    agent_id: str,
    occurrence_id: str,
    *,
    run_id: str,
    terminal_at: datetime,
) -> RoutineOccurrence | None:
    loaded = _load_routine_occurrence_row(connection, agent_id, occurrence_id)
    if loaded is None:
        return None
    current, encoded = loaded
    if (
        current.disposition
        is RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
    ):
        return current
    if (
        current.disposition is not RoutineOccurrenceDisposition.RUNNING
        or current.reserved_run_id != run_id
    ):
        return None
    row = connection.execute(
        "SELECT result FROM runs WHERE id = ? AND agent_id = ?",
        (run_id, agent_id),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    result = decode_loop_exit(row[0])
    if result.run_id != run_id:
        raise ValueError("routine terminal run identity is invalid")
    updated = replace(
        current,
        run_terminal_at=terminal_at,
        terminal_run_id=run_id,
        lease_expires_at=None,
        disposition=(RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION),
        updated_at=terminal_at,
    )
    _replace_routine_occurrence_row(connection, encoded, updated)
    return updated


def finalize_occurrence(
    connection: SQLConnection,
    agent_id: str,
    occurrence_id: str,
    *,
    delivery_id: str,
    finalized_at: datetime,
    skipped_no_change_observation: ResourceRevisionObservation | None,
    failure_code: str | None,
    artifact_references: tuple[OutcomeArtifactReference, ...],
    outcome_contract_failure_code: str | None,
) -> tuple[RoutineOccurrence, Delivery | None] | None:
    loaded = _load_routine_occurrence_row(connection, agent_id, occurrence_id)
    if loaded is None:
        return None
    current, occurrence_data = loaded
    if skipped_no_change_observation is not None and failure_code is not None:
        raise ValueError("routine finalization outcomes are mutually exclusive")
    if outcome_contract_failure_code is not None and not re.fullmatch(
        r"[a-z][a-z0-9_]{0,127}", outcome_contract_failure_code
    ):
        raise ValueError("routine outcome contract failure code is invalid")
    contract_failure_code = outcome_contract_failure_code
    routine_loaded = _load_routine_row(
        connection,
        agent_id,
        current.routine_id,
    )
    if routine_loaded is None:
        return None
    routine, routine_data = routine_loaded
    target = routine.distribution_plan.targets[0]
    existing_row = connection.execute(
        "SELECT delivery_id, data FROM deliveries "
        "WHERE agent_id = ? AND subject_kind = ? AND subject_id = ? "
        "AND target_fingerprint = ?",
        (
            agent_id,
            DeliverySubjectKind.ROUTINE_OCCURRENCE.value,
            occurrence_id,
            target.target_fingerprint,
        ),
    ).fetchone()
    if existing_row is not None:
        existing_delivery = decode_delivery(
            existing_row[1],
            agent_id=agent_id,
            delivery_id=existing_row[0],
        )
        return current, existing_delivery
    if routine.active_occurrence_id != occurrence_id:
        return None
    pre_run_outcome = (
        skipped_no_change_observation is not None or failure_code is not None
    )
    if pre_run_outcome:
        bound_without_transcript = (
            failure_code == "routine_run_not_started"
            and current.disposition is RoutineOccurrenceDisposition.RUNNING
            and current.reserved_run_id is not None
            and connection.execute(
                "SELECT 1 FROM runs WHERE id = ? AND agent_id = ?",
                (current.reserved_run_id, agent_id),
            ).fetchone()
            is None
        )
        expected_disposition = (
            RoutineOccurrenceDisposition.SKIPPED_NO_CHANGE
            if skipped_no_change_observation is not None
            else RoutineOccurrenceDisposition.TERMINAL_FAILED
        )
        if current.disposition is expected_disposition:
            return current, None
        if not bound_without_transcript and (
            current.disposition
            not in {
                RoutineOccurrenceDisposition.CLAIMED,
                RoutineOccurrenceDisposition.PRECHECKING,
            }
            or current.reserved_run_id is not None
        ):
            return None
    elif (
        current.disposition
        is not RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
        or current.reserved_run_id is None
    ):
        return None

    occurrence_observation = current.precheck_observation
    resulting_run_id: str | None = None
    charged_tokens = 0
    charged_cost = Decimal("0")
    sensitivity = routine.sensitivity_ceiling
    report_digest: str | None = None
    report_preview: str | None = None
    report_truncated = False
    validated_artifact_references: tuple[OutcomeArtifactReference, ...] = ()
    effect_receipt_ids: tuple[str, ...] = ()
    uncertain_effect = False
    rate_limited = False
    if pre_run_outcome:
        occurrence_observation = skipped_no_change_observation
        successful = occurrence_observation is not None
        terminal_failure_code = failure_code
        disposition = (
            RoutineOccurrenceDisposition.SKIPPED_NO_CHANGE
            if successful
            else RoutineOccurrenceDisposition.TERMINAL_FAILED
        )
        outcome = "skipped_no_change" if successful else "failed"
        reason = "resource_revision_unchanged" if successful else terminal_failure_code
        if occurrence_observation is not None:
            precheck = routine.precheck
            if (
                precheck is None
                or occurrence_observation.source_id != precheck.source_id
                or occurrence_observation.resource_id != precheck.resource_id
            ):
                raise ValueError("routine precheck observation is out of scope")
        if successful and any(
            requirement.minimum_count > 0
            for requirement in routine.outcome_contract.artifact_requirements
        ):
            successful = False
            terminal_failure_code = "outcome_artifact_contract_failed"
            disposition = RoutineOccurrenceDisposition.TERMINAL_FAILED
            outcome = "failed"
            reason = terminal_failure_code
    else:
        assert current.reserved_run_id is not None
        run_row = connection.execute(
            "SELECT input, result FROM runs WHERE id = ? AND agent_id = ?",
            (current.reserved_run_id, agent_id),
        ).fetchone()
        if run_row is None or run_row[1] is None:
            return None
        run_input = decode_run_input(run_row[0])
        result = decode_loop_exit(run_row[1])
        if (
            run_input.origin is not RunOrigin.SCHEDULED_ROUTINE
            or run_input.execution_scope is None
            or run_input.execution_scope.routine_id != current.routine_id
            or run_input.execution_scope.routine_revision != current.routine_revision
            or run_input.execution_scope.occurrence_id != current.occurrence_id
            or run_input.execution_scope.distribution_plan_digest
            != routine.distribution_plan.plan_digest
            or result.run_id != current.reserved_run_id
        ):
            raise ValueError("routine terminal run scope is invalid")
        message_rows = connection.execute(
            "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
            (current.reserved_run_id,),
        ).fetchall()
        transcript = Transcript(
            run=run_input,
            messages=tuple(
                decode_message(message_data) for (message_data,) in message_rows
            ),
        )
        successful = (
            result.kind is LoopExitKind.COMPLETED
            and isinstance(result.final_text, str)
            and bool(result.final_text.strip())
        )
        receipt_rows = connection.execute(
            "SELECT data FROM effect_receipts WHERE agent_id = ? AND run_id = ? ORDER BY id LIMIT 257",
            (agent_id, current.reserved_run_id),
        ).fetchall()
        if len(receipt_rows) > 256:
            raise ValueError("routine effect receipt references exceed their bound")
        receipts = tuple(decode_receipt(row[0]) for row in receipt_rows)
        effect_receipt_ids = tuple(receipt.receipt_id for receipt in receipts)
        grants = {grant.capability_id: grant for grant in routine.capability_grants}
        scope_contracts = (
            run_input.execution_scope.contract_bindings.capability_contracts
        )
        successful_calls: dict[str, int] = {}
        result_blocks: dict[str, list[ToolResultBlock]] = {}
        for exchange in transcript.messages:
            if exchange.role is MessageRole.TOOL:
                for tool_result in exchange.content:
                    if isinstance(tool_result, ToolResultBlock):
                        result_blocks.setdefault(tool_result.call_id, []).append(
                            tool_result
                        )
        for receipt in receipts:
            if receipt.sensitivity.routing_rank > sensitivity.routing_rank:
                sensitivity = receipt.sensitivity
            if receipt.outcome in {
                EffectOutcome.STARTED,
                EffectOutcome.UNCERTAIN,
            }:
                uncertain_effect = True
            grant = grants.get(receipt.capability_id)
            # Approved at the time: bind to the contract fixed at run start. A
            # receipt that used a grant must still match that exact grant.
            if (
                receipt.routine_id != current.routine_id
                or receipt.routine_revision != current.routine_revision
                or receipt.occurrence_id != current.occurrence_id
                or receipt.capability_contract_digest
                != scope_contracts.get(receipt.capability_id)
                or (
                    receipt.capability_grant_digest is not None
                    and (
                        grant is None
                        or receipt.capability_grant_digest != grant.grant_digest
                        or receipt.domain_owner_id != grant.domain_owner_id
                    )
                )
            ):
                contract_failure_code = "outcome_effect_binding_invalid"
                continue
            if receipt.outcome in {
                EffectOutcome.STARTED,
                EffectOutcome.UNCERTAIN,
            }:
                uncertain_effect = True
                contract_failure_code = "outcome_effect_uncertain"
                continue
            if receipt.outcome is not EffectOutcome.SUCCEEDED:
                continue
            results = result_blocks.get(receipt.call_id, [])
            expected_reference = {
                "receipt_id": receipt.receipt_id,
                "receipt_digest": receipt.receipt_digest,
                "outcome": receipt.outcome.value,
                "evidence_basis": receipt.evidence_basis.value,
            }
            if (
                len(results) != 1
                or results[0].is_error
                or results[0].capability_id != receipt.capability_id
                or results[0].output.get("effect_receipt")
                != FrozenJsonObject.from_mapping(expected_reference)
                or results[0].output_sha256
                != "sha256:"
                + sha256(canonical_json(results[0].output).encode("utf-8")).hexdigest()
            ):
                contract_failure_code = "outcome_effect_result_invalid"
                continue
            requirement = next(
                (
                    item
                    for item in routine.outcome_contract.effect_requirements
                    if item.capability_id == receipt.capability_id
                ),
                None,
            )
            # A write the routine never declared still counts once a person approved it.
            if (
                requirement is not None
                and receipt.evidence_basis not in requirement.accepted_evidence_bases
            ):
                contract_failure_code = "outcome_effect_evidence_unsupported"
                continue
            successful_calls[receipt.capability_id] = (
                successful_calls.get(receipt.capability_id, 0) + 1
            )
        if any(
            successful_calls.get(item.capability_id, 0) < item.minimum_successful_calls
            for item in routine.outcome_contract.effect_requirements
        ):
            contract_failure_code = (
                contract_failure_code or "outcome_effect_requirement_unsatisfied"
            )
        if uncertain_effect:
            contract_failure_code = "outcome_effect_uncertain"
        if contract_failure_code is not None:
            successful = False
        for message in transcript.messages:
            for block in message.content:
                if (
                    isinstance(block, ToolResultBlock)
                    and block.sensitivity is not None
                    and block.sensitivity.routing_rank > sensitivity.routing_rank
                ):
                    sensitivity = block.sensitivity
        if (
            sensitivity.routing_rank
            > routine.outcome_contract.maximum_effective_sensitivity.routing_rank
        ):
            successful = False
            contract_failure_code = "outcome_sensitivity_contract_failed"
        estimate = result.usage.cost_estimate
        charged_tokens = result.usage.total_tokens
        charged_cost = estimate.amount_usd or Decimal("0")
        dollar_capped = routine.per_run_max_cost_usd is not None
        # A provider quota rejection bills nothing for the rejected request.
        rate_limited = (
            result.kind is LoopExitKind.FAILED
            and result.reason == ProviderErrorCode.RATE_LIMIT_ERROR.value
        )
        # Without a dollar ceiling, an unpriced route (a subscription or a
        # model with no price schedule) is complete when it reported tokens.
        unpriced_with_known_tokens = (
            not dollar_capped
            and estimate.code in _UNPRICED_COST_CODES
            and (result.usage.total_tokens > 0 or rate_limited)
        )
        if (
            estimate.status is not CostEstimateStatus.COMPLETE
            and estimate.code != "no_model_attempts"
            and not unpriced_with_known_tokens
        ):
            # Unknown consumption retains the reservation as a conservative
            # charge, without discarding a larger known partial amount.
            charged_cost = max(charged_cost, current.reserved_cost_usd)
            charged_tokens = max(charged_tokens, current.reserved_tokens)
            # A token-only run whose unknown is a replayed stall's
            # unreported attempt is bounded by that charge and keeps its
            # outcome. Other unknown consumption fails the run.
            unreported_stall = (
                not dollar_capped
                and estimate.code == STALL_USAGE_UNREPORTED
                and result.usage.total_tokens > 0
            )
            if not unreported_stall:
                if successful:
                    contract_failure_code = (
                        contract_failure_code or "routine_run_usage_incomplete"
                    )
                successful = False
        if result.usage.total_tokens > current.reserved_tokens or (
            dollar_capped and charged_cost > current.reserved_cost_usd
        ):
            contract_failure_code = (
                contract_failure_code or "routine_run_budget_exceeded"
            )
            successful = False
        # Authenticate committed partial artifacts even when a different
        # required action failed. Minimum counts decide completion only.
        try:
            expected_artifact_references = tuple(
                sorted(
                    (outcome_artifact_reference(ref) for ref in result.artifacts),
                    key=lambda item: item.artifact_id,
                )
            )
            validated_artifact_references = validate_outcome_artifact_references(
                artifact_references,
                contract=routine.outcome_contract,
                resulting_run_id=current.reserved_run_id,
                require_minimum_counts=False,
            )
            if validated_artifact_references != expected_artifact_references:
                raise ValueError("validated artifact references differ")
        except (TypeError, ValueError):
            contract_failure_code = (
                contract_failure_code or "outcome_artifact_contract_failed"
            )
            validated_artifact_references = ()
            successful = False
        else:
            try:
                validate_outcome_artifact_references(
                    validated_artifact_references,
                    contract=routine.outcome_contract,
                    resulting_run_id=current.reserved_run_id,
                )
            except (TypeError, ValueError):
                contract_failure_code = (
                    contract_failure_code or "outcome_artifact_contract_failed"
                )
                successful = False
        if result.final_text is not None:
            report_digest, report_preview, report_truncated = (
                conclusion_preview_projection(result.final_text)
            )
        if successful and routine.outcome_contract.effect_requirements and not receipts:
            report_digest, report_preview, report_truncated = (
                conclusion_preview_projection(
                    "No external action was taken in this occurrence. "
                    "The approved optional action path was used.\n\n"
                    + (result.final_text or "")
                )
            )
        resulting_run_id = current.reserved_run_id
        terminal_failure_code = (
            None
            if successful
            else (contract_failure_code or f"routine_run_{result.reason}")
        )
        disposition = (
            RoutineOccurrenceDisposition.COMPLETED
            if successful
            else RoutineOccurrenceDisposition.TERMINAL_FAILED
        )
        outcome = "completed" if successful else "failed"
        reason = "completed" if successful else terminal_failure_code

    accounted_routine = replace(
        routine,
        reserved_tokens=routine.reserved_tokens - current.reserved_tokens,
        reserved_cost_usd=routine.reserved_cost_usd - current.reserved_cost_usd,
        charged_tokens=routine.charged_tokens + charged_tokens,
        charged_cost_usd=routine.charged_cost_usd + charged_cost,
    )
    if current.slot_kind is RoutineSlotKind.MANUAL:
        following = routine.next_due_at
    else:
        following = next_slot(
            routine.schedule,
            after=current.scheduled_for,
            expires_at=routine.expires_at,
        )
        while following is not None and following <= finalized_at:
            following = next_slot(
                routine.schedule,
                after=following,
                expires_at=routine.expires_at,
            )
    # A quota rejection is the provider's pause, not the routine failing;
    # the supervisor holds new claims until the quota window passes.
    failures = (
        0 if successful else routine.consecutive_failures + (0 if rate_limited else 1)
    )
    if successful:
        next_state = (
            RoutineState.ACTIVE
            if following is not None
            else (
                RoutineState.EXPIRED
                if finalized_at >= routine.expires_at
                else RoutineState.COMPLETED
            )
        )
    elif failures >= routine.maximum_consecutive_failures or following is None:
        next_state = RoutineState.NEEDS_ATTENTION
        following = None
    else:
        next_state = RoutineState.ACTIVE
    if next_state is RoutineState.ACTIVE and (
        accounted_routine.model_budget_exhausted
        or accounted_routine.occurrence_allowance_exhausted
    ):
        next_state = RoutineState.NEEDS_ATTENTION
        following = None
    if uncertain_effect:
        next_state = RoutineState.PAUSED
        following = None
    if routine.state in {
        RoutineState.PAUSED,
        RoutineState.DISABLED,
        RoutineState.EXPIRED,
    }:
        # A foreground stop/recovery decision survives delayed producer
        # finalization; that finalizer cannot reactivate the assignment.
        next_state = routine.state
        following = None
    escalation = not successful and next_state is RoutineState.NEEDS_ATTENTION

    payload = {
        "subject": {
            "kind": DeliverySubjectKind.ROUTINE_OCCURRENCE.value,
            "subject_id": occurrence_id,
        },
        "routine_id": current.routine_id,
        "routine_revision": current.routine_revision,
        "occurrence_id": current.occurrence_id,
        "scheduled_for": current.scheduled_for.isoformat(),
        "run_id": resulting_run_id,
        "effect_receipt_ids": effect_receipt_ids,
        "outcome": outcome,
        "reason": reason,
        "escalation": escalation,
        "routine_state": next_state.value,
        "observation_digest": (
            None if occurrence_observation is None else occurrence_observation.digest
        ),
        "report_digest": report_digest,
        "report_preview": report_preview,
        "report_truncated": report_truncated,
    }
    conclusion_digest = (
        "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    )
    conclusion_kind = (
        OutcomeConclusionKind.TERMINAL_RUN
        if resulting_run_id is not None
        else OutcomeConclusionKind.NO_MODEL_OCCURRENCE
    )
    delivery = construct_logical_delivery(
        delivery_id=delivery_id,
        agent_id=agent_id,
        conversation_id=routine.conversation_id,
        subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
        subject_id=occurrence_id,
        target=target,
        conclusion_kind=conclusion_kind,
        conclusion_state=(
            OutcomeState.SKIPPED_NO_CHANGE
            if disposition is RoutineOccurrenceDisposition.SKIPPED_NO_CHANGE
            else (OutcomeState.SUCCEEDED if successful else OutcomeState.FAILED)
        ),
        conclusion_id=resulting_run_id or occurrence_id,
        conclusion_digest=(
            report_digest if successful and report_digest else conclusion_digest
        ),
        conclusion_preview=report_preview or "",
        conclusion_preview_truncated=report_truncated,
        resulting_run_id=resulting_run_id,
        artifact_references=(
            validated_artifact_references if resulting_run_id is not None else ()
        ),
        effective_sensitivity=sensitivity,
        provenance_digest=(
            current.execution_scope_digest
            or (
                None
                if occurrence_observation is None
                else occurrence_observation.digest
            )
            or conclusion_digest
        ),
        failure_code=terminal_failure_code,
        observed_at=finalized_at,
        effect_receipt_ids=effect_receipt_ids,
    )
    _insert_delivery(connection, delivery)

    completed_occurrence = replace(
        current,
        precheck_observation=occurrence_observation,
        reserved_tokens=0,
        reserved_cost_usd=Decimal("0"),
        charged_tokens=charged_tokens,
        charged_cost_usd=charged_cost,
        conclusion_digest=conclusion_digest,
        delivery_ids=(delivery.delivery_id,),
        effect_receipt_ids=effect_receipt_ids,
        failure_code=terminal_failure_code,
        lease_expires_at=None,
        disposition=disposition,
        updated_at=finalized_at,
    )
    acknowledged_observation = routine.last_acknowledged_precheck_observation
    if successful and occurrence_observation is not None:
        acknowledged_observation = occurrence_observation
    completed_routine = replace(
        accounted_routine,
        consecutive_failures=failures,
        last_acknowledged_precheck_observation=acknowledged_observation,
        active_occurrence_id=None,
        last_occurrence_id=current.occurrence_id,
        last_delivery_ids=(delivery.delivery_id,),
        next_due_at=following,
        state=next_state,
        updated_at=finalized_at,
    )
    _replace_routine_occurrence_row(
        connection,
        occurrence_data,
        completed_occurrence,
    )
    _replace_routine_row(connection, routine_data, completed_routine)
    return completed_occurrence, delivery


def recover_stale_occurrences(
    connection: SQLConnection,
    agent_id: str,
    *,
    recovered_at: datetime,
    recovered_us: int | None,
    claim_token_factory: Callable[[str], str],
) -> tuple[RoutineOccurrence, ...]:
    rows = tuple(
        connection.execute(
            """SELECT occurrence_id, data FROM routine_occurrences
               WHERE agent_id = ?
                 AND (state = ?
                      OR (state IN (?, ?, ?)
                          AND (lease_expires_at_us IS NULL
                               OR lease_expires_at_us <= ?)))
               ORDER BY COALESCE(lease_expires_at_us, 0), occurrence_id""",
            (
                agent_id,
                (RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION.value),
                RoutineOccurrenceDisposition.CLAIMED.value,
                RoutineOccurrenceDisposition.PRECHECKING.value,
                RoutineOccurrenceDisposition.RUNNING.value,
                recovered_us,
            ),
        )
    )
    recovered: list[RoutineOccurrence] = []
    for occurrence_id_value, occurrence_data in rows:
        current = decode_routine_occurrence(
            occurrence_data,
            agent_id=agent_id,
            occurrence_id=occurrence_id_value,
        )
        if (
            current.disposition
            is RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
        ):
            recovered.append(current)
            continue
        if current.disposition not in {
            RoutineOccurrenceDisposition.CLAIMED,
            RoutineOccurrenceDisposition.PRECHECKING,
            RoutineOccurrenceDisposition.RUNNING,
        }:
            continue
        if current.reserved_run_id is not None:
            run_row = connection.execute(
                "SELECT result FROM runs WHERE id = ? AND agent_id = ?",
                (current.reserved_run_id, agent_id),
            ).fetchone()
            if run_row is not None and run_row[0] is not None:
                updated = replace(
                    current,
                    run_terminal_at=recovered_at,
                    terminal_run_id=current.reserved_run_id,
                    lease_expires_at=None,
                    disposition=(
                        RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                    ),
                    updated_at=recovered_at,
                )
                _replace_routine_occurrence_row(
                    connection,
                    occurrence_data,
                    updated,
                )
                recovered.append(updated)
                continue
            # A reserved ID is never rebound, including a crash before
            # the loop created its transcript. The ordinary finalizer
            # rechecks that absence and records a no-model failure.
            if run_row is None:
                recovered.append(current)
            continue
        routine_loaded = _load_routine_row(
            connection,
            agent_id,
            current.routine_id,
        )
        if routine_loaded is None:
            continue
        routine, routine_data = routine_loaded
        if routine.active_occurrence_id != current.occurrence_id:
            continue
        if (
            current.attempt_count >= MAX_ROUTINE_ATTEMPTS
            or routine.attempt_count >= routine.cumulative_max_attempts
        ):
            # Return the expired claim for atomic failure/delivery,
            # without consuming another attempt or renewing its lease.
            recovered.append(current)
            continue
        token = claim_token_factory(current.occurrence_id)
        updated = replace(
            current,
            claimed_at=recovered_at,
            claim_token=token,
            lease_expires_at=recovered_at
            + timedelta(seconds=ROUTINE_CLAIM_LEASE_SECONDS),
            attempt_count=current.attempt_count + 1,
            updated_at=recovered_at,
        )
        updated_routine = replace(
            routine,
            attempt_count=routine.attempt_count + 1,
            updated_at=recovered_at,
        )
        _replace_routine_occurrence_row(
            connection,
            occurrence_data,
            updated,
        )
        _replace_routine_row(connection, routine_data, updated_routine)
        recovered.append(updated)
    return tuple(recovered)
