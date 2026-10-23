"""Routine rows as released revisions 1-3 stored them, for upgrade tests."""

from __future__ import annotations

import json
import sqlite3

from daita.routines.models import RoutineOccurrence, ScheduledRoutine
from daita.storage.sqlite_codecs.routines import (
    decode_routine_occurrence,
    decode_scheduled_routine,
    encode_routine_occurrence,
    encode_scheduled_routine,
)

_EXPIRES_AT = {"__datetime__": "2027-01-01T00:00:00Z"}


def released_routine(
    routine: ScheduledRoutine,
    *,
    capability_grants: tuple[dict[str, object], ...] = (),
    precheck: dict[str, object] | None = None,
) -> str:
    """The frozen-contract shape: access, grants, budgets, expiry and outcome."""

    payload = json.loads(encode_scheduled_routine(routine))
    fields = payload["fields"]
    fields["allowed_source_ids"] = fields.pop("source_ids")
    fields.update(
        run_immediately=False,
        misfire_policy="latest_only",
        reporting_mode="always" if precheck is None else "changes_only",
        precheck=precheck,
        last_acknowledged_precheck_observation=None,
        allowed_connector_binding_ids=[],
        allowed_resource_ids=[],
        allowed_capability_ids=["artifact.create_document"],
        contract_bindings={
            "__record__": "ExecutionContractBindings",
            "fields": {"capability_contracts": {}, "model_routes": {}},
        },
        capability_grants=list(capability_grants),
        allowed_access_modes=["none"],
        allowed_operational_effects=["none"],
        sensitivity_ceiling="internal",
        eligible_model_routes=["mock:released"],
        skill_bindings=[],
        outcome_contract={"__record__": "OutcomeContract", "fields": {}},
        distribution_plan={"__record__": "DistributionPlan", "fields": {}},
        per_run_max_tokens=2_000,
        per_run_max_cost_usd=None,
        cumulative_max_tokens=20_000,
        cumulative_max_cost_usd=None,
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        reserved_tokens=0,
        reserved_cost_usd={"__decimal__": "0"},
        maximum_consecutive_failures=3,
        expires_at=_EXPIRES_AT,
        promotion_evidence=None,
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def released_occurrence(occurrence: RoutineOccurrence, *, disposition: str) -> str:
    """The shape with precheck observations and per-occurrence reservations."""

    payload = json.loads(encode_routine_occurrence(occurrence))
    payload["fields"].update(
        precheck_observation=None,
        reserved_tokens=0,
        reserved_cost_usd={"__decimal__": "0"},
        disposition=disposition,
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def restore_released_routines(connection: sqlite3.Connection) -> None:
    """Rewrite every current routine and occurrence row to the released shape."""

    for agent_id, routine_id, data in connection.execute(
        "SELECT agent_id, routine_id, data FROM scheduled_routines"
    ).fetchall():
        routine = decode_scheduled_routine(
            data, agent_id=agent_id, routine_id=routine_id
        )
        connection.execute(
            "UPDATE scheduled_routines SET data = ? WHERE agent_id = ? AND routine_id = ?",
            (released_routine(routine), agent_id, routine_id),
        )
    for agent_id, occurrence_id, data in connection.execute(
        "SELECT agent_id, occurrence_id, data FROM routine_occurrences"
    ).fetchall():
        occurrence = decode_routine_occurrence(
            data, agent_id=agent_id, occurrence_id=occurrence_id
        )
        connection.execute(
            "UPDATE routine_occurrences SET data = ? WHERE agent_id = ? AND occurrence_id = ?",
            (
                released_occurrence(
                    occurrence, disposition=occurrence.disposition.value
                ),
                agent_id,
                occurrence_id,
            ),
        )
