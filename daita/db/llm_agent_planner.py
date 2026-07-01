"""LLM-backed outer planner for the DB agent loop."""

from __future__ import annotations

import json
import re
from typing import Any

from .llm_service import DbLLMService
from .planner_protocol import (
    DbAgentPlanner,
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)

_PLANNER_DECISION_KEYS = frozenset(DbPlannerDecision.__dataclass_fields__)
_PLANNER_ACTION_KEYS = frozenset(DbPlannerAction.__dataclass_fields__)
_PLANNER_DECISION_ENVELOPES = (
    "decision",
    "planner_decision",
    "DbPlannerDecision",
)


class DbLLMAgentPlanner(DbAgentPlanner):
    """Production DB agent planner backed by ``DbLLMService``."""

    def __init__(self, service: DbLLMService) -> None:
        self.service = service

    async def plan(self, state: DbLoopState) -> DbPlannerDecision:
        """Ask the configured LLM for one typed DB planner decision."""
        if not self.service.available:
            return DbPlannerDecision(
                status=DbPlannerDecisionStatus.FAILED,
                stop_conditions=("configuration_required",),
                rationale="DB LLM service is required for semantic DB planning.",
                metadata={
                    "failure": "db_llm_service_unavailable",
                    "configuration_required": True,
                },
            )
        try:
            response = await self.service.generate_json(_planner_messages(state))
        except Exception as exc:
            return DbPlannerDecision(
                status=DbPlannerDecisionStatus.FAILED,
                stop_conditions=("planner_llm_failed",),
                rationale="DB LLM planner request failed.",
                metadata={"failure": "planner_llm_failed", "error": str(exc)},
            )
        parsed, diagnostics = _parse_planner_json(response.content)
        if parsed is None:
            return DbPlannerDecision(
                status=DbPlannerDecisionStatus.FAILED,
                stop_conditions=("planner_json_invalid",),
                rationale="DB LLM planner returned invalid structured JSON.",
                metadata={
                    "failure": "planner_json_invalid",
                    "diagnostics": diagnostics,
                    "llm": response.diagnostics,
                },
            )
        try:
            decision = DbPlannerDecision.from_dict(parsed)
        except Exception as exc:
            return DbPlannerDecision(
                status=DbPlannerDecisionStatus.FAILED,
                stop_conditions=("planner_decision_invalid",),
                rationale="DB LLM planner returned an invalid decision shape.",
                metadata={
                    "failure": "planner_decision_invalid",
                    "error": str(exc),
                    "planner_json_normalization": diagnostics,
                    "llm": response.diagnostics,
                },
            )
        metadata = {
            **decision.metadata,
            "llm": response.diagnostics,
            "planner_json_normalization": diagnostics,
        }
        return DbPlannerDecision(
            status=decision.status,
            intent=decision.intent,
            actions=decision.actions,
            stop_conditions=decision.stop_conditions,
            clarification_question=decision.clarification_question,
            rationale=decision.rationale,
            metadata=metadata,
        )


def _planner_messages(state: DbLoopState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the Daita from_db outer planner. Return only strict JSON "
                "for a DbPlannerDecision. You choose semantic intent and typed "
                "planner actions; you never execute database work. Use only the "
                "provided action vocabulary. SQL execution actions must depend on "
                "runtime validation and must not bypass governance. For user "
                "requests that ask for rows, counts, aggregates, metrics, or "
                "other database values, set intent.operation_type to "
                "'data.query'. If schema evidence is missing, plan schema "
                "inspection or schema search first; do not clarify only because "
                "schema has not been inspected. Continue data queries through "
                "query planning and validated read execution until query.result "
                "evidence exists, unless the request is genuinely ambiguous after "
                "available runtime facts have been gathered. Use depends_on only "
                "for actions in the same decision. When prior accepted "
                "query.plan.proposal evidence already contains SQL, an "
                "execute_validated_read action should provide that SQL in input.sql "
                "or otherwise rely on that accepted evidence; do not depend on a "
                "previous-turn action id."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "decision_schema": _decision_schema_hint(),
                    "available_action_kinds": [
                        kind.value for kind in DbPlannerActionKind
                    ],
                    "state": state.to_dict(),
                },
                sort_keys=True,
                default=str,
            ),
        },
    ]


def _decision_schema_hint() -> dict[str, Any]:
    return {
        "status": "continue | finish | clarify | blocked | failed",
        "intent": {"operation_type": "semantic operation label, if known"},
        "actions": [
            {
                "action_id": "stable id unique within this decision",
                "kind": "one available action kind",
                "input": {},
                "depends_on": [],
                "rationale": "optional",
                "metadata": {},
            }
        ],
        "stop_conditions": [],
        "clarification_question": None,
        "rationale": "optional",
        "metadata": {},
        "monitor_action_inputs": {
            "plan_monitor_create": {
                "intent": {
                    "target": {
                        "target_type": "table",
                        "name": "table_or_asset_name",
                        "source_scope": [],
                    },
                    "condition": {
                        "kind": "new_rows | rows_present | threshold | freshness | report",
                        "operator": "optional comparison operator",
                        "value": "optional threshold",
                        "path": "optional value path",
                    },
                    "schedule": {
                        "kind": "interval",
                        "interval_seconds": 300,
                    },
                    "delivery": {
                        "delivery_kind": "local | in_app | email | slack",
                        "target": {},
                    },
                    "display": {
                        "explicit_name": "optional name",
                        "description": "optional description",
                    },
                    "policy": {},
                    "budget": {},
                },
                "owner": {},
            },
            "commit_monitor_create": {
                "depends_on": ["plan_monitor_create_action_id"],
                "input": {},
            },
            "plan_monitor_lifecycle": {
                "action": "update | pause | resume | delete | disable",
                "monitor_id": "structured monitor id or reference",
                "patch": {},
                "paused_until": "optional ISO timestamp",
            },
            "commit_monitor_lifecycle": {
                "action": "update | pause | resume | delete | disable",
                "depends_on": ["plan_monitor_lifecycle_action_id"],
            },
            "read_monitor_state": {
                "read_kind": "list | inspect | explain_run | approvals",
                "monitor_id": "optional structured monitor id or reference",
                "status": "optional status filter for list",
                "pending_only": True,
            },
            "resolve_monitor_approval": {
                "approval_action": "approve | reject | cancel",
                "approval_id": "optional approval id",
                "monitor_id": "optional structured monitor id or reference",
            },
        },
    }


def _parse_planner_json(content: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"normalization_steps": []}
    raw = _strip_json_fence(content)
    if raw != content.strip():
        _add_normalization_step(diagnostics, "json_fence_stripped")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        diagnostics.update({"error": "planner_json_decode_error", "message": str(exc)})
        return None, diagnostics
    if not isinstance(parsed, dict):
        diagnostics["error"] = "planner_json_not_object"
        return None, diagnostics
    return _normalize_planner_decision_payload(parsed, diagnostics)


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _normalize_planner_decision_payload(
    parsed: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    diagnostics = diagnostics or {"normalization_steps": []}
    payload = _unwrap_planner_decision(parsed, diagnostics)
    normalized = {
        key: value for key, value in payload.items() if key in _PLANNER_DECISION_KEYS
    }
    dropped_decision_keys = sorted(set(payload) - _PLANNER_DECISION_KEYS)
    if dropped_decision_keys:
        diagnostics["dropped_decision_keys"] = dropped_decision_keys
        _add_normalization_step(
            diagnostics,
            "dropped_decision_keys",
            keys=dropped_decision_keys,
        )

    if "stop_conditions" in normalized:
        normalized["stop_conditions"] = _coerce_boundary_string_list(
            normalized["stop_conditions"],
            diagnostics=diagnostics,
            path="stop_conditions",
        )

    actions = normalized.get("actions")
    if isinstance(actions, list):
        normalized["actions"] = [
            _normalize_planner_action_payload(action, index, diagnostics)
            for index, action in enumerate(actions)
        ]
    return normalized, diagnostics


def _unwrap_planner_decision(
    parsed: dict[str, Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    if "status" in parsed:
        return parsed
    for key in _PLANNER_DECISION_ENVELOPES:
        value = parsed.get(key)
        if isinstance(value, dict):
            diagnostics["unwrapped_envelope"] = key
            _add_normalization_step(diagnostics, "unwrapped_envelope", key=key)
            dropped_envelope_keys = sorted(set(parsed) - {key})
            if dropped_envelope_keys:
                diagnostics["dropped_envelope_keys"] = dropped_envelope_keys
                _add_normalization_step(
                    diagnostics,
                    "dropped_envelope_keys",
                    keys=dropped_envelope_keys,
                )
            return value
    return parsed


def _normalize_planner_action_payload(
    action: Any, index: int, diagnostics: dict[str, Any]
) -> Any:
    if not isinstance(action, dict):
        return action
    normalized = {
        key: value for key, value in action.items() if key in _PLANNER_ACTION_KEYS
    }
    dropped_keys = sorted(set(action) - _PLANNER_ACTION_KEYS)
    if dropped_keys:
        diagnostics.setdefault("dropped_action_keys", []).append(
            {
                "index": index,
                "action_id": str(action.get("action_id") or ""),
                "keys": dropped_keys,
            }
        )
        _add_normalization_step(
            diagnostics,
            "dropped_action_keys",
            index=index,
            keys=dropped_keys,
        )
    if "depends_on" in normalized:
        normalized["depends_on"] = _coerce_boundary_string_list(
            normalized["depends_on"],
            diagnostics=diagnostics,
            path=f"actions[{index}].depends_on",
        )
    return normalized


def _coerce_boundary_string_list(
    value: Any,
    *,
    diagnostics: dict[str, Any],
    path: str,
) -> list[str]:
    if value is None:
        coerced: list[str] = []
    elif isinstance(value, str):
        coerced = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        coerced = [_coerce_boundary_string_item(item) for item in value]
    else:
        coerced = [_coerce_boundary_string_item(value)]
    if coerced != value:
        diagnostics.setdefault("coerced_fields", []).append(
            {"path": path, "before": value, "after": coerced}
        )
        _add_normalization_step(
            diagnostics,
            "coerced_field",
            path=path,
        )
    return coerced


def _coerce_boundary_string_item(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("action_id", "id", "kind", "name"):
            item = value.get(key)
            if isinstance(item, str) and item:
                return item
    return str(value)


def _add_normalization_step(
    diagnostics: dict[str, Any], step: str, **details: Any
) -> None:
    diagnostics.setdefault("normalization_steps", []).append({"step": step, **details})
