"""LLM-backed outer planner for the DB agent loop."""

from __future__ import annotations

import json
from typing import Any

from .json_normalization import strip_json_fence
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
_TERMINAL_ACTION_KINDS = frozenset(
    {DbPlannerActionKind.FINISH, DbPlannerActionKind.CLARIFY}
)
_LLM_ACTION_KINDS = tuple(
    kind for kind in DbPlannerActionKind if kind not in _TERMINAL_ACTION_KINDS
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
        decision, terminal_error = _normalize_terminal_decision(decision)
        if terminal_error is not None:
            return DbPlannerDecision(
                status=DbPlannerDecisionStatus.FAILED,
                intent=decision.intent,
                stop_conditions=(terminal_error,),
                rationale="DB LLM planner returned an invalid terminal decision.",
                metadata={
                    "failure": "planner_decision_invalid",
                    "error": terminal_error,
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
    state_payload = state.to_dict()
    state_payload["available_action_kinds"] = [
        kind
        for kind in state_payload.get("available_action_kinds", ())
        if kind not in {item.value for item in _TERMINAL_ACTION_KINDS}
    ]
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
                "for actions in the same decision. propose_sql_read creates a new "
                "query plan and must not include input.query_plan_ref or "
                "input.plan_evidence_id. query_plan_ref is valid only for "
                "execute_validated_read. To use SQL from accepted "
                "query.plan.proposal evidence during execution, set "
                "input.plan_evidence_id to that evidence id or set "
                'input.query_plan_ref="latest_accepted_query_plan". For '
                "repair_query_plan, the runtime binds durable prior-plan, "
                "failure, and planning-context evidence. If the user explicitly "
                "asks for "
                "catalog column values, gather them with search_column_values "
                "before SQL; set input.tables and input.columns to the targets. "
                "If validation reports unobserved_filter_literal, repair to "
                "observed values when intent is clear; otherwise block or "
                "clarify. For explicit SQL writes, put the concrete SQL in "
                "input.sql: use propose_sql_write for write.propose, "
                "execute_validated_write for write.execute, and propose_sql_write "
                "for destructive SQL so policy can deny it before execution."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "decision_schema": _decision_schema_hint(),
                    "available_action_kinds": [
                        kind.value for kind in _LLM_ACTION_KINDS
                    ],
                    "state": state_payload,
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
                "input": {
                    "sql": (
                        "optional direct SQL for execute_validated_read, "
                        "propose_sql_write, or execute_validated_write"
                    ),
                    "plan_evidence_id": (
                        "only for execute_validated_read or repair_query_plan; "
                        "do not include on propose_sql_read"
                    ),
                    "query_plan_ref": (
                        "only latest_accepted_query_plan for "
                        "execute_validated_read; do not include on "
                        "propose_sql_read or repair_query_plan"
                    ),
                },
                "depends_on": ["same-decision action ids only"],
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
    raw = strip_json_fence(content)
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


def _normalize_terminal_decision(
    decision: DbPlannerDecision,
) -> tuple[DbPlannerDecision, str | None]:
    terminal_actions = tuple(
        action for action in decision.actions if action.kind in _TERMINAL_ACTION_KINDS
    )
    executable_actions = tuple(
        action
        for action in decision.actions
        if action.kind not in _TERMINAL_ACTION_KINDS
    )
    if terminal_actions and (len(terminal_actions) != 1 or executable_actions):
        return decision, "terminal_action_mixed_with_executable_actions"

    status = decision.status
    actions = decision.actions
    if terminal_actions:
        terminal_status = DbPlannerDecisionStatus(terminal_actions[0].kind.value)
        if status not in {DbPlannerDecisionStatus.CONTINUE, terminal_status}:
            return decision, "terminal_action_conflicts_with_decision_status"
        status = terminal_status
        actions = ()
    elif (
        status
        in {
            DbPlannerDecisionStatus.FINISH,
            DbPlannerDecisionStatus.CLARIFY,
        }
        and executable_actions
    ):
        return decision, "terminal_status_mixed_with_executable_actions"

    if (
        status is DbPlannerDecisionStatus.CLARIFY
        and not str(decision.clarification_question or "").strip()
    ):
        return decision, "clarification_question_required"

    if status is decision.status and actions == decision.actions:
        return decision, None
    return (
        DbPlannerDecision(
            status=status,
            intent=decision.intent,
            actions=actions,
            stop_conditions=decision.stop_conditions,
            clarification_question=decision.clarification_question,
            rationale=decision.rationale,
            metadata=decision.metadata,
        ),
        None,
    )


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
