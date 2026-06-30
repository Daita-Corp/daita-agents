"""LLM-backed outer planner for the DB agent loop."""

from __future__ import annotations

import json
import re
from typing import Any

from .llm_service import DbLLMService
from .planner_protocol import (
    DbAgentPlanner,
    DbLoopState,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
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
                    "llm": response.diagnostics,
                },
            )
        metadata = {
            **decision.metadata,
            "llm": response.diagnostics,
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
                "runtime validation and must not bypass governance."
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
    }


def _parse_planner_json(content: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    raw = _strip_json_fence(content)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, {"error": "planner_json_decode_error", "message": str(exc)}
    if not isinstance(parsed, dict):
        return None, {"error": "planner_json_not_object"}
    return parsed, {}


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped
