"""LLM-backed DB query planner and repair executors."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

from daita.runtime import Evidence, Operation, Task

from .query_plan import DbQueryPlan
from .query_sql_validation import sql_fingerprint


@dataclass(frozen=True)
class DbLLMPlannerExecutor:
    """Strict structured planner executor for `db.query.plan`."""

    id: str = "db_runtime.query.plan.llm"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.query.plan"})
    runtime: Any = None

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = _runtime_from_boundary(self.runtime)
        service = runtime.db_llm_service
        if not service.available:
            return [
                Evidence(
                    kind="query.plan.proposal",
                    owner=self.owner,
                    operation_id=operation.id,
                    task_id=task.id,
                    accepted=False,
                    payload={
                        "valid": False,
                        "failure": "planner_unavailable",
                        "diagnostics": {"reason": "db_llm_service_unavailable"},
                    },
                )
            ]
        planning_context = await _load_context_evidence(runtime, task, operation)
        messages = _planner_messages(planning_context.payload)
        response = await service.generate_json(messages)
        parsed, diagnostics = _parse_plan_response(response.content)
        if parsed is None:
            return [
                Evidence(
                    kind="query.plan.proposal",
                    owner=self.owner,
                    operation_id=operation.id,
                    task_id=task.id,
                    payload={
                        "valid": False,
                        "failure": "planner_json_invalid",
                        "raw_model_response": response.content,
                        "parse_diagnostics": diagnostics,
                        "planner_diagnostics": response.diagnostics,
                        "planning_context_evidence_id": planning_context.id,
                    },
                )
            ]
        plan = DbQueryPlan.from_mapping(parsed)
        payload = _plan_payload(
            plan,
            raw_model_response=response.content,
            planning_context=planning_context,
            planner_diagnostics=response.diagnostics,
            parse_diagnostics=diagnostics,
        )
        return [
            Evidence(
                kind="query.plan.proposal",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
                metadata={
                    "schema_fingerprint": planning_context.payload.get(
                        "schema_fingerprint"
                    ),
                    "plan_fingerprint": payload["plan_fingerprint"],
                    "sql_fingerprint": payload.get("sql_fingerprint"),
                },
            )
        ]


@dataclass(frozen=True)
class DbLLMRepairExecutor:
    """One-shot repair executor for failed plan/SQL/result attempts."""

    id: str = "db_runtime.query.repair.llm"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.query.repair"})
    runtime: Any = None

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = _runtime_from_boundary(self.runtime)
        service = runtime.db_llm_service
        if not service.available:
            return [
                Evidence(
                    kind="query.plan.repair",
                    owner=self.owner,
                    operation_id=operation.id,
                    task_id=task.id,
                    accepted=False,
                    payload={"valid": False, "failure": "planner_unavailable"},
                )
            ]
        planning_context = await _load_context_evidence(runtime, task, operation)
        failure = await _load_evidence_id(
            runtime,
            operation.id,
            str(task.input.get("failure_evidence_id") or ""),
        )
        prior_plan = await _load_evidence_id(
            runtime,
            operation.id,
            str(task.input.get("prior_plan_evidence_id") or ""),
        )
        messages = _repair_messages(
            planning_context.payload,
            prior_plan.payload if prior_plan is not None else {},
            failure.payload if failure is not None else {},
        )
        response = await service.generate_json(messages)
        parsed, diagnostics = _parse_plan_response(response.content)
        repair = Evidence(
            kind="query.plan.repair",
            owner=self.owner,
            operation_id=operation.id,
            task_id=task.id,
            payload={
                "valid": parsed is not None,
                "failure_evidence_id": getattr(failure, "id", None),
                "prior_plan_evidence_id": getattr(prior_plan, "id", None),
                "raw_model_response": response.content,
                "parse_diagnostics": diagnostics,
                "planner_diagnostics": response.diagnostics,
            },
        )
        if parsed is None:
            return [repair]
        plan = DbQueryPlan.from_mapping(parsed)
        prior_sql = (
            prior_plan.payload.get("structured_plan", {}).get("selected_sql")
            if prior_plan is not None
            else None
        )
        repeated = bool(prior_sql and plan.selected_sql == prior_sql)
        proposal = Evidence(
            kind="query.plan.proposal",
            owner=self.owner,
            operation_id=operation.id,
            task_id=task.id,
            accepted=not repeated,
            payload={
                **_plan_payload(
                    plan,
                    raw_model_response=response.content,
                    planning_context=planning_context,
                    planner_diagnostics=response.diagnostics,
                    parse_diagnostics=diagnostics,
                ),
                "repair_attempt": int(task.input.get("repair_attempt") or 1),
                "repaired_failure_evidence_id": getattr(failure, "id", None),
                "repeated_sql_blocked": repeated,
            },
        )
        return [repair, proposal]


def _planner_messages(context_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a database query planner inside a governed runtime. "
                "Return only strict JSON matching the requested schema. You may "
                "propose SQL or a clarification question, but you must never "
                "execute database work. The operation field must be one of: "
                "read, write_propose, schema, analysis. Use read for SELECT "
                "queries. DB memory is advisory semantic context only. Schema, "
                "catalog, policy, SQL validation, and connector guardrails "
                "override memory. Memory may explain business meaning, metric "
                "or unit conventions, aliases, and caveats, but it must never "
                "justify inventing tables, columns, relationships, values, or "
                "permissions absent from schema/catalog evidence. Ignore memory "
                "with mismatched source identity, stale schema fingerprint, "
                "inactive status, low confidence, or missing catalog citation. "
                "If memory changes SQL, cite its keys or evidence refs in "
                "assumptions or diagnostics."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{context_payload.get('rendered_context', '')}\n\n"
                "Return JSON with keys: operation, selected_sql, candidates, "
                "selected_tables, joins, filters, aggregations, group_by, "
                "order_by, limit, assumptions, clarification_question, "
                "confidence, planner. For read queries set operation exactly "
                'to "read".'
            ),
        },
    ]


def _runtime_from_boundary(boundary: Any) -> Any:
    runtime = getattr(boundary, "runtime", None)
    return runtime if runtime is not None else boundary


def _repair_messages(
    context_payload: dict[str, Any],
    prior_plan_payload: dict[str, Any],
    failure_payload: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair a failed database query plan. Return only strict JSON "
                "for a complete revised plan. Do not repeat the same SQL unless "
                "the failure facts show the context changed. The operation "
                "field must be one of: read, write_propose, schema, analysis. "
                "Use read for SELECT queries."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "planning_context": context_payload,
                    "prior_plan": prior_plan_payload,
                    "failure": failure_payload,
                },
                sort_keys=True,
                default=str,
            ),
        },
    ]


async def _load_context_evidence(
    runtime: Any, task: Task, operation: Operation
) -> Evidence:
    context_id = str(task.input.get("planning_context_evidence_id") or "")
    evidence = await _load_evidence_id(runtime, operation.id, context_id)
    if evidence is None:
        evidence = await runtime._latest_accepted_evidence(
            operation.id, "planning.context"
        )
    if evidence is None:
        raise RuntimeError("planning.context evidence is required")
    return evidence


async def _load_evidence_id(
    runtime: Any,
    operation_id: str,
    evidence_id: str,
) -> Evidence | None:
    if not evidence_id:
        return None
    for evidence in await runtime.store.list_evidence(operation_id):
        if evidence.id == evidence_id:
            return evidence
    return None


def _parse_plan_response(content: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    raw = _strip_json_fence(content)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, {"error": str(exc), "raw_length": len(content)}
    if not isinstance(parsed, dict):
        return None, {"error": "planner_json_not_object"}
    try:
        DbQueryPlan.from_mapping(parsed)
    except Exception as exc:
        return None, {"error": str(exc), "error_type": type(exc).__name__}
    return parsed, {"parsed": True}


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _plan_payload(
    plan: DbQueryPlan,
    *,
    raw_model_response: str,
    planning_context: Evidence,
    planner_diagnostics: dict[str, Any],
    parse_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    plan_dict = plan.to_dict()
    return {
        "valid": bool(plan.selected_sql or plan.clarification_question),
        "sql": plan.selected_sql,
        "structured_plan": plan_dict,
        "raw_model_response": raw_model_response,
        "parse_diagnostics": parse_diagnostics,
        "planner_diagnostics": planner_diagnostics,
        "planning_context_evidence_id": planning_context.id,
        "planning_context_refs": [planning_context.id] if planning_context.id else [],
        "schema_fingerprint": planning_context.payload.get("schema_fingerprint"),
        "plan_fingerprint": _fingerprint(plan_dict),
        "sql_fingerprint": (
            sql_fingerprint(plan.selected_sql) if plan.selected_sql else None
        ),
    }


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()
