"""LLM-backed DB query planner and repair executors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from daita.runtime import Evidence, Operation, Task

from .evidence import load_evidence
from .fingerprints import persisted_fingerprint
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
                    accepted=False,
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
        accepted = _query_plan_proposal_accepted(payload)
        return [
            Evidence(
                kind="query.plan.proposal",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                accepted=accepted,
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
                    payload={
                        "valid": False,
                        "failure": "planner_unavailable",
                        "parse_succeeded": False,
                    },
                )
            ]
        required_inputs = _repair_input_diagnostics(task)
        if required_inputs["missing_input_ids"]:
            return _missing_repair_input_evidence(
                owner=self.owner,
                operation=operation,
                task=task,
                diagnostics=required_inputs,
            )
        planning_context = await load_evidence(
            runtime,
            operation.id,
            str(task.input.get("planning_context_evidence_id") or ""),
        )
        failure = await load_evidence(
            runtime,
            operation.id,
            str(task.input.get("failure_evidence_id") or ""),
        )
        prior_plan = await load_evidence(
            runtime,
            operation.id,
            str(task.input.get("prior_plan_evidence_id") or ""),
        )
        required_inputs = _repair_input_diagnostics(
            task,
            planning_context=planning_context,
            failure=failure,
            prior_plan=prior_plan,
        )
        if planning_context is None or failure is None or prior_plan is None:
            return _missing_repair_input_evidence(
                owner=self.owner,
                operation=operation,
                task=task,
                diagnostics=required_inputs,
            )
        messages = _repair_messages(
            planning_context.payload,
            prior_plan.payload,
            failure.payload,
        )
        response = await service.generate_json(messages)
        parsed, diagnostics = _parse_plan_response(response.content)
        repair = Evidence(
            kind="query.plan.repair",
            owner=self.owner,
            operation_id=operation.id,
            task_id=task.id,
            accepted=False,
            payload={
                "valid": parsed is not None,
                "parse_succeeded": parsed is not None,
                "repair_inputs_present": True,
                "failure_evidence_id": failure.id,
                "prior_plan_evidence_id": prior_plan.id,
                "planning_context_evidence_id": planning_context.id,
                "raw_model_response": response.content,
                "parse_diagnostics": diagnostics,
                "planner_diagnostics": response.diagnostics,
            },
        )
        if parsed is None:
            return [repair]
        plan = DbQueryPlan.from_mapping(parsed)
        prior_sql = _sql_from_plan_payload(prior_plan.payload)
        context_changed = _repair_failure_context_changed(
            failure.payload,
            planning_context=planning_context,
        )
        repeated = _same_sql(plan.selected_sql, prior_sql)
        repeated_blocked = repeated and not context_changed
        non_executable_reason = _repair_non_executable_reason(plan)
        payload = {
            **_plan_payload(
                plan,
                raw_model_response=response.content,
                planning_context=planning_context,
                planner_diagnostics=response.diagnostics,
                parse_diagnostics=diagnostics,
            ),
            "repair_attempt": int(task.input.get("repair_attempt") or 1),
            "repaired_failure_evidence_id": failure.id,
            "repeated_sql_blocked": repeated_blocked,
            "repair_context_changed": context_changed,
            "repair_inputs_present": True,
        }
        if repeated and context_changed:
            payload["repeated_sql_allowed_context_changed"] = True
        if non_executable_reason is not None:
            payload.update(
                {
                    "valid": False,
                    "failure": "repair_non_executable_plan",
                    "repair_rejection_reason": non_executable_reason,
                }
            )
        accepted = (
            non_executable_reason is None
            and _query_plan_proposal_accepted(payload)
            and not repeated_blocked
        )
        repair_payload = {
            **repair.payload,
            "valid": repair.payload["parse_succeeded"],
            "proposal_accepted": accepted,
            "repeated_sql_blocked": repeated_blocked,
            "repair_context_changed": context_changed,
        }
        if repeated and context_changed:
            repair_payload["repeated_sql_allowed_context_changed"] = True
        if non_executable_reason is not None:
            repair_payload.update(
                {
                    "failure": "repair_non_executable_plan",
                    "repair_rejection_reason": non_executable_reason,
                }
            )
        repair = Evidence(
            kind=repair.kind,
            owner=repair.owner,
            operation_id=repair.operation_id,
            task_id=repair.task_id,
            accepted=accepted,
            payload=repair_payload,
        )
        proposal = Evidence(
            kind="query.plan.proposal",
            owner=self.owner,
            operation_id=operation.id,
            task_id=task.id,
            accepted=accepted,
            payload=payload,
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
                "queries. DB memory is semantic business context. Source-matched, "
                "active, high-confidence DB memory grounded in the current "
                "schema/catalog may define metric meanings, unit conventions, "
                "aliases, and caveats. If eligible DB memory defines a metric "
                "or unit convention in the user prompt, the plan must satisfy "
                "the memory contract or ask a clarification question. Schema, "
                "catalog, policy, SQL validation, and connector guardrails still "
                "override memory. Never invent tables, columns, relationships, "
                "values, or permissions from memory alone. Ignore memory with "
                "mismatched source identity, stale schema fingerprint, inactive "
                "status, low confidence, or missing catalog citation. If memory "
                "changes SQL, cite its keys or evidence refs in assumptions or "
                "diagnostics."
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
                "Repair a failed executable read database query plan. Return "
                "only strict JSON for a complete revised DbQueryPlan. The "
                'operation field must be exactly "read", selected_sql must be '
                "a non-empty executable SELECT query, and candidates should "
                "describe the executable SQL. Do not return analysis, "
                "write_propose, schema, revised_plan, or meta-plan JSON. Do "
                "not repeat the same SQL unless the failure facts show the "
                "planning context changed."
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
    evidence = await load_evidence(runtime, operation.id, context_id)
    if evidence is None:
        evidence = await runtime.tasks.latest_accepted_evidence(
            operation.id, "planning.context"
        )
    if evidence is None:
        raise RuntimeError("planning.context evidence is required")
    return evidence


def _parse_plan_response(content: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"parsed": False}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        diagnostics.update(
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "raw_length": len(content),
            }
        )
        return None, diagnostics
    if not isinstance(parsed, dict):
        diagnostics["error"] = "planner_json_not_object"
        return None, diagnostics
    normalized = dict(parsed)
    if (
        "selected_sql" not in normalized
        and isinstance(normalized.get("sql"), str)
        and normalized["sql"].strip()
    ):
        normalized["selected_sql"] = normalized["sql"]
        diagnostics["normalized_aliases"] = {"sql": "selected_sql"}
    normalized.pop("sql", None)
    try:
        DbQueryPlan.from_mapping(normalized)
    except Exception as exc:
        diagnostics.update({"error": str(exc), "error_type": type(exc).__name__})
        return None, diagnostics
    diagnostics["parsed"] = True
    return normalized, diagnostics


def _query_plan_proposal_accepted(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("valid") is True
        and isinstance(payload.get("sql"), str)
        and bool(payload["sql"].strip())
    )


def _repair_non_executable_reason(plan: DbQueryPlan) -> str | None:
    if plan.operation != "read":
        return f"operation_not_read:{plan.operation}"
    if not isinstance(plan.selected_sql, str) or not plan.selected_sql.strip():
        return "missing_selected_sql"
    return None


def _repair_failure_context_changed(
    failure_payload: Mapping[str, Any],
    *,
    planning_context: Evidence,
) -> bool:
    failed_context_id = str(
        failure_payload.get("planning_context_evidence_id") or ""
    ).strip()
    if not failed_context_id or failed_context_id == planning_context.id:
        return False
    return _failure_facts_are_context_sensitive(failure_payload)


def _failure_facts_are_context_sensitive(
    failure_payload: Mapping[str, Any],
) -> bool:
    context_sensitive_kinds = {
        "filter_literal_requires_grounding",
        "unobserved_filter_literal",
        "ambiguous_literal_column",
    }
    values: list[Any] = []
    for key in ("validation_facts", "warnings", "validation_warnings", "errors"):
        value = failure_payload.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(value)
        elif value is not None:
            values.append(value)
    for value in values:
        if isinstance(value, Mapping):
            kind = str(value.get("kind") or "").strip()
            if kind in context_sensitive_kinds:
                return True
        text = str(value or "")
        if any(kind in text for kind in context_sensitive_kinds):
            return True
    return False


def _plan_payload(
    plan: DbQueryPlan,
    *,
    raw_model_response: str,
    planning_context: Evidence,
    planner_diagnostics: dict[str, Any],
    parse_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    plan_dict = plan.to_dict()
    executable_sql = isinstance(plan.selected_sql, str) and bool(
        plan.selected_sql.strip()
    )
    return {
        "valid": executable_sql,
        "sql": plan.selected_sql,
        "structured_plan": plan_dict,
        "raw_model_response": raw_model_response,
        "parse_diagnostics": parse_diagnostics,
        "planner_diagnostics": planner_diagnostics,
        "planning_context_evidence_id": planning_context.id,
        "planning_context_refs": [planning_context.id] if planning_context.id else [],
        "schema_fingerprint": planning_context.payload.get("schema_fingerprint"),
        "plan_fingerprint": persisted_fingerprint(plan_dict),
        "sql_fingerprint": (
            sql_fingerprint(plan.selected_sql) if plan.selected_sql else None
        ),
    }


def _repair_input_diagnostics(
    task: Task,
    *,
    planning_context: Evidence | None = None,
    failure: Evidence | None = None,
    prior_plan: Evidence | None = None,
) -> dict[str, Any]:
    required = {
        "planning_context_evidence_id": planning_context,
        "failure_evidence_id": failure,
        "prior_plan_evidence_id": prior_plan,
    }
    missing_ids = [
        key for key in required if not str(task.input.get(key) or "").strip()
    ]
    missing_evidence = [
        key
        for key, evidence in required.items()
        if key not in missing_ids and evidence is None
    ]
    return {
        "missing_input_ids": missing_ids,
        "missing_input_evidence": missing_evidence,
        "input_ids": {
            key: str(task.input.get(key) or "").strip()
            for key in required
            if str(task.input.get(key) or "").strip()
        },
    }


def _missing_repair_input_evidence(
    *,
    owner: str,
    operation: Operation,
    task: Task,
    diagnostics: Mapping[str, Any],
) -> list[Evidence]:
    payload = {
        "valid": False,
        "failure": "repair_inputs_missing",
        "parse_succeeded": False,
        "repair_inputs_present": False,
        "missing_input_ids": list(diagnostics.get("missing_input_ids") or ()),
        "missing_input_evidence": list(diagnostics.get("missing_input_evidence") or ()),
        "input_ids": dict(diagnostics.get("input_ids") or {}),
    }
    return [
        Evidence(
            kind="query.plan.repair",
            owner=owner,
            operation_id=operation.id,
            task_id=task.id,
            accepted=False,
            payload=payload,
        ),
        Evidence(
            kind="query.plan.proposal",
            owner=owner,
            operation_id=operation.id,
            task_id=task.id,
            accepted=False,
            payload=payload,
        ),
    ]


def _sql_from_plan_payload(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("sql")
    if isinstance(value, str) and value.strip():
        return value
    value = payload.get("selected_sql")
    if isinstance(value, str) and value.strip():
        return value
    structured_plan = payload.get("structured_plan")
    if isinstance(structured_plan, Mapping):
        value = structured_plan.get("selected_sql")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _same_sql(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return sql_fingerprint(left) == sql_fingerprint(right)
