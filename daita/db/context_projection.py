"""Governed DB context projections for planner, result, and audit views."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Any, Iterable, Mapping

from daita.runtime import (
    Evidence,
    OperationStatus,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStreamEvent,
)

from .memory import PII_COLUMN_PATTERNS, _detect_pii_value as _memory_detect_pii_value
from .memory_contracts import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    db_memory_contract_refs,
)
from .models import DbOperationResult

_PUBLIC_WARNING_CODE = re.compile(r"[a-z0-9_.:-]+")
_PUBLIC_ERROR_TYPE = re.compile(r"[A-Za-z0-9_.:-]+")
_PUBLIC_TELEMETRY_FIELDS = (
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "llm_calls",
    "estimated_cost_usd",
    "latency_ms",
    "mode",
)
_PUBLIC_TRACE_ID_FIELDS = (
    "trace_id",
    "span_id",
    "root_span_id",
    "parent_span_id",
    "correlation_id",
    "operation_id",
    "execution_id",
    "request_id",
    "run_id",
)
_REDACTED_WARNING_CODE = "db_runtime_warning_redacted"
_REDACTED_ERROR_CODE = "db_runtime_error_redacted"
_PUBLIC_MACHINE_CODE_MAX_LENGTH = 128
_PUBLIC_RUNTIME_MESSAGES = {
    RuntimeEventType.OPERATION_CREATED: "Operation started.",
    RuntimeEventType.OPERATION_UPDATED: "Operation updated.",
    RuntimeEventType.TASK_CREATED: "Task planned.",
    RuntimeEventType.TASK_UPDATED: "Task updated.",
    RuntimeEventType.EVIDENCE_ACCEPTED: "Evidence accepted.",
    RuntimeEventType.POLICY_DECISION: "Policy evaluated.",
    RuntimeEventType.APPROVAL_REQUESTED: "Approval requested.",
    RuntimeEventType.APPROVAL_UPDATED: "Approval updated.",
    RuntimeEventType.LLM_REQUESTED: "Model request started.",
    RuntimeEventType.LLM_COMPLETED: "Model request completed.",
    RuntimeEventType.EXECUTOR_STARTED: "Executor started.",
    RuntimeEventType.EXECUTOR_COMPLETED: "Executor completed.",
    RuntimeEventType.EXECUTOR_FAILED: "Executor failed.",
    RuntimeEventType.MONITOR_TICKED: "Monitor ticked.",
    RuntimeEventType.MONITOR_TRIGGERED: "Monitor triggered.",
    RuntimeEventType.MONITOR_SKIPPED: "Monitor skipped.",
    RuntimeEventType.WORKER_HANDOFF: "Worker handoff recorded.",
    RuntimeEventType.WORKER_DELEGATED: "Work delegated.",
    RuntimeEventType.WORKER_LEASE_CLAIMED: "Worker lease claimed.",
    RuntimeEventType.WORKER_HEARTBEAT: "Worker heartbeat recorded.",
    RuntimeEventType.WORKER_COMPLETED: "Worker completed.",
    RuntimeEventType.WORKER_FAILED: "Worker failed.",
    RuntimeEventType.WORKER_TIMEOUT: "Worker timed out.",
    RuntimeEventType.WORKER_CANCELLED: "Worker cancelled.",
    RuntimeEventType.OPERATION_RESUMED: "Operation resumed.",
    RuntimeEventType.TASK_SKIPPED: "Task skipped.",
    RuntimeEventType.DIAGNOSTIC: "Runtime diagnostic.",
    RuntimeEventType.ERROR: "Runtime error.",
}


class ProjectionMode(str, Enum):
    """Caller mode for DB-owned context and evidence projection."""

    PLANNER = "planner"
    DIAGNOSTIC = "diagnostic"
    PUBLIC_RESULT = "public_result"
    AUDIT = "audit"


@dataclass(frozen=True)
class ProjectionContext:
    """Facts produced by existing owners and consumed by projection helpers."""

    mode: ProjectionMode = ProjectionMode.PLANNER
    operation_intent: str | None = None
    safety_frame: dict[str, Any] | None = None
    policy_summary: dict[str, Any] | None = None
    guardrail_summary: dict[str, Any] | None = None
    source_identity: str | None = None
    schema_fingerprint: str | None = None
    session_id: str | None = None
    user_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ProjectionMode(self.mode))
        object.__setattr__(self, "safety_frame", dict(self.safety_frame or {}))
        object.__setattr__(self, "policy_summary", dict(self.policy_summary or {}))
        object.__setattr__(
            self,
            "guardrail_summary",
            dict(self.guardrail_summary or {}),
        )

    @property
    def blocked_tables(self) -> frozenset[str]:
        values = []
        sources = (
            self.policy_summary or {},
            self.safety_frame or {},
            self.guardrail_summary or {},
        )
        for source in sources:
            values.extend(_string_list(source.get("blocked_tables")))
            values.extend(_string_list(source.get("deny_tables")))
            values.extend(_string_list(source.get("restricted_tables")))
        return frozenset(_ref_key(value) for value in values if _ref_key(value))

    @property
    def blocked_columns(self) -> frozenset[str]:
        values = []
        sources = (
            self.policy_summary or {},
            self.safety_frame or {},
            self.guardrail_summary or {},
        )
        for source in sources:
            values.extend(_string_list(source.get("blocked_columns")))
            values.extend(_string_list(source.get("blocked_fields")))
            values.extend(_string_list(source.get("sensitive_columns")))
            values.extend(_string_list(source.get("deny_columns")))
        return frozenset(_ref_key(value) for value in values if _ref_key(value))

    @property
    def blocked_values(self) -> frozenset[str]:
        values = []
        sources = (
            self.policy_summary or {},
            self.safety_frame or {},
            self.guardrail_summary or {},
        )
        for source in sources:
            values.extend(_string_list(source.get("blocked_values")))
            values.extend(_string_list(source.get("sensitive_values")))
        return frozenset(str(value).strip().lower() for value in values if str(value))


def policy_summary_from_source(source: Any) -> dict[str, Any]:
    """Return connector-owned policy facts without evaluating policy."""

    return {
        "read_only": getattr(source, "read_only", None),
        "allowed_tables": sorted(getattr(source, "allowed_tables", set()) or []),
        "blocked_tables": sorted(getattr(source, "blocked_tables", set()) or []),
        "blocked_columns": sorted(getattr(source, "blocked_columns", set()) or []),
    }


def project_policy_summary(
    policy_summary: Mapping[str, Any],
    projection: ProjectionContext,
) -> dict[str, Any]:
    """Project connector policy facts without exposing denied identifiers."""

    if projection.mode is ProjectionMode.AUDIT:
        return dict(policy_summary)
    blocked_tables = _string_list(policy_summary.get("blocked_tables"))
    blocked_columns = _string_list(policy_summary.get("blocked_columns"))
    result = {
        "read_only": policy_summary.get("read_only"),
        "allowed_tables": _string_list(policy_summary.get("allowed_tables")),
        "blocked_tables": ["<redacted>"] if blocked_tables else [],
        "blocked_columns": ["<redacted>"] if blocked_columns else [],
        "blocked_table_count": len(blocked_tables),
        "blocked_column_count": len(blocked_columns),
    }
    return result


def project_session_context(
    session_context: Mapping[str, Any] | None,
    projection: ProjectionContext,
) -> dict[str, Any]:
    """Project compact session context for one caller mode."""

    if not isinstance(session_context, Mapping):
        return {}
    if projection.mode is ProjectionMode.AUDIT:
        return dict(session_context)

    result: dict[str, Any] = {}
    if session_context.get("session_id") is not None:
        result["session_id"] = session_context.get("session_id")
    if (
        projection.mode is not ProjectionMode.PUBLIC_RESULT
        and session_context.get("user_id") is not None
    ):
        result["user_id"] = session_context.get("user_id")

    referents = _project_session_referents(session_context.get("referents"), projection)
    if referents:
        result["referents"] = referents

    recent_operations = session_context.get("recent_operations")
    if (
        isinstance(recent_operations, list)
        and projection.mode is not ProjectionMode.PUBLIC_RESULT
    ):
        result["recent_operations"] = [
            dict(item) for item in recent_operations if isinstance(item, Mapping)
        ]

    query_scopes = project_session_query_scopes(
        session_context.get("query_scopes") or (),
        projection,
    )
    if query_scopes:
        result["query_scopes"] = list(query_scopes)

    durable_ids = session_context.get("durable_ids")
    if (
        isinstance(durable_ids, Mapping)
        and projection.mode is not ProjectionMode.PUBLIC_RESULT
    ):
        result["durable_ids"] = dict(durable_ids)

    diagnostics = session_context.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        result["diagnostics"] = {
            **dict(diagnostics),
            "projection": _projection_summary(projection),
        }
    elif projection.mode is not ProjectionMode.PUBLIC_RESULT:
        result["diagnostics"] = {"projection": _projection_summary(projection)}
    return result


def project_session_query_scopes(
    query_scopes: Iterable[Any],
    projection: ProjectionContext,
) -> tuple[dict[str, Any], ...]:
    """Project prior query scopes without blocked filters or values."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(dict(item) for item in query_scopes if isinstance(item, Mapping))

    projected: list[dict[str, Any]] = []
    for scope in query_scopes:
        if not isinstance(scope, Mapping):
            continue
        tables = [
            table
            for table in _string_list(scope.get("tables"))
            if not _table_blocked(table, projection)
        ]
        filters = [
            item
            for item in (
                _project_session_filter(filter_item, projection)
                for filter_item in scope.get("filters", ()) or ()
            )
            if item is not None
        ]
        joins = _project_session_joins(scope.get("joins"), projection)
        selected_columns = [
            column
            for column in _string_list(scope.get("selected_columns"))
            if not _column_blocked_or_sensitive(column, projection)
        ]

        item: dict[str, Any] = {}
        if (
            projection.mode is not ProjectionMode.PUBLIC_RESULT
            and scope.get("scope_id") is not None
        ):
            item["scope_id"] = scope.get("scope_id")
        if (
            projection.mode is not ProjectionMode.PUBLIC_RESULT
            and scope.get("operation_id") is not None
        ):
            item["operation_id"] = scope.get("operation_id")
        if tables:
            item["tables"] = tables
        if filters:
            item["filters"] = filters
        if joins:
            item["joins"] = joins
        if selected_columns:
            item["selected_columns"] = selected_columns
        if isinstance(scope.get("result_row_count"), int):
            item["result_row_count"] = max(0, int(scope["result_row_count"]))
        if item and (
            tables or filters or joins or selected_columns or "result_row_count" in item
        ):
            projected.append(item)
    return tuple(projected)


def project_memory_refs(
    refs: Iterable[Mapping[str, Any]],
    projection: ProjectionContext,
) -> tuple[dict[str, Any], ...]:
    """Project recalled DB memory refs for planner/result visibility."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(dict(ref) for ref in refs)

    projected: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        reason = _memory_ref_redaction_reason(ref, projection)
        if reason:
            projected.append(_redacted_memory_ref(ref, reason, projection))
            continue
        item = dict(ref)
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            item.pop("text", None)
            item.pop(DB_MEMORY_SEMANTIC_CONTRACT_KEY, None)
        projected.append(item)
    return tuple(projected)


def project_memory_semantics(
    semantics: Iterable[Mapping[str, Any]],
    projection: ProjectionContext,
) -> tuple[dict[str, Any], ...]:
    """Project compact memory semantic contracts without blocked refs."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(dict(item) for item in semantics)

    projected: list[dict[str, Any]] = []
    for semantic in semantics:
        if not isinstance(semantic, Mapping):
            continue
        item = dict(semantic)
        if _memory_semantic_blocked(item, projection):
            item["enforceable"] = False
            item["projection"] = {
                "redacted": True,
                "reason": "blocked_by_policy",
            }
            for key in (
                "required_refs",
                "required_relationships",
                "required_filters",
                "required_aggregations",
                "result_shape",
                "unit_conversion",
            ):
                item.pop(key, None)
        elif projection.mode is ProjectionMode.PUBLIC_RESULT:
            for key in (
                "required_refs",
                "required_relationships",
                "required_filters",
                "required_aggregations",
            ):
                item.pop(key, None)
        projected.append(item)
    return tuple(projected)


def project_catalog_hints(
    hints: Iterable[Mapping[str, Any]],
    projection: ProjectionContext,
) -> tuple[dict[str, Any], ...]:
    """Project catalog value hints for planner-safe rendering."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(dict(item) for item in hints)

    projected: list[dict[str, Any]] = []
    for hint in hints:
        if not isinstance(hint, Mapping):
            continue
        table = str(hint.get("table") or "").strip()
        column = str(hint.get("column") or "").strip()
        if _table_blocked(table, projection) or _column_blocked_or_sensitive(
            f"{table}.{column}" if table and column else column,
            projection,
        ):
            continue
        item = dict(hint)
        if projection.blocked_values:
            item["observed_values"] = [
                value
                for value in hint.get("observed_values", []) or []
                if not _value_blocked(value, projection)
            ]
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            item.pop("observed_values", None)
            item.pop("candidate_mapping", None)
        projected.append(item)
    return tuple(projected)


def project_operation_evidence(
    evidence: Iterable[Evidence],
    projection: ProjectionContext,
) -> tuple[Evidence, ...]:
    """Project persisted evidence for caller-facing result views."""

    if projection.mode is ProjectionMode.AUDIT:
        return tuple(evidence)
    return tuple(
        replace(
            item,
            payload=_project_evidence_payload(item, projection),
            metadata={
                **(
                    {}
                    if projection.mode is ProjectionMode.PUBLIC_RESULT
                    else item.metadata
                ),
                "projection_mode": projection.mode.value,
                "projected": True,
            },
        )
        for item in evidence
    )


def project_operation_result(
    result: DbOperationResult,
    projection: ProjectionContext,
) -> DbOperationResult:
    """Build the allowlisted public view of one DB operation result."""

    if projection.mode is not ProjectionMode.PUBLIC_RESULT:
        raise ValueError("operation results require public_result projection")

    evidence = project_operation_evidence(result.evidence, projection)
    diagnostics = result.diagnostics
    projected_diagnostics: dict[str, Any] = {}

    planner = _project_result_planner_diagnostics(diagnostics.get("planner"))
    if planner:
        projected_diagnostics["planner"] = planner

    projected_diagnostics["execution"] = _project_result_execution_diagnostics(
        diagnostics.get("execution"),
        operation_id=result.operation_id,
        evidence=evidence,
    )

    verification = _project_result_verification_diagnostics(
        diagnostics.get("verification")
    )
    if verification:
        projected_diagnostics["verification"] = verification

    telemetry_source = diagnostics.get("telemetry")
    if not isinstance(telemetry_source, Mapping):
        telemetry_source = result.telemetry
    telemetry = _project_result_telemetry(telemetry_source)
    projected_diagnostics["synthesis"] = dict(telemetry)
    projected_diagnostics["telemetry"] = dict(telemetry)

    trace = _project_result_trace_diagnostics(diagnostics.get("trace"))
    if trace:
        projected_diagnostics["trace"] = trace

    error = _project_result_error_diagnostics(diagnostics.get("error"))
    if error:
        projected_diagnostics["error"] = error

    return replace(
        result,
        answer=_project_result_answer(result),
        evidence=evidence,
        warnings=_public_warning_codes(result.warnings),
        diagnostics=projected_diagnostics,
    )


def project_runtime_stream_event(
    event: RuntimeEvent,
    projection: ProjectionContext,
) -> RuntimeStreamEvent:
    """Build an allowlisted public progress event from one durable event."""

    if projection.mode is not ProjectionMode.PUBLIC_RESULT:
        raise ValueError("runtime stream events require public_result projection")
    stream_event = RuntimeStreamEvent.from_runtime_event(event)
    return replace(
        stream_event,
        message=(
            "Operation stream completed."
            if event.payload.get("terminal") is True
            else _PUBLIC_RUNTIME_MESSAGES[event.type]
        ),
        policy_id=None,
        payload=_project_runtime_event_payload(event),
    )


def project_runtime_stream_drop_diagnostic(
    *,
    operation_id: str,
    runtime_id: str,
    runtime_kind: str,
    dropped_count: int,
) -> RuntimeStreamEvent:
    """Return a bounded-delivery diagnostic without exposing dropped payloads."""

    return RuntimeStreamEvent(
        type=RuntimeEventType.DIAGNOSTIC,
        operation_id=operation_id,
        runtime_id=runtime_id,
        runtime_kind=runtime_kind,
        message="Runtime progress events were dropped for a slow consumer.",
        payload={
            "diagnostic": "runtime_stream.events_dropped",
            "dropped_event_count": max(0, int(dropped_count)),
            "projected": True,
        },
    )


def project_runtime_stream_terminal(
    result: DbOperationResult,
    *,
    runtime_id: str,
    runtime_kind: str,
) -> RuntimeStreamEvent:
    """Return the single public terminal delivery event for a DB result."""

    return RuntimeStreamEvent(
        type=RuntimeEventType.OPERATION_UPDATED,
        operation_id=result.operation_id,
        runtime_id=runtime_id,
        runtime_kind=runtime_kind,
        message="Operation stream completed.",
        payload=project_runtime_stream_terminal_payload(result),
    )


def project_runtime_stream_terminal_payload(
    result: DbOperationResult,
) -> dict[str, Any]:
    """Build the safe terminal payload persisted on the final runtime event."""

    execution = result.diagnostics.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    task_count = execution.get("task_count")
    if not isinstance(task_count, int) or isinstance(task_count, bool):
        task_count = len(execution.get("task_refs") or ())
    return {
        "terminal": True,
        "answer": result.answer,
        "result_summary": {
            "status": result.status.value,
            "warning_codes": list(_public_warning_codes(result.warnings)),
            "task_count": max(0, task_count),
            "evidence_count": len(result.evidence),
        },
        "projected": True,
    }


def _project_runtime_event_payload(event: RuntimeEvent) -> dict[str, Any]:
    payload = event.payload if isinstance(event.payload, Mapping) else {}
    if payload.get("terminal") is True:
        return _project_persisted_terminal_payload(payload)
    projected: dict[str, Any] = {"projected": True}

    for key in ("operation_type", "kind"):
        code = _public_machine_code(payload.get(key))
        if code is not None:
            projected[key] = code
    status = _public_machine_code(payload.get("status"))
    if status not in {"succeeded", "failed", "cancelled", "blocked"}:
        if status is not None:
            projected["status"] = status
    for key in ("accepted", "truncated", "success"):
        if isinstance(payload.get(key), bool):
            projected[key] = payload[key]
    for key in (
        "attempt_count",
        "payload_size",
        "row_count",
        "total_rows",
        "task_count",
        "evidence_count",
    ):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            projected[key] = max(0, value)
    warnings = _public_warning_codes(payload.get("warnings"))
    if warnings:
        projected["warning_codes"] = list(warnings)
    return projected


def _project_persisted_terminal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("result_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    status = _public_machine_code(summary.get("status"))
    if status not in {
        "pending",
        "planning",
        "running",
        "verifying",
        "succeeded",
        "failed",
        "cancelled",
        "blocked",
    }:
        status = "failed"
    projected_summary: dict[str, Any] = {
        "status": status,
        "warning_codes": list(_public_warning_codes(summary.get("warning_codes"))),
    }
    for key in ("task_count", "evidence_count"):
        value = summary.get(key)
        projected_summary[key] = (
            max(0, value)
            if isinstance(value, int) and not isinstance(value, bool)
            else 0
        )
    answer = payload.get("answer")
    return {
        "terminal": True,
        "answer": answer if isinstance(answer, str) or answer is None else None,
        "result_summary": projected_summary,
        "projected": True,
    }


def _project_result_answer(result: DbOperationResult) -> str | None:
    if result.status is OperationStatus.FAILED:
        return "DB operation failed before final synthesis."
    if result.status is OperationStatus.CANCELLED:
        return "DB operation was cancelled before completion."
    if (
        result.status is OperationStatus.BLOCKED
        and result.diagnostics.get("error") is not None
    ):
        return "DB operation was blocked before completion."
    return result.answer


def _project_result_planner_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    status = _public_machine_code(value.get("status"))
    if status is not None:
        result["status"] = status
    turn_count = _planner_turn_count(value)
    if turn_count is not None:
        result["turn_count"] = turn_count
    warnings = value.get("warning_codes", value.get("warnings"))
    result["warning_codes"] = list(_public_warning_codes(warnings))
    terminal_reason = value.get("terminal_reason_code")
    nested = value.get("diagnostics")
    if terminal_reason is None and isinstance(nested, Mapping):
        terminal_reason = nested.get("terminal_reason_code")
    if terminal_reason is None:
        terminal_reason = status
    terminal_reason_code = _public_machine_code(terminal_reason)
    if terminal_reason_code is not None:
        result["terminal_reason_code"] = terminal_reason_code
    return result


def _planner_turn_count(value: Mapping[str, Any]) -> int | None:
    candidates = [value.get("turn_count"), value.get("turn")]
    diagnostics = value.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        candidates.extend(
            (
                diagnostics.get("turn_count"),
                diagnostics.get("turn"),
                diagnostics.get("turn_budget"),
            )
        )
        observation = diagnostics.get("observation")
        if isinstance(observation, Mapping):
            observation_diagnostics = observation.get("diagnostics")
            if isinstance(observation_diagnostics, Mapping):
                candidates.extend(
                    (
                        observation_diagnostics.get("turn_count"),
                        observation_diagnostics.get("turn"),
                    )
                )
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return max(0, candidate)
    return None


def _project_result_execution_diagnostics(
    value: Any,
    *,
    operation_id: str,
    evidence: tuple[Evidence, ...],
) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    raw_task_refs = source.get("task_refs")
    if not isinstance(raw_task_refs, (list, tuple)):
        raw_task_refs = source.get("tasks")
    task_refs = _project_result_task_refs(raw_task_refs)
    task_count = source.get("task_count")
    if not isinstance(task_count, int) or isinstance(task_count, bool):
        task_count = len(task_refs)
    evidence_refs = [_public_evidence_ref(item) for item in evidence]
    return {
        "operation_id": operation_id,
        "task_count": max(0, task_count),
        "evidence_count": len(evidence_refs),
        "evidence_refs": evidence_refs,
        "task_refs": task_refs,
    }


def _project_result_task_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        task_id = item.get("id", item.get("task_id"))
        capability_id = item.get("capability_id")
        status = item.get("status")
        result.append(
            {
                "id": str(task_id) if task_id is not None else None,
                "capability_id": (
                    str(capability_id) if capability_id is not None else None
                ),
                "status": str(status) if status is not None else None,
            }
        )
    return result


def _public_evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "task_id": evidence.task_id,
        "accepted": evidence.accepted,
    }


def _project_result_verification_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    diagnostics = value.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    required = value.get("required_evidence_kinds")
    if required is None:
        required = value.get("required_evidence", diagnostics.get("required_evidence"))
    missing = value.get("missing_evidence_kinds")
    if missing is None:
        missing = value.get("missing_evidence")
    warnings = value.get("warning_codes", value.get("warnings"))
    return {
        "passed": value.get("passed") is True,
        "required_evidence_kinds": _public_machine_codes(required),
        "missing_evidence_kinds": _public_machine_codes(missing),
        "warning_codes": list(_public_warning_codes(warnings)),
    }


def _project_result_telemetry(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: value[key] for key in _PUBLIC_TELEMETRY_FIELDS if key in value}


def _project_result_trace_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: str(value[key])
        for key in _PUBLIC_TRACE_ID_FIELDS
        if value.get(key) is not None
    }


def _project_result_error_diagnostics(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    source = value if isinstance(value, Mapping) else {}
    error_type = source.get("type", source.get("error_type"))
    error_type_text = str(error_type or "Error")
    if _PUBLIC_ERROR_TYPE.fullmatch(error_type_text) is None:
        error_type_text = "Error"
    elif len(error_type_text) > _PUBLIC_MACHINE_CODE_MAX_LENGTH:
        error_type_text = "Error"
    code = _public_machine_code(source.get("code", source.get("error_code")))
    return {
        "type": error_type_text,
        "code": code or _REDACTED_ERROR_CODE,
    }


def _public_warning_codes(value: Any) -> tuple[str, ...]:
    result: list[str] = []
    for item in _string_list(value):
        code = _public_machine_code(item) or _REDACTED_WARNING_CODE
        if code not in result:
            result.append(code)
    return tuple(result)


def _public_machine_codes(value: Any) -> list[str]:
    return [
        code
        for item in _string_list(value)
        if (code := _public_machine_code(item)) is not None
    ]


def _public_machine_code(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if (
        len(text) > _PUBLIC_MACHINE_CODE_MAX_LENGTH
        or _PUBLIC_WARNING_CODE.fullmatch(text) is None
    ):
        return None
    return text


def _project_evidence_payload(
    evidence: Evidence,
    projection: ProjectionContext,
) -> dict[str, Any]:
    payload = evidence.payload if isinstance(evidence.payload, Mapping) else {}
    base: dict[str, Any] = {
        "projection_mode": projection.mode.value,
        "source_kind": evidence.kind,
        "accepted": evidence.accepted,
        "payload_keys": sorted(str(key) for key in payload.keys()),
    }
    if projection.mode is ProjectionMode.PUBLIC_RESULT:
        base["redacted"] = True

    if evidence.kind in {"sql.validation", "query.plan.validation"}:
        if "valid" in payload:
            base["valid"] = payload.get("valid") is True
        operation = payload.get("operation")
        if operation is not None:
            operation_code = _public_machine_code(operation)
            if operation_code is not None:
                base["operation"] = operation_code
        facts = _project_validation_items(
            payload.get("validation_facts")
            or payload.get("warnings")
            or payload.get("validation_warnings"),
            projection,
        )
        if facts and projection.mode is ProjectionMode.DIAGNOSTIC:
            base["validation_facts"] = facts
    elif evidence.kind == "query.result":
        rows = payload.get("rows")
        if isinstance(rows, list):
            base["row_count"] = len(rows)
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            total_rows = payload.get("total_rows")
            if isinstance(total_rows, int) and not isinstance(total_rows, bool):
                base["total_rows"] = max(0, total_rows)
            for key in ("truncated", "success"):
                if isinstance(payload.get(key), bool):
                    base[key] = payload[key]
        else:
            for key in ("total_rows", "truncated", "success"):
                if key in payload:
                    base[key] = payload[key]
        if "error" in payload:
            error = payload["error"]
            if projection.mode is ProjectionMode.PUBLIC_RESULT:
                base["error_code"] = _REDACTED_ERROR_CODE
                base["redacted_fields"] = ["error"]
            elif error is not None and _text_contains_blocked_or_sensitive(
                str(error),
                projection,
            ):
                base["error"] = "<redacted>"
                base["redacted_fields"] = ["error"]
            else:
                base["error"] = error
    elif evidence.kind == "planning.context":
        base["included_sections"] = list(payload.get("included_sections") or [])
        base["omitted_sections"] = list(payload.get("omitted_sections") or [])
        memory_refs = project_memory_refs(
            payload.get("db_memory_refs") or (),
            projection,
        )
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            memory_refs = tuple(
                item
                for item in memory_refs
                if not (
                    isinstance(item.get("projection"), Mapping)
                    and item["projection"].get("redacted") is True
                )
            )
        if memory_refs:
            base["db_memory_refs"] = [dict(item) for item in memory_refs]
        memory_semantics = project_memory_semantics(
            payload.get("db_memory_semantics") or (),
            projection,
        )
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            memory_semantics = tuple(
                item
                for item in memory_semantics
                if not (
                    isinstance(item.get("projection"), Mapping)
                    and item["projection"].get("redacted") is True
                )
            )
        if memory_semantics:
            base["db_memory_semantics"] = [dict(item) for item in memory_semantics]
        memory_diagnostics = _project_memory_diagnostics(
            payload.get("db_memory_diagnostics"),
        )
        if memory_diagnostics and memory_refs:
            base["db_memory_diagnostics"] = memory_diagnostics
        contract_diagnostics = _project_memory_contract_diagnostics(
            payload.get("db_memory_contract_diagnostics"),
        )
        if contract_diagnostics and memory_semantics:
            base["db_memory_contract_diagnostics"] = contract_diagnostics
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            base["diagnostics"] = {
                key: diagnostics[key]
                for key in (
                    "schema_table_count",
                    "column_value_hint_count",
                    "db_memory_ref_count",
                    "db_memory_contract_count",
                    "schema_fingerprint",
                )
                if key in diagnostics
            }
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            session_context = project_session_context(
                payload.get("session_context"),
                replace(projection, mode=ProjectionMode.DIAGNOSTIC),
            )
            if session_context:
                base["session_context"] = session_context
            base["column_value_hints"] = list(
                project_catalog_hints(
                    payload.get("column_value_hints") or (),
                    replace(projection, mode=ProjectionMode.DIAGNOSTIC),
                )
            )
            base["db_memory_ref_count"] = len(payload.get("db_memory_refs") or ())
    elif evidence.kind == "memory.semantic.recall":
        results = payload.get("results")
        if isinstance(results, list):
            base["result_count"] = len(results)
        diagnostics = _project_recall_diagnostics(payload.get("diagnostics"))
        if diagnostics:
            base["diagnostics"] = diagnostics
    elif evidence.kind == "db.memory.selection":
        base["raw_candidate_count"] = int(payload.get("raw_candidate_count") or 0)
        base["included_count"] = int(payload.get("included_count") or 0)
        omitted = payload.get("omitted_counts_by_reason")
        if isinstance(omitted, Mapping):
            base["omitted_count"] = sum(
                int(count) for count in omitted.values() if isinstance(count, int)
            )
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["omitted_counts_by_reason"] = dict(
                payload.get("omitted_counts_by_reason") or {}
            )
            base["safe_diagnostic_omission_summaries"] = [
                dict(item)
                for item in payload.get("safe_diagnostic_omission_summaries", ()) or ()
                if isinstance(item, Mapping)
            ]
            budget = payload.get("budget_usage")
            if isinstance(budget, Mapping):
                base["budget_usage"] = dict(budget)
    elif evidence.kind == "db.memory.contracts":
        enforceable = payload.get("enforceable_contracts")
        advisory = payload.get("advisory_contracts")
        if isinstance(enforceable, list):
            base["enforceable_count"] = len(enforceable)
        if isinstance(advisory, list):
            base["advisory_count"] = len(advisory)
        omitted = payload.get("contract_omission_reasons")
        if isinstance(omitted, Mapping):
            base["omitted_count"] = sum(
                int(count) for count in omitted.values() if isinstance(count, int)
            )
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["contract_omission_reasons"] = dict(
                payload.get("contract_omission_reasons") or {}
            )
            base["safe_diagnostic_summaries"] = [
                dict(item)
                for item in payload.get("safe_diagnostic_summaries", ()) or ()
                if isinstance(item, Mapping)
            ]
            applicability = payload.get("source_schema_applicability")
            if isinstance(applicability, Mapping):
                base["source_schema_applicability"] = dict(applicability)
    elif evidence.kind == "session.query_scope":
        base["table_count"] = len(_string_list(payload.get("tables")))
        base["filter_count"] = len(
            [
                item
                for item in payload.get("filters", ()) or ()
                if isinstance(item, Mapping)
            ]
        )
        base["join_count"] = len(
            [
                item
                for item in payload.get("joins", ()) or ()
                if isinstance(item, Mapping)
            ]
        )
        if isinstance(payload.get("result_row_count"), int):
            base["result_row_count"] = max(0, int(payload["result_row_count"]))
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["scope_id"] = payload.get("scope_id")
            base["source_operation_id"] = payload.get(
                "source_operation_id"
            ) or payload.get("operation_id")
            base["tables"] = [
                table
                for table in _string_list(payload.get("tables"))
                if not _table_blocked(table, projection)
            ]
            base["filters"] = [
                item
                for item in (
                    _project_session_filter(filter_item, projection)
                    for filter_item in payload.get("filters", ()) or ()
                )
                if item is not None
            ]
            base["joins"] = _project_session_joins(payload.get("joins"), projection)
    elif evidence.kind == "session.scope_binding":
        filters = [
            item
            for item in payload.get("required_filters", ()) or ()
            if isinstance(item, Mapping)
        ]
        joins = [
            item
            for item in payload.get("required_joins", ()) or ()
            if isinstance(item, Mapping)
        ]
        omitted = [
            item
            for item in payload.get("omitted_unsafe_referents", ()) or ()
            if isinstance(item, Mapping)
        ]
        base["required_filter_count"] = len(filters)
        base["required_join_count"] = len(joins)
        base["omitted_unsafe_referent_count"] = len(omitted)
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["binding_status"] = payload.get("binding_status")
            base["source_scope_id"] = payload.get("source_scope_id")
            base["source_operation_id"] = payload.get("source_operation_id")
            base["required_filters"] = [
                item
                for item in (
                    _project_session_filter(filter_item, projection)
                    for filter_item in filters
                )
                if item is not None
            ]
            base["required_joins"] = _project_session_joins(joins, projection)
            base["omitted_unsafe_referents"] = [
                {"reason": item.get("reason"), "count": item.get("count", 1)}
                for item in omitted
            ]
    elif evidence.kind == "schema.column_value_hint":
        hints = project_catalog_hints(
            payload.get("hints") or (),
            replace(projection, mode=ProjectionMode.DIAGNOSTIC),
        )
        base["hint_count"] = len(hints)
        if projection.mode is ProjectionMode.DIAGNOSTIC:
            base["hints"] = list(hints)
    elif evidence.kind == "schema.asset_profile":
        tables = payload.get("tables")
        if isinstance(tables, list):
            safe_tables = [
                table
                for table in tables
                if isinstance(table, Mapping)
                and not _table_blocked(str(table.get("name") or ""), projection)
            ]
            base["table_count"] = len(safe_tables)
            if projection.mode is ProjectionMode.DIAGNOSTIC:
                base["tables"] = [
                    {
                        "name": table.get("name"),
                        "columns": [
                            column.get("name")
                            for column in table.get("columns", []) or []
                            if isinstance(column, Mapping)
                            and not _column_blocked_or_sensitive(
                                f"{table.get('name')}.{column.get('name')}",
                                projection,
                            )
                        ],
                    }
                    for table in safe_tables
                ]
    else:
        if projection.mode is ProjectionMode.PUBLIC_RESULT:
            if isinstance(payload.get("success"), bool):
                base["success"] = payload["success"]
        else:
            for key in ("status", "success", "error", "reason"):
                value = payload.get(key)
                if value is not None and not _text_contains_blocked_or_sensitive(
                    str(value),
                    projection,
                ):
                    base[key] = value
    return base


def _project_recall_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value[key]
        for key in (
            "retrieval_mode",
            "embedding_available",
            "structured_candidate_count",
            "embedding_candidate_count",
            "returned_count",
            "limit",
        )
        if key in value
    }


def _project_memory_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value[key]
        for key in (
            "candidate_count",
            "included_count",
            "used_chars",
            "char_budget",
            "limit",
            "score_threshold",
            "omitted_reasons",
        )
        if key in value
    }


def _project_memory_contract_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value[key]
        for key in (
            "candidate_count",
            "enforced_count",
            "advisory_count",
            "omitted_count",
            "omitted_reasons",
        )
        if key in value
    }


def _project_session_referents(
    referents: Any,
    projection: ProjectionContext,
) -> dict[str, list[str]]:
    if not isinstance(referents, Mapping):
        return {}
    result: dict[str, list[str]] = {}
    for key, values in referents.items():
        strings = _string_list(values)
        if key == "tables":
            strings = [item for item in strings if not _table_blocked(item, projection)]
        elif key == "columns":
            strings = [
                item
                for item in strings
                if not _column_blocked_or_sensitive(item, projection)
            ]
        result[str(key)] = strings
    return result


def _project_session_filter(
    value: Any,
    projection: ProjectionContext,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    column = str(value.get("column") or "").strip()
    if not column or _column_blocked_or_sensitive(column, projection):
        return None
    values = [
        item
        for item in value.get("values", []) or []
        if not _value_blocked(item, projection)
    ]
    if not values:
        return None
    return {
        "column": column,
        "operator": str(value.get("operator") or "").strip(),
        "values": values,
    }


def _project_session_joins(
    value: Any,
    projection: ProjectionContext,
) -> list[dict[str, Any]]:
    joins = value if isinstance(value, (list, tuple)) else []
    projected = []
    for item in joins:
        if not isinstance(item, Mapping):
            continue
        left_table = str(item.get("left_table") or "").strip()
        right_table = str(item.get("right_table") or "").strip()
        left_column = str(item.get("left_column") or "").strip()
        right_column = str(item.get("right_column") or "").strip()
        if _table_blocked(left_table, projection) or _table_blocked(
            right_table,
            projection,
        ):
            continue
        left_ref = f"{left_table}.{left_column}" if left_column else left_table
        right_ref = f"{right_table}.{right_column}" if right_column else right_table
        if _column_blocked_or_sensitive(
            left_ref,
            projection,
        ) or _column_blocked_or_sensitive(right_ref, projection):
            continue
        projected.append(
            {
                "left_table": left_table,
                **({"left_column": left_column} if left_column else {}),
                "right_table": right_table,
                **({"right_column": right_column} if right_column else {}),
            }
        )
    return projected


def _project_validation_items(
    value: Any,
    projection: ProjectionContext,
) -> list[dict[str, Any]]:
    items = value if isinstance(value, (list, tuple)) else [value]
    projected: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        table = str(item.get("table") or item.get("table_name") or "").strip()
        column = str(item.get("column") or item.get("column_name") or "").strip()
        column_ref = f"{table}.{column}" if table and column else column
        if _table_blocked(table, projection) or _column_blocked_or_sensitive(
            column_ref,
            projection,
        ):
            projected.append(
                {
                    "kind": item.get("kind") or "validation_fact",
                    "redacted": True,
                    "reason": "blocked_by_policy",
                }
            )
            continue
        safe = {
            key: item[key]
            for key in (
                "kind",
                "table",
                "table_name",
                "column",
                "column_name",
                "operator",
                "candidates",
            )
            if key in item
        }
        for key in ("literal", "value", "filter_literal"):
            if key in item and not _value_blocked(item[key], projection):
                safe[key] = item[key]
        projected.append(safe)
    return projected


def _memory_ref_redaction_reason(
    ref: Mapping[str, Any],
    projection: ProjectionContext,
) -> str | None:
    contract = ref.get(DB_MEMORY_SEMANTIC_CONTRACT_KEY)
    if _contract_has_blocked_refs(contract, projection):
        return "blocked_by_policy"
    text = " ".join(
        str(ref.get(key) or "")
        for key in ("key", "text", "schema_fingerprint")
        if ref.get(key) is not None
    )
    if _text_contains_blocked_or_sensitive(text, projection):
        return "blocked_by_policy"
    return None


def _redacted_memory_ref(
    ref: Mapping[str, Any],
    reason: str,
    projection: ProjectionContext,
) -> dict[str, Any]:
    key = str(ref.get("key") or "").strip()
    if _text_contains_blocked_or_sensitive(key, projection):
        key = "redacted"
    result = {
        "chunk_id": ref.get("chunk_id"),
        "kind": ref.get("kind"),
        "key": key or "redacted",
        "confidence": ref.get("confidence"),
        "importance": ref.get("importance"),
        "source_identity": ref.get("source_identity"),
        "evidence_refs": list(ref.get("evidence_refs") or []),
        "schema_fingerprint": ref.get("schema_fingerprint"),
        "projection": {"redacted": True, "reason": reason},
    }
    if projection.mode is not ProjectionMode.PUBLIC_RESULT:
        result["text"] = "Memory reference redacted by projection policy."
    return {key: value for key, value in result.items() if value is not None}


def _memory_semantic_blocked(
    semantic: Mapping[str, Any],
    projection: ProjectionContext,
) -> bool:
    values = []
    values.extend(_string_list(semantic.get("required_refs")))
    values.extend(_string_list(semantic.get("required_relationships")))
    for item in semantic.get("required_filters", []) or []:
        if isinstance(item, Mapping):
            values.append(str(item.get("ref") or ""))
            values.append(str(item.get("value") or ""))
    for item in semantic.get("required_aggregations", []) or []:
        if isinstance(item, Mapping):
            values.append(str(item.get("ref") or ""))
    return any(
        _text_contains_blocked_or_sensitive(value, projection) for value in values
    )


def _contract_has_blocked_refs(
    contract: Any,
    projection: ProjectionContext,
) -> bool:
    if not isinstance(contract, Mapping):
        return False
    for ref in db_memory_contract_refs(dict(contract)):
        table = ref.get("table")
        column = ref.get("column")
        if table and _table_blocked(table, projection):
            return True
        if column and _column_blocked_or_sensitive(
            f"{table}.{column}" if table else column,
            projection,
        ):
            return True
    return False


def _table_blocked(table: Any, projection: ProjectionContext) -> bool:
    table_key = _ref_key(table)
    if not table_key:
        return False
    short = table_key.split(".")[-1]
    return table_key in projection.blocked_tables or short in projection.blocked_tables


def _column_blocked_or_sensitive(
    column_ref: Any,
    projection: ProjectionContext,
) -> bool:
    if _column_blocked(column_ref, projection):
        return True
    return _looks_sensitive_column(column_ref)


def _column_blocked(column_ref: Any, projection: ProjectionContext) -> bool:
    ref = _ref_key(column_ref)
    if not ref:
        return False
    table, column = _split_column_ref(ref)
    if table and _table_blocked(table, projection):
        return True
    if ref in projection.blocked_columns or column in projection.blocked_columns:
        return True
    for blocked in projection.blocked_columns:
        blocked_table, blocked_column = _split_column_ref(blocked)
        if ref == blocked:
            return True
        if column == blocked_column and (
            not table or not blocked_table or table == blocked_table
        ):
            return True
    return False


def _value_blocked(value: Any, projection: ProjectionContext) -> bool:
    raw = value.get("value") if isinstance(value, Mapping) else value
    text = str(raw or "").strip().lower()
    if not text:
        return False
    if text in projection.blocked_values:
        return True
    return bool(_memory_detect_pii_value(text))


def _text_contains_blocked_or_sensitive(
    text: str,
    projection: ProjectionContext,
) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if _memory_detect_pii_value(lowered):
        return True
    for table in projection.blocked_tables:
        if table and _ref_token_in_text(table, lowered):
            return True
    for column in projection.blocked_columns:
        if column and _ref_token_in_text(column, lowered):
            return True
        _table, short = _split_column_ref(column)
        if short and _ref_token_in_text(short, lowered):
            return True
    return False


def _looks_sensitive_column(column_ref: Any) -> bool:
    lowered = _ref_key(column_ref).replace(".", "_")
    if not lowered:
        return False
    return any(pattern in lowered for pattern in PII_COLUMN_PATTERNS)


def _ref_token_in_text(ref: str, text: str) -> bool:
    normalized = _ref_key(ref)
    if not normalized:
        return False
    variants = {normalized, normalized.replace(".", "_")}
    _table, column = _split_column_ref(normalized)
    if column:
        variants.add(column)
    return any(
        re.search(rf"(?<![a-z0-9_]){re.escape(value)}(?![a-z0-9_])", text)
        for value in variants
        if value
    )


def _split_column_ref(ref: str) -> tuple[str | None, str]:
    cleaned = _ref_key(ref)
    parts = [part for part in cleaned.split(".") if part]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, parts[-1] if parts else ""


def _ref_key(value: Any) -> str:
    text = str(value or "").strip().strip('`"[]').lower()
    return ".".join(part.strip('`"[]') for part in text.split(".") if part)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(value)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _projection_summary(projection: ProjectionContext) -> dict[str, Any]:
    return {
        "mode": projection.mode.value,
        "blocked_table_count": len(projection.blocked_tables),
        "blocked_column_count": len(projection.blocked_columns),
    }
