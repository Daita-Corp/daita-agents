"""Runtime-native evidence extraction for evals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Mapping

from .models import (
    RunMetrics,
    RuntimeEvidenceRecord,
    RuntimeGovernanceRecord,
    RuntimeTaskEvidence,
    StabilitySummary,
)


@dataclass
class RunEvidence:
    """Normalized runtime evidence for one agent run."""

    answer: str
    prompt_hash: str
    answer_hash: str
    operation_id: str | None
    operation_status: str | None
    operation_type: str | None
    intent: str | None
    tasks: list[RuntimeTaskEvidence]
    evidence: list[RuntimeEvidenceRecord]
    governance: RuntimeGovernanceRecord | None
    approvals: list[dict[str, Any]]
    warnings: list[str]
    metrics: RunMetrics
    trace_id: str | None = None


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def preview_text(value: Any, max_chars: int = 240) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def extract_run_evidence(prompt: str, raw_result: Any) -> RunEvidence:
    """Return eval evidence from a runtime-native operation result.

    Accepted inputs are ``DbOperationResult``-like objects, ``OperationSnapshot``-
    like objects, or dictionaries that wrap those objects under ``runtime_result``
    with optional runner metrics.
    """

    wrapper = raw_result if isinstance(raw_result, Mapping) else {}
    runtime_result = wrapper.get("runtime_result", raw_result)
    result = _runtime_result_mapping(runtime_result)
    diagnostics = _mapping(result.get("diagnostics"))
    execution = _mapping(diagnostics.get("execution"))
    operation = _mapping(result.get("operation"))

    answer = str(result.get("answer") or _snapshot_answer(result) or "")
    operation_id = _optional_string(
        result.get("operation_id") or operation.get("id") or result.get("id")
    )
    operation_status = _optional_string(
        _enum_value(result.get("status") or operation.get("status"))
    )
    contract = _mapping(result.get("contract"))
    intent = _mapping(result.get("intent"))
    tasks = [_task_from_mapping(item) for item in _list(execution.get("tasks"))]
    if not tasks and "tasks" in result:
        tasks = [_task_from_mapping(item) for item in _list(result.get("tasks"))]
    evidence = [_evidence_from_mapping(item) for item in _list(result.get("evidence"))]
    governance = _governance_from_mapping(diagnostics.get("governance"))
    approvals = _approval_records(result, diagnostics, governance)
    metrics = summarize_run_metrics(result, wrapper, execution)

    return RunEvidence(
        answer=answer,
        prompt_hash=stable_hash(prompt),
        answer_hash=stable_hash(answer),
        operation_id=operation_id,
        operation_status=operation_status,
        operation_type=_optional_string(
            contract.get("operation_type") or operation.get("operation_type")
        ),
        intent=_optional_string(_enum_value(intent.get("kind"))),
        tasks=tasks,
        evidence=evidence,
        governance=governance,
        approvals=approvals,
        warnings=[str(item) for item in _list(result.get("warnings"))],
        metrics=metrics,
        trace_id=operation_id,
    )


def extract_capability_sequence(
    tasks: Iterable[RuntimeTaskEvidence],
) -> tuple[str, ...]:
    return tuple(task.capability_id for task in tasks)


def extract_sql_statements(evidence: Iterable[RuntimeEvidenceRecord]) -> list[str]:
    statements = []
    for item in evidence:
        sql = item.payload.get("sql") or item.payload.get("query")
        if isinstance(sql, str):
            statements.append(sql)
        facts = item.payload.get("statement_facts")
        if isinstance(facts, dict):
            statement_sql = facts.get("sql")
            if isinstance(statement_sql, str):
                statements.append(statement_sql)
    return statements


def summarize_stability(runs: list[RunEvidence]) -> StabilitySummary:
    costs = [r.metrics.cost for r in runs if r.metrics.cost is not None]
    latencies = [r.metrics.latency_ms for r in runs if r.metrics.latency_ms is not None]
    tokens = [
        r.metrics.tokens_total for r in runs if r.metrics.tokens_total is not None
    ]
    return StabilitySummary(
        answer_variants=len({r.answer_hash for r in runs}),
        capability_sequence_variants=len(
            {extract_capability_sequence(r.tasks) for r in runs}
        ),
        cost_min=min(costs) if costs else None,
        cost_max=max(costs) if costs else None,
        latency_ms_min=min(latencies) if latencies else None,
        latency_ms_max=max(latencies) if latencies else None,
        latency_ms_p50=percentile(latencies, 50) if latencies else None,
        latency_ms_p95=percentile(latencies, 95) if latencies else None,
        latency_ms_p99=percentile(latencies, 99) if latencies else None,
        token_min=min(tokens) if tokens else None,
        token_max=max(tokens) if tokens else None,
    )


def percentile(values: list[float], percentile_value: float) -> float:
    """Return an interpolated percentile for runtime latency gates."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if percentile_value <= 0:
        return min(values)
    if percentile_value >= 100:
        return max(values)

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile_value / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def metric_delta_pct(
    min_value: float | int | None, max_value: float | int | None
) -> float:
    if min_value is None or max_value is None or min_value == 0:
        return 0.0
    return ((max_value - min_value) / min_value) * 100


def metric_snapshot(target: Any) -> dict[str, Any]:
    llm = getattr(target, "llm", target)
    tokens = {}
    if hasattr(llm, "get_accumulated_tokens"):
        tokens = llm.get_accumulated_tokens()
    cost = llm.get_accumulated_cost() if hasattr(llm, "get_accumulated_cost") else None
    return {"tokens": tokens or {}, "cost": cost}


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not before or not after:
        return {}
    before_tokens = before.get("tokens") or {}
    after_tokens = after.get("tokens") or {}
    return {
        "tokens_total": (after_tokens.get("total_tokens") or 0)
        - (before_tokens.get("total_tokens") or 0),
        "cost": (after.get("cost") or 0) - (before.get("cost") or 0),
    }


def summarize_run_metrics(
    result: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> RunMetrics:
    llm = _mapping(_mapping(result.get("diagnostics")).get("llm"))
    tokens = _mapping(llm.get("tokens") or wrapper.get("tokens"))
    deltas = _mapping(wrapper.get("_eval_metric_delta"))
    return RunMetrics(
        latency_ms=_optional_float(wrapper.get("latency_ms")),
        tokens_total=deltas.get("tokens_total")
        or tokens.get("total_tokens")
        or wrapper.get("tokens_total"),
        cost=deltas.get("cost") if "cost" in deltas else llm.get("cost"),
        iterations=execution.get("task_count") or len(_list(execution.get("tasks"))),
    )


def _runtime_result_mapping(raw: Any) -> dict[str, Any]:
    if raw is None:
        raise TypeError("Eval target returned no runtime result.")
    if isinstance(raw, Mapping):
        if "tool_calls" in raw:
            raise TypeError(
                "daita.evals is runtime-native and no longer accepts legacy "
                "tool_calls results. Return a DbOperationResult from run_detailed()."
            )
        if "runtime_result" in raw:
            return _runtime_result_mapping(raw["runtime_result"])
        if "operation" in raw and "tasks" in raw:
            return dict(raw)
        if "operation_id" in raw and ("evidence" in raw or "diagnostics" in raw):
            return dict(raw)
        raise TypeError(
            "Eval target must return a runtime-native DbOperationResult or "
            "OperationSnapshot payload."
        )
    if hasattr(raw, "to_dict"):
        return _runtime_result_mapping(raw.to_dict())

    values = {}
    for name in (
        "operation_id",
        "request",
        "intent",
        "contract",
        "status",
        "answer",
        "evidence",
        "warnings",
        "diagnostics",
    ):
        if hasattr(raw, name):
            values[name] = getattr(raw, name)
    if values:
        return {key: _jsonable(value) for key, value in values.items()}
    raise TypeError(
        "Eval target must return a runtime-native DbOperationResult or "
        "OperationSnapshot payload."
    )


def _task_from_mapping(raw: Any) -> RuntimeTaskEvidence:
    data = _mapping(raw)
    metadata = _mapping(data.get("metadata"))
    return RuntimeTaskEvidence(
        id=str(data.get("id") or ""),
        capability_id=str(data.get("capability_id") or ""),
        executor_id=str(data.get("executor_id") or ""),
        owner=_optional_string(metadata.get("owner")),
        status=str(_enum_value(data.get("status")) or ""),
        input=_mapping(data.get("input")),
        required_evidence=[str(item) for item in _list(data.get("required_evidence"))],
        metadata=metadata,
    )


def _evidence_from_mapping(raw: Any) -> RuntimeEvidenceRecord:
    data = _mapping(raw)
    return RuntimeEvidenceRecord(
        id=_optional_string(data.get("id")),
        kind=str(data.get("kind") or ""),
        owner=_optional_string(data.get("owner")),
        operation_id=_optional_string(data.get("operation_id")),
        task_id=_optional_string(data.get("task_id")),
        accepted=bool(data.get("accepted", True)),
        payload=_mapping(data.get("payload")),
        metadata=_mapping(data.get("metadata")),
    )


def _governance_from_mapping(raw: Any) -> RuntimeGovernanceRecord | None:
    if raw is None:
        return None
    data = _mapping(raw)
    return RuntimeGovernanceRecord(
        allowed=_optional_bool(data.get("allowed")),
        blocked=_optional_bool(data.get("blocked")),
        pending_approval=_optional_bool(data.get("pending_approval")),
        decisions=[_mapping(item) for item in _list(data.get("decisions"))],
        approval_requests=[
            _mapping(item) for item in _list(data.get("approval_requests"))
        ],
        metadata=_mapping(data.get("metadata")),
    )


def _approval_records(
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    governance: RuntimeGovernanceRecord | None,
) -> list[dict[str, Any]]:
    approvals = [_mapping(item) for item in _list(result.get("approval_requests"))]
    if approvals:
        return approvals
    snapshot_approvals = [
        _mapping(item) for item in _list(diagnostics.get("approvals"))
    ]
    if snapshot_approvals:
        return snapshot_approvals
    return list(governance.approval_requests) if governance else []


def _snapshot_answer(result: Mapping[str, Any]) -> str | None:
    operation = _mapping(result.get("operation"))
    metadata = _mapping(operation.get("metadata"))
    answer = metadata.get("answer") or metadata.get("result")
    return str(answer) if answer is not None else None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(child) for child in value]
    enum_value = _enum_value(value)
    if enum_value is not value:
        return enum_value
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_bool(value: Any) -> bool | None:
    return bool(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
