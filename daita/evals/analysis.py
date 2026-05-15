"""Evidence extraction and operation inspection for evals."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

from .models import ExecutionSpan, RunMetrics, StabilitySummary, ToolCallEvidence

WRITE_WORDS = {
    "write",
    "create",
    "insert",
    "update",
    "upsert",
    "delete",
    "remove",
    "execute",
    "post",
    "put",
    "patch",
}
DELETE_WORDS = {"delete", "remove", "drop", "truncate"}
READ_WORDS = {"read", "get", "list", "query", "select", "inspect", "count", "sample"}


@dataclass
class DataOperation:
    """A normalized data operation inferred from a tool call."""

    category: str
    action: str
    tool_name: str
    resource: str | None = None
    method: str | None = None
    host: str | None = None
    bucket: str | None = None
    path: str | None = None
    table: str | None = None
    sql: str | None = None
    top_k: int | None = None
    filters: list[str] = field(default_factory=list)
    call_index: int = 0


@dataclass
class RunEvidence:
    """Normalized evidence for one agent run."""

    answer: str
    prompt_hash: str
    answer_hash: str
    tool_calls: list[ToolCallEvidence]
    execution_spans: list[ExecutionSpan]
    operations: list[DataOperation]
    metrics: RunMetrics
    trace_id: str | None = None


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def preview_text(value: Any, max_chars: int = 240) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def normalize_tool_calls(raw_calls: Iterable[Any]) -> list[ToolCallEvidence]:
    calls: list[ToolCallEvidence] = []
    for raw in raw_calls or []:
        if not isinstance(raw, dict):
            continue
        name = raw.get("tool") or raw.get("name") or raw.get("tool_name") or "unknown"
        arguments = raw.get("arguments") or raw.get("input") or raw.get("args") or {}
        result = raw.get("result", raw.get("output"))
        calls.append(
            ToolCallEvidence(
                name=str(name),
                arguments=arguments if isinstance(arguments, dict) else {},
                result=result,
            )
        )
    return calls


def summarize_run_metrics(raw_result: dict[str, Any]) -> RunMetrics:
    deltas = raw_result.get("_eval_metric_delta") or {}
    tokens = raw_result.get("tokens") or {}
    return RunMetrics(
        latency_ms=raw_result.get("processing_time_ms")
        or raw_result.get("latency_ms")
        or raw_result.get("duration_ms"),
        tokens_total=deltas.get("tokens_total")
        or tokens.get("total_tokens")
        or raw_result.get("tokens_total"),
        cost=deltas.get("cost") if "cost" in deltas else raw_result.get("cost"),
        iterations=raw_result.get("iterations"),
    )


def extract_run_evidence(prompt: str, raw_result: Any) -> RunEvidence:
    if isinstance(raw_result, str):
        result_dict: dict[str, Any] = {"result": raw_result}
    elif isinstance(raw_result, dict):
        result_dict = raw_result
    else:
        result_dict = {"result": str(raw_result)}

    answer = str(result_dict.get("result") or result_dict.get("answer") or "")
    tool_calls = normalize_tool_calls(result_dict.get("tool_calls", []))
    execution_spans = normalize_execution_spans(result_dict)
    operations = inspect_data_operations(tool_calls)
    return RunEvidence(
        answer=answer,
        prompt_hash=stable_hash(prompt),
        answer_hash=stable_hash(answer),
        tool_calls=tool_calls,
        execution_spans=execution_spans,
        operations=operations,
        metrics=summarize_run_metrics(result_dict),
        trace_id=result_dict.get("_daita_trace_id") or result_dict.get("trace_id"),
    )


def extract_tool_sequence(tool_calls: Iterable[ToolCallEvidence]) -> tuple[str, ...]:
    return tuple(call.name for call in tool_calls)


def extract_sql_statements(tool_calls: Iterable[ToolCallEvidence]) -> list[str]:
    statements = []
    for call in tool_calls:
        sql = call.arguments.get("sql") or call.arguments.get("query")
        if isinstance(sql, str):
            statements.append(sql)
    return statements


def inspect_data_operations(
    tool_calls: Iterable[ToolCallEvidence],
) -> list[DataOperation]:
    return [_inspect_tool_call(call, index) for index, call in enumerate(tool_calls)]


def normalize_execution_spans(raw_result: dict[str, Any]) -> list[ExecutionSpan]:
    spans: list[ExecutionSpan] = []
    for raw in raw_result.get("spans") or raw_result.get("trace") or []:
        span = _span_from_mapping(raw)
        if span is not None:
            spans.append(span)
    for raw in raw_result.get("skill_calls") or []:
        span = _span_from_mapping(raw, default_kind="skill")
        if span is not None:
            spans.append(span)
    for raw in raw_result.get("plugin_calls") or []:
        span = _span_from_mapping(raw, default_kind="plugin")
        if span is not None:
            spans.append(span)
    spans.extend(
        _spans_from_tool_metadata(
            raw_result.get("tool_calls") or [],
            include_skills=not any(span.kind == "skill" for span in spans),
            include_plugins=not any(span.kind == "plugin" for span in spans),
        )
    )
    return spans


def summarize_stability(runs: list[RunEvidence]) -> StabilitySummary:
    costs = [r.metrics.cost for r in runs if r.metrics.cost is not None]
    latencies = [r.metrics.latency_ms for r in runs if r.metrics.latency_ms is not None]
    tokens = [
        r.metrics.tokens_total for r in runs if r.metrics.tokens_total is not None
    ]
    return StabilitySummary(
        answer_variants=len({r.answer_hash for r in runs}),
        tool_sequence_variants=len({extract_tool_sequence(r.tool_calls) for r in runs}),
        cost_min=min(costs) if costs else None,
        cost_max=max(costs) if costs else None,
        latency_ms_min=min(latencies) if latencies else None,
        latency_ms_max=max(latencies) if latencies else None,
        token_min=min(tokens) if tokens else None,
        token_max=max(tokens) if tokens else None,
    )


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


def _inspect_tool_call(call: ToolCallEvidence, index: int) -> DataOperation:
    name = call.name.lower()
    args = call.arguments
    action = _infer_action(name, args)
    category = _infer_category(name, args)
    resource = _first_string(
        args,
        "resource",
        "path",
        "file",
        "file_path",
        "filename",
        "url",
        "uri",
        "bucket",
        "key",
        "table",
        "collection",
        "index",
    )
    url = _first_string(args, "url", "uri", "endpoint")
    parsed = urlparse(url) if url else None
    sql = args.get("sql") or args.get("query")

    bucket = _first_string(args, "bucket", "bucket_name")
    path = _first_string(args, "path", "file", "file_path", "filename", "key")
    table = _first_string(args, "table", "table_name", "collection", "index")

    if not bucket and path and path.startswith("s3://"):
        parsed_s3 = urlparse(path)
        bucket = parsed_s3.netloc

    return DataOperation(
        category=category,
        action=action,
        tool_name=call.name,
        resource=resource,
        method=_first_string(args, "method", "http_method"),
        host=parsed.netloc if parsed else _first_string(args, "host"),
        bucket=bucket,
        path=path,
        table=table,
        sql=sql if isinstance(sql, str) else None,
        top_k=_first_int(args, "top_k", "k", "limit"),
        filters=_extract_filters(args),
        call_index=index,
    )


def _span_from_mapping(
    raw: Any, *, default_kind: str | None = None
) -> ExecutionSpan | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or default_kind or "").lower()
    if kind not in {"skill", "plugin", "tool", "workflow"}:
        return None
    name = raw.get("name") or raw.get(kind) or raw.get(f"{kind}_name")
    if not name:
        return None
    return ExecutionSpan(
        kind=kind,
        name=str(name),
        operation=_optional_string(raw.get("operation") or raw.get("action")),
        status=str(raw.get("status") or "passed"),
        latency_ms=_optional_float(
            raw.get("latency_ms") or raw.get("duration_ms") or raw.get("elapsed_ms")
        ),
        error=_optional_string(raw.get("error")),
        parent_id=_optional_string(raw.get("parent_id")),
        trace_id=_optional_string(raw.get("trace_id")),
    )


def _spans_from_tool_metadata(
    raw_calls: Iterable[Any],
    *,
    include_skills: bool,
    include_plugins: bool,
) -> list[ExecutionSpan]:
    spans: list[ExecutionSpan] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        tool_name = raw.get("tool") or raw.get("name") or raw.get("tool_name")
        operation = raw.get("operation") or raw.get("action")
        if tool_name:
            spans.append(
                ExecutionSpan(
                    kind="tool",
                    name=str(tool_name),
                    operation=_optional_string(operation),
                    status=str(raw.get("status") or "passed"),
                    latency_ms=_optional_float(
                        raw.get("latency_ms") or raw.get("duration_ms")
                    ),
                    error=_optional_string(raw.get("error")),
                    parent_id=_optional_string(raw.get("parent_id")),
                    trace_id=_optional_string(raw.get("trace_id")),
                )
            )
        skill = raw.get("skill") or raw.get("skill_name")
        if include_skills and skill:
            spans.append(
                ExecutionSpan(
                    kind="skill",
                    name=str(skill),
                    operation=_optional_string(operation),
                    status=str(raw.get("status") or "passed"),
                    latency_ms=_optional_float(
                        raw.get("skill_latency_ms") or raw.get("latency_ms")
                    ),
                    error=_optional_string(raw.get("error")),
                    trace_id=_optional_string(raw.get("trace_id")),
                )
            )
        plugin = raw.get("plugin") or raw.get("plugin_name")
        if include_plugins and plugin:
            spans.append(
                ExecutionSpan(
                    kind="plugin",
                    name=str(plugin),
                    operation=_optional_string(operation),
                    status=str(raw.get("status") or "passed"),
                    latency_ms=_optional_float(
                        raw.get("plugin_latency_ms") or raw.get("latency_ms")
                    ),
                    error=_optional_string(raw.get("error")),
                    trace_id=_optional_string(raw.get("trace_id")),
                )
            )
    return spans


def _infer_action(tool_name: str, args: dict[str, Any]) -> str:
    method = str(args.get("method") or args.get("http_method") or "").lower()
    if method in {"post", "put", "patch"}:
        return "write"
    if method == "delete":
        return "delete"
    tokens = set(re.split(r"[_\W]+", tool_name))
    if tokens & DELETE_WORDS:
        return "delete"
    if tokens & (WRITE_WORDS - DELETE_WORDS):
        return "write"
    sql = str(args.get("sql") or args.get("query") or "").lstrip().lower()
    if re.match(r"^(delete|drop|truncate)\b", sql):
        return "delete"
    if re.match(r"^(insert|update|merge|create|alter)\b", sql):
        return "write"
    if tokens & READ_WORDS:
        return "read"
    return "unknown"


def _infer_category(tool_name: str, args: dict[str, Any]) -> str:
    text = " ".join([tool_name, *[str(v) for v in args.values() if isinstance(v, str)]])
    low = text.lower()
    if "sql" in args or re.search(r"\b(select|insert|update|delete)\b", low):
        return "sql"
    if any(word in low for word in ["vector", "embedding", "similarity", "top_k"]):
        return "vector"
    if any(word in low for word in ["s3", "bucket", "blob", "gcs", "storage"]):
        return "storage"
    if any(word in low for word in ["http", "api", "url", "endpoint"]):
        return "api"
    if any(word in low for word in ["file", "csv", "xlsx", "sheet", "path"]):
        return "file"
    if any(word in low for word in ["workflow", "relay", "channel"]):
        return "workflow"
    return "tool"


def _first_string(args: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _first_int(args: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = args.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _extract_filters(args: dict[str, Any]) -> list[str]:
    filters = args.get("filters") or args.get("filter") or args.get("where")
    if isinstance(filters, str):
        return [filters]
    if isinstance(filters, dict):
        return [str(k) for k in filters]
    if isinstance(filters, list):
        return [str(item) for item in filters]
    return []


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
