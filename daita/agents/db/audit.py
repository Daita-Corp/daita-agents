"""
Audit log wrappers for agent.run() and agent.stream().

Wraps the original methods to record every invocation in agent._db_audit_log.
Each entry: {timestamp, prompt, tool_calls: [{tool, arguments, result}]}
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent


def make_audited_run(agent: "Agent", original_run: Callable) -> Callable:
    """Return an async wrapper around *original_run* that appends to the audit log."""

    async def _audited_run(prompt: str, *, detailed: bool = False, **kwargs: Any):
        if "history" not in kwargs and hasattr(agent, "_db_history"):
            kwargs["history"] = agent._db_history
        try:
            result = await original_run(prompt, detailed=True, **kwargs)
            agent._db_audit_log.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt": prompt,
                    "tool_calls": _sanitize_tool_calls(result.get("tool_calls", [])),
                }
            )
            return result if detailed else result["result"]
        except Exception as exc:
            agent._db_audit_log.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt": prompt,
                    "tool_calls": _sanitize_tool_calls(list(agent._tool_call_history)),
                    "error": str(exc),
                }
            )
            raise

    return _audited_run


def make_audited_stream(agent: "Agent", original_stream: Callable) -> Callable:
    """Return an async generator wrapper around *original_stream* that appends to the audit log."""

    async def _audited_stream(prompt: str, **kwargs: Any):
        if "history" not in kwargs and hasattr(agent, "_db_history"):
            kwargs["history"] = agent._db_history
        from ...core.streaming import EventType

        tool_calls_this_run: List[Dict[str, Any]] = []

        async for event in original_stream(prompt, **kwargs):
            if event.type == EventType.TOOL_CALL:
                tool_calls_this_run.append(
                    {
                        "tool": event.tool_name,
                        "arguments": event.tool_args or {},
                    }
                )
            elif event.type == EventType.TOOL_RESULT:
                pending = next(
                    (
                        call
                        for call in tool_calls_this_run
                        if call.get("tool") == event.tool_name and "result" not in call
                    ),
                    None,
                )
                if pending is None:
                    pending = {"tool": event.tool_name, "arguments": {}}
                    tool_calls_this_run.append(pending)
                pending["result"] = event.result
            elif event.type == EventType.COMPLETE:
                agent._db_audit_log.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "prompt": prompt,
                        "tool_calls": _sanitize_tool_calls(tool_calls_this_run),
                    }
                )
            elif event.type == EventType.ERROR:
                agent._db_audit_log.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "prompt": prompt,
                        "tool_calls": _sanitize_tool_calls(tool_calls_this_run),
                        "error": event.error,
                    }
                )
            yield event

    return _audited_stream


def _sanitize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep DB audit useful without retaining raw row/result payloads."""
    return [_sanitize_tool_call(call) for call in tool_calls]


def _sanitize_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    if "tool" in call:
        sanitized["tool"] = call["tool"]
    if "arguments" in call:
        sanitized["arguments"] = _sanitize_arguments(call.get("arguments"))
    if "result" in call:
        sanitized["result"] = _sanitize_result(call.get("result"))
    if "error" in call:
        sanitized["error"] = str(call["error"])
    return sanitized


def _sanitize_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return _safe_scalar(arguments)

    allowed_keys = {
        "sql",
        "table",
        "column",
        "columns",
        "method",
        "sample_size",
        "n",
        "store",
    }
    sanitized = {
        key: _safe_scalar(value)
        for key, value in arguments.items()
        if key in allowed_keys
    }
    if "params" in arguments:
        params = arguments.get("params") or []
        sanitized["param_count"] = len(params) if isinstance(params, list) else 1
    return sanitized


def _sanitize_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return _safe_scalar(result)

    sanitized: Dict[str, Any] = {}
    for key in (
        "success",
        "sql",
        "total_rows",
        "truncated",
        "affected_rows",
        "row_count",
        "columns_profiled",
        "anomaly_count",
        "total_rows_scanned",
        "is_fresh",
        "age_hours",
        "expected_interval_hours",
        "completeness_score",
        "graph_persisted",
        "metric_node_id",
        "error",
        "note",
    ):
        if key in result:
            sanitized[key] = _safe_scalar(result[key])

    if "rows" in result:
        rows = result.get("rows") or []
        sanitized["row_count"] = len(rows) if isinstance(rows, list) else None
    if "profile" in result and "columns_profiled" not in sanitized:
        profile = result.get("profile")
        if isinstance(profile, dict):
            sanitized["columns_profiled"] = len(profile)
    if "anomaly_values" in result:
        values = result.get("anomaly_values")
        sanitized["anomaly_value_count"] = (
            len(values) if isinstance(values, list) else 0
        )
    if not sanitized:
        sanitized["result_type"] = type(result).__name__
    return sanitized


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        if all(
            item is None or isinstance(item, (str, int, float, bool)) for item in value
        ):
            return list(value)
        return {"type": type(value).__name__, "count": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in value.keys())}
    return str(value)
