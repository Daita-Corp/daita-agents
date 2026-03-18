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
            agent._db_audit_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "tool_calls": result.get("tool_calls", []),
            })
            return result if detailed else result["result"]
        except Exception as exc:
            agent._db_audit_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "tool_calls": list(agent._tool_call_history),
                "error": str(exc),
            })
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
            if event.type == EventType.TOOL_RESULT:
                tool_calls_this_run.append({
                    "tool": event.tool_name,
                    "result": event.result,
                })
            elif event.type == EventType.COMPLETE:
                agent._db_audit_log.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt": prompt,
                    "tool_calls": tool_calls_this_run,
                })
            elif event.type == EventType.ERROR:
                agent._db_audit_log.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt": prompt,
                    "tool_calls": tool_calls_this_run,
                    "error": event.error,
                })
            yield event

    return _audited_stream
