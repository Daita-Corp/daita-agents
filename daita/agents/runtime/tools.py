"""Tool execution helpers for the Agent runtime."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from ...core.tracing import TraceType

if TYPE_CHECKING:
    from ...core.tools import AgentTool


async def execute_tool_call(tool_call: Dict[str, Any], tools: List["AgentTool"]) -> Any:
    """Execute a single tool call with timeout and error handling."""
    tool_name = tool_call["name"]
    arguments = tool_call["arguments"]

    tool = next((t for t in tools if t.name == tool_name), None)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}

    try:
        return await asyncio.wait_for(
            tool.handler(arguments), timeout=tool.timeout_seconds
        )
    except asyncio.TimeoutError:
        return {"error": f"Tool '{tool_name}' timed out after {tool.timeout_seconds}s"}
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {str(e)}"}


async def execute_and_track_tool(
    agent,
    tool_call: Dict[str, Any],
    tools: List["AgentTool"],
    on_event: Optional[Any],
) -> Dict[str, Any]:
    """Execute a tool, trace it, record agent history, and emit a result event."""
    from ...core.streaming import EventType

    tool_name = tool_call["name"]
    tool = next((t for t in tools if t.name == tool_name), None)

    async with agent.trace_manager.span(
        operation_name=f"tool_{tool_name}",
        trace_type=TraceType.TOOL_EXECUTION,
        agent_id=agent.agent_id,
        tool_name=tool_name,
        input_data=tool_call.get("arguments"),
    ) as span_id:
        start_time = time.time()
        result = await execute_tool_call(tool_call, tools)
        agent.trace_manager.record_output(span_id, result)
        duration_ms = int((time.time() - start_time) * 1000)

        tool_call_record = {
            "name": tool_name,
            "duration_ms": duration_ms,
            "input": tool_call.get("arguments"),
            "output": result,
        }
        agent._tool_call_history.append(tool_call_record)

        agent._emit_event(
            on_event, EventType.TOOL_RESULT, tool_name=tool_name, result=result
        )

        return {
            "tool": tool_name,
            "arguments": tool_call["arguments"],
            "result": result,
            "duration_ms": duration_ms,
            "retry_safe": bool(getattr(tool, "retry_safe", False)),
            "replay_safe": bool(getattr(tool, "replay_safe", False)),
            "idempotent": bool(getattr(tool, "idempotent", False)),
            "side_effecting": bool(getattr(tool, "side_effecting", True)),
        }


def append_tool_messages(
    agent, conversation: List[Dict], tool_calls: List[Dict], results: List[Any]
) -> None:
    """Append assistant tool-call and tool-result messages to the conversation."""
    conversation.append({"role": "assistant", "tool_calls": tool_calls})

    for tool_call, result in zip(tool_calls, results):
        content_result = result["result"]
        compactor = getattr(agent, "_compact_tool_result_for_context", None)
        if callable(compactor):
            content_result = compactor(tool_call["name"], content_result)
        conversation.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["name"],
                "content": json.dumps(content_result, default=json_serializer),
            }
        )


def json_serializer(obj):
    """JSON serializer for values commonly returned by plugins."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
