"""Helpers for executing one LLM turn inside the Agent runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Unified LLM response format for both streaming and non-streaming turns."""

    text: str
    tool_calls: List[Dict[str, Any]]
    finish_reason: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_response(cls, response: Any) -> "LLMResult":
        """Create an LLMResult from a provider response."""
        if isinstance(response, str):
            return cls(text=response, tool_calls=[])
        if isinstance(response, dict):
            return cls(
                text=response.get("content", ""),
                tool_calls=response.get("tool_calls", []),
                finish_reason=response.get("finish_reason"),
                provider=response.get("provider"),
                model=response.get("model"),
                raw_metadata=response.get("metadata", {}) or {},
                warnings=list(response.get("warnings", []) or []),
            )
        logger.warning("Unexpected response type: %s", type(response))
        return cls(text=str(response), tool_calls=[])


async def stream_llm_turn(agent, conversation, tools, on_event, **kwargs) -> LLMResult:
    """Execute a streaming LLM turn and emit THINKING / TOOL_CALL events."""
    from ...core.streaming import EventType

    thinking_text = ""
    tool_calls = []
    finish_reason = None
    raw_metadata: Dict[str, Any] = {}
    emitted_event = False

    try:
        async for chunk in await agent.llm.generate(
            messages=conversation, tools=tools, stream=True, **kwargs
        ):
            if chunk.type == "text":
                thinking_text += chunk.content
                agent._emit_event(on_event, EventType.THINKING, content=chunk.content)
                emitted_event = True

            elif chunk.type == "tool_call_complete":
                tool_calls.append(
                    {
                        "id": chunk.tool_call_id,
                        "name": chunk.tool_name,
                        "arguments": chunk.tool_args,
                    }
                )
                agent._emit_event(
                    on_event,
                    EventType.TOOL_CALL,
                    tool_name=chunk.tool_name,
                    tool_args=chunk.tool_args,
                )
                emitted_event = True

            metadata = getattr(chunk, "metadata", None)
            if isinstance(metadata, dict):
                raw_metadata.update(metadata)
                finish_reason = metadata.get("finish_reason") or finish_reason
    except Exception as error:
        if emitted_event:
            setattr(error, "_daita_stream_event_emitted", True)
        raise

    return LLMResult(
        text=thinking_text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        raw_metadata=raw_metadata,
    )


async def nonstream_llm_turn(agent, conversation, tools, **kwargs) -> LLMResult:
    """Execute a non-streaming LLM turn."""
    return LLMResult.from_response(
        await agent.llm.generate(
            messages=conversation, tools=tools, stream=False, **kwargs
        )
    )
