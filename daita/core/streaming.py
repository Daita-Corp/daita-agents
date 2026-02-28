"""
Streaming event system for real-time agent execution updates.

This module provides event types and data structures for streaming
agent execution progress to UIs, APIs, or logging systems.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Dict


class EventType(Enum):
    """Types of events emitted during agent execution."""
    THINKING = "thinking"           # LLM reasoning/text output
    TOOL_CALL = "tool_call"         # Tool about to be executed
    TOOL_RESULT = "tool_result"     # Tool execution completed
    ITERATION = "iteration"         # New iteration starting
    COMPLETE = "complete"           # Execution finished
    ERROR = "error"                 # Error occurred


@dataclass
class AgentEvent:
    """
    Event emitted during streaming agent execution.

    Different event types populate different fields:
    - THINKING: content
    - TOOL_CALL: tool_name, tool_args
    - TOOL_RESULT: tool_name, result
    - ITERATION: iteration, max_iterations
    - COMPLETE: final_result, iterations, token_usage, cost
    - ERROR: error

    Example:
        event = AgentEvent(
            type=EventType.THINKING,
            content="Analyzing the data..."
        )
    """
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)

    # THINKING events
    content: Optional[str] = None

    # TOOL_CALL and TOOL_RESULT events
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None

    # ITERATION events
    iteration: Optional[int] = None
    max_iterations: Optional[int] = None

    # COMPLETE events
    final_result: Optional[str] = None
    iterations: Optional[int] = None
    token_usage: Optional[Dict[str, int]] = None
    cost: Optional[float] = None

    # ERROR events
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary for serialization.

        Useful for sending over WebSocket or storing in database.
        """
        data = {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat()
        }

        # Add populated fields
        if self.content is not None:
            data["content"] = self.content
        if self.tool_name is not None:
            data["tool_name"] = self.tool_name
        if self.tool_args is not None:
            data["tool_args"] = self.tool_args
        if self.result is not None:
            data["result"] = str(self.result)[:500]  # Truncate large results
        if self.iteration is not None:
            data["iteration"] = self.iteration
        if self.max_iterations is not None:
            data["max_iterations"] = self.max_iterations
        if self.final_result is not None:
            data["final_result"] = self.final_result
        if self.iterations is not None:
            data["iterations"] = self.iterations
        if self.token_usage is not None:
            data["token_usage"] = self.token_usage
        if self.cost is not None:
            data["cost"] = self.cost
        if self.error is not None:
            data["error"] = self.error

        return data


@dataclass
class LLMChunk:
    """
    Chunk from streaming LLM response.

    LLM providers emit these chunks which get converted to AgentEvents.
    """
    type: str  # "text" or "tool_call_complete"

    # Text chunks
    content: Optional[str] = None

    # Tool call chunks (complete only, no partial)
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None

    # Metadata
    model: Optional[str] = None
    finish_reason: Optional[str] = None
