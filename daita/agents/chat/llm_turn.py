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
