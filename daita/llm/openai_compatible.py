"""
Shared helpers for providers that expose an OpenAI-compatible chat API.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compact_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop parameters left unset so provider SDKs don't receive explicit nulls."""
    return {key: value for key, value in params.items() if value is not None}


class OpenAICompatibleMixin:
    """Message, usage, and tool-call helpers for OpenAI-compatible providers."""

    def _convert_messages_to_openai(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert Daita's flat tool-call format to OpenAI's nested format.
        """
        converted = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                converted_tool_calls = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
                converted.append(
                    {"role": "assistant", "tool_calls": converted_tool_calls}
                )
            else:
                converted.append(msg)
        return converted

    def _safe_parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        """Parse model-emitted tool arguments into a dict."""
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool arguments: %s", arguments)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _tool_calls_from_openai_message(self, message) -> list[dict[str, Any]]:
        """Extract Daita-style tool calls from an OpenAI-compatible message."""
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return []
        return [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": self._safe_parse_tool_arguments(tc.function.arguments),
            }
            for tc in tool_calls
        ]

    def _apply_openai_tool_call_deltas(
        self, buffers: dict[int, dict[str, str]], tool_call_deltas
    ) -> None:
        """Accumulate streamed OpenAI-compatible tool-call deltas."""
        if not tool_call_deltas:
            return

        for tc_delta in tool_call_deltas:
            index = tc_delta.index
            buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})

            if tc_delta.id:
                buffers[index]["id"] = tc_delta.id
            if tc_delta.function and tc_delta.function.name:
                buffers[index]["name"] = tc_delta.function.name
            if tc_delta.function and tc_delta.function.arguments:
                buffers[index]["arguments"] += tc_delta.function.arguments
