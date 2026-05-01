"""
Anthropic LLM provider implementation with integrated tracing.
"""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional

from ..core.exceptions import LLMError
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation with automatic call tracing."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Anthropic model name
            api_key: Anthropic API key
            **kwargs: Additional Anthropic-specific parameters
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        super().__init__(model=model, api_key=api_key, **kwargs)

        # Anthropic-specific default parameters
        self.default_params.update({"timeout": kwargs.get("timeout", 60)})

        # Lazy-load Anthropic client
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                from anthropic import Timeout

                self._validate_api_key()

                # Get timeout from default params (set during __init__)
                timeout_seconds = self.default_params.get("timeout", 60)

                # Create client with extended timeout
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.api_key,
                    timeout=Timeout(
                        timeout_seconds, read=timeout_seconds, write=10.0, connect=5.0
                    ),
                )
                logger.debug(
                    f"Anthropic client initialized with {timeout_seconds}s timeout"
                )
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install 'daita-agents[anthropic]'"
                ) from None
        return self._client

    def _build_api_params(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build API parameters for Anthropic API call.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in Anthropic format (already converted by base class)
            **kwargs: Optional parameters

        Returns:
            API parameters dict ready for Anthropic API
        """
        # Extract system message (Anthropic expects it as separate parameter)
        system_message = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        api_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature"),
            "messages": self._convert_messages_to_anthropic(filtered_messages),
            "timeout": kwargs.get("timeout"),
        }

        # Add system message if present
        if system_message:
            api_params["system"] = system_message

        # Add tools if provided (already in Anthropic format from base class)
        if tools:
            api_params["tools"] = tools

        return api_params

    async def _generate_impl(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs,
    ):
        """
        Anthropic non-streaming with optional tools.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in Anthropic format (already converted by base class)
            **kwargs: Optional parameters

        Returns:
            - If no tools or LLM returns text: str
            - If LLM wants to call tools: {"tool_calls": [...]}
        """
        try:
            # Build API parameters
            api_params = self._build_api_params(messages, tools, **kwargs)

            # Make API call
            response = await self.client.messages.create(**api_params)

            self._record_usage(response.usage)

            # Check response content
            # First pass: collect all tool_use blocks (takes priority over text)
            tool_calls = [
                {"id": b.id, "name": b.name, "arguments": b.input}
                for b in response.content
                if b.type == "tool_use"
            ]

            # If tools were called, return them (takes priority)
            if tool_calls:
                return {"tool_calls": tool_calls}

            # Otherwise, return text content
            for block in response.content:
                if block.type == "text":
                    return block.text

            return ""

        except Exception as e:
            logger.error(f"Anthropic generation failed: {str(e)}")
            raise LLMError(f"Anthropic generation failed: {str(e)}")

    async def _stream_impl(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs,
    ):
        """
        Anthropic streaming with optional tools.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in Anthropic format (already converted by base class)
            **kwargs: Optional parameters

        Yields:
            LLMChunk objects with type "text" or "tool_call_complete"
        """
        from ..core.streaming import LLMChunk

        try:
            # Build API parameters
            api_params = self._build_api_params(messages, tools, **kwargs)

            # Stream with context manager
            async with self.client.messages.stream(**api_params) as stream:
                current_tool_use = None

                async for event in stream:
                    # Text content deltas
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield LLMChunk(
                                type="text", content=event.delta.text, model=self.model
                            )
                        # Accumulate tool input deltas
                        elif hasattr(event.delta, "partial_json") and current_tool_use:
                            # Anthropic streams tool arguments as partial JSON
                            if "partial_json" not in current_tool_use:
                                current_tool_use["partial_json"] = ""
                            current_tool_use["partial_json"] += event.delta.partial_json

                    # Tool use start
                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool_use = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": {},  # Will be populated from stream or final message
                                "partial_json": "",
                            }

                    # Tool use complete
                    elif event.type == "content_block_stop":
                        if current_tool_use:
                            # Parse accumulated JSON or use the input
                            import json

                            if current_tool_use.get("partial_json"):
                                try:
                                    tool_args = json.loads(
                                        current_tool_use["partial_json"]
                                    )
                                except json.JSONDecodeError:
                                    tool_args = current_tool_use.get("input", {})
                            else:
                                tool_args = current_tool_use.get("input", {})

                            yield LLMChunk(
                                type="tool_call_complete",
                                tool_name=current_tool_use["name"],
                                tool_args=tool_args,
                                tool_call_id=current_tool_use["id"],
                                model=self.model,
                            )
                            current_tool_use = None

                    # Message stop - get usage
                    elif event.type == "message_stop":
                        final_message = await stream.get_final_message()
                        if hasattr(final_message, "usage"):
                            self._record_usage(final_message.usage)

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {str(e)}")
            raise LLMError(f"Anthropic streaming failed: {str(e)}")

    async def generate_with_system(
        self, prompt: str, system_message: str, **kwargs
    ) -> str:
        """
        Generate text with a system message using Anthropic's system parameter.

        Note: This method bypasses automatic tracing since it's not part of the
        base interface. If you want tracing for system messages, call the base
        generate() method with a formatted prompt instead.

        Args:
            prompt: User prompt
            system_message: System message to set context
            **kwargs: Optional parameters

        Returns:
            Generated text
        """
        try:
            # Merge parameters
            params = self._merge_params(kwargs)

            # Make API call with system parameter
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
                system=system_message,
                messages=[{"role": "user", "content": prompt}],
                timeout=params.get("timeout"),
            )

            self._record_usage(response.usage)

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation with system message failed: {str(e)}")
            raise LLMError(f"Anthropic generation failed: {str(e)}")

    def _convert_tools_to_format(
        self, tools: list["AgentTool"]
    ) -> list[Dict[str, Any]]:
        """
        Convert AgentTool list to Anthropic tool format.

        Anthropic uses a different tool format than OpenAI.
        """
        return [tool.to_anthropic_tool() for tool in tools]

    def _convert_messages_to_anthropic(
        self, messages: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Anthropic format.

        Anthropic uses a different message format, especially for tool results.
        """
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "tool":
                # Tool result - convert to Anthropic format
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg["tool_call_id"],
                                "content": msg["content"],
                            }
                        ],
                    }
                )
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                # Assistant with tool calls (already in flat format)
                content_blocks = []
                for tc in msg["tool_calls"]:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments"],
                        }
                    )
                anthropic_messages.append(
                    {"role": "assistant", "content": content_blocks}
                )
            else:
                # Regular message
                anthropic_messages.append(msg)

        return anthropic_messages

    @property
    def info(self) -> Dict[str, Any]:
        """Get information about the Anthropic provider."""
        base_info = super().info
        base_info.update({"provider_name": "Anthropic", "api_compatible": "Anthropic"})
        return base_info
