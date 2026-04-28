"""
Grok (xAI) LLM provider implementation with integrated tracing.
"""

import os
import logging
from typing import Dict, Any, Optional, List

from ..core.exceptions import LLMError
from .base import BaseLLMProvider
from .openai_compatible import OpenAICompatibleMixin, compact_params

logger = logging.getLogger(__name__)


class GrokProvider(OpenAICompatibleMixin, BaseLLMProvider):
    """Grok (xAI) LLM provider implementation with automatic call tracing."""

    def __init__(
        self, model: str = "grok-4.20", api_key: Optional[str] = None, **kwargs
    ):
        """
        Initialize Grok provider.

        Args:
            model: Grok model name (e.g., "grok-4.20", "grok-4")
            api_key: xAI API key
            **kwargs: Additional Grok-specific parameters
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")

        super().__init__(model=model, api_key=api_key, **kwargs)

        # Grok-specific default parameters
        self.default_params.update(
            {
                "stream": kwargs.get("stream", False),
                "timeout": kwargs.get("timeout", 60),
            }
        )

        # Base URL for xAI API
        self.base_url = kwargs.get("base_url", "https://api.x.ai/v1")

        # Lazy-load OpenAI client (Grok uses OpenAI-compatible API)
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client configured for xAI."""
        if self._client is None:
            try:
                import openai

                self._validate_api_key()
                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key, base_url=self.base_url
                )
                logger.debug("Grok client initialized")
            except ImportError:
                raise LLMError(
                    "openai package not found. It is a core dependency — reinstall with: pip install daita-agents"
                )
        return self._client

    async def _generate_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs,
    ):
        """
        Grok non-streaming with optional tools.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in OpenAI format (or None)
            **kwargs: Optional parameters

        Returns:
            - If no tools or LLM returns text: str
            - If LLM wants to call tools: {"tool_calls": [...]}
        """
        try:
            # Build API call params
            api_params = compact_params(
                {
                    "model": self.model,
                    "messages": self._convert_messages_to_openai(messages),
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                    "top_p": kwargs.get("top_p"),
                    "timeout": kwargs.get("timeout"),
                }
            )

            # Add tools if provided
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            # Make API call
            response = await self.client.chat.completions.create(**api_params)

            self._record_usage(response.usage)

            message = response.choices[0].message

            # Check if tool calls
            tool_calls = self._tool_calls_from_openai_message(message)
            if tool_calls:
                return {"tool_calls": tool_calls}
            return message.content

        except Exception as e:
            logger.error(f"Grok generation failed: {str(e)}")
            raise LLMError(f"Grok generation failed: {str(e)}")

    async def _stream_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs,
    ):
        """
        Grok streaming with optional tools.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in OpenAI format (or None)
            **kwargs: Optional parameters

        Yields:
            LLMChunk objects with type "text" or "tool_call_complete"
        """
        from ..core.streaming import LLMChunk

        try:
            # Build API call params
            api_params = compact_params(
                {
                    "model": self.model,
                    "messages": self._convert_messages_to_openai(messages),
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                    "top_p": kwargs.get("top_p"),
                    "timeout": kwargs.get("timeout"),
                    "stream": True,
                }
            )

            # Add tools if provided
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            # Stream response
            stream = await self.client.chat.completions.create(**api_params)

            # Buffer for accumulating partial tool calls
            tool_call_buffers = {}

            async for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta

                # Stream text content
                if delta.content:
                    yield LLMChunk(type="text", content=delta.content, model=self.model)

                # Handle tool calls (streamed as deltas)
                self._apply_openai_tool_call_deltas(tool_call_buffers, delta.tool_calls)

                # On stream end, emit complete tool calls
                if choice.finish_reason == "tool_calls":
                    for tool_call in tool_call_buffers.values():
                        yield LLMChunk(
                            type="tool_call_complete",
                            tool_name=tool_call["name"],
                            tool_args=self._safe_parse_tool_arguments(
                                tool_call["arguments"]
                            ),
                            tool_call_id=tool_call["id"],
                            model=self.model,
                        )

                # Store usage if available
                if hasattr(chunk, "usage") and chunk.usage:
                    self._record_usage(chunk.usage)

        except Exception as e:
            logger.error(f"Grok streaming failed: {str(e)}")
            raise LLMError(f"Grok streaming failed: {str(e)}")

    @property
    def info(self) -> Dict[str, Any]:
        """Get information about the Grok provider."""
        base_info = super().info
        base_info.update(
            {
                "base_url": self.base_url,
                "provider_name": "Grok (xAI)",
                "api_compatible": "OpenAI",
            }
        )
        return base_info
