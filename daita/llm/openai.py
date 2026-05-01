"""
OpenAI LLM provider implementation with integrated tracing.
"""

import os
import logging
from typing import Dict, Any, Optional

from ..core.exceptions import LLMError
from .base import BaseLLMProvider
from .openai_compatible import OpenAICompatibleMixin, compact_params

logger = logging.getLogger(__name__)


def _build_token_param(
    max_tokens: Any = None,
    max_completion_tokens: Any = None,
    use_legacy_max_tokens: bool = False,
) -> Dict[str, Any]:
    """Build OpenAI's token-cap parameter while preserving Daita's max_tokens alias."""
    if max_completion_tokens is not None:
        return {"max_completion_tokens": max_completion_tokens}
    if max_tokens is None:
        return {}
    key = "max_tokens" if use_legacy_max_tokens else "max_completion_tokens"
    return {key: max_tokens}


class OpenAIProvider(OpenAICompatibleMixin, BaseLLMProvider):
    """OpenAI LLM provider implementation with automatic call tracing."""

    def __init__(
        self, model: str = "gpt-5.4-mini", api_key: Optional[str] = None, **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name (e.g., "gpt-5.4-mini", "gpt-5.4", "gpt-5.5")
            api_key: OpenAI API key
            **kwargs: Additional OpenAI-specific parameters
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        super().__init__(model=model, api_key=api_key, **kwargs)

        # OpenAI-specific default parameters
        self.default_params.update(
            {
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
                "reasoning_effort": kwargs.get("reasoning_effort"),
                "service_tier": kwargs.get("service_tier"),
                "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
                "max_completion_tokens": kwargs.get("max_completion_tokens"),
                "use_legacy_max_tokens": kwargs.get("use_legacy_max_tokens", False),
                "timeout": kwargs.get("timeout", 60),
            }
        )

        # Lazy-load OpenAI client
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._validate_api_key()
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
                logger.debug("OpenAI client initialized")
            except ImportError:
                raise LLMError(
                    "openai package not found. It is a core dependency — reinstall with: pip install daita-agents"
                )
        return self._client

    async def _generate_impl(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs,
    ):
        """
        OpenAI non-streaming with optional tools.

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
                    "temperature": kwargs.get("temperature"),
                    "top_p": kwargs.get("top_p"),
                    "frequency_penalty": kwargs.get("frequency_penalty"),
                    "presence_penalty": kwargs.get("presence_penalty"),
                    "reasoning_effort": kwargs.get("reasoning_effort"),
                    "service_tier": kwargs.get("service_tier"),
                    "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
                    "timeout": kwargs.get("timeout"),
                    **_build_token_param(
                        max_tokens=kwargs.get("max_tokens"),
                        max_completion_tokens=kwargs.get("max_completion_tokens"),
                        use_legacy_max_tokens=kwargs.get(
                            "use_legacy_max_tokens", False
                        ),
                    ),
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
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise LLMError(f"OpenAI generation failed: {str(e)}")

    async def _stream_impl(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs,
    ):
        """
        OpenAI streaming with optional tools.

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
                    "temperature": kwargs.get("temperature"),
                    "top_p": kwargs.get("top_p"),
                    "frequency_penalty": kwargs.get("frequency_penalty"),
                    "presence_penalty": kwargs.get("presence_penalty"),
                    "reasoning_effort": kwargs.get("reasoning_effort"),
                    "service_tier": kwargs.get("service_tier"),
                    "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
                    "timeout": kwargs.get("timeout"),
                    "stream": True,
                    "stream_options": {
                        "include_usage": True
                    },  # Get token usage in streaming
                    **_build_token_param(
                        max_tokens=kwargs.get("max_tokens"),
                        max_completion_tokens=kwargs.get("max_completion_tokens"),
                        use_legacy_max_tokens=kwargs.get(
                            "use_legacy_max_tokens", False
                        ),
                    ),
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
                # Handle usage-only chunks (from stream_options={"include_usage": True})
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        self._record_usage(chunk.usage)
                    continue

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
            logger.error(f"OpenAI streaming failed: {str(e)}")
            raise LLMError(f"OpenAI streaming failed: {str(e)}")

    @property
    def info(self) -> Dict[str, Any]:
        """Get information about the OpenAI provider."""
        base_info = super().info
        base_info.update({"provider_name": "OpenAI", "api_compatible": "OpenAI"})
        return base_info
