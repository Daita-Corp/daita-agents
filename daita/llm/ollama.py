"""
Ollama LLM provider for running local models.

Uses Ollama's OpenAI-compatible API, so any model available via
``ollama pull`` works: llama3.1, mistral, gemma2, codestral, phi4, etc.

Requires Ollama running locally (https://ollama.com).
Default endpoint: http://localhost:11434

Usage:
    agent = Agent(
        name="local",
        llm_provider="ollama",
        llm_model="llama3.1",
    )
"""

import os
import logging
from typing import Dict, Any, Optional

from ..core.exceptions import LLMError
from .base import BaseLLMProvider
from .openai_compatible import OpenAICompatibleMixin, compact_params

logger = logging.getLogger(__name__)


class OllamaProvider(OpenAICompatibleMixin, BaseLLMProvider):
    """Ollama LLM provider using the OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "llama3.1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model: Ollama model name (e.g. "llama3.1", "llama3.2", "mistral")
            api_key: Not required for Ollama. Passed through for compatibility.
            base_url: Ollama server URL (default: http://localhost:11434/v1)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        # Ollama doesn't need an API key — use a placeholder for the base class
        super().__init__(model=model, api_key=api_key or "ollama", **kwargs)

        self._base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
        )
        self._client = None

        self.default_params.update(
            {
                "timeout": kwargs.get("timeout", 120),
            }
        )

    @property
    def client(self):
        """Lazy-load OpenAI client pointed at Ollama."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise LLMError(
                    "openai package not found. It is a core dependency — "
                    "reinstall with: pip install daita-agents"
                )
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self._base_url,
            )
            logger.debug(f"Ollama client initialized at {self._base_url}")
        return self._client

    async def _generate_impl(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs,
    ):
        """Non-streaming generation via Ollama's OpenAI-compatible endpoint."""
        try:
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

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**api_params)

            self._record_usage(response.usage)

            message = response.choices[0].message

            tool_calls = self._tool_calls_from_openai_message(message)
            if tool_calls:
                return {"tool_calls": tool_calls}
            return message.content

        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "refused" in error_msg:
                raise LLMError(self._connection_error_message())
            raise LLMError(f"Ollama generation failed: {error_msg}")

    async def _stream_impl(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs,
    ):
        """Streaming generation via Ollama's OpenAI-compatible endpoint."""
        from ..core.streaming import LLMChunk

        try:
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

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            stream = await self.client.chat.completions.create(**api_params)

            tool_call_buffers = {}

            async for chunk in stream:
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        self._record_usage(chunk.usage)
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    yield LLMChunk(type="text", content=delta.content, model=self.model)

                self._apply_openai_tool_call_deltas(tool_call_buffers, delta.tool_calls)

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

                if hasattr(chunk, "usage") and chunk.usage:
                    self._record_usage(chunk.usage)

        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "refused" in error_msg:
                raise LLMError(self._connection_error_message())
            raise LLMError(f"Ollama streaming failed: {error_msg}")

    def _connection_error_message(self) -> str:
        if os.getenv("DAITA_RUNTIME") == "lambda":
            return (
                "Ollama is a local-only LLM provider and cannot run in Daita Cloud. "
                "Use a cloud provider instead (openai, anthropic, gemini, grok)."
            )
        return (
            f"Cannot connect to Ollama at {self._base_url}. "
            f"Is Ollama running? Start it with: ollama serve"
        )

    @property
    def info(self) -> Dict[str, Any]:
        base_info = super().info
        base_info.update(
            {
                "provider_name": "Ollama",
                "base_url": self._base_url,
                "api_compatible": "OpenAI",
            }
        )
        return base_info
