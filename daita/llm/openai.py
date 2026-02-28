"""
OpenAI LLM provider implementation with integrated tracing.
"""
import os
import logging
from typing import Dict, Any, Optional

from ..core.exceptions import LLMError
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation with automatic call tracing."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key
            **kwargs: Additional OpenAI-specific parameters
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        super().__init__(model=model, api_key=api_key, **kwargs)
        
        # OpenAI-specific default parameters
        self.default_params.update({
            'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
            'presence_penalty': kwargs.get('presence_penalty', 0.0),
            'timeout': kwargs.get('timeout', 60)
        })
        
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
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client
    
    async def _generate_impl(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs
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
        import json

        try:
            # Build API call params
            api_params = {
                "model": self.model,
                "messages": self._convert_messages_to_openai(messages),
                "max_tokens": kwargs.get('max_tokens'),
                "temperature": kwargs.get('temperature'),
                "top_p": kwargs.get('top_p'),
                "frequency_penalty": kwargs.get('frequency_penalty'),
                "presence_penalty": kwargs.get('presence_penalty'),
                "timeout": kwargs.get('timeout')
            }

            # Add tools if provided
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            # Make API call
            response = await self.client.chat.completions.create(**api_params)

            # Store usage
            self._last_usage = response.usage

            # Update accumulated metrics for cost tracking
            if response.usage:
                token_usage = {
                    'total_tokens': response.usage.total_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
                self._update_accumulated_metrics(token_usage)

            message = response.choices[0].message

            # Check if tool calls
            if message.tool_calls:
                return {
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments)
                        }
                        for tc in message.tool_calls
                    ]
                }
            else:
                return message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise LLMError(f"OpenAI generation failed: {str(e)}")

    async def _stream_impl(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]],
        **kwargs
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
        import json

        try:
            # Build API call params
            api_params = {
                "model": self.model,
                "messages": self._convert_messages_to_openai(messages),
                "max_tokens": kwargs.get('max_tokens'),
                "temperature": kwargs.get('temperature'),
                "top_p": kwargs.get('top_p'),
                "frequency_penalty": kwargs.get('frequency_penalty'),
                "presence_penalty": kwargs.get('presence_penalty'),
                "timeout": kwargs.get('timeout'),
                "stream": True,
                "stream_options": {"include_usage": True}  # Get token usage in streaming
            }

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
                    if hasattr(chunk, 'usage') and chunk.usage:
                        self._last_usage = chunk.usage
                        # Update accumulated metrics for cost tracking
                        token_usage = {
                            'total_tokens': chunk.usage.total_tokens,
                            'prompt_tokens': chunk.usage.prompt_tokens,
                            'completion_tokens': chunk.usage.completion_tokens
                        }
                        self._update_accumulated_metrics(token_usage)
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Stream text content
                if delta.content:
                    yield LLMChunk(
                        type="text",
                        content=delta.content,
                        model=self.model
                    )

                # Handle tool calls (streamed as deltas)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        index = tc_delta.index

                        # Initialize buffer for this tool call
                        if index not in tool_call_buffers:
                            tool_call_buffers[index] = {
                                "id": "",
                                "name": "",
                                "arguments": ""
                            }

                        # Accumulate partial data
                        if tc_delta.id:
                            tool_call_buffers[index]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_call_buffers[index]["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_buffers[index]["arguments"] += tc_delta.function.arguments

                # On stream end, emit complete tool calls
                if choice.finish_reason == "tool_calls":
                    for tool_call in tool_call_buffers.values():
                        yield LLMChunk(
                            type="tool_call_complete",
                            tool_name=tool_call["name"],
                            tool_args=json.loads(tool_call["arguments"]),
                            tool_call_id=tool_call["id"],
                            model=self.model
                        )

                # Store usage if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    self._last_usage = chunk.usage
                    # Update accumulated metrics for cost tracking
                    token_usage = {
                        'total_tokens': chunk.usage.total_tokens,
                        'prompt_tokens': chunk.usage.prompt_tokens,
                        'completion_tokens': chunk.usage.completion_tokens
                    }
                    self._update_accumulated_metrics(token_usage)

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {str(e)}")
            raise LLMError(f"OpenAI streaming failed: {str(e)}")

    def _convert_messages_to_openai(
        self,
        messages: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Convert universal flat format to OpenAI's nested format.

        OpenAI expects tool_calls in nested format:
        {"id": "x", "type": "function", "function": {"name": "...", "arguments": "..."}}

        Our internal format is flat:
        {"id": "x", "name": "...", "arguments": {...}}
        """
        import json

        openai_messages = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Convert flat format to OpenAI's nested format
                converted_tool_calls = []
                for tc in msg["tool_calls"]:
                    converted_tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"]
                        }
                    })

                openai_messages.append({
                    "role": "assistant",
                    "tool_calls": converted_tool_calls
                })
            else:
                # Pass through other messages unchanged
                openai_messages.append(msg)

        return openai_messages


    @property
    def info(self) -> Dict[str, Any]:
        """Get information about the OpenAI provider."""
        base_info = super().info
        base_info.update({
            'provider_name': 'OpenAI',
            'api_compatible': 'OpenAI'
        })
        return base_info