"""
Grok (xAI) LLM provider implementation with integrated tracing.
"""
import os
import logging
from typing import Dict, Any, Optional, List

from ..core.exceptions import LLMError
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class GrokProvider(BaseLLMProvider):
    """Grok (xAI) LLM provider implementation with automatic call tracing."""
    
    def __init__(
        self,
        model: str = "grok-3",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Grok provider.

        Args:
            model: Grok model name (e.g., "grok-3", "grok-vision-beta")
            api_key: xAI API key
            **kwargs: Additional Grok-specific parameters
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        
        super().__init__(model=model, api_key=api_key, **kwargs)
        
        # Grok-specific default parameters
        self.default_params.update({
            'stream': kwargs.get('stream', False),
            'timeout': kwargs.get('timeout', 60)
        })
        
        # Base URL for xAI API
        self.base_url = kwargs.get('base_url', 'https://api.x.ai/v1')
        
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
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.debug("Grok client initialized")
            except ImportError:
                raise LLMError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client
    
    def _safe_parse_arguments(self, arguments_str: str) -> Dict[str, Any]:
        """
        Safely parse JSON arguments with error handling.

        Args:
            arguments_str: JSON string to parse

        Returns:
            Parsed dict or empty dict if parsing fails
        """
        import json

        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments: {arguments_str}")
            return {}

    def _convert_messages_to_openai(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert universal flat format to OpenAI's nested format.

        Grok uses OpenAI-compatible API, so we need the same conversion.
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


    async def _generate_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs
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
        import json

        try:
            # Build API call params
            api_params = {
                "model": self.model,
                "messages": self._convert_messages_to_openai(messages),
                "max_tokens": kwargs.get('max_tokens'),
                "temperature": kwargs.get('temperature'),
                "top_p": kwargs.get('top_p'),
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
                            "arguments": self._safe_parse_arguments(tc.function.arguments)
                        }
                        for tc in message.tool_calls
                    ]
                }
            else:
                return message.content

        except Exception as e:
            logger.error(f"Grok generation failed: {str(e)}")
            raise LLMError(f"Grok generation failed: {str(e)}")

    async def _stream_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs
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
        import json

        try:
            # Build API call params
            api_params = {
                "model": self.model,
                "messages": self._convert_messages_to_openai(messages),
                "max_tokens": kwargs.get('max_tokens'),
                "temperature": kwargs.get('temperature'),
                "top_p": kwargs.get('top_p'),
                "timeout": kwargs.get('timeout'),
                "stream": True,
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
                            tool_args=self._safe_parse_arguments(tool_call["arguments"]),
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
            logger.error(f"Grok streaming failed: {str(e)}")
            raise LLMError(f"Grok streaming failed: {str(e)}")

    @property
    def info(self) -> Dict[str, Any]:
        """Get information about the Grok provider."""
        base_info = super().info
        base_info.update({
            'base_url': self.base_url,
            'provider_name': 'Grok (xAI)',
            'api_compatible': 'OpenAI'
        })
        return base_info