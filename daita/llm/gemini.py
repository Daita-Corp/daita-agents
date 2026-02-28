"""
Google Gemini LLM provider implementation with integrated tracing.
Uses the new google.genai package (replaces deprecated google.generativeai).
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional, List

from ..core.exceptions import LLMError
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation with automatic call tracing."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gemini provider.

        Args:
            model: Gemini model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash")
            api_key: Google AI API key
            **kwargs: Additional Gemini-specific parameters
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        super().__init__(model=model, api_key=api_key, **kwargs)

        # Gemini-specific default parameters
        self.default_params.update({
            'timeout': kwargs.get('timeout', 60),
            'safety_settings': kwargs.get('safety_settings', None),
            'generation_config': kwargs.get('generation_config', None)
        })

        # Lazy-load Gemini client
        self._client = None

    @property
    def client(self):
        """Lazy-load Google Genai client (new package)."""
        if self._client is None:
            try:
                from google import genai
                self._validate_api_key()

                # Create client with API key
                self._client = genai.Client(api_key=self.api_key)
                logger.debug("Gemini client initialized with google.genai")
            except ImportError:
                raise LLMError(
                    "Google Genai package not installed. Install with: pip install google-genai"
                )
        return self._client

    async def _generate_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs
    ):
        """
        Gemini non-streaming with optional tools.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in Gemini format (already converted by base class)
            **kwargs: Optional parameters

        Returns:
            - If no tools or LLM returns text: str
            - If LLM wants to call tools: {"tool_calls": [...]}
        """
        try:
            from google.genai import types

            # Convert to Gemini format
            gemini_contents = self._convert_messages_to_gemini(messages)

            # Build generation config
            generation_config_params = {}
            if kwargs.get('max_tokens'):
                generation_config_params['max_output_tokens'] = kwargs['max_tokens']
            if kwargs.get('temperature') is not None:
                generation_config_params['temperature'] = kwargs['temperature']
            if kwargs.get('top_p') is not None:
                generation_config_params['top_p'] = kwargs['top_p']

            generation_config = types.GenerateContentConfig(
                **generation_config_params
            ) if generation_config_params else None

            # Prepare API call params
            api_params = {
                "model": self.model,
                "contents": gemini_contents,
            }

            if generation_config:
                api_params["config"] = generation_config

            # Add tools if provided
            if tools:
                gemini_tools = self._convert_tools_to_gemini_format(tools)
                api_params["config"] = api_params.get("config") or types.GenerateContentConfig()
                api_params["config"].tools = gemini_tools

            # Generate response
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                **api_params
            )

            # Store usage metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self._last_usage = response.usage_metadata
                token_usage = {
                    'total_tokens': response.usage_metadata.total_token_count or 0,
                    'prompt_tokens': response.usage_metadata.prompt_token_count or 0,
                    'completion_tokens': response.usage_metadata.candidates_token_count or 0
                }
                self._update_accumulated_metrics(token_usage)

            # Check for function calls - collect ALL of them
            tool_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for idx, part in enumerate(response.candidates[0].content.parts):
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        # Only collect tool calls with valid names
                        if fc.name:
                            # Convert args to dict
                            args_dict = {}
                            if hasattr(fc, 'args') and fc.args:
                                args_dict = dict(fc.args)

                            tool_calls.append({
                                "id": f"{fc.name}_{idx}_{id(fc)}",  # Unique ID using name + index + object id
                                "name": fc.name,
                                "arguments": args_dict
                            })

            # Return tool calls if any were found
            if tool_calls:
                return {"tool_calls": tool_calls}

            # Return text content
            if response.text:
                return response.text

            return ""

        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise LLMError(f"Gemini generation failed: {str(e)}")

    async def _stream_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs
    ):
        """
        Gemini streaming with optional tools.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in Gemini format (already converted by base class)
            **kwargs: Optional parameters

        Yields:
            LLMChunk objects with type "text" or "tool_call_complete"
        """
        from ..core.streaming import LLMChunk
        from google.genai import types

        try:
            # Convert to Gemini format
            gemini_contents = self._convert_messages_to_gemini(messages)

            # Build generation config
            generation_config_params = {}
            if kwargs.get('max_tokens'):
                generation_config_params['max_output_tokens'] = kwargs['max_tokens']
            if kwargs.get('temperature') is not None:
                generation_config_params['temperature'] = kwargs['temperature']
            if kwargs.get('top_p') is not None:
                generation_config_params['top_p'] = kwargs['top_p']

            generation_config = types.GenerateContentConfig(
                **generation_config_params
            ) if generation_config_params else None

            # Prepare API call params
            api_params = {
                "model": self.model,
                "contents": gemini_contents,
            }

            if generation_config:
                api_params["config"] = generation_config

            # Add tools if provided
            if tools:
                gemini_tools = self._convert_tools_to_gemini_format(tools)
                api_params["config"] = api_params.get("config") or types.GenerateContentConfig()
                api_params["config"].tools = gemini_tools

            # Stream response
            response_stream = await asyncio.to_thread(
                self.client.models.generate_content_stream,
                **api_params
            )

            # Process stream chunks
            for chunk in response_stream:
                # Text content
                if hasattr(chunk, 'text') and chunk.text:
                    yield LLMChunk(type="text", content=chunk.text, model=self.model)

                # Function calls
                if chunk.candidates and chunk.candidates[0].content.parts:
                    for idx, part in enumerate(chunk.candidates[0].content.parts):
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            # Only yield if function call has a valid name
                            if fc.name:
                                # Convert args to dict
                                args_dict = {}
                                if hasattr(fc, 'args') and fc.args:
                                    args_dict = dict(fc.args)

                                yield LLMChunk(
                                    type="tool_call_complete",
                                    tool_name=fc.name,
                                    tool_args=args_dict,
                                    tool_call_id=f"{fc.name}_{idx}_{id(fc)}",  # Unique ID
                                    model=self.model
                                )

                # Usage metadata (typically in last chunk)
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    self._last_usage = chunk.usage_metadata
                    token_usage = {
                        'total_tokens': chunk.usage_metadata.total_token_count or 0,
                        'prompt_tokens': chunk.usage_metadata.prompt_token_count or 0,
                        'completion_tokens': chunk.usage_metadata.candidates_token_count or 0
                    }
                    self._update_accumulated_metrics(token_usage)

        except Exception as e:
            logger.error(f"Gemini streaming failed: {str(e)}")
            raise LLMError(f"Gemini streaming failed: {str(e)}")

    def _get_last_token_usage(self) -> Dict[str, int]:
        """
        Override base class method to handle Gemini's token format.

        Gemini uses different token field names in usage_metadata.
        """
        if self._last_usage:
            # Gemini format
            prompt_tokens = getattr(self._last_usage, 'prompt_token_count', 0)
            completion_tokens = getattr(self._last_usage, 'candidates_token_count', 0)
            total_tokens = getattr(self._last_usage, 'total_token_count', prompt_tokens + completion_tokens)

            return {
                'total_tokens': total_tokens,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }

        # Fallback to base class estimation
        return super()._get_last_token_usage()

    def _convert_tools_to_format(self, tools: List['AgentTool']) -> List[Dict[str, Any]]:
        """
        Convert AgentTool list to Gemini function declaration format.

        Gemini uses a simpler format than OpenAI.
        """
        gemini_tools = []
        for tool in tools:
            openai_format = tool.to_openai_function()

            # Convert OpenAI format to Gemini dict format
            gemini_tools.append({
                "name": openai_format["function"]["name"],
                "description": openai_format["function"]["description"],
                "parameters": openai_format["function"]["parameters"]
            })

        return gemini_tools

    def _convert_tools_to_gemini_format(self, tools: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert tool dicts to Gemini Tool objects for the new API.

        Args:
            tools: List of tool dicts with name, description, parameters

        Returns:
            List of google.genai Tool objects
        """
        from google.genai import types

        function_declarations = []
        for tool in tools:
            # Create FunctionDeclaration
            func_decl = types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=tool.get("parameters", {})
            )
            function_declarations.append(func_decl)

        # Wrap in Tool object
        return [types.Tool(function_declarations=function_declarations)]

    def _convert_messages_to_gemini(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Convert universal flat format to Gemini's Content format.

        Gemini uses "user" and "model" roles (not "assistant").
        Uses Content objects with Part objects.
        """
        from google.genai import types

        gemini_contents = []

        for msg in messages:
            if msg["role"] == "user":
                gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=msg["content"])]
                    )
                )
            elif msg["role"] == "assistant":
                if msg.get("tool_calls"):
                    # Assistant with tool calls
                    parts = []
                    for tc in msg["tool_calls"]:
                        # Skip tool calls with empty names
                        if tc.get("name"):
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=tc["name"],
                                        args=tc["arguments"]
                                    )
                                )
                            )
                    # Only add message if we have valid tool calls
                    if parts:
                        gemini_contents.append(
                            types.Content(role="model", parts=parts)
                        )
                else:
                    # Regular assistant message
                    gemini_contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(text=msg.get("content", ""))]
                        )
                    )
            elif msg["role"] == "tool":
                # Tool result
                tool_name = msg.get("name", "")
                if tool_name:
                    gemini_contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    function_response=types.FunctionResponse(
                                        name=tool_name,
                                        response={"result": msg["content"]}
                                    )
                                )
                            ]
                        )
                    )

        return gemini_contents

    @property
    def info(self) -> Dict[str, Any]:
        """Get information about the Gemini provider."""
        base_info = super().info
        base_info.update({
            'provider_name': 'Google Gemini',
            'api_compatible': 'Google AI',
            'package': 'google-genai'
        })
        return base_info
