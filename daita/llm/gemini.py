"""
Google Gemini LLM provider implementation with integrated tracing.
Uses the new google.genai package (replaces deprecated google.generativeai).
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import TYPE_CHECKING, Dict, Any, Optional, List

from ..core.exceptions import LLMError
from .base import BaseLLMProvider

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation with automatic call tracing."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Gemini provider.

        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash-lite", "gemini-2.5-flash")
            api_key: Google AI API key
            **kwargs: Additional Gemini-specific parameters
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        super().__init__(model=model, api_key=api_key, **kwargs)

        # Gemini-specific default parameters
        self.default_params.update(
            {
                "timeout": kwargs.get("timeout", 60),
                "safety_settings": kwargs.get("safety_settings", None),
                "generation_config": kwargs.get("generation_config", None),
                "top_k": kwargs.get("top_k", None),
                "stop_sequences": kwargs.get("stop_sequences", None),
                "response_mime_type": kwargs.get("response_mime_type", None),
                "response_schema": kwargs.get("response_schema", None),
                "thinking_config": kwargs.get("thinking_config", None),
            }
        )

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
                raise ImportError(
                    "Google Genai package not installed. Install with: pip install 'daita-agents[google]'"
                ) from None
        return self._client

    def _prepare_api_params(
        self,
        types,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs,
    ) -> Dict[str, Any]:
        system_instruction = "\n\n".join(
            msg.get("content", "") for msg in messages if msg.get("role") == "system"
        )
        conversation_messages = [msg for msg in messages if msg.get("role") != "system"]
        api_params = {
            "model": self.model,
            "contents": self._convert_messages_to_gemini(conversation_messages),
        }

        config = self._build_generation_config(
            types, tools=tools, system_instruction=system_instruction, **kwargs
        )
        if config:
            api_params["config"] = config
        return api_params

    def _build_generation_config(
        self,
        types,
        tools: Optional[List[Dict[str, Any]]],
        system_instruction: str = "",
        **kwargs,
    ):
        config = kwargs.get("generation_config")
        config_params = {}

        if isinstance(config, dict):
            config_params.update(config)
            config = None

        option_map = {
            "max_tokens": "max_output_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "stop_sequences": "stop_sequences",
            "response_mime_type": "response_mime_type",
            "response_schema": "response_schema",
            "thinking_config": "thinking_config",
        }
        for source_key, config_key in option_map.items():
            value = kwargs.get(source_key)
            if value is not None:
                config_params[config_key] = value

        if system_instruction:
            config_params["system_instruction"] = system_instruction

        safety_settings = kwargs.get("safety_settings")
        if safety_settings is not None:
            config_params["safety_settings"] = safety_settings

        if tools:
            config_params["tools"] = self._convert_tools_to_gemini_format(tools)

        if config is not None:
            for key, value in config_params.items():
                setattr(config, key, value)
            return config

        return types.GenerateContentConfig(**config_params) if config_params else None

    async def _generate_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs,
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

            api_params = self._prepare_api_params(types, messages, tools, **kwargs)

            # Generate response
            response = await asyncio.to_thread(
                self.client.models.generate_content, **api_params
            )

            # Store usage metadata
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self._record_usage(response.usage_metadata)

            # Check for function calls - collect ALL of them
            tool_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for idx, part in enumerate(response.candidates[0].content.parts):
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        # Only collect tool calls with valid names
                        if fc.name:
                            # Convert args to dict
                            args_dict = {}
                            if hasattr(fc, "args") and fc.args:
                                args_dict = dict(fc.args)

                            tool_calls.append(
                                {
                                    "id": f"{fc.name}_{idx}_{id(fc)}",  # Unique ID using name + index + object id
                                    "name": fc.name,
                                    "arguments": args_dict,
                                }
                            )

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
        **kwargs,
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
            api_params = self._prepare_api_params(types, messages, tools, **kwargs)

            # Stream response
            response_stream = await asyncio.to_thread(
                self.client.models.generate_content_stream, **api_params
            )

            # Process stream chunks
            for chunk in response_stream:
                # Text content
                if hasattr(chunk, "text") and chunk.text:
                    yield LLMChunk(type="text", content=chunk.text, model=self.model)

                # Function calls
                if chunk.candidates and chunk.candidates[0].content.parts:
                    for idx, part in enumerate(chunk.candidates[0].content.parts):
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            # Only yield if function call has a valid name
                            if fc.name:
                                # Convert args to dict
                                args_dict = {}
                                if hasattr(fc, "args") and fc.args:
                                    args_dict = dict(fc.args)

                                yield LLMChunk(
                                    type="tool_call_complete",
                                    tool_name=fc.name,
                                    tool_args=args_dict,
                                    tool_call_id=f"{fc.name}_{idx}_{id(fc)}",  # Unique ID
                                    model=self.model,
                                )

                # Usage metadata (typically in last chunk)
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    self._record_usage(chunk.usage_metadata)

        except Exception as e:
            logger.error(f"Gemini streaming failed: {str(e)}")
            raise LLMError(f"Gemini streaming failed: {str(e)}")

    def _convert_tools_to_format(
        self, tools: List["AgentTool"]
    ) -> List[Dict[str, Any]]:
        """
        Convert AgentTool list to Gemini function declaration format.

        Gemini uses a simpler format than OpenAI.
        """
        gemini_tools = []
        for tool in tools:
            openai_format = tool.to_openai_function()

            # Convert OpenAI format to Gemini dict format
            gemini_tools.append(
                {
                    "name": openai_format["function"]["name"],
                    "description": openai_format["function"]["description"],
                    "parameters": openai_format["function"]["parameters"],
                }
            )

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
                parameters=tool.get("parameters", {}),
            )
            function_declarations.append(func_decl)

        # Wrap in Tool object
        return [types.Tool(function_declarations=function_declarations)]

    def _convert_messages_to_gemini(self, messages: List[Dict[str, Any]]) -> List[Any]:
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
                    types.Content(role="user", parts=[types.Part(text=msg["content"])])
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
                                        name=tc["name"], args=tc["arguments"]
                                    )
                                )
                            )
                    # Only add message if we have valid tool calls
                    if parts:
                        gemini_contents.append(types.Content(role="model", parts=parts))
                else:
                    # Regular assistant message
                    gemini_contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(text=msg.get("content", ""))],
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
                                        response={"result": msg["content"]},
                                    )
                                )
                            ],
                        )
                    )

        return gemini_contents

    @property
    def info(self) -> Dict[str, Any]:
        """Get information about the Gemini provider."""
        base_info = super().info
        base_info.update(
            {
                "provider_name": "Google Gemini",
                "api_compatible": "Google AI",
                "package": "google-genai",
            }
        )
        return base_info
