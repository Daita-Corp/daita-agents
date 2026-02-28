"""
Mock LLM provider for testing with integrated tracing.
"""
import asyncio
import logging
from typing import Dict, Any, Optional

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing purposes with automatic call tracing."""
    
    def __init__(
        self,
        model: str = "mock-model",
        responses: Optional[Dict[str, str]] = None,
        delay: float = 0.1,
        **kwargs
    ):
        """
        Initialize mock provider.
        
        Args:
            model: Mock model name
            responses: Dictionary mapping prompts to responses
            delay: Artificial delay to simulate API calls
            **kwargs: Additional parameters
        """
        # Remove api_key from kwargs to avoid conflict, then pass it explicitly
        kwargs.pop('api_key', None)  # Remove if exists
        super().__init__(model=model, api_key="mock-key", **kwargs)
        
        # Predefined responses
        self.responses = responses or {}
        self.delay = delay
        
        # Default responses
        self.default_responses = {
            "default": "This is a mock response from the LLM.",
            "analyze": "Based on the data provided, here are the key insights: [mock analysis]",
            "summarize": "Summary: [mock summary of the content]",
            "error": "This is an error response for testing."
        }
        
        # Track calls for testing
        self.call_history = []
    
    async def _generate_impl(self, messages, tools=None, **kwargs):
        """
        Provider-specific implementation of mock text generation.

        This method contains the mock generation logic and is automatically
        wrapped with tracing by the base class generate() method.

        Args:
            messages: List of message dicts or a single prompt string
            tools: Optional tool specifications (ignored in mock)
            **kwargs: Optional parameters

        Returns:
            Mock response (string or dict depending on tools)
        """
        # Handle both old-style string prompts and new-style message lists
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list):
            # Extract last user message
            prompt = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    break
        else:
            prompt = str(messages)

        # Record the call
        self.call_history.append({
            'prompt': prompt,
            'messages': messages if isinstance(messages, list) else None,
            'tools': tools,
            'params': kwargs,
            'timestamp': asyncio.get_event_loop().time()
        })

        # Simulate API delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # If tools provided, return dict format
        if tools:
            return {
                "content": f"Mock response for: {prompt[:50]}...",
                "tool_calls": None
            }

        # Check for specific response
        if prompt in self.responses:
            return self.responses[prompt]

        # Check for keyword-based responses
        prompt_lower = prompt.lower()
        for keyword, response in self.default_responses.items():
            if keyword in prompt_lower:
                return response

        # Default response
        return f"Mock response for: {prompt[:50]}..."
    
    async def _generate_with_tools_single(
        self,
        messages: list,
        tools: list,
        **kwargs
    ) -> dict:
        """
        Mock implementation of single LLM call with tools.

        Returns a mock response without tool calls (final answer).
        Override this in tests if you need to test tool calling behavior.
        """
        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # Record the call
        self.call_history.append({
            'messages': messages,
            'tools': tools,
            'params': kwargs,
            'timestamp': asyncio.get_event_loop().time()
        })

        # Simulate API delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Return final answer (no tool calls for basic mock)
        return {
            "content": f"Mock response for: {user_message[:50]}...",
            "tool_calls": None
        }

    async def _stream_impl(
        self,
        messages: list,
        tools: list,
        **kwargs
    ):
        """
        Mock streaming implementation.

        Yields LLMChunk objects simulating streaming response.
        Override this in tests if you need custom streaming behavior.
        """
        from ..core.streaming import LLMChunk

        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # Record the call
        self.call_history.append({
            'messages': messages,
            'tools': tools,
            'params': kwargs,
            'timestamp': asyncio.get_event_loop().time()
        })

        # Simulate streaming with delays
        mock_response = f"Mock response for: {user_message[:50]}..."

        # Yield text chunks character by character (simplified)
        for i, char in enumerate(mock_response):
            if self.delay > 0:
                await asyncio.sleep(self.delay / len(mock_response))

            yield LLMChunk(
                type="text",
                content=char,
                index=i
            )

    def _get_last_token_usage(self) -> Dict[str, int]:
        """
        Override base class method to return mock token usage.

        Provides realistic but fake token counts for testing.
        """
        if self.call_history:
            # Get the last call to estimate tokens
            last_call = self.call_history[-1]
            prompt = last_call.get('prompt', '')

            # Mock realistic token counts
            estimated_prompt_tokens = max(5, len(prompt) // 4)  # Rough estimate
            # Assume a moderate response length for mocking
            estimated_completion_tokens = max(10, estimated_prompt_tokens // 2)

            return {
                'total_tokens': estimated_prompt_tokens + estimated_completion_tokens,
                'prompt_tokens': estimated_prompt_tokens,
                'completion_tokens': estimated_completion_tokens
            }

        # Fallback to default
        return super()._get_last_token_usage()
    
    def set_response(self, prompt: str, response: str) -> None:
        """Set a specific response for a prompt."""
        self.responses[prompt] = response
    
    def clear_history(self) -> None:
        """Clear call history."""
        self.call_history.clear()
    
    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the last call made to the provider."""
        return self.call_history[-1] if self.call_history else None
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get information about the mock provider."""
        base_info = super().info
        base_info.update({
            'provider_name': 'Mock LLM (Testing)',
            'call_count': len(self.call_history),
            'configured_responses': len(self.responses),
            'delay_seconds': self.delay
        })
        return base_info