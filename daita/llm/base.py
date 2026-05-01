"""
Base class for all LLM providers.

All LLM calls are automatically traced (tokens, cost, latency) without any
configuration required. Subclass this to add a new LLM provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Optional, List
from contextlib import asynccontextmanager
import logging

from ..core.tracing import get_trace_manager, TraceType
from ..core.interfaces import LLMProvider

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class BaseLLMProvider(LLMProvider, ABC):
    """
    Base class for LLM providers with automatic call tracing.

    Every LLM call is automatically traced with:
    - Provider and model details
    - Token usage and costs
    - Latency and performance
    - Input/output content (preview)
    - Error tracking

    Users get full LLM observability without any configuration.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM provider with automatic tracing.

        Args:
            model: Model identifier
            api_key: API key for authentication
            **kwargs: Additional provider-specific options
        """
        self.model = model
        self.api_key = api_key
        self.config = kwargs

        # Default parameters
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0),
        }

        # Agent ID for tracing (set by agent)
        self.agent_id = kwargs.get("agent_id")

        # Get trace manager for automatic tracing
        self.trace_manager = get_trace_manager()

        # Provider name for tracing
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()

        # Last usage for cost estimation
        self._last_usage = None

        # Accumulated cost tracking for operations
        self._accumulated_cost = 0.0

        # Accumulated token tracking
        self._accumulated_tokens = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        logger.debug(
            f"Initialized {self.__class__.__name__} with model {model} (automatic tracing enabled)"
        )

    async def generate(
        self,
        messages,
        tools: Optional[List[AgentTool]] = None,
        stream: bool = False,
        **kwargs,
    ):
        """
        Unified LLM generation - handles everything.

        Args:
            messages: Text prompt (str) or conversation messages (List[Dict])
            tools: Optional tools for LLM to call
            stream: If True, returns AsyncIterator of chunks
            **kwargs: LLM parameters (temperature, etc.)

        Returns:
            - stream=False, no tools: str (text response)
            - stream=False, with tools: Dict (with "content" or "tool_calls")
            - stream=True: AsyncIterator[LLMChunk] (real-time chunks)

        Examples:
            # Simple text
            text = await llm.generate("Hello")

            # Streaming text
            async for chunk in llm.generate("Hello", stream=True):
                print(chunk.content)

            # Tool calling
            result = await llm.generate(messages, tools=tools)

            # Streaming with tools
            async for chunk in llm.generate(messages, tools=tools, stream=True):
                if chunk.type == "text":
                    print(chunk.content)
                elif chunk.type == "tool_call_complete":
                    execute_tool(chunk.tool_name, chunk.tool_args)
        """
        # Normalize messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Merge parameters
        params = self._merge_params(kwargs)

        # Convert tools to provider format if provided
        tool_specs = None
        if tools:
            tool_specs = self._convert_tools_to_format(tools)

        # Route to streaming or non-streaming
        if stream:
            return self._stream_impl(messages, tool_specs, **params)
        else:
            async with self.trace_manager.span(
                operation_name=f"llm_{self.provider_name}",
                trace_type=TraceType.LLM_CALL,
                agent_id=self.agent_id,
                model=self.model,
            ) as span_id:
                result = await self._generate_impl(messages, tool_specs, **params)

                # Record token usage on the span
                token_usage = self._get_last_token_usage()
                if token_usage.get("total_tokens"):
                    self.trace_manager.record_llm_call(
                        span_id=span_id,
                        model=self.model,
                        prompt_tokens=token_usage.get("prompt_tokens", 0),
                        completion_tokens=token_usage.get("completion_tokens", 0),
                        total_tokens=token_usage.get("total_tokens", 0),
                    )

                return result

    @abstractmethod
    async def _generate_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs,
    ):
        """
        Provider-specific non-streaming implementation.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in provider-specific format (or None)
            **kwargs: Optional parameters

        Returns:
            - If no tools or LLM returns text: str
            - If LLM wants to call tools: {"tool_calls": [...]}
        """
        pass

    @abstractmethod
    async def _stream_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs,
    ):
        """
        Provider-specific streaming implementation.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in provider-specific format (or None)
            **kwargs: Optional parameters

        Yields:
            LLMChunk objects with type "text" or "tool_call_complete"
        """
        pass

    @staticmethod
    def _extract_tokens(usage) -> Dict[str, int]:
        """Extract token counts from an OpenAI or Anthropic usage object."""
        if not usage:
            return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        if hasattr(usage, "total_tokens"):
            return {
                "total_tokens": getattr(usage, "total_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            }
        if hasattr(usage, "input_tokens"):
            inp = getattr(usage, "input_tokens", 0)
            out = getattr(usage, "output_tokens", 0)
            return {
                "total_tokens": inp + out,
                "prompt_tokens": inp,
                "completion_tokens": out,
            }
        if hasattr(usage, "total_token_count"):
            prompt = getattr(usage, "prompt_token_count", 0) or 0
            completion = getattr(usage, "candidates_token_count", 0) or 0
            total = getattr(usage, "total_token_count", prompt + completion) or 0
            return {
                "total_tokens": total,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            }

        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _record_usage(self, usage) -> None:
        """Store provider usage and update accumulated token/cost metrics."""
        self._last_usage = usage
        token_usage = self._extract_tokens(usage)
        if token_usage.get("total_tokens"):
            self._update_accumulated_metrics(token_usage)

    def _get_last_token_usage(self) -> Dict[str, int]:
        """Get token usage from the last API call."""
        return self._extract_tokens(self._last_usage)

    def _estimate_cost(self, token_usage: Dict[str, int]) -> Optional[float]:
        """
        Simple cost estimation for MVP.

        Providers can override with specific pricing.
        """
        total_tokens = token_usage.get("total_tokens", 0)
        if total_tokens == 0:
            return None

        # Generic estimation for MVP - providers should override
        cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens
        return (total_tokens / 1000) * cost_per_1k_tokens

    def _merge_params(self, override_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default parameters with overrides."""
        params = self.default_params.copy()
        params.update(override_params)
        return params

    def _validate_api_key(self) -> None:
        """Validate that API key is available."""
        if not self.api_key:
            raise ValueError(f"API key required for {self.__class__.__name__}")

    def set_agent_id(self, agent_id: str):
        """
        Set the agent ID for tracing context.

        This is called automatically by BaseAgent during initialization.
        """
        self.agent_id = agent_id
        logger.debug(f"Set agent ID {agent_id} for {self.provider_name} provider")

    def get_recent_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent LLM calls for this provider's agent from unified tracing."""
        if not self.agent_id:
            return []

        operations = self.trace_manager.get_recent_operations(
            agent_id=self.agent_id, limit=limit * 2
        )

        # Filter for LLM calls from this provider
        llm_calls = [
            op
            for op in operations
            if (
                op.get("type") == "llm_call"
                and op.get("metadata", {}).get("llm_provider") == self.provider_name
            )
        ]

        return llm_calls[:limit]

    def get_token_stats(self) -> Dict[str, Any]:
        """Get token usage statistics from unified tracing."""
        if not self.agent_id:
            return {"total_calls": 0, "total_tokens": 0, "estimated_cost": 0.0}

        metrics = self.trace_manager.get_agent_metrics(self.agent_id)

        # Get accumulated token usage across all calls (not just the last one)
        accumulated = self._accumulated_tokens.copy()

        return {
            "total_calls": metrics.get("total_operations", 0),  # All operations
            "total_tokens": accumulated["total_tokens"],  # From ALL API calls
            "prompt_tokens": accumulated["prompt_tokens"],
            "completion_tokens": accumulated["completion_tokens"],
            "estimated_cost": self._accumulated_cost,  # Accumulated cost across all calls
            "success_rate": metrics.get("success_rate", 0),
            "avg_latency_ms": metrics.get("avg_latency_ms", 0),
        }

    def _convert_tools_to_format(self, tools: List[AgentTool]) -> List[Dict[str, Any]]:
        """
        Convert AgentTool list to provider-specific format.

        Default implementation uses OpenAI format. Providers can override
        to use their own format (e.g., Anthropic).
        """
        return [tool.to_openai_function() for tool in tools]

    @property
    def info(self) -> Dict[str, Any]:
        """Get information about this LLM provider."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "agent_id": self.agent_id,
            "config": {k: v for k, v in self.config.items() if "key" not in k.lower()},
            "default_params": self.default_params,
            "tracing_enabled": True,
        }

    @property
    def model_name(self) -> str:
        """Get the model name (alias for self.model for backwards compatibility)."""
        return self.model

    def get_accumulated_cost(self) -> float:
        """
        Get the accumulated cost across all LLM calls for this provider instance.

        Returns:
            float: Total estimated cost in USD
        """
        return self._accumulated_cost

    def get_accumulated_tokens(self) -> Dict[str, int]:
        """
        Get the accumulated token usage across all LLM calls for this provider instance.

        Returns:
            Dict with total_tokens, prompt_tokens, completion_tokens
        """
        return self._accumulated_tokens.copy()

    def _update_accumulated_metrics(
        self, token_usage: Dict[str, int], cost: Optional[float] = None
    ):
        """
        Update accumulated metrics after an LLM call.

        Args:
            token_usage: Token usage from the call
            cost: Estimated cost (if None, will be calculated)
        """
        # Update accumulated tokens
        self._accumulated_tokens["total_tokens"] += token_usage.get("total_tokens", 0)
        self._accumulated_tokens["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
        self._accumulated_tokens["completion_tokens"] += token_usage.get(
            "completion_tokens", 0
        )

        # Update accumulated cost
        if cost is None:
            cost = self._estimate_cost(token_usage) or 0.0
        self._accumulated_cost += cost


# Context manager for batch LLM operations
@asynccontextmanager
async def traced_llm_batch(
    llm_provider: BaseLLMProvider, batch_name: str = "llm_batch"
):
    """
    Context manager for tracing batch LLM operations.

    Usage:
        async with traced_llm_batch(llm, "document_analysis"):
            summary = await llm.generate("Summarize: " + doc1)
            analysis = await llm.generate("Analyze: " + doc2)
    """
    trace_manager = get_trace_manager()

    async with trace_manager.span(
        operation_name=f"llm_batch_{batch_name}",
        trace_type=TraceType.LLM_CALL,
        agent_id=llm_provider.agent_id,
        llm_provider=llm_provider.provider_name,
        batch_operation=batch_name,
    ):
        try:
            yield llm_provider
        except Exception as e:
            logger.error(f"LLM batch {batch_name} failed: {e}")
            raise


# Utility functions
def get_llm_traces(
    agent_id: Optional[str] = None, provider: Optional[str] = None, limit: int = 20
) -> List[Dict[str, Any]]:
    """Get recent LLM call traces from unified tracing."""
    trace_manager = get_trace_manager()
    operations = trace_manager.get_recent_operations(agent_id=agent_id, limit=limit * 2)

    # Filter for LLM operations
    llm_ops = [op for op in operations if op.get("type") == "llm_call"]

    # Filter by provider if specified
    if provider:
        llm_ops = [
            op
            for op in llm_ops
            if op.get("metadata", {}).get("llm_provider") == provider
        ]

    return llm_ops[:limit]


def get_llm_stats(
    agent_id: Optional[str] = None, provider: Optional[str] = None
) -> Dict[str, Any]:
    """Get LLM usage statistics from unified tracing."""
    traces = get_llm_traces(agent_id, provider, limit=50)

    if not traces:
        return {"total_calls": 0, "success_rate": 0, "avg_latency_ms": 0}

    total_calls = len(traces)
    successful_calls = len([t for t in traces if t.get("status") == "success"])

    # Calculate averages
    latencies = [t.get("duration_ms", 0) for t in traces if t.get("duration_ms")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Token aggregation (simplified for MVP)
    total_tokens = 0
    for trace in traces:
        metadata = trace.get("metadata", {})
        total_tokens += metadata.get("tokens_total", 0)

    return {
        "total_calls": total_calls,
        "successful_calls": successful_calls,
        "failed_calls": total_calls - successful_calls,
        "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "agent_id": agent_id,
        "provider": provider,
    }


# Export everything
__all__ = [
    # Base class
    "BaseLLMProvider",
    # Context managers
    "traced_llm_batch",
    # Utility functions
    "get_llm_traces",
    "get_llm_stats",
]
