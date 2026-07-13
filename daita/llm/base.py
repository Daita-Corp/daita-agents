"""
Base class for all LLM providers.

All LLM calls are automatically traced (tokens, cost, latency) without any
configuration required. Subclass this to add a new LLM provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
from contextlib import asynccontextmanager
import logging
import inspect

from ..core.exceptions import LLMError
from ..core.tracing import get_trace_manager, TraceType
from ..core.interfaces import LLMProvider
from .pricing import CostEstimate, TokenUsage, estimate_llm_cost

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
        self._last_cost_estimate: Optional[CostEstimate] = None

        # Accumulated token tracking
        self._accumulated_tokens = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        }

        logger.debug(
            f"Initialized {self.__class__.__name__} with model {model} (automatic tracing enabled)"
        )

    async def aclose(self) -> None:
        """Close an already-created provider client without forcing initialization.

        Providers in this package keep their optional SDK client in ``_client``.
        Async SDKs expose ``aclose()`` while synchronous clients such as Gemini
        expose ``close()``.  Clearing the owned reference in ``finally`` makes
        repeated closure safe and permits a later use to create a fresh client.
        """
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            close = getattr(client, "aclose", None)
            if close is None:
                close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            if getattr(self, "_client", None) is client:
                self._client = None

    def structured_output_options(
        self,
        schema: Dict[str, Any],
        *,
        name: str,
    ) -> Dict[str, Any]:
        """Return provider-native JSON-schema options when supported."""
        return {}

    async def generate(
        self,
        messages,
        tools: Optional[List[Any]] = None,
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

        trace_input = {
            "messages": messages,
            "tools": tool_specs,
            "params": params,
        }

        # Route to streaming or non-streaming
        if stream:
            return self._stream_with_tracing(
                messages, tool_specs, trace_input, **params
            )
        else:
            async with self.trace_manager.span(
                operation_name=f"llm_{self.provider_name}",
                trace_type=TraceType.LLM_CALL,
                agent_id=self.agent_id,
                model=self.model,
                input_data=trace_input,
            ) as span_id:
                result = await self._generate_impl(messages, tool_specs, **params)
                self.trace_manager.record_output(span_id, result)

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

    async def _stream_with_tracing(self, messages, tool_specs, trace_input, **params):
        """Wrap provider streaming with one LLM span and aggregate output events."""
        text_parts = []
        tool_calls = []

        async with self.trace_manager.span(
            operation_name=f"llm_{self.provider_name}",
            trace_type=TraceType.LLM_CALL,
            agent_id=self.agent_id,
            model=self.model,
            input_data=trace_input,
        ) as span_id:
            try:
                async for chunk in self._stream_impl(messages, tool_specs, **params):
                    if getattr(chunk, "type", None) == "text" and chunk.content:
                        text_parts.append(chunk.content)
                    elif getattr(chunk, "type", None) == "tool_call_complete":
                        tool_calls.append(
                            {
                                "id": chunk.tool_call_id,
                                "name": chunk.tool_name,
                                "arguments": chunk.tool_args,
                            }
                        )
                    yield chunk
            finally:
                output: Dict[str, Any] = {}
                if text_parts:
                    output["content"] = "".join(text_parts)
                if tool_calls:
                    output["tool_calls"] = tool_calls
                if output:
                    self.trace_manager.record_output(span_id, output)

                token_usage = self._get_last_token_usage()
                if token_usage.get("total_tokens"):
                    self.trace_manager.record_llm_call(
                        span_id=span_id,
                        model=self.model,
                        prompt_tokens=token_usage.get("prompt_tokens", 0),
                        completion_tokens=token_usage.get("completion_tokens", 0),
                        total_tokens=token_usage.get("total_tokens", 0),
                    )

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
    def _stream_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        **kwargs,
    ) -> AsyncIterator[Any]:
        """
        Provider-specific streaming implementation.

        Args:
            messages: Conversation history in standardized format
            tools: Tool specifications in provider-specific format (or None)
            **kwargs: Optional parameters

        Yields:
            LLMChunk objects with type "text" or "tool_call_complete"
        """
        ...

    @staticmethod
    def _extract_tokens(usage) -> Dict[str, int]:
        """Extract token counts from provider usage into legacy framework keys."""
        if not usage:
            return {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            }

        if isinstance(usage, dict):
            prompt_details = usage.get("prompt_tokens_details") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            prompt_tokens = (
                usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            )
            completion_tokens = (
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
            total_tokens = (
                usage.get("total_tokens") or prompt_tokens + completion_tokens
            )
            return {
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_input_tokens": usage.get(
                    "cached_input_tokens",
                    _get_usage_field(prompt_details, "cached_tokens", 0),
                ),
                "reasoning_tokens": usage.get(
                    "reasoning_tokens",
                    _get_usage_field(completion_details, "reasoning_tokens", 0),
                ),
            }

        if hasattr(usage, "total_tokens"):
            prompt_details = _get_usage_field(usage, "prompt_tokens_details") or {}
            completion_details = (
                _get_usage_field(usage, "completion_tokens_details") or {}
            )
            return {
                "total_tokens": getattr(usage, "total_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "cached_input_tokens": _get_usage_field(
                    prompt_details, "cached_tokens", 0
                ),
                "reasoning_tokens": _get_usage_field(
                    completion_details, "reasoning_tokens", 0
                ),
            }
        if hasattr(usage, "input_tokens"):
            inp = getattr(usage, "input_tokens", 0)
            out = getattr(usage, "output_tokens", 0)
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            return {
                "total_tokens": inp + out,
                "prompt_tokens": inp,
                "completion_tokens": out,
                "cached_input_tokens": cache_read,
                "reasoning_tokens": 0,
            }
        if hasattr(usage, "total_token_count"):
            prompt = getattr(usage, "prompt_token_count", 0) or 0
            completion = getattr(usage, "candidates_token_count", 0) or 0
            total = getattr(usage, "total_token_count", prompt + completion) or 0
            return {
                "total_tokens": total,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cached_input_tokens": getattr(usage, "cached_content_token_count", 0)
                or 0,
                "reasoning_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
            }

        return {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        }

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
        """Estimate cost using the shared model pricing catalog."""
        estimate = self._estimate_cost_details(token_usage)
        return estimate.as_float() if estimate.usd is not None else None

    def _estimate_cost_details(self, token_usage: Dict[str, int]) -> CostEstimate:
        """Return detailed pricing metadata for a token usage payload."""
        usage = TokenUsage.from_counts(
            input_tokens=token_usage.get("prompt_tokens", 0),
            output_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0),
            cached_input_tokens=token_usage.get("cached_input_tokens", 0),
            reasoning_tokens=token_usage.get("reasoning_tokens", 0),
        )
        estimate = estimate_llm_cost(self.provider_name, self.model, usage)
        if estimate.warning:
            logger.debug(estimate.warning)
        return estimate

    def get_last_cost_estimate(self) -> Optional[CostEstimate]:
        """Return pricing details for the most recent recorded LLM call."""
        return self._last_cost_estimate

    def get_pricing_metadata(self) -> Dict[str, Any]:
        """Return pricing metadata for operation reporting and debugging."""
        if self._last_cost_estimate is None:
            return {}
        return {
            "pricing_provider": self._last_cost_estimate.provider,
            "pricing_model": self._last_cost_estimate.pricing_model,
            "pricing_source": self._last_cost_estimate.pricing_source,
            "pricing_confidence": self._last_cost_estimate.pricing_confidence,
            "pricing_warning": self._last_cost_estimate.warning,
        }

    def _merge_params(self, override_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default parameters with overrides."""
        params = self.default_params.copy()
        params.update(override_params)
        return params

    def _validate_api_key(self) -> None:
        """Validate that API key is available."""
        if not self.api_key:
            raise ValueError(f"API key required for {self.__class__.__name__}")

    def _provider_error(self, message: str, error: Exception) -> LLMError:
        """Wrap provider SDK errors while preserving retry-relevant context."""
        status_code = _get_error_status_code(error)
        error_type = error.__class__.__name__
        retry_after = getattr(error, "retry_after", None)
        if retry_after is None:
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            if headers is not None:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")

        retry_hint = "retryable"
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            retry_hint = "transient"
        elif status_code in {400, 401, 402, 403, 404, 422}:
            retry_hint = "permanent"
        elif error_type in {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectionError",
            "InternalServerError",
            "RateLimitError",
            "ReadTimeout",
            "ServiceUnavailableError",
            "TimeoutError",
        }:
            retry_hint = "transient"
        elif error_type in {
            "AuthenticationError",
            "BadRequestError",
            "BillingError",
            "InvalidRequestError",
            "NotFoundError",
            "PermissionDeniedError",
            "PermissionError",
            "QuotaExceededError",
            "ValidationError",
        }:
            retry_hint = "permanent"

        context = {
            "provider_error_type": error_type,
            "status_code": status_code,
            "retry_after": retry_after,
        }
        return LLMError(
            f"{message}: {error}",
            provider=self.provider_name,
            model=self.model,
            retry_hint=retry_hint,
            context=context,
        )

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
            "cached_input_tokens": accumulated["cached_input_tokens"],
            "reasoning_tokens": accumulated["reasoning_tokens"],
            "estimated_cost": self._accumulated_cost,  # Accumulated cost across all calls
            **self.get_pricing_metadata(),
            "success_rate": metrics.get("success_rate", 0),
            "avg_latency_ms": metrics.get("avg_latency_ms", 0),
        }

    def _convert_tools_to_format(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert provider-neutral tool specs to provider-specific format.

        Default implementation uses OpenAI format. Providers can override
        to use their own format (e.g., Anthropic).
        """
        return [_tool_to_openai_function(tool) for tool in tools]

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
        self._accumulated_tokens["cached_input_tokens"] += token_usage.get(
            "cached_input_tokens", 0
        )
        self._accumulated_tokens["reasoning_tokens"] += token_usage.get(
            "reasoning_tokens", 0
        )

        # Update accumulated cost
        if cost is None:
            self._last_cost_estimate = self._estimate_cost_details(token_usage)
            cost = self._last_cost_estimate.as_float() or 0.0
        self._accumulated_cost += cost


def _get_usage_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _get_error_status_code(error: Exception) -> Optional[int]:
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _tool_to_openai_function(tool: Any) -> Dict[str, Any]:
    """Convert LocalTool or ModelToolSpec-like objects to OpenAI tool format."""
    converter = getattr(tool, "to_openai_function", None)
    if callable(converter):
        converted = converter()
        if not isinstance(converted, dict):
            raise TypeError("OpenAI tool conversion must return a dictionary")
        return {str(key): value for key, value in converted.items()}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        },
    }


def _tool_to_anthropic_tool(tool: Any) -> Dict[str, Any]:
    """Convert LocalTool or ModelToolSpec-like objects to Anthropic tool format."""
    converter = getattr(tool, "to_anthropic_tool", None)
    if callable(converter):
        converted = converter()
        if not isinstance(converted, dict):
            raise TypeError("Anthropic tool conversion must return a dictionary")
        return {str(key): value for key, value in converted.items()}
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.parameters),
    }


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
