"""Internal LLM service for DB runtime-owned planning tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import time
from typing import Any

_TOKEN_DIAGNOSTIC_KEYS = frozenset(
    {
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
    }
)
_PRICING_DIAGNOSTIC_KEYS = frozenset(
    {
        "pricing_provider",
        "pricing_model",
        "pricing_source",
        "pricing_confidence",
        "pricing_warning",
    }
)


@dataclass(frozen=True)
class DbLLMConfig:
    """Safe DB LLM configuration plus non-persisted secret material."""

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    temperature: float | None = None
    options: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("llm.provider is required")
        if not self.model:
            raise ValueError("llm.model is required")
        object.__setattr__(self, "options", dict(self.options))

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "roles": ["planner", "repair", "synthesizer"],
        }


@dataclass(frozen=True)
class DbLLMResponse:
    """LLM response plus safe diagnostics."""

    content: str
    diagnostics: dict[str, Any]


class DbLLMService:
    """Lazy boundary around LLM providers used by DB runtime executors."""

    def __init__(
        self, config: DbLLMConfig | None, *, agent_id: str | None = None
    ) -> None:
        self.config = config
        self.agent_id = agent_id
        self._provider = None

    @property
    def available(self) -> bool:
        return self.config is not None and bool(
            self.config.provider and self.config.model
        )

    @property
    def safe_metadata(self) -> dict[str, Any]:
        return self.config.safe_metadata() if self.config is not None else {}

    @property
    def provider(self) -> Any:
        if self.config is None:
            raise RuntimeError("DB LLM service is not configured")
        if self._provider is None:
            from daita.llm.factory import create_llm_provider

            options = {
                **self.config.options,
                **(
                    {"temperature": self.config.temperature}
                    if self.config.temperature is not None
                    else {}
                ),
            }
            self._provider = create_llm_provider(
                self.config.provider,
                self.config.model,
                api_key=self.config.api_key,
                agent_id=self.agent_id,
                **options,
            )
        return self._provider

    async def aclose(self) -> None:
        """Close the already-created provider while preserving lazy setup."""
        provider = self._provider
        if provider is None:
            return
        try:
            close = getattr(provider, "aclose", None)
            if close is None:
                close = getattr(provider, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            if self._provider is provider:
                self._provider = None

    async def generate_json(self, messages: list[dict[str, str]]) -> DbLLMResponse:
        if not self.available:
            raise RuntimeError("DB LLM service is not configured")
        started = time.perf_counter()
        result = await self.provider.generate(messages, stream=False)
        latency_ms = (time.perf_counter() - started) * 1000
        content = _content_from_result(result)
        diagnostics = {
            **self.safe_metadata,
            "latency_ms": latency_ms,
        }
        credential_values = _credential_values(self.config)
        raw_usage: dict[str, Any] = getattr(
            self.provider, "_get_last_token_usage", lambda: {}
        )()
        usage = _safe_diagnostics_mapping(
            raw_usage,
            _TOKEN_DIAGNOSTIC_KEYS,
            credential_values=credential_values,
        )
        if usage:
            diagnostics["tokens"] = usage
        estimate = getattr(self.provider, "_estimate_cost", lambda usage: None)(usage)
        if isinstance(estimate, (int, float)) and not isinstance(estimate, bool):
            diagnostics["estimated_cost_usd"] = estimate
        raw_pricing: dict[str, Any] = getattr(
            self.provider, "get_pricing_metadata", lambda: {}
        )()
        pricing = _safe_diagnostics_mapping(
            raw_pricing,
            _PRICING_DIAGNOSTIC_KEYS,
            credential_values=credential_values,
        )
        if pricing:
            diagnostics["pricing"] = pricing
        return DbLLMResponse(content=content, diagnostics=diagnostics)

    async def generate_synthesis_json(
        self, messages: list[dict[str, str]]
    ) -> DbLLMResponse:
        """Generate strict JSON for DB answer synthesis."""
        return await self.generate_json(messages)


def db_llm_service_from_config(
    config: DbLLMConfig | None,
    *,
    agent_id: str | None = None,
) -> DbLLMService:
    if config is not None and not isinstance(config, DbLLMConfig):
        raise TypeError("llm must be a DbLLMConfig instance")
    return DbLLMService(config, agent_id=agent_id)


def _content_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str):
            return content
    return str(result)


def _safe_diagnostics_mapping(
    values: Any,
    allowed_keys: frozenset[str],
    *,
    credential_values: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    return {
        key: value
        for key, value in values.items()
        if key in allowed_keys
        and isinstance(value, (str, int, float, bool, type(None)))
        and not (
            isinstance(value, str)
            and any(secret in value for secret in credential_values)
        )
    }


def _credential_values(config: DbLLMConfig | None) -> frozenset[str]:
    if config is None:
        return frozenset()
    values = {config.api_key} if config.api_key else set()
    values.update(_nested_string_values(config.options))
    return frozenset(value for value in values if value)


def _nested_string_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        values: set[str] = set()
        for nested in value.values():
            values.update(_nested_string_values(nested))
        return values
    if isinstance(value, (list, tuple, set, frozenset)):
        values = set()
        for nested in value:
            values.update(_nested_string_values(nested))
        return values
    return set()
