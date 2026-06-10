"""Internal LLM service for DB runtime-owned planning tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass(frozen=True)
class DbLLMConfig:
    """Safe DB LLM configuration plus non-persisted secret material."""

    provider: str
    model: str
    api_key: str | None = None
    temperature: float | None = None
    agent_id: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

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

    def __init__(self, config: DbLLMConfig | None) -> None:
        self.config = config
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
                agent_id=self.config.agent_id,
                **options,
            )
        return self._provider

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
        usage = getattr(self.provider, "_get_last_token_usage", lambda: {})()
        if usage:
            diagnostics["tokens"] = dict(usage)
        estimate = getattr(self.provider, "_estimate_cost", lambda usage: None)(usage)
        if estimate is not None:
            diagnostics["estimated_cost_usd"] = estimate
        pricing = getattr(self.provider, "get_pricing_metadata", lambda: {})()
        if pricing:
            diagnostics["pricing"] = dict(pricing)
        return DbLLMResponse(content=content, diagnostics=diagnostics)

    async def generate_synthesis_json(
        self, messages: list[dict[str, str]]
    ) -> DbLLMResponse:
        """Generate strict JSON for DB answer synthesis."""
        return await self.generate_json(messages)


def db_llm_service_from_config(
    *,
    model: str | None,
    llm_provider: str | None,
    api_key: str | None = None,
    temperature: float | None = None,
    agent_id: str | None = None,
    options: dict[str, Any] | None = None,
) -> DbLLMService:
    if not model and not llm_provider:
        return DbLLMService(None)
    provider = llm_provider or "openai"
    if not model:
        return DbLLMService(None)
    return DbLLMService(
        DbLLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            agent_id=agent_id,
            options=dict(options or {}),
        )
    )


def db_llm_service_from_metadata(metadata: dict[str, Any]) -> DbLLMService:
    options = metadata.get("from_db_options")
    if not isinstance(options, dict):
        return DbLLMService(None)
    return db_llm_service_from_config(
        model=options.get("model"),
        llm_provider=options.get("llm_provider"),
        temperature=options.get("temperature"),
        options={
            key: value
            for key, value in options.items()
            if key.startswith("planner_") or key.startswith("llm_")
        },
    )


def _content_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str):
            return content
    return str(result)
