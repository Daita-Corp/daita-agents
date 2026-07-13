"""Internal LLM service for DB runtime-owned planning tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
import os
import re
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
_PRIVATE_DIAGNOSTIC_SENSITIVE_KEYS = frozenset(
    {
        "address",
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "customer",
        "customer_name",
        "date_of_birth",
        "email",
        "first_name",
        "full_name",
        "last_name",
        "password",
        "phone",
        "secret",
        "social_security_number",
        "ssn",
        "token",
        "user_name",
        "username",
    }
)
_PRIVATE_DIAGNOSTIC_CONTENT_KEYS = frozenset(
    {
        "accepted_evidence_summaries",
        "catalog_context",
        "content",
        "db_memory_refs",
        "execution_error_summaries",
        "input",
        "memory_context",
        "normalized_user_request",
        "planning_context",
        "query_result",
        "rejected_evidence_summaries",
        "rendered_context",
        "result_rows",
        "row",
        "rows",
        "state",
        "task_summaries",
        "validation_summaries",
    }
)
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_INTERNATIONAL_PHONE_PATTERN = re.compile(r"\+\d[\d\s().-]{7,}\d")
_US_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
_US_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_API_KEY_PATTERN = re.compile(r"\b(?:sk|key|token)-[A-Za-z0-9_-]{8,}\b", re.I)
_BEARER_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}", re.I)
_PRIVATE_DIAGNOSTIC_MAX_ITEMS = 50


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

    async def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "db_json_response",
    ) -> DbLLMResponse:
        if not self.available:
            raise RuntimeError("DB LLM service is not configured")
        started = time.perf_counter()
        provider = self.provider
        structured_options: dict[str, Any] = {}
        if response_schema is not None:
            options = getattr(provider, "structured_output_options", None)
            if options is not None:
                structured_options = dict(
                    options(response_schema, name=schema_name) or {}
                )
        result = await provider.generate(
            messages,
            stream=False,
            **structured_options,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        content = _content_from_result(result)
        diagnostics = {
            **self.safe_metadata,
            "latency_ms": latency_ms,
        }
        if response_schema is not None:
            diagnostics["structured_output"] = {
                "schema_name": schema_name,
                "provider_native": bool(structured_options),
            }
        credential_values = _credential_values(self.config)
        raw_usage: dict[str, Any] = getattr(
            provider, "_get_last_token_usage", lambda: {}
        )()
        usage = _safe_diagnostics_mapping(
            raw_usage,
            _TOKEN_DIAGNOSTIC_KEYS,
            credential_values=credential_values,
        )
        if usage:
            diagnostics["tokens"] = usage
        estimate = getattr(provider, "_estimate_cost", lambda usage: None)(usage)
        if isinstance(estimate, (int, float)) and not isinstance(estimate, bool):
            diagnostics["estimated_cost_usd"] = estimate
        raw_pricing: dict[str, Any] = getattr(
            provider, "get_pricing_metadata", lambda: {}
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
    values = set()
    if config is not None:
        if config.api_key:
            values.add(config.api_key)
        values.update(_nested_string_values(config.options))
    secret_markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    values.update(
        value
        for key, value in os.environ.items()
        if value
        and len(value) >= 8
        and any(marker in key.upper() for marker in secret_markers)
    )
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


def redact_db_llm_private_diagnostic(
    value: Any,
    *,
    config: DbLLMConfig | None = None,
    max_chars: int = 4096,
) -> Any:
    """Return bounded private LLM diagnostics with secrets and content removed."""
    credentials = _credential_values(config)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                redacted = _redact_private_diagnostic_value(parsed, credentials)
                serialized = json.dumps(redacted, sort_keys=True, default=str)
                return _bounded_private_diagnostic(serialized, max_chars=max_chars)
    redacted = _redact_private_diagnostic_value(value, credentials)
    return _bounded_private_diagnostic(redacted, max_chars=max_chars)


def _redact_private_diagnostic_value(
    value: Any,
    credentials: frozenset[str],
) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for index, (key, nested) in enumerate(value.items()):
            if index >= _PRIVATE_DIAGNOSTIC_MAX_ITEMS:
                redacted["__truncated_items__"] = len(value) - index
                break
            normalized = str(key).lower()
            if (
                normalized in _PRIVATE_DIAGNOSTIC_SENSITIVE_KEYS
                or normalized in _PRIVATE_DIAGNOSTIC_CONTENT_KEYS
                or normalized.endswith(
                    (
                        "_api_key",
                        "_authorization",
                        "_credential",
                        "_password",
                        "_secret",
                    )
                )
            ):
                redacted[str(key)] = "[REDACTED_PRIVATE]"
            else:
                redacted[str(key)] = _redact_private_diagnostic_value(
                    nested, credentials
                )
        return redacted
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        redacted_items = [
            _redact_private_diagnostic_value(item, credentials)
            for item in items[:_PRIVATE_DIAGNOSTIC_MAX_ITEMS]
        ]
        if len(items) > _PRIVATE_DIAGNOSTIC_MAX_ITEMS:
            redacted_items.append(
                {"__truncated_items__": len(items) - _PRIVATE_DIAGNOSTIC_MAX_ITEMS}
            )
        return redacted_items
    if not isinstance(value, str):
        return value
    redacted = value
    for credential in credentials:
        redacted = redacted.replace(credential, "[REDACTED_SECRET]")
    redacted = _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", redacted)
    redacted = _BEARER_PATTERN.sub("[REDACTED_AUTHORIZATION]", redacted)
    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    redacted = _US_SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
    redacted = _CREDIT_CARD_PATTERN.sub(_redact_credit_card_match, redacted)
    redacted = _INTERNATIONAL_PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    return _US_PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)


def _redact_credit_card_match(match: re.Match[str]) -> str:
    candidate = match.group(0)
    if not _looks_like_credit_card(candidate):
        return candidate
    return "[REDACTED_CREDIT_CARD]"


def _looks_like_credit_card(candidate: str) -> bool:
    digits = [int(character) for character in candidate if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _bounded_private_diagnostic(value: Any, *, max_chars: int) -> Any:
    limit = max(256, min(int(max_chars), 8192))
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    serialized = json.dumps(value, sort_keys=True, default=str)
    if len(serialized) <= limit:
        return value
    return {
        "truncated": True,
        "preview": serialized[:limit] + "...[truncated]",
    }
