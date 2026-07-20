"""Stable public error categories for the replacement runtime.

The hierarchy is intentionally small.  Concrete services may expose narrower
subclasses, but callers can make retry and subsystem decisions without parsing
messages or depending on provider/adapter exceptions.
"""

from __future__ import annotations

from enum import Enum
import math
import re

_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")


def _required_message(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("error message must be a non-empty string")
    if len(value) > 4_096:
        raise ValueError("error message exceeds 4096 characters")
    return value


def _error_code(value: str) -> str:
    if not isinstance(value, str) or _ERROR_CODE.fullmatch(value) is None:
        raise ValueError("error_code must be a bounded snake-case identifier")
    return value


def _optional_identifier(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    if len(value) > 256 or any(character in "\r\n\x00" for character in value):
        raise ValueError(f"{field_name} must be a bounded single-line identifier")
    return value


class ErrorRetryability(str, Enum):
    """Stable retry guidance attached to public Daita failures."""

    UNKNOWN = "unknown"
    TRANSIENT = "transient"
    RETRYABLE = "retryable"
    PERMANENT = "permanent"


class DaitaError(RuntimeError):
    """Base for normalized public failures.

    ``error_code`` and ``retryability`` are safe machine-readable facts.  Raw
    provider responses, connector diagnostics, arguments, and secret values do
    not belong on this record.
    """

    def __init__(
        self,
        message: str = "Daita operation failed.",
        *,
        error_code: str = "daita_error",
        retryability: ErrorRetryability = ErrorRetryability.UNKNOWN,
    ) -> None:
        if not isinstance(retryability, ErrorRetryability):
            raise TypeError("retryability must be an ErrorRetryability")
        self.error_code = _error_code(error_code)
        self.retryability = retryability
        super().__init__(_required_message(message))

    @property
    def retry_hint(self) -> str:
        """String form retained for logging and boundary serialization."""

        return self.retryability.value

    def is_transient(self) -> bool:
        return self.retryability is ErrorRetryability.TRANSIENT

    def is_retryable(self) -> bool:
        return self.retryability in {
            ErrorRetryability.TRANSIENT,
            ErrorRetryability.RETRYABLE,
        }

    def is_permanent(self) -> bool:
        return self.retryability is ErrorRetryability.PERMANENT


class AgentError(DaitaError):
    """Normalized public agent lifecycle or operation failure."""

    def __init__(
        self,
        message: str = "Agent operation failed.",
        *,
        agent_id: str | None = None,
        error_code: str = "agent_error",
        retryability: ErrorRetryability = ErrorRetryability.UNKNOWN,
    ) -> None:
        self.agent_id = _optional_identifier(agent_id, "agent_id")
        super().__init__(
            message,
            error_code=error_code,
            retryability=retryability,
        )


class ConfigError(DaitaError):
    """Permanent invalid or unavailable runtime configuration."""

    def __init__(
        self,
        message: str = "Configuration is invalid.",
        *,
        section: str | None = None,
        error_code: str = "config_error",
    ) -> None:
        self.section = _optional_identifier(section, "section")
        super().__init__(
            message,
            error_code=error_code,
            retryability=ErrorRetryability.PERMANENT,
        )


class LLMError(DaitaError):
    """Provider-neutral model or routing failure."""

    def __init__(
        self,
        message: str = "Model request failed.",
        *,
        provider_id: str | None = None,
        error_code: str = "llm_error",
        retryability: ErrorRetryability = ErrorRetryability.UNKNOWN,
    ) -> None:
        self.provider_id = _optional_identifier(provider_id, "provider_id")
        DaitaError.__init__(
            self,
            message,
            error_code=error_code,
            retryability=retryability,
        )


class PluginError(DaitaError):
    """Failure owned by an extension or resource adapter."""

    def __init__(
        self,
        message: str = "Extension operation failed.",
        *,
        plugin_id: str | None = None,
        error_code: str = "plugin_error",
        retryability: ErrorRetryability = ErrorRetryability.UNKNOWN,
    ) -> None:
        self.plugin_id = _optional_identifier(plugin_id, "plugin_id")
        super().__init__(
            message,
            error_code=error_code,
            retryability=retryability,
        )


class SkillError(PluginError):
    """Failure owned by skill discovery, selection, or lifecycle."""

    def __init__(
        self,
        message: str = "Skill operation failed.",
        *,
        skill_id: str | None = None,
        error_code: str = "skill_error",
        retryability: ErrorRetryability = ErrorRetryability.UNKNOWN,
    ) -> None:
        self.skill_id = _optional_identifier(skill_id, "skill_id")
        super().__init__(
            message,
            plugin_id=None,
            error_code=error_code,
            retryability=retryability,
        )


class TransientError(DaitaError):
    """Temporary failure that may clear without changing the request."""

    def __init__(
        self,
        message: str = "A temporary failure occurred.",
        *,
        error_code: str = "transient_error",
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            retryability=ErrorRetryability.TRANSIENT,
        )


class RetryableError(DaitaError):
    """Failure for which a bounded retry or alternate route may succeed."""

    def __init__(
        self,
        message: str = "The operation may be retried.",
        *,
        error_code: str = "retryable_error",
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            retryability=ErrorRetryability.RETRYABLE,
        )


class PermanentError(DaitaError):
    """Failure that must not be retried without changing configuration/input."""

    def __init__(
        self,
        message: str = "The operation cannot be retried as configured.",
        *,
        error_code: str = "permanent_error",
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            retryability=ErrorRetryability.PERMANENT,
        )


class RateLimitError(TransientError):
    """A normalized rate-limit response with an optional safe delay fact."""

    def __init__(self, *, retry_after_seconds: float | None = None) -> None:
        if retry_after_seconds is not None:
            if (
                not isinstance(retry_after_seconds, (int, float))
                or isinstance(retry_after_seconds, bool)
                or not math.isfinite(retry_after_seconds)
                or retry_after_seconds < 0
            ):
                raise ValueError("retry_after_seconds must be finite and non-negative")
            retry_after_seconds = float(retry_after_seconds)
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            "The model provider rate limit was reached.",
            error_code="rate_limit_error",
        )


class AuthenticationError(PermanentError):
    """A normalized authentication failure without credential diagnostics."""

    def __init__(self, *, provider_id: str | None = None) -> None:
        self.provider_id = _optional_identifier(provider_id, "provider_id")
        super().__init__(
            "Authentication failed.",
            error_code="authentication_error",
        )


class ValidationError(PermanentError):
    """A normalized validation failure that never retains the rejected value."""

    def __init__(self, *, field: str | None = None) -> None:
        self.field = _optional_identifier(field, "field")
        super().__init__(
            "Validation failed.",
            error_code="validation_error",
        )


class FocusDSLError(ValidationError):
    """Typed migration failure for the deferred standalone Focus DSL."""

    def __init__(self) -> None:
        super().__init__(field="focus_expression")
        self.error_code = "focus_dsl_error"


class DataQualityError(ValidationError):
    """Typed data-quality failure containing counts, never rejected rows."""

    def __init__(self, *, violation_count: int = 0) -> None:
        if (
            not isinstance(violation_count, int)
            or isinstance(violation_count, bool)
            or violation_count < 0
        ):
            raise ValueError("violation_count must be a non-negative integer")
        self.violation_count = violation_count
        super().__init__(field=None)
        self.error_code = "data_quality_error"


__all__ = [
    "AgentError",
    "AuthenticationError",
    "ConfigError",
    "DaitaError",
    "DataQualityError",
    "ErrorRetryability",
    "FocusDSLError",
    "LLMError",
    "PermanentError",
    "PluginError",
    "RateLimitError",
    "RetryableError",
    "SkillError",
    "TransientError",
    "ValidationError",
]
