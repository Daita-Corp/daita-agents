"""Define stable public error categories and safe exception normalization."""

from __future__ import annotations

import math
import re
from enum import Enum
from pathlib import Path

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


class StateCompatibilityCode(str, Enum):
    """Stable classification of local agent-home admission failures."""

    NEWER_REVISION = "state_revision_newer"
    LEGACY = "state_legacy"
    DAMAGED = "state_database_damaged"
    REVISION_UNSUPPORTED = "state_revision_unsupported"
    UPGRADE_FAILED = "state_upgrade_failed"


class StateCompatibilityError(DaitaError):
    """A local state database cannot be safely admitted by this release."""

    def __init__(
        self,
        code: StateCompatibilityCode,
        path: Path,
        message: str,
        *,
        current_revision: str,
        found_revision: str | None = None,
    ) -> None:
        if not isinstance(code, StateCompatibilityCode):
            raise TypeError("state compatibility code is invalid")
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("state compatibility path must be absolute")
        if not isinstance(current_revision, str) or not current_revision.strip():
            raise ValueError("current state revision must be non-empty text")
        if found_revision is not None and (
            not isinstance(found_revision, str) or not found_revision.strip()
        ):
            raise ValueError("found state revision must be non-empty text")
        self.code = code
        self.path = path
        self.found_revision = found_revision
        self.current_revision = current_revision
        super().__init__(
            message,
            error_code=code.value,
            retryability=ErrorRetryability.PERMANENT,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": str(self),
            "state_path": str(self.path),
            "found_revision": self.found_revision,
            "current_revision": self.current_revision,
            "state_changed": False,
        }


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


__all__ = [
    "AgentError",
    "AuthenticationError",
    "ConfigError",
    "DaitaError",
    "ErrorRetryability",
    "LLMError",
    "PermanentError",
    "RateLimitError",
    "StateCompatibilityCode",
    "StateCompatibilityError",
    "TransientError",
]
