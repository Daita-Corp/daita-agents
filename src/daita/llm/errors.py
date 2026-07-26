"""Stable provider-neutral model failure contracts."""

from __future__ import annotations

from enum import Enum
import math
from typing import cast

from ..errors import (
    AuthenticationError,
    ErrorRetryability,
    LLMError,
    RateLimitError,
)
from .models import ModelUsage


class ContextWindowExceeded(LLMError):
    """The mandatory provider-neutral model request cannot fit its profile."""

    def __init__(self) -> None:
        super().__init__(
            "The required model context exceeds the configured input window.",
            error_code="context_window_exceeded",
            retryability=ErrorRetryability.PERMANENT,
        )


class ProviderErrorCode(str, Enum):
    """Canonical failures that every model adapter may expose to the runtime."""

    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_NOT_FOUND = "model_not_found"
    CONTEXT_OVERFLOW = "context_overflow"
    INVALID_REQUEST = "invalid_request"
    CONTENT_BLOCKED = "content_blocked"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    MALFORMED_RESPONSE = "malformed_response"
    CONFIGURATION_ERROR = "configuration_error"


class ModelProviderError(LLMError):
    """One adapter failure already normalized at the provider boundary."""

    def __new__(
        cls,
        code: ProviderErrorCode,
        message: str | None = None,
        *,
        provider_id: str | None = None,
        retry_after_seconds: float | None = None,
        usage: ModelUsage = ModelUsage(),
    ) -> ModelProviderError:
        del message, provider_id, retry_after_seconds, usage
        concrete: type[ModelProviderError] = cls
        if cls is ModelProviderError:
            if code is ProviderErrorCode.RATE_LIMIT_ERROR:
                concrete = _ProviderRateLimitError
            elif code is ProviderErrorCode.AUTHENTICATION_ERROR:
                concrete = _ProviderAuthenticationError
        return cast(ModelProviderError, BaseException.__new__(concrete))

    def __init__(
        self,
        code: ProviderErrorCode,
        message: str | None = None,
        *,
        provider_id: str | None = None,
        retry_after_seconds: float | None = None,
        usage: ModelUsage = ModelUsage(),
    ) -> None:
        if not isinstance(code, ProviderErrorCode):
            raise TypeError("code must be a ProviderErrorCode")
        if message is not None and (
            not isinstance(message, str) or not message.strip()
        ):
            raise ValueError("message must be a non-empty string when provided")
        self.code = code
        if retry_after_seconds is not None:
            if code is not ProviderErrorCode.RATE_LIMIT_ERROR:
                raise ValueError(
                    "retry_after_seconds is valid only for a rate-limit error"
                )
            if (
                not isinstance(retry_after_seconds, (int, float))
                or isinstance(retry_after_seconds, bool)
                or not math.isfinite(retry_after_seconds)
                or retry_after_seconds < 0
            ):
                raise ValueError("retry_after_seconds must be finite and non-negative")
            retry_after_seconds = float(retry_after_seconds)
        self.retry_after_seconds: float | None = retry_after_seconds
        if not isinstance(usage, ModelUsage):
            raise TypeError("usage must be a ModelUsage record")
        self.usage = usage
        retryability = (
            ErrorRetryability.TRANSIENT
            if code
            in {
                ProviderErrorCode.RATE_LIMIT_ERROR,
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
                ProviderErrorCode.TIMEOUT,
            }
            else ErrorRetryability.PERMANENT
        )
        super().__init__(
            message or code.value,
            provider_id=provider_id,
            error_code=code.value,
            retryability=retryability,
        )


class _ProviderRateLimitError(ModelProviderError, RateLimitError):
    """Normalized provider rate limit catchable through both public types."""


class _ProviderAuthenticationError(ModelProviderError, AuthenticationError):
    """Normalized provider authentication failure with both public types."""


def detached_provider_error(error: ModelProviderError) -> ModelProviderError:
    """Return a normalized error without retaining vendor diagnostics.

    Adapter and router boundaries deliberately raise the returned exception only
    after leaving their ``except`` blocks.  Clearing the existing traceback and
    chain prevents raw SDK exceptions (and their frame locals) from surviving in
    logs or uncaught-exception formatting while preserving the canonical code and
    normalized code.
    """

    if not isinstance(error, ModelProviderError):
        raise TypeError("error must be a ModelProviderError")
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True
    return error
