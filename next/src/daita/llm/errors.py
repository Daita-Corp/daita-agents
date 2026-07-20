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
from .models import ModelRoutingTrace


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


class ModelProviderError(LLMError):
    """One adapter failure already normalized at the provider boundary."""

    def __new__(
        cls,
        code: ProviderErrorCode,
        message: str | None = None,
        *,
        routing: ModelRoutingTrace | None = None,
        provider_id: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> ModelProviderError:
        del message, routing, provider_id, retry_after_seconds
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
        routing: ModelRoutingTrace | None = None,
        provider_id: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        if not isinstance(code, ProviderErrorCode):
            raise TypeError("code must be a ProviderErrorCode")
        if message is not None and (
            not isinstance(message, str) or not message.strip()
        ):
            raise ValueError("message must be a non-empty string when provided")
        if routing is not None:
            if not isinstance(routing, ModelRoutingTrace):
                raise TypeError("routing must be a ModelRoutingTrace or None")
            if routing.terminal_error_code != code.value:
                raise ValueError("routing terminal error must match the provider code")
        self.code = code
        self.routing = routing
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
    routing trace.
    """

    if not isinstance(error, ModelProviderError):
        raise TypeError("error must be a ModelProviderError")
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True
    return error
