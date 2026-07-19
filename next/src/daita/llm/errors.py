"""Stable provider-neutral model failure contracts."""

from __future__ import annotations

from enum import Enum

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


class ModelProviderError(RuntimeError):
    """One adapter failure already normalized at the provider boundary."""

    def __init__(
        self,
        code: ProviderErrorCode,
        message: str | None = None,
        *,
        routing: ModelRoutingTrace | None = None,
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
        super().__init__(message or code.value)


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
