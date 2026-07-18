"""Stable provider-neutral model failure contracts."""

from __future__ import annotations

from enum import Enum


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
    ) -> None:
        if not isinstance(code, ProviderErrorCode):
            raise TypeError("code must be a ProviderErrorCode")
        if message is not None and (
            not isinstance(message, str) or not message.strip()
        ):
            raise ValueError("message must be a non-empty string when provided")
        self.code = code
        super().__init__(message or code.value)
