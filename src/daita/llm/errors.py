"""Define normalized model failures and request-limit errors for routing and loops."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import Enum
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


class ContextEvidencePressureExceeded(LLMError):
    """The exact current run contains more evidence than its fixed bound."""

    def __init__(self) -> None:
        super().__init__(
            "Current tool evidence exceeds the fixed context-pressure bound; "
            "narrow rows, columns, filters, or aggregation.",
            error_code="context_evidence_limit_exceeded",
            retryability=ErrorRetryability.PERMANENT,
        )


class ToolSurfaceLimitExceeded(LLMError):
    """The fixed provider-facing direct/control surface exceeds its run bound."""

    def __init__(
        self,
        *,
        observed_tools: int,
        maximum_tools: int,
        observed_definition_bytes: int,
        maximum_definition_bytes: int,
    ) -> None:
        self.observed_tools = observed_tools
        self.maximum_tools = maximum_tools
        self.observed_definition_bytes = observed_definition_bytes
        self.maximum_definition_bytes = maximum_definition_bytes
        super().__init__(
            "The projected model tool surface exceeds its configured count or "
            "definition-byte bound.",
            error_code="tool_surface_limit_exceeded",
            retryability=ErrorRetryability.PERMANENT,
        )


class ToolCatalogLimitExceeded(LLMError):
    """The complete applicable run catalog exceeds its independent bound."""

    def __init__(
        self,
        *,
        observed_tools: int,
        maximum_tools: int,
        observed_catalog_bytes: int,
        maximum_catalog_bytes: int,
    ) -> None:
        self.observed_tools = observed_tools
        self.maximum_tools = maximum_tools
        self.observed_catalog_bytes = observed_catalog_bytes
        self.maximum_catalog_bytes = maximum_catalog_bytes
        super().__init__(
            "The applicable run tool catalog exceeds its configured count or "
            "canonical-byte bound.",
            error_code="tool_catalog_limit_exceeded",
            retryability=ErrorRetryability.PERMANENT,
        )


class ToolManifestLimitExceeded(LLMError):
    """The trusted compact toolbox manifest exceeds its independent bound."""

    def __init__(self) -> None:
        super().__init__(
            "The toolbox manifest exceeds its configured count, byte, or token bound.",
            error_code="tool_manifest_limit_exceeded",
            retryability=ErrorRetryability.PERMANENT,
        )


class RequestSensitivityUnavailable(LLMError):
    """The current admitted resource scope cannot be classified safely."""

    def __init__(self) -> None:
        super().__init__(
            "The admitted resource scope has no complete sensitivity classification.",
            error_code="request_sensitivity_unavailable",
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
    OUTPUT_LIMIT = "output_limit"
    MALFORMED_RESPONSE = "malformed_response"
    CONFIGURATION_ERROR = "configuration_error"
    LOCAL_ACCESS_ERROR = "local_access_error"
    TOKEN_LIMIT_REACHED = "token_limit_reached"
    TOKEN_BUDGET_INSUFFICIENT = "token_budget_insufficient"
    TOKEN_COUNT_UNAVAILABLE = "token_count_unavailable"
    COST_LIMIT_REACHED = "cost_limit_reached"
    COST_BUDGET_INSUFFICIENT = "cost_budget_insufficient"
    COST_LIMIT_UNPRICED_ROUTE = "cost_limit_unpriced_route"


class ProviderFailurePhase(str, Enum):
    """Bounded provider boundary phase where a normalized failure arose."""

    PROVIDER_BOUNDARY = "provider_boundary"
    REQUEST_ADMISSION = "request_admission"
    RESPONSE_DECODE = "response_decode"
    STREAM_EVENT = "stream_event"
    STREAM_TERMINAL = "stream_terminal"
    SUBSCRIPTION_OUTPUT = "subscription_output"


_DIAGNOSTIC_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_STRUCTURAL_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,95}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_OUTPUT_ITEM_TYPES = 8


@dataclass(frozen=True, slots=True)
class ProviderFailureDiagnostic:
    """Privacy-safe structural detail retained after vendor errors are detached."""

    phase: ProviderFailurePhase
    code: str
    event_type: str | None = None
    terminal_status: str | None = None
    output_item_types: tuple[str, ...] = ()
    response_id_digest: str | None = None
    input_tokens: int | None = None
    remaining_tokens: int | None = None
    maximum_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, ProviderFailurePhase):
            raise TypeError("provider failure phase must be ProviderFailurePhase")
        if not isinstance(self.code, str) or not _DIAGNOSTIC_CODE.fullmatch(self.code):
            raise ValueError("provider failure diagnostic code is invalid")
        for value, label in (
            (self.event_type, "event type"),
            (self.terminal_status, "terminal status"),
        ):
            if value is not None and (
                not isinstance(value, str) or not _STRUCTURAL_TOKEN.fullmatch(value)
            ):
                raise ValueError(f"provider failure {label} is invalid")
        for token_value in (
            self.input_tokens,
            self.remaining_tokens,
            self.maximum_output_tokens,
        ):
            if token_value is not None and (
                type(token_value) is not int or token_value < 0
            ):
                raise ValueError(
                    "provider admission token values must be non-negative integers"
                )
        if self.phase is not ProviderFailurePhase.REQUEST_ADMISSION and any(
            value is not None
            for value in (
                self.input_tokens,
                self.remaining_tokens,
                self.maximum_output_tokens,
            )
        ):
            raise ValueError("token admission values require the admission phase")
        output_item_types = tuple(self.output_item_types)
        if len(output_item_types) > _MAX_OUTPUT_ITEM_TYPES:
            raise ValueError("provider failure output item types exceed their bound")
        if any(
            not isinstance(item, str) or not _STRUCTURAL_TOKEN.fullmatch(item)
            for item in output_item_types
        ):
            raise ValueError("provider failure output item type is invalid")
        if self.response_id_digest is not None and (
            not isinstance(self.response_id_digest, str)
            or not _SHA256.fullmatch(self.response_id_digest)
        ):
            raise ValueError("provider failure response ID digest is invalid")
        object.__setattr__(self, "output_item_types", output_item_types)


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
        diagnostic: ProviderFailureDiagnostic | None = None,
    ) -> ModelProviderError:
        del message, provider_id, retry_after_seconds, usage, diagnostic
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
        diagnostic: ProviderFailureDiagnostic | None = None,
    ) -> None:
        if not isinstance(code, ProviderErrorCode):
            raise TypeError("code must be a ProviderErrorCode")
        if message is not None and (
            not isinstance(message, str) or not message.strip()
        ):
            raise ValueError("message must be a non-empty string when provided")
        self.code = code
        if retry_after_seconds is not None:
            if code not in {
                ProviderErrorCode.RATE_LIMIT_ERROR,
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
            }:
                raise ValueError(
                    "retry_after_seconds requires a rate-limit or unavailable-provider error"
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
        if diagnostic is not None and not isinstance(
            diagnostic, ProviderFailureDiagnostic
        ):
            raise TypeError(
                "diagnostic must be a ProviderFailureDiagnostic when provided"
            )
        self.diagnostic = diagnostic
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


def token_count_error(*, invalid: bool = False) -> ModelProviderError:
    """Missing counting support or invalid data cannot authorize generation."""
    from decimal import Decimal

    from .pricing import CostEstimate

    return ModelProviderError(
        ProviderErrorCode.TOKEN_COUNT_UNAVAILABLE,
        (
            "The provider returned an invalid input count; generation was not submitted."
            if invalid
            else "Complete input counting is unavailable; generation was not submitted."
        ),
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
        diagnostic=ProviderFailureDiagnostic(
            phase=ProviderFailurePhase.REQUEST_ADMISSION,
            code=(
                "input_token_count_invalid"
                if invalid
                else "input_token_count_unavailable"
            ),
        ),
    )


def before_generation(error: ModelProviderError, *, code: str) -> ModelProviderError:
    """Preserve normalized failure semantics with proven zero generation usage."""
    from decimal import Decimal
    from .pricing import CostEstimate

    return ModelProviderError(
        error.code,
        str(error),
        provider_id=error.provider_id,
        retry_after_seconds=error.retry_after_seconds,
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
        diagnostic=ProviderFailureDiagnostic(
            phase=ProviderFailurePhase.REQUEST_ADMISSION,
            code=code,
        ),
    )


def retry_after_from_headers(
    headers: object, *, now: datetime | None = None
) -> float | None:
    """Read bounded HTTP retry metadata; the router owns waiting and its ceiling."""
    if not isinstance(headers, Mapping):
        return None
    for key, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(key)
        if not isinstance(value, str) or not 0 < len(value) <= 128:
            continue
        try:
            seconds = float(value) / divisor
        except ValueError:
            if divisor != 1:
                continue
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    continue
                seconds = max(0.0, (date - (now or datetime.now(UTC))).total_seconds())
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def with_cancelled_model_usage(
    error: asyncio.CancelledError, usage: ModelUsage
) -> asyncio.CancelledError:
    """Attach canonical usage while preserving asyncio's exact cancellation type.

    Python 3.11 timeout contexts recognize CancelledError itself, not subclasses.
    Replacing it with a subclass would turn deadline expiry into cancellation.
    """
    if not isinstance(error, asyncio.CancelledError) or not isinstance(
        usage, ModelUsage
    ):
        raise TypeError("cancelled model usage requires cancellation and ModelUsage")
    setattr(error, "_daita_model_usage", usage)
    return error


def interrupted_model_usage(error: BaseException) -> ModelUsage:
    """Count cancellation before generation as zero; preserve unknown dispatches."""
    from .pricing import CostEstimate

    # asyncio.timeout converts CancelledError into TimeoutError at the enclosing
    # deadline while retaining the original cancellation as its cause.
    cancellation = error.__cause__ if isinstance(error, TimeoutError) else error
    if isinstance(cancellation, asyncio.CancelledError):
        usage = getattr(cancellation, "_daita_model_usage", None)
        if isinstance(usage, ModelUsage):
            return usage
    return ModelUsage(
        cost_estimate=CostEstimate.unavailable("model_attempt_interrupted")
    )


def detached_provider_error(
    error: ModelProviderError,
    *,
    provider_id: str | None = None,
) -> ModelProviderError:
    """Return a normalized error without retaining vendor diagnostics.

    Adapter and router boundaries deliberately raise the returned exception only
    after leaving their ``except`` blocks.  Clearing the existing traceback and
    chain prevents raw SDK exceptions (and their frame locals) from surviving in
    logs or uncaught-exception formatting while preserving the canonical code and
    normalized code.
    """

    if not isinstance(error, ModelProviderError):
        raise TypeError("error must be a ModelProviderError")
    if error.provider_id is None and provider_id is not None:
        error = ModelProviderError(
            error.code,
            str(error),
            provider_id=provider_id,
            retry_after_seconds=error.retry_after_seconds,
            usage=error.usage,
            diagnostic=error.diagnostic,
        )
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True
    return error
