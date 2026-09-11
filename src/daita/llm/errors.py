"""Define normalized model failures and request-limit errors for routing and loops."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import Enum
from hashlib import sha256
from time import monotonic
from typing import cast

from .._json import FrozenJsonObject, canonical_json
from ..errors import (
    AuthenticationError,
    ErrorRetryability,
    LLMError,
    RateLimitError,
)
from .models import ModelRequest, ModelResponse, ModelUsage


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
    CLEANUP_FAILED = "cleanup_failed"
    CLEANUP_TIMEOUT = "cleanup_timeout"
    OWNER_UNAVAILABLE = "owner_unavailable"
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

# One immutable final snapshot in the current task, never a request/event history.
# The recorder clears it before entry and consumes it after generator cleanup.
# Responses/errors carry their own snapshots; this slot also covers normal aclose,
# which cannot return metadata through an already yielded immutable response.
_last_attempt: ContextVar[FrozenJsonObject | None] = ContextVar(
    "daita_last_provider_attempt", default=None
)


def take_provider_attempt_diagnostic() -> FrozenJsonObject | None:
    value = _last_attempt.get()
    _last_attempt.set(None)
    return value


class ProviderAttempt:
    """Bounded local phase observations shared by the three API adapters.

    No payloads, vendor errors, raw IDs, callbacks, I/O, or execution decisions.
    Native/header extraction stays with the adapter that understands its SDK.
    """

    def __init__(self, request: ModelRequest, *, headers_supported: bool) -> None:
        self.started = monotonic()
        self.events: dict[str, int] = {}
        self.values: dict[str, object] = {
            "deadline_remaining_seconds": (
                None
                if request.deadline is None
                else max(0.0, request.deadline - self.started)
            ),
            "count_state": "not_requested",
            "count_started_seconds": None,
            "count_finished_seconds": None,
            "counted_input_tokens": None,
            "output_cap": None,
            "generation_submitted_seconds": None,
            "first_native_event_seconds": None,
            "first_native_event_type": None,
            "last_native_event_seconds": None,
            "last_native_event_type": None,
            "native_event_overflow": 0,
            "response_id_digest": None,
            "terminal_observed": False,
            "cleanup_started_seconds": None,
            "cleanup_finished_seconds": None,
            "cleanup_failure": None,
            "failure_code": None,
            "transport_error_kind": None,
            "transport_error_phase": None,
        }
        for phase in ("count", "generation"):
            self.values.update(
                {
                    f"{phase}_headers_availability": (
                        "not_observed" if headers_supported else "unsupported"
                    ),
                    f"{phase}_headers_seconds": None,
                    f"{phase}_http_status": None,
                    f"{phase}_request_id_digest": None,
                }
            )
        _last_attempt.set(None)

    def mark(self, field: str) -> None:
        self.values[field] = round(max(0.0, monotonic() - self.started), 6)

    def start_count(self) -> None:
        self.values["count_state"] = "started"
        self.mark("count_started_seconds")

    def counted(self, tokens: int) -> None:
        self.values["count_state"] = "succeeded"
        self.values["counted_input_tokens"] = tokens
        self.mark("count_finished_seconds")

    def transport_failure(self, error: BaseException, *, phase: str) -> None:
        """Retain public transport exception categories before SDK causes are detached.

        No exception messages, requests or URLs are retained. This observation
        cannot affect normalization, retries, accounting or execution.
        """
        try:
            import httpx

            if phase not in {"count", "generation"}:
                return
            current: BaseException | None = error
            visited: set[int] = set()
            for _ in range(8):
                if current is None or id(current) in visited:
                    break
                visited.add(id(current))
                for error_type, kind in (
                    (httpx.ConnectTimeout, "connect_timeout"),
                    (httpx.ReadTimeout, "read_timeout"),
                    (httpx.WriteTimeout, "write_timeout"),
                    (httpx.PoolTimeout, "pool_timeout"),
                    (httpx.ConnectError, "connect_error"),
                    (httpx.RemoteProtocolError, "remote_protocol_error"),
                ):
                    if isinstance(current, error_type):
                        self.values["transport_error_kind"] = kind
                        self.values["transport_error_phase"] = phase
                        return
                current = current.__cause__ or current.__context__
        except Exception:
            # Diagnostics are best effort, including unusual exception chains.
            return

    def headers(
        self, phase: str, status: object, request_id: object, *, arrived: bool = True
    ) -> None:
        if type(status) is int and 100 <= status <= 599:
            if arrived:
                self.mark(f"{phase}_headers_seconds")
            self.values[f"{phase}_http_status"] = status
            self.values[f"{phase}_headers_availability"] = (
                "observed" if arrived else "status_only"
            )
        self.values[f"{phase}_request_id_digest"] = self._id_digest(request_id)

    @staticmethod
    def _id_digest(value: object) -> str | None:
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:-]{1,256}", value):
            return "sha256:" + sha256(value.encode("ascii")).hexdigest()
        return None

    def native(self, event_type: object, response_id: object = None) -> None:
        token = (
            event_type
            if isinstance(event_type, str) and _STRUCTURAL_TOKEN.fullmatch(event_type)
            else "unavailable"
        )
        if self.values["first_native_event_seconds"] is None:
            self.mark("first_native_event_seconds")
            self.values["first_native_event_type"] = token
        self.mark("last_native_event_seconds")
        self.values["last_native_event_type"] = token
        if token in self.events or len(self.events) < 16:
            self.events[token] = min(self.events.get(token, 0) + 1, 2**31 - 1)
        else:
            self.values["native_event_overflow"] = min(
                cast(int, self.values["native_event_overflow"]) + 1, 2**31 - 1
            )
        digest = self._id_digest(response_id)
        if digest is not None:
            self.values["response_id_digest"] = digest

    def snapshot(self) -> FrozenJsonObject:
        value = FrozenJsonObject.from_mapping(
            {**self.values, "native_event_counts": self.events}
        )
        # Fixed field count and capped counters/identifiers keep this below 8 KiB.
        if len(canonical_json(value).encode("utf-8")) > 8192:
            return FrozenJsonObject.from_mapping(
                {"measurement_availability": "unavailable"}
            )
        return value

    def response(self, response: ModelResponse) -> ModelResponse:
        self.values["terminal_observed"] = True
        try:
            return replace(
                response,
                provider_metadata={
                    **response.provider_metadata,
                    "attempt_diagnostic": self.snapshot(),
                },
            )
        except Exception:
            return response

    def finish(self, error: BaseException | None) -> None:
        if self.values["count_state"] == "started":
            self.values["count_state"] = "failed"
            self.mark("count_finished_seconds")
        if error is not None and not (
            isinstance(error, GeneratorExit) and self.values["terminal_observed"]
        ):
            self.values["failure_code"] = (
                error.code.value
                if isinstance(error, ModelProviderError)
                else (
                    "cancelled"
                    if isinstance(error, (asyncio.CancelledError, GeneratorExit))
                    else "provider_boundary_failure"
                )
            )
        try:
            snapshot = self.snapshot()
        except Exception:
            snapshot = FrozenJsonObject.from_mapping(
                {"measurement_availability": "unavailable"}
            )
        _last_attempt.set(snapshot)
        if isinstance(error, ModelProviderError):
            diagnostic = error.diagnostic or ProviderFailureDiagnostic(
                phase=ProviderFailurePhase.PROVIDER_BOUNDARY, code=error.code.value
            )
            error.diagnostic = replace(diagnostic, attempt=snapshot)
        elif isinstance(error, asyncio.CancelledError):
            setattr(error, "_daita_attempt_diagnostic", snapshot)


def interrupted_attempt_diagnostic(error: BaseException) -> FrozenJsonObject | None:
    cancellation = error.__cause__ if isinstance(error, TimeoutError) else error
    value = getattr(cancellation, "_daita_attempt_diagnostic", None)
    return value if isinstance(value, FrozenJsonObject) else None


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
    attempt: FrozenJsonObject | None = None

    def __post_init__(self) -> None:
        if self.attempt is not None and (
            not isinstance(self.attempt, FrozenJsonObject)
            or len(canonical_json(self.attempt).encode("utf-8")) > 8192
        ):
            raise ValueError("provider attempt diagnostic exceeds its bound")
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
        terminal_observed: bool = False,
        cleanup_unresolved: bool = False,
        canonical_emitted: bool = False,
    ) -> ModelProviderError:
        del (
            message,
            provider_id,
            retry_after_seconds,
            usage,
            diagnostic,
            terminal_observed,
            cleanup_unresolved,
            canonical_emitted,
        )
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
        terminal_observed: bool = False,
        cleanup_unresolved: bool = False,
        canonical_emitted: bool = False,
    ) -> None:
        if not isinstance(code, ProviderErrorCode):
            raise TypeError("code must be a ProviderErrorCode")
        if message is not None and (
            not isinstance(message, str) or not message.strip()
        ):
            raise ValueError("message must be a non-empty string when provided")
        self.code = code
        for name, value in (
            ("terminal_observed", terminal_observed),
            ("cleanup_unresolved", cleanup_unresolved),
            ("canonical_emitted", canonical_emitted),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
            setattr(self, name, value)
        self.terminal_observed: bool = terminal_observed
        self.cleanup_unresolved: bool = cleanup_unresolved
        self.canonical_emitted: bool = canonical_emitted
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
            attempt=None if error.diagnostic is None else error.diagnostic.attempt,
        ),
        terminal_observed=error.terminal_observed,
        cleanup_unresolved=error.cleanup_unresolved,
        canonical_emitted=error.canonical_emitted,
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
            terminal_observed=error.terminal_observed,
            cleanup_unresolved=error.cleanup_unresolved,
            canonical_emitted=error.canonical_emitted,
        )
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True
    return error
