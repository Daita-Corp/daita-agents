"""Small retry helpers for model turns inside the Agent runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar

from ...core.exceptions import DaitaError
from .state import RunState

T = TypeVar("T")


_TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectionError",
    "InternalServerError",
    "RateLimitError",
    "ReadTimeout",
    "ServiceUnavailableError",
    "TimeoutError",
}

_PERMANENT_ERROR_NAMES = {
    "AuthenticationError",
    "BadRequestError",
    "BillingError",
    "ConflictError",
    "InvalidRequestError",
    "NotFoundError",
    "PermissionDeniedError",
    "PermissionError",
    "QuotaExceededError",
    "ValidationError",
}


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    classification: str
    reason: str
    retry_after: Optional[float] = None


def classify_model_error(error: BaseException) -> RetryDecision:
    """Classify a model/provider failure for runtime-scoped retry."""
    if isinstance(error, asyncio.CancelledError):
        return RetryDecision(False, "cancelled", "cancelled")

    if getattr(error, "_daita_stream_event_emitted", False):
        return RetryDecision(
            False,
            "stream_partial_output",
            "stream already emitted user-visible output",
        )

    retry_after = _extract_retry_after(error)

    if isinstance(error, DaitaError):
        if error.retry_hint == "transient":
            return RetryDecision(True, "transient", error.retry_hint, retry_after)
        if error.retry_hint == "retryable":
            return RetryDecision(True, "retryable", error.retry_hint, retry_after)
        if error.retry_hint == "permanent":
            return RetryDecision(False, "permanent", error.retry_hint, retry_after)
        return RetryDecision(False, "unknown", "unknown daita retry hint", retry_after)

    error_name = error.__class__.__name__
    cause_name = error.__cause__.__class__.__name__ if error.__cause__ else ""
    names = {error_name, cause_name}

    if names & _TRANSIENT_ERROR_NAMES:
        return RetryDecision(
            True, "transient", "known transient exception", retry_after
        )
    if names & _PERMANENT_ERROR_NAMES:
        return RetryDecision(
            False, "permanent", "known permanent exception", retry_after
        )

    status_code = _extract_status_code(error)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return RetryDecision(
            True, "transient", f"transient status {status_code}", retry_after
        )
    if status_code in {400, 401, 402, 403, 404, 422}:
        return RetryDecision(
            False, "permanent", f"permanent status {status_code}", retry_after
        )

    return RetryDecision(False, "unknown", "unknown model error", retry_after)


def get_retry_delay(policy: Any, attempt: int, error: BaseException) -> float:
    """Return policy delay, honoring retry-after when present within max_delay."""
    retry_after = _extract_retry_after(error)
    if retry_after is not None and retry_after >= 0:
        return min(float(retry_after), float(policy.max_delay))
    return float(policy.calculate_delay(attempt))


async def run_model_turn_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    policy: Any,
    run_state: RunState,
    scope: str = "model_turn",
) -> T:
    """Run one model turn with policy-driven retry and RunState diagnostics."""
    max_attempts = policy.max_retries + 1
    last_error: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = await call()
            if attempt > 1:
                run_state.record_retry_event(
                    {
                        "scope": scope,
                        "model_turn": run_state.model_turn_count,
                        "iteration": run_state.iteration_count,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "decision": "succeeded_after_retry",
                    }
                )
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            last_error = error
            decision = classify_model_error(error)
            should_retry = decision.should_retry and attempt < max_attempts
            event = {
                "scope": scope,
                "model_turn": run_state.model_turn_count,
                "iteration": run_state.iteration_count,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "decision": (
                    "retry"
                    if should_retry
                    else "exhausted" if decision.should_retry else "do_not_retry"
                ),
                "reason": decision.reason,
                "classification": decision.classification,
                "exception_type": type(error).__name__,
            }
            if decision.retry_after is not None:
                event["retry_after_seconds"] = decision.retry_after

            if should_retry:
                delay = get_retry_delay(policy, attempt, error)
                event["delay_seconds"] = delay
                run_state.record_retry_event(event)
                await asyncio.sleep(delay)
                continue

            run_state.record_retry_event(event)
            raise

    raise last_error or RuntimeError("Unknown model retry failure")


def mark_whole_run_retry_suppressed(error: Exception, run_state: RunState) -> None:
    """Mark an exception so the outer whole-run retry scaffold will not replay it."""
    if run_state.tool_call_count <= 0:
        return
    unsafe_tool_calls = [
        call for call in run_state.tool_calls if not call.get("replay_safe", False)
    ]
    if not unsafe_tool_calls:
        return
    run_state.record_retry_event(
        {
            "scope": "whole_run",
            "decision": "suppressed",
            "reason": "committed tool work is not known replay-safe",
            "tool_call_count": run_state.tool_call_count,
            "unsafe_tool_count": len(unsafe_tool_calls),
        }
    )
    setattr(error, "_daita_suppress_whole_run_retry", True)
    setattr(error, "_daita_run_diagnostics", run_state.diagnostic_summary())


def _extract_retry_after(error: BaseException) -> Optional[float]:
    value = getattr(error, "retry_after", None)
    if value is None and isinstance(error, DaitaError):
        value = error.context.get("retry_after")
    if value is None and error.__cause__ is not None:
        value = getattr(error.__cause__, "retry_after", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_status_code(error: BaseException) -> Optional[int]:
    candidates = [error, getattr(error, "__cause__", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ("status_code", "status"):
            value = getattr(candidate, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    if isinstance(error, DaitaError):
        value = error.context.get("status_code")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    return None
