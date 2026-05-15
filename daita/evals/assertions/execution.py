"""Skill and plugin execution assertions."""

from __future__ import annotations

from ..analysis import RunEvidence
from ..config import ExecutionExpectations, Expectations
from ..models import AssertionResult, ExecutionSpan
from .common import fail, matches, matches_any


def execution_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    results: list[AssertionResult] = []
    results.extend(_span_assertions("skill", "skills", exp.skills, evidence))
    results.extend(_span_assertions("plugin", "plugins", exp.plugins, evidence))
    return results


def _span_assertions(
    kind: str,
    config_key: str,
    exp: ExecutionExpectations,
    evidence: RunEvidence,
) -> list[AssertionResult]:
    spans = [span for span in evidence.execution_spans if span.kind == kind]
    names = [span.name for span in spans]
    return [
        *_required_assertions(kind, config_key, exp.required, names),
        *_forbidden_assertions(kind, config_key, exp.forbidden, names),
        *_max_call_assertions(kind, config_key, exp.max_calls, spans),
        *_latency_assertions(kind, config_key, exp.max_latency_ms, spans),
        *_error_assertions(kind, config_key, exp.max_errors, spans),
    ]


def _required_assertions(
    kind: str, config_key: str, required: list[str], names: list[str]
) -> list[AssertionResult]:
    return [
        fail(
            f"{config_key}.required[{index}]",
            f"required_{kind}_missing",
            f"Required {kind} was not used: {name}.",
            f"expectations.{config_key}.required[{index}]",
            observed=names,
            expected=name,
        )
        for index, name in enumerate(required)
        if not matches_any(name, names)
    ]


def _forbidden_assertions(
    kind: str, config_key: str, forbidden: list[str], names: list[str]
) -> list[AssertionResult]:
    return [
        fail(
            f"{config_key}.forbidden[{index}]",
            f"forbidden_{kind}_called",
            f"Forbidden {kind} was used: {name}.",
            f"expectations.{config_key}.forbidden[{index}]",
            observed=name,
            expected=f"not {name}",
        )
        for index, name in enumerate(forbidden)
        if any(matches(name, observed) for observed in names)
    ]


def _max_call_assertions(
    kind: str,
    config_key: str,
    max_calls: int | None,
    spans: list[ExecutionSpan],
) -> list[AssertionResult]:
    if max_calls is None or len(spans) <= max_calls:
        return []
    return [
        fail(
            f"{config_key}.max_calls",
            f"too_many_{kind}_calls",
            f"Too many {kind} calls were observed.",
            f"expectations.{config_key}.max_calls",
            observed=len(spans),
            expected=max_calls,
        )
    ]


def _latency_assertions(
    kind: str,
    config_key: str,
    max_latency_ms: float | None,
    spans: list[ExecutionSpan],
) -> list[AssertionResult]:
    if max_latency_ms is None:
        return []
    offenders = [
        span
        for span in spans
        if span.latency_ms is not None and span.latency_ms > max_latency_ms
    ]
    if not offenders:
        return []
    return [
        fail(
            f"{config_key}.max_latency_ms",
            f"{kind}_latency_over_budget",
            f"{kind.title()} latency exceeded the configured maximum.",
            f"expectations.{config_key}.max_latency_ms",
            observed=_span_latency_observed(offenders),
            expected=max_latency_ms,
        )
    ]


def _error_assertions(
    kind: str,
    config_key: str,
    max_errors: int | None,
    spans: list[ExecutionSpan],
) -> list[AssertionResult]:
    if max_errors is None:
        return []
    errors = [_span_error(span) for span in spans if _span_error(span)]
    if len(errors) <= max_errors:
        return []
    return [
        fail(
            f"{config_key}.max_errors",
            f"too_many_{kind}_errors",
            f"Too many {kind} errors were observed.",
            f"expectations.{config_key}.max_errors",
            observed=errors,
            expected=max_errors,
        )
    ]


def _span_latency_observed(spans: list[ExecutionSpan]) -> list[dict[str, float | str]]:
    return [{"name": span.name, "latency_ms": span.latency_ms or 0.0} for span in spans]


def _span_error(span: ExecutionSpan) -> str:
    status = span.status.lower()
    if span.error:
        return f"{span.name}: {span.error}"
    if status in {"failed", "error", "errored"}:
        return f"{span.name}: {span.status}"
    return ""
