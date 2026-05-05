"""Repeat-run stability assertions."""

from __future__ import annotations


from ..analysis import metric_delta_pct
from ..config import EvalCaseConfig
from ..models import AssertionResult, StabilitySummary
from .common import fail


def stability_assertions(
    case: EvalCaseConfig,
    stability: StabilitySummary,
) -> list[AssertionResult]:
    exp = case.expectations.stability
    results: list[AssertionResult] = []
    if exp.require_same_tools and stability.tool_sequence_variants > 1:
        results.append(
            fail(
                "stability.require_same_tools",
                "unstable_tools",
                f"{stability.tool_sequence_variants} tool sequences observed.",
                "expectations.stability.require_same_tools",
                observed=stability.tool_sequence_variants,
                expected=1,
            )
        )
    if exp.max_answer_variants and stability.answer_variants > exp.max_answer_variants:
        results.append(
            fail(
                "stability.max_answer_variants",
                "unstable_answer",
                f"{stability.answer_variants} answer variants observed.",
                "expectations.stability.max_answer_variants",
                observed=stability.answer_variants,
                expected=exp.max_answer_variants,
            )
        )
    _delta_check(
        results, "cost", stability.cost_min, stability.cost_max, exp.max_cost_delta_pct
    )
    _delta_check(
        results,
        "latency_ms",
        stability.latency_ms_min,
        stability.latency_ms_max,
        exp.max_latency_delta_pct,
    )
    _delta_check(
        results,
        "token",
        stability.token_min,
        stability.token_max,
        exp.max_token_delta_pct,
    )
    return results


def _delta_check(
    results: list[AssertionResult],
    metric: str,
    min_value: float | int | None,
    max_value: float | int | None,
    allowed: float | None,
) -> None:
    if allowed is None:
        return
    observed = metric_delta_pct(min_value, max_value)
    if observed > allowed:
        results.append(
            fail(
                f"stability.max_{metric}_delta_pct",
                f"unstable_{metric}",
                f"{metric} varied by {observed:.1f}%.",
                f"expectations.stability.max_{metric}_delta_pct",
                observed=observed,
                expected=allowed,
            )
        )
