"""Budget assertions for cost, tokens, latency, and iterations."""

from __future__ import annotations

from ..analysis import RunEvidence
from ..config import EvalCaseConfig, Expectations, SuiteDefaults
from ..models import AssertionResult
from .common import fail


def budget_assertions(
    exp: Expectations,
    evidence: RunEvidence,
    case: EvalCaseConfig,
    defaults: SuiteDefaults,
) -> list[AssertionResult]:
    results = []
    max_cost = (
        exp.budgets.max_cost if exp.budgets.max_cost is not None else defaults.max_cost
    )
    if (
        max_cost is not None
        and evidence.metrics.cost is not None
        and evidence.metrics.cost > max_cost
    ):
        results.append(
            fail(
                "budgets.max_cost",
                "cost_over_budget",
                f"Cost {evidence.metrics.cost} exceeded budget {max_cost}.",
                "expectations.budgets.max_cost",
                observed=evidence.metrics.cost,
                expected=max_cost,
            )
        )
    if exp.budgets.max_tokens is not None and evidence.metrics.tokens_total is not None:
        if evidence.metrics.tokens_total > exp.budgets.max_tokens:
            results.append(
                fail(
                    "budgets.max_tokens",
                    "tokens_over_budget",
                    f"Tokens {evidence.metrics.tokens_total} exceeded budget {exp.budgets.max_tokens}.",
                    "expectations.budgets.max_tokens",
                    observed=evidence.metrics.tokens_total,
                    expected=exp.budgets.max_tokens,
                )
            )
    if (
        exp.budgets.max_latency_ms is not None
        and evidence.metrics.latency_ms is not None
    ):
        if evidence.metrics.latency_ms > exp.budgets.max_latency_ms:
            results.append(
                fail(
                    "budgets.max_latency_ms",
                    "latency_over_budget",
                    f"Latency {evidence.metrics.latency_ms}ms exceeded budget.",
                    "expectations.budgets.max_latency_ms",
                    observed=evidence.metrics.latency_ms,
                    expected=exp.budgets.max_latency_ms,
                )
            )
    max_iterations = (
        exp.budgets.max_iterations or case.max_iterations or defaults.max_iterations
    )
    if (
        evidence.metrics.iterations is not None
        and evidence.metrics.iterations > max_iterations
    ):
        results.append(
            fail(
                "budgets.max_iterations",
                "iterations_over_budget",
                f"Iterations {evidence.metrics.iterations} exceeded budget {max_iterations}.",
                "expectations.budgets.max_iterations",
                observed=evidence.metrics.iterations,
                expected=max_iterations,
            )
        )
    return results
