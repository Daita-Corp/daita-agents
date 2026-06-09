"""Deterministic eval assertions."""

from __future__ import annotations

from ..analysis import RunEvidence
from ..config import EvalCaseConfig, SuiteDefaults
from ..models import AssertionResult, StabilitySummary
from .answer import answer_assertions
from .budgets import budget_assertions
from .common import has_errors
from .runtime import (
    approval_assertions,
    capability_assertions,
    evidence_assertions,
    governance_assertions,
    task_assertions,
)
from .results import result_assertions
from .sql import sql_assertions
from .stability import stability_assertions


def evaluate_run_assertions(
    case: EvalCaseConfig,
    evidence: RunEvidence,
    defaults: SuiteDefaults,
) -> list[AssertionResult]:
    expectations = case.expectations
    results: list[AssertionResult] = []
    results.extend(answer_assertions(expectations, evidence))
    results.extend(capability_assertions(expectations, evidence, defaults))
    results.extend(task_assertions(expectations, evidence))
    results.extend(evidence_assertions(expectations, evidence))
    results.extend(result_assertions(expectations, evidence))
    results.extend(governance_assertions(expectations, evidence))
    results.extend(approval_assertions(expectations, evidence))
    results.extend(budget_assertions(expectations, evidence, case, defaults))
    results.extend(sql_assertions(expectations, evidence))
    return results


def evaluate_stability_assertions(
    case: EvalCaseConfig,
    stability: StabilitySummary,
) -> list[AssertionResult]:
    return stability_assertions(case, stability)


__all__ = [
    "evaluate_run_assertions",
    "evaluate_stability_assertions",
    "has_errors",
]
