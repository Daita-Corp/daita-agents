"""Deterministic eval assertions."""

from __future__ import annotations

from ..analysis import RunEvidence
from ..config import EvalCaseConfig, SuiteDefaults
from ..models import AssertionResult, StabilitySummary
from .answer import answer_assertions
from .budgets import budget_assertions
from .common import has_errors
from .execution import execution_assertions
from .operations import (
    api_assertions,
    file_assertions,
    operation_assertions,
    storage_assertions,
    vector_assertions,
)
from .sql import sql_assertions
from .stability import stability_assertions
from .tools import tool_assertions


def evaluate_run_assertions(
    case: EvalCaseConfig,
    evidence: RunEvidence,
    defaults: SuiteDefaults,
) -> list[AssertionResult]:
    expectations = case.expectations
    results: list[AssertionResult] = []
    results.extend(answer_assertions(expectations, evidence))
    results.extend(tool_assertions(expectations, evidence, defaults))
    results.extend(budget_assertions(expectations, evidence, case, defaults))
    results.extend(sql_assertions(expectations, evidence))
    results.extend(execution_assertions(expectations, evidence))
    results.extend(operation_assertions(expectations, evidence))
    results.extend(file_assertions(expectations, evidence))
    results.extend(api_assertions(expectations, evidence))
    results.extend(storage_assertions(expectations, evidence))
    results.extend(vector_assertions(expectations, evidence))
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
