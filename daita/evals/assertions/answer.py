"""Answer text and numeric assertions."""

from __future__ import annotations

import re


from ..analysis import RunEvidence
from ..config import Expectations
from ..models import AssertionResult
from .common import fail


def answer_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    answer = evidence.answer
    results = []
    if exp.answer.equals is not None and answer != exp.answer.equals:
        results.append(
            fail(
                "answer.equals",
                "answer_mismatch",
                "Answer did not equal expected text.",
                "expectations.answer.equals",
                observed=answer,
                expected=exp.answer.equals,
            )
        )
    for index, value in enumerate(exp.answer.contains):
        if value not in answer:
            results.append(
                fail(
                    f"answer.contains[{index}]",
                    "missing_text",
                    f"Answer did not contain expected text: {value}.",
                    f"expectations.answer.contains[{index}]",
                    observed=answer,
                    expected=value,
                )
            )
    for index, value in enumerate(exp.answer.not_contains):
        if value in answer:
            results.append(
                fail(
                    f"answer.not_contains[{index}]",
                    "forbidden_text",
                    f"Answer contained forbidden text: {value}.",
                    f"expectations.answer.not_contains[{index}]",
                    observed=value,
                    expected=f"not {value}",
                )
            )
    for index, pattern in enumerate(exp.answer.regex):
        if not re.search(pattern, answer):
            results.append(
                fail(
                    f"answer.regex[{index}]",
                    "regex_no_match",
                    f"Answer did not match regex: {pattern}.",
                    f"expectations.answer.regex[{index}]",
                    observed=answer,
                    expected=pattern,
                )
            )
    for index, numeric in enumerate(exp.answer.numeric):
        observed = extract_number_near_label(answer, numeric.label)
        if observed is None or abs(observed - numeric.expected) > numeric.tolerance:
            results.append(
                fail(
                    f"answer.numeric[{index}]",
                    "numeric_mismatch",
                    f"Expected {numeric.label} to be {numeric.expected} +/- {numeric.tolerance}.",
                    f"expectations.answer.numeric[{index}]",
                    observed=observed,
                    expected=numeric.expected,
                    fix_hints=[
                        "Inspect aggregation logic and units in accepted runtime evidence."
                    ],
                )
            )
    return results


def extract_number_near_label(answer: str, label: str) -> float | None:
    label_re = re.escape(label)
    patterns = (
        rf"{label_re}[^\n\d\-+]*([-+]?\d[\d,]*(?:\.\d+)?)",
        rf"([-+]?\d[\d,]*(?:\.\d+)?)[^\n]{{0,80}}{label_re}",
    )
    for line in answer.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line, re.I)
            if match:
                return float(match.group(1).replace(",", ""))
    return None
