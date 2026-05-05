"""Tool usage assertions."""

from __future__ import annotations

from ..analysis import RunEvidence
from ..config import Expectations, SuiteDefaults
from ..models import AssertionResult
from .common import fail


def tool_assertions(
    exp: Expectations, evidence: RunEvidence, defaults: SuiteDefaults
) -> list[AssertionResult]:
    names = [call.name for call in evidence.tool_calls]
    results = []
    for index, name in enumerate(exp.tools.required):
        if name not in names:
            results.append(
                fail(
                    f"tools.required[{index}]",
                    "required_tool_missing",
                    f"Required tool was not called: {name}.",
                    f"expectations.tools.required[{index}]",
                    observed=names,
                    expected=name,
                )
            )
    for index, name in enumerate(exp.tools.forbidden):
        if name in names:
            call_indexes = [i for i, call_name in enumerate(names) if call_name == name]
            results.append(
                fail(
                    f"tools.forbidden[{index}]",
                    "forbidden_tool_called",
                    f"Forbidden tool was called: {name}.",
                    f"expectations.tools.forbidden[{index}]",
                    observed=name,
                    expected=f"not {name}",
                    related_tool_calls=call_indexes,
                )
            )
    max_calls = (
        exp.tools.max_calls
        if exp.tools.max_calls is not None
        else defaults.max_tool_calls
    )
    if max_calls is not None and len(names) > max_calls:
        results.append(
            fail(
                "tools.max_calls",
                "too_many_tool_calls",
                f"Expected at most {max_calls} tool calls, got {len(names)}.",
                "expectations.tools.max_calls",
                observed=len(names),
                expected=max_calls,
            )
        )
    return results
