"""Run exit policy for Agent runtime stops and partial exits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExitDecision:
    """Internal decision for a run that cannot complete normally."""

    mode: str
    result_text: Optional[str] = None
    reason: str = "max_iterations"

    @property
    def should_return(self) -> bool:
        return self.mode in {"partial"}


class RunExitPolicy:
    """Conservative exit policy.

    Default behavior stays compatible: max iterations still raises. Partial
    return is opt-in and requires evidence from at least one successful tool.
    """

    def __init__(self, allow_partial: bool = False):
        self.allow_partial = allow_partial

    def decide_max_iterations(self, run_state, prompt: str) -> ExitDecision:
        if not self.allow_partial or not run_state.evidence:
            return ExitDecision(mode="raise", reason="max_iterations")
        return ExitDecision(
            mode="partial",
            reason="max_iterations_partial",
            result_text=self._partial_text(run_state, prompt),
        )

    def _partial_text(self, run_state, prompt: str) -> str:
        evidence_summaries = [
            _summarize_evidence(record) for record in run_state.evidence[-5:]
        ]
        lines = [
            "I could not complete the full run before hitting the iteration limit.",
            "Partial result based on evidence collected so far:",
        ]
        lines.extend(f"- {summary}" for summary in evidence_summaries if summary)
        if not evidence_summaries:
            lines.append("- No usable evidence was collected.")
        lines.append(f"Original request: {prompt}")
        return "\n".join(lines)


def _summarize_evidence(record) -> str:
    if record.domain == "generic" and record.kind == "tool_result":
        return _summarize_tool_result(record.source_tool, record.payload)
    return f"{record.domain}:{record.kind} from {record.source_tool or 'unknown'}"


def _summarize_tool_result(tool_name: Optional[str], payload: Dict[str, Any]) -> str:
    result = payload.get("result")
    arguments = payload.get("arguments", {})
    if isinstance(result, dict):
        compact_result = _compact_mapping(result)
    else:
        compact_result = repr(result)
    return f"{tool_name or 'tool'}({arguments}) returned {compact_result}"


def _compact_mapping(value: Dict[str, Any]) -> str:
    items: List[str] = []
    for key, val in list(value.items())[:5]:
        items.append(f"{key}={val!r}")
    suffix = ", ..." if len(value) > 5 else ""
    return "{" + ", ".join(items) + suffix + "}"
