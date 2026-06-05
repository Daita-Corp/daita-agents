"""Transient state for a single Agent.run() invocation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4


class RunPhase(str, Enum):
    INITIALIZING = "initializing"
    MODEL_TURN = "model_turn"
    TOOL_EXECUTION = "tool_execution"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class EvidenceRecord:
    evidence_id: str
    domain: str
    kind: str
    source_tool: Optional[str] = None
    confidence: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class FinalAnswerReadiness:
    """Domain-neutral decision about whether a model response may finalize."""

    allow_final: bool = True
    guidance: Optional[str] = None
    warning: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


FinalAnswerReadinessHook = Callable[
    ["RunState", str, List[Any]], Optional[FinalAnswerReadiness]
]


@dataclass
class RunState:
    """Generic per-run execution facts and diagnostics.

    This state is not durable memory. It exists to coordinate one model/tool
    loop, expose diagnostics, and give tools a task-local place for attachments.
    """

    agent_id: str
    run_id: str = field(default_factory=lambda: f"run_{uuid4().hex}")
    started_at: float = field(default_factory=time.time)
    phase: RunPhase = RunPhase.INITIALIZING
    iteration_count: int = 0
    model_turn_count: int = 0
    tool_call_count: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    failed_tool_fingerprints: Dict[str, int] = field(default_factory=dict)
    repeated_result_fingerprints: Dict[str, int] = field(default_factory=dict)
    last_progress_marker: Optional[str] = None
    progress_events: List[Dict[str, Any]] = field(default_factory=list)
    exit_reason: Optional[str] = None
    partial_result: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    retry_events: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceRecord] = field(default_factory=list)
    domains: Dict[str, Any] = field(default_factory=dict)
    final_answer_readiness_hooks: List[FinalAnswerReadinessHook] = field(
        default_factory=list,
        repr=False,
    )

    def set_phase(self, phase: RunPhase) -> None:
        self.phase = phase

    def record_progress(self, marker: str, **metadata: Any) -> None:
        self.last_progress_marker = marker
        self.progress_events.append(
            {"marker": marker, "timestamp": time.time(), **metadata}
        )

    def record_tool_call(self, result: Dict[str, Any]) -> None:
        self.tool_call_count += 1
        self.tool_calls.append(result)

    def add_evidence(self, record: EvidenceRecord) -> None:
        self.evidence.append(record)

    def record_retry_event(self, event: Dict[str, Any]) -> None:
        self.retry_events.append({"timestamp": time.time(), **event})

    def diagnostic_summary(self) -> Dict[str, Any]:
        retry_summary: Dict[str, Any] = {
            "retry_event_count": len(self.retry_events),
            "retry_events": _compact_retry_events(self.retry_events),
        }
        if self.retry_events:
            retry_summary["last_retry_decision"] = self.retry_events[-1].get("decision")
        exhausted = next(
            (
                event
                for event in reversed(self.retry_events)
                if event.get("decision") == "exhausted"
            ),
            None,
        )
        if exhausted is not None:
            retry_summary["retry_exhaustion_reason"] = exhausted.get("reason")
        replay_suppressed = any(
            event.get("decision") == "suppressed" and event.get("scope") == "whole_run"
            for event in self.retry_events
        )
        if replay_suppressed:
            retry_summary["whole_run_retry_suppressed"] = True

        return {
            "run_id": self.run_id,
            "phase": self.phase.value,
            "iteration_count": self.iteration_count,
            "model_turn_count": self.model_turn_count,
            "tool_call_count": self.tool_call_count,
            "exit_reason": self.exit_reason,
            "warnings": list(self.warnings),
            "progress_event_count": len(self.progress_events),
            "evidence_count": len(self.evidence),
            "domains": sorted(self.domains.keys()),
            **retry_summary,
            **self.diagnostics,
        }


def _compact_retry_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return stable, bounded retry diagnostics for result payloads."""
    compact_keys = (
        "scope",
        "model_turn",
        "iteration",
        "attempt",
        "max_attempts",
        "decision",
        "reason",
        "classification",
        "exception_type",
        "delay_seconds",
        "retry_after_seconds",
        "tool_call_count",
        "unsafe_tool_count",
    )
    return [
        {key: event[key] for key in compact_keys if key in event}
        for event in events[-20:]
    ]
