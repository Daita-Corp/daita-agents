"""Shared per-run evidence helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .contextvars import get_active_run_state
from .state import EvidenceRecord, RunState


def add_evidence(
    run_state: RunState,
    *,
    domain: str,
    kind: str,
    source_tool: Optional[str] = None,
    confidence: Optional[float] = None,
    payload: Optional[Dict[str, Any]] = None,
    notes: Optional[list[str]] = None,
) -> EvidenceRecord:
    """Append an evidence record to a run and return it."""
    record = EvidenceRecord(
        evidence_id=f"evidence_{len(run_state.evidence) + 1}",
        domain=domain,
        kind=kind,
        source_tool=source_tool,
        confidence=confidence,
        payload=dict(payload or {}),
        notes=list(notes or []),
    )
    run_state.add_evidence(record)
    return record


def add_active_evidence(
    *,
    domain: str,
    kind: str,
    source_tool: Optional[str] = None,
    confidence: Optional[float] = None,
    payload: Optional[Dict[str, Any]] = None,
    notes: Optional[list[str]] = None,
) -> Optional[EvidenceRecord]:
    """Append evidence to the active run state, when one exists."""
    run_state = get_active_run_state()
    if run_state is None:
        return None
    return add_evidence(
        run_state,
        domain=domain,
        kind=kind,
        source_tool=source_tool,
        confidence=confidence,
        payload=payload,
        notes=notes,
    )
