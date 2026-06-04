"""Internal runtime machinery for Agent.run().

This package is intentionally private to the agent layer. Public lifecycle
behavior still lives on BaseAgent and Agent.
"""

from .contextvars import active_run_state, get_active_run_state
from .evidence import add_active_evidence, add_evidence
from .exit import ExitDecision, RunExitPolicy
from .state import EvidenceRecord, RunPhase, RunState

__all__ = [
    "EvidenceRecord",
    "ExitDecision",
    "RunPhase",
    "RunExitPolicy",
    "RunState",
    "active_run_state",
    "add_active_evidence",
    "add_evidence",
    "get_active_run_state",
]
