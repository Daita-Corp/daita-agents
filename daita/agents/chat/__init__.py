"""Internal chat machinery for the secondary generic Agent.

This package is intentionally private to the agent layer. Public lifecycle
behavior still lives on BaseAgent and Agent.
"""

from .contextvars import active_run_state, get_active_run_state
from .evidence import add_active_evidence, add_evidence
from .exit import ExitDecision, RunExitPolicy
from .facade import ChatAgentFacadeMixin
from .runtime import ChatRunResult, ChatRuntime, ChatToolCallResult, ModelToolSpec
from .state import EvidenceRecord, RunPhase, RunState

__all__ = [
    "ChatAgentFacadeMixin",
    "ChatRunResult",
    "ChatRuntime",
    "ChatToolCallResult",
    "EvidenceRecord",
    "ExitDecision",
    "ModelToolSpec",
    "RunPhase",
    "RunExitPolicy",
    "RunState",
    "active_run_state",
    "add_active_evidence",
    "add_evidence",
    "get_active_run_state",
]
