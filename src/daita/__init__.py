"""Daita agent."""

from .adapters import LocalDirectorySource, PostgreSQLSource, SQLiteSource
from .agent import Agent
from .catalog import CatalogSummary
from .capabilities import ApprovalDecision, ApprovalHandler, ApprovalRequest
from .config import AgentConfig
from .llm import (
    ModelRoute,
    ModelRouteCandidate,
    RetryPolicy,
    create_llm_provider,
)
from .loop import ConversationRun, LoopExit, LoopExitKind, LoopLimits, Transcript
from .observation import AgentEvent, AgentEventKind, AgentObserver
from .skills import Skill, SkillSummary

__version__ = "2.0.0a0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "AgentEventKind",
    "AgentObserver",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalRequest",
    "ConversationRun",
    "CatalogSummary",
    "LocalDirectorySource",
    "LoopExit",
    "LoopExitKind",
    "LoopLimits",
    "ModelRoute",
    "ModelRouteCandidate",
    "PostgreSQLSource",
    "RetryPolicy",
    "SQLiteSource",
    "Skill",
    "SkillSummary",
    "Transcript",
    "__version__",
    "create_llm_provider",
]
