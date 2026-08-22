"""Transient exact-candidate admission for ordinary foreground mutations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..capabilities import CapabilityInputError
from ..llm.models import ToolCall

if TYPE_CHECKING:
    from ..learning_candidates import LearningCandidate

_MEMORY_SET_TOOL_NAME = "memory_set"
_SKILL_SAVE_TOOL_NAME = "skill_save"
_SKILL_DELETE_TOOL_NAME = "skill_delete"
_SEMANTIC_SAVE_TOOL_NAME = "semantic_save"
_SEMANTIC_DELETE_TOOL_NAME = "semantic_delete"


class LearningCandidateGuard:
    """Own the one live candidate selection and its exact success outcome."""

    def __init__(self) -> None:
        self._selected: dict[str, LearningCandidate] = {}
        self._successful: set[str] = set()

    def select(self, run_id: str, candidate: LearningCandidate) -> None:
        from ..learning_candidates import LearningCandidate

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("candidate guard run_id must be non-empty text")
        if not isinstance(candidate, LearningCandidate):
            raise TypeError("candidate guard requires LearningCandidate")
        if self._selected:
            raise RuntimeError("candidate mutation guard exceeds its live bound")
        self._successful.discard(run_id)
        self._selected[run_id] = candidate

    def clear(self, run_id: str) -> None:
        self._selected.pop(run_id, None)

    def mutation_succeeded(self, run_id: str) -> bool:
        return run_id in self._successful

    def clear_outcome(self, run_id: str) -> None:
        self._successful.discard(run_id)

    def allows(self, run_id: str, tool_name: str, *, effectful: bool) -> bool:
        selected = self._selected.get(run_id)
        return (
            selected is None
            or not effectful
            or tool_name == _candidate_mutation_tool(selected)
        )

    def selected_mutation_tool(self, run_id: str) -> str | None:
        selected = self._selected.get(run_id)
        return None if selected is None else _candidate_mutation_tool(selected)

    def validate_effect(self, run_id: str, call: ToolCall) -> None:
        from ..learning_candidates import candidate_matches_mutation_call

        selected = self._selected.get(run_id)
        if selected is not None and not candidate_matches_mutation_call(selected, call):
            raise CapabilityInputError(
                "candidate_mismatch",
                "This acceptance run may mutate only the explicitly selected "
                "candidate's exact target content.",
                {"candidate_id": selected.id},
            )

    def mark_effect_succeeded(self, run_id: str) -> None:
        if run_id in self._selected:
            self._successful.add(run_id)


def _candidate_mutation_tool(candidate: LearningCandidate) -> str:
    from ..learning_candidates import (
        LearningCandidateAction,
        LearningCandidateTarget,
        SemanticCandidateContent,
        SkillCandidateContent,
    )

    if candidate.target in {
        LearningCandidateTarget.MEMORY,
        LearningCandidateTarget.USER,
    }:
        return _MEMORY_SET_TOOL_NAME
    if candidate.target is LearningCandidateTarget.SKILL:
        skill_content = cast(SkillCandidateContent, candidate.content)
        return (
            _SKILL_DELETE_TOOL_NAME
            if skill_content.action is LearningCandidateAction.DELETE
            else _SKILL_SAVE_TOOL_NAME
        )
    semantic_content = cast(SemanticCandidateContent, candidate.content)
    return (
        _SEMANTIC_DELETE_TOOL_NAME
        if semantic_content.action is LearningCandidateAction.DELETE
        else _SEMANTIC_SAVE_TOOL_NAME
    )


__all__ = ["LearningCandidateGuard"]
