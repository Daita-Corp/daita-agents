"""Admit only the selected transient learning candidate for foreground mutation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..capabilities import CapabilityInputError
from ..llm.models import ToolCall

if TYPE_CHECKING:
    from ..learning_candidates import LearningCandidate
    from ..loop.session import RunSession

_MEMORY_SET_TOOL_NAME = "memory_set"
_SKILL_SAVE_TOOL_NAME = "skill_save"
_SKILL_DELETE_TOOL_NAME = "skill_delete"
_SEMANTIC_SAVE_TOOL_NAME = "semantic_save"
_SEMANTIC_DELETE_TOOL_NAME = "semantic_delete"


class LearningCandidateGuard:
    """Evaluate the candidate carried by one immutable run session."""

    def selected(self, session: RunSession | None) -> LearningCandidate | None:
        if session is None:
            return None
        from ..learning_candidates import LearningCandidate

        selected = session.options.learning_candidate
        if selected is not None and not isinstance(selected, LearningCandidate):
            raise TypeError("session learning candidate is invalid")
        return selected

    def allows(
        self,
        session: RunSession | None,
        tool_name: str,
        *,
        effectful: bool,
    ) -> bool:
        selected = self.selected(session)
        return (
            selected is None
            or not effectful
            or tool_name == _candidate_mutation_tool(selected)
        )

    def selected_mutation_tool(self, session: RunSession | None) -> str | None:
        selected = self.selected(session)
        return None if selected is None else _candidate_mutation_tool(selected)

    def validate_effect(self, session: RunSession | None, call: ToolCall) -> None:
        from ..learning_candidates import candidate_matches_mutation_call

        selected = self.selected(session)
        if selected is not None and not candidate_matches_mutation_call(selected, call):
            raise CapabilityInputError(
                "candidate_mismatch",
                "This acceptance run may mutate only the explicitly selected "
                "candidate's exact target content.",
                {"candidate_id": selected.id},
            )

    def mark_effect_succeeded(
        self,
        session: RunSession | None,
        call_id: str,
    ) -> None:
        if self.selected(session) is not None:
            assert session is not None
            session.evidence.record_learning_mutation(call_id)


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
