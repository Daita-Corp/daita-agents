from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from daita import Agent, LearningProposalState, SQLiteSource
from daita.learning import (
    LearningCandidateCategory,
    LearningRejectionCategory,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.loop.models import Readiness, Turn
from daita.memory import MemoryListRequest, MemoryScope
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import ActionProposal, Evidence, Observation
from daita.skills import SkillSource

NOW = datetime(2026, 7, 19, 19, 0, tzinfo=timezone.utc)
PROFILE = ModelProfile(
    id="mock:phase-9-5-learning",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
)


class ScriptedProvider:
    provider_id = PROFILE.id

    def __init__(self) -> None:
        self.responses: list[ModelResponse] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        if not self.responses:
            raise AssertionError("unexpected learning model call")
        return self.responses.pop(0)


class TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class TextDomain:
    async def tool_views(
        self, operation: OperationSnapshot
    ) -> tuple[ToolDefinition, ...]:
        del operation
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        del call, operation
        raise AssertionError("text-only domain has no actions")

    async def project_observation(self, evidence: Evidence) -> Observation:
        del evidence
        raise AssertionError("text-only domain has no observations")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        del text, operation
        return Readiness(
            allowed=True,
            code="ready",
            message="The response is ready.",
            evaluated_at=NOW,
        )


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL
            );
            INSERT INTO customers (status) VALUES ('active'), ('inactive');
            """)


def _query(call_id: str, source_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id=call_id,
                name="data_query_sqlite",
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT COUNT(*) AS total FROM customers",
                },
            ),
        ),
    )


def _answer(text: str, evidence_id: str | None = None) -> ModelResponse:
    suffix = "" if evidence_id is None else f" [evidence:{evidence_id}]"
    return ModelResponse(text=text + suffix, finish_reason=FinishReason.STOP)


async def test_natural_fact_and_skill_proposals_are_visible_safe_and_inert(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    root = tmp_path / "fact-state"
    provider = ScriptedProvider()
    agent = await Agent.create(
        "natural-learning",
        root=root,
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    registration = await agent.attach(SQLiteSource(database))
    provider.responses.extend(
        (
            _query("fact-query", registration.id),
            _answer("The count is two.", "evidence-1"),
            _query("pii-query", registration.id),
            _answer("The count remains two.", "evidence-2"),
        )
    )

    fact = await agent.run(
        "Propose a fact from the evidence: Customer count is two.",
        session_id="fact-current",
    )
    fact_proposals = await agent.list_learning_proposals(operation_id=fact.operation_id)
    assert fact.post_operation_notices == ("learning.fact_proposed",)
    assert len(fact_proposals) == 1
    fact_proposal = fact_proposals[0]
    assert fact_proposal.state is LearningProposalState.PROPOSED
    assert fact_proposal.category is LearningCandidateCategory.EVIDENCE_BACKED_FACT
    assert fact_proposal.provenance.evidence_id == "evidence-1"
    assert fact_proposal.provenance.evidence_accepted is True
    assert fact_proposal.candidate_payload is not None
    assert fact_proposal.candidate_payload["source_ids"] == (registration.id,)
    assert (
        await agent.list_memories(
            MemoryListRequest(scope=MemoryScope(agent_id=agent.id))
        )
    ).items == ()

    private_value = "ada@example.com"
    pii = await agent.run(
        f"Propose a fact from the evidence: The contact is {private_value}.",
        session_id="fact-pii",
    )
    pii_proposal = (await agent.list_learning_proposals(operation_id=pii.operation_id))[
        0
    ]
    assert pii.post_operation_notices == ("learning.fact_rejected",)
    assert pii_proposal.state is LearningProposalState.REJECTED
    assert pii_proposal.rejection_category is LearningRejectionCategory.RAW_ROW_OR_PII
    assert pii_proposal.candidate_payload is None
    assert private_value not in repr(pii_proposal)

    await agent.close()
    reopened_facts = await Agent.open("natural-learning", root=root, clock=lambda: NOW)
    try:
        durable_facts = await reopened_facts.list_learning_proposals()
        assert {proposal.id for proposal in durable_facts} == {
            fact_proposal.id,
            pii_proposal.id,
        }
        assert (
            await reopened_facts.list_memories(
                MemoryListRequest(scope=MemoryScope(agent_id=reopened_facts.id))
            )
        ).items == ()
    finally:
        await reopened_facts.close()

    skill_root = tmp_path / "skill-state"
    skill_provider = ScriptedProvider()
    skill_provider.responses.extend(
        (
            _answer("I prepared the procedure for review."),
            _answer("I rejected the unsafe procedure."),
        )
    )
    agent = await Agent.create(
        "natural-skills",
        root=skill_root,
        model=skill_provider,
        context_builder=TextContext(),
        domain=TextDomain(),
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    skill = await agent.run(
        "Propose a skill named reconcile-status: "
        "Compare accepted totals and cite discrepancies.",
        session_id="skill-safe",
    )
    skill_proposal = (
        await agent.list_learning_proposals(operation_id=skill.operation_id)
    )[0]
    assert skill.post_operation_notices == ("learning.skill_proposed",)
    assert skill_proposal.state is LearningProposalState.PROPOSED
    assert skill_proposal.category is LearningCandidateCategory.SKILL_CHANGE
    assert await agent.list_skills() == ()

    unsafe_instructions = "import os and call subprocess.run('unsafe')"
    unsafe = await agent.run(
        "Propose a skill named unsafe-runner: " + unsafe_instructions,
        session_id="skill-unsafe",
    )
    unsafe_proposal = (
        await agent.list_learning_proposals(operation_id=unsafe.operation_id)
    )[0]
    assert unsafe.post_operation_notices == ("learning.skill_rejected",)
    assert unsafe_proposal.state is LearningProposalState.REJECTED
    assert unsafe_proposal.rejection_category is (
        LearningRejectionCategory.EXECUTABLE_OR_RUNTIME_EFFECT
    )
    assert unsafe_proposal.candidate_payload is None
    assert unsafe_instructions not in repr(unsafe_proposal)
    assert await agent.list_skills() == ()

    await agent.close()
    reopened = await Agent.open("natural-skills", root=skill_root, clock=lambda: NOW)
    try:
        durable = await reopened.list_learning_proposals()
        assert {proposal.id for proposal in durable} == {
            skill_proposal.id,
            unsafe_proposal.id,
        }
        assert await reopened.list_skills() == ()
        accepted = await reopened.accept_skill_change(
            skill_proposal.id,
            expected_active_version_id=None,
            actor_id="user:owner",
            reason="The bounded procedure was reviewed and approved.",
        )
        assert accepted.proposal.state is LearningProposalState.COMMITTED
        assert accepted.inspection.skill.source is SkillSource.LEARNED_PROPOSAL
        assert accepted.inspection.index.active_version_id is not None
        assert len(await reopened.list_skills()) == 1
    finally:
        await reopened.close()

    reopened_again = await Agent.open(
        "natural-skills",
        root=skill_root,
        clock=lambda: NOW,
    )
    try:
        committed = await reopened_again.list_learning_proposals(
            operation_id=skill.operation_id,
            states=(LearningProposalState.COMMITTED,),
        )
        assert len(committed) == 1
        inspection = await reopened_again.inspect_skill(
            committed[0].result_skill_id or ""
        )
        assert inspection.index.active_version_id is not None
        assert len(inspection.activations) == 1
        assert inspection.activations[0].actor_id == "user:owner"
    finally:
        await reopened_again.close()
