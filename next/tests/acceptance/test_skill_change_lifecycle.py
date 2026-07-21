from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from daita import Agent, SQLiteSource
from daita.learning import LearningProposalState
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from daita.loop.models import LoopExitKind
from daita.skills import (
    SkillActivationMode,
    SkillChangeCandidate,
    SkillSource,
)

NOW = datetime(2026, 7, 19, 14, 0, tzinfo=timezone.utc)
PROVIDER_ID = "mock:skill-change-lifecycle"
PROFILE = ModelProfile(
    id=PROVIDER_ID,
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
)


class JourneyProvider:
    provider_id = PROVIDER_ID

    def __init__(self) -> None:
        self.script: list[ModelResponse] = []
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("unexpected model call")
        return self.script.pop(0)


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
                name TEXT NOT NULL,
                status TEXT NOT NULL
            );
            INSERT INTO customers (name, status) VALUES
                ('Ada', 'complete'),
                ('Grace', 'complete'),
                ('Linus', 'pending');
            """)


async def test_public_skill_change_propose_accept_reopen_lifecycle(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    root = tmp_path / "state"
    provider = JourneyProvider()
    agent = await Agent.create(
        "atlas",
        root=root,
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    registration = await agent.attach(SQLiteSource(database, name="Customers"))
    provider.script.extend(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="call-source-query",
                        name="data_query_sqlite",
                        arguments={
                            "source_id": registration.id,
                            "sql": (
                                "SELECT COUNT(*) AS customer_count FROM customers "
                                "WHERE status = ?"
                            ),
                            "parameters": ["complete"],
                        },
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="There are 2 completed customers. [evidence:evidence-1]",
            ),
        )
    )
    source = await agent.run(
        "Determine the exact completed-customer query procedure.",
        session_id="session-skill-source",
    )
    assert source.kind is LoopExitKind.COMPLETED

    candidate = SkillChangeCandidate(
        stable_name="completed-customer-count",
        version="1.0.0",
        description="Count completed customers through accepted evidence.",
        instructions=(
            "Use data.sqlite.query with the exact stored value complete and cite "
            "the accepted evidence."
        ),
        activation_mode=SkillActivationMode.ALWAYS,
        domains=("data",),
        resource_kinds=("table",),
        required_capability_ids=("data.sqlite.query",),
        sensitivity_notes="Do not expose raw rows.",
    )
    proposed = await agent.propose_skill_change(
        source.operation_id,
        candidate,
    )

    assert proposed.proposal.state is LearningProposalState.PROPOSED
    assert proposed.proposal.provenance.operation_id == source.operation_id
    assert proposed.proposed_version is not None
    assert proposed.proposed_version.source is SkillSource.LEARNED_PROPOSAL
    assert await agent.list_skills() == ()

    accepted = await agent.accept_skill_change(
        proposed.proposal.id,
        expected_active_version_id=None,
        actor_id="user:owner",
        reason="Explicitly accept the reviewed procedure.",
    )
    skill_id = accepted.inspection.skill.id
    version_id = accepted.inspection.index.active_version_id
    assert version_id is not None
    assert accepted.proposal.state is LearningProposalState.COMMITTED
    assert accepted.proposal.result_skill_version == 1
    assert accepted.inspection.skill.source is SkillSource.LEARNED_PROPOSAL
    assert accepted.inspection.index.active_version_id == version_id
    assert len(accepted.inspection.versions) == 1
    assert len(accepted.inspection.activations) == 1

    await agent.close()
    reopened = await Agent.open("atlas", root=root, clock=lambda: NOW)
    recovered = await reopened.inspect_skill(skill_id)
    replay = await reopened.accept_skill_change(
        proposed.proposal.id,
        expected_active_version_id=None,
        actor_id="user:owner",
        reason="Idempotent acceptance retry.",
    )
    await reopened.close()

    assert recovered == accepted.inspection
    assert replay.replayed is True
    assert replay.proposal == accepted.proposal
    assert replay.inspection == recovered
