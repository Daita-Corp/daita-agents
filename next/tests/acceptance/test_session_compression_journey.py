from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject
from daita.catalog import ResourceKind, catalog_resource_id
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.loop.models import LoopExitKind

NOW = datetime(2026, 7, 18, 22, 0, tzinfo=timezone.utc)
SESSION_ID = "session-compression"


class CompressionJourneyProvider:
    provider_id = "mock:compression-journey"

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
            CREATE TABLE facts (value TEXT NOT NULL);
            INSERT INTO facts (value) VALUES ('stable');
            """)


def _intent(position: int) -> str:
    return f"Journal request {position}: " + ("retain-intent " * 65)


def _answer(position: int) -> str:
    return (
        f"Journal answer {position}: "
        + ("durable-detail " * 260)
        + f"[evidence:evidence-{position}]"
    )


def _selected_records(request: ModelRequest) -> tuple[FrozenJsonObject, ...]:
    selected = request.context_selection["selected_blocks"]
    assert isinstance(selected, tuple)
    assert all(isinstance(item, FrozenJsonObject) for item in selected)
    return selected


def _message_texts(request: ModelRequest) -> tuple[str, ...]:
    return tuple(
        block.text
        for message in request.messages
        for block in message.content
        if isinstance(block, TextBlock)
    )


async def test_public_session_compression_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / "facts.db"
    _database(database)
    state_root = tmp_path / "state"
    provider = CompressionJourneyProvider()
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=9_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "atlas",
        root=state_root,
        model=provider,
        model_profile=profile,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    operation_ids: list[str] = []
    prompts: list[str] = []
    answers: list[str] = []
    later_request: ModelRequest
    table_resource_id: str
    agent_id = agent.id
    try:
        registration = await agent.attach(SQLiteSource(database, name="Facts"))
        table_resource_id = catalog_resource_id(
            registration.id,
            ResourceKind.TABLE,
            "main.facts",
        )
        for position in range(1, 10):
            prompt = _intent(position)
            answer = _answer(position)
            prompts.append(prompt)
            answers.append(answer)
            provider.script.extend(
                (
                    ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        tool_calls=(
                            ToolCall(
                                id=f"call-{position}",
                                name="data_query_sqlite",
                                arguments={
                                    "source_id": registration.id,
                                    "sql": "SELECT value FROM facts",
                                },
                            ),
                        ),
                    ),
                    ModelResponse(
                        finish_reason=FinishReason.STOP,
                        text=answer,
                    ),
                )
            )
            request_start = len(provider.requests)
            result = await agent.run(prompt, session_id=SESSION_ID)
            assert result.kind is LoopExitKind.COMPLETED
            assert result.final_text == answer
            operation_ids.append(result.operation_id)
            if position == 9:
                later_request = provider.requests[request_start]

        assert provider.script == []
        later_texts = _message_texts(later_request)
        assert any(
            text.startswith("UNTRUSTED_SESSION_SUMMARY=") for text in later_texts
        )
        assert any("Journal request 8:" in text for text in later_texts)
        selected = _selected_records(later_request)
        assert any(
            record["kind"] == "session_summary" and record["required"] is True
            for record in selected
        )
        assert any(
            record["kind"] == "session_recent"
            and record["required"] is True
            and record["id"] == "session.recent.7"
            for record in selected
        )
    finally:
        await agent.close()

    reopened = await Agent.open("atlas", root=state_root, clock=lambda: NOW)
    try:
        persisted = await reopened.inspect(operation_ids[-1])
        persisted_request = persisted.model_calls[0].request
        assert persisted_request == later_request
        persisted_selected = _selected_records(persisted_request)
        persisted_session_kinds = {
            record["kind"]
            for record in persisted_selected
            if record["owner"] == "sessions"
        }
        assert persisted_session_kinds == {"session_recent", "session_summary"}
        assert all(
            record["required"] is True
            for record in persisted_selected
            if record["owner"] == "sessions"
        )

        transcript = await reopened.transcript(SESSION_ID)
        assert transcript.operation_ids == tuple(operation_ids)
        assert len(transcript.messages) == 3 * len(operation_ids)
        for position, operation_id in enumerate(operation_ids, start=1):
            user, tool_call, final = transcript.messages[
                (position - 1) * 3 : position * 3
            ]
            assert [user.role, tool_call.role, final.role] == [
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.ASSISTANT,
            ]
            assert user.operation_id == tool_call.operation_id == final.operation_id
            assert user.operation_id == operation_id
            assert user.content == (TextBlock(prompts[position - 1]),)
            assert tuple(call.id for call in tool_call.tool_calls) == (
                f"call-{position}",
            )
            assert final.content == (TextBlock(answers[position - 1]),)

        store = reopened._embedded._store
        checkpoint = await store.load_session_compression(agent_id, SESSION_ID)
        assert checkpoint is not None
        assert checkpoint.version == 1
        assert checkpoint.through_position == 3
        assert checkpoint.operation_ids == tuple(operation_ids[:4])
        assert checkpoint.through_operation_id == operation_ids[3]
        assert checkpoint.evidence_ids == tuple(
            f"evidence-{position}" for position in range(1, 5)
        )
        assert checkpoint.resource_ids == (table_resource_id,)
        assert checkpoint.source_fingerprint.startswith("sha256:")

        for position, operation_id in enumerate(checkpoint.operation_ids, start=1):
            facts = await store.load_session_operation(operation_id)
            assert facts is not None
            assert facts.objective == prompts[position - 1]
            assert facts.evidence_ids == (f"evidence-{position}",)
            assert facts.resource_ids == (table_resource_id,)
            assert tuple(scope.resource_id for scope in facts.resource_scope_facts) == (
                table_resource_id,
            )
    finally:
        await reopened.close()
