from __future__ import annotations

from collections.abc import Mapping, Sequence
import io
from pathlib import Path
import re
import sqlite3
from typing import Any, cast

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from daita import Agent, SQLiteSource
from daita.agent import PostgreSQLProbeResult
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference
from daita.terminal import run_terminal_application
import daita.hosting.embedded as embedded


class _FakeKeychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.events: list[tuple[str, str]] = []

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.events.append(("set", reference.name))
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        self.events.append(("delete", reference.name))
        self.values.pop(reference.name, None)


class _GroundedProvider:
    provider_id = "openai:stage5-manual-model"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.answer: str | None = None

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="stage5-catalog",
                        name="catalog_search",
                        arguments={"query": "revenue", "limit": 5},
                    ),
                ),
                provider_id=self.provider_id,
            )
        if len(self.requests) == 2:
            search_results = tuple(
                block
                for message in request.messages
                for block in message.content
                if isinstance(block, ToolResultBlock)
            )
            if not search_results or search_results[-1].is_error:
                raise AssertionError("catalog search did not expose a source ID")
            search_data = cast(
                Mapping[str, object],
                search_results[-1].output["data"],
            )
            hits = cast(Sequence[Mapping[str, object]], search_data["hits"])
            source_id = cast(str, hits[0]["source_id"])
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="stage5-grounding",
                        name="data_query_sqlite",
                        arguments={
                            "source_id": source_id,
                            "sql": "SELECT region, paid_revenue FROM revenue",
                            "parameters": [],
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )

        results = tuple(
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        if not results or results[-1].is_error:
            raise AssertionError("expected one successful grounded tool result")
        result_data = cast(Mapping[str, object], results[-1].output["data"])
        rows = cast(Sequence[Mapping[str, object]], result_data["rows"])
        self.answer = (
            f"{rows[0]['region']} has {rows[0]['paid_revenue']} in paid revenue."
        )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text=self.answer,
            provider_id=self.provider_id,
        )


async def _empty_agent(root: Path, name: str) -> None:
    agent = await Agent.create(name, root=root)
    await agent.close()


def _source_state_counts(root: Path, name: str) -> tuple[int, int]:
    with sqlite3.connect(root / "agents" / name / "state.db") as connection:
        return (
            connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],
        )


async def test_first_run_keyboard_postgresql_path_reaches_grounded_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    await _empty_agent(tmp_path, "alpha")
    await _empty_agent(tmp_path, "beta")
    keychain = _FakeKeychain()
    validator = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="validation-call",
                        name="daita_validate_tool_support",
                        arguments={},
                    ),
                ),
                provider_id="openai:stage5-manual-model",
            ),
        ),
        provider_id="openai:stage5-manual-model",
    )
    provider = _GroundedProvider()
    monkeypatch.setattr(
        embedded,
        "create_model_route_provider",
        lambda route, *, secret_provider=None: provider,
    )
    selected_attachments: list[dict[str, Any]] = []
    database = tmp_path / "fake-postgresql.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE revenue (region TEXT NOT NULL, paid_revenue INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO revenue VALUES ('EMEA', 4200)")

    async def fake_probe(self: Agent, **kwargs: Any) -> PostgreSQLProbeResult:
        del self, kwargs
        return PostgreSQLProbeResult.build(
            (
                ("analytics", True),
                ("empty", False),
                ("reporting", True),
            )
        )

    async def fake_attach(self: Agent, **kwargs: Any):
        selected_attachments.append(kwargs)
        return await self.attach(SQLiteSource(database, name=kwargs["name"]))

    monkeypatch.setattr(Agent, "probe_postgresql", fake_probe)
    monkeypatch.setattr(Agent, "attach_postgresql", fake_attach)
    hidden_prompts: list[str] = []
    hidden_values = iter(("provider-secret", "database-secret"))

    def hidden_input(prompt: str) -> str:
        hidden_prompts.append(prompt)
        return next(hidden_values)

    output = io.StringIO()
    with create_pipe_input() as pipe:
        pipe.send_text(
            "\x1b[B\x1b[B\r"  # create a new agent after two existing agents
            "\r"  # OpenAI
            "\x1b[B\x1b[B\x1b[B\r"  # manual model entry
            "\x1b[B\x1b[B\r"  # PostgreSQL
            "\x1b[B\r"  # individual connection fields
            " \x1b[B\x1b[B \r"  # analytics and reporting
        )
        code = await run_terminal_application(
            root=tmp_path,
            input_stream=io.StringIO(
                "customer\n"
                "stage5-manual-model\n"
                "128000\n"
                "4096\n"
                "Fixture PostgreSQL\n"
                "db.example.test\n"
                "5432\n"
                "warehouse\n"
                "reader\n"
                "disable\n"
                "Which region leads paid revenue?\n"
                "/exit\n"
            ),
            output_stream=output,
            hidden_input=hidden_input,
            keychain=keychain,
            model_validator=validator,
            selection_input=pipe,
            selection_output=DummyOutput(),
        )

    assert code == 0
    assert hidden_prompts == ["API key: ", "Password: "]
    assert len(selected_attachments) == 1
    assert selected_attachments[0]["schemas"] == ("analytics", "reporting")
    assert "empty" not in selected_attachments[0]["schemas"]
    assert "information_schema" not in selected_attachments[0]["schemas"]
    assert "pg_catalog" not in selected_attachments[0]["schemas"]
    text = output.getvalue()
    assert "\x1b" not in text
    assert "[bold]" not in text
    assert "[/bold]" not in text
    assert "Status" in text and "Ready" in text
    assert "Ask a question about your data" in text
    assert "Validating model configuration" in text
    assert "Model configuration validated" in text
    assert "Validating PostgreSQL connection" in text
    assert "✓ Connection validated" in text
    assert "✓ Schemas selected: analytics, reporting" in text
    assert provider.answer == "EMEA has 4200 in paid revenue."
    assert provider.answer in text
    assert "provider-secret" not in text
    assert "database-secret" not in text
    assert len(provider.requests) == 3
    validator.assert_consumed()

    reopened = await Agent.open("customer", root=tmp_path, keychain=keychain)
    try:
        conversation = re.search(
            r"Conversation  (conversation-[A-Za-z0-9]+)",
            text,
        )
        assert conversation is not None
        records = await reopened.conversation_runs(conversation.group(1))
        transcript = records[0].transcript
        transcript_text = "\n".join(
            block.text
            for message in transcript.messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "Which region leads paid revenue?" in transcript_text
        for local_text in (
            "Select an agent",
            "Select a model provider",
            "Select an OpenAI model",
            "Select a source type",
            "Select one or more schemas",
            "Space toggle",
        ):
            assert local_text not in transcript_text
            assert all(
                local_text not in block.text
                for request in provider.requests
                for message in request.messages
                if message.role is not MessageRole.SYSTEM
                for block in message.content
                if isinstance(block, TextBlock)
            )
    finally:
        await reopened.close()


async def test_returning_keyboard_selection_skips_onboarding_and_writes_no_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keychain = _FakeKeychain()
    database = tmp_path / "returning.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
    validator = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="validation-call",
                        name="daita_validate_tool_support",
                        arguments={},
                    ),
                ),
                provider_id="openai:returning-model",
            ),
        ),
        provider_id="openai:returning-model",
    )
    for name in ("alpha", "beta"):
        await _empty_agent(tmp_path, name)
    agent = await Agent.create(
        "customer",
        root=tmp_path,
        keychain=keychain,
        model_validator=validator,
    )
    await agent.configure_model(
        provider="openai",
        model="returning-model",
        api_key="returning-secret",
        context_window_tokens=128_000,
        max_output_tokens=4_096,
    )
    registration = await agent.attach(SQLiteSource(database, name="Retained source"))
    before_summary = await agent.catalog_summary()
    await agent.close()
    config_path = tmp_path / "agents" / "customer" / "config.json"
    config_before = config_path.read_bytes()
    config_mtime_before = config_path.stat().st_mtime_ns
    keychain_events_before = tuple(keychain.events)
    source_state_before = _source_state_counts(tmp_path, "customer")
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="returning answer",
                provider_id="openai:returning-model",
            ),
        ),
        provider_id="openai:returning-model",
    )
    monkeypatch.setattr(
        embedded,
        "create_model_route_provider",
        lambda route, *, secret_provider=None: provider,
    )
    output = io.StringIO()

    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[B\x1b[B\r")
        code = await run_terminal_application(
            root=tmp_path,
            input_stream=io.StringIO("returning question\n/exit\n"),
            output_stream=output,
            hidden_input=lambda prompt: (_ for _ in ()).throw(
                AssertionError(f"unexpected hidden prompt: {prompt}")
            ),
            keychain=keychain,
            selection_input=pipe,
            selection_output=DummyOutput(),
        )

    assert code == 0
    text = output.getvalue()
    assert "Agent     customer" in text
    assert "Model     OpenAI · returning-model · configured" in text
    assert "Source    Retained source · cataloged" in text
    assert "Status" in text and "Ready" in text
    assert "Ask a question about your data" in text
    assert "returning answer" in text
    assert "Agent name:" not in text
    assert "Select a model provider" not in text
    assert "Select a source type" not in text
    assert config_path.read_bytes() == config_before
    assert config_path.stat().st_mtime_ns == config_mtime_before
    assert tuple(keychain.events) == keychain_events_before
    assert _source_state_counts(tmp_path, "customer") == source_state_before
    assert len(provider.requests) == 1
    request_text = tuple(
        block.text
        for message in provider.requests[0].messages
        if message.role is not MessageRole.SYSTEM
        for block in message.content
        if isinstance(block, TextBlock)
    )
    assert request_text == ("returning question",)

    reopened = await Agent.open("customer", root=tmp_path, keychain=keychain)
    try:
        sources = await reopened.list_sources()
        after_summary = await reopened.catalog_summary()
        assert tuple(source.id for source in sources) == (registration.id,)
        assert after_summary.active_source_count == before_summary.active_source_count
        assert after_summary.resource_count == before_summary.resource_count
        assert after_summary.relationship_count == before_summary.relationship_count
    finally:
        await reopened.close()
