from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sqlite3

from daita import Agent, LocalDirectorySource, SQLiteSource
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider


def _database(path: Path, table: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)'  # noqa: S608
        )


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _user_texts(request: object) -> tuple[str, ...]:
    messages = getattr(request, "messages")
    return tuple(
        block.text
        for message in messages
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextBlock)
    )


async def test_selected_source_persists_and_detach_falls_back_to_the_only_source(
    tmp_path: Path,
):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    _database(first_path, "first_records")
    _database(second_path, "second_records")
    agent = await Agent.create("source-persistence", root=tmp_path)
    first = await agent.attach(SQLiteSource(first_path, name="First Source"))
    second = await agent.attach(SQLiteSource(second_path, name="Second Source"))

    assert await agent.active_source() == first
    assert await agent.resolve_source("second-source") == second
    assert await agent.select_source("second-source") == second
    assert await agent.active_source() == second
    await agent.close()

    reopened = await Agent.open("source-persistence", root=tmp_path)
    try:
        assert await reopened.active_source() == second
        await reopened.detach(second.id)
        assert await reopened.active_source() == first
    finally:
        await reopened.close()


async def test_one_run_override_keeps_conversation_source_and_history_isolated(
    tmp_path: Path,
):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    _database(first_path, "first_records")
    _database(second_path, "second_records")
    provider = MockModelProvider(
        (
            _stop("first answer"),
            _stop("override answer"),
            _stop("follow-up answer"),
        )
    )
    agent = await Agent.create(
        "source-override",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        first_source = await agent.attach(SQLiteSource(first_path, name="First Source"))
        second_source = await agent.attach(
            SQLiteSource(second_path, name="Second Source")
        )
        first = await agent.run("Question about first_records")
        override = await agent.run(
            "Question about second_records",
            conversation_id=first.conversation_id,
            source_id=second_source.id,
        )
        follow_up = await agent.run(
            "Follow up about first_records",
            conversation_id=first.conversation_id,
        )

        assert (
            await agent.active_source(conversation_id=first.conversation_id)
            == first_source
        )
        override_run = (await agent.transcript(override.run_id)).run
        assert override_run.source_id == second_source.id
        assert override_run.conversation_source_id == first_source.id
        assert _user_texts(provider.requests[0]) == ("Question about first_records",)
        assert _user_texts(provider.requests[1]) == ("Question about second_records",)
        assert _user_texts(provider.requests[2]) == (
            "Question about first_records",
            "Follow up about first_records",
        )
        first_system = repr(provider.requests[0].messages[0])
        override_system = repr(provider.requests[1].messages[0])
        follow_up_system = repr(provider.requests[2].messages[0])
        assert first_source.id in first_system
        assert second_source.id not in first_system
        assert second_source.id in override_system
        assert first_source.id not in override_system
        assert first_source.id in follow_up_system
        assert second_source.id not in follow_up_system
        assert follow_up.conversation_id == first.conversation_id
    finally:
        await agent.close()


async def test_runtime_injects_catalog_scope_and_rejects_cross_source_query(
    tmp_path: Path,
):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    _database(first_path, "shared_first")
    _database(second_path, "shared_second")
    provider = MockModelProvider(())
    agent = await Agent.create(
        "source-enforcement",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    first_source = await agent.attach(SQLiteSource(first_path, name="First Source"))
    second_source = await agent.attach(SQLiteSource(second_path, name="Second Source"))
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="catalog",
                    name="catalog_search",
                    arguments={"query": "shared"},
                ),
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="cross-source",
                    name="data_query_sqlite",
                    arguments={
                        "source_id": second_source.id,
                        "sql": "SELECT id FROM shared_second",
                    },
                ),
            ),
        ),
        _stop("The cross-source query was blocked."),
    )
    try:
        result = await agent.run("Find shared records")
        transcript = await agent.transcript(result.run_id)
        catalog_result = transcript.messages[2].content[0]
        blocked_result = transcript.messages[4].content[0]
        assert isinstance(catalog_result, ToolResultBlock)
        assert isinstance(blocked_result, ToolResultBlock)
        catalog_data = catalog_result.output["data"]
        assert isinstance(catalog_data, Mapping)
        hits = catalog_data["hits"]
        assert isinstance(hits, tuple)
        assert hits
        assert all(
            isinstance(hit, Mapping) and hit["source_id"] == first_source.id
            for hit in hits
        )
        error = blocked_result.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "source_scope_violation"
    finally:
        await agent.close()


async def test_selected_source_projects_only_its_data_adapter_tools(tmp_path: Path):
    database = tmp_path / "records.sqlite"
    files = tmp_path / "files"
    files.mkdir()
    _database(database, "database_records")
    (files / "file_records.csv").write_text("id\n1\n", encoding="utf-8")
    provider = MockModelProvider((_stop("file source selected"),))
    agent = await Agent.create(
        "source-tool-projection",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        await agent.attach(SQLiteSource(database, name="SQLite Source"))
        await agent.attach(LocalDirectorySource(files, name="File Source"))
        await agent.select_source("file-source")

        await agent.run("What file data is available?")

        tool_names = {tool.name for tool in provider.requests[0].tools}
        assert "data_read_file" in tool_names
        assert "data_query_sqlite" not in tool_names
    finally:
        await agent.close()
