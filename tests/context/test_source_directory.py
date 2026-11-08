"""The prompt carries a short directory, never a catalog ranked for the question."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from daita import Agent, SQLiteSource
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sqlite_codecs import (
    decode_catalog_sync,
    encode_catalog_sync,
)
from tests.support.workspace import workspace_for

_NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


def _system(request: ModelRequest) -> str:
    block = request.messages[0].content[0]
    assert isinstance(block, TextBlock)
    return block.text


def _database(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (id INTEGER PRIMARY KEY);
            CREATE TABLE orders_east (id INTEGER PRIMARY KEY);
            CREATE TABLE orders_west (id INTEGER PRIMARY KEY);
            """)
    return path


def _stop(text: str = "Done.") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


async def _agent(
    tmp_path: Path, *script: ModelResponse
) -> tuple[Agent, MockModelProvider]:
    provider = MockModelProvider(script)
    agent = await Agent.create(
        "source-directory",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
        clock=lambda: _NOW,
    )
    await agent.attach(SQLiteSource(_database(tmp_path / "shop.sqlite"), name="Shop"))
    return agent, provider


async def test_the_prompt_is_the_same_whatever_the_question_names(
    tmp_path: Path,
) -> None:
    agent, provider = await _agent(tmp_path, _stop(), _stop())
    try:
        first = await agent.run("How many east orders are there?")
        second = await agent.run("Which customers are new?")
    finally:
        await agent.close()
    assert first.conversation_id != second.conversation_id
    one, other = (_system(request) for request in provider.requests)
    strip = lambda text: "\n\n".join(  # noqa: E731
        part for part in text.split("\n\n") if "time at run start" not in part
    )
    assert strip(one) == strip(other)


async def test_the_directory_names_each_source_but_no_table(tmp_path: Path) -> None:
    agent, provider = await _agent(tmp_path, _stop())
    try:
        await agent.run("Show me the east orders.")
    finally:
        await agent.close()
    system = _system(provider.requests[0])
    assert '"label":"Shop"' in system
    assert '"type":"SQLite"' in system
    assert '"dialect":"sqlite"' in system
    assert '"approximate_objects":3' in system
    assert "catalog_partial" not in system
    for ranked in ("orders_east", "match_outcomes", "binding_status"):
        assert ranked not in system


async def test_a_truncated_catalog_says_the_directory_is_only_part_of_the_source(
    tmp_path: Path,
) -> None:
    agent, provider = await _agent(tmp_path, _stop())
    try:
        with sqlite3.connect(agent.home / "state.db") as connection:
            ((data,),) = connection.execute(
                "SELECT y.data FROM syncs AS y JOIN catalog_source_state AS s "
                "ON y.agent_id = s.agent_id AND y.source_id = s.source_id"
            )
            sync = decode_catalog_sync(data)
            connection.execute(
                "UPDATE syncs SET data = ? WHERE id = ?",
                (encode_catalog_sync(replace(sync, truncated=True)), sync.id),
            )
        await agent.run("Hello.")
    finally:
        await agent.close()
    system = _system(provider.requests[0])
    assert '"approximate_objects":3' in system
    assert "catalog_partial" in system


async def test_a_failing_catalog_leaves_the_directory_without_a_count(
    tmp_path: Path, monkeypatch
) -> None:
    from daita.catalog.protocols import CatalogStoreError

    agent, provider = await _agent(tmp_path, _stop())

    class _Broken:
        def __getattr__(self, name: str) -> object:
            raise CatalogStoreError(name)

    try:
        monkeypatch.setattr(agent._embedded._data_view, "_store", _Broken())
        monkeypatch.setattr(agent._embedded._catalog_service, "_store", _Broken())
        result = await agent.run("Hello.")
        assert result.reason == "completed"
    finally:
        await agent.close()
    system = _system(provider.requests[0])
    assert '"approximate_objects":null' in system
    assert "catalog_partial" not in system


async def test_later_turns_read_tables_again_rather_than_trust_old_descriptions(
    tmp_path: Path,
) -> None:
    agent, provider = await _agent(
        tmp_path,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    "describe",
                    "catalog_describe",
                    {"source": "Shop", "name": "customers"},
                ),
            ),
        ),
        _stop("It has an id."),
        _stop("Still an id."),
    )
    try:
        first = await agent.run("What columns does customers have?")
        await agent.run("And now?", conversation_id=first.conversation_id)
    finally:
        await agent.close()
    later = repr(provider.requests[-1].messages)
    assert "It has an id." in later
    assert "catalog.description" not in later
