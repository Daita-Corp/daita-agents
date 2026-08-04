import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from daita import Agent, SQLiteSource
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop import LoopExitKind


async def test_default_agent_root_uses_final_daita_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    agent = await Agent.create("default-root")
    try:
        assert agent.home == tmp_path / ".daita" / "agents" / "default-root"
    finally:
        await agent.close()


async def test_public_agent_queries_sqlite_and_reopens_exact_transcript(tmp_path):
    database = tmp_path / "sales.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE sales(product TEXT, revenue INTEGER)")
    connection.executemany(
        "INSERT INTO sales VALUES (?, ?)", (("alpha", 10), ("beta", 20))
    )
    connection.commit()
    connection.close()

    provider = MockModelProvider(())
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )
    agent = await Agent.create(
        "sales",
        root=tmp_path,
        model=provider,
        model_profile=profile,
    )
    source = await agent.attach(SQLiteSource(database))
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="query-1",
                    name="data_query_sqlite",
                    arguments={
                        "source_id": source.id,
                        "sql": "SELECT product, revenue FROM sales ORDER BY revenue DESC",
                    },
                ),
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            text="beta has the highest revenue.",
        ),
    )

    result = await agent.run("Which product has the highest revenue?")
    transcript = await agent.transcript(result.run_id)
    await agent.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == "beta has the highest revenue."
    tool_result = transcript.messages[2].content[0]
    assert isinstance(tool_result, ToolResultBlock)
    data = tool_result.output["data"]
    assert isinstance(data, Mapping)
    rows = data["rows"]
    assert isinstance(rows, tuple)
    assert isinstance(rows[0], Mapping)
    assert rows[0]["product"] == "beta"

    reopened = await Agent.open("sales", root=tmp_path)
    try:
        restored = await reopened.transcript(result.run_id)
        assert restored == transcript
        assert (await reopened.list_sources())[0] == source
        assert len(await reopened.list_catalog_resources()) == 1
    finally:
        await reopened.close()


async def test_invalid_sql_is_a_model_visible_tool_error(tmp_path):
    database = tmp_path / "sales.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE sales(value INTEGER)")
    connection.commit()
    connection.close()
    provider = MockModelProvider(())
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "sales", root=tmp_path, model=provider, model_profile=profile
    )
    source = await agent.attach(SQLiteSource(database))
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="bad-query",
                    name="data_query_sqlite",
                    arguments={
                        "source_id": source.id,
                        "sql": "DELETE FROM sales",
                    },
                ),
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            text="I could not run a write because this agent is read-only.",
        ),
    )
    try:
        result = await agent.run("Delete the rows")
        transcript = await agent.transcript(result.run_id)
        error = transcript.messages[2].content[0]
        assert isinstance(error, ToolResultBlock)
        assert error.is_error is True
        details = error.output["error"]
        assert isinstance(details, Mapping)
        assert details["code"] == "sql_validation_failed"
        assert provider.requests[1].messages[-1] == transcript.messages[2]
    finally:
        await agent.close()
