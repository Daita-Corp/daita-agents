"""A question found, described and answered from the source through the catalog tools."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from daita import Agent, SQLiteSource
from daita.llm.models import FinishReason, ModelResponse, ToolCall, ToolResultBlock
from daita.llm.providers.mock import MockModelProvider
from tests.support.workspace import workspace_for


def _step(call_id: str, tool: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(call_id, tool, arguments),),
    )


async def test_a_table_created_after_attach_is_found_described_and_read(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shop.sqlite"
    sqlite3.connect(database).close()
    provider = MockModelProvider(
        (
            _step("find", "catalog_search", {"query": "shipments"}),
            _step(
                "describe",
                "catalog_describe",
                {"source": "Shop", "name": "main.shipments"},
            ),
            _step(
                "read",
                "data_query",
                {"source": "Shop", "sql": "SELECT count(*) AS n FROM shipments"},
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="One shipment."),
        )
    )
    agent = await Agent.create(
        "catalog-workflow",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE shipments (id INTEGER PRIMARY KEY)")
            connection.execute("INSERT INTO shipments VALUES (1)")
        result = await agent.run("How many shipments are there?")
        assert result.final_text == "One shipment."
        pairs = await agent.tool_evidence(result.run_id)
        assert [call.id for call, _ in pairs] == ["find", "describe", "read"]
        for _, block in pairs:
            assert isinstance(block, ToolResultBlock) and not block.is_error, block
    finally:
        await agent.close()
