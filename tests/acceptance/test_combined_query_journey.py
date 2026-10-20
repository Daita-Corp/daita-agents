"""One model call joins two sources and its result backs a derived artifact."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, SQLiteSource
from daita.artifacts.models import ArtifactAuthorship
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from tests.support.toolbox_model import ToolboxAwareMockModelProvider, result_ref
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.acceptance


def _call(call_id: str, name: str, arguments: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


async def test_revenue_by_region_joins_two_sources_and_backs_a_table_artifact(
    tmp_path: Path,
):
    orders_database = tmp_path / "orders.sqlite"
    customers_database = tmp_path / "customers.sqlite"
    with sqlite3.connect(orders_database) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, "
            "total INTEGER)"
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            ((1, 1, 100), (2, 2, 40), (3, 3, 25), (4, 1, 15)),
        )
    with sqlite3.connect(customers_database) as connection:
        connection.execute(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, region TEXT)"
        )
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?)",
            ((1, "north"), (2, "south"), (3, "north")),
        )
    provider = ToolboxAwareMockModelProvider((), provider_id="mock:combined-query")
    agent = await Agent.create(
        "combined-journey",
        root=tmp_path,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        workspace=workspace_for(tmp_path),
    )
    try:
        orders_source = await agent.attach(SQLiteSource(orders_database))
        customers_source = await agent.attach(SQLiteSource(customers_database))
        (orders,) = await agent.list_catalog_resources(source_id=orders_source.id)
        (customers,) = await agent.list_catalog_resources(source_id=customers_source.id)
        provider.replace_script(
            (
                _call(
                    "revenue",
                    "data_query",
                    {
                        "inputs": [
                            {
                                "alias": "o",
                                "source_id": orders_source.id,
                                "resource_ids": [orders.id],
                                "sql": "SELECT customer_id, SUM(total) AS revenue "
                                "FROM orders GROUP BY customer_id",
                            },
                            {
                                "alias": "c",
                                "source_id": customers_source.id,
                                "resource_ids": [customers.id],
                                "sql": "SELECT id, region FROM customers",
                            },
                        ],
                        "sql": "SELECT c.region, SUM(o.revenue) AS revenue FROM o "
                        "JOIN c ON c.id = o.customer_id GROUP BY c.region "
                        "ORDER BY c.region",
                    },
                ),
                _call(
                    "table",
                    "artifact_create",
                    {
                        "from": "table",
                        "columns": ["region", "revenue"],
                        "rows": [["north", 140], ["south", 40]],
                        "format": "csv",
                        "filename": "revenue.csv",
                        "evidence_refs": [result_ref("revenue")],
                    },
                ),
                ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="North earned 140 and south 40.",
                ),
            )
        )
        result = await agent.run("What is revenue by customer region?")
        transcript = await agent.transcript(result.run_id)
        (artifact,) = result.artifacts
        content = (await agent.read_artifact(artifact.artifact_id)).content
    finally:
        await agent.close()

    revenue = next(
        block
        for message in transcript.messages
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == "revenue"
    )
    data = cast(Mapping[str, object], revenue.output["data"])
    assert [
        dict(row) for row in cast(tuple[Mapping[str, object], ...], data["rows"])
    ] == [
        {"region": "north", "revenue": 140},
        {"region": "south", "revenue": 40},
    ]
    assert result.final_text == "North earned 140 and south 40."
    assert artifact.provenance.authorship is ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
    assert artifact.provenance.evidence_call_ids == ("revenue",)
    assert {
        (binding.source_id, binding.resource_id)
        for binding in artifact.provenance.resource_bindings
    } == {(orders_source.id, orders.id), (customers_source.id, customers.id)}
    assert content == b'"region","revenue"\r\n"north",140\r\n"south",40\r\n'
