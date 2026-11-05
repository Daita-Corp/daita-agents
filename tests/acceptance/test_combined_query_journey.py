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


async def test_sixteen_inputs_over_shared_sources_report_how_the_planted_join_matched(
    tmp_path: Path,
):
    # Four databases hold four inputs each; input i reads group i. Inputs 0 and 1
    # join on n with one key missing from each side.
    groups: dict[int, tuple[int, ...]] = {i: (1, 2) for i in range(16)}
    groups[0], groups[1] = (1, 2, 3), (1, 2, 9)
    databases = []
    for db in range(4):
        path = tmp_path / f"db{db}.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE items (grp INTEGER, n INTEGER, w INTEGER)")
            connection.executemany(
                "INSERT INTO items VALUES (?, ?, ?)",
                [
                    (grp, n, (grp + 1) * n)
                    for grp in range(db, 16, 4)
                    for n in groups[grp]
                ],
            )
        databases.append(path)
    provider = ToolboxAwareMockModelProvider((), provider_id="mock:combined-sixteen")
    agent = await Agent.create(
        "combined-sixteen",
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
        sources = []
        for index, path in enumerate(databases):
            source = await agent.attach(SQLiteSource(path, name=f"Db{index}"))
            (table,) = await agent.list_catalog_resources(source_id=source.id)
            sources.append((source.id, table.id))
        inputs = [
            {
                "alias": f"a{i}",
                "source_id": sources[i % 4][0],
                "resource_ids": [sources[i % 4][1]],
                "sql": f"SELECT n, w FROM items WHERE grp = {i}",
            }
            for i in range(16)
        ]
        joins = " ".join(f"JOIN a{i} ON a{i}.n = a0.n" for i in range(1, 16))
        total = " + ".join(f"a{i}.w" for i in range(16))
        provider.replace_script(
            (
                _call(
                    "sixteen",
                    "data_query",
                    {
                        "inputs": inputs,
                        "sql": f"SELECT a0.n AS n, {total} AS total FROM a0 "
                        f"{joins} ORDER BY a0.n",
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="Joined."),
            )
        )
        result = await agent.run("Total w per n across all sixteen inputs.")
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    block = next(
        block
        for message in transcript.messages
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == "sixteen"
    )
    assert not block.is_error, block.output
    data = cast(Mapping[str, object], block.output["data"])
    # Each n totals n times 1 + 2 + ... + 16.
    assert [
        dict(row) for row in cast(tuple[Mapping[str, object], ...], data["rows"])
    ] == [
        {"n": 1, "total": 136},
        {"n": 2, "total": 272},
    ]
    assert data["truncated"] is False
    assert "read_at" in cast(str, data["read_consistency"])
    inputs_out = cast(tuple[Mapping[str, object], ...], data["inputs"])
    assert [item["alias"] for item in inputs_out] == [f"a{i}" for i in range(16)]
    assert [item["rows"] for item in inputs_out] == [3, 3] + [2] * 14
    assert all(
        isinstance(item["read_ms"], int) and item["read_ms"] >= 0 for item in inputs_out
    )
    checks = [
        dict(item)
        for item in cast(tuple[Mapping[str, object], ...], data["join_checks"])
    ]
    planted = next(
        item for item in checks if {item["left"], item["right"]} == {"a0.n", "a1.n"}
    )
    assert (
        planted["left_rows"],
        planted["left_unmatched"],
        planted["right_rows"],
        planted["right_unmatched"],
    ) == (3, 1, 3, 1)
    warnings = cast(tuple[str, ...], data["warnings"])
    assert (
        "a1.n = a0.n: 1 of 3 rows in a1.n have no match; "
        "1 of 3 rows in a0.n have no match." in warnings
    )
    # Every other input shares a0's missing key 3, and nothing else warns.
    assert len(warnings) == 15
