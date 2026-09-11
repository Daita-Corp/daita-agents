"""Component-owned tests split from ``test_schema_slices.py``."""

from __future__ import annotations

from tests.catalog._schema_support import (
    Agent,
    ModelProfile,
    Path,
    SQLiteSource,
    _BridgePlanningProvider,
    _fixture_database,
    _InventoryProvider,
    _RegionalMarginProvider,
    _RevisionReuseProvider,
    sqlite3,
    workspace_for,
)


async def test_inventory_uses_one_catalog_tool_call_and_records_efficiency(
    tmp_path: Path,
):
    database = tmp_path / "inventory.sqlite"
    _fixture_database(database)
    provider = _InventoryProvider()
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=80_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "catalog-inventory",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.attach(SQLiteSource(database))
        result = await agent.run("What tables and relationships are available?")
        assert (
            result.final_text == "Eight tables and their relationships are available."
        )
        assert provider.catalog_tool_call_count == 1
        assert provider.projected_resource_count == 8
        assert provider.duplicated_relationship_id_count == 0
        assert provider.schema_result_bytes > 0
        assert provider.truncation is not None
        assert provider.truncation["resources"] is False
        assert provider.truncation["relationships"] is False
        calls = tuple(
            call
            for request in provider.requests
            for message in request.messages
            for call in message.tool_calls
            if call.name.startswith("catalog_")
        )
        assert tuple(call.name for call in calls) == ("catalog_schema",)
    finally:
        await agent.close()


async def test_regional_margin_plan_uses_one_schema_slice_before_querying(
    tmp_path: Path,
):
    database = tmp_path / "regional-margin.sqlite"
    _fixture_database(database)
    provider = _RegionalMarginProvider()
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=80_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "catalog-regional-margin",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.attach(SQLiteSource(database))
        result = await agent.run("Summarize paid revenue and gross margin by region.")
        assert result.final_text == "EMEA paid revenue is 110 with gross margin 60."
        assert provider.planned_from_schema is True
        assert provider.catalog_tool_call_count == 1
        assert provider.schema_result_bytes > 0
        calls_by_id = {
            call.id: call.name
            for request in provider.requests
            for message in request.messages
            for call in message.tool_calls
        }
        assert tuple(calls_by_id.values()) == (
            "catalog_schema",
            "data_query",
        )
    finally:
        await agent.close()


async def test_one_connected_schema_call_supplies_bridges_before_one_data_query(
    tmp_path: Path,
):
    database = tmp_path / "bridge-planning.sqlite"
    _fixture_database(database)
    provider = _BridgePlanningProvider()
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=80_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "catalog-bridge-planning",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.attach(SQLiteSource(database))
        result = await agent.run("What is gross margin by customer segment?")
        assert result.final_text == "Enterprise gross margin is 60."
        assert provider.saw_complete_bridge_evidence is True
        calls = {
            call.id: call.name
            for request in provider.requests
            for message in request.messages
            for call in message.tool_calls
        }
        assert tuple(calls.values()) == ("catalog_schema", "data_query")
    finally:
        await agent.close()


async def test_unchanged_revision_reuses_schema_but_refresh_requires_new_slice(
    tmp_path: Path,
):
    database = tmp_path / "schema-reuse.sqlite"
    _fixture_database(database)
    provider = _RevisionReuseProvider()
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=80_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "catalog-schema-reuse",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        prompt = "What tables and relationships are available now?"
        first = await agent.run(prompt)
        assert provider.schema_calls == 1

        await agent.run(prompt, conversation_id=first.conversation_id)
        assert provider.schema_calls == 1

        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE orders ADD COLUMN sales_note TEXT")
        await agent.refresh_source(source.id)
        await agent.run(prompt, conversation_id=first.conversation_id)
        assert provider.schema_calls == 2
    finally:
        await agent.close()
