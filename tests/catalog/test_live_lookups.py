"""catalog_search and catalog_describe ask the source; the catalog only adds to it."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, FolderSource, SQLiteSource
from daita._json import canonical_json
from daita.catalog.protocols import CatalogStoreError
from daita.context import _estimate_input_tokens
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
)
from daita.llm.providers.mock import MockModelProvider
from tests.support.workspace import workspace_for


class _BrokenCatalogStore:
    def __getattr__(self, name: str) -> object:
        raise CatalogStoreError(f"catalog store is unavailable: {name}")


def _database(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (id INTEGER PRIMARY KEY, email TEXT UNIQUE);
            CREATE TABLE orders (
              id INTEGER PRIMARY KEY,
              customer_id INTEGER REFERENCES customers(id),
              status TEXT NOT NULL
            );
            CREATE TABLE secrets (id INTEGER PRIMARY KEY, order_ref INTEGER);
            CREATE VIEW open_orders AS SELECT id FROM orders;
            """)
    return path


async def _agent(tmp_path: Path, *calls: ToolCall) -> tuple[Agent, MockModelProvider]:
    provider = MockModelProvider(
        (
            *(
                ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,))
                for call in calls
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Done."),
        )
    )
    agent = await Agent.create(
        "live-lookups",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )
    return agent, provider


async def _results(agent: Agent, question: str = "Look.") -> list[ToolResultBlock]:
    run = await agent.run(question)
    pairs = await agent.tool_evidence(run.run_id)
    return [cast(ToolResultBlock, result) for _, result in pairs]


def _data(result: ToolResultBlock) -> Mapping[str, object]:
    assert not result.is_error, result.output
    return cast(Mapping[str, object], result.output["data"])


def _hits(result: ToolResultBlock) -> dict[str, Mapping[str, object]]:
    hits = cast(Sequence[Mapping[str, object]], _data(result)["hits"])
    return {cast(str, hit["name"]): hit for hit in hits}


def _search(query: str, call_id: str = "search", **extra: object) -> ToolCall:
    return ToolCall(call_id, "catalog_search", {"query": query, **extra})


def _describe(name: str, call_id: str = "describe") -> ToolCall:
    return ToolCall(call_id, "catalog_describe", {"source": "Shop", "name": name})


async def test_search_finds_a_table_the_catalog_has_never_seen(tmp_path: Path) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _search("invoice"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE invoices (id INTEGER, total REAL)")
        (result,) = await _results(agent)
        hit = _hits(result)["main.invoices"]
        assert hit["seen_live"] is True
        assert hit["kind"] == "table"
        assert hit["dialect"] == "sqlite"
        assert hit["columns"] == "id INTEGER, total REAL"
    finally:
        await agent.close()


async def test_search_matches_column_names_and_lists_views(tmp_path: Path) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(
        tmp_path, _search("status"), _search("open", "views", kinds=["view"])
    )
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        by_column, views = await _results(agent)
        assert _hits(by_column)["main.orders"]["seen_live"] is True
        assert _hits(by_column)["main.customers"]["seen_live"] is False
        assert _hits(views)["main.open_orders"]["kind"] == "view"
    finally:
        await agent.close()


async def test_a_table_only_the_catalog_remembers_is_marked_not_seen_live(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _search("secrets"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE secrets")
        (result,) = await _results(agent)
        assert _hits(result)["main.secrets"]["seen_live"] is False
    finally:
        await agent.close()


async def test_search_answers_from_the_source_while_the_catalog_is_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _search("orders"), _describe("orders"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        for owner in (agent._embedded._catalog_service, agent._embedded._data_view):
            monkeypatch.setattr(owner, "_store", _BrokenCatalogStore())
        search, describe = await _results(agent)
        assert _hits(search)["main.orders"]["seen_live"] is True
        assert _data(search)["catalog_available"] is False
        assert _data(describe)["answered_from"] == "live"
    finally:
        await agent.close()


async def test_an_unreachable_source_answers_from_the_catalog_and_says_so(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, provider = await _agent(tmp_path, _search("orders"), _describe("orders"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        database.unlink()
        search, describe = await _results(agent)
        assert _hits(search)["main.orders"]["seen_live"] is False
        rendered = model_tool_result(search, None, provider.requests[-1].scope_names)
        output = cast(Mapping[str, Mapping[str, object]], rendered["output"])
        assert output["data"]["live_unavailable_sources"] == ["Shop"]
        described = _data(describe)
        assert described["answered_from"] == "cache"
        assert "could not be reached" in cast(str, described["note"])
        assert [
            cast(Mapping[str, object], column)["name"]
            for column in cast(Sequence[object], described["columns"])
        ] == ["id", "customer_id", "status"]
    finally:
        await agent.close()


async def test_describe_reads_the_live_table_with_keys_and_relationships(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(
        tmp_path, _describe("orders"), _describe("customers", "customers")
    )
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        # A column added after attach shows: a cached answer never overrides.
        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE orders ADD COLUMN placed_at TEXT")
        orders, customers = await _results(agent)
        data = _data(orders)
        assert data["name"] == "main.orders"
        assert data["answered_from"] == "live"
        assert data["primary_key"] == ("id",)
        assert [
            (column["name"], column["type"], column["nullable"])
            for column in cast(Sequence[Mapping[str, object]], data["columns"])
        ] == [
            ("id", "INTEGER", False),
            ("customer_id", "INTEGER", True),
            ("status", "TEXT", False),
            ("placed_at", "TEXT", True),
        ]
        (relationship,) = cast(Sequence[Mapping[str, object]], data["relationships"])
        assert (relationship["from"], relationship["to"]) == (
            "main.orders",
            "main.customers",
        )
        assert relationship["columns"] == (("customer_id", "id"),)
        assert _data(customers)["unique_keys"] == (("email",),)
    finally:
        await agent.close()


async def test_a_table_the_source_no_longer_has_is_unknown_even_if_cached(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _describe("secrets"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE secrets")
        (result,) = await _results(agent)
        assert result.is_error
        error = cast(Mapping[str, object], result.output["error"])
        assert error["code"] == "table_unknown"
    finally:
        await agent.close()


async def test_describing_a_schema_is_capped_and_says_how_many_exist(
    tmp_path: Path,
) -> None:
    database = tmp_path / "wide.sqlite"
    with sqlite3.connect(database) as connection:
        for index in range(130):
            connection.execute(f"CREATE TABLE t{index:03d} (id INTEGER)")
    agent, _provider = await _agent(tmp_path, _describe("main"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        (result,) = await _results(agent)
        data = _data(result)
        assert data["kind"] == "schema"
        assert data["total_tables"] == 130
        assert len(cast(Sequence[object], data["tables"])) == 100
    finally:
        await agent.close()


async def test_search_returns_compact_capped_hits(tmp_path: Path) -> None:
    database = tmp_path / "wide.sqlite"
    with sqlite3.connect(database) as connection:
        for index in range(60):
            columns = ", ".join(f"column_{item} TEXT" for item in range(30))
            connection.execute(f"CREATE TABLE orders_{index:03d} ({columns})")
    agent, _provider = await _agent(
        tmp_path, _search("orders"), _search("orders", "two", limit=3)
    )
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        default, narrowed = await _results(agent)
        assert _data(default)["returned_count"] == 12
        assert _data(default)["truncated"] is True
        assert len(canonical_json(default.output)) < 6_000
        assert "(+18 more)" in cast(str, _hits(default)["main.orders_000"]["columns"])
        assert _data(narrowed)["returned_count"] == 3
    finally:
        await agent.close()


async def test_selected_scope_limits_what_lookups_show(tmp_path: Path) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(
        tmp_path,
        _search("orders secrets customers"),
        _describe("secrets", "denied"),
        _describe("main", "listing"),
        _describe("orders", "orders"),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        orders = next(
            item
            for item in await agent.list_catalog_resources(source_id=source.id)
            if item.name == "orders"
        )
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="selected",
            read_resource_ids=(orders.id,),
            relational_write_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )
        search, denied, listing, described = await _results(agent)
        assert set(_hits(search)) == {"main.orders"}
        assert denied.is_error
        assert "secrets" not in canonical_json(_data(listing))
        assert [
            cast(Mapping[str, object], table)["name"]
            for table in cast(Sequence[object], _data(listing)["tables"])
        ] == ["main.orders"]
        # orders references customers, which this run cannot read.
        assert _data(described)["relationships"] == ()
    finally:
        await agent.close()


async def test_lookups_render_names_and_never_ids(tmp_path: Path) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, provider = await _agent(tmp_path, _search("orders"), _describe("orders"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        results = await _results(agent)
        rendered = canonical_json(
            [
                model_tool_result(item, None, provider.requests[-1].scope_names)
                for item in results
            ]
        )
        assert '"source":"Shop"' in rendered
        assert "sha256:" not in rendered.replace(
            str(_data(results[1]).get("revision")), ""
        )
    finally:
        await agent.close()


async def test_a_folder_search_lists_its_data_files_and_describes_columns(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "data"
    (folder / "reports").mkdir(parents=True)
    (folder / "reports" / "orders.csv").write_text("id,total\n1,2.5\n")
    (folder / "notes.txt").write_text("orders are late\n")
    home = tmp_path / "home"
    home.mkdir()
    agent, _provider = await _agent(
        home,
        _search("orders", source="Shop"),
        _describe("reports/orders.csv"),
    )
    try:
        await agent.attach(FolderSource(folder, name="Shop"))
        search, describe = await _results(agent)
        assert _hits(search)["reports/orders.csv"]["kind"] == "file"
        assert _hits(search)["reports/orders.csv"]["seen_live"] is True
        assert "notes.txt" not in _hits(search)
        assert [
            cast(Mapping[str, object], column)["name"]
            for column in cast(Sequence[object], _data(describe)["columns"])
        ] == ["id", "total"]
    finally:
        await agent.close()


async def test_hunting_for_a_table_cannot_exhaust_the_context_window(
    tmp_path: Path,
) -> None:
    """Stored provenance once read as hundreds of thousands of tokens per result."""

    database = tmp_path / "wide.sqlite"
    with sqlite3.connect(database) as connection:
        for index in range(300):
            connection.execute(f"CREATE TABLE orders_{index:03d} (id INTEGER)")
    calls = tuple(_search(f"orders {index}", f"call-{index}") for index in range(10))
    provider = MockModelProvider(
        (
            *(
                ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,))
                for call in calls
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Done."),
        )
    )
    agent = await Agent.create(
        "hunting",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=120_000,
            max_output_tokens=2_048,
            supports_tools=True,
        ),
    )
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        result = await agent.run("How many shipped orders are there?")
        assert result.reason == "completed"
        assert max(_estimate_input_tokens(item) for item in provider.requests) < 60_000
    finally:
        await agent.close()
