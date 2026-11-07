"""catalog_search and catalog_describe ask the source; the catalog only adds to it."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, FolderSource, SQLiteSource
from daita._json import canonical_json
from daita.adapters.models import SourceRegistration
from daita.capabilities import ToolOutput
from daita.catalog.capabilities import (
    CATALOG_DESCRIBE_CAPABILITY_ID,
    CATALOG_DESCRIBE_EVIDENCE_KIND,
)
from daita.catalog.protocols import CatalogStoreError
from daita.context import _estimate_input_tokens
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
)
from daita.llm.providers.mock import MockModelProvider
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticKind,
    SemanticSubject,
)
from tests.support.source_refresh import settled, sync_count
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
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE invoices (id INTEGER, total REAL)")
        (result,) = await _results(agent)
        hit = _hits(result)["main.invoices"]
        assert hit["seen_live"] is True
        assert hit["kind"] == "table"
        assert hit["dialect"] == "sqlite"
        assert hit["columns"] == "id INTEGER, total REAL"
        # Seeing a table only the source knows brings the catalog up to date.
        await settled(agent)
        names = {
            item.name
            for item in await agent.list_catalog_resources(source_id=source.id)
        }
        assert "invoices" in names
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
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        # A column added after attach shows: a cached answer never overrides.
        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE orders ADD COLUMN placed_at TEXT")
        orders, customers = await _results(agent)
        await settled(agent)
        cached = await agent._embedded._data_view.cached_description(
            agent.id, source.id, "orders"
        )
        assert cached is not None
        columns = cast(Sequence[Mapping[str, object]], cached["columns"])
        assert columns[-1]["name"] == "placed_at"
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
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE secrets")
        (result,) = await _results(agent)
        assert result.is_error
        error = cast(Mapping[str, object], result.output["error"])
        assert error["code"] == "table_unknown"
        await settled(agent)
        names = {
            item.name
            for item in await agent.list_catalog_resources(source_id=source.id)
        }
        assert "secrets" not in names
    finally:
        await agent.close()


async def test_looking_up_tables_that_have_not_changed_starts_no_refresh(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(
        tmp_path,
        _describe("orders"),
        _describe("open_orders", "view"),
        _describe("main", "listing"),
        _describe("secrets", "secrets"),
        _search("orders", "search"),
    )
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        results = await _results(agent)
        assert not any(result.is_error for result in results)
        await settled(agent)
        assert sync_count(agent) == 1
    finally:
        await agent.close()


async def test_a_name_neither_the_source_nor_the_catalog_knows_starts_no_refresh(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _describe("nothing_here"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        (result,) = await _results(agent)
        assert result.is_error
        await settled(agent)
        assert sync_count(agent) == 1
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


_ESTIMATES = {"main.orders": 1_000, "main.customers": 50, "main.products": 7}


def _crm(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE products (sku_code TEXT, label TEXT);
            CREATE TABLE notes (note_key TEXT);
            """)
    return path


def _shop_with_keys(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (id INTEGER PRIMARY KEY);
            CREATE TABLE orders (
              id INTEGER PRIMARY KEY,
              customer_id INTEGER REFERENCES customers(id),
              sku_code TEXT,
              note_key INTEGER
            );
            """)
    return path


def _with_estimates(agent: Agent) -> None:
    """SQLite reports no row counts, so give the real catalog some."""

    service = agent._embedded._catalog_service
    real = service.tabular_resources

    async def estimated(
        agent_id: str, source_id: str, **lookup: object
    ) -> tuple[object, ...]:
        return tuple(
            replace(
                item,
                tabular=replace(
                    item.tabular,
                    row_count_estimate=_ESTIMATES.get(item.resource.native_identity),
                ),
            )
            for item in await real(agent_id, source_id, **lookup)  # type: ignore[arg-type]
        )

    service.tabular_resources = estimated  # type: ignore[method-assign,assignment]


def _candidates(result: ToolResultBlock) -> list[Mapping[str, object]]:
    return list(
        cast(Sequence[Mapping[str, object]], _data(result).get("link_candidates", ()))
    )


async def _two_sources(
    agent: Agent, tmp_path: Path
) -> tuple[SourceRegistration, SourceRegistration]:
    shop = await agent.attach(
        SQLiteSource(_shop_with_keys(tmp_path / "shop.sqlite"), name="Shop")
    )
    crm = await agent.attach(SQLiteSource(_crm(tmp_path / "crm.sqlite"), name="CRM"))
    return shop, crm


async def test_describe_suggests_unconfirmed_joins_to_other_sources(
    tmp_path: Path,
) -> None:
    agent, provider = await _agent(tmp_path, _describe("orders"))
    try:
        _with_estimates(agent)
        await _two_sources(agent, tmp_path)
        (result,) = await _results(agent)
        data = _data(result)
        # The foreign key is a hint with row counts from the catalog.
        (relationship,) = cast(Sequence[Mapping[str, object]], data["relationships"])
        assert relationship["from_row_estimate"] == 1_000
        assert relationship["to_row_estimate"] == 50
        # An equal name comes first, then the id convention; id alone, another
        # type and the same source's customers table give nothing.
        assert [
            (item["column"], item["to"], item["to_column"], item.get("to_row_estimate"))
            for item in _candidates(result)
        ] == [
            ("sku_code", "main.products", "sku_code", 7),
            ("customer_id", "main.customers", "id", 50),
        ]
        assert {item["status"] for item in _candidates(result)} == {"unconfirmed"}
        assert await agent.list_catalog_links() == ()
        rendered = canonical_json(
            model_tool_result(result, None, provider.requests[-1].scope_names)
        )
        assert '"to_source":"CRM"' in rendered
        assert "sha256:" not in rendered.replace(str(data.get("revision")), "")
        # An older stored result has no candidates and still validates.
        old = {k: v for k, v in data.items() if k != "link_candidates"}
        agent._embedded._capabilities.validate_output(
            CATALOG_DESCRIBE_CAPABILITY_ID,
            ToolOutput(kind=CATALOG_DESCRIBE_EVIDENCE_KIND, data=old),
        )
    finally:
        await agent.close()


async def test_a_more_sensitive_candidate_source_raises_the_describe_sensitivity(
    tmp_path: Path,
) -> None:
    agent, _provider = await _agent(tmp_path / "home", _describe("orders"))
    try:
        await agent.attach(
            SQLiteSource(_shop_with_keys(tmp_path / "shop.sqlite"), name="Shop")
        )
        folder = tmp_path / "restricted"
        folder.mkdir()
        (folder / "products.csv").write_text("sku_code,label\nA1,Anvil\n", "utf-8")
        restricted = await agent.attach(
            FolderSource(
                folder, name="Restricted", sensitivity=ModelSensitivity.RESTRICTED
            )
        )
        (result,) = await _results(agent)
        candidates = _candidates(result)
        assert any(item["to_source_id"] == restricted.id for item in candidates)
        # The candidate's source is not a declared link, yet it counts.
        assert result.sensitivity is ModelSensitivity.RESTRICTED
        assert result.sensitivity_provenance is not None
        assert restricted.id in cast(
            tuple[str, ...], result.sensitivity_provenance["source_ids"]
        )
    finally:
        await agent.close()


async def test_describe_shows_estimates_on_declared_links_and_does_not_repeat_them(
    tmp_path: Path,
) -> None:
    agent, _provider = await _agent(tmp_path, _describe("orders"))
    try:
        _with_estimates(agent)
        shop, crm = await _two_sources(agent, tmp_path)

        async def resource(source_id: str, name: str) -> str:
            return next(
                item.id
                for item in await agent.list_catalog_resources(source_id=source_id)
                if item.name == name
            )

        await agent.declare_catalog_link(
            await resource(shop.id, "orders"),
            await resource(crm.id, "customers"),
            (("customer_id", "id"),),
        )
        (result,) = await _results(agent)
        declared = next(
            item
            for item in cast(
                Sequence[Mapping[str, object]], _data(result)["relationships"]
            )
            if item.get("provenance") == "declared"
        )
        assert (declared["from_row_estimate"], declared["to_row_estimate"]) == (
            1_000,
            50,
        )
        assert [item["column"] for item in _candidates(result)] == ["sku_code"]
    finally:
        await agent.close()


async def test_describe_suggests_nothing_from_unreadable_sources_or_a_failing_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _provider = await _agent(tmp_path, _describe("orders", "scoped"))
    try:
        _shop, crm = await _two_sources(agent, tmp_path)
        products = next(
            item
            for item in await agent.list_catalog_resources(source_id=crm.id)
            if item.name == "products"
        )
        preview = await agent.preview_source_permissions(
            source_id=crm.id,
            read_mode="selected",
            read_resource_ids=(products.id,),
            relational_write_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=crm.id, confirmation_fingerprint=preview.confirmation_fingerprint
        )
        (scoped,) = await _results(agent)
        assert [item["to"] for item in _candidates(scoped)] == ["main.products"]
    finally:
        await agent.close()

    broken = tmp_path / "broken"
    broken.mkdir()
    agent, _provider = await _agent(broken, _describe("orders"))
    try:
        await _two_sources(agent, broken)
        for owner in (agent._embedded._catalog_service, agent._embedded._data_view):
            monkeypatch.setattr(owner, "_store", _BrokenCatalogStore())
        (result,) = await _results(agent)
        assert _data(result)["answered_from"] == "live"
        assert _candidates(result) == []
    finally:
        await agent.close()


def _with_archive(path: Path) -> Path:
    with sqlite3.connect(_database(path)) as connection:
        connection.executescript("""
            CREATE TABLE orders_archive (
              id INTEGER, customer_id INTEGER, status TEXT NOT NULL
            );
            CREATE TABLE order_items (id INTEGER, order_id INTEGER, sku TEXT);
            """)
    return path


async def test_search_labels_look_alike_tables_and_hides_nothing(
    tmp_path: Path,
) -> None:
    database = _with_archive(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _search("order"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        (result,) = await _results(agent)
        hits = _hits(result)
        assert {"main.orders", "main.orders_archive", "main.order_items"} <= set(hits)
        assert hits["main.orders"]["status"] == "unknown"
        assert hits["main.orders"]["look_alikes"] == ("main.orders_archive",)
        assert "status" not in hits["main.order_items"]
    finally:
        await agent.close()


async def test_describe_names_the_look_alikes_of_one_table(tmp_path: Path) -> None:
    database = _with_archive(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _describe("orders"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        (result,) = await _results(agent)
        data = _data(result)
        assert data["status"] == "unknown"
        (other,) = cast(Sequence[Mapping[str, object]], data["look_alikes"])
        assert other["name"] == "main.orders_archive"
        assert other["status"] == "unknown"
    finally:
        await agent.close()


async def test_a_failing_label_lookup_leaves_describe_unlabelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _with_archive(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _describe("orders"))
    try:
        await agent.attach(SQLiteSource(database, name="Shop"))
        monkeypatch.setattr(agent._embedded._data_view, "_store", _BrokenCatalogStore())
        (result,) = await _results(agent)
        data = _data(result)
        assert data["answered_from"] == "live"
        assert "status" not in data and "look_alikes" not in data
    finally:
        await agent.close()


async def test_a_current_note_marks_a_look_alike_deprecated(tmp_path: Path) -> None:
    database = _with_archive(tmp_path / "shop.sqlite")
    agent, _provider = await _agent(tmp_path, _search("order"))
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        resource = next(
            item
            for item in await agent.list_catalog_resources(source_id=source.id)
            if item.name == "orders_archive"
        )
        now = datetime(2026, 11, 6, tzinfo=UTC)
        await agent._embedded._store.save_semantic_annotation(
            agent.id,
            SemanticAnnotation(
                id="archive-note",
                agent_id=agent.id,
                subject=SemanticSubject((source.id,), (resource.id,)),
                kind=SemanticKind.GLOSSARY,
                statement="Deprecated: orders live in orders now.",
                evidence=(
                    SemanticEvidence(SemanticEvidenceKind.USER_ASSERTION, "run", 0),
                ),
                catalog_revisions=(
                    ResourceRevisionBinding(resource.id, resource.current_revision),
                ),
                created_at=now,
                confirmed_at=now,
            ),
        )
        (result,) = await _results(agent)
        hits = _hits(result)
        assert hits["main.orders_archive"]["status"] == "deprecated"
        assert hits["main.orders"]["status"] == "canonical"
    finally:
        await agent.close()
