"""Component-owned tests split from ``test_schema_slices.py``."""

from __future__ import annotations

from tests.catalog._schema_support import (
    Agent,
    CatalogStoreError,
    Path,
    ResourceKind,
    SQLiteSource,
    TabularFacet,
    _commit_schema_graph,
    catalog_service,
    pytest,
    sqlite3,
    workspace_for,
)


async def test_validation_schemas_use_one_bulk_current_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "validation-bulk.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE accounts (
                account_id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'active'
            );
            CREATE VIEW account_lookup AS
            SELECT account_id, email FROM accounts;
        """)
    agent = await Agent.create(
        "catalog-validation-bulk", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        service = agent._embedded._catalog_service
        data_view = agent._embedded._data_view
        store = agent._embedded._store
        original_bulk = service.tabular_resources
        original_refs = store.list_current_snapshot_refs
        original_snapshot = store.load_current_snapshot
        original_compile = catalog_service._compile_source_index
        counts = {
            "bulk": 0,
            "refs": 0,
            "snapshot": 0,
            "compile": 0,
        }
        projected = []

        async def counting_bulk(agent_id: str, source_id: str):
            counts["bulk"] += 1
            result = await original_bulk(agent_id, source_id)
            projected.append(result)
            return result

        async def counting_refs(agent_id: str, source_ids: tuple[str, ...]):
            counts["refs"] += 1
            return await original_refs(agent_id, source_ids)

        async def counting_snapshot(ref):
            counts["snapshot"] += 1
            return await original_snapshot(ref)

        def counting_compile(snapshot):
            counts["compile"] += 1
            return original_compile(snapshot)

        async def legacy_read(*args, **kwargs):
            pytest.fail("validation schemas performed a superseded store read")

        monkeypatch.setattr(service, "tabular_resources", counting_bulk)
        monkeypatch.setattr(store, "list_current_snapshot_refs", counting_refs)
        monkeypatch.setattr(store, "load_current_snapshot", counting_snapshot)
        monkeypatch.setattr(catalog_service, "_compile_source_index", counting_compile)
        for method_name in (
            "list_resources",
            "load_resource",
            "load_revision",
            "load_facets",
            "load_sync",
        ):
            monkeypatch.setattr(store, method_name, legacy_read)

        first = await data_view.resource_schemas(agent.id, source.id)
        second = await data_view.resource_schemas(agent.id, source.id)

        assert first == second
        assert counts == {
            "bulk": 2,
            "refs": 4,
            "snapshot": 1,
            "compile": 1,
        }
        assert tuple(item.name for item in first) == ("account_lookup", "accounts")
        by_name = {item.name: item for item in first}
        assert by_name["accounts"].columns == ("account_id", "email", "status")
        assert by_name["accounts"].unique_key_columns == ("account_id", "email")
        assert by_name["accounts"].column_declared_types == (
            ("account_id", "INTEGER"),
            ("email", "TEXT"),
            ("status", "TEXT"),
        )
        assert by_name["accounts"].aliases == ("main.accounts",)
        assert by_name["accounts"].resource_kind == "table"
        assert by_name["accounts"].writable is True
        assert by_name["account_lookup"].resource_kind == "view"
        assert by_name["account_lookup"].writable is False

        assert len(projected) == 2
        facts = projected[0]
        assert tuple(item.resource.name for item in facts) == (
            "account_lookup",
            "accounts",
        )
        for item in facts:
            assert item.sync.agent_id == agent.id
            assert item.sync.source_id == source.id
            assert item.resource.agent_id == agent.id
            assert item.resource.source_id == source.id
            assert item.resource.current_sync_id == item.sync.id
            assert item.revision.resource_id == item.resource.id
            assert item.revision.revision == item.resource.current_revision
            assert item.revision.sync_id == item.sync.id
            assert item.facet.resource_id == item.resource.id
            assert item.facet.sync_id == item.sync.id
            assert item.facet.revision in item.revision.facet_revisions
            assert item.facet.kind.value == "tabular"
            assert item.tabular == TabularFacet.from_payload(item.facet.payload)
            assert item.sync.source_revision == item.revision.source_revision
    finally:
        await agent.close()


async def test_validation_projection_enforces_active_source_and_agent_isolation(
    tmp_path: Path,
):
    first_database = tmp_path / "validation-first.sqlite"
    second_database = tmp_path / "validation-second.sqlite"
    with sqlite3.connect(first_database) as connection:
        connection.execute("CREATE TABLE first_private (id INTEGER PRIMARY KEY)")
    with sqlite3.connect(second_database) as connection:
        connection.execute("CREATE TABLE second_private (id INTEGER PRIMARY KEY)")
    first = await Agent.create(
        "catalog-validation-first", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    second = await Agent.create(
        "catalog-validation-second", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        first_source = await first.attach(SQLiteSource(first_database))
        first_other_source = await first.attach(SQLiteSource(second_database))
        second_source = await second.attach(SQLiteSource(second_database))
        first_service = first._embedded._catalog_service
        facts = await first_service.tabular_resources(first.id, first_source.id)
        assert tuple(item.resource.name for item in facts) == ("first_private",)
        assert all(item.resource.agent_id == first.id for item in facts)
        assert all(item.resource.source_id == first_source.id for item in facts)
        other_facts = await first_service.tabular_resources(
            first.id,
            first_other_source.id,
        )
        assert tuple(item.resource.name for item in other_facts) == ("second_private",)
        assert all(
            item.resource.source_id == first_other_source.id for item in other_facts
        )

        with pytest.raises(CatalogStoreError, match="unknown active catalog source"):
            await first_service.tabular_resources(second.id, first_source.id)
        with pytest.raises(CatalogStoreError, match="unknown active catalog source"):
            await first_service.tabular_resources(first.id, second_source.id)

        await first.detach(first_source.id)
        with pytest.raises(CatalogStoreError, match="unknown active catalog source"):
            await first_service.tabular_resources(first.id, first_source.id)
        assert all(
            key[:2] != (first.id, first_source.id)
            for key in first_service._source_indexes
        )
    finally:
        await first.close()
        await second.close()


async def test_validation_projection_translates_tabular_resource_kinds_and_order(
    tmp_path: Path,
):
    database = tmp_path / "validation-kinds.sqlite"
    database.touch()
    agent = await Agent.create(
        "catalog-validation-kinds", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        nodes = {
            "zeta": ("warehouse.zeta", ResourceKind.TABLE, True),
            "alpha_events": ("alpha.events", ResourceKind.TABLE, True),
            "beta_events": ("beta.events", ResourceKind.VIEW, True),
            "csv": ("files/records.csv", ResourceKind.FILE, True),
            "json": ("files/records.json", ResourceKind.FILE, True),
            "blob": ("files/blob.json", ResourceKind.FILE, False),
        }
        await _commit_schema_graph(
            agent,
            source.id,
            nodes=nodes,
            edges=(),
            sync_id="validation-kinds-1",
        )
        service = agent._embedded._catalog_service
        data_view = agent._embedded._data_view
        first_facts = await service.tabular_resources(agent.id, source.id)
        first_schemas = await data_view.resource_schemas(agent.id, source.id)

        assert tuple(item.resource.native_identity for item in first_facts) == tuple(
            item.aliases[0] for item in first_schemas
        )
        assert tuple(item.name for item in first_schemas) == (
            "csv",
            "events",
            "events",
            "json",
            "zeta",
        )
        event_schemas = tuple(item for item in first_schemas if item.name == "events")
        assert len(event_schemas) == 2
        assert len({item.resource_id for item in event_schemas}) == 2
        assert {item.aliases for item in event_schemas} == {
            ("alpha.events",),
            ("beta.events",),
        }
        assert {item.resource_kind for item in first_schemas} == {
            "file",
            "table",
            "view",
        }
        assert "files/blob.json" not in {
            item.resource.native_identity for item in first_facts
        }

        await _commit_schema_graph(
            agent,
            source.id,
            nodes=dict(reversed(tuple(nodes.items()))),
            edges=(),
            sync_id="validation-kinds-2",
        )
        second_schemas = await data_view.resource_schemas(agent.id, source.id)
        assert tuple(
            (item.name, item.aliases, item.resource_id, item.columns)
            for item in second_schemas
        ) == tuple(
            (item.name, item.aliases, item.resource_id, item.columns)
            for item in first_schemas
        )
    finally:
        await agent.close()


async def test_validation_projection_empty_source_is_bounded_without_source_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "validation-empty.sqlite"
    database.touch()
    agent = await Agent.create(
        "catalog-validation-empty", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        database.rename(tmp_path / "validation-empty-unavailable.sqlite")
        store = agent._embedded._store

        async def legacy_read(*args, **kwargs):
            pytest.fail("empty validation projection used a per-resource store read")

        for method_name in (
            "list_resources",
            "load_resource",
            "load_revision",
            "load_facets",
            "load_sync",
        ):
            monkeypatch.setattr(store, method_name, legacy_read)
        assert (
            await agent._embedded._data_view.resource_schemas(agent.id, source.id) == ()
        )
    finally:
        await agent.close()
