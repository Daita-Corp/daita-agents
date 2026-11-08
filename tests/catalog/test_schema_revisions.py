"""Component-owned tests split from ``test_describe.py``."""

from __future__ import annotations

from tests.catalog._schema_support import (
    Agent,
    CatalogStoreError,
    Path,
    SQLiteSource,
    TabularFacet,
    _fixture_database,
    _mapping,
    _mapping_sequence,
    _path_name_signatures,
    _schema,
    _shift_generation,
    canonical_json,
    pytest,
    replace,
    sqlite3,
    workspace_for,
)


async def test_schema_refresh_filters_old_resources_and_carries_new_revisions(
    tmp_path: Path,
):
    database = tmp_path / "refresh.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE current_table (id INTEGER PRIMARY KEY, value TEXT)"
        )
    agent = await Agent.create(
        "catalog-schema-refresh", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        old_resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        old_projection = await _schema(agent, resource_ids=(old_resource.id,))
        old_revision = _mapping_sequence(old_projection["resources"])[0]["revision"]

        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE current_table ADD COLUMN later TEXT")
        await agent.refresh_source(source.id)
        refreshed = await agent.list_catalog_resources(source_id=source.id)
        assert refreshed[0].id == old_resource.id
        new_projection = await _schema(agent, resource_ids=(old_resource.id,))
        current = _mapping_sequence(new_projection["resources"])[0]
        assert current["revision"] != old_revision
        assert tuple(
            column["name"] for column in _mapping_sequence(current["columns"])
        ) == (
            "id",
            "value",
            "later",
        )
    finally:
        await agent.close()


async def test_connected_schema_refresh_removes_stale_join_paths(tmp_path: Path):
    database = tmp_path / "stale-path.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE parent (id INTEGER PRIMARY KEY);
            CREATE TABLE bridge (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER NOT NULL REFERENCES parent(id)
            );
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                bridge_id INTEGER NOT NULL REFERENCES bridge(id)
            );
        """)
    agent = await Agent.create(
        "catalog-schema-stale-path", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = {
            resource.name: resource
            for resource in await agent.list_catalog_resources(source_id=source.id)
        }
        initial = await _schema(
            agent,
            resource_ids=(resources["parent"].id, resources["child"].id),
            limit=3,
        )
        assert _path_name_signatures(initial) == (
            ("main.child", "main.bridge", "main.parent"),
        )

        with sqlite3.connect(database) as connection:
            connection.executescript("""
                PRAGMA foreign_keys = OFF;
                DROP TABLE child;
                CREATE TABLE child (
                    id INTEGER PRIMARY KEY,
                    bridge_id INTEGER NOT NULL
                );
            """)
        await agent.refresh_source(source.id)
        refreshed = await _schema(
            agent,
            resource_ids=(resources["parent"].id, resources["child"].id),
            limit=3,
        )
        assert refreshed["paths"] == ()
        refreshed_selection = _mapping(refreshed["selection"])
        assert refreshed_selection["bridge_resource_ids"] == ()
        assert refreshed_selection["unresolved_reasons"] == ("no_path",)
    finally:
        await agent.close()


async def test_schema_slice_reopens_with_an_identical_payload(
    tmp_path: Path,
):
    database = tmp_path / "coherent-reopen.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-coherent-reopen",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database))
    resources = await agent.list_catalog_resources(source_id=source.id)
    resource_ids = tuple(resource.id for resource in resources)
    expected = canonical_json(await _schema(agent, resource_ids=resource_ids))
    await agent.close()

    reopened = await Agent.open(
        "catalog-schema-coherent-reopen",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await _schema(reopened, resource_ids=resource_ids)
        second = await _schema(reopened, resource_ids=resource_ids)
        assert canonical_json(first) == expected
        assert canonical_json(second) == expected
    finally:
        await reopened.close()


async def test_schema_slice_retries_a_generation_change_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "generation-conflict.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-generation-conflict",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        expected = await _schema(agent, resource_ids=(resource.id,))
        _shift_generation(monkeypatch, agent._embedded._store, reads=1)
        assert await _schema(agent, resource_ids=(resource.id,)) == expected

        _shift_generation(monkeypatch, agent._embedded._store)
        with pytest.raises(
            CatalogStoreError,
            match="generation changed repeatedly",
        ):
            await _schema(agent, resource_ids=(resource.id,))
    finally:
        await agent.close()


async def test_catalog_boundary_rejects_tabular_facet_revision_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "validation-revision-mismatch.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY)")
    agent = await Agent.create(
        "catalog-validation-revision-mismatch",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        store = agent._embedded._store
        service = agent._embedded._catalog_service
        entries = await store.list_entries(agent.id, source.id)
        assert len(entries) == 1
        entry = entries[0]
        facet = next(item for item in entry.facets if item.kind.value == "tabular")
        decoded = TabularFacet.from_payload(facet.payload)
        changed_column = replace(decoded.columns[0], native_type="changed_type")
        changed_facet = TabularFacet(
            columns=(changed_column, *decoded.columns[1:]),
            indexes=decoded.indexes,
            row_count_estimate=decoded.row_count_estimate,
        )
        mismatched_facet = replace(facet, payload=changed_facet.to_payload())
        mismatched = replace(
            entry,
            facets=tuple(
                mismatched_facet if item is facet else item for item in entry.facets
            ),
        )

        async def list_mismatched(_agent_id, _source_id):
            return (mismatched,)

        monkeypatch.setattr(store, "list_entries", list_mismatched)
        with pytest.raises(CatalogStoreError, match="revision does not match"):
            await service.tabular_resources(agent.id, source.id)
    finally:
        await agent.close()


async def test_validation_schemas_survive_reopen(
    tmp_path: Path,
):
    database = tmp_path / "validation-reopen.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-validation-reopen", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await agent.attach(SQLiteSource(database))
    agent_id = agent.id
    source_id = source.id
    expected = await agent._embedded._data_view.resource_schemas(agent_id, source_id)
    await agent.close()

    reopened = await Agent.open(
        "catalog-validation-reopen", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert expected
        assert (
            await reopened._embedded._data_view.resource_schemas(agent_id, source_id)
            == expected
        )
    finally:
        await reopened.close()


async def test_validation_schema_refresh_replaces_all_current_structural_facts(
    tmp_path: Path,
):
    database = tmp_path / "validation-refresh.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE current_table (
                id INTEGER PRIMARY KEY,
                code TEXT NOT NULL
            );
            CREATE TABLE stale_table (id INTEGER PRIMARY KEY, stale_value TEXT);
        """)
    agent = await Agent.create(
        "catalog-validation-refresh", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        service = agent._embedded._catalog_service
        data_view = agent._embedded._data_view
        old_facts = {
            item.resource.name: item
            for item in await service.tabular_resources(agent.id, source.id)
        }
        old_schemas = {
            item.name: item
            for item in await data_view.resource_schemas(agent.id, source.id)
        }

        with sqlite3.connect(database) as connection:
            connection.executescript("""
                ALTER TABLE current_table ADD COLUMN later TEXT;
                CREATE UNIQUE INDEX current_table_code_unique
                    ON current_table(code);
                DROP TABLE stale_table;
                CREATE TABLE replacement_table (
                    replacement_id INTEGER PRIMARY KEY,
                    value NUMERIC
                );
            """)
        await agent.refresh_source(source.id)

        new_facts = {
            item.resource.name: item
            for item in await service.tabular_resources(agent.id, source.id)
        }
        new_schemas = {
            item.name: item
            for item in await data_view.resource_schemas(agent.id, source.id)
        }
        assert set(old_facts) == {"current_table", "stale_table"}
        assert set(new_facts) == {"current_table", "replacement_table"}
        assert set(new_schemas) == {"current_table", "replacement_table"}
        assert "stale_table" not in new_schemas

        old_current = old_facts["current_table"]
        new_current = new_facts["current_table"]
        assert new_current.resource.id == old_current.resource.id
        assert new_current.sync.id != old_current.sync.id
        assert new_current.sync.source_revision != old_current.sync.source_revision
        assert new_current.resource.current_revision != (
            old_current.resource.current_revision
        )
        assert new_current.revision.revision == new_current.resource.current_revision
        assert new_current.revision.sync_id == new_current.sync.id
        assert new_current.facet.sync_id == new_current.sync.id
        assert tuple(
            column["name"]
            for column in _mapping_sequence(new_current.facet.payload["columns"])
        ) == ("id", "code", "later")
        assert old_schemas["current_table"].unique_key_columns == ("id",)
        assert new_schemas["current_table"].unique_key_columns == ("id", "code")
        assert new_schemas["current_table"].columns == ("id", "code", "later")
        assert new_schemas["current_table"].revision == (
            new_current.resource.current_revision
        )
        assert new_schemas["current_table"].source_revision == (
            new_current.sync.source_revision
        )
    finally:
        await agent.close()


async def test_validation_projection_retries_one_generation_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "validation-transient-conflict.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-validation-transient-conflict",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        _shift_generation(monkeypatch, agent._embedded._store, reads=1)
        facts = await agent._embedded._catalog_service.tabular_resources(
            agent.id,
            source.id,
        )
        assert facts
    finally:
        await agent.close()


async def test_validation_projection_reports_repeated_generation_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "validation-repeated-conflict.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-validation-repeated-conflict",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        _shift_generation(monkeypatch, agent._embedded._store)
        with pytest.raises(
            CatalogStoreError,
            match="generation changed repeatedly",
        ):
            await agent._embedded._catalog_service.tabular_resources(
                agent.id,
                source.id,
            )
    finally:
        await agent.close()
