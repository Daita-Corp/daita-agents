"""Component-owned tests split from ``test_schema_slices.py``."""

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
    canonical_json,
    catalog_service,
    pytest,
    replace,
    sqlite3,
    sqlite_store,
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


async def test_connected_schema_reuses_one_compilation_per_exact_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "schema-compilation.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-compilation", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = {
            resource.name: resource
            for resource in await agent.list_catalog_resources(source_id=source.id)
        }
        original_compile = catalog_service._compile_source_index
        compile_count = 0

        def counting_compile(snapshot):
            nonlocal compile_count
            compile_count += 1
            return original_compile(snapshot)

        monkeypatch.setattr(catalog_service, "_compile_source_index", counting_compile)
        request_ids = (resources["customers"].id, resources["products"].id)
        first = await _schema(agent, resource_ids=request_ids, limit=5)
        second = await _schema(agent, resource_ids=request_ids, limit=5)
        assert first == second
        assert compile_count == 1
    finally:
        await agent.close()


async def test_schema_slice_reopens_with_one_decode_and_identical_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    original_decode = sqlite_store.decode_catalog_snapshot
    decode_count = 0

    def counting_decode(value: str):
        nonlocal decode_count
        decode_count += 1
        return original_decode(value)

    monkeypatch.setattr(sqlite_store, "decode_catalog_snapshot", counting_decode)
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
        assert decode_count == 1
    finally:
        await reopened.close()


async def test_schema_slice_retries_a_generation_conflict_only_once(
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
        store = agent._embedded._store
        load_count = 0

        async def changing_generation(ref):
            nonlocal load_count
            load_count += 1
            return None

        monkeypatch.setattr(store, "load_current_snapshot", changing_generation)
        with pytest.raises(
            CatalogStoreError,
            match="generation changed repeatedly",
        ):
            await _schema(agent, resource_ids=(resource.id,))
        assert load_count == 2
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
        refs = await store.list_current_snapshot_refs(agent.id, (source.id,))
        assert len(refs) == 1
        snapshot = await store.load_current_snapshot(refs[0])
        assert snapshot is not None
        facet = next(item for item in snapshot.facets if item.kind.value == "tabular")
        decoded = TabularFacet.from_payload(facet.payload)
        changed_column = replace(decoded.columns[0], native_type="changed_type")
        changed_facet = TabularFacet(
            columns=(changed_column, *decoded.columns[1:]),
            indexes=decoded.indexes,
            row_count_estimate=decoded.row_count_estimate,
        )
        mismatched_facet = replace(facet, payload=changed_facet.to_payload())
        mismatched_snapshot = replace(
            snapshot,
            facets=tuple(
                mismatched_facet if item is facet else item for item in snapshot.facets
            ),
        )

        async def load_mismatched(_ref):
            return mismatched_snapshot

        service._source_indexes.clear()
        monkeypatch.setattr(store, "load_current_snapshot", load_mismatched)
        with pytest.raises(CatalogStoreError, match="revision does not match"):
            await service.tabular_resources(agent.id, source.id)
    finally:
        await agent.close()


async def test_validation_schema_reopen_decodes_and_compiles_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "validation-reopen.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-validation-reopen", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await agent.attach(SQLiteSource(database))
    agent_id = agent.id
    source_id = source.id
    await agent.close()

    original_decode = sqlite_store.decode_catalog_snapshot
    original_compile = catalog_service._compile_source_index
    decode_count = 0
    compile_count = 0

    def counting_decode(value: str):
        nonlocal decode_count
        decode_count += 1
        return original_decode(value)

    def counting_compile(snapshot):
        nonlocal compile_count
        compile_count += 1
        return original_compile(snapshot)

    monkeypatch.setattr(sqlite_store, "decode_catalog_snapshot", counting_decode)
    monkeypatch.setattr(catalog_service, "_compile_source_index", counting_compile)
    reopened = await Agent.open(
        "catalog-validation-reopen", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        store = reopened._embedded._store
        snapshot_read_count = 0
        original_snapshot = store.load_current_snapshot

        async def counting_snapshot(ref):
            nonlocal snapshot_read_count
            snapshot_read_count += 1
            return await original_snapshot(ref)

        async def legacy_read(*args, **kwargs):
            pytest.fail("validation schemas performed a superseded store read")

        monkeypatch.setattr(store, "load_current_snapshot", counting_snapshot)
        for method_name in (
            "list_resources",
            "load_resource",
            "load_revision",
            "load_facets",
            "load_sync",
        ):
            monkeypatch.setattr(store, method_name, legacy_read)

        first = await reopened._embedded._data_view.resource_schemas(
            agent_id,
            source_id,
        )
        second = await reopened._embedded._data_view.resource_schemas(
            agent_id,
            source_id,
        )
        assert first == second
        assert decode_count == 1
        assert compile_count == 1
        assert snapshot_read_count == 1
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
        assert all(
            key != (agent.id, source.id, old_current.sync.id)
            for key in service._source_indexes
        )
    finally:
        await agent.close()


async def test_validation_projection_retries_one_generation_conflict(
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
        store = agent._embedded._store
        original_snapshot = store.load_current_snapshot
        load_count = 0

        async def one_conflict(ref):
            nonlocal load_count
            load_count += 1
            if load_count == 1:
                return None
            return await original_snapshot(ref)

        monkeypatch.setattr(store, "load_current_snapshot", one_conflict)
        facts = await agent._embedded._catalog_service.tabular_resources(
            agent.id,
            source.id,
        )
        assert facts
        assert load_count == 2
    finally:
        await agent.close()


async def test_validation_projection_reports_repeated_generation_conflict(
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
        store = agent._embedded._store
        load_count = 0

        async def changing_generation(ref):
            nonlocal load_count
            load_count += 1
            return None

        monkeypatch.setattr(store, "load_current_snapshot", changing_generation)
        with pytest.raises(
            CatalogStoreError,
            match="generation changed repeatedly",
        ):
            await agent._embedded._catalog_service.tabular_resources(
                agent.id,
                source.id,
            )
        assert load_count == 2
    finally:
        await agent.close()
