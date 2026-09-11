"""Component-owned tests split from ``test_schema_slices.py``."""

from __future__ import annotations

from tests.catalog._schema_support import (
    Agent,
    CatalogResourceNotFoundError,
    FrozenJsonObject,
    Mapping,
    Path,
    RelationshipProvenance,
    ResourceKind,
    SQLiteSource,
    _commit_schema_graph,
    _fixture_database,
    _mapping,
    _mapping_sequence,
    _object_sequence,
    _path_name_signatures,
    _schema,
    _SchemaEdge,
    catalog_service,
    pytest,
    sqlite3,
    workspace_for,
)


async def test_connected_schema_selects_direct_and_required_bridge_paths(
    tmp_path: Path,
):
    database = tmp_path / "connected.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-connected", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = {
            resource.name: resource
            for resource in await agent.list_catalog_resources(source_id=source.id)
        }

        direct = await _schema(
            agent,
            resource_ids=(resources["orders"].id, resources["customers"].id),
            limit=4,
        )
        assert _path_name_signatures(direct) == (("main.customers", "main.orders"),)
        direct_roles = {
            item["name"]: item["selection_role"]
            for item in _mapping_sequence(direct["resources"])
        }
        assert direct_roles["main.customers"] == "seed"
        assert direct_roles["main.orders"] == "seed"
        direct_path = _mapping_sequence(direct["paths"])[0]
        assert len(_object_sequence(direct_path["relationship_ids"])) == 1
        assert set(_object_sequence(direct_path["seed_resource_ids"])) == {
            resources["customers"].id,
            resources["orders"].id,
        }

        bridged = await _schema(
            agent,
            resource_ids=(resources["customers"].id, resources["products"].id),
            limit=5,
        )
        assert _path_name_signatures(bridged) == (
            (
                "main.customers",
                "main.orders",
                "main.order_items",
                "main.products",
            ),
        )
        roles = {
            item["name"]: item["selection_role"]
            for item in _mapping_sequence(bridged["resources"])
        }
        assert roles == {
            "main.customers": "seed",
            "main.order_items": "bridge",
            "main.orders": "bridge",
            "main.products": "seed",
        }
        selection = bridged["selection"]
        assert isinstance(selection, Mapping)
        assert set(selection["seed_resource_ids"]) == {
            resources["customers"].id,
            resources["products"].id,
        }
        assert set(selection["bridge_resource_ids"]) == {
            resources["orders"].id,
            resources["order_items"].id,
        }
        assert selection["unresolved_reasons"] == ()
    finally:
        await agent.close()


async def test_connected_schema_builds_deterministic_three_seed_join_tree(
    tmp_path: Path,
):
    database = tmp_path / "join-tree.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-join-tree", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = {
            resource.name: resource
            for resource in await agent.list_catalog_resources(source_id=source.id)
        }
        projection = await _schema(
            agent,
            resource_ids=(
                resources["products"].id,
                resources["regions"].id,
                resources["customers"].id,
            ),
            limit=6,
        )

        paths = _path_name_signatures(projection)
        assert paths == (
            ("main.customers", "main.regions"),
            (
                "main.customers",
                "main.orders",
                "main.order_items",
                "main.products",
            ),
        )
        relationships = _mapping_sequence(projection["relationships"])
        path_relationship_ids = {
            relationship_id
            for path in _mapping_sequence(projection["paths"])
            for relationship_id in _object_sequence(path["relationship_ids"])
        }
        assert path_relationship_ids <= {
            relationship["relationship_id"] for relationship in relationships
        }
        assert _mapping(projection["selection"])["unresolved_reasons"] == ()
    finally:
        await agent.close()


async def test_connected_schema_three_seed_tree_reuses_one_shared_bridge(
    tmp_path: Path,
):
    database = tmp_path / "shared-bridge.sqlite"
    database.touch()
    agent = await Agent.create(
        "catalog-schema-shared-bridge", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resource_ids, relationships = await _commit_schema_graph(
            agent,
            source.id,
            nodes={
                "a": ("main.a", ResourceKind.TABLE, True),
                "b": ("main.b", ResourceKind.TABLE, True),
                "c": ("main.c", ResourceKind.TABLE, True),
                "hub": ("main.hub", ResourceKind.TABLE, True),
            },
            edges=(
                _SchemaEdge("a", "hub"),
                _SchemaEdge("b", "hub"),
                _SchemaEdge("c", "hub"),
            ),
            sync_id="shared-bridge-1",
        )
        projection = await _schema(
            agent,
            resource_ids=(resource_ids["c"], resource_ids["b"], resource_ids["a"]),
            limit=4,
        )

        first_terminal = "b" if relationships[1].id < relationships[2].id else "c"
        second_terminal = "c" if first_terminal == "b" else "b"
        assert _path_name_signatures(projection) == (
            ("main.a", "main.hub", f"main.{first_terminal}"),
            ("main.hub", f"main.{second_terminal}"),
        )
        roles = {
            resource["name"]: resource["selection_role"]
            for resource in _mapping_sequence(projection["resources"])
        }
        assert roles == {
            "main.a": "seed",
            "main.b": "seed",
            "main.c": "seed",
            "main.hub": "bridge",
        }
        assert _mapping(projection["selection"])["bridge_resource_ids"] == (
            resource_ids["hub"],
        )
    finally:
        await agent.close()


async def test_connected_schema_labels_bounded_single_seed_neighbors(tmp_path: Path):
    database = tmp_path / "neighbors.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-neighbors", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        customer = next(
            resource
            for resource in await agent.list_catalog_resources(source_id=source.id)
            if resource.name == "customers"
        )
        projection = await _schema(
            agent,
            resource_ids=(customer.id,),
            limit=3,
        )
        roles = tuple(
            resource["selection_role"]
            for resource in _mapping_sequence(projection["resources"])
        )
        assert roles.count("seed") == 1
        assert roles.count("neighbor") == 2
        assert projection["paths"] == ()
    finally:
        await agent.close()


async def test_connected_schema_query_seeds_are_diversified_and_keep_match_terms(
    tmp_path: Path,
):
    database = tmp_path / "query-seeds.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-query-seeds", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        await agent.attach(SQLiteSource(database))
        projection = await _schema(
            agent,
            query="customers products",
            limit=5,
        )

        roles = {
            item["name"]: item["selection_role"]
            for item in _mapping_sequence(projection["resources"])
        }
        assert roles == {
            "main.customers": "seed",
            "main.order_items": "bridge",
            "main.orders": "bridge",
            "main.products": "seed",
        }
        assert _path_name_signatures(projection) == (
            (
                "main.customers",
                "main.orders",
                "main.order_items",
                "main.products",
            ),
        )
        selection = projection["selection"]
        assert isinstance(selection, Mapping)
        assert selection["covered_terms"] == ("customers", "products")
        assert selection["unresolved_terms"] == ()
        seed_resources = tuple(
            resource
            for resource in _mapping_sequence(projection["resources"])
            if resource["selection_role"] == "seed"
        )
        assert {
            term
            for resource in seed_resources
            for term in _object_sequence(resource["matched_terms"])
        } == {
            "customers",
            "products",
        }
    finally:
        await agent.close()


async def test_connected_schema_preserves_composite_field_pairs_and_reverse_paths(
    tmp_path: Path,
):
    database = tmp_path / "composite.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE parent (
                tenant_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, entity_id)
            );
            CREATE TABLE zchild (
                child_id INTEGER PRIMARY KEY,
                tenant_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                FOREIGN KEY (tenant_id, entity_id)
                    REFERENCES parent (tenant_id, entity_id)
            );
        """)
    agent = await Agent.create(
        "catalog-schema-composite", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = {
            resource.name: resource
            for resource in await agent.list_catalog_resources(source_id=source.id)
        }
        projection = await _schema(
            agent,
            resource_ids=(resources["zchild"].id, resources["parent"].id),
            limit=2,
        )

        assert _path_name_signatures(projection) == (("main.parent", "main.zchild"),)
        relationship = _mapping_sequence(projection["relationships"])[0]
        assert tuple(
            (pair["source_field"], pair["target_field"])
            for pair in _mapping_sequence(relationship["field_pairs"])
        ) == (("tenant_id", "tenant_id"), ("entity_id", "entity_id"))
    finally:
        await agent.close()


async def test_connected_schema_uses_only_connector_paths_with_duplicate_names(
    tmp_path: Path,
):
    database = tmp_path / "connector-only.sqlite"
    database.touch()
    agent = await Agent.create(
        "catalog-schema-connector-only",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resource_ids, relationships = await _commit_schema_graph(
            agent,
            source.id,
            nodes={
                "current": ("public.orders", ResourceKind.TABLE, True),
                "archive": ("archive.orders", ResourceKind.TABLE, True),
                "connector_bridge": (
                    "public.order_links",
                    ResourceKind.TABLE,
                    True,
                ),
                "declared_bridge": (
                    "archive.order_links",
                    ResourceKind.TABLE,
                    True,
                ),
            },
            edges=(
                _SchemaEdge(
                    "current", "declared_bridge", RelationshipProvenance.DECLARED
                ),
                _SchemaEdge(
                    "declared_bridge", "archive", RelationshipProvenance.DECLARED
                ),
                _SchemaEdge("current", "connector_bridge"),
                _SchemaEdge("connector_bridge", "archive"),
            ),
            sync_id="connector-only-1",
        )

        explicit = await _schema(
            agent,
            resource_ids=(resource_ids["archive"], resource_ids["current"]),
            limit=3,
        )
        assert _path_name_signatures(explicit) == (
            ("archive.orders", "public.order_links", "public.orders"),
        )
        path_relationship_ids = set(
            _object_sequence(
                _mapping_sequence(explicit["paths"])[0]["relationship_ids"]
            )
        )
        connector_ids = {
            relationship.id
            for relationship in relationships
            if relationship.provenance is RelationshipProvenance.CONNECTOR
        }
        assert path_relationship_ids == connector_ids
        assert all(
            relationship["provenance"] == "connector"
            for relationship in _mapping_sequence(explicit["relationships"])
        )

        queried = await _schema(
            agent,
            query="public.orders archive.orders",
            limit=3,
        )
        assert set(
            _object_sequence(_mapping(queried["selection"])["seed_resource_ids"])
        ) == {
            resource_ids["current"],
            resource_ids["archive"],
        }
        assert _path_name_signatures(queried) == _path_name_signatures(explicit)
    finally:
        await agent.close()


async def test_connected_schema_paths_ignore_snapshot_insertion_order(tmp_path: Path):
    database = tmp_path / "path-order.sqlite"
    database.touch()
    agent = await Agent.create(
        "catalog-schema-path-order", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        nodes = {
            "a": ("main.a", ResourceKind.TABLE, True),
            "b": ("main.b", ResourceKind.TABLE, True),
            "left": ("main.left_bridge", ResourceKind.TABLE, True),
            "right": ("main.right_bridge", ResourceKind.TABLE, True),
        }
        edges = (
            _SchemaEdge("a", "left"),
            _SchemaEdge("left", "b"),
            _SchemaEdge("a", "right"),
            _SchemaEdge("right", "b"),
        )
        resource_ids, _ = await _commit_schema_graph(
            agent,
            source.id,
            nodes=nodes,
            edges=edges,
            sync_id="path-order-1",
        )
        first = await _schema(
            agent,
            resource_ids=(resource_ids["a"], resource_ids["b"]),
            limit=3,
        )

        reversed_resource_ids, _ = await _commit_schema_graph(
            agent,
            source.id,
            nodes=dict(reversed(tuple(nodes.items()))),
            edges=tuple(reversed(edges)),
            sync_id="path-order-2",
        )
        second = await _schema(
            agent,
            resource_ids=(reversed_resource_ids["a"], reversed_resource_ids["b"]),
            limit=3,
        )

        assert resource_ids == reversed_resource_ids
        assert _path_name_signatures(first) == _path_name_signatures(second)
        assert tuple(
            path["relationship_ids"] for path in _mapping_sequence(first["paths"])
        ) == tuple(
            path["relationship_ids"] for path in _mapping_sequence(second["paths"])
        )
    finally:
        await agent.close()


async def test_connected_schema_is_agent_isolated(tmp_path: Path):
    first_database = tmp_path / "agent-first.sqlite"
    second_database = tmp_path / "agent-second.sqlite"
    first_database.touch()
    second_database.touch()
    first_agent = await Agent.create(
        "catalog-schema-agent-first", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    second_agent = await Agent.create(
        "catalog-schema-agent-second", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        await first_agent.attach(SQLiteSource(first_database))
        second_source = await second_agent.attach(SQLiteSource(second_database))
        second_ids, _ = await _commit_schema_graph(
            second_agent,
            second_source.id,
            nodes={"private": ("main.private", ResourceKind.TABLE, True)},
            edges=(),
            sync_id="agent-isolation-1",
        )
        with pytest.raises(CatalogResourceNotFoundError):
            await _schema(
                first_agent,
                resource_ids=(second_ids["private"],),
            )
    finally:
        await first_agent.close()
        await second_agent.close()


async def test_connected_schema_rejects_non_tabular_file_paths(tmp_path: Path):
    database = tmp_path / "non-tabular.sqlite"
    database.touch()
    agent = await Agent.create(
        "catalog-schema-non-tabular", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resource_ids, _ = await _commit_schema_graph(
            agent,
            source.id,
            nodes={
                "left": ("main.left", ResourceKind.TABLE, True),
                "blob": ("files/blob.json", ResourceKind.FILE, False),
                "right": ("main.right", ResourceKind.TABLE, True),
            },
            edges=(
                _SchemaEdge("left", "blob"),
                _SchemaEdge("blob", "right"),
            ),
            sync_id="non-tabular-1",
        )

        through_file = await _schema(
            agent,
            resource_ids=(resource_ids["left"], resource_ids["right"]),
            limit=3,
        )
        assert through_file["paths"] == ()
        assert _mapping(through_file["selection"])["unresolved_reasons"] == ("no_path",)

        file_seed = await _schema(
            agent,
            resource_ids=(resource_ids["blob"], resource_ids["left"]),
            limit=2,
        )
        assert file_seed["paths"] == ()
        assert _mapping(file_seed["selection"])["unresolved_reasons"] == (
            "non_tabular_seed",
        )
    finally:
        await agent.close()


async def test_connected_schema_reports_no_path_depth_and_resource_bounds(
    tmp_path: Path,
):
    database = tmp_path / "unresolved.sqlite"
    _fixture_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE isolated (id INTEGER PRIMARY KEY)")
    agent = await Agent.create(
        "catalog-schema-unresolved", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = {
            resource.name: resource
            for resource in await agent.list_catalog_resources(source_id=source.id)
        }

        no_path = await _schema(
            agent,
            resource_ids=(resources["customers"].id, resources["isolated"].id),
            limit=4,
        )
        assert no_path["paths"] == ()
        assert _mapping(no_path["selection"])["unresolved_reasons"] == ("no_path",)
        assert _mapping(no_path["truncation"])["paths"] is False

        depth = await _schema(
            agent,
            resource_ids=(resources["customers"].id, resources["products"].id),
            limit=5,
            max_join_depth=2,
        )
        assert depth["paths"] == ()
        assert _mapping(depth["selection"])["unresolved_reasons"] == ("max_join_depth",)
        depth_truncation = _mapping(depth["truncation"])
        assert depth_truncation["paths"] is True
        assert depth_truncation["reason"] == "max_join_depth"

        resource_bound = await _schema(
            agent,
            resource_ids=(resources["customers"].id, resources["products"].id),
            limit=2,
        )
        assert resource_bound["paths"] == ()
        resource_selection = _mapping(resource_bound["selection"])
        assert resource_selection["bridge_resource_ids"] == ()
        assert resource_selection["unresolved_reasons"] == ("resource_limit",)
        resource_truncation = _mapping(resource_bound["truncation"])
        assert resource_truncation["resources"] is True
        assert resource_truncation["paths"] is True
        assert resource_truncation["reason"] == "resource_limit"
    finally:
        await agent.close()


async def test_connected_schema_reports_graph_and_relationship_work_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "work-bounds.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-work-bounds", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = {
            resource.name: resource
            for resource in await agent.list_catalog_resources(source_id=source.id)
        }

        monkeypatch.setattr(catalog_service, "_SCHEMA_JOIN_MAX_EDGES", 1)
        edge_bound = await _schema(
            agent,
            resource_ids=(resources["customers"].id, resources["products"].id),
            limit=5,
        )
        assert edge_bound["paths"] == ()
        assert _mapping(edge_bound["selection"])["unresolved_reasons"] == (
            "graph_edge_limit",
        )
        assert _mapping(edge_bound["truncation"])["reason"] == "graph_edge_limit"

        monkeypatch.setattr(catalog_service, "_SCHEMA_JOIN_MAX_EDGES", 200)
        monkeypatch.setattr(catalog_service, "_SCHEMA_JOIN_MAX_NODES", 2)
        node_bound = await _schema(
            agent,
            resource_ids=(resources["customers"].id, resources["products"].id),
            limit=5,
        )
        assert node_bound["paths"] == ()
        assert _mapping(node_bound["selection"])["unresolved_reasons"] == (
            "graph_node_limit",
        )
        assert _mapping(node_bound["truncation"])["reason"] == "graph_node_limit"

        monkeypatch.setattr(catalog_service, "_SCHEMA_JOIN_MAX_NODES", 100)
        monkeypatch.setattr(catalog_service, "_SCHEMA_RELATIONSHIP_LIMIT", 1)
        relationship_bound = await _schema(
            agent,
            resource_ids=(
                resources["customers"].id,
                resources["order_items"].id,
            ),
            limit=3,
        )
        assert relationship_bound["paths"] == ()
        assert _mapping(relationship_bound["selection"])["unresolved_reasons"] == (
            "relationship_limit",
        )
        relationship_truncation = _mapping(relationship_bound["truncation"])
        assert relationship_truncation["relationships"] is True
        assert relationship_truncation["reason"] == "relationship_limit"
    finally:
        await agent.close()


async def test_connected_schema_never_invents_cross_source_paths(tmp_path: Path):
    first_database = tmp_path / "cross-first.sqlite"
    second_database = tmp_path / "cross-second.sqlite"
    with sqlite3.connect(first_database) as connection:
        connection.execute("CREATE TABLE duplicate (id INTEGER PRIMARY KEY)")
    with sqlite3.connect(second_database) as connection:
        connection.execute("CREATE TABLE duplicate (id INTEGER PRIMARY KEY)")
    agent = await Agent.create(
        "catalog-schema-cross-source", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        first = await agent.attach(SQLiteSource(first_database, name="First"))
        second = await agent.attach(SQLiteSource(second_database, name="Second"))
        first_resource = (await agent.list_catalog_resources(source_id=first.id))[0]
        second_resource = (await agent.list_catalog_resources(source_id=second.id))[0]

        projection = await _schema(
            agent,
            resource_ids=(first_resource.id, second_resource.id),
            limit=4,
        )
        assert projection["paths"] == ()
        assert _mapping(projection["selection"])["unresolved_reasons"] == (
            "cross_source_unsupported",
        )
        assert {
            item["source_id"] for item in _mapping_sequence(projection["resources"])
        } == {first.id, second.id}
        assert all(
            item["selection_role"] == "seed"
            for item in _mapping_sequence(projection["resources"])
        )
    finally:
        await agent.close()


async def test_schema_resource_and_relationship_bounds_are_explicit(tmp_path: Path):
    database = tmp_path / "bounded.sqlite"
    wide_columns = ", ".join(f"c{index} TEXT" for index in range(257))
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"CREATE TABLE parent (id INTEGER PRIMARY KEY, {wide_columns})"
        )
        for index in range(201):
            connection.execute(
                f"CREATE TABLE child_{index:03d} ("
                "id INTEGER PRIMARY KEY, "
                "parent_id INTEGER REFERENCES parent(id))"
            )
    agent = await Agent.create(
        "catalog-schema-bounds", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        parent = next(resource for resource in resources if resource.name == "parent")
        projection = await _schema(
            agent,
            resource_ids=(parent.id,),
            limit=1,
        )
        resource = _mapping_sequence(projection["resources"])[0]
        assert len(_mapping_sequence(resource["columns"])) == 256
        bounds = projection["bounds"]
        assert isinstance(bounds, FrozenJsonObject)
        assert bounds.to_dict() == {
            "columns_per_resource": 256,
            "join_depth": 3,
            "join_graph_edges": 2_000,
            "join_graph_nodes": 1_000,
            "primary_key_fields_per_resource": 64,
            "relationships": 200,
            "resources": 1,
            "structural_facts_per_resource": 32,
            "unique_key_fields_per_resource": 64,
        }
        truncation = projection["truncation"]
        assert isinstance(truncation, Mapping)
        assert truncation["columns"] is True
        assert truncation["relationships"] is True
        assert len(_mapping_sequence(projection["relationships"])) == 200

        resource_projection = await _schema(
            agent,
            query="child",
            limit=1,
            include_relationships=False,
        )
        resource_truncation = resource_projection["truncation"]
        assert isinstance(resource_truncation, Mapping)
        assert resource_truncation["resources"] is True
        assert resource_truncation["relationships"] is False
        assert resource_projection["total_matches"] == 202
    finally:
        await agent.close()
