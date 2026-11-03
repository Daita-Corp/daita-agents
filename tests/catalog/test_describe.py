"""Exercise the catalog schema projections and the catalog tools' input contracts."""

from __future__ import annotations

from collections.abc import Sequence

from daita.capabilities import CallTargetLane
from tests.catalog._schema_support import (
    CATALOG_DESCRIBE_CAPABILITY_ID,
    CATALOG_MAX_LIMIT,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_SEARCH_REQUEST_DEFAULT_LIMIT,
    CATALOG_SEARCH_REQUEST_MAX_QUERY_CHARACTERS,
    CATALOG_TOOL_DEFAULT_LIMIT,
    CATALOG_TOOL_QUERY_MAX_CHARACTERS,
    Agent,
    CapabilityInputError,
    CapabilityRegistry,
    CatalogResourceNotFoundError,
    CatalogSchemaRequest,
    CatalogSearchRequest,
    CatalogStoreError,
    FrozenJsonObject,
    Mapping,
    Path,
    SQLiteSource,
    _fixture_database,
    _mapping,
    _mapping_sequence,
    _schema,
    cast,
    pytest,
    sqlite3,
    workspace_for,
)
from tests.support.capability_runtime import argument_issues


async def test_schema_slice_spans_eight_tables_with_exact_compact_structure(
    tmp_path: Path,
):
    database = tmp_path / "fixture.sqlite"
    _fixture_database(database)
    agent = await Agent.create(
        "catalog-schema-eight", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        projection = await _schema(
            agent,
            resource_ids=tuple(resource.id for resource in resources),
        )

        projected_resources = _mapping_sequence(projection["resources"])
        assert tuple(item["name"] for item in projected_resources) == (
            "main.customers",
            "main.order_items",
            "main.orders",
            "main.payments",
            "main.products",
            "main.regions",
            "main.shipments",
            "main.support_tickets",
        )
        assert projection["total_matches"] == 8
        assert projection["trust_classification"] == "untrusted_external_data"
        truncation = projection["truncation"]
        assert isinstance(truncation, FrozenJsonObject)
        assert truncation.to_dict() == {
            "columns": False,
            "paths": False,
            "primary_key_fields": False,
            "reason": None,
            "relationships": False,
            "resources": False,
            "structural_facts": False,
            "unique_key_fields": False,
        }

        by_name = {str(item["name"]): item for item in projected_resources}
        orders = by_name["main.orders"]
        assert tuple(
            dict(column) for column in _mapping_sequence(orders["columns"])
        ) == (
            {"name": "order_id", "nullable": False, "type": "INTEGER"},
            {"name": "customer_id", "nullable": False, "type": "INTEGER"},
            {"name": "ordered_at", "nullable": False, "type": "TEXT"},
            {"name": "status", "nullable": False, "type": "TEXT"},
            {"name": "sales_channel", "nullable": False, "type": "TEXT"},
            {"name": "subtotal", "nullable": False, "type": "NUMERIC"},
            {"name": "tax_amount", "nullable": False, "type": "NUMERIC"},
            {"name": "total_amount", "nullable": False, "type": "NUMERIC"},
        )
        assert orders["primary_key_fields"] == ("order_id",)
        assert by_name["main.regions"]["unique_key_fields"] == (("region_name",),)
        assert by_name["main.customers"]["unique_key_fields"] == (("email",),)
        assert by_name["main.products"]["unique_key_fields"] == (("sku",),)
        assert by_name["main.payments"]["unique_key_fields"] == (("order_id",),)
        assert by_name["main.shipments"]["unique_key_fields"] == (("order_id",),)

        relationships = _mapping_sequence(projection["relationships"])
        relationship_ids = tuple(item["relationship_id"] for item in relationships)
        assert len(relationship_ids) == len(set(relationship_ids)) == 8
        pairs = {
            (
                by_id[item["from_resource_id"]]["name"],
                by_id[item["to_resource_id"]]["name"],
                tuple(
                    (pair["source_field"], pair["target_field"])
                    for pair in _mapping_sequence(item["field_pairs"])
                ),
            )
            for item in relationships
            for by_id in (
                {resource["resource_id"]: resource for resource in projected_resources},
            )
        }
        assert pairs == {
            ("main.customers", "main.regions", (("region_code", "region_code"),)),
            ("main.order_items", "main.orders", (("order_id", "order_id"),)),
            ("main.order_items", "main.products", (("product_id", "product_id"),)),
            ("main.orders", "main.customers", (("customer_id", "customer_id"),)),
            ("main.payments", "main.orders", (("order_id", "order_id"),)),
            ("main.shipments", "main.orders", (("order_id", "order_id"),)),
            (
                "main.support_tickets",
                "main.customers",
                (("customer_id", "customer_id"),),
            ),
            (
                "main.support_tickets",
                "main.orders",
                (("order_id", "order_id"),),
            ),
        }
        assert all(item["provenance"] == "connector" for item in relationships)
        assert all(
            item["from_resource_revision"] and item["to_resource_revision"]
            for item in relationships
        )
        sources = _mapping_sequence(projection["sources"])
        assert len(sources) == 1
        assert sources[0]["source_id"] == source.id
        assert isinstance(sources[0]["source_revision"], str)
        assert isinstance(sources[0]["sync_id"], str)
    finally:
        await agent.close()


async def test_catalog_tools_take_names_within_explicit_bounds(tmp_path: Path):
    agent = await Agent.create(
        "catalog-input-contracts", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        registry: CapabilityRegistry = agent._embedded._capabilities
        search = registry.tool_definition(
            "catalog_search", lane=CallTargetLane.FOREGROUND
        ).input_schema
        describe = registry.tool_definition(
            "catalog_describe", lane=CallTargetLane.FOREGROUND
        ).input_schema
        assert set(_mapping(search["properties"])) == {
            "kinds",
            "limit",
            "query",
            "source",
        }
        assert set(_mapping(describe["properties"])) == {"name", "source"}
        assert set(cast(Sequence[str], describe["required"])) == {"name", "source"}
        assert _mapping(_mapping(search["properties"])["limit"])["maximum"] == (
            CATALOG_MAX_LIMIT
        )

        with pytest.raises(CapabilityInputError) as invalid_limit:
            registry.validate_arguments(
                CATALOG_SEARCH_CAPABILITY_ID,
                {"query": "orders", "limit": CATALOG_MAX_LIMIT + 1},
            )
        assert _validation_details(invalid_limit.value) == {
            "constraint": "maximum",
            "name": "limit",
        }
        with pytest.raises(CapabilityInputError) as invalid_kind:
            registry.validate_arguments(
                CATALOG_SEARCH_CAPABILITY_ID,
                {"query": "orders", "kinds": ("spreadsheet",)},
            )
        assert _validation_details(invalid_kind.value) == {
            "constraint": "enum",
            "name": "kinds[0]",
        }
        # The executed contract admits only a full source ID, never a guess.
        with pytest.raises(CapabilityInputError) as garbled:
            registry.validate_arguments(
                CATALOG_DESCRIBE_CAPABILITY_ID,
                {"source_id": "source:sha256:" + "a" * 63, "name": "orders"},
            )
        assert _validation_details(garbled.value) == {
            "constraint": "pattern",
            "name": "source",
        }

        assert CatalogSearchRequest(agent_id="agent", query="orders").limit == (
            CATALOG_SEARCH_REQUEST_DEFAULT_LIMIT
        )
        assert CatalogSchemaRequest(agent_id="agent", query="orders").limit == (
            CATALOG_TOOL_DEFAULT_LIMIT
        )
        oversized_search_query = "q" * (CATALOG_SEARCH_REQUEST_MAX_QUERY_CHARACTERS + 1)
        with pytest.raises(ValueError, match="catalog search query exceeds"):
            CatalogSearchRequest(agent_id="agent", query=oversized_search_query)
        oversized_tool_query = "q" * (CATALOG_TOOL_QUERY_MAX_CHARACTERS + 1)
        with pytest.raises(ValueError, match="catalog schema query exceeds"):
            CatalogSchemaRequest(agent_id="agent", query=oversized_tool_query)
    finally:
        await agent.close()


async def test_structural_search_ranks_direct_matches_before_one_hop_neighbors(
    tmp_path: Path,
):
    database = tmp_path / "structural.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE needle (id INTEGER PRIMARY KEY);
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                needle_code TEXT NOT NULL UNIQUE
            );
            CREATE TABLE neighbor (
                id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(id)
            );
            """)
    agent = await Agent.create(
        "catalog-structural-search", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        await agent.attach(SQLiteSource(database))
        result = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="needle",
                limit=3,
            )
        )
        assert tuple(hit.name for hit in result.hits) == (
            "needle",
            "orders",
            "neighbor",
        )
        assert result.hits[0].match_reasons == ("resource_name_exact",)
        assert result.hits[1].match_reasons == ("structural_field_contains",)
        assert "column:needle_code" in result.hits[1].matched_fields
        assert result.hits[2].match_reasons == ("relationship_neighbor",)
        no_synonym = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="margin",
                limit=3,
            )
        )
        assert no_synonym.total_matches == 0
        assert no_synonym.total_candidates == 3
        assert len(no_synonym.hits) == 3
        assert all(
            hit.match_reasons == ("unmatched_fallback",) for hit in no_synonym.hits
        )
    finally:
        await agent.close()


async def test_schema_scope_is_strict_current_and_source_file_is_not_read(
    tmp_path: Path,
):
    first_database = tmp_path / "first.sqlite"
    second_database = tmp_path / "second.sqlite"
    with sqlite3.connect(first_database) as connection:
        connection.execute(
            "CREATE TABLE first_table (id INTEGER PRIMARY KEY, hidden_term TEXT)"
        )
    with sqlite3.connect(second_database) as connection:
        connection.execute("CREATE TABLE second_table (id INTEGER PRIMARY KEY)")
    agent = await Agent.create(
        "catalog-schema-scope", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        first = await agent.attach(SQLiteSource(first_database))
        second = await agent.attach(SQLiteSource(second_database))
        first_resource = (await agent.list_catalog_resources(source_id=first.id))[0]
        second_resource = (await agent.list_catalog_resources(source_id=second.id))[0]

        first_database.rename(tmp_path / "first-source-is-unavailable.sqlite")
        projection = await _schema(
            agent,
            resource_ids=(first_resource.id,),
            source_id=first.id,
        )
        assert len(_mapping_sequence(projection["resources"])) == 1

        with pytest.raises(CatalogStoreError):
            await _schema(
                agent,
                resource_ids=(second_resource.id,),
                source_id=first.id,
            )
        with pytest.raises(CatalogResourceNotFoundError):
            await _schema(agent, resource_ids=("catalog-resource:sha256:" + "0" * 64,))

        await agent.detach(first.id)
        with pytest.raises(CatalogStoreError):
            await _schema(agent, query="hidden_term", source_id=first.id)
    finally:
        await agent.close()


async def test_schema_and_structural_search_order_ignore_catalog_insertion_order(
    tmp_path: Path,
):
    first_database = tmp_path / "ordered.sqlite"
    second_database = tmp_path / "reverse.sqlite"
    _fixture_database(first_database)
    _fixture_database(second_database, reverse=True)
    first_agent = await Agent.create(
        "catalog-order-first", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    second_agent = await Agent.create(
        "catalog-order-second", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        first_source = await first_agent.attach(SQLiteSource(first_database))
        second_source = await second_agent.attach(SQLiteSource(second_database))
        first_search = await first_agent.search_catalog(
            CatalogSearchRequest(
                agent_id=first_agent.id,
                query="id",
                limit=50,
            )
        )
        second_search = await second_agent.search_catalog(
            CatalogSearchRequest(
                agent_id=second_agent.id,
                query="id",
                limit=50,
            )
        )
        assert tuple(
            (
                hit.name,
                hit.match_reasons,
                (
                    ()
                    if hit.match_reasons == ("relationship_neighbor",)
                    else hit.matched_fields
                ),
            )
            for hit in first_search.hits
        ) == tuple(
            (
                hit.name,
                hit.match_reasons,
                (
                    ()
                    if hit.match_reasons == ("relationship_neighbor",)
                    else hit.matched_fields
                ),
            )
            for hit in second_search.hits
        )

        first_resources = await first_agent.list_catalog_resources(
            source_id=first_source.id
        )
        second_resources = await second_agent.list_catalog_resources(
            source_id=second_source.id
        )
        first_projection = await _schema(
            first_agent,
            resource_ids=tuple(resource.id for resource in first_resources),
        )
        second_projection = await _schema(
            second_agent,
            resource_ids=tuple(resource.id for resource in second_resources),
        )
        assert tuple(
            item["name"] for item in _mapping_sequence(first_projection["resources"])
        ) == tuple(
            item["name"] for item in _mapping_sequence(second_projection["resources"])
        )

        def signatures(projection: Mapping[str, object]) -> tuple[object, ...]:
            resources = _mapping_sequence(projection["resources"])
            names = {item["resource_id"]: item["name"] for item in resources}
            return tuple(
                sorted(
                    (
                        names[item["from_resource_id"]],
                        names[item["to_resource_id"]],
                        item["kind"],
                        tuple(
                            (pair["source_field"], pair["target_field"])
                            for pair in _mapping_sequence(item["field_pairs"])
                        ),
                    )
                    for item in _mapping_sequence(projection["relationships"])
                )
            )

        assert signatures(first_projection) == signatures(second_projection)
    finally:
        await first_agent.close()
        await second_agent.close()


def _validation_details(error: CapabilityInputError) -> dict[str, object]:
    """The one structural issue of an ``invalid_arguments`` error."""

    details = dict(error.details.to_dict())
    recovery = details.pop("recovery")
    assert isinstance(recovery, dict)
    assert recovery["kind"] == "correct_arguments"
    assert error.code == "invalid_arguments"
    (issue,) = argument_issues(details)
    assert issue["code"] == "invalid_argument_value"
    return {"constraint": issue["constraint"], "name": issue["name"]}


def _field_names(value: object) -> set[str]:
    if isinstance(value, Mapping):
        names = {str(key) for key in value}
        for item in value.values():
            names |= _field_names(item)
        return names
    if isinstance(value, (tuple, list)):
        names = set()
        for item in value:
            names |= _field_names(item)
        return names
    return set()
