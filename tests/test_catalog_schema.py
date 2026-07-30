from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sqlite3

import pytest

from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import (
    CapabilityInputError,
    CapabilityRegistry,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
)
from daita.catalog import CatalogSchemaRequest, CatalogSearchRequest
from daita.catalog.capabilities import (
    CATALOG_SCHEMA_EVIDENCE_KIND,
    CATALOG_TRAVERSE_CAPABILITY_ID,
)
from daita.catalog.protocols import (
    CatalogResourceNotFoundError,
    CatalogStoreError,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)


def _fixture_database(path: Path, *, reverse: bool = False) -> None:
    statements = (
        """
        CREATE TABLE regions (
            region_code TEXT PRIMARY KEY,
            region_name TEXT NOT NULL UNIQUE,
            currency_code TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            region_code TEXT NOT NULL REFERENCES regions(region_code),
            segment TEXT NOT NULL,
            signed_up_at TEXT NOT NULL,
            is_active BOOLEAN NOT NULL
        )
        """,
        """
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            sku TEXT NOT NULL UNIQUE,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            unit_price NUMERIC NOT NULL,
            unit_cost NUMERIC NOT NULL,
            is_active BOOLEAN NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            ordered_at TEXT NOT NULL,
            status TEXT NOT NULL,
            sales_channel TEXT NOT NULL,
            subtotal NUMERIC NOT NULL DEFAULT 0,
            tax_amount NUMERIC NOT NULL DEFAULT 0,
            total_amount NUMERIC NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL REFERENCES orders(order_id),
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            quantity INTEGER NOT NULL,
            unit_price NUMERIC NOT NULL,
            discount_percent NUMERIC NOT NULL,
            line_total NUMERIC NOT NULL
        )
        """,
        """
        CREATE TABLE payments (
            payment_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL UNIQUE REFERENCES orders(order_id),
            processed_at TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            payment_status TEXT NOT NULL,
            amount NUMERIC NOT NULL
        )
        """,
        """
        CREATE TABLE shipments (
            shipment_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL UNIQUE REFERENCES orders(order_id),
            warehouse_code TEXT NOT NULL,
            shipment_status TEXT NOT NULL,
            shipped_at TEXT,
            delivered_at TEXT
        )
        """,
        """
        CREATE TABLE support_tickets (
            ticket_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            order_id INTEGER REFERENCES orders(order_id),
            opened_at TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            ticket_status TEXT NOT NULL,
            satisfaction_score INTEGER
        )
        """,
    )
    ordered = tuple(reversed(statements)) if reverse else statements
    if reverse:
        # Reverse creation cannot precede referenced tables when FK enforcement is
        # enabled, but SQLite still records the declared references deterministically.
        ordered = tuple(reversed(statements))
    with sqlite3.connect(path) as connection:
        for statement in ordered:
            connection.execute(statement)
        connection.executescript("""
            CREATE INDEX customers_region_code_idx ON customers(region_code);
            CREATE INDEX orders_customer_id_idx ON orders(customer_id);
            CREATE INDEX order_items_order_id_idx ON order_items(order_id);
            CREATE INDEX order_items_product_id_idx ON order_items(product_id);

            INSERT INTO regions VALUES ('EMEA', 'Europe', 'EUR');
            INSERT INTO customers VALUES
                (1, 'Example', 'example@test', 'EMEA', 'enterprise',
                 '2026-01-01T00:00:00Z', 1);
            INSERT INTO products VALUES
                (1, 'SKU-1', 'Example product', 'software', 100, 40, 1,
                 '2026-01-01T00:00:00Z');
            INSERT INTO orders VALUES
                (1, 1, '2026-01-02T00:00:00Z', 'paid', 'direct',
                 100, 10, 110);
            INSERT INTO order_items VALUES (1, 1, 1, 1, 100, 0, 100);
            """)


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    assert isinstance(value, tuple)
    assert all(isinstance(item, Mapping) for item in value)
    return value


def _tool_results(request: ModelRequest) -> tuple[ToolResultBlock, ...]:
    return tuple(
        block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )


class _InventoryProvider:
    provider_id = "mock:catalog-inventory"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.catalog_tool_call_count = 0
        self.schema_result_bytes = 0
        self.projected_resource_count = 0
        self.duplicated_relationship_id_count = 0
        self.truncation: Mapping[str, object] | None = None

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            guidance = repr(request.messages)
            assert "catalog_schema first for SQL and direct relationships" in guidance
            assert "never call catalog_traverse alongside it" in guidance
            assert "traverse later only for unresolved multi-hop paths" in guidance
            assert "catalog_inspect for full facets" in guidance
            self.catalog_tool_call_count += 1
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="inventory-schema",
                        name="catalog_schema",
                        arguments={
                            "query": "tables relationships",
                            "limit": 50,
                            "include_relationships": True,
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )

        results = _tool_results(request)
        schema = next(block for block in results if block.call_id == "inventory-schema")
        assert schema.is_error is False
        data = schema.output["data"]
        assert isinstance(data, Mapping)
        resources = _mapping_sequence(data["resources"])
        relationships = _mapping_sequence(data["relationships"])
        relationship_ids = tuple(item["relationship_id"] for item in relationships)
        self.schema_result_bytes = len(canonical_json(data).encode("utf-8"))
        self.projected_resource_count = len(resources)
        self.duplicated_relationship_id_count = len(relationship_ids) - len(
            set(relationship_ids)
        )
        truncation = data["truncation"]
        assert isinstance(truncation, Mapping)
        self.truncation = truncation
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text="Eight tables and their relationships are available.",
            provider_id=self.provider_id,
        )


class _RegionalMarginProvider:
    provider_id = "mock:catalog-regional-margin"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.catalog_tool_call_count = 0
        self.schema_result_bytes = 0
        self.planned_from_schema = False

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            self.catalog_tool_call_count += 1
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="regional-schema",
                        name="catalog_schema",
                        arguments={
                            "query": (
                                "customers regions orders order_items products "
                                "region_code total_amount unit_cost unit_price"
                            ),
                            "limit": 8,
                            "include_relationships": True,
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )

        results = _tool_results(request)
        if len(self.requests) == 2:
            schema = next(
                block for block in results if block.call_id == "regional-schema"
            )
            assert schema.is_error is False
            data = schema.output["data"]
            assert isinstance(data, Mapping)
            self.schema_result_bytes = len(canonical_json(data).encode("utf-8"))
            resources = _mapping_sequence(data["resources"])
            by_name = {item["name"]: item for item in resources}
            assert {
                "main.customers",
                "main.regions",
                "main.orders",
                "main.order_items",
                "main.products",
            } <= set(by_name)
            relationship_pairs = {
                (
                    item["from_resource_id"],
                    item["to_resource_id"],
                    tuple(
                        (pair["source_field"], pair["target_field"])
                        for pair in _mapping_sequence(item["field_pairs"])
                    ),
                )
                for item in _mapping_sequence(data["relationships"])
            }
            name_by_id = {item["resource_id"]: item["name"] for item in resources}
            named_pairs = {
                (name_by_id[source], name_by_id[target], fields)
                for source, target, fields in relationship_pairs
            }
            assert (
                "main.orders",
                "main.customers",
                (("customer_id", "customer_id"),),
            ) in named_pairs
            assert (
                "main.customers",
                "main.regions",
                (("region_code", "region_code"),),
            ) in named_pairs
            assert (
                "main.order_items",
                "main.products",
                (("product_id", "product_id"),),
            ) in named_pairs
            source = _mapping_sequence(data["sources"])[0]
            self.planned_from_schema = True
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="regional-query",
                        name="data_query_sqlite",
                        arguments={
                            "source_id": source["source_id"],
                            "sql": (
                                "SELECT c.region_code, "
                                "SUM(o.total_amount) AS paid_revenue, "
                                "SUM(oi.line_total - "
                                "(oi.quantity * p.unit_cost)) AS gross_margin "
                                "FROM customers AS c "
                                "JOIN orders AS o "
                                "ON o.customer_id = c.customer_id "
                                "JOIN order_items AS oi "
                                "ON oi.order_id = o.order_id "
                                "JOIN products AS p "
                                "ON p.product_id = oi.product_id "
                                "WHERE o.status = 'paid' "
                                "GROUP BY c.region_code"
                            ),
                            "parameters": (),
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )

        query = next(block for block in results if block.call_id == "regional-query")
        assert query.is_error is False
        data = query.output["data"]
        assert isinstance(data, Mapping)
        rows = _mapping_sequence(data["rows"])
        assert len(rows) == 1
        assert rows[0]["region_code"] == "EMEA"
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text="EMEA paid revenue is 110 with gross margin 60.",
            provider_id=self.provider_id,
        )


class _RevisionReuseProvider:
    provider_id = "mock:catalog-schema-revision-reuse"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.schema_calls = 0

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        schema_results = tuple(
            block
            for block in _tool_results(request)
            if block.output.get("kind") == CATALOG_SCHEMA_EVIDENCE_KIND
        )
        if not schema_results:
            self.schema_calls += 1
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id=f"schema-revision-{self.schema_calls}",
                        name="catalog_schema",
                        arguments={
                            "query": "tables relationships",
                            "limit": 50,
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text="Current schema is available.",
            provider_id=self.provider_id,
        )


async def _schema(
    agent: Agent,
    *,
    query: str | None = None,
    resource_ids: tuple[str, ...] = (),
    source_id: str | None = None,
    limit: int = 50,
    include_relationships: bool = True,
) -> Mapping[str, object]:
    projection = await agent._embedded._catalog_service.schema_slice(
        CatalogSchemaRequest(
            agent_id=agent.id,
            query=query,
            resource_ids=resource_ids,
            source_id=source_id,
            limit=limit,
            include_relationships=include_relationships,
        )
    )
    return projection


async def test_schema_slice_spans_eight_tables_with_exact_compact_structure(
    tmp_path: Path,
):
    database = tmp_path / "fixture.sqlite"
    _fixture_database(database)
    agent = await Agent.create("catalog-schema-eight", root=tmp_path)
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
            "primary_key_fields": False,
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


async def test_schema_capability_validates_output_and_is_smaller_than_inspections(
    tmp_path: Path,
):
    database = tmp_path / "capability.sqlite"
    _fixture_database(database)
    agent = await Agent.create("catalog-schema-capability", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        resources = await agent.list_catalog_resources(source_id=source.id)
        registry: CapabilityRegistry = agent._embedded._capabilities
        view, capability = registry.resolve_tool("catalog_schema")
        assert view.capability_id == capability.id
        _, executor = registry.resolve_execution(capability.id)
        output = await executor.execute(
            ToolExecution(
                run_id="schema-capability",
                capability_id=capability.id,
                arguments={
                    "resource_ids": tuple(resource.id for resource in resources),
                    "limit": 50,
                    "include_relationships": True,
                },
            )
        )
        assert output.kind == CATALOG_SCHEMA_EVIDENCE_KIND
        assert registry.validate_output(capability.id, output) == output
        with pytest.raises(ToolOutputValidationError):
            registry.validate_output(
                capability.id,
                ToolOutput(
                    kind=CATALOG_SCHEMA_EVIDENCE_KIND,
                    data={"resources": (), "relationships": ()},
                ),
            )

        schema_bytes = len(canonical_json(output.data).encode("utf-8"))
        inspections = tuple(
            [
                await agent._embedded._catalog_service.inspect_resource(
                    agent.id,
                    resource.id,
                )
                for resource in resources
            ]
        )
        inspection_bytes = sum(
            len(canonical_json(inspection).encode("utf-8"))
            for inspection in inspections
        )
        assert schema_bytes < inspection_bytes * 0.7
        assert len(_mapping_sequence(output.data["resources"])) == 8
        output_truncation = output.data["truncation"]
        assert isinstance(output_truncation, Mapping)
        assert output_truncation["resources"] is False
        assert len(
            {
                item["relationship_id"]
                for item in _mapping_sequence(output.data["relationships"])
            }
        ) == len(_mapping_sequence(output.data["relationships"]))
    finally:
        await agent.close()


async def test_catalog_tool_contract_exposes_and_enforces_progressive_bounds(
    tmp_path: Path,
):
    agent = await Agent.create("catalog-progressive-contract", root=tmp_path)
    try:
        registry: CapabilityRegistry = agent._embedded._capabilities
        schema_definition = registry.tool_definition("catalog_schema")
        traversal_definition = registry.tool_definition("catalog_traverse")

        assert "Use first for SQL planning" in schema_definition.description
        assert "Do not call catalog_traverse in the same response" in (
            schema_definition.description
        )
        assert "later step" in traversal_definition.description
        assert "Do not call alongside catalog_schema" in (
            traversal_definition.description
        )

        traversal_properties = traversal_definition.input_schema["properties"]
        assert isinstance(traversal_properties, Mapping)
        endpoint_rule = {
            "items": {
                "maxLength": 256,
                "minLength": 1,
                "type": "string",
            },
            "maxItems": 16,
            "minItems": 1,
            "type": "array",
            "uniqueItems": True,
        }
        assert canonical_json(
            traversal_properties["from_resource_ids"]
        ) == canonical_json(endpoint_rule)
        assert canonical_json(
            traversal_properties["to_resource_ids"]
        ) == canonical_json(endpoint_rule)
        assert canonical_json(
            traversal_properties["relationship_kinds"]
        ) == canonical_json(
            {
                "default": [],
                "items": {
                    "enum": [
                        "contains",
                        "references",
                        "derived_from",
                        "produces",
                        "writes_to",
                        "reads_from",
                        "observes",
                    ],
                    "type": "string",
                },
                "maxItems": 7,
                "type": "array",
                "uniqueItems": True,
            }
        )
        for name, default, maximum in (
            ("max_depth", 4, 6),
            ("max_paths", 5, 8),
            ("max_nodes", 100, 1_000),
            ("max_edges", 200, 2_000),
        ):
            assert canonical_json(traversal_properties[name]) == canonical_json(
                {
                    "default": default,
                    "maximum": maximum,
                    "minimum": 1,
                    "type": "integer",
                }
            )

        with pytest.raises(CapabilityInputError) as invalid_paths:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": ("resource-a",),
                    "to_resource_ids": ("resource-b",),
                    "max_paths": 20,
                },
            )
        assert invalid_paths.value.code == "invalid_argument_value"
        assert invalid_paths.value.details.to_dict() == {
            "constraint": "maximum",
            "name": "max_paths",
        }

        with pytest.raises(CapabilityInputError) as duplicate_endpoints:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": ("resource-a", "resource-a"),
                    "to_resource_ids": ("resource-b",),
                },
            )
        assert duplicate_endpoints.value.code == "invalid_argument_value"
        assert duplicate_endpoints.value.details.to_dict() == {
            "constraint": "uniqueItems",
            "name": "from_resource_ids",
        }

        with pytest.raises(CapabilityInputError) as oversized_endpoints:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": tuple(
                        f"resource-{index}" for index in range(17)
                    ),
                    "to_resource_ids": ("resource-b",),
                },
            )
        assert oversized_endpoints.value.code == "invalid_argument_value"
        assert oversized_endpoints.value.details.to_dict() == {
            "constraint": "maxItems",
            "name": "from_resource_ids",
        }

        with pytest.raises(CapabilityInputError) as oversized_endpoint:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": ("r" * 257,),
                    "to_resource_ids": ("resource-b",),
                },
            )
        assert oversized_endpoint.value.code == "invalid_argument_value"
        assert oversized_endpoint.value.details.to_dict() == {
            "constraint": "maxLength",
            "name": "from_resource_ids[0]",
        }

        with pytest.raises(CapabilityInputError) as invalid_relationship_kind:
            registry.validate_arguments(
                CATALOG_TRAVERSE_CAPABILITY_ID,
                {
                    "from_resource_ids": ("resource-a",),
                    "to_resource_ids": ("resource-b",),
                    "relationship_kinds": ("invented",),
                },
            )
        assert invalid_relationship_kind.value.code == "invalid_argument_value"
        assert invalid_relationship_kind.value.details.to_dict() == {
            "constraint": "enum",
            "name": "relationship_kinds[0]",
        }

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
    agent = await Agent.create("catalog-structural-search", root=tmp_path)
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
        assert no_synonym.hits == ()
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
    agent = await Agent.create("catalog-schema-scope", root=tmp_path)
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


async def test_schema_refresh_filters_old_resources_and_carries_new_revisions(
    tmp_path: Path,
):
    database = tmp_path / "refresh.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE current_table (id INTEGER PRIMARY KEY, value TEXT)"
        )
    agent = await Agent.create("catalog-schema-refresh", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-bounds", root=tmp_path)
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


async def test_schema_and_structural_search_order_ignore_catalog_insertion_order(
    tmp_path: Path,
):
    first_database = tmp_path / "ordered.sqlite"
    second_database = tmp_path / "reverse.sqlite"
    _fixture_database(first_database)
    _fixture_database(second_database, reverse=True)
    first_agent = await Agent.create("catalog-order-first", root=tmp_path)
    second_agent = await Agent.create("catalog-order-second", root=tmp_path)
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
            "data_query_sqlite",
        )
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
