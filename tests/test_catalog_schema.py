from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
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
from daita.catalog import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSchemaRequest,
    CatalogSearchRequest,
    CatalogSync,
    CatalogSyncStatus,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from daita.catalog.capabilities import (
    CATALOG_SCHEMA_CAPABILITY_ID,
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
import daita.storage.sqlite as sqlite_store
import daita.catalog.service as catalog_service

_OBSERVED_AT = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class _SchemaEdge:
    source: str
    target: str
    provenance: RelationshipProvenance = RelationshipProvenance.CONNECTOR
    field_pairs: tuple[tuple[str, str], ...] = (("id", "id"),)


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


async def _commit_schema_graph(
    agent: Agent,
    source_id: str,
    *,
    nodes: Mapping[str, tuple[str, ResourceKind, bool]],
    edges: tuple[_SchemaEdge, ...],
    sync_id: str,
) -> tuple[dict[str, str], tuple[CatalogRelationship, ...]]:
    resource_ids = {
        key: catalog_resource_id(source_id, kind, native_identity)
        for key, (native_identity, kind, _) in nodes.items()
    }
    field_names_by_node: dict[str, set[str]] = {key: {"id"} for key in nodes}
    for edge in edges:
        field_names_by_node[edge.source].update(pair[0] for pair in edge.field_pairs)
        field_names_by_node[edge.target].update(pair[1] for pair in edge.field_pairs)
    facets_by_node = {
        key: CatalogFacet.from_tabular(
            resource_id=resource_ids[key],
            sync_id=sync_id,
            observed_at=_OBSERVED_AT,
            facet=TabularFacet(
                columns=tuple(
                    TabularColumn(
                        name=field_name,
                        native_type="INTEGER",
                        ordinal=ordinal,
                        nullable=False,
                        primary_key_ordinal=(1 if field_name == "id" else None),
                    )
                    for ordinal, field_name in enumerate(
                        sorted(
                            field_names_by_node[key],
                            key=lambda value: (value != "id", value),
                        )
                    )
                )
            ),
        )
        for key, (_, _, tabular) in nodes.items()
        if tabular
    }
    relationships = tuple(
        CatalogRelationship.build(
            source_id=source_id,
            from_resource_id=resource_ids[edge.source],
            to_resource_id=resource_ids[edge.target],
            kind=RelationshipKind.REFERENCES,
            provenance=edge.provenance,
            confidence=(
                1.0 if edge.provenance is RelationshipProvenance.CONNECTOR else 0.9
            ),
            sync_id=sync_id,
            observed_at=_OBSERVED_AT,
            field_pairs=tuple(
                RelationshipFieldPair(
                    source_field=source_field,
                    target_field=target_field,
                    ordinal=ordinal,
                )
                for ordinal, (source_field, target_field) in enumerate(edge.field_pairs)
            ),
        )
        for edge in edges
    )
    relationship_revisions_by_node: dict[str, list[str]] = {key: [] for key in nodes}
    for edge, relationship in zip(edges, relationships, strict=True):
        relationship_revisions_by_node[edge.source].append(relationship.revision)
        relationship_revisions_by_node[edge.target].append(relationship.revision)
    revisions = tuple(
        CatalogResourceRevision.build(
            resource_id=resource_ids[key],
            sync_id=sync_id,
            observed_at=_OBSERVED_AT,
            facet_revisions=(
                () if key not in facets_by_node else (facets_by_node[key].revision,)
            ),
            relationship_revisions=relationship_revisions_by_node[key],
            source_revision=f"test:{sync_id}",
        )
        for key in nodes
    )
    revision_by_resource_id = {revision.resource_id: revision for revision in revisions}
    resources = tuple(
        CatalogResource.build(
            agent_id=agent.id,
            source_id=source_id,
            native_identity=native_identity,
            external_uri=f"test://{source_id}/{native_identity}",
            kind=kind,
            name=native_identity.rsplit(".", 1)[-1],
            sensitivity=Sensitivity.INTERNAL,
            revision=revision_by_resource_id[resource_ids[key]],
            first_observed_at=_OBSERVED_AT,
            last_observed_at=_OBSERVED_AT,
        )
        for key, (native_identity, kind, _) in nodes.items()
    )
    await agent._embedded._store.commit_snapshot(
        SourceCatalogSnapshot(
            sync=CatalogSync(
                id=sync_id,
                agent_id=agent.id,
                source_id=source_id,
                adapter_id="test-schema-graph",
                status=CatalogSyncStatus.SUCCEEDED,
                started_at=_OBSERVED_AT,
                completed_at=_OBSERVED_AT,
                source_revision=f"test:{sync_id}",
                resource_count=len(resources),
                relationship_count=len(relationships),
            ),
            resources=resources,
            revisions=revisions,
            facets=tuple(facets_by_node.values()),
            relationships=relationships,
        )
    )
    return resource_ids, relationships


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    assert isinstance(value, tuple)
    assert all(isinstance(item, Mapping) for item in value)
    return value


def _object_sequence(value: object) -> tuple[object, ...]:
    assert isinstance(value, tuple)
    return value


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
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
            assert (
                "catalog_schema first for SQL (bounded bridges and paths)" in guidance
            )
            assert "Only then use catalog_traverse" in guidance
            assert "never call both together" in guidance
            assert "catalog_inspect gives full facets" in guidance
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


class _BridgePlanningProvider:
    provider_id = "mock:catalog-bridge-planning"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.saw_complete_bridge_evidence = False

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bridge-schema",
                        name="catalog_schema",
                        arguments={
                            "query": "customers products",
                            "limit": 5,
                            "max_join_depth": 3,
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )

        results = _tool_results(request)
        if len(self.requests) == 2:
            schema = next(
                block for block in results if block.call_id == "bridge-schema"
            )
            assert schema.is_error is False
            data = schema.output["data"]
            assert isinstance(data, Mapping)
            resources = _mapping_sequence(data["resources"])
            names_by_id = {
                resource["resource_id"]: resource["name"] for resource in resources
            }
            assert {
                resource["name"]: resource["selection_role"] for resource in resources
            } == {
                "main.customers": "seed",
                "main.order_items": "bridge",
                "main.orders": "bridge",
                "main.products": "seed",
            }
            path = _mapping_sequence(data["paths"])[0]
            assert tuple(
                names_by_id[resource_id]
                for resource_id in _object_sequence(path["resource_ids"])
            ) == (
                "main.customers",
                "main.orders",
                "main.order_items",
                "main.products",
            )
            relationships = {
                relationship["relationship_id"]: relationship
                for relationship in _mapping_sequence(data["relationships"])
            }
            assert set(_object_sequence(path["relationship_ids"])) <= set(relationships)
            assert all(
                relationship["field_pairs"] for relationship in relationships.values()
            )
            self.saw_complete_bridge_evidence = True
            source = _mapping_sequence(data["sources"])[0]
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bridge-query",
                        name="data_query_sqlite",
                        arguments={
                            "source_id": source["source_id"],
                            "sql": (
                                "SELECT c.segment, "
                                "SUM(oi.line_total - (oi.quantity * p.unit_cost)) "
                                "AS gross_margin "
                                "FROM customers AS c "
                                "JOIN orders AS o ON o.customer_id = c.customer_id "
                                "JOIN order_items AS oi ON oi.order_id = o.order_id "
                                "JOIN products AS p ON p.product_id = oi.product_id "
                                "GROUP BY c.segment"
                            ),
                            "parameters": (),
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )

        query = next(block for block in results if block.call_id == "bridge-query")
        assert query.is_error is False
        query_data = _mapping(query.output["data"])
        rows = _mapping_sequence(query_data["rows"])
        assert rows[0]["segment"] == "enterprise"
        assert rows[0]["gross_margin"] == 60
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text="Enterprise gross margin is 60.",
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
    max_join_depth: int = 3,
) -> Mapping[str, object]:
    projection = await agent._embedded._catalog_service.schema_slice(
        CatalogSchemaRequest(
            agent_id=agent.id,
            query=query,
            resource_ids=resource_ids,
            source_id=source_id,
            limit=limit,
            include_relationships=include_relationships,
            max_join_depth=max_join_depth,
        )
    )
    return projection


def _resource_names_by_id(projection: Mapping[str, object]) -> dict[str, str]:
    return {
        str(resource["resource_id"]): str(resource["name"])
        for resource in _mapping_sequence(projection["resources"])
    }


def _path_name_signatures(
    projection: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    names = _resource_names_by_id(projection)
    return tuple(
        tuple(
            names[str(resource_id)]
            for resource_id in _object_sequence(path["resource_ids"])
        )
        for path in _mapping_sequence(projection["paths"])
    )


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


async def test_connected_schema_selects_direct_and_required_bridge_paths(
    tmp_path: Path,
):
    database = tmp_path / "connected.sqlite"
    _fixture_database(database)
    agent = await Agent.create("catalog-schema-connected", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-join-tree", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-shared-bridge", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-neighbors", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-query-seeds", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-composite", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-connector-only", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-path-order", root=tmp_path)
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
    first_agent = await Agent.create("catalog-schema-agent-first", root=tmp_path)
    second_agent = await Agent.create("catalog-schema-agent-second", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-non-tabular", root=tmp_path)
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

        assert "SQL schema, bridges, paths" in schema_definition.description
        assert "Do not use with catalog_traverse" in schema_definition.description
        assert "later step" in traversal_definition.description
        assert "Do not call alongside catalog_schema" in (
            traversal_definition.description
        )

        schema_properties = schema_definition.input_schema["properties"]
        assert isinstance(schema_properties, Mapping)
        assert canonical_json(schema_properties["max_join_depth"]) == canonical_json(
            {
                "default": 3,
                "maximum": 6,
                "minimum": 1,
                "type": "integer",
            }
        )
        with pytest.raises(CapabilityInputError):
            registry.validate_arguments(
                CATALOG_SCHEMA_CAPABILITY_ID,
                {"query": "orders", "max_join_depth": True},
            )
        with pytest.raises(CapabilityInputError):
            registry.validate_arguments(
                CATALOG_SCHEMA_CAPABILITY_ID,
                {"query": "orders", "max_join_depth": 7},
            )
        with pytest.raises(ValueError, match="max_join_depth"):
            CatalogSchemaRequest(
                agent_id="agent",
                query="orders",
                max_join_depth=0,
            )
        with pytest.raises(ValueError, match="max_join_depth"):
            CatalogSchemaRequest(
                agent_id="agent",
                query="orders",
                max_join_depth=True,
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
    agent = await Agent.create("catalog-schema-stale-path", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-compilation", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-coherent-reopen", root=tmp_path)
    source = await agent.attach(SQLiteSource(database))
    resources = await agent.list_catalog_resources(source_id=source.id)
    resource_ids = tuple(resource.id for resource in resources)
    expected = canonical_json(await _schema(agent, resource_ids=resource_ids))
    await agent.close()

    original_loads = sqlite_store._loads
    decode_count = 0

    def counting_loads(value: str) -> object:
        nonlocal decode_count
        if '"__record__":"SourceCatalogSnapshot"' in value:
            decode_count += 1
        return original_loads(value)

    monkeypatch.setattr(sqlite_store, "_loads", counting_loads)
    reopened = await Agent.open("catalog-schema-coherent-reopen", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-generation-conflict", root=tmp_path)
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


async def test_connected_schema_reports_no_path_depth_and_resource_bounds(
    tmp_path: Path,
):
    database = tmp_path / "unresolved.sqlite"
    _fixture_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE isolated (id INTEGER PRIMARY KEY)")
    agent = await Agent.create("catalog-schema-unresolved", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-work-bounds", root=tmp_path)
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
    agent = await Agent.create("catalog-schema-cross-source", root=tmp_path)
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
        assert tuple(calls.values()) == ("catalog_schema", "data_query_sqlite")
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
