from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import daita.catalog.service as catalog_service
import daita.storage.sqlite as sqlite_store
from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import (
    CapabilityInputError,
    CapabilityRegistry,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
)
from daita.capability_runtime import CapabilityRuntime
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
    TabularIndex,
    catalog_resource_id,
)
from daita.catalog.capabilities import (
    CATALOG_SCHEMA_CAPABILITY_ID,
    CATALOG_SCHEMA_EVIDENCE_KIND,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_TRAVERSE_CAPABILITY_ID,
)
from daita.catalog.models import (
    CATALOG_MAX_LIMIT,
    CATALOG_RESOURCE_ID_MAX_CHARACTERS,
    CATALOG_SCHEMA_MAX_RESOURCE_IDS,
    CATALOG_SEARCH_REQUEST_DEFAULT_LIMIT,
    CATALOG_SEARCH_REQUEST_MAX_QUERY_CHARACTERS,
    CATALOG_SOURCE_ID_MAX_CHARACTERS,
    CATALOG_TOOL_DEFAULT_LIMIT,
    CATALOG_TOOL_QUERY_MAX_CHARACTERS,
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
from daita.loop.models import RunInput
from tests.support.capability_runtime import execute_projected
from tests.support.workspace import workspace_for

_OBSERVED_AT = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


def _tabular_decoder_fixture() -> TabularFacet:
    return TabularFacet(
        columns=(
            TabularColumn(
                name="tenant_id",
                native_type="bigint",
                ordinal=1,
                nullable=False,
                primary_key_ordinal=2,
                native_type_namespace="pg_catalog",
                native_type_name="int8",
                updatable=True,
            ),
            TabularColumn(
                name="account_id",
                native_type="bigint",
                ordinal=0,
                nullable=False,
                primary_key_ordinal=1,
                native_type_namespace="pg_catalog",
                native_type_name="int8",
                identity=True,
            ),
        ),
        indexes=(
            TabularIndex(
                name="accounts_tenant_unique",
                kind="btree",
                columns=("tenant_id",),
                unique=True,
            ),
        ),
        row_count_estimate=12,
    )


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
            assert "reported unresolved paths after schema inspection" in guidance
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
                        name="data_query",
                        arguments={
                            "source_id": source["source_id"],
                            "resource_ids": tuple(
                                by_name[name]["resource_id"]
                                for name in (
                                    "main.customers",
                                    "main.orders",
                                    "main.order_items",
                                    "main.products",
                                )
                            ),
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
                        name="data_query",
                        arguments={
                            "source_id": source["source_id"],
                            "resource_ids": tuple(
                                resource["resource_id"] for resource in resources
                            ),
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
