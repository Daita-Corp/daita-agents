"""Declared catalog links join resources across sources in one edge set."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import daita.storage.sql as sql_storage
from daita import Agent, FolderSource, SQLiteSource
from daita.adapters.models import SourceRegistration
from daita.catalog import (
    CatalogDescribeDetail,
    CatalogDescribeRequest,
    CatalogLinkStaleReason,
    CatalogResourceNotFoundError,
    CatalogSchemaRequest,
    CatalogSearchRequest,
    CatalogStoreError,
    CatalogTraversalRequest,
)
from daita.catalog.models import ResourceKind
from daita.llm.models import ModelProfile, ModelSensitivity, ToolCall, ToolResultBlock
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from tests.catalog._schema_support import (
    _mapping,
    _mapping_sequence,
    _object_sequence,
    _schema,
)
from tests.support.capability_runtime import execute_projected
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class _TwoSources:
    agent: Agent
    orders_source: SourceRegistration
    customers_source: SourceRegistration
    customers_database: Path
    orders: str
    customers: str


async def _two_sources(tmp_path: Path) -> _TwoSources:
    orders_database = tmp_path / "orders.sqlite"
    customers_database = tmp_path / "customers.sqlite"
    with sqlite3.connect(orders_database) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, "
            "total REAL)"
        )
    with sqlite3.connect(customers_database) as connection:
        connection.execute(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, region TEXT)"
        )
    agent = await Agent.create(
        "declared-links", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    orders_source = await agent.attach(SQLiteSource(orders_database, name="Orders"))
    customers_source = await agent.attach(
        SQLiteSource(customers_database, name="Customers")
    )
    (orders,) = await agent.list_catalog_resources(source_id=orders_source.id)
    (customers,) = await agent.list_catalog_resources(source_id=customers_source.id)
    return _TwoSources(
        agent=agent,
        orders_source=orders_source,
        customers_source=customers_source,
        customers_database=customers_database,
        orders=orders.id,
        customers=customers.id,
    )


async def _traverse(
    agent: Agent,
    from_id: str,
    to_id: str,
    *,
    readable_resource_ids: frozenset[str] | None = None,
) -> Mapping[str, object]:
    return await agent._embedded._data_view.traverse(
        CatalogTraversalRequest(
            agent_id=agent.id,
            from_resource_ids=(from_id,),
            to_resource_ids=(to_id,),
        ),
        readable_resource_ids=readable_resource_ids,
    )


def _unresolved(projection: Mapping[str, object]) -> tuple[object, ...]:
    return _object_sequence(_mapping(projection["selection"])["unresolved_reasons"])


async def test_declared_link_joins_tables_across_two_sqlite_sources(tmp_path: Path):
    sources = await _two_sources(tmp_path)
    agent = sources.agent
    try:
        unlinked = await _schema(
            agent, resource_ids=(sources.orders, sources.customers), limit=4
        )
        assert unlinked["paths"] == ()
        assert "cross_source_unsupported" in _unresolved(unlinked)

        link = await agent.declare_catalog_link(
            sources.orders,
            sources.customers,
            (("customer_id", "id"),),
            note="Orders belong to customers.",
        )

        traversal = await _traverse(agent, sources.orders, sources.customers)
        assert traversal["reachable"] is True
        (path,) = _mapping_sequence(traversal["paths"])
        (step,) = _mapping_sequence(path["steps"])
        assert step["relationship_id"] == link.id
        assert step["provenance"] == "declared"
        assert step["cross_source"] is True
        assert step["direction"] == "forward"

        linked = await _schema(
            agent, resource_ids=(sources.orders, sources.customers), limit=4
        )
        (join_path,) = _mapping_sequence(linked["paths"])
        assert join_path["relationship_ids"] == (link.id,)
        (relationship,) = _mapping_sequence(linked["relationships"])
        assert relationship["relationship_id"] == link.id
        assert relationship["provenance"] == "declared"
        assert relationship["cross_source"] is True
        assert "cross_source_unsupported" not in _unresolved(linked)

        search = await agent._embedded._data_view.search(
            CatalogSearchRequest(agent_id=agent.id, query="orders")
        )
        neighbor = next(
            hit for hit in search.hits if hit.resource_id == sources.customers
        )
        assert neighbor.match_reasons == (
            "relationship_neighbor",
            "declared_link",
            "cross_source",
        )

        (status,) = await agent.list_catalog_links()
        assert status.link == link
        assert status.current
    finally:
        await agent.close()


async def test_link_staleness_follows_paired_fields_not_resource_revisions(
    tmp_path: Path,
):
    sources = await _two_sources(tmp_path)
    agent = sources.agent
    try:
        link = await agent.declare_catalog_link(
            sources.orders, sources.customers, (("customer_id", "id"),)
        )
        with sqlite3.connect(sources.customers_database) as connection:
            connection.execute("ALTER TABLE customers ADD COLUMN segment TEXT")
        await agent.refresh_source(sources.customers_source.id)
        (customers,) = await agent.list_catalog_resources(
            source_id=sources.customers_source.id
        )
        assert customers.current_revision != link.declared_revisions[1]
        (status,) = await agent.list_catalog_links()
        assert status.current
        assert (await _traverse(agent, sources.orders, sources.customers))["reachable"]

        with sqlite3.connect(sources.customers_database) as connection:
            connection.execute("DROP TABLE customers")
            connection.execute("CREATE TABLE customers (key INTEGER, region TEXT)")
        await agent.refresh_source(sources.customers_source.id)

        (status,) = await agent.list_catalog_links()
        assert status.link == link
        assert status.stale_reason is CatalogLinkStaleReason.FIELD_MISSING
        stale = await _schema(
            agent, resource_ids=(sources.orders, sources.customers), limit=4
        )
        assert stale["paths"] == ()
        assert stale["relationships"] == ()
        assert "cross_source_unsupported" in _unresolved(stale)
        assert (await _traverse(agent, sources.orders, sources.customers))[
            "reachable"
        ] is False

        # The endpoint table itself is gone while its source stays active.
        with sqlite3.connect(sources.customers_database) as connection:
            connection.execute("DROP TABLE customers")
            connection.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY)")
        await agent.refresh_source(sources.customers_source.id)
        (status,) = await agent.list_catalog_links()
        assert status.stale_reason is CatalogLinkStaleReason.RESOURCE_MISSING
    finally:
        await agent.close()


async def test_detached_endpoint_source_makes_link_stale_until_reattached(
    tmp_path: Path,
):
    sources = await _two_sources(tmp_path)
    agent = sources.agent
    try:
        link = await agent.declare_catalog_link(
            sources.orders, sources.customers, (("customer_id", "id"),)
        )
        await agent.detach(sources.customers_source.id)
        (status,) = await agent.list_catalog_links()
        assert status.stale_reason is CatalogLinkStaleReason.SOURCE_INACTIVE

        reattached = await agent.attach(
            SQLiteSource(sources.customers_database, name="Customers")
        )
        assert reattached.id == sources.customers_source.id
        (status,) = await agent.list_catalog_links()
        assert status.link == link
        assert status.current
        assert (await _traverse(agent, sources.orders, sources.customers))["reachable"]
    finally:
        await agent.close()


async def test_run_scope_excluding_a_linked_source_never_reveals_its_resources(
    tmp_path: Path,
):
    sources = await _two_sources(tmp_path)
    agent = sources.agent
    try:
        with sqlite3.connect(tmp_path / "orders.sqlite") as connection:
            connection.execute(
                "CREATE TABLE refunds (id INTEGER PRIMARY KEY, customer_id INTEGER)"
            )
        await agent.refresh_source(sources.orders_source.id)
        refunds = next(
            resource.id
            for resource in await agent.list_catalog_resources(
                source_id=sources.orders_source.id
            )
            if resource.name == "refunds"
        )
        # orders and refunds share a source but connect only through customers.
        await agent.declare_catalog_link(
            sources.orders, sources.customers, (("customer_id", "id"),)
        )
        await agent.declare_catalog_link(
            refunds, sources.customers, (("customer_id", "id"),)
        )
        assert (await _traverse(agent, sources.orders, refunds))["reachable"]

        scope = frozenset({sources.orders, refunds})
        view = agent._embedded._data_view
        scoped = await _traverse(
            agent, sources.orders, refunds, readable_resource_ids=scope
        )
        assert scoped["reachable"] is False
        assert scoped["paths"] == ()
        with pytest.raises(CatalogResourceNotFoundError):
            await _traverse(
                agent,
                sources.orders,
                sources.customers,
                readable_resource_ids=scope,
            )
        search = await view.search(
            CatalogSearchRequest(agent_id=agent.id, query="orders"),
            readable_resource_ids=scope,
        )
        assert sources.customers not in {hit.resource_id for hit in search.hits}
        projection = await view.schema_slice(
            CatalogSchemaRequest(
                agent_id=agent.id, resource_ids=(sources.orders, refunds)
            ),
            readable_resource_ids=scope,
        )
        assert {
            item["resource_id"] for item in _mapping_sequence(projection["resources"])
        } == {sources.orders, refunds}
        assert projection["paths"] == ()
        assert projection["relationships"] == ()
    finally:
        await agent.close()


async def test_source_narrowed_describe_lists_declared_links_leaving_that_source(
    tmp_path: Path,
):
    sources = await _two_sources(tmp_path)
    agent = sources.agent
    try:
        with sqlite3.connect(tmp_path / "orders.sqlite") as connection:
            connection.execute(
                "CREATE TABLE refunds (id INTEGER PRIMARY KEY, customer_id INTEGER)"
            )
        await agent.refresh_source(sources.orders_source.id)
        refunds = next(
            resource.id
            for resource in await agent.list_catalog_resources(
                source_id=sources.orders_source.id
            )
            if resource.name == "refunds"
        )
        orders_link = await agent.declare_catalog_link(
            sources.orders, sources.customers, (("customer_id", "id"),)
        )
        refunds_link = await agent.declare_catalog_link(
            refunds, sources.customers, (("customer_id", "id"),)
        )
        view = agent._embedded._data_view

        async def describe(
            *resource_ids: str, readable: frozenset[str] | None = None
        ) -> Mapping[str, object]:
            return await view.describe(
                CatalogDescribeRequest(
                    agent_id=agent.id,
                    detail=CatalogDescribeDetail.SCHEMA,
                    resource_ids=resource_ids,
                    source_id=sources.orders_source.id,
                ),
                readable_resource_ids=readable,
            )

        # One seed or several: source_id narrows the described resources, while
        # each current link leaving that source is listed with its far endpoint.
        for resource_ids, links in (
            ((sources.orders,), {orders_link.id}),
            ((sources.orders, refunds), {orders_link.id, refunds_link.id}),
        ):
            narrowed = await describe(*resource_ids)
            assert {
                item["resource_id"] for item in _mapping_sequence(narrowed["resources"])
            } == set(resource_ids)
            relationships = _mapping_sequence(narrowed["relationships"])
            assert {item["relationship_id"] for item in relationships} == links
            for relationship in relationships:
                assert relationship["provenance"] == "declared"
                assert relationship["cross_source"] is True
                assert [
                    endpoint["resource_id"]
                    for endpoint in _mapping_sequence(
                        relationship["unselected_endpoints"]
                    )
                ] == [sources.customers]

        # A scope without the far endpoint still never reveals it.
        scoped = await describe(
            sources.orders, readable=frozenset({sources.orders, refunds})
        )
        assert scoped["relationships"] == ()
    finally:
        await agent.close()


async def test_source_narrowed_describe_takes_the_sensitivity_of_listed_link_endpoints(
    tmp_path: Path,
):
    orders_database = tmp_path / "orders.sqlite"
    with sqlite3.connect(orders_database) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER)"
        )
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    (restricted / "customers.csv").write_text("id,region\n1,north\n", "utf-8")
    provider = MockModelProvider(())
    agent = await Agent.create(
        "describe-sensitivity",
        root=tmp_path / "home",
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=80_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        workspace=workspace_for(tmp_path),
    )
    try:
        orders_source = await agent.attach(SQLiteSource(orders_database, name="Orders"))
        folder = await agent.attach(
            FolderSource(
                restricted, name="Restricted", sensitivity=ModelSensitivity.RESTRICTED
            )
        )
        (orders,) = await agent.list_catalog_resources(source_id=orders_source.id)
        (customers,) = [
            item
            for item in await agent.list_catalog_resources(source_id=folder.id)
            if item.kind is ResourceKind.FILE
        ]
        await agent.declare_catalog_link(
            orders.id, customers.id, (("customer_id", "id"),)
        )
        loop = agent._embedded._loop
        assert loop is not None
        (result,) = await execute_projected(
            loop._tools,
            RunInput(
                id="describe-sensitivity-run",
                agent_id=agent.id,
                message="describe orders",
                created_at=datetime(2026, 10, 19, 12, 0, tzinfo=UTC),
                conversation_id="describe-sensitivity-conversation",
            ),
            (
                ToolCall(
                    id="describe",
                    name="catalog_describe",
                    arguments={
                        "source_id": orders_source.id,
                        "resource_ids": [orders.id],
                    },
                ),
            ),
        )
    finally:
        await agent.close()

    assert isinstance(result, ToolResultBlock)
    assert "error" not in result.output, result.output
    # The far endpoint's metadata is restricted, so the whole result is.
    assert result.sensitivity is ModelSensitivity.RESTRICTED
    assert result.sensitivity_provenance is not None
    assert folder.id in cast(
        tuple[str, ...], result.sensitivity_provenance["source_ids"]
    )


async def test_invalid_link_declarations_are_rejected_before_persistence(
    tmp_path: Path,
):
    sources = await _two_sources(tmp_path)
    agent = sources.agent
    unknown = "catalog-resource:sha256:" + "0" * 64
    try:
        with pytest.raises(CatalogResourceNotFoundError):
            await agent.declare_catalog_link(
                sources.orders, unknown, (("customer_id", "id"),)
            )
        with pytest.raises(ValueError, match="not a current column"):
            await agent.declare_catalog_link(
                sources.orders, sources.customers, (("customer_id", "missing"),)
            )
        with pytest.raises(ValueError, match="distinct"):
            await agent.declare_catalog_link(
                sources.orders, sources.orders, (("customer_id", "id"),)
            )
        assert await agent.list_catalog_links() == ()
    finally:
        await agent.close()


async def test_declared_links_persist_across_reopen_until_removed(tmp_path: Path):
    sources = await _two_sources(tmp_path)
    link = await sources.agent.declare_catalog_link(
        sources.orders, sources.customers, (("customer_id", "id"),)
    )
    await sources.agent.close()

    reopened = await Agent.open(
        "declared-links", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        (status,) = await reopened.list_catalog_links()
        assert status.link == link
        assert status.current
        assert await reopened.remove_catalog_link(link.id) is True
        assert await reopened.remove_catalog_link(link.id) is False
        assert await reopened.list_catalog_links() == ()
    finally:
        await reopened.close()


async def test_link_count_is_bounded_per_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sources = await _two_sources(tmp_path)
    agent = sources.agent
    monkeypatch.setattr(sql_storage, "CATALOG_MAX_LINKS", 1)
    try:
        first = await agent.declare_catalog_link(
            sources.orders, sources.customers, (("customer_id", "id"),)
        )
        assert (
            await agent.declare_catalog_link(
                sources.orders, sources.customers, (("customer_id", "id"),)
            )
        ).id == first.id
        with pytest.raises(CatalogStoreError, match="at most 1"):
            await agent.declare_catalog_link(
                sources.customers, sources.orders, (("id", "customer_id"),)
            )
        assert len(await agent.list_catalog_links()) == 1
    finally:
        await agent.close()
