"""data_query joins complete bounded reads from several sources in DuckDB."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import daita.domains.data.capabilities as data_capabilities
from daita import Agent, AgentConfig, SQLiteSource
from daita.adapters.models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from daita.adapters.protocols import StaleResourceError
from daita.adapters.registry import SourceBackendContext, SourceOperation, SourceType
from daita.capabilities import CallTargetLane, CapabilityRegistry
from daita.capability_runtime import CapabilityRuntime
from daita.catalog.models import (
    CatalogFacet,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from daita.domains.data import (
    DATA_QUERY_CAPABILITY_ID,
    DATA_QUERY_COMBINED_CAPABILITY_ID,
    DATA_QUERY_COMBINED_EVIDENCE_KIND,
)
from daita.domains.data.capabilities import ReadTabularRequest, SqlReadResult
from daita.domains.data.results import project_result_rows
from daita.llm.models import ModelProfile, ModelSensitivity, ToolCall, ToolResultBlock
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from daita.scope import EffectiveSourceScope, scope_refs
from tests.support.capability_runtime import execute_projected
from tests.support.memory_source import MEMORY_ADAPTER_ID, MEMORY_TYPE, MemorySource
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.contract

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
REVENUE_BY_REGION = (
    "SELECT c.region, SUM(o.revenue) AS revenue FROM o "
    "JOIN c ON c.id = o.customer_id GROUP BY c.region ORDER BY c.region"
)
REGION_CUSTOMERS = (
    "SELECT r.region, r.revenue, COUNT(c.id) AS customers FROM r "
    "JOIN c ON c.region = r.region GROUP BY r.region, r.revenue ORDER BY r.region"
)
# Tabular inputs need a type that declares TABULAR_READ: the in-memory one.
_MEMORY = AgentConfig(source_types=(MEMORY_TYPE,))


@dataclass(frozen=True, slots=True)
class _Shop:
    agent: Agent
    orders_source: str
    customers_source: str
    orders: str
    customers: str
    customers_database: Path


async def _agent(tmp_path: Path, config: AgentConfig | None = None) -> Agent:
    provider = MockModelProvider(())
    return await Agent.create(
        "combined-query",
        root=tmp_path,
        config=config,
        clock=lambda: NOW,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=80_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        workspace=workspace_for(tmp_path),
    )


async def _shop(tmp_path: Path, config: AgentConfig | None = None) -> _Shop:
    orders_database = tmp_path / "orders.sqlite"
    customers_database = tmp_path / "customers.sqlite"
    with sqlite3.connect(orders_database) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, "
            "total REAL)"
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            ((1, 1, 10.5), (2, 2, 4.0), (3, 3, 2.25), (4, 1, 1.25), (5, 2, 0.5)),
        )
    with sqlite3.connect(customers_database) as connection:
        connection.execute(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, region TEXT)"
        )
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?)",
            ((1, "north"), (2, "south"), (3, "north")),
        )
    agent = await _agent(tmp_path, config)
    orders_source = await agent.attach(SQLiteSource(orders_database, name="Orders"))
    customers_source = await agent.attach(
        SQLiteSource(customers_database, name="Customers")
    )
    (orders,) = await agent.list_catalog_resources(source_id=orders_source.id)
    (customers,) = await agent.list_catalog_resources(source_id=customers_source.id)
    return _Shop(
        agent=agent,
        orders_source=orders_source.id,
        customers_source=customers_source.id,
        orders=orders.id,
        customers=customers.id,
        customers_database=customers_database,
    )


def _inputs(shop: _Shop) -> tuple[dict[str, object], ...]:
    return (
        {
            "alias": "o",
            "source_id": shop.orders_source,
            "resource_ids": (shop.orders,),
            "sql": "SELECT customer_id, SUM(total) AS revenue FROM orders "
            "GROUP BY customer_id",
        },
        {
            "alias": "c",
            "source_id": shop.customers_source,
            "resource_ids": (shop.customers,),
            "sql": "SELECT id, region FROM customers",
        },
    )


async def _regions(shop: _Shop) -> tuple[str, str]:
    """Attach the in-memory regions table, whose type has only tabular reads."""

    source = await shop.agent.attach(MemorySource("Regions"))
    (table,) = await shop.agent.list_catalog_resources(source_id=source.id)
    return source.id, table.id


def _tabular_regions(
    source_id: str, resource_id: str, **changes: object
) -> dict[str, object]:
    return {
        "alias": "r",
        "source_id": source_id,
        "resource_id": resource_id,
        "columns": ("region", "revenue"),
        **changes,
    }


def _tabular_requests(agent: Agent) -> list[ReadTabularRequest]:
    backends = agent._embedded._source_backends[SourceOperation.TABULAR_READ]
    return backends[MEMORY_ADAPTER_ID].requests  # type: ignore[no-any-return]


def _runtime(agent: Agent) -> CapabilityRuntime:
    loop = agent._embedded._loop
    assert loop is not None
    return cast(CapabilityRuntime, loop._tools)


def _run(agent: Agent, scope: tuple[str, ...]) -> RunInput:
    return RunInput(
        id="combined-query-run",
        agent_id=agent.id,
        message="join the shop sources",
        created_at=NOW,
        conversation_id="combined-query-conversation",
        source_scope_ids=scope,
    )


async def _call(
    shop: _Shop,
    arguments: Mapping[str, object],
    *,
    scope: tuple[str, ...] | None = None,
) -> ToolResultBlock:
    (result,) = await execute_projected(
        _runtime(shop.agent),
        _run(
            shop.agent,
            (shop.orders_source, shop.customers_source) if scope is None else scope,
        ),
        (ToolCall(id="combined", name="data_query", arguments=arguments),),
    )
    return cast(ToolResultBlock, result)


def _data(result: ToolResultBlock) -> Mapping[str, object]:
    assert "error" not in result.output, result.output
    assert result.output["kind"] == DATA_QUERY_COMBINED_EVIDENCE_KIND
    return cast(Mapping[str, object], result.output["data"])


def _error(result: ToolResultBlock) -> Mapping[str, object]:
    assert "data" not in result.output, result.output
    return cast(Mapping[str, object], result.output["error"])


class _CountingBackend:
    """Delegate source reads, counting them and running a hook before each."""

    def __init__(
        self,
        delegate: object,
        before: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self.delegate = delegate
        self.before = before
        self.reads: list[str] = []

    async def execute_read(self, **request: object) -> SqlReadResult:
        self.reads.append(cast(str, request["source_id"]))
        if self.before is not None:
            self.before(request)
        return await self.delegate.execute_read(**request)  # type: ignore[attr-defined]


def _wrap_sqlite_reads(
    agent: Agent, before: Callable[[Mapping[str, object]], None] | None = None
) -> _CountingBackend:
    registry: CapabilityRegistry = agent._embedded._capabilities
    _capability, executor = registry.resolve_execution(
        DATA_QUERY_COMBINED_CAPABILITY_ID
    )
    backends = executor._backends  # type: ignore[attr-defined]
    counting = _CountingBackend(backends["sqlite"], before)
    backends["sqlite"] = counting
    return counting


async def test_combined_query_totals_revenue_by_region_across_two_sqlite_sources(
    tmp_path: Path,
):
    shop = await _shop(tmp_path)
    try:
        result = await _call(shop, {"inputs": _inputs(shop), "sql": REVENUE_BY_REGION})
        data = _data(result)
        catalog = {
            item.id: item
            for source_id in (shop.orders_source, shop.customers_source)
            for item in await shop.agent.list_catalog_resources(source_id=source_id)
        }
    finally:
        await shop.agent.close()

    assert tuple(dict(row) for row in data["rows"]) == (  # type: ignore[attr-defined]
        {"region": "north", "revenue": 14.0},
        {"region": "south", "revenue": 4.5},
    )
    assert data["columns"] == ("region", "revenue")
    assert data["truncated"] is False
    assert data["resource_ids"] == tuple(sorted((shop.orders, shop.customers)))
    assert {
        item["resource_id"]: item["revision"]
        for item in cast(tuple[Mapping[str, object], ...], data["resource_revisions"])
    } == {resource_id: item.current_revision for resource_id, item in catalog.items()}
    inputs = cast(tuple[Mapping[str, object], ...], data["inputs"])
    assert [(item["alias"], item["source_id"], item["rows"]) for item in inputs] == [
        ("o", shop.orders_source, 3),
        ("c", shop.customers_source, 3),
    ]
    for item in inputs:
        assert str(item["canonical_sql"]).upper().startswith("SELECT")
        assert str(item["sql_fingerprint"]).startswith("sha256:")
    assert str(data["sql_fingerprint"]).startswith("sha256:")
    assert result.capability_id == DATA_QUERY_COMBINED_CAPABILITY_ID
    assert result.sensitivity_provenance is not None
    assert result.sensitivity_provenance["source_ids"] == tuple(
        sorted((shop.orders_source, shop.customers_source))
    )


async def test_tabular_input_named_by_run_handles_joins_a_sqlite_input(
    tmp_path: Path,
):
    shop = await _shop(tmp_path, _MEMORY)
    try:
        regions_source, regions = await _regions(shop)
        # The model may name the tabular input's identities by run handles.
        refs = scope_refs(
            EffectiveSourceScope(
                frozenset({shop.customers_source, regions_source}),
                frozenset({shop.customers, regions}),
            )
        )
        result = await _call(
            shop,
            {
                "inputs": (
                    _inputs(shop)[1],
                    _tabular_regions(
                        refs.handles[regions_source],
                        refs.handles[regions],
                        filters=({"column": "revenue", "operator": "gte", "value": 5},),
                    ),
                ),
                "sql": REGION_CUSTOMERS,
            },
            scope=(shop.customers_source, regions_source),
        )
        (table,) = await shop.agent.list_catalog_resources(source_id=regions_source)
    finally:
        await shop.agent.close()

    data = _data(result)
    assert tuple(dict(row) for row in data["rows"]) == (  # type: ignore[attr-defined]
        {"region": "north", "revenue": 15, "customers": 2},
        {"region": "south", "revenue": 7, "customers": 1},
    )
    tabular = cast(tuple[Mapping[str, object], ...], data["inputs"])[1]
    assert tabular["source_id"] == regions_source
    assert tabular["columns"] == ("region", "revenue")
    assert [dict(item) for item in tabular["filters"]] == [  # type: ignore[attr-defined]
        {"column": "revenue", "operator": "gte", "value": 5}
    ]
    assert "canonical_sql" not in tabular
    assert [dict(item) for item in tabular["resource_revisions"]] == [  # type: ignore[attr-defined]
        {"resource_id": regions, "revision": table.current_revision}
    ]


async def test_an_input_with_no_rows_still_aggregates(tmp_path: Path):
    shop = await _shop(tmp_path)
    try:
        orders, customers = _inputs(shop)
        result = await _call(
            shop,
            {
                "inputs": (
                    {**orders, "sql": f"{orders['sql']} HAVING SUM(total) > 1000"},
                    customers,
                ),
                "sql": "SELECT c.region, SUM(o.revenue) AS revenue FROM c "
                "LEFT JOIN o ON o.customer_id = c.id GROUP BY c.region "
                "ORDER BY c.region",
            },
        )
    finally:
        await shop.agent.close()

    assert tuple(dict(row) for row in _data(result)["rows"]) == (  # type: ignore[attr-defined]
        {"region": "north", "revenue": None},
        {"region": "south", "revenue": None},
    )


async def test_sql_input_without_catalog_revisions_is_refused_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shop = await _shop(tmp_path)
    view = shop.agent._embedded._data_view
    schemas = view.resource_schemas

    async def without_revisions(agent_id: str, source_id: str, *args, **kwargs):
        found = await schemas(agent_id, source_id, *args, **kwargs)
        if source_id != shop.customers_source:
            return found
        return tuple(replace(item, revision=None) for item in found)

    monkeypatch.setattr(view, "resource_schemas", without_revisions)
    try:
        reads = _wrap_sqlite_reads(shop.agent)
        result = await _call(shop, {"inputs": _inputs(shop), "sql": REVENUE_BY_REGION})
    finally:
        await shop.agent.close()

    error = _error(result)
    assert error["code"] == "catalog_provenance_missing"
    assert cast(Mapping[str, object], error["details"])["input_alias"] == "c"
    assert reads.reads == []


async def test_combining_sql_may_self_join_and_stack_inputs(tmp_path: Path):
    shop = await _shop(tmp_path)
    try:
        stacked = await _call(
            shop,
            {
                "inputs": _inputs(shop),
                "sql": "SELECT COUNT(*) AS n FROM (SELECT customer_id AS id FROM o "
                "UNION ALL SELECT id FROM c) AS ids",
            },
        )
        paired = await _call(
            shop,
            {
                "inputs": _inputs(shop),
                "sql": "SELECT COUNT(*) AS n FROM c JOIN c AS twin "
                "ON twin.region = c.region AND twin.id <> c.id "
                "JOIN o ON o.customer_id = c.id",
            },
        )
    finally:
        await shop.agent.close()

    assert tuple(dict(row) for row in _data(stacked)["rows"]) == ({"n": 6},)  # type: ignore[attr-defined]
    # Customers 1 and 3 share north; both have orders.
    assert tuple(dict(row) for row in _data(paired)["rows"]) == ({"n": 2},)  # type: ignore[attr-defined]


async def test_inputs_route_only_with_the_default_output(tmp_path: Path):
    shop = await _shop(tmp_path)
    try:
        registry: CapabilityRegistry = shop.agent._embedded._capabilities
        combined = {"inputs": _inputs(shop), "sql": REVENUE_BY_REGION}
        assert (
            registry.call_capability("data_query", combined).id
            == DATA_QUERY_COMBINED_CAPABILITY_ID
        )
        assert (
            registry.call_capability("data_query", {**combined, "output": "rows"}).id
            == DATA_QUERY_COMBINED_CAPABILITY_ID
        )
        assert (
            registry.call_capability(
                "data_query", {"source_id": shop.orders_source, "sql": "SELECT 1"}
            ).id
            == DATA_QUERY_CAPABILITY_ID
        )
        schema = registry.tool_definition(
            "data_query", lane=CallTargetLane.FOREGROUND
        ).input_schema
        properties = cast(Mapping[str, Mapping[str, object]], schema["properties"])
        assert "inputs" not in cast(tuple[str, ...], properties["output"]["enum"])
        assert "inputs" in properties

        exported = await _call(shop, {**combined, "output": "csv"})
    finally:
        await shop.agent.close()

    error = _error(exported)
    assert error["code"] == "invalid_argument_for_mode"
    assert error["message"] == "inputs do not apply when output=csv."


async def test_definition_describes_only_the_routes_a_run_admits(tmp_path: Path):
    shop = await _shop(tmp_path)
    try:
        registry: CapabilityRegistry = shop.agent._embedded._capabilities
        full = registry.tool_definition("data_query", lane=CallTargetLane.FOREGROUND)
        # A machine run admits no combined or live-folder route.
        narrowed = registry.tool_definition(
            "data_query", lane=None, routes=("rows", "csv", "xlsx", "arrow")
        )
    finally:
        await shop.agent.close()

    full_properties = cast(Mapping[str, object], full.input_schema["properties"])
    assert "Join sources" in full.description and "Folder files" in full.description
    assert {"inputs", "path_pattern"} <= set(full_properties)
    properties = cast(Mapping[str, object], narrowed.input_schema["properties"])
    assert "Join sources" not in narrowed.description
    assert "Folder files" not in narrowed.description
    assert not {"inputs", "path_pattern"} & set(properties)
    assert {"source_id", "resource_ids", "sql", "output"} <= set(properties)


async def test_two_inputs_on_one_sqlite_source_share_its_single_permit(tmp_path: Path):
    shop = await _shop(tmp_path)
    try:
        result = await asyncio.wait_for(
            _call(
                shop,
                {
                    "inputs": (
                        {
                            "alias": "o",
                            "source_id": shop.orders_source,
                            "resource_ids": (shop.orders,),
                            "sql": "SELECT customer_id, total FROM orders",
                        },
                        {
                            "alias": "big",
                            "source_id": shop.orders_source,
                            "resource_ids": (shop.orders,),
                            "sql": "SELECT id FROM orders WHERE total > 2",
                        },
                    ),
                    "sql": "SELECT COUNT(*) AS n FROM o JOIN big ON big.id = "
                    "o.customer_id",
                },
            ),
            timeout=60,
        )
    finally:
        await shop.agent.close()

    # Every order's customer id is an order id above 2.
    assert tuple(dict(row) for row in _data(result)["rows"]) == (  # type: ignore[attr-defined]
        {"n": 5},
    )


class _RecordingCoordinator:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def source_resource_permit(self, key: str, **_: object) -> _Permit:
        self.events.append(f"acquire {key}")
        return _Permit(self.events, key)


class _Permit:
    def __init__(self, events: list[str], key: str) -> None:
        self._events = events
        self._key = key

    async def __aenter__(self) -> _Permit:
        return self

    async def __aexit__(self, *_: object) -> None:
        self._events.append(f"release {self._key}")


async def test_input_permits_are_taken_once_each_in_sorted_source_order(
    tmp_path: Path,
):
    agent = await _agent(tmp_path)
    try:
        runtime = _runtime(agent)
        recorder = _RecordingCoordinator()
        runtime._admission_coordinator = recorder  # type: ignore[assignment]
        capability = agent._embedded._capabilities.capability(
            DATA_QUERY_COMBINED_CAPABILITY_ID
        )
        permit = await runtime._call_io_permit(
            _run(agent, ()),
            capability,
            {
                "inputs": (
                    {"source_id": "source:b"},
                    {"source_id": "source:a"},
                    {"source_id": "source:b"},
                )
            },
            None,
            session=None,
        )
        assert permit is not None
        assert recorder.events == ["acquire source:source:a", "acquire source:source:b"]
        async with permit:
            pass
    finally:
        await agent.close()

    assert recorder.events[2:] == ["release source:source:b", "release source:source:a"]


async def test_input_exactly_at_its_row_ceiling_joins_and_past_its_bytes_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Both inputs return exactly three rows.
    monkeypatch.setattr(data_capabilities, "COMBINED_QUERY_INPUT_MAX_ROWS", 3)
    shop = await _shop(tmp_path)
    try:
        at_ceiling = await _call(
            shop, {"inputs": _inputs(shop), "sql": REVENUE_BY_REGION}
        )
        monkeypatch.setattr(data_capabilities, "COMBINED_QUERY_INPUT_MAX_BYTES", 16)
        past_bytes = await _call(
            shop, {"inputs": _inputs(shop), "sql": REVENUE_BY_REGION}
        )
    finally:
        await shop.agent.close()

    assert _data(at_ceiling)["truncated"] is False
    error = _error(past_bytes)
    assert error["code"] == "combined_input_truncated"
    details = cast(Mapping[str, object], error["details"])
    assert details["truncation_reasons"] == ("byte_limit",)


@pytest.mark.parametrize("orders_form", ["sql", "tabular"])
async def test_input_past_its_internal_ceiling_refuses_the_whole_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orders_form: str
):
    monkeypatch.setattr(data_capabilities, "COMBINED_QUERY_INPUT_MAX_ROWS", 2)
    shop = await _shop(tmp_path, _MEMORY)
    try:
        arguments: dict[str, object] = {
            "inputs": _inputs(shop),
            "sql": REVENUE_BY_REGION,
        }
        scope = None
        if orders_form == "tabular":
            regions_source, regions = await _regions(shop)
            arguments = {
                "inputs": (_tabular_regions(regions_source, regions), _inputs(shop)[1]),
                "sql": REGION_CUSTOMERS,
            }
            scope = (shop.customers_source, regions_source)
        result = await _call(shop, arguments, scope=scope)
    finally:
        await shop.agent.close()

    error = _error(result)
    assert error["code"] == "combined_input_truncated"
    details = cast(Mapping[str, object], error["details"])
    assert details["alias"] == ("r" if orders_form == "tabular" else "o")
    assert details["truncation_reasons"] == ("row_limit",)


async def test_schema_change_after_validation_fails_the_whole_call_as_stale(
    tmp_path: Path,
):
    shop = await _shop(tmp_path)
    inputs = _inputs(shop)

    def alter_customers(request: Mapping[str, object]) -> None:
        if request["source_id"] == shop.customers_source:
            with sqlite3.connect(shop.customers_database) as connection:
                connection.execute("ALTER TABLE customers ADD COLUMN tier TEXT")

    try:
        reads = _wrap_sqlite_reads(shop.agent, alter_customers)
        result = await _call(shop, {"inputs": inputs, "sql": REVENUE_BY_REGION})
    finally:
        await shop.agent.close()

    assert reads.reads == [shop.orders_source, shop.customers_source]
    assert _error(result)["code"] == "catalog_source_stale"


@pytest.mark.parametrize("customers_form", ["sql", "tabular"])
@pytest.mark.parametrize("foreign", ["source", "resource"])
async def test_input_outside_the_run_scope_is_refused_before_any_source_read(
    tmp_path: Path, foreign: str, customers_form: str
):
    shop = await _shop(tmp_path, _MEMORY)
    try:
        reads = _wrap_sqlite_reads(shop.agent)
        inputs = list(_inputs(shop))
        if customers_form == "tabular":
            inputs[1] = _tabular_regions(*await _regions(shop), alias="c")
        if foreign == "resource":
            # The input names an in-scope source but reads a table of another.
            inputs[1] = {**inputs[1], "source_id": shop.orders_source}
        result = await _call(
            shop,
            {"inputs": tuple(inputs), "sql": REVENUE_BY_REGION},
            scope=(shop.orders_source,),
        )
        tabular_reads = _tabular_requests(shop.agent)
    finally:
        await shop.agent.close()

    error = _error(result)
    assert error["code"] == (
        "source_scope_violation" if foreign == "source" else "resource_read_not_allowed"
    )
    assert cast(Mapping[str, object], error["details"])["input_alias"] == "c"
    assert reads.reads == []
    assert tabular_reads == []


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("SELECT * FROM read_csv('/etc/passwd')", "table_function"),
        ("SELECT * FROM read_json('https://example.test/a.json')", "table_function"),
        ("ATTACH '/tmp/evil.db' AS evil", "select_only"),
        ("COPY o TO '/tmp/evil.csv'", "select_only"),
        ("INSTALL httpfs", "select_only"),
        ("LOAD httpfs", "select_only"),
        ("SELECT * FROM o JOIN c ON true JOIN secrets ON true", "relation_forbidden"),
        ("SELECT * FROM o; SELECT * FROM c", "statement_count"),
        ("SELECT * FROM o", "data_reference_count"),
    ],
    ids=[
        "file",
        "http",
        "attach",
        "copy",
        "install",
        "load",
        "unknown_table",
        "two_statements",
        "unused_input",
    ],
)
async def test_combining_sql_outside_the_input_aliases_is_rejected_before_execution(
    tmp_path: Path, sql: str, reason: str
):
    shop = await _shop(tmp_path)
    try:
        reads = _wrap_sqlite_reads(shop.agent)
        result = await _call(shop, {"inputs": _inputs(shop), "sql": sql})
    finally:
        await shop.agent.close()

    error = _error(result)
    assert error["code"] == "combined_query_invalid"
    assert cast(Mapping[str, object], error["details"])["reason"] == reason
    assert reads.reads == []


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        (
            {"filters": ({"column": "region", "operator": "ne", "value": "north"},)},
            "invalid_argument_value",
        ),
        (
            {"filters": ({"column": "region", "operator": "like", "value": "n%"},)},
            "invalid_argument_value",
        ),
        (
            {"filters": ({"column": "region", "operator": "is_null", "value": True},)},
            "invalid_argument_value",
        ),
        (
            {"filters": ({"column": "region", "operator": "in", "value": "north"},)},
            "tabular_filter_invalid",
        ),
        (
            {"filters": ({"column": "region", "operator": "in", "value": ()},)},
            "tabular_filter_invalid",
        ),
        (
            {"filters": ({"column": "region", "operator": "in", "value": (("n",),)},)},
            "tabular_filter_invalid",
        ),
        (
            {"filters": ({"column": "revenue", "operator": "eq", "value": (1, 2)},)},
            "tabular_filter_invalid",
        ),
        (
            {"filters": ({"column": "revenue", "operator": "gte", "value": None},)},
            "tabular_filter_invalid",
        ),
        (
            {"filters": ({"column": "revenue", "operator": "lt", "value": {"n": 2}},)},
            "tabular_filter_invalid",
        ),
        ({"columns": ("region", "country")}, "tabular_column_unknown"),
        (
            {"filters": ({"column": "tier", "operator": "eq", "value": "gold"},)},
            "tabular_column_unknown",
        ),
    ],
    ids=[
        "not_equal",
        "like",
        "null_test",
        "in_scalar",
        "in_empty",
        "in_nested",
        "eq_array",
        "range_null",
        "range_object",
        "unknown_column",
        "unknown_filter_column",
    ],
)
async def test_tabular_input_outside_equality_in_and_range_on_catalog_columns_is_refused(
    tmp_path: Path, changes: Mapping[str, object], code: str
):
    shop = await _shop(tmp_path, _MEMORY)
    try:
        regions_source, regions = await _regions(shop)
        reads = _wrap_sqlite_reads(shop.agent)
        result = await _call(
            shop,
            {
                "inputs": (
                    _inputs(shop)[1],
                    _tabular_regions(regions_source, regions, **changes),
                ),
                "sql": REGION_CUSTOMERS,
            },
            scope=(shop.customers_source, regions_source),
        )
        tabular_reads = _tabular_requests(shop.agent)
    finally:
        await shop.agent.close()

    assert _error(result)["code"] == code
    assert reads.reads == []
    assert tabular_reads == []


CONFIDENTIAL_ADAPTER_ID = "confidential-targets"


class _TargetsAdapter:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

    async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
        registration = self.registration
        now = request.requested_at
        resource_id = catalog_resource_id(
            registration.id, ResourceKind.TABLE, "targets"
        )
        facet = CatalogFacet.from_tabular(
            resource_id=resource_id,
            sync_id=request.sync_id,
            observed_at=now,
            facet=TabularFacet(
                columns=(
                    TabularColumn(
                        name="region", native_type="text", ordinal=0, nullable=False
                    ),
                    TabularColumn(
                        name="target", native_type="real", ordinal=1, nullable=False
                    ),
                )
            ),
        )
        revision = CatalogResourceRevision.build(
            resource_id=resource_id,
            sync_id=request.sync_id,
            observed_at=now,
            facet_revisions=(facet.revision,),
            source_revision="targets:v1",
        )
        resource = CatalogResource.build(
            agent_id=registration.agent_id,
            source_id=registration.id,
            native_identity="targets",
            external_uri="memory://targets",
            kind=ResourceKind.TABLE,
            name="targets",
            sensitivity=Sensitivity.CONFIDENTIAL,
            revision=revision,
            first_observed_at=now,
            last_observed_at=now,
        )
        snapshot = SourceCatalogSnapshot(
            sync=CatalogSync(
                id=request.sync_id,
                agent_id=registration.agent_id,
                source_id=registration.id,
                adapter_id=CONFIDENTIAL_ADAPTER_ID,
                status=CatalogSyncStatus.SUCCEEDED,
                started_at=now,
                completed_at=now,
                source_revision="targets:v1",
                resource_count=1,
            ),
            resources=(resource,),
            revisions=(revision,),
            facets=(facet,),
        )
        return DiscoveryResult(request=request, snapshot=snapshot, completed_at=now)

    async def inspect(self, resource: ResourceRef) -> ResourceSnapshot:
        raise StaleResourceError(self.registration.id, resource.resource_id)

    async def health(self) -> SourceHealth:
        return SourceHealth(
            agent_id=self.registration.agent_id,
            source_id=self.registration.id,
            adapter_id=CONFIDENTIAL_ADAPTER_ID,
            healthy=True,
            checked_at=NOW,
        )

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _TargetsSource:
    async def open(self, *, agent_id: str, attached_at: datetime, clock: object):
        del clock
        return _TargetsAdapter(
            SourceRegistration.build(
                agent_id=agent_id,
                adapter_id=CONFIDENTIAL_ADAPTER_ID,
                native_identity="memory:targets",
                display_name="Targets",
                configuration={},
                attached_at=attached_at,
            )
        )


class _TargetsBackend:
    async def execute_read(self, **request: object) -> SqlReadResult:
        revisions = cast(
            tuple[tuple[str, str], ...], request["expected_resource_revisions"]
        )
        return SqlReadResult(
            source_id=str(request["source_id"]),
            canonical_sql=str(request["sql"]),
            sql_fingerprint="sha256:" + "2" * 64,
            resource_ids=tuple(resource_id for resource_id, _ in revisions),
            resource_revisions=revisions,
            source_revision="targets:v1",
            columns=("region", "target"),
            projection=project_result_rows(
                (
                    {"region": "north", "target": 20.0},
                    {"region": "south", "target": 4.0},
                ),
                max_rows=cast(int, request["max_rows"]),
                max_bytes=cast(int, request["max_bytes"]),
            ),
        )


def _targets_type() -> SourceType:
    backend = _TargetsBackend()

    def backends(context: SourceBackendContext) -> Mapping[SourceOperation, object]:
        del context
        return {SourceOperation.SQL_READ: backend}

    return SourceType(
        adapter_id=CONFIDENTIAL_ADAPTER_ID,
        label="Confidential targets",
        from_registration=lambda registration, *, secret_provider: _TargetsSource(),
        sql_dialect="sqlite",
        operations=frozenset({SourceOperation.SQL_READ}),
        create_backends=backends,
    )


async def test_result_takes_the_highest_input_sensitivity_and_each_type_backend(
    tmp_path: Path,
):
    shop = await _shop(tmp_path, AgentConfig(source_types=(_targets_type(),)))
    try:
        targets_source = await shop.agent.attach(_TargetsSource())
        (targets,) = await shop.agent.list_catalog_resources(
            source_id=targets_source.id
        )
        result = await _call(
            shop,
            {
                "inputs": (
                    _inputs(shop)[0],
                    _inputs(shop)[1],
                    {
                        "alias": "t",
                        "source_id": targets_source.id,
                        "resource_ids": (targets.id,),
                        "sql": "SELECT region, target FROM targets",
                    },
                ),
                "sql": "SELECT c.region, SUM(o.revenue) AS revenue, "
                "MAX(t.target) AS target FROM o JOIN c ON c.id = o.customer_id "
                "JOIN t ON t.region = c.region GROUP BY c.region ORDER BY c.region",
            },
            scope=(shop.orders_source, shop.customers_source, targets_source.id),
        )
    finally:
        await shop.agent.close()

    assert tuple(dict(row) for row in _data(result)["rows"]) == (  # type: ignore[attr-defined]
        {"region": "north", "revenue": 14.0, "target": 20.0},
        {"region": "south", "revenue": 4.5, "target": 4.0},
    )
    assert result.sensitivity is ModelSensitivity.CONFIDENTIAL


async def test_tabular_input_on_a_type_without_tabular_reads_is_refused(
    tmp_path: Path,
):
    shop = await _shop(tmp_path, AgentConfig(source_types=(_targets_type(),)))
    try:
        targets_source = await shop.agent.attach(_TargetsSource())
        (targets,) = await shop.agent.list_catalog_resources(
            source_id=targets_source.id
        )
        reads = _wrap_sqlite_reads(shop.agent)
        result = await _call(
            shop,
            {
                "inputs": (
                    _inputs(shop)[1],
                    {
                        "alias": "t",
                        "source_id": targets_source.id,
                        "resource_id": targets.id,
                        "columns": ("region", "target"),
                    },
                ),
                "sql": "SELECT c.region, t.target FROM c JOIN t ON t.region = c.region",
            },
            scope=(shop.customers_source, targets_source.id),
        )
    finally:
        await shop.agent.close()

    error = _error(result)
    assert error["code"] == "tabular_source_adapter_unsupported"
    assert cast(Mapping[str, object], error["details"])["input_alias"] == "t"
    assert reads.reads == []
