"""
Phase 6 validation for catalog-first from_db behavior.

These tests are intentionally diagnostic-heavy. They print and optionally write
full JSON result bundles so a live run shows how the agent behaved, which tools
it used, and which trace spans were emitted.

Run examples:
    OPENAI_API_KEY=sk-... pytest tests/integration/from_db/test_from_db_phase6_live.py -v -s
    DAITA_FROM_DB_PHASE6_MODELS=gpt-4o-mini,gpt-4.1-mini pytest \
      tests/integration/from_db/test_from_db_phase6_live.py -v -s -m "requires_llm and requires_db"
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from daita.agents.agent import Agent
from daita.core.graph import LocalGraphBackend
from daita.core.tracing import get_trace_manager
from daita.llm.pricing.catalog import PRICE_CATALOG
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.base_discoverer import DiscoveredStore
from daita.plugins.catalog.base_profiler import (
    NormalizedColumn,
    NormalizedForeignKey,
    NormalizedSchema,
    NormalizedTable,
)

from tests.integration._harness import start_container

POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_USER = "daita"
POSTGRES_PASSWORD = "daita_test_pw"
POSTGRES_DB = "daita_phase6"

MYSQL_IMAGE = "mysql:8.0"
MYSQL_ROOT_PASSWORD = "daita_root_pw"
MYSQL_USER = "daita"
MYSQL_PASSWORD = "daita_test_pw"
MYSQL_DB = "daita_phase6"

PHASE6_POSTGRES_SQL = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;

CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    segment      TEXT NOT NULL
);

CREATE TABLE orders (
    order_id     SERIAL PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    status       TEXT NOT NULL,
    created_at   TIMESTAMP NOT NULL
);

CREATE TABLE products (
    product_id   SERIAL PRIMARY KEY,
    sku          TEXT NOT NULL,
    category     TEXT NOT NULL
);

CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    amount        NUMERIC(10, 2) NOT NULL
);

CREATE TABLE shipments (
    shipment_id SERIAL PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(order_id),
    carrier     TEXT NOT NULL
);

INSERT INTO customers (name, segment) VALUES
    ('Alice', 'enterprise'),
    ('Bob', 'commercial');
INSERT INTO orders (customer_id, status, created_at) VALUES
    (1, 'shipped', '2026-05-01T10:00:00'),
    (2, 'pending', '2026-05-02T10:00:00');
INSERT INTO products (sku, category) VALUES
    ('SKU-001', 'analytics'),
    ('SKU-002', 'storage');
INSERT INTO order_items (order_id, product_id, amount) VALUES
    (1, 1, 50.00),
    (1, 2, 75.00),
    (2, 2, 150.00);
INSERT INTO shipments (order_id, carrier) VALUES
    (1, 'UPS');
"""

PHASE6_MYSQL_STATEMENTS = [
    "DROP TABLE IF EXISTS order_items",
    "DROP TABLE IF EXISTS orders",
    "DROP TABLE IF EXISTS customers",
    """
    CREATE TABLE customers (
        customer_id INT AUTO_INCREMENT PRIMARY KEY,
        name        VARCHAR(100) NOT NULL,
        segment     VARCHAR(80) NOT NULL
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE orders (
        order_id    INT AUTO_INCREMENT PRIMARY KEY,
        customer_id INT NOT NULL,
        amount      DECIMAL(10,2) NOT NULL,
        status      VARCHAR(40) NOT NULL,
        CONSTRAINT fk_phase6_orders_customer
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE order_items (
        order_item_id INT AUTO_INCREMENT PRIMARY KEY,
        order_id      INT NOT NULL,
        amount        DECIMAL(10,2) NOT NULL,
        CONSTRAINT fk_phase6_items_order
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
    ) ENGINE=InnoDB
    """,
    "INSERT INTO customers (name, segment) VALUES ('Alice', 'enterprise'), ('Bob', 'commercial')",
    "INSERT INTO orders (customer_id, amount, status) VALUES (1, 125.00, 'shipped'), (2, 150.00, 'pending')",
    "INSERT INTO order_items (order_id, amount) VALUES (1, 50.00), (1, 75.00), (2, 150.00)",
]


def _phase6_models() -> list[str]:
    raw = os.environ.get("DAITA_FROM_DB_PHASE6_MODELS", "gpt-4o-mini,gpt-4.1-mini")
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if len(models) < 2:
        pytest.skip("DAITA_FROM_DB_PHASE6_MODELS must include at least two models")
    missing = [model for model in models if not _catalog_has_openai_model(model)]
    if missing:
        raise AssertionError(f"Models missing from llm pricing catalog: {missing}")
    return models


def _catalog_has_openai_model(model: str) -> bool:
    return any(fnmatch.fnmatch(model, pattern) for pattern in PRICE_CATALOG["openai"])


def _openai_kwargs(model: str) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping live phase 6 LLM test")
    kwargs: dict[str, Any] = {
        "llm_provider": "openai",
        "api_key": api_key,
        "model": model,
        "max_tokens": 700,
    }
    temperature = os.environ.get("DAITA_FROM_DB_PHASE6_TEMPERATURE")
    kwargs["temperature"] = float(temperature) if temperature is not None else 0
    return kwargs


def _phase6_output_dir() -> Path | None:
    raw = os.environ.get("DAITA_FROM_DB_PHASE6_OUTPUT")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _emit_phase6_output(case_id: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, default=str)
    print(f"\n[PHASE6:{case_id}]\n{text}\n", file=sys.stderr)
    output_dir = _phase6_output_dir()
    if output_dir is not None:
        (output_dir / f"{case_id}.json").write_text(text)


def _trace_bundle(trace_id: str | None, agent_id: str | None = None) -> dict[str, Any]:
    trace_manager = get_trace_manager()
    trace_manager.flush()
    spans = trace_manager._memory_exporter.get_finished_spans()
    selected = []
    for span in spans:
        context = span.get_span_context()
        span_trace_id = format(context.trace_id, "032x") if context else None
        attrs = dict(span.attributes or {})
        if (
            trace_id
            and span_trace_id != trace_id
            and attrs.get("daita.agent.id") != agent_id
        ):
            continue
        if not trace_id and agent_id and attrs.get("daita.agent.id") != agent_id:
            continue
        selected.append(_span_to_full_dict(span))
    operations = [span["operation"] for span in selected]
    return {
        "trace_id": trace_id,
        "span_count": len(selected),
        "operations": operations,
        "spans": selected,
    }


def _span_to_full_dict(span) -> dict[str, Any]:
    context = span.get_span_context()
    parent = format(span.parent.span_id, "016x") if span.parent else None
    return {
        "trace_id": format(context.trace_id, "032x") if context else None,
        "span_id": format(context.span_id, "016x") if context else None,
        "parent_span_id": parent,
        "operation": span.name,
        "status": span.status.status_code.name if span.status else None,
        "duration_ms": (
            (span.end_time - span.start_time) / 1_000_000
            if span.start_time and span.end_time
            else None
        ),
        "attributes": dict(span.attributes or {}),
        "events": [
            {
                "name": event.name,
                "attributes": dict(event.attributes or {}),
            }
            for event in span.events
        ],
    }


async def _close_agent_db(agent) -> None:
    plugin = getattr(getattr(agent, "db", None), "plugin", None)
    if plugin is not None and hasattr(plugin, "disconnect"):
        await plugin.disconnect()


@pytest.fixture(scope="module")
def phase6_postgres_container():
    container = start_container(
        POSTGRES_IMAGE,
        container_port=5432,
        env={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DB,
        },
        tag_prefix="daita-phase6-pg",
    )
    try:
        yield container
    finally:
        container.remove()


@pytest.fixture(scope="module")
def phase6_postgres_url(phase6_postgres_container) -> str:
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{phase6_postgres_container.host}:{phase6_postgres_container.host_port}"
        f"/{POSTGRES_DB}?sslmode=disable"
    )


@pytest.fixture(scope="module")
def seeded_phase6_postgres(phase6_postgres_url) -> str:
    asyncpg = pytest.importorskip(
        "asyncpg",
        reason="asyncpg required for PostgreSQL phase 6 integration tests",
    )

    async def _seed() -> None:
        deadline = time.time() + 30
        last_err = None
        while time.time() < deadline:
            try:
                conn = await asyncpg.connect(phase6_postgres_url)
                await conn.execute(PHASE6_POSTGRES_SQL)
                await conn.close()
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Could not seed phase 6 Postgres: {last_err}")

    asyncio.run(_seed())
    return phase6_postgres_url


@pytest.fixture(scope="module")
def phase6_mysql_container():
    container = start_container(
        MYSQL_IMAGE,
        container_port=3306,
        env={
            "MYSQL_ROOT_PASSWORD": MYSQL_ROOT_PASSWORD,
            "MYSQL_USER": MYSQL_USER,
            "MYSQL_PASSWORD": MYSQL_PASSWORD,
            "MYSQL_DATABASE": MYSQL_DB,
        },
        tag_prefix="daita-phase6-mysql",
        readiness_timeout=180.0,
    )
    try:
        yield container
    finally:
        container.remove()


@pytest.fixture(scope="module")
def phase6_mysql_url(phase6_mysql_container) -> str:
    return (
        f"mysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{phase6_mysql_container.host}:{phase6_mysql_container.host_port}/{MYSQL_DB}"
        "?sslmode=disable"
    )


@pytest.fixture(scope="module")
def seeded_phase6_mysql(phase6_mysql_url, phase6_mysql_container) -> str:
    aiomysql = pytest.importorskip(
        "aiomysql",
        reason="aiomysql required for MySQL phase 6 integration tests",
    )

    async def _seed() -> None:
        deadline = time.time() + 120
        last_err = None
        while time.time() < deadline:
            try:
                conn = await aiomysql.connect(
                    host=phase6_mysql_container.host,
                    port=phase6_mysql_container.host_port,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    db=MYSQL_DB,
                    autocommit=True,
                )
                try:
                    async with conn.cursor() as cursor:
                        for statement in PHASE6_MYSQL_STATEMENTS:
                            await cursor.execute(statement)
                finally:
                    conn.close()
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await asyncio.sleep(1.0)
        raise RuntimeError(f"Could not seed phase 6 MySQL: {last_err}")

    asyncio.run(_seed())
    return phase6_mysql_url


def _schema(
    store_id: str,
    database_type: str,
    database_name: str,
    tables: list[NormalizedTable],
    foreign_keys: list[NormalizedForeignKey] | None = None,
) -> NormalizedSchema:
    return NormalizedSchema(
        database_type=database_type,
        database_name=database_name,
        tables=tables,
        foreign_keys=foreign_keys or [],
        table_count=len(tables),
        store_id=store_id,
        profiled_at=datetime.now(timezone.utc).isoformat(),
    )


def _table(
    name: str,
    *,
    columns: int = 4,
    asset_type: str = "table",
    row_count: int | None = 10,
    comment_prefix: str = "",
) -> NormalizedTable:
    generated = [
        NormalizedColumn(
            name="id" if index == 0 else f"field_{index:03d}",
            type="text" if index else "integer",
            nullable=index != 0,
            is_primary_key=index == 0,
            comment=f"{comment_prefix} field {index}" if comment_prefix else None,
        )
        for index in range(columns)
    ]
    return NormalizedTable(
        name=name,
        row_count=row_count,
        columns=generated,
        metadata={"asset_type": asset_type, "description": comment_prefix},
    )


def _phase6_catalog(tmp_path: Path) -> CatalogPlugin:
    plugin = CatalogPlugin(
        backend=LocalGraphBackend(graph_type=f"phase6_{tmp_path.name}"),
        auto_persist=False,
    )
    plugin.initialize("phase6-catalog")

    huge_tables = [_table(f"warehouse_table_{index:05d}") for index in range(10_000)]
    huge_tables.append(
        NormalizedTable(
            name="wide_fact_events",
            row_count=2,
            columns=[
                NormalizedColumn(
                    name="event_id" if index == 0 else f"feature_{index - 1:03d}",
                    type="text",
                    nullable=index != 0,
                    is_primary_key=index == 0,
                    comment="late channel feature" if index == 520 else None,
                )
                for index in range(521)
            ],
            metadata={"asset_type": "table", "description": "wide event features"},
        )
    )
    huge_tables.extend(
        [
            _table("public.orders", comment_prefix="ambiguous order facts"),
            _table("analytics.orders", comment_prefix="ambiguous revenue facts"),
            _table("customers", comment_prefix="customer dimension"),
            _table("order_items", comment_prefix="line item facts"),
            _table("products", comment_prefix="product dimension"),
        ]
    )
    huge_fks = [
        NormalizedForeignKey("order_items", "order_id", "public.orders", "id"),
        NormalizedForeignKey("public.orders", "customer_id", "customers", "id"),
        NormalizedForeignKey("order_items", "product_id", "products", "id"),
    ]
    _register_schema(
        plugin,
        _schema("phase6:warehouse", "postgresql", "warehouse", huge_tables, huge_fks),
    )

    mixed_tables = [
        _table(
            "/v1/orders",
            asset_type="endpoint",
            columns=8,
            comment_prefix="orders API endpoint parameters and response fields",
        ),
        _table(
            "drive/orders.xlsx",
            asset_type="sheet",
            columns=12,
            comment_prefix="orders spreadsheet columns",
        ),
        _table(
            "docs/orders-runbook",
            asset_type="document",
            columns=6,
            comment_prefix="orders runbook metadata",
        ),
        _table(
            "orders",
            asset_type="table",
            columns=8,
            comment_prefix="orders database table",
        ),
    ]
    _register_schema(
        plugin,
        _schema("phase6:mixed", "catalog", "mixed_sources", mixed_tables),
    )
    return plugin


def _register_schema(plugin: CatalogPlugin, schema: NormalizedSchema) -> None:
    assert schema.store_id
    plugin._schemas[schema.store_id] = schema
    plugin._discovered_stores[schema.store_id] = DiscoveredStore(
        id=schema.store_id,
        store_type=schema.database_type,
        display_name=schema.database_name,
        connection_hint={},
        source="phase6-test",
        confidence=1.0,
        tags=["phase6"],
        metadata={"profiled_at": schema.profiled_at},
    )


async def _direct_postgres_catalog_discovery(connection_string: str) -> dict[str, Any]:
    plugin = CatalogPlugin(
        backend=LocalGraphBackend(graph_type="phase6_direct_postgres"),
        auto_persist=False,
    )
    plugin.initialize("phase6-direct-postgres")
    discover_tool = next(
        tool for tool in plugin.get_tools() if tool.name == "discover_schema"
    )
    discovery = await discover_tool.execute(
        {
            "store_type": "postgresql",
            "connection_string": connection_string,
            "options": {"schema": "public", "ssl_mode": "disable"},
            "persist": False,
        }
    )
    store_id = discovery["store_id"]
    search = plugin.catalog_search_schema(store_id, "customers orders", limit=5)
    inspect = plugin.get_table_schema(store_id, "orders", limit=20)
    return {
        "discovery": discovery,
        "search": search,
        "inspect": inspect,
    }


@pytest.mark.integration
def test_phase6_synthetic_catalog_scale_and_mixed_sources(tmp_path):
    plugin = _phase6_catalog(tmp_path)

    warehouse_summary = plugin.summarize_store("phase6:warehouse", limit=25)
    wide_page_1 = plugin.get_table_schema(
        "phase6:warehouse", "wide_fact_events", offset=0, limit=100
    )
    wide_page_late = plugin.get_table_schema(
        "phase6:warehouse", "wide_fact_events", offset=500, limit=40
    )
    search = plugin.catalog_search_schema(
        "phase6:warehouse", "late channel feature", limit=5
    )
    ambiguous = plugin.get_table_schema("phase6:warehouse", "orders", limit=10)
    relationships = plugin.find_relationship_paths(
        "phase6:warehouse",
        ["order_items"],
        ["customers"],
        max_hops=3,
        max_paths=2,
    )
    mixed_api = plugin.search_catalog(
        "phase6:mixed", "orders", asset_types=["endpoint"], limit=3
    )
    mixed_files = plugin.search_catalog(
        "phase6:mixed", "orders", asset_types=["sheet", "document"], limit=3
    )
    mixed_tables = plugin.search_catalog(
        "phase6:mixed", "orders", asset_types=["table"], limit=3
    )

    diagnostics = {
        "warehouse_summary": warehouse_summary,
        "wide_page_1": wide_page_1,
        "wide_page_late": wide_page_late,
        "search": search,
        "ambiguous": ambiguous,
        "relationships": relationships,
        "mixed_api": mixed_api,
        "mixed_files": mixed_files,
        "mixed_tables": mixed_tables,
    }
    _emit_phase6_output("synthetic_catalog_scale_mixed_sources", diagnostics)

    assert warehouse_summary["table_count"] == 10_006
    assert warehouse_summary["truncated"] is True
    assert wide_page_1["column_count"] == 521
    assert wide_page_1["truncated"] is True
    assert any(column["name"] == "feature_519" for column in wide_page_late["columns"])
    assert search["total_matches"] >= 1
    assert len(search["tables"]) <= 5
    assert ambiguous["success"] is False
    assert len(ambiguous["candidates"]) >= 2
    assert relationships["reachable"] is True
    assert relationships["path_count"] <= 2
    assert mixed_api["assets"][0]["asset_type"] == "endpoint"
    assert {asset["asset_type"] for asset in mixed_files["assets"]} <= {
        "sheet",
        "document",
    }
    assert mixed_tables["assets"][0]["asset_type"] == "table"


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.requires_llm
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping live phase 6 LLM test",
)
@pytest.mark.parametrize("model", _phase6_models())
async def test_phase6_postgres_from_db_openai_models_emit_full_traces(
    seeded_phase6_postgres, model, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    get_trace_manager()._memory_exporter.clear()
    agent = await Agent.from_db(
        seeded_phase6_postgres,
        name=f"phase6-postgres-{model}",
        mode="data_team",
        db_schema="public",
        cache_ttl=None,
        query_default_limit=20,
        query_max_rows=20,
        query_max_chars=4000,
        memory=True,
        lineage=True,
        **_openai_kwargs(model),
    )
    try:
        direct_catalog = await _direct_postgres_catalog_discovery(
            seeded_phase6_postgres
        )
        store_id = agent._db_catalog_store_id
        catalog_search = agent._db_catalog.catalog_search_schema(
            store_id, "customers orders", limit=5
        )
        catalog_inspect = agent._db_catalog.get_table_schema(
            store_id, "orders", limit=20
        )
        relationship_paths = agent._db_catalog.find_relationship_paths(
            store_id,
            ["order_items"],
            ["customers"],
            max_hops=3,
            max_paths=3,
        )

        result = await agent.run(
            "Use db_query exactly once with this SQL: "
            "SELECT name FROM customers WHERE customer_id = 1. "
            "Return the customer name and do not call catalog tools for this simple select.",
            detailed=True,
            max_iterations=3,
        )
        trace = _trace_bundle(result.get("_daita_trace_id"), result.get("agent_id"))
        tool_names = [call.get("tool") for call in result.get("tool_calls") or []]
        diagnostics = {
            "model": model,
            "catalog_model_entry_verified": _catalog_has_openai_model(model),
            "store_id": store_id,
            "direct_catalog_discovery": direct_catalog,
            "catalog_search": catalog_search,
            "catalog_inspect": catalog_inspect,
            "relationship_paths": relationship_paths,
            "agent_result": result,
            "trace": trace,
            "agent_metrics": get_trace_manager().get_agent_metrics(result["agent_id"]),
        }
        _emit_phase6_output(f"postgres_openai_{model}", diagnostics)

        assert "Alice" in result.get("result", "")
        assert tool_names == ["db_query"], tool_names
        assert result["iterations"] <= 3
        assert result["tokens"]["total_tokens"] > 0
        assert result["cost"] >= 0
        assert relationship_paths["reachable"] is True
        assert direct_catalog["search"]["total_matches"] >= 2
        assert direct_catalog["inspect"]["success"] is True
        assert catalog_search["total_matches"] >= 2
        assert catalog_inspect["success"] is True
        _assert_trace_operations(
            trace,
            {
                "from_db.fast_path",
                "from_db.prepare_runtime_context",
                "from_db.memory_recall",
                "from_db.build_runtime_context",
                "from_db.select_tools",
                "agent_run",
                "llm_openai",
                "tool_db_query",
                "from_db.validate_sql",
                "from_db.execute_sql",
            },
        )
    finally:
        await _close_agent_db(agent)


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.requires_llm
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping live phase 6 LLM test",
)
async def test_phase6_mysql_from_db_live_trace_and_simple_select(
    seeded_phase6_mysql, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    model = _phase6_models()[0]
    get_trace_manager()._memory_exporter.clear()
    agent = await Agent.from_db(
        seeded_phase6_mysql,
        name=f"phase6-mysql-{model}",
        mode="governed",
        db_schema=MYSQL_DB,
        cache_ttl=None,
        query_default_limit=20,
        query_max_rows=20,
        query_max_chars=4000,
        memory=True,
        lineage=True,
        **_openai_kwargs(model),
    )
    try:
        store_id = agent._db_catalog_store_id
        catalog_search = agent._db_catalog.catalog_search_schema(
            store_id, "orders amount customers", limit=5
        )
        catalog_inspect = agent._db_catalog.get_table_schema(
            store_id, "orders", limit=20
        )
        result = await agent.run(
            "Use db_query exactly once with this SQL: "
            "SELECT amount FROM orders WHERE order_id = 1. "
            "Return the amount and do not inspect schema for this simple select.",
            detailed=True,
            max_iterations=3,
        )
        trace = _trace_bundle(result.get("_daita_trace_id"), result.get("agent_id"))
        tool_names = [call.get("tool") for call in result.get("tool_calls") or []]
        diagnostics = {
            "model": model,
            "store_id": store_id,
            "catalog_search": catalog_search,
            "catalog_inspect": catalog_inspect,
            "agent_result": result,
            "trace": trace,
            "agent_metrics": get_trace_manager().get_agent_metrics(result["agent_id"]),
        }
        _emit_phase6_output("mysql_openai_simple_select", diagnostics)

        assert "125" in result.get("result", "")
        assert tool_names == ["db_query"], tool_names
        assert result["iterations"] <= 3
        assert result["tokens"]["total_tokens"] > 0
        assert result["cost"] >= 0
        assert catalog_search["total_matches"] >= 2
        assert catalog_inspect["success"] is True
        _assert_trace_operations(
            trace,
            {
                "from_db.fast_path",
                "from_db.prepare_runtime_context",
                "from_db.memory_recall",
                "from_db.build_runtime_context",
                "from_db.select_tools",
                "agent_run",
                "llm_openai",
                "tool_db_query",
                "from_db.validate_sql",
                "from_db.execute_sql",
            },
        )
    finally:
        await _close_agent_db(agent)


def _assert_trace_operations(trace: dict[str, Any], expected: set[str]) -> None:
    operations = set(trace.get("operations") or [])
    missing = expected - operations
    assert not missing, {
        "missing": sorted(missing),
        "operations": sorted(operations),
    }
