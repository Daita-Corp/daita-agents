"""
Live integration test for CatalogPlugin + MySQL.

Mirror of test_catalog_postgres_live.py against a throwaway MySQL
container. Same four pillars: plugin works, graph correct, soft speed
logging, live-LLM accuracy.

Requirements:
  - docker
  - aiomysql (pip install 'daita-agents[mysql]')
  - OPENAI_API_KEY (for the live-LLM section)

Run:
    OPENAI_API_KEY=sk-... pytest \\
      tests/integration/test_catalog_mysql_live.py -v -s -m "integration"
"""

from __future__ import annotations

import asyncio
import time

import pytest

aiomysql = pytest.importorskip(
    "aiomysql", reason="aiomysql required: pip install 'daita-agents[mysql]'"
)

from daita.core.graph import LocalGraphBackend
from daita.core.graph.models import NodeType
from daita.plugins.catalog import CatalogPlugin

from ._harness import (
    assert_answer_mentions,
    assert_tool_called,
    build_live_agent,
    start_container,
    timed,
)

MYSQL_IMAGE = "mysql:8.0"
MYSQL_ROOT_PASSWORD = "daita_root_pw"
MYSQL_USER = "daita"
MYSQL_PASSWORD = "daita_test_pw"
MYSQL_DB = "daita_test"


# MySQL's entrypoint script creates $MYSQL_DATABASE owned by $MYSQL_USER
# automatically. We use that DB and connect as the non-root user.
SEED_SQL_STATEMENTS = [
    """
    CREATE TABLE customers (
        customer_id  INT AUTO_INCREMENT PRIMARY KEY,
        name         VARCHAR(100) NOT NULL,
        email        VARCHAR(200) UNIQUE,
        signup_date  DATE NOT NULL
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE orders (
        order_id     INT AUTO_INCREMENT PRIMARY KEY,
        customer_id  INT NOT NULL,
        amount       DECIMAL(10,2) NOT NULL,
        status       VARCHAR(40),
        created_at   DATETIME NOT NULL,
        CONSTRAINT fk_orders_customer
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    ) ENGINE=InnoDB
    """,
    "CREATE INDEX orders_customer_idx ON orders(customer_id)",
    "INSERT INTO customers (name, email, signup_date) VALUES "
    "('Alice', 'alice@example.com', '2026-01-01'), "
    "('Bob',   'bob@example.com',   '2026-01-02')",
    "INSERT INTO orders (customer_id, amount, status, created_at) VALUES "
    "(1, 50.00, 'shipped', '2026-01-03 10:00:00'), "
    "(2, 150.00, 'pending', '2026-01-03 11:00:00')",
]


EXPECTED_TABLES = {"customers", "orders"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mysql_container():
    container = start_container(
        MYSQL_IMAGE,
        container_port=3306,
        env={
            "MYSQL_ROOT_PASSWORD": MYSQL_ROOT_PASSWORD,
            "MYSQL_USER": MYSQL_USER,
            "MYSQL_PASSWORD": MYSQL_PASSWORD,
            "MYSQL_DATABASE": MYSQL_DB,
        },
        tag_prefix="daita-it-mysql",
        # MySQL's bootstrap is slow — it writes its datadir, creates the
        # user, restarts. 120s is a safe cap; real time is ~20–45s.
        readiness_timeout=180.0,
    )
    try:
        yield container
    finally:
        container.remove()


@pytest.fixture(scope="module")
def mysql_url(mysql_container) -> str:
    return (
        f"mysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{mysql_container.host}:{mysql_container.host_port}/{MYSQL_DB}"
    )


@pytest.fixture(scope="module")
def seeded_mysql(mysql_url, mysql_container) -> str:
    """Seed the known schema; tolerate the auth-not-ready-yet window that
    follows TCP readiness."""

    async def _seed():
        deadline = time.time() + 120
        last_err = None
        while time.time() < deadline:
            try:
                conn = await aiomysql.connect(
                    host=mysql_container.host,
                    port=mysql_container.host_port,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    db=MYSQL_DB,
                    autocommit=True,
                )
                try:
                    async with conn.cursor() as cur:
                        for stmt in SEED_SQL_STATEMENTS:
                            await cur.execute(stmt)
                finally:
                    conn.close()
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await asyncio.sleep(1.0)
        raise RuntimeError(f"Could not seed MySQL: {last_err}")

    asyncio.run(_seed())
    return mysql_url


@pytest.fixture
async def plugin_with_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="catalog_mysql_live")
    plugin = CatalogPlugin(backend=backend, auto_persist=False)
    plugin.initialize("catalog-mysql-live")
    return plugin, backend


# ---------------------------------------------------------------------------
# (1) Catalog plugin works with MySQL
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestCatalogPluginMySQL:
    async def test_discover_schema_returns_seeded_tables(
        self, plugin_with_backend, seeded_mysql
    ):
        plugin, _ = plugin_with_backend

        discover_tool = next(
            t for t in plugin.get_tools() if t.name == "discover_schema"
        )
        async with timed("discover_schema mysql"):
            result = await discover_tool.execute(
                {
                    "store_type": "mysql",
                    "connection_string": seeded_mysql,
                    "options": {"schema": MYSQL_DB, "ssl_mode": "disable"},
                    "persist": False,
                }
            )

        # Raw discover_* shape: tables[].table_name, columns in a flat sibling array.
        schema = result.get("schema") or result
        table_names = {t["table_name"] for t in schema.get("tables", [])}
        assert (
            EXPECTED_TABLES <= table_names
        ), f"Expected {EXPECTED_TABLES} in {table_names}"

        fk_triples = {
            (fk["source_table"], fk["source_column"], fk["target_table"])
            for fk in schema.get("foreign_keys", [])
        }
        assert ("orders", "customer_id", "customers") in fk_triples


# ---------------------------------------------------------------------------
# (2) Graph built correctly
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestMySQLGraphCorrectness:
    async def test_persist_populates_graph(self, plugin_with_backend, seeded_mysql):
        plugin, backend = plugin_with_backend

        discover_tool = next(
            t for t in plugin.get_tools() if t.name == "discover_schema"
        )
        async with timed("discover_schema mysql + persist"):
            await discover_tool.execute(
                {
                    "store_type": "mysql",
                    "connection_string": seeded_mysql,
                    "options": {"schema": MYSQL_DB, "ssl_mode": "disable"},
                    "persist": True,
                }
            )

        tables = await backend.find_nodes(NodeType.TABLE)
        table_names = {t.name.split(".")[-1] for t in tables}
        assert EXPECTED_TABLES <= table_names, f"Got: {table_names}"


# ---------------------------------------------------------------------------
# (3) + (4) Live-LLM traversal: accuracy + speed
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.requires_db
class TestAgentMySQLLive:
    async def test_agent_discovers_and_describes_schema(
        self, plugin_with_backend, seeded_mysql
    ):
        plugin, _ = plugin_with_backend

        agent = build_live_agent(name="MySQLCatalogAgent", tools=[plugin])
        async with timed("agent.run mysql describe"):
            result = await agent.run(
                f"Use the catalog tools to profile this MySQL database: "
                f"`{seeded_mysql}`. Important: this is a local container "
                f"without TLS, so pass options={{'schema': '{MYSQL_DB}', "
                f"'ssl_mode': 'disable'}} to discover_schema. Then tell me "
                f"the names of all tables and how `orders` relates to "
                f"`customers`. Be concise.",
                detailed=True,
            )

        assert_tool_called(result, "discover_schema")
        assert_answer_mentions(result, ["customers", "orders"])
        assert_answer_mentions(
            result,
            ["customer_id", "foreign key", "references"],
            any_of=True,
        )
