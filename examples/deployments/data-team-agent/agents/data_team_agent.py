"""Local data team agent template.

This module owns only project setup: a small SQLite fixture, local storage
paths, and helper functions for creating a DbAgent. Schema discovery, catalog
search, query planning, SQL execution, quality, lineage, memory, evidence,
verification, synthesis, approval/resume, and monitors stay owned by
``Agent.from_db()`` and ``DbRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from daita.agents.agent import Agent
from daita.db import DbRuntimeOptions
from daita.plugins.sqlite import SQLitePlugin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DIR = PROJECT_ROOT / ".daita" / "local"
MONITOR_ID = "data_team_pending_orders"
DEMO_NOW = "2026-06-25T12:00:00+00:00"

QUALITY_REQUEST = "Profile data quality for the orders table."
LINEAGE_REQUEST = "Trace lineage for the orders table."
MEMORY_RULE = "Completed orders use status value 'complete', not 'completed'."
CATALOG_QUERY = "Join orders to customers using their relationship and return records."

SEED_SQL = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    segment TEXT NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    ordered_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);

INSERT INTO customers (id, name, region, segment) VALUES
    (1, 'Ada Lovelace', 'North America', 'enterprise'),
    (2, 'Grace Hopper', 'North America', 'mid-market'),
    (3, 'Katherine Johnson', 'Europe', 'enterprise'),
    (4, 'Mary Jackson', 'Europe', 'startup'),
    (5, 'Dorothy Vaughan', 'North America', 'enterprise');

INSERT INTO products (id, name, category, unit_price) VALUES
    (1, 'Analytics Notebook', 'software', 120.00),
    (2, 'Data Quality Audit', 'service', 300.00),
    (3, 'Pipeline Support Plan', 'service', 450.00),
    (4, 'Query Optimization Guide', 'content', 80.00),
    (5, 'Lineage Workshop', 'service', 700.00);

INSERT INTO orders (id, customer_id, ordered_at, status) VALUES
    (1, 1, '2026-01-05', 'complete'),
    (2, 1, '2026-01-09', 'pending'),
    (3, 2, '2026-01-11', 'complete'),
    (4, 3, '2026-01-12', 'complete'),
    (5, 4, '2026-01-14', 'cancelled'),
    (6, 5, '2026-01-17', 'complete');

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 2, 120.00),
    (2, 1, 4, 1, 80.00),
    (3, 2, 2, 1, 300.00),
    (4, 3, 3, 1, 450.00),
    (5, 3, 1, 1, 120.00),
    (6, 4, 2, 2, 300.00),
    (7, 5, 4, 3, 80.00),
    (8, 6, 5, 1, 700.00),
    (9, 6, 2, 1, 300.00);
"""


@dataclass(frozen=True)
class DataTeamPaths:
    """Local files used by the deployment template."""

    base_dir: Path
    db_path: Path
    runtime_store_path: Path
    memory_dir: Path


def default_paths(base_dir: str | Path | None = None) -> DataTeamPaths:
    """Return the local, git-ignored project paths."""
    root = Path(base_dir) if base_dir is not None else DEFAULT_LOCAL_DIR
    root = root.expanduser().resolve()
    return DataTeamPaths(
        base_dir=root,
        db_path=root / "data" / "team_sales.sqlite",
        runtime_store_path=root / "runtime" / "runtime_store.sqlite",
        memory_dir=root / "memory",
    )


async def seed_local_sqlite(db_path: str | Path) -> Path:
    """Create the copyable local SQLite fixture."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plugin = SQLitePlugin(path=str(path))
    try:
        await plugin.execute_script(SEED_SQL)
    except ImportError as exc:
        raise ImportError(
            "SQLite support requires aiosqlite. Install with: "
            "pip install 'daita-agents[sqlite]'"
        ) from exc
    finally:
        await plugin.disconnect()
    return path


def memory_options(memory_dir: str | Path) -> dict[str, Any]:
    """Use local structured memory so the template works without embeddings."""
    from daita.plugins.memory import LocalMemoryBackend

    memory_path = Path(memory_dir)
    memory_path.mkdir(parents=True, exist_ok=True)
    return {
        "backend": LocalMemoryBackend(
            workspace="data_team_agent",
            scope="project",
            base_dir=memory_path,
        ),
        "recall": "auto",
        "learning": "off",
        "retrieval_mode": "structured",
        "limit": 3,
        "char_budget": 800,
    }


def llm_options(use_live_llm: bool) -> dict[str, Any]:
    """Keep live LLM calls behind an explicit flag."""
    if not use_live_llm:
        return {}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}
    return {
        "llm_provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        "api_key": api_key,
        "temperature": 0,
    }


async def create_data_team_agent(
    paths: DataTeamPaths | None = None,
    *,
    use_live_llm: bool = False,
):
    """Create the data-team DbAgent over the local SQLite fixture."""
    resolved = paths or default_paths()
    resolved.runtime_store_path.parent.mkdir(parents=True, exist_ok=True)
    return await Agent.from_db(
        str(resolved.db_path),
        name="DataTeamAgent",
        mode="data_team",
        quality=True,
        lineage=True,
        memory=memory_options(resolved.memory_dir),
        cache_ttl=0,
        runtime=DbRuntimeOptions(
            store="sqlite",
            store_path=resolved.runtime_store_path,
        ),
        **llm_options(use_live_llm),
    )


def pending_orders_observation() -> dict[str, Any]:
    """Executable monitor observation consumed by the runtime scheduler."""
    return {
        "kind": "metric_sql",
        "metric": "pending_count",
        "sql": "select count(*) as pending_count from orders where status = 'pending'",
        "value_path": "rows.0.pending_count",
        "source_scope": ["orders"],
        "capability_owner": "sqlite",
    }


async def ensure_pending_orders_monitor(agent):
    """Create or load the durable pending-orders monitor."""
    existing = await agent.inspect_monitor(MONITOR_ID)
    if existing is not None:
        return existing.monitor
    return await agent.monitor(
        monitor_id=MONITOR_ID,
        name="Pending Orders",
        schedule={"interval_seconds": 0},
        watch="Count pending orders in the local sales fixture.",
        observation_plan=pending_orders_observation(),
        trigger={"path": "pending_count", "gt": 10},
        source_scope=("orders",),
        metadata={"template": "data-team-agent"},
    )


def evidence_kinds(result) -> list[str]:
    """Return operation evidence kind names for display/tests."""
    return [item.kind for item in result.evidence]


def task_capability_sequence(result) -> list[str]:
    """Return executed task capability ids for display/tests."""
    tasks = result.diagnostics.get("execution", {}).get("tasks", [])
    return [task.get("capability_id", "") for task in tasks]
