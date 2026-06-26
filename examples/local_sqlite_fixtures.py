"""Local SQLite fixtures for the data-first examples.

The helpers in this file only create demo data. Query planning, schema
inspection, SQL validation, execution, evidence, and joins stay owned by
``Agent.from_db()`` and the DB runtime.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import AsyncIterator

from daita.plugins.sqlite import SQLitePlugin

SALES_SEED_SQL = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL
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

INSERT INTO customers (id, name, region) VALUES
    (1, 'Ada Lovelace', 'North America'),
    (2, 'Grace Hopper', 'North America'),
    (3, 'Katherine Johnson', 'Europe'),
    (4, 'Mary Jackson', 'Europe');

INSERT INTO products (id, name, category, unit_price) VALUES
    (1, 'Analytics Notebook', 'software', 120.00),
    (2, 'Data Quality Audit', 'service', 300.00),
    (3, 'Pipeline Support Plan', 'service', 450.00),
    (4, 'Query Optimization Guide', 'content', 80.00);

INSERT INTO orders (id, customer_id, ordered_at, status) VALUES
    (1, 1, '2026-01-05', 'complete'),
    (2, 1, '2026-01-09', 'pending'),
    (3, 2, '2026-01-11', 'complete'),
    (4, 3, '2026-01-12', 'complete'),
    (5, 4, '2026-01-14', 'cancelled');

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 2, 120.00),
    (2, 1, 4, 1, 80.00),
    (3, 2, 2, 1, 300.00),
    (4, 3, 3, 1, 450.00),
    (5, 3, 1, 1, 120.00),
    (6, 4, 2, 2, 300.00),
    (7, 5, 4, 3, 80.00);
"""


async def seed_sales_sqlite(path: str | Path) -> Path:
    """Create a small relational sales database and return its path."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    plugin = SQLitePlugin(path=str(db_path))
    try:
        await plugin.execute_script(SALES_SEED_SQL)
    except ImportError as exc:
        raise ImportError(
            "SQLite examples require aiosqlite. Install with: "
            "pip install 'daita-agents[sqlite]'"
        ) from exc
    finally:
        await plugin.disconnect()
    return db_path


@asynccontextmanager
async def temporary_sales_sqlite(
    filename: str = "daita_sales.sqlite",
) -> AsyncIterator[Path]:
    """Yield a freshly seeded SQLite database in a temporary directory."""
    with TemporaryDirectory(prefix="daita_examples_") as tmpdir:
        db_path = await seed_sales_sqlite(Path(tmpdir) / filename)
        yield db_path
