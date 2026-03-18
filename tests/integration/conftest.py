"""
Shared fixtures for integration tests.

Tests in this directory use an in-memory SQLite database — no external service
required, just `pip install 'daita-agents[sqlite]'`.
"""

import pytest

# Skip the entire integration suite if aiosqlite is not installed
aiosqlite = pytest.importorskip(
    "aiosqlite",
    reason="aiosqlite required: pip install 'daita-agents[sqlite]'",
)

from daita.plugins.sqlite import SQLitePlugin


@pytest.fixture
async def db():
    """
    Fresh in-memory SQLite database for each test.

    WAL mode is disabled for :memory: databases (not meaningful in-process).
    The fixture connects on setup and disconnects on teardown so tests never
    need to call connect()/disconnect() themselves.
    """
    plugin = SQLitePlugin(path=":memory:", wal_mode=False)
    await plugin.connect()
    yield plugin
    await plugin.disconnect()


@pytest.fixture
async def orders_db(db):
    """
    SQLite database pre-seeded with an orders table and summary target table.

    Schema
    ------
    orders      : id, customer_id, amount, status
    order_totals: customer_id, total          (empty — filled by transformations)
    """
    await db.execute_script("""
        CREATE TABLE orders (
            id          INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            amount      REAL,
            status      TEXT,
            created_at  TEXT
        );
        CREATE TABLE order_totals (
            customer_id INTEGER,
            total       REAL
        );
    """)
    await db.insert_many(
        "orders",
        [
            {
                "id": 1,
                "customer_id": 100,
                "amount": 50.0,
                "status": "active",
                "created_at": "2024-01-01T10:00:00",
            },
            {
                "id": 2,
                "customer_id": 101,
                "amount": 150.0,
                "status": "inactive",
                "created_at": "2024-01-01T11:00:00",
            },
            {
                "id": 3,
                "customer_id": 100,
                "amount": 75.0,
                "status": "active",
                "created_at": "2024-01-01T12:00:00",
            },
            {
                "id": 4,
                "customer_id": 102,
                "amount": 200.0,
                "status": "active",
                "created_at": "2024-01-01T13:00:00",
            },
            {
                "id": 5,
                "customer_id": 102,
                "amount": 25.0,
                "status": "inactive",
                "created_at": "2024-01-01T14:00:00",
            },
        ],
    )
    return db
