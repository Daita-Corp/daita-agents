from daita.db import DbIntentKind, DbRequest, DbRuntime
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus


async def _seed(plugin: SQLitePlugin) -> None:
    await plugin.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total REAL NOT NULL
        );
        INSERT INTO customers (id, email, status) VALUES
            (1, 'ada@example.com', 'active'),
            (2, 'grace@example.com', 'inactive');
        INSERT INTO orders (id, customer_id, total) VALUES
            (10, 1, 42.5),
            (11, 2, 12.0);
        """)


async def _runtime() -> tuple[DbRuntime, SQLitePlugin]:
    sqlite = SQLitePlugin(path=":memory:", query_default_limit=10)
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False), sqlite))
    await runtime.setup(agent_id="phase-8-test")
    await _seed(sqlite)
    return runtime, sqlite


def _schema_with_legacy_status(schema: dict) -> dict:
    return {
        **schema,
        "tables": [
            {
                **table,
                "columns": [
                    *table.get("columns", []),
                    *(
                        [{"name": "legacy_status", "data_type": "TEXT"}]
                        if table.get("name") == "customers"
                        else []
                    ),
                ],
            }
            for table in schema.get("tables", [])
        ],
    }
