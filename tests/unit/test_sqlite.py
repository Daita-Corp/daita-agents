"""
Unit tests for SQLitePlugin.

All tests use :memory: databases — no files, no external services required.
"""

import pytest
from daita.plugins.sqlite import SQLitePlugin, sqlite

SCHEMA = """
CREATE TABLE users (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL,
    age  INTEGER
);
CREATE TABLE orders (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    total   REAL
);
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db():
    """Connected in-memory SQLite database with a basic schema."""
    async with SQLitePlugin(path=":memory:") as plugin:
        await plugin.execute_script(SCHEMA)
        yield plugin


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_connect_disconnect():
    plugin = SQLitePlugin(path=":memory:")
    assert not plugin.is_connected
    await plugin.connect()
    assert plugin.is_connected
    await plugin.disconnect()
    assert not plugin.is_connected


async def test_context_manager():
    async with sqlite(path=":memory:") as db:
        assert db.is_connected
    assert not db.is_connected


async def test_double_connect_is_idempotent():
    async with sqlite(path=":memory:") as db:
        await db.connect()  # second call — should not raise
        assert db.is_connected


# ---------------------------------------------------------------------------
# execute_script
# ---------------------------------------------------------------------------


async def test_execute_script_creates_tables(db):
    tables = await db.tables()
    assert "users" in tables
    assert "orders" in tables


# ---------------------------------------------------------------------------
# execute + query
# ---------------------------------------------------------------------------


async def test_insert_and_query(db):
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Alice", 30])
    rows = await db.query("SELECT * FROM users")
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["age"] == 30


async def test_execute_returns_affected_rows(db):
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Bob", 25])
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Carol", 35])
    affected = await db.execute("UPDATE users SET age = 99 WHERE age < 30")
    assert affected == 1


async def test_delete_returns_affected_rows(db):
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Dave", 20])
    affected = await db.execute("DELETE FROM users WHERE name = ?", ["Dave"])
    assert affected == 1


async def test_query_with_no_results(db):
    rows = await db.query("SELECT * FROM users WHERE name = ?", ["Nobody"])
    assert rows == []


async def test_query_returns_list_of_dicts(db):
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Eve", 28])
    rows = await db.query("SELECT id, name FROM users")
    assert isinstance(rows[0], dict)
    assert "id" in rows[0]
    assert "name" in rows[0]


# ---------------------------------------------------------------------------
# insert_many
# ---------------------------------------------------------------------------


async def test_insert_many(db):
    data = [
        {"name": "Frank", "age": 40},
        {"name": "Grace", "age": 32},
        {"name": "Hank", "age": 27},
    ]
    count = await db.insert_many("users", data)
    assert count == 3
    rows = await db.query("SELECT * FROM users ORDER BY name")
    assert len(rows) == 3
    assert rows[0]["name"] == "Frank"


async def test_insert_many_empty_returns_zero(db):
    count = await db.insert_many("users", [])
    assert count == 0


# ---------------------------------------------------------------------------
# tables + describe
# ---------------------------------------------------------------------------


async def test_tables(db):
    tables = await db.tables()
    assert set(tables) == {"users", "orders"}


async def test_describe_column_names(db):
    columns = await db.describe("users")
    names = [c["column_name"] for c in columns]
    assert "id" in names
    assert "name" in names
    assert "age" in names


async def test_describe_primary_key(db):
    columns = await db.describe("users")
    pk_col = next(c for c in columns if c["column_name"] == "id")
    assert pk_col["is_primary_key"] is True


async def test_describe_nullable(db):
    columns = await db.describe("users")
    name_col = next(c for c in columns if c["column_name"] == "name")
    age_col = next(c for c in columns if c["column_name"] == "age")
    assert name_col["is_nullable"] == "NO"
    assert age_col["is_nullable"] == "YES"


# ---------------------------------------------------------------------------
# pragma
# ---------------------------------------------------------------------------


async def test_pragma_read(db):
    val = await db.pragma("journal_mode")
    assert val is not None  # e.g. "memory" for :memory:


async def test_pragma_write(db):
    await db.pragma("cache_size", -4000)
    val = await db.pragma("cache_size")
    assert int(val) == -4000


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


async def test_get_tools_returns_correct_tools(db):
    tools = db.get_tools()
    names = [t.name for t in tools]
    assert "sqlite_query" in names
    assert "sqlite_execute" in names
    assert "sqlite_inspect" in names
    assert "sqlite_count" in names
    assert "sqlite_sample" in names
    # list_tables and get_schema removed from default tools
    assert "sqlite_list_tables" not in names
    assert "sqlite_get_schema" not in names


async def test_tool_query(db):
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Ivy", 22])
    result = await db._tool_query({"sql": "SELECT * FROM users LIMIT 10"})
    assert result["total_rows"] == 1
    assert result["rows"][0]["name"] == "Ivy"


async def test_tool_query_with_params(db):
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Jack", 45])
    result = await db._tool_query(
        {"sql": "SELECT * FROM users WHERE age > ? LIMIT 10", "params": [30]}
    )
    assert result["total_rows"] == 1


async def test_tool_execute(db):
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Kate", 29])
    result = await db._tool_execute(
        {"sql": "UPDATE users SET age = ? WHERE name = ?", "params": [30, "Kate"]}
    )
    assert result["affected_rows"] == 1


async def test_tool_list_tables(db):
    """_tool_list_tables kept for backward compat."""
    result = await db._tool_list_tables({})
    assert "users" in result["tables"]


async def test_tool_get_schema(db):
    """_tool_get_schema kept for backward compat."""
    result = await db._tool_get_schema({"table_name": "users"})
    assert result["table"] == "users"
    assert len(result["columns"]) == 3


async def test_tool_inspect(db):
    result = await db._tool_inspect({})
    assert result["total_tables"] == 2
    table_names = [t["name"] for t in result["tables"]]
    assert "users" in table_names
    assert "orders" in table_names


async def test_tool_inspect_filtered(db):
    result = await db._tool_inspect({"tables": ["users"]})
    assert result["total_tables"] == 1
    assert result["tables"][0]["name"] == "users"


async def test_tool_count(db):
    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Leo", 25])
    result = await db._tool_count({"table": "users"})
    assert result["count"] == 1
    assert result["table"] == "users"


async def test_tool_sample(db):
    data = [{"name": "Sam", "age": 30}, {"name": "Tina", "age": 28}]
    await db.insert_many("users", data)
    result = await db._tool_sample({"table": "users", "n": 2})
    assert len(result["rows"]) == 2
    assert result["table"] == "users"


# ---------------------------------------------------------------------------
# compact_column helper
# ---------------------------------------------------------------------------


async def test_compact_column_format(db):
    columns = await db.describe("users")
    compact = [db._compact_column(c) for c in columns]
    # Should be "colname:datatype:null_status" format
    for c in compact:
        parts = c.split(":")
        assert len(parts) == 3
        assert parts[2] in ("nn", "null")


# ---------------------------------------------------------------------------
# query_checked (inherited from BaseDatabasePlugin)
# ---------------------------------------------------------------------------


async def test_query_checked_passes(db):
    from daita.core.assertions import ItemAssertion

    await db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ["Leo", 25])
    rows = await db.query_checked(
        "SELECT * FROM users",
        assertions=[
            ItemAssertion(lambda r: r["name"] is not None, "name must not be null")
        ],
    )
    assert len(rows) == 1


async def test_query_checked_raises_on_violation(db):
    from daita.core.assertions import ItemAssertion
    from daita.core.exceptions import DataQualityError

    # Insert a row that will violate a not-null check on age
    await db.execute("INSERT INTO users (name) VALUES (?)", ["Mia"])

    with pytest.raises(DataQualityError):
        await db.query_checked(
            "SELECT * FROM users",
            assertions=[
                ItemAssertion(lambda r: r["age"] is not None, "age must not be null")
            ],
        )


# ---------------------------------------------------------------------------
# Missing dependency path
# ---------------------------------------------------------------------------


async def test_missing_aiosqlite_raises_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "aiosqlite":
            raise ImportError("aiosqlite not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    plugin = SQLitePlugin(path=":memory:")
    from daita.core.exceptions import PluginError

    with pytest.raises(PluginError):
        await plugin.connect()
