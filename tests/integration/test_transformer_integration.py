"""
Integration tests for TransformerPlugin against a real SQLite database.

These tests verify that transformations produce correct data, not just that
the right methods were called. No mocks are used for the database layer.

Run with:
    pytest tests/integration/test_transformer_integration.py -v
"""

import pytest
from daita.plugins.transformer import TransformerPlugin


def make_tx(db=None):
    plugin = TransformerPlugin(db=db)
    plugin._agent_id = "integration-test"
    return plugin


# ---------------------------------------------------------------------------
# Basic execution — does the SQL actually run and produce rows?
# ---------------------------------------------------------------------------


async def test_transform_run_inserts_real_rows(orders_db):
    """Running a transformation actually writes data to the target table."""
    tx = make_tx(db=orders_db)
    await tx.transform_create(
        "summarize_orders",
        sql=(
            "INSERT INTO order_totals "
            "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id"
        ),
        description="Aggregate orders per customer",
    )

    result = await tx.transform_run(orders_db, "summarize_orders")
    assert result["success"] is True

    rows = await orders_db.query(
        "SELECT customer_id, total FROM order_totals ORDER BY customer_id"
    )
    assert len(rows) == 3
    totals = {r["customer_id"]: r["total"] for r in rows}
    assert totals[100] == pytest.approx(125.0)  # 50 + 75
    assert totals[101] == pytest.approx(150.0)
    assert totals[102] == pytest.approx(225.0)  # 200 + 25


async def test_transform_run_respects_parameter_substitution(orders_db):
    """
    :param substitution actually filters rows — verified by inspecting
    the target table contents, not just the SQL string.
    """
    await orders_db.execute_script(
        "CREATE TABLE active_orders (id INTEGER, customer_id INTEGER, amount REAL);"
    )
    tx = make_tx(db=orders_db)
    await tx.transform_create(
        "copy_active",
        sql=(
            "INSERT INTO active_orders "
            "SELECT id, customer_id, amount FROM orders WHERE status = :status"
        ),
    )

    result = await tx.transform_run(
        orders_db, "copy_active", parameters={"status": "active"}
    )
    assert result["success"] is True

    rows = await orders_db.query("SELECT * FROM active_orders")
    # Only active rows (ids 1, 3, 4) should be present
    assert len(rows) == 3
    assert all(r["amount"] > 0 for r in rows)


async def test_transform_run_increments_run_count_correctly(orders_db):
    """run_count is exactly N after N executions — not just 'incremented'."""
    tx = make_tx(db=orders_db)
    await tx.transform_create(
        "count_orders",
        sql="INSERT INTO order_totals SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
    )

    for _ in range(3):
        await tx.transform_run(orders_db, "count_orders")

    assert tx._definitions["count_orders"]["run_count"] == 3


async def test_transform_run_records_last_run_timestamp(orders_db):
    """last_run is populated with execution metadata after a run."""
    tx = make_tx(db=orders_db)
    await tx.transform_create("noop", sql="SELECT 1")

    await tx.transform_run(orders_db, "noop")

    last_run = tx._definitions["noop"].get("last_run")
    assert last_run is not None
    assert last_run["success"] is True
    assert last_run["duration_ms"] >= 0
    assert "executed_at" in last_run


# ---------------------------------------------------------------------------
# Chained transformations — A → B → C
# ---------------------------------------------------------------------------


async def test_chained_transformations_produce_correct_output(orders_db):
    """
    Two transformations run in sequence: raw orders → filtered → aggregated.
    The final table contents must match the expected values.
    """
    await orders_db.execute_script("""
        CREATE TABLE active_only (id INTEGER, customer_id INTEGER, amount REAL);
        CREATE TABLE active_totals (customer_id INTEGER, total REAL);
    """)

    tx = make_tx(db=orders_db)

    await tx.transform_create(
        "filter_active",
        sql="INSERT INTO active_only SELECT id, customer_id, amount FROM orders WHERE status = 'active'",
    )
    await tx.transform_create(
        "aggregate_active",
        sql="INSERT INTO active_totals SELECT customer_id, SUM(amount) FROM active_only GROUP BY customer_id",
    )

    await tx.transform_run(orders_db, "filter_active")
    await tx.transform_run(orders_db, "aggregate_active")

    rows = await orders_db.query(
        "SELECT customer_id, total FROM active_totals ORDER BY customer_id"
    )
    assert len(rows) == 2  # only customers 100 and 102 have active orders
    totals = {r["customer_id"]: r["total"] for r in rows}
    assert totals[100] == pytest.approx(125.0)  # 50 + 75
    assert totals[102] == pytest.approx(200.0)  # 200 only (25 is inactive)


# ---------------------------------------------------------------------------
# transform_test — EXPLAIN against a real DB
# ---------------------------------------------------------------------------


async def test_transform_test_valid_sql_passes(orders_db):
    """transform_test returns valid=True for SQL that SQLite can parse and plan."""
    tx = make_tx(db=orders_db)
    await tx.transform_create(
        "explain_me",
        sql="INSERT INTO order_totals SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
    )

    result = await tx.transform_test(orders_db, "explain_me")
    assert result["success"] is True
    assert result["valid"] is True
    assert result["plan"]  # SQLite EXPLAIN returns opcode rows


async def test_transform_test_invalid_sql_returns_valid_false(orders_db):
    """transform_test returns valid=False and an error message for bad SQL."""
    tx = make_tx(db=orders_db)
    await tx.transform_create("bad_sql", sql="INSERT INTO nonexistent_table SELECT 1")

    result = await tx.transform_test(orders_db, "bad_sql")
    assert result["success"] is True
    assert result["valid"] is False
    assert result["error"]


async def test_transform_test_dummy_params_substituted_before_explain(orders_db):
    """
    TX-10: :param placeholders are replaced with '__test__' before EXPLAIN
    so the DB sees a syntactically valid query rather than bare :param tokens.
    """
    tx = make_tx(db=orders_db)
    await tx.transform_create(
        "parameterized",
        sql="SELECT * FROM orders WHERE status = :status AND customer_id = :cid",
    )

    # If dummy substitution is broken, SQLite will see `:status` as an unbound
    # parameter and raise a binding error, making valid=False even for good SQL.
    result = await tx.transform_test(orders_db, "parameterized")
    assert result["valid"] is True


# ---------------------------------------------------------------------------
# Versioning and diff
# ---------------------------------------------------------------------------


async def test_version_then_diff_shows_real_sql_changes(orders_db):
    """
    Snapshot → update SQL → diff proves the actual SQL change is captured.
    """
    tx = make_tx(db=orders_db)
    await tx.transform_create(
        "evolving",
        sql="INSERT INTO order_totals SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
    )
    await tx.transform_version("evolving")  # snapshot v0

    # Update to a different SQL
    await tx.transform_create(
        "evolving",
        sql="INSERT INTO order_totals SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
    )

    diff_result = await tx.transform_diff("evolving", version_a=0, version_b="current")
    assert diff_result["success"] is True
    assert diff_result["changed"] is True
    # The diff should show COUNT → SUM
    assert "COUNT" in diff_result["diff"]
    assert "SUM" in diff_result["diff"]


async def test_no_diff_when_sql_unchanged(orders_db):
    """Snapshotting the same SQL and diffing against current shows no change."""
    tx = make_tx(db=orders_db)
    await tx.transform_create("stable", sql="SELECT * FROM orders")
    await tx.transform_version("stable")

    diff_result = await tx.transform_diff("stable", version_a=0, version_b="current")
    assert diff_result["changed"] is False


# ---------------------------------------------------------------------------
# transform_list — in-memory path after creates
# ---------------------------------------------------------------------------


async def test_transform_list_reflects_created_transformations(orders_db):
    tx = make_tx(db=orders_db)
    await tx.transform_create("t1", sql="SELECT 1")
    await tx.transform_create("t2", sql="SELECT 2")
    await tx.transform_create("t3", sql="SELECT 3")

    result = await tx.transform_list()
    assert result["success"] is True
    assert result["count"] == 3
    names = {t["name"] for t in result["transformations"]}
    assert names == {"t1", "t2", "t3"}


async def test_transform_list_run_count_appears_in_listing(orders_db):
    """run_count in transform_list reflects actual execution count."""
    tx = make_tx(db=orders_db)
    await tx.transform_create("tracked", sql="SELECT 1")
    await tx.transform_run(orders_db, "tracked")
    await tx.transform_run(orders_db, "tracked")

    result = await tx.transform_list()
    entry = next(t for t in result["transformations"] if t["name"] == "tracked")
    assert entry["run_count"] == 2
