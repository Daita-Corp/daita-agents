"""
Unit tests for the SQL focus compiler.

Tests compile_focus_to_sql() as a pure function — no database connection required.
Also verifies that evaluate_remaining() correctly skips clauses in the applied set,
confirming the compiler output integrates cleanly with the Python fallback.
"""

import pytest

from daita.core.focus import parse
from daita.core.focus.backends.sql import compile_focus_to_sql
from daita.core.focus.evaluator import evaluate_remaining

# ── Helpers ───────────────────────────────────────────────────────────────────

BASE = "SELECT * FROM orders"


def compile(dsl: str, dialect: str = "postgresql", param_offset: int = 0, mode: str = "safe"):
    fq = parse(dsl)
    return compile_focus_to_sql(BASE, fq, dialect=dialect, param_offset=param_offset, mode=mode)


# ── Filter pushdown ───────────────────────────────────────────────────────────


def test_filter_equality_postgresql():
    sql, params, applied = compile("status == 'completed'")
    assert "WHERE" in sql
    assert "status" in sql
    assert "$1" in sql
    assert params == ["completed"]
    assert "filter" in applied


def test_filter_equality_mysql():
    sql, params, applied = compile("status == 'completed'", dialect="mysql")
    assert "%s" in sql
    assert params == ["completed"]
    assert "filter" in applied


def test_filter_equality_sqlite():
    sql, params, applied = compile("status == 'completed'", dialect="sqlite")
    assert "?" in sql
    assert params == ["completed"]
    assert "filter" in applied


def test_filter_numeric_comparison():
    sql, params, applied = compile("amount > 100")
    assert ">" in sql
    assert params == [100]
    assert "filter" in applied


def test_filter_and():
    sql, params, applied = compile("status == 'active' and amount > 50")
    assert "AND" in sql
    assert len(params) == 2
    assert "filter" in applied


def test_filter_or():
    sql, params, applied = compile("region == 'US' or region == 'EU'")
    assert "OR" in sql
    assert "filter" in applied


def test_filter_not():
    sql, params, applied = compile("not status == 'cancelled'")
    assert "NOT" in sql
    assert "filter" in applied


def test_filter_in_list():
    sql, params, applied = compile("status in ['active', 'pending']")
    assert "IN" in sql
    assert "active" in params
    assert "pending" in params
    assert "filter" in applied


def test_filter_not_in_list():
    sql, params, applied = compile("status not in ['cancelled', 'deleted']")
    assert "NOT IN" in sql
    assert "filter" in applied


def test_filter_none_eq_uses_is_null():
    """== None must compile to IS NULL, not = NULL (which is always FALSE in SQL)."""
    sql, params, applied = compile("status == None")
    assert "IS NULL" in sql, "Expected IS NULL for == None"
    assert "= NULL" not in sql, "= NULL is always FALSE in SQL — must not be emitted"
    assert params == []
    assert "filter" in applied


def test_filter_none_neq_uses_is_not_null():
    """!= None must compile to IS NOT NULL."""
    sql, params, applied = compile("status != None")
    assert "IS NOT NULL" in sql, "Expected IS NOT NULL for != None"
    assert "!= NULL" not in sql
    assert params == []
    assert "filter" in applied


def test_filter_none_comparison_not_pushed():
    """Comparisons other than == / != against None can't be expressed in SQL."""
    sql, params, applied = compile("amount > None")
    assert "filter" not in applied  # falls back to Python


def test_filter_dot_notation_not_pushed():
    """Dot-notation (nested field) cannot be pushed to SQL — filter stays in Python."""
    sql, params, applied = compile("order.status == 'paid'")
    assert "filter" not in applied
    assert params == []
    # SQL should still be a valid subquery (just with SELECT *)
    assert "_focus_q" in sql


# ── SELECT pushdown ───────────────────────────────────────────────────────────


def test_select_projection():
    # SELECT pushdown only happens in full mode (developer-controlled schema)
    sql, params, applied = compile("SELECT id, amount", mode="full")
    assert '"id"' in sql
    assert '"amount"' in sql
    assert "select" in applied
    assert params == []


def test_select_not_pushed_in_safe_mode():
    # In safe mode (LLM-generated SQL), SELECT stays in Python to avoid column mismatch
    sql, params, applied = compile("SELECT id, amount", mode="safe")
    assert "select" not in applied
    assert '"id"' not in sql  # outer SELECT is just *


def test_select_mysql_backtick_quoting():
    sql, params, applied = compile("SELECT id, amount", dialect="mysql", mode="full")
    assert "`id`" in sql
    assert "`amount`" in sql
    assert "select" in applied


# ── ORDER BY pushdown ─────────────────────────────────────────────────────────


def test_order_by_asc():
    sql, params, applied = compile("ORDER BY amount", mode="full")
    assert "ORDER BY" in sql
    assert '"amount"' in sql
    assert "ASC" in sql
    assert "order_by" in applied


def test_order_by_desc():
    sql, params, applied = compile("ORDER BY amount DESC", mode="full")
    assert "DESC" in sql
    assert "order_by" in applied


def test_order_by_not_pushed_in_safe_mode():
    sql, params, applied = compile("ORDER BY amount DESC", mode="safe")
    assert "order_by" not in applied
    assert "ORDER BY" not in sql


# ── LIMIT pushdown ────────────────────────────────────────────────────────────


def test_limit():
    sql, params, applied = compile("LIMIT 25")
    assert "LIMIT 25" in sql
    assert "limit" in applied


# ── Combined clauses ──────────────────────────────────────────────────────────


def test_combined_filter_select_limit():
    # full mode: all three clauses pushed to SQL
    sql, params, applied = compile(
        "status == 'completed' | SELECT id, amount | LIMIT 100", mode="full"
    )
    assert "WHERE" in sql
    assert '"id"' in sql
    assert "LIMIT 100" in sql
    assert {"filter", "select", "limit"} <= applied
    assert params == ["completed"]


def test_combined_filter_limit_safe_mode():
    # safe mode: WHERE and LIMIT pushed; SELECT left for Python
    sql, params, applied = compile(
        "status == 'completed' | SELECT id, amount | LIMIT 100", mode="safe"
    )
    assert "WHERE" in sql
    assert "LIMIT 100" in sql
    assert "filter" in applied
    assert "limit" in applied
    assert "select" not in applied


def test_clause_order_in_sql():
    """WHERE must precede ORDER BY, LIMIT (full mode)."""
    sql, params, applied = compile("amount > 0 | ORDER BY amount DESC | LIMIT 10", mode="full")
    where_pos = sql.index("WHERE")
    order_pos = sql.index("ORDER BY")
    limit_pos = sql.index("LIMIT")
    assert where_pos < order_pos < limit_pos


# ── GROUP BY + aggregates pushdown ────────────────────────────────────────────


def test_group_by_with_aggregates():
    sql, params, applied = compile(
        "GROUP BY region | SELECT region, SUM(revenue) AS total", mode="full"
    )
    assert "GROUP BY" in sql
    assert "SUM(" in sql
    assert '"total"' in sql
    assert "group_by" in applied
    assert "aggregates" in applied


def test_group_by_count_star():
    sql, params, applied = compile("GROUP BY status | SELECT status, COUNT(*) AS cnt", mode="full")
    assert "COUNT(*)" in sql
    assert "group_by" in applied


def test_group_by_not_pushed_in_safe_mode():
    sql, params, applied = compile(
        "GROUP BY region | SELECT region, SUM(revenue) AS total", mode="safe"
    )
    assert "group_by" not in applied
    assert "GROUP BY" not in sql


# ── Param offset (PostgreSQL numbered params) ─────────────────────────────────


def test_param_offset_postgresql():
    """Existing $1,$2 in base query → focus params start at $3."""
    sql, params, applied = compile(
        "status == 'active'", dialect="postgresql", param_offset=2
    )
    assert "$3" in sql
    assert "$1" not in sql
    assert params == ["active"]


def test_param_offset_mysql_unaffected():
    """MySQL uses %s regardless of offset — offset has no effect on placeholder style."""
    sql, params, applied = compile(
        "status == 'active'", dialect="mysql", param_offset=5
    )
    assert "%s" in sql
    assert params == ["active"]


# ── Subquery structure ────────────────────────────────────────────────────────


def test_base_query_is_wrapped():
    sql, _, _ = compile("LIMIT 10")
    assert f"({BASE})" in sql
    assert "_focus_q" in sql


# ── evaluate_remaining integration ───────────────────────────────────────────


def test_applied_clauses_skipped_by_evaluator():
    """Clauses in applied are not re-applied by evaluate_remaining."""
    rows = [
        {"status": "active", "amount": 200},
        {"status": "inactive", "amount": 100},
    ]
    fq = parse("status == 'active' | LIMIT 5")
    # Simulate: filter was pushed to SQL, only 'active' row came back
    # But tell evaluate_remaining that filter is already applied
    # (it should NOT re-filter and should not drop any rows)
    result = evaluate_remaining(rows, fq, applied={"filter"})
    # limit still applies (not in applied) — both rows pass since filter was "already done"
    assert len(result) == 2


def test_no_applied_clauses_python_handles_all():
    """With empty applied set, evaluate_remaining processes everything."""
    rows = [
        {"status": "active", "amount": 200},
        {"status": "inactive", "amount": 100},
    ]
    fq = parse("status == 'active' | LIMIT 5")
    result = evaluate_remaining(rows, fq, applied=set())
    assert len(result) == 1
    assert result[0]["status"] == "active"
