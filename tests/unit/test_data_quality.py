"""
Unit tests for daita/plugins/data_quality.py.

All tests use mocked database plugins — no real database required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from daita.plugins.data_quality import (
    DataQualityPlugin,
    data_quality,
    _validate_identifier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_db(rows_map: Dict[str, List[Any]] = None, dialect: str = "postgresql"):
    """Create a mock db plugin returning pre-configured rows for each SQL prefix."""
    rows_map = rows_map or {}
    db = MagicMock()
    db.sql_dialect = dialect

    async def query(sql, params=None):
        for prefix, rows in rows_map.items():
            if prefix.lower() in sql.lower():
                return rows
        return []

    db.query = query
    return db


def make_dq(db=None, backend=None):
    plugin = DataQualityPlugin(db=db, backend=backend)
    plugin._agent_id = "test-agent"
    return plugin


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def test_factory_function_returns_plugin():
    assert isinstance(data_quality(), DataQualityPlugin)


def test_factory_function_passes_db():
    db = MagicMock()
    assert data_quality(db=db)._db is db


def test_factory_function_passes_thresholds():
    plugin = data_quality(thresholds={"z_score": 2.5})
    assert plugin._thresholds["z_score"] == 2.5


# ---------------------------------------------------------------------------
# _validate_identifier
# ---------------------------------------------------------------------------


def test_validate_identifier_valid():
    assert _validate_identifier("orders") == "orders"
    assert _validate_identifier("schema.table") == "schema.table"
    assert _validate_identifier("col_1") == "col_1"


def test_validate_identifier_invalid():
    with pytest.raises(ValueError):
        _validate_identifier("'; DROP TABLE users; --")
    with pytest.raises(ValueError):
        _validate_identifier("col name")


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


def test_initialize_sets_agent_id():
    plugin = DataQualityPlugin()
    with patch("daita.core.graph.backend.auto_select_backend") as mock_backend:
        mock_backend.return_value = MagicMock()
        plugin.initialize("agent-123")
    assert plugin._agent_id == "agent-123"


def test_initialize_auto_selects_backend():
    plugin = DataQualityPlugin()
    mock_be = MagicMock()
    with patch("daita.core.graph.backend.auto_select_backend", return_value=mock_be):
        plugin.initialize("agent-123")
    assert plugin._graph_backend is mock_be


def test_initialize_skips_backend_if_provided():
    custom_backend = MagicMock()
    plugin = DataQualityPlugin(backend=custom_backend)
    with patch("daita.core.graph.backend.auto_select_backend") as mock_select:
        plugin.initialize("agent-123")
    mock_select.assert_not_called()
    assert plugin._graph_backend is custom_backend


# ---------------------------------------------------------------------------
# _validate_db
# ---------------------------------------------------------------------------


def test_validate_db_raises_when_none():
    plugin = make_dq()
    with pytest.raises(ValueError, match="No database plugin configured"):
        plugin._validate_db()


def test_validate_db_returns_db():
    db = MagicMock()
    plugin = make_dq(db=db)
    assert plugin._validate_db() is db


# ---------------------------------------------------------------------------
# get_tools — now 4 tools (profile, detect_anomaly, check_freshness, report)
# ---------------------------------------------------------------------------


def test_get_tools_returns_four_tools():
    plugin = make_dq()
    assert len(plugin.get_tools()) == 4


def test_get_tools_names():
    plugin = make_dq()
    names = {t.name for t in plugin.get_tools()}
    assert names == {
        "dq_profile",
        "dq_detect_anomaly",
        "dq_check_freshness",
        "dq_report",
    }


def test_all_tools_have_handlers():
    plugin = make_dq()
    for tool in plugin.get_tools():
        assert callable(tool.handler)


# ---------------------------------------------------------------------------
# profile — DQ-09: 2 queries per column, DQ-01 sample fix, DQ-03/DQ-06
# ---------------------------------------------------------------------------


async def test_profile_returns_stats():
    async def mock_query(sql, params=None):
        sql_l = sql.lower()
        if "pragma_table_info" in sql_l or "information_schema" in sql_l:
            return [{"column_name": "amount"}]
        # Combined count query
        if "count(*) as total" in sql_l:
            return [{"total": 105, "non_null": 100, "distinct_count": 80}]
        # Stats query
        if "min(" in sql_l:
            return [{"min_val": 1.0, "max_val": 999.0, "avg_val": 50.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query
    plugin = make_dq(db=db)
    result = await plugin.profile(db, "orders", columns=["amount"])

    assert result["success"] is True
    assert "amount" in result["profile"]
    stats = result["profile"]["amount"]
    assert stats["total_rows"] == 105
    assert stats["non_null_count"] == 100
    assert stats["null_count"] == 5
    assert stats["null_rate"] == round(5 / 105, 4)
    assert stats["distinct_count"] == 80
    assert stats["min"] == 1.0
    assert stats["avg"] == 50.0


async def test_profile_sample_size_uses_subquery():
    """sample_size must be in a subquery, not applied to the outer COUNT (DQ-01)."""
    captured_sqls = []

    async def mock_query(sql, params=None):
        captured_sqls.append(sql)
        if "count(*) as total" in sql.lower():
            return [{"total": 50, "non_null": 50, "distinct_count": 30}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 10, "avg_val": 5.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query
    plugin = make_dq(db=db)
    await plugin.profile(db, "orders", columns=["amount"], sample_size=100)

    count_sqls = [s for s in captured_sqls if "count(*) as total" in s.lower()]
    assert count_sqls, "Expected a count query"
    # LIMIT must appear inside a subquery, not on the outer SELECT
    count_sql = count_sqls[0]
    assert "limit" in count_sql.lower()
    limit_pos = count_sql.lower().index("limit")
    from_pos = count_sql.lower().index("from")
    assert (
        limit_pos > from_pos
    ), "LIMIT must be inside the FROM subquery, not the outer query"


async def test_profile_no_dead_sample_clause():
    """The dead `sample_clause = TABLESAMPLE...` variable must not appear (DQ-03)."""
    import inspect
    from daita.plugins import data_quality as dq_module

    source = inspect.getsource(dq_module)
    assert "TABLESAMPLE" not in source


async def test_profile_stats_failure_logged_not_swallowed(caplog):
    """Non-numeric column stat failure must log at DEBUG, not silently pass (DQ-06)."""
    import logging

    async def mock_query(sql, params=None):
        if "count(*) as total" in sql.lower():
            return [{"total": 10, "non_null": 10, "distinct_count": 5}]
        if "min(" in sql.lower():
            raise Exception("cannot cast text to float")
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query
    plugin = make_dq(db=db)

    with caplog.at_level(logging.DEBUG, logger="daita.plugins.data_quality"):
        result = await plugin.profile(db, "orders", columns=["name"])

    assert result["success"] is True
    assert result["profile"]["name"]["min"] is None
    assert result["profile"]["name"]["avg"] is None
    assert any("skipping numeric stats" in r.message for r in caplog.records)


async def test_profile_discovers_columns_postgresql():
    async def mock_query(sql, params=None):
        if "information_schema" in sql.lower():
            return [{"column_name": "id"}, {"column_name": "name"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 10, "non_null": 10, "distinct_count": 10}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 9, "avg_val": 5.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query
    plugin = make_dq(db=db)
    result = await plugin.profile(db, "users")
    assert "id" in result["profile"]
    assert "name" in result["profile"]


async def test_profile_discovers_columns_sqlite():
    async def mock_query(sql, params=None):
        if "pragma_table_info" in sql.lower():
            return [{"column_name": "id"}, {"column_name": "email"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 5, "non_null": 5, "distinct_count": 5}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 5, "avg_val": 3.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "sqlite"
    db.query = mock_query
    plugin = make_dq(db=db)
    result = await plugin.profile(db, "users")
    assert "id" in result["profile"]
    assert "email" in result["profile"]


# ---------------------------------------------------------------------------
# detect_anomaly
# ---------------------------------------------------------------------------


async def test_detect_anomaly_insufficient_data():
    db = MagicMock()
    db.query = AsyncMock(return_value=[{"amount": 1.0}, {"amount": 2.0}])
    plugin = make_dq(db=db)
    result = await plugin.detect_anomaly(db, "orders", "amount")
    assert result["success"] is True
    assert "Insufficient data" in result["note"]


async def test_detect_anomaly_zscore_detects_outlier():
    values = [0.0] * 99 + [1000.0]
    rows = [{"amount": v} for v in values]
    db = MagicMock()
    db.query = AsyncMock(return_value=rows)
    plugin = make_dq(db=db)
    result = await plugin.detect_anomaly(db, "orders", "amount", method="zscore")
    assert result["success"] is True
    assert result["anomaly_count"] >= 1
    assert 1000.0 in result["anomaly_values"]


async def test_detect_anomaly_iqr():
    values = [10.0] * 95 + [500.0, 600.0, 700.0, 800.0, 900.0]
    rows = [{"val": v} for v in values]
    db = MagicMock()
    db.query = AsyncMock(return_value=rows)
    plugin = make_dq(db=db)
    result = await plugin.detect_anomaly(db, "t", "val", method="iqr")
    assert result["success"] is True
    assert result["anomaly_count"] >= 1


# ---------------------------------------------------------------------------
# check_freshness — DQ-05: string timestamp handling
# ---------------------------------------------------------------------------


async def test_check_freshness_fresh():
    from datetime import datetime, timezone, timedelta

    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    db = MagicMock()
    db.query = AsyncMock(return_value=[{"latest": recent}])
    plugin = make_dq(db=db)
    result = await plugin.check_freshness(
        db, "events", "created_at", expected_interval_hours=24
    )
    assert result["is_fresh"] is True
    assert result["age_hours"] < 24


async def test_check_freshness_stale():
    from datetime import datetime, timezone, timedelta

    old = datetime.now(timezone.utc) - timedelta(hours=48)
    db = MagicMock()
    db.query = AsyncMock(return_value=[{"latest": old}])
    plugin = make_dq(db=db)
    result = await plugin.check_freshness(
        db, "events", "created_at", expected_interval_hours=24
    )
    assert result["is_fresh"] is False


async def test_check_freshness_null_timestamp():
    db = MagicMock()
    db.query = AsyncMock(return_value=[{"latest": None}])
    plugin = make_dq(db=db)
    result = await plugin.check_freshness(db, "events", "created_at")
    assert result["is_fresh"] is False
    assert "NULL" in result["detail"]


async def test_check_freshness_string_timestamp(monkeypatch):
    """DB adapters that return timestamps as ISO strings must not crash (DQ-05)."""
    from datetime import datetime, timezone, timedelta

    recent_str = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    db = MagicMock()
    db.query = AsyncMock(return_value=[{"latest": recent_str}])
    plugin = make_dq(db=db)
    result = await plugin.check_freshness(
        db, "events", "created_at", expected_interval_hours=24
    )
    assert result["success"] is True
    assert result["is_fresh"] is True


async def test_check_freshness_string_timestamp_Z_suffix():
    """ISO string with Z suffix (e.g. aiosqlite) must be parsed correctly."""
    from datetime import datetime, timezone, timedelta

    recent_str = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    ) + "Z"
    db = MagicMock()
    db.query = AsyncMock(return_value=[{"latest": recent_str}])
    plugin = make_dq(db=db)
    result = await plugin.check_freshness(
        db, "events", "ts", expected_interval_hours=24
    )
    assert result["success"] is True
    assert result["is_fresh"] is True


# ---------------------------------------------------------------------------
# report — DQ-12: stable metric node ID, removed assertions param
# ---------------------------------------------------------------------------


async def test_report_returns_completeness_score():
    async def mock_query(sql, params=None):
        if "information_schema" in sql.lower() or "pragma_table_info" in sql.lower():
            return [{"column_name": "id"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 100, "non_null": 90, "distinct_count": 80}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 100, "avg_val": 50.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query
    plugin = make_dq(db=db)
    result = await plugin.report(db, "orders")
    assert result["success"] is True
    assert "completeness_score" in result
    assert 0.0 <= result["completeness_score"] <= 1.0


async def test_report_has_no_contract_field():
    """report() no longer accepts assertions — contract key must not appear."""

    async def mock_query(sql, params=None):
        if "information_schema" in sql.lower() or "pragma_table_info" in sql.lower():
            return [{"column_name": "id"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 10, "non_null": 10, "distinct_count": 5}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 10, "avg_val": 5.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query
    plugin = make_dq(db=db)
    result = await plugin.report(db, "orders")
    assert "contract" not in result


async def test_report_persists_stable_metric_node_id():
    """Metric node ID must be quality_latest:{table}, not timestamp-based (DQ-12)."""

    async def mock_query(sql, params=None):
        if "information_schema" in sql.lower() or "pragma_table_info" in sql.lower():
            return [{"column_name": "id"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 100, "non_null": 100, "distinct_count": 50}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 100, "avg_val": 50.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query

    backend = MagicMock()
    backend.add_node = AsyncMock()
    backend.add_edge = AsyncMock()
    backend.flush = AsyncMock()

    plugin = make_dq(db=db, backend=backend)
    result = await plugin.report(db, "orders")

    assert result["graph_persisted"] is True
    # Node ID must contain quality_latest:orders, not a timestamp
    metric_node_id = result["metric_node_id"]
    assert "quality_latest:orders" in metric_node_id
    assert "T" not in metric_node_id.split("quality_latest")[-1]  # no ISO timestamp


async def test_report_metric_node_id_is_stable_across_calls():
    """Two calls for the same table must produce the same metric node ID."""

    async def mock_query(sql, params=None):
        if "information_schema" in sql.lower() or "pragma_table_info" in sql.lower():
            return [{"column_name": "id"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 50, "non_null": 50, "distinct_count": 20}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 50, "avg_val": 25.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query

    backend = MagicMock()
    backend.add_node = AsyncMock()
    backend.add_edge = AsyncMock()
    backend.flush = AsyncMock()

    plugin = make_dq(db=db, backend=backend)
    r1 = await plugin.report(db, "orders")
    r2 = await plugin.report(db, "orders")
    assert r1["metric_node_id"] == r2["metric_node_id"]


async def test_report_graph_failure_does_not_raise():
    async def mock_query(sql, params=None):
        if "information_schema" in sql.lower() or "pragma_table_info" in sql.lower():
            return [{"column_name": "id"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 10, "non_null": 10, "distinct_count": 5}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 10, "avg_val": 5.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query

    backend = MagicMock()
    backend.add_node = AsyncMock(side_effect=Exception("graph unavailable"))
    backend.add_edge = AsyncMock()

    plugin = make_dq(db=db, backend=backend)
    result = await plugin.report(db, "orders")
    assert result["success"] is True
    assert result["graph_persisted"] is False


# ---------------------------------------------------------------------------
# _extract_count — DQ-07: safe last-resort path
# ---------------------------------------------------------------------------


def test_extract_count_from_cnt_key():
    plugin = make_dq()
    assert plugin._extract_count([{"cnt": 42}]) == 42


def test_extract_count_from_count_key():
    plugin = make_dq()
    assert plugin._extract_count([{"count": 7}]) == 7


def test_extract_count_none_value_returns_zero():
    plugin = make_dq()
    assert plugin._extract_count([{"cnt": None}]) == 0


def test_extract_count_empty_rows_returns_zero():
    plugin = make_dq()
    assert plugin._extract_count([]) == 0


def test_extract_count_non_numeric_first_value_returns_zero():
    """Last-resort path must not raise on non-numeric values (DQ-07)."""
    plugin = make_dq()
    assert plugin._extract_count([{"unexpected_col": "not_a_number"}]) == 0


def test_extract_count_tuple_row():
    plugin = make_dq()
    assert plugin._extract_count([(99,)]) == 99


def test_extract_count_tuple_row_none():
    plugin = make_dq()
    assert plugin._extract_count([(None,)]) == 0


# ---------------------------------------------------------------------------
# Tool handler dispatches (smoke tests)
# ---------------------------------------------------------------------------


async def test_tool_profile_dispatches():
    async def mock_query(sql, params=None):
        if "information_schema" in sql.lower() or "pragma_table_info" in sql.lower():
            return [{"column_name": "amount"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 50, "non_null": 45, "distinct_count": 40}]
        if "min(" in sql.lower():
            return [{"min_val": 0, "max_val": 100, "avg_val": 50.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query
    plugin = make_dq(db=db)
    result = await plugin._tool_profile({"table": "orders"})
    assert result["success"] is True


async def test_tool_check_freshness_no_db_raises():
    plugin = make_dq()
    with pytest.raises(ValueError):
        await plugin._tool_check_freshness(
            {"table": "events", "timestamp_column": "ts", "expected_interval_hours": 24}
        )


async def test_tool_report_dispatches():
    async def mock_query(sql, params=None):
        if "information_schema" in sql.lower() or "pragma_table_info" in sql.lower():
            return [{"column_name": "id"}]
        if "count(*) as total" in sql.lower():
            return [{"total": 10, "non_null": 10, "distinct_count": 5}]
        if "min(" in sql.lower():
            return [{"min_val": 1, "max_val": 10, "avg_val": 5.0}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = mock_query
    plugin = make_dq(db=db)
    result = await plugin._tool_report({"table": "orders"})
    assert result["success"] is True
