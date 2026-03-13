"""
Integration tests for DataQualityPlugin and query_checked() against real SQLite.

These tests verify that profiling stats, freshness checks, anomaly detection,
and assertion enforcement produce correct results on actual data — not just that
the right SQL strings were constructed.

Run with:
    pytest tests/integration/test_data_quality_integration.py -v
"""

import pytest
from datetime import datetime, timedelta, timezone

from daita.plugins.data_quality import DataQualityPlugin
from daita.core.assertions import ItemAssertion
from daita.core.exceptions import DataQualityError


# ---------------------------------------------------------------------------
# Shared fixture: table with known quality characteristics
# ---------------------------------------------------------------------------


@pytest.fixture
async def quality_db(db):
    """
    Database with a 'sales' table that has deliberate quality issues:
      - 2 NULL amounts (out of 10 rows) → null_rate = 0.2
      - amounts: 10, 20, 30, 40, 50, 60, 70, 80, NULL, NULL
      - one extreme outlier (9999) added by individual tests that need it
      - two rows with negative amounts for assertion violation tests
    """
    await db.execute_script("""
        CREATE TABLE sales (
            id         INTEGER PRIMARY KEY,
            product    TEXT NOT NULL,
            amount     REAL,
            region     TEXT,
            updated_at TEXT
        );
    """)
    now_str = datetime.now(timezone.utc).isoformat()
    await db.insert_many("sales", [
        {"id": 1,  "product": "A", "amount": 10.0,  "region": "east", "updated_at": now_str},
        {"id": 2,  "product": "B", "amount": 20.0,  "region": "west", "updated_at": now_str},
        {"id": 3,  "product": "A", "amount": 30.0,  "region": "east", "updated_at": now_str},
        {"id": 4,  "product": "C", "amount": 40.0,  "region": "west", "updated_at": now_str},
        {"id": 5,  "product": "B", "amount": 50.0,  "region": "east", "updated_at": now_str},
        {"id": 6,  "product": "A", "amount": 60.0,  "region": "west", "updated_at": now_str},
        {"id": 7,  "product": "C", "amount": 70.0,  "region": "east", "updated_at": now_str},
        {"id": 8,  "product": "B", "amount": 80.0,  "region": "west", "updated_at": now_str},
        {"id": 9,  "product": "A", "amount": None,  "region": "east", "updated_at": now_str},
        {"id": 10, "product": "C", "amount": None,  "region": None,   "updated_at": now_str},
    ])
    return db


# ---------------------------------------------------------------------------
# query_checked — enforcement at query time
# ---------------------------------------------------------------------------


async def test_query_checked_passes_silently_on_clean_data(quality_db):
    """
    query_checked returns all rows unchanged when every row satisfies
    every assertion.
    """
    rows = await quality_db.query_checked(
        "SELECT * FROM sales WHERE amount IS NOT NULL",
        assertions=[
            ItemAssertion(lambda r: r["amount"] > 0, "amount must be positive"),
        ],
    )
    assert len(rows) == 8


async def test_query_checked_raises_data_quality_error_on_violation(quality_db):
    """A single failing row triggers DataQualityError."""
    await quality_db.execute(
        "INSERT INTO sales VALUES (99, 'X', -5.0, 'north', '2024-01-01T00:00:00')"
    )

    with pytest.raises(DataQualityError):
        await quality_db.query_checked(
            "SELECT * FROM sales WHERE amount IS NOT NULL",
            assertions=[
                ItemAssertion(lambda r: r["amount"] > 0, "amount must be positive"),
            ],
        )


async def test_query_checked_violation_count_is_exact(quality_db):
    """violation_count in the error matches the actual number of bad rows."""
    # Insert 3 negative rows
    for i in range(3):
        await quality_db.execute(
            f"INSERT INTO sales VALUES ({200 + i}, 'Z', {-10.0 * (i + 1)}, 'x', '2024-01-01T00:00:00')"
        )

    with pytest.raises(DataQualityError) as exc_info:
        await quality_db.query_checked(
            "SELECT * FROM sales WHERE amount IS NOT NULL",
            assertions=[
                ItemAssertion(lambda r: r["amount"] > 0, "amount must be positive"),
            ],
        )
    violation = exc_info.value.violations[0]
    assert violation["violation_count"] == 3


async def test_query_checked_sample_contains_bad_rows(quality_db):
    """The sample in the violation holds the actual offending rows."""
    bad_amount = -42.0
    await quality_db.execute(
        f"INSERT INTO sales VALUES (300, 'BAD', {bad_amount}, 'x', '2024-01-01T00:00:00')"
    )

    with pytest.raises(DataQualityError) as exc_info:
        await quality_db.query_checked(
            "SELECT * FROM sales WHERE amount IS NOT NULL",
            assertions=[
                ItemAssertion(lambda r: r["amount"] > 0, "amount must be positive"),
            ],
        )
    sample = exc_info.value.violations[0]["sample"]
    assert any(r["amount"] == bad_amount for r in sample)


async def test_query_checked_multiple_assertions_all_violations_collected(quality_db):
    """Both failing assertions are reported, not just the first one."""
    await quality_db.execute(
        "INSERT INTO sales VALUES (400, 'M', -1.0, NULL, '2024-01-01T00:00:00')"
    )

    with pytest.raises(DataQualityError) as exc_info:
        await quality_db.query_checked(
            "SELECT * FROM sales WHERE amount IS NOT NULL",
            assertions=[
                ItemAssertion(lambda r: r["amount"] > 0, "amount positive"),
                ItemAssertion(lambda r: r["region"] is not None, "region not null"),
            ],
        )
    assert len(exc_info.value.violations) == 2


# ---------------------------------------------------------------------------
# profile — statistical accuracy
# ---------------------------------------------------------------------------


async def test_profile_null_count_matches_actual_nulls(quality_db):
    """null_count for 'amount' is exactly 2 (rows 9 and 10 have NULL)."""
    dq = DataQualityPlugin()
    result = await dq.profile(quality_db, "sales", columns=["amount"])

    assert result["success"] is True
    amount_stats = result["profile"]["amount"]
    assert amount_stats["null_count"] == 2
    assert amount_stats["total_rows"] == 10
    assert amount_stats["null_rate"] == pytest.approx(0.2)


async def test_profile_non_null_count_is_correct(quality_db):
    """non_null_count is total_rows minus null_count."""
    dq = DataQualityPlugin()
    result = await dq.profile(quality_db, "sales", columns=["amount"])

    stats = result["profile"]["amount"]
    assert stats["non_null_count"] == stats["total_rows"] - stats["null_count"]


async def test_profile_numeric_min_max_avg_are_correct(quality_db):
    """min, max, and avg are computed over non-null values only."""
    dq = DataQualityPlugin()
    result = await dq.profile(quality_db, "sales", columns=["amount"])

    stats = result["profile"]["amount"]
    # Non-null amounts: 10, 20, 30, 40, 50, 60, 70, 80
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(80.0)
    assert stats["avg"] == pytest.approx(45.0)  # (10+20+...+80)/8


async def test_profile_distinct_count_is_correct(quality_db):
    """distinct_count for 'product' matches the number of unique values."""
    dq = DataQualityPlugin()
    result = await dq.profile(quality_db, "sales", columns=["product"])

    stats = result["profile"]["product"]
    assert stats["distinct_count"] == 3  # A, B, C


async def test_profile_discovers_all_columns_when_unspecified(quality_db):
    """Calling profile without specifying columns discovers all 5."""
    dq = DataQualityPlugin()
    result = await dq.profile(quality_db, "sales")

    assert result["success"] is True
    assert result["columns_profiled"] == 5
    assert set(result["profile"].keys()) == {"id", "product", "amount", "region", "updated_at"}


async def test_profile_sample_size_limits_scan(quality_db):
    """sample_size caps total_rows in the profile to the requested limit."""
    dq = DataQualityPlugin()
    result = await dq.profile(quality_db, "sales", columns=["amount"], sample_size=5)

    assert result["success"] is True
    assert result["profile"]["amount"]["total_rows"] <= 5


# ---------------------------------------------------------------------------
# check_freshness
# ---------------------------------------------------------------------------


async def test_check_freshness_returns_fresh_for_recent_data(quality_db):
    """is_fresh is True when the most recent timestamp is within the window."""
    dq = DataQualityPlugin()
    result = await dq.check_freshness(
        quality_db, "sales", "updated_at", expected_interval_hours=24
    )

    assert result["success"] is True
    assert result["is_fresh"] is True
    assert result["age_hours"] < 1.0


async def test_check_freshness_returns_stale_for_old_data(db):
    """is_fresh is False when all timestamps are older than the window."""
    await db.execute_script(
        "CREATE TABLE stale_events (id INTEGER, ts TEXT);"
    )
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    await db.execute(f"INSERT INTO stale_events VALUES (1, '{old_ts}')")

    dq = DataQualityPlugin()
    result = await dq.check_freshness(
        db, "stale_events", "ts", expected_interval_hours=24
    )

    assert result["success"] is True
    assert result["is_fresh"] is False
    assert result["age_hours"] > 24.0


async def test_check_freshness_empty_table(db):
    """is_fresh is False and detail explains when the table is empty."""
    await db.execute_script("CREATE TABLE empty_events (id INTEGER, ts TEXT);")

    dq = DataQualityPlugin()
    result = await dq.check_freshness(db, "empty_events", "ts")

    assert result["success"] is True
    assert result["is_fresh"] is False
    assert result["latest_timestamp"] is None


async def test_check_freshness_custom_interval(db):
    """expected_interval_hours is respected — 1-hour-old data fails a 30-min window."""
    await db.execute_script("CREATE TABLE recent_events (id INTEGER, ts TEXT);")
    one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await db.execute(f"INSERT INTO recent_events VALUES (1, '{one_hour_ago}')")

    dq = DataQualityPlugin()
    result = await dq.check_freshness(
        db, "recent_events", "ts", expected_interval_hours=0.5  # 30-min window
    )

    assert result["is_fresh"] is False


# ---------------------------------------------------------------------------
# detect_anomaly
# ---------------------------------------------------------------------------


async def test_detect_anomaly_finds_clear_outlier(db):
    """A value far outside the normal distribution is flagged as an anomaly."""
    await db.execute_script("CREATE TABLE metrics (value REAL);")
    # Normal values around 50 with one extreme outlier
    normal_values = [45, 48, 50, 51, 49, 52, 50, 47, 53, 50]
    for v in normal_values:
        await db.execute(f"INSERT INTO metrics VALUES ({v})")
    await db.execute("INSERT INTO metrics VALUES (9999)")  # clear outlier

    dq = DataQualityPlugin(thresholds={"z_score": 2.5})
    result = await dq.detect_anomaly(db, "metrics", "value", method="zscore")

    assert result["success"] is True
    assert result["anomaly_count"] >= 1
    assert 9999.0 in result["anomaly_values"]


async def test_detect_anomaly_no_anomalies_in_uniform_data(db):
    """No anomalies when all values are close to the mean."""
    await db.execute_script("CREATE TABLE uniform (value REAL);")
    for v in [10, 11, 10, 12, 10, 11, 10, 11, 12, 10]:
        await db.execute(f"INSERT INTO uniform VALUES ({v})")

    dq = DataQualityPlugin()
    result = await dq.detect_anomaly(db, "uniform", "value", method="zscore")

    assert result["success"] is True
    assert result["anomaly_count"] == 0


async def test_detect_anomaly_iqr_method(db):
    """IQR method also detects the same clear outlier."""
    await db.execute_script("CREATE TABLE iqr_data (value REAL);")
    normal = [40, 42, 45, 43, 44, 41, 46, 43, 45, 44]
    for v in normal:
        await db.execute(f"INSERT INTO iqr_data VALUES ({v})")
    await db.execute("INSERT INTO iqr_data VALUES (9999)")

    dq = DataQualityPlugin(thresholds={"iqr_multiplier": 1.5})
    result = await dq.detect_anomaly(db, "iqr_data", "value", method="iqr")

    assert result["anomaly_count"] >= 1
    assert 9999.0 in result["anomaly_values"]


async def test_detect_anomaly_insufficient_data_returns_note(db):
    """Fewer than 4 rows returns a note rather than raising."""
    await db.execute_script("CREATE TABLE tiny (value REAL);")
    await db.execute("INSERT INTO tiny VALUES (1)")
    await db.execute("INSERT INTO tiny VALUES (2)")

    dq = DataQualityPlugin()
    result = await dq.detect_anomaly(db, "tiny", "value")

    assert result["success"] is True
    assert result["anomaly_count"] == 0
    assert "Insufficient data" in result["note"]


# ---------------------------------------------------------------------------
# report — end-to-end completeness score
# ---------------------------------------------------------------------------


async def test_report_completeness_score_reflects_nulls(quality_db):
    """
    Completeness score is the average non-null rate across all columns.
    With 2/10 NULL amounts and 1/10 NULL regions, the score should be < 1.
    """
    dq = DataQualityPlugin()
    result = await dq.report(quality_db, "sales")

    assert result["success"] is True
    assert 0.0 < result["completeness_score"] < 1.0


async def test_report_completeness_score_is_1_for_complete_table(db):
    """completeness_score is 1.0 when there are no NULLs anywhere."""
    await db.execute_script("CREATE TABLE complete (id INTEGER, name TEXT, value REAL);")
    await db.insert_many("complete", [
        {"id": 1, "name": "a", "value": 1.0},
        {"id": 2, "name": "b", "value": 2.0},
    ])

    dq = DataQualityPlugin()
    result = await dq.report(db, "complete")

    assert result["completeness_score"] == pytest.approx(1.0)


async def test_report_includes_per_column_profile(quality_db):
    """report() embeds the full column profile in its output."""
    dq = DataQualityPlugin()
    result = await dq.report(quality_db, "sales")

    assert "profile" in result
    assert "amount" in result["profile"]
    assert result["profile"]["amount"]["null_count"] == 2
