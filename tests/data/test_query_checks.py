"""Advisory checks on a read: fan-out, repeated keys and empty-result probes."""

from __future__ import annotations

import pytest

from daita.domains.data.sql.query_checks import (
    duplicate_key_warnings,
    empty_result_probe,
    empty_result_warning,
    fan_out_warnings,
)

KEYS = {"sales.orders": [["order_id"]], "sales.order_items": [["id"]]}
JOIN = "FROM sales.orders o JOIN sales.order_items i ON i.order_id = o.order_id"


@pytest.mark.parametrize(
    ("select", "warns"),
    [
        ("SELECT sum(o.shipping_fee) {j}", True),
        ("SELECT avg(o.shipping_fee) {j}", True),
        ("SELECT count(o.shipping_fee) {j}", True),
        ("SELECT sum(i.qty) {j}", False),
        ("SELECT count(*), min(o.fee), max(o.fee) {j}", False),
        ("SELECT sum(DISTINCT o.fee) {j}", False),
        ("SELECT sum(o.fee) FROM sales.orders o", False),
        (
            (
                "SELECT sum(o.fee) FROM sales.orders o JOIN "
                "(SELECT order_id, count(*) n FROM sales.order_items GROUP BY 1) i "
                "ON i.order_id = o.order_id"
            ),
            False,
        ),
    ],
)
def test_fan_out_warning_follows_whether_the_aggregated_rows_repeat(
    select: str, warns: bool
) -> None:
    assert bool(fan_out_warnings(select.format(j=JOIN), "postgresql", KEYS)) is warns


def test_fan_out_stays_silent_when_keys_are_unknown() -> None:
    assert fan_out_warnings(f"SELECT sum(o.fee) {JOIN}", "postgresql", {}) == ()
    assert fan_out_warnings("not sql at all (", "sqlite", KEYS) == ()


def test_fan_out_matches_unqualified_table_names_to_catalog_keys() -> None:
    sql = "SELECT sum(o.fee) FROM orders o JOIN order_items i ON i.order_id = o.id"
    keys = {"main.orders": [["id"]], "main.order_items": [["item_id"]]}
    assert fan_out_warnings(sql, "sqlite", keys)


def test_duplicate_primary_keys_in_joined_rows_are_reported() -> None:
    sql = f"SELECT o.order_id, i.id {JOIN}"
    rows = [{"order_id": 1, "id": 1}, {"order_id": 1, "id": 2}]
    columns = ["order_id", "id"]
    assert duplicate_key_warnings(sql, "postgresql", KEYS, columns, rows)
    assert not duplicate_key_warnings(
        sql,
        "postgresql",
        KEYS,
        columns,
        [{"order_id": 1, "id": 1}, {"order_id": 2, "id": 2}],
    )
    assert not duplicate_key_warnings(
        f"SELECT DISTINCT o.order_id {JOIN}", "postgresql", KEYS, ["order_id"], rows
    )


def test_probe_asks_for_the_range_of_a_range_filtered_column() -> None:
    kind, sql = empty_result_probe(
        "SELECT * FROM sales.orders WHERE ordered_at >= '2027-01-01' AND status = 'x'",
        "postgresql",
    ) or ("", "")
    assert kind == "range"
    assert "MIN(ORDERED_AT)" in sql.upper() and "FROM SALES.ORDERS" in sql.upper()


def test_probe_asks_for_distinct_values_of_an_equality_filtered_column() -> None:
    kind, sql = empty_result_probe(
        "SELECT * FROM orders o WHERE o.status IN ('a', 'b')", "sqlite"
    ) or ("", "")
    assert kind == "values"
    assert "LIMIT 10" in sql


def test_probe_is_none_without_a_literal_filter() -> None:
    assert empty_result_probe("SELECT * FROM orders WHERE a > b", "sqlite") is None
    assert empty_result_probe("SELECT * FROM orders", "sqlite") is None


def test_range_warning_reads_the_bounds_by_name_not_by_row_order() -> None:
    kind, sql = empty_result_probe(
        "SELECT * FROM orders WHERE ordered_at > '2099-01-01'", "sqlite"
    ) or ("", "")
    # Stored rows are key-sorted, so "highest" arrives before "lowest".
    text = empty_result_warning(
        kind, sql, [{"highest": "2026-04-30", "lowest": "2026-01-01"}]
    )
    assert text and text.index("2026-01-01") < text.index("2026-04-30")
