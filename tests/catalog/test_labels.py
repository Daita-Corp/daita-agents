"""Look-alike tables are grouped by name and columns; labels are only advice."""

from __future__ import annotations

from daita.catalog.labels import TableFacts, column_names, label_tables

_ORDER_COLUMNS = frozenset({"id", "customer_id", "status", "total"})


def _table(name: str, columns: set[str] | None = None, **facts: object) -> TableFacts:
    return TableFacts(
        name,
        None if columns is None else frozenset(columns),
        **facts,  # type: ignore[arg-type]
    )


def test_look_alike_names_with_shared_columns_form_one_group() -> None:
    labels = label_tables(
        [
            _table("sales.orders", set(_ORDER_COLUMNS)),
            _table("sales.orders_archive", set(_ORDER_COLUMNS)),
            _table("legacy.orders", set(_ORDER_COLUMNS)),
            _table("events.signup", {"id", "at"}),
            _table("events.signups", {"id", "at"}),
            _table("finance.revenue", {"id", "amount", "day"}),
            _table("finance.revenue_entries", {"id", "amount", "day", "note"}),
        ]
    )
    assert labels["sales.orders"].look_alikes == (
        "legacy.orders",
        "sales.orders_archive",
    )
    assert labels["events.signup"].look_alikes == ("events.signups",)
    assert labels["finance.revenue"].look_alikes == ("finance.revenue_entries",)


def test_half_shared_columns_and_extra_name_parts_do_not_group() -> None:
    labels = label_tables(
        [
            _table("crm.accounts", {"id", "name", "owner", "region"}),
            _table("crm.account_segments", {"id", "name", "segment", "weight"}),
            _table("sales.orders", set(_ORDER_COLUMNS)),
            _table("sales.order_items", {"id", "order_id", "sku", "qty"}),
        ]
    )
    assert labels == {}


def test_unknown_columns_group_only_equal_names() -> None:
    labels = label_tables(
        [_table("a.orders"), _table("b.orders_v2"), _table("a.order_items")]
    )
    assert set(labels) == {"a.orders", "b.orders_v2"}


def test_a_table_with_far_more_rows_and_references_is_canonical() -> None:
    labels = label_tables(
        [
            _table("sales.orders", set(_ORDER_COLUMNS), row_estimate=5_000),
            _table("sales.orders_archive", set(_ORDER_COLUMNS), row_estimate=12),
        ]
    )
    assert labels["sales.orders"].status == "canonical"
    stale = labels["sales.orders_archive"]
    assert stale.status == "likely_stale"
    assert stale.evidence == ("12 rows vs 5,000",)


def test_inbound_foreign_keys_decide_when_rows_are_unknown() -> None:
    labels = label_tables(
        [
            _table("a.orders", set(_ORDER_COLUMNS), inbound_foreign_keys=1),
            _table("b.orders", set(_ORDER_COLUMNS), inbound_foreign_keys=0),
        ]
    )
    assert labels["a.orders"].status == "canonical"
    assert labels["b.orders"].evidence == ("no inbound foreign keys vs 1",)


def test_nothing_clearly_ahead_leaves_every_look_alike_unknown() -> None:
    labels = label_tables(
        [
            _table("a.orders", set(_ORDER_COLUMNS), row_estimate=100),
            _table("b.orders", set(_ORDER_COLUMNS), row_estimate=150),
            _table("c.orders", set(_ORDER_COLUMNS)),
        ]
    )
    assert {label.status for label in labels.values()} == {"unknown"}


def test_a_table_ahead_on_rows_but_behind_on_references_is_not_canonical() -> None:
    labels = label_tables(
        [
            _table(
                "a.orders",
                set(_ORDER_COLUMNS),
                row_estimate=1_000,
                inbound_foreign_keys=0,
            ),
            _table(
                "b.orders",
                set(_ORDER_COLUMNS),
                row_estimate=10,
                inbound_foreign_keys=2,
            ),
        ]
    )
    assert {label.status for label in labels.values()} == {"unknown"}


def test_a_deprecated_comment_or_note_marks_a_table_even_without_look_alikes() -> None:
    labels = label_tables(
        [
            _table("a.ledger", comment="Superseded by the finance schema"),
            _table("a.payments", notes=("Do not use for reporting.",)),
            _table("a.customers"),
        ]
    )
    assert labels["a.ledger"].status == "deprecated"
    assert labels["a.ledger"].evidence == ("comment says deprecated",)
    assert labels["a.payments"].evidence == ("note says deprecated",)
    assert "a.customers" not in labels


def test_the_only_look_alike_not_deprecated_is_canonical() -> None:
    labels = label_tables(
        [
            _table("a.orders", set(_ORDER_COLUMNS)),
            _table("a.orders_old", set(_ORDER_COLUMNS), comment="obsolete"),
        ]
    )
    assert labels["a.orders"].status == "canonical"
    assert labels["a.orders_old"].status == "deprecated"


def test_column_names_come_from_a_hit_summary() -> None:
    assert column_names("id integer, price numeric(10, 2), name text (+4 more)") == {
        "id",
        "price",
        "name",
    }
    assert column_names(None) is None
