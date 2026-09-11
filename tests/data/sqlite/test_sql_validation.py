import pytest

from daita.domains.data.sql import (
    MAX_SQL_CHARACTERS,
    ResourceSchema,
    analyze_sqlite_sql,
    validate_sqlite_read,
)

_REVISION = "sha256:" + "3" * 64


def _resource(
    resource_id: str,
    name: str,
    columns: tuple[str, ...],
) -> ResourceSchema:
    return ResourceSchema(
        resource_id=resource_id,
        source_id="source-sqlite",
        name=name,
        columns=columns,
        revision=_REVISION,
        source_revision="schema_version:1",
        resource_kind="table",
    )


_ORDERS = _resource(
    "resource-orders",
    "orders",
    ("id", "customer_id", "status", "total_amount", "ordered_at"),
)
_CUSTOMERS = _resource(
    "resource-customers",
    "customers",
    ("id", "name", "region"),
)
_RESOURCES = (_ORDERS, _CUSTOMERS)


def _validate(
    sql: str,
    *,
    parameters: tuple[object, ...] = (),
):
    return validate_sqlite_read(
        sql,
        source_id="source-sqlite",
        resources=_RESOURCES,
        parameters=parameters,
    )


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT o.id, o.status FROM orders o",
        "SELECT c.id FROM customers c WHERE EXISTS "
        "(SELECT 1 FROM orders o WHERE o.customer_id = c.id)",
        "WITH recent AS (SELECT o.id, o.status FROM orders o) "
        "SELECT recent.id FROM recent UNION ALL SELECT o.id FROM orders o",
        "SELECT o.status, count(*), sum(o.total_amount) FROM orders o "
        "GROUP BY o.status HAVING count(*) > 0",
    ),
)
def test_catalog_scoped_sqlite_query_shapes_are_valid(sql: str) -> None:
    result = _validate(sql)

    assert result.valid, result.issues
    assert result.resource_ids
    assert result.source_revision == "schema_version:1"


def test_cte_order_by_output_alias_wins_before_parent_scope_lookup() -> None:
    result = _validate("""
        WITH selected AS (
          SELECT c.name AS customer_id
          FROM customers AS c
          ORDER BY customer_id
        )
        SELECT selected.customer_id
        FROM orders AS o
        CROSS JOIN selected
        """)

    assert result.valid, result.issues


def test_cte_cannot_escape_to_same_named_parent_column_without_local_alias() -> None:
    result = _validate("""
        WITH selected AS (
          SELECT c.name
          FROM customers AS c
          ORDER BY customer_id
        )
        SELECT selected.name
        FROM orders AS o
        CROSS JOIN selected
        """)

    assert not result.valid
    assert "column_scope_escape" in result.issue_codes


@pytest.mark.parametrize(
    ("sql", "issue_code"),
    (
        ("SELECT 1", "resource_scope_empty"),
        ("DELETE FROM orders", "mutation_not_allowed"),
        ("SELECT o.missing FROM orders o", "missing_column"),
        ("SELECT o.id FROM absent o", "unknown_resource"),
        (
            "SELECT id FROM orders o JOIN customers c ON o.customer_id = c.id",
            "ambiguous_column",
        ),
        ("SELECT o.id FROM orders o; SELECT 1", "multiple_statements"),
        ("ATTACH DATABASE '/tmp/value.db' AS value", "mutation_not_allowed"),
    ),
)
def test_invalid_sqlite_statement_and_catalog_shapes_fail_closed(
    sql: str,
    issue_code: str,
) -> None:
    result = _validate(sql)

    assert not result.valid
    assert issue_code in result.issue_codes


def test_sqlite_parameters_must_match_placeholders() -> None:
    valid = _validate(
        "SELECT o.id FROM orders o WHERE o.id = ?",
        parameters=(1,),
    )
    invalid = _validate(
        "SELECT o.id FROM orders o WHERE o.id = ?",
        parameters=(),
    )

    assert valid.valid
    assert not invalid.valid
    assert "parameter_count_mismatch" in invalid.issue_codes


def test_sqlite_analysis_is_bounded_and_canonical() -> None:
    first = analyze_sqlite_sql("SELECT o.id FROM orders o")
    second = analyze_sqlite_sql(" SELECT o.id\nFROM orders AS o; ")
    oversized = _validate("SELECT o.id FROM orders o " + (" " * MAX_SQL_CHARACTERS))

    assert first.canonical_sql == second.canonical_sql
    assert first.sql_fingerprint == second.sql_fingerprint
    assert not oversized.valid
    assert oversized.analysis is None
    assert oversized.issue_codes == ("sql_too_large",)
