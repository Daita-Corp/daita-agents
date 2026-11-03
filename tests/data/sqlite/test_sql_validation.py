import pytest

from daita.catalog.models import ResourceKind, catalog_resource_id
from daita.domains.data.sql import (
    MAX_SQL_CHARACTERS,
    analyze_sqlite_sql,
    validate_read,
)

_SOURCE = "source-sqlite"


def _table(name: str) -> str:
    return catalog_resource_id(_SOURCE, ResourceKind.TABLE, f"main.{name}")


def _validate(
    sql: str,
    *,
    parameters: tuple[object, ...] = (),
    allowed: frozenset[str] | None = None,
):
    return validate_read(
        sql,
        dialect="sqlite",
        source_id=_SOURCE,
        parameters=parameters,
        allowed_resource_ids=allowed,
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
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) "
        "SELECT r.n FROM r, orders",
    ),
)
def test_ordinary_sqlite_query_shapes_are_valid(sql: str) -> None:
    result = _validate(sql)

    assert result.valid, result.issues
    assert result.resource_ids


@pytest.mark.parametrize(
    ("sql", "issue_code"),
    (
        ("DELETE FROM orders", "mutation_not_allowed"),
        ("INSERT INTO orders (id) VALUES (1)", "mutation_not_allowed"),
        ("UPDATE orders SET id = 1", "mutation_not_allowed"),
        ("DROP TABLE orders", "mutation_not_allowed"),
        ("CREATE TABLE copy AS SELECT * FROM orders", "mutation_not_allowed"),
        (
            "WITH d AS (DELETE FROM orders RETURNING id) SELECT id FROM d",
            "mutation_not_allowed",
        ),
        ("/* SELECT 1; */ DROP TABLE orders", "mutation_not_allowed"),
        ("SELECT o.id FROM temp.absent o", "resource_out_of_scope"),
        ("SELECT * FROM pragma_database_list", "resource_out_of_scope"),
        ("SELECT * FROM pragma_table_info('orders')", "function_not_allowed"),
        ("SELECT o.id FROM orders o; SELECT 1", "multiple_statements"),
        ("ATTACH DATABASE '/tmp/value.db' AS value", "mutation_not_allowed"),
        ("PRAGMA writable_schema = 1", "read_statement_required"),
        ("COMMIT", "read_statement_required"),
        ("SELECT id FROM orders ORDER BY 1 USING <", "sql_parse_error"),
    ),
)
def test_invalid_sqlite_statements_fail_closed_before_any_source_io(
    sql: str,
    issue_code: str,
) -> None:
    result = _validate(sql)

    assert not result.valid
    assert issue_code in result.issue_codes


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT * FROM companies",
        "SELECT * FROM Companies",
        'SELECT * FROM "COMPANIES"',
        "SELECT * FROM main.companies",
        "SELECT * FROM MAIN.Companies",
    ),
)
def test_every_spelling_of_a_table_is_one_identity(sql: str) -> None:
    result = _validate(sql, allowed=frozenset({_table("companies")}))

    assert result.valid
    assert result.resource_names == ((_table("companies"), "main.companies"),)


def test_selected_scope_refuses_other_tables_and_the_schema_table() -> None:
    allowed = frozenset({_table("companies")})

    assert (
        "resource_out_of_scope"
        in _validate("SELECT * FROM orders", allowed=allowed).issue_codes
    )
    assert (
        "resource_out_of_scope"
        in _validate("SELECT * FROM sqlite_schema", allowed=allowed).issue_codes
    )
    assert _validate("SELECT * FROM sqlite_schema").valid
    assert _validate("SELECT 1").valid


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
