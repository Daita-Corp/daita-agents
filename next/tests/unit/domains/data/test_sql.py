from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError

import pytest

from daita.domains.data import (
    ResourceSchema,
    SqlAnalysisError,
    analyze_sqlite_sql,
    normalize_sql,
    validate_sqlite_read,
)


@pytest.fixture
def resources() -> tuple[ResourceSchema, ...]:
    return (
        ResourceSchema(
            resource_id="resource:sqlite:orders",
            source_id="source:sqlite:test",
            name="orders",
            aliases=("main.orders",),
            columns=("id", "customer_id", "status", "total"),
        ),
        ResourceSchema(
            resource_id="resource:sqlite:customers",
            source_id="source:sqlite:test",
            name="customers",
            aliases=("main.customers",),
            columns=("id", "name"),
        ),
    )


def test_normalize_sql_removes_only_trailing_statement_punctuation() -> None:
    assert normalize_sql("SELECT ';' AS marker;  ") == "SELECT ';' AS marker"
    assert normalize_sql("SELECT 1") == "SELECT 1"


@pytest.mark.parametrize("sql", ("SELECT 1;;", "SELECT 1; ;", "SELECT 1;;;"))
def test_repeated_trailing_statement_delimiters_fail_closed(
    resources: tuple[ResourceSchema, ...],
    sql: str,
) -> None:
    analysis = analyze_sqlite_sql(sql)
    result = validate_sqlite_read(
        sql,
        source_id="source:sqlite:test",
        resources=resources,
    )

    assert normalize_sql(sql).count(";") >= 2
    assert analysis.statement_count > 1
    assert result.valid is False
    assert "multiple_statements" in result.issue_codes


def test_analysis_extracts_aliases_tables_columns_and_positional_parameters() -> None:
    analysis = analyze_sqlite_sql("""
        SELECT o.id AS order_id, c.name
        FROM main.orders AS o
        JOIN customers c ON c.id = o.customer_id
        WHERE o.status = ? AND c.name <> '?'
        LIMIT 20;
        """)

    assert analysis.is_read is True
    assert analysis.statement_count == 1
    assert analysis.has_limit is True
    assert analysis.positional_parameter_count == 1
    assert analysis.mutation_types == ()
    assert [(item.qualified_name, item.alias) for item in analysis.tables] == [
        ("main.orders", "o"),
        ("customers", "c"),
    ]
    assert {
        (item.name, item.qualifier, item.resource_name) for item in analysis.columns
    } >= {
        ("id", "o", "main.orders"),
        ("name", "c", "customers"),
        ("customer_id", "o", "main.orders"),
        ("status", "o", "main.orders"),
    }
    assert analysis.sql_fingerprint.startswith("sha256:")


def test_equivalent_formatting_and_trailing_semicolon_have_same_fingerprint() -> None:
    first = analyze_sqlite_sql("select id from orders where status = ?;")
    second = analyze_sqlite_sql("  SELECT id\nFROM orders\nWHERE status = ?   ")

    assert first.canonical_sql == second.canonical_sql
    assert first.sql_fingerprint == second.sql_fingerprint


def test_analysis_classifies_cte_hidden_mutation() -> None:
    analysis = analyze_sqlite_sql(
        "WITH deleted AS ("
        "DELETE FROM orders WHERE id = 1 RETURNING *"
        ") SELECT * FROM deleted"
    )

    assert analysis.is_read is False
    assert analysis.mutation_types == ("DELETE",)
    assert any(item.name == "deleted" and item.is_cte for item in analysis.tables)
    assert any(item.name == "orders" and not item.is_cte for item in analysis.tables)


def test_nested_cte_name_does_not_hide_unrelated_outer_sqlite_table() -> None:
    allowed = ResourceSchema(
        resource_id="resource:sqlite:allowed",
        source_id="source:sqlite:test",
        name="allowed",
        columns=("id",),
    )
    result = validate_sqlite_read(
        "SELECT allowed.id FROM secret JOIN allowed ON true "
        "WHERE EXISTS (WITH secret AS (SELECT id FROM allowed) "
        "SELECT id FROM secret)",
        source_id="source:sqlite:test",
        resources=(allowed,),
    )

    assert result.valid is False
    assert "unknown_resource" in result.issue_codes
    assert result.analysis is not None
    assert any(
        table.name == "secret" and not table.is_cte for table in result.analysis.tables
    )
    assert any(
        table.name == "secret" and table.is_cte for table in result.analysis.tables
    )


def test_analysis_classifies_explain_by_its_inner_statement() -> None:
    read = analyze_sqlite_sql("EXPLAIN QUERY PLAN SELECT id FROM orders")
    mutation = analyze_sqlite_sql("EXPLAIN DELETE FROM orders WHERE id = 1")

    assert read.statement_type == "explain"
    assert read.is_read is True
    assert mutation.statement_type == "explain"
    assert mutation.is_read is False
    assert mutation.mutation_types == ("DELETE",)


def test_missing_sqlglot_uses_the_sqlite_extra_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name == "sqlglot" or name.startswith("sqlglot."):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError) as exc_info:
        analyze_sqlite_sql("SELECT 1")

    assert str(exc_info.value) == (
        "sqlglot is required for SQLite SQL validation. "
        "Install with: pip install 'daita-agents[sqlite]'"
    )


def test_empty_sql_has_a_stable_bounded_error() -> None:
    with pytest.raises(SqlAnalysisError) as exc_info:
        analyze_sqlite_sql(" ; ")

    assert exc_info.value.code == "empty_sql"
    assert str(exc_info.value) == "SQL must not be empty."


def test_valid_read_is_checked_against_catalog_scope_and_parameter_arity(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_sqlite_read(
        """
        SELECT o.id, c.name
        FROM orders o
        JOIN customers c ON c.id = o.customer_id
        WHERE o.status = ?
        """,
        source_id="source:sqlite:test",
        resources=resources,
        parameters=("complete",),
    )

    assert result.valid is True
    assert result.issue_codes == ()
    assert result.resource_ids == (
        "resource:sqlite:orders",
        "resource:sqlite:customers",
    )


@pytest.mark.parametrize(
    ("sql", "parameters", "expected_code"),
    [
        ("SELECT id FROM orders; SELECT id FROM customers", (), "multiple_statements"),
        ("DELETE FROM orders WHERE id = ?", (1,), "mutation_not_allowed"),
        ("PRAGMA journal_mode", (), "read_statement_required"),
        ("SELECT id FROM orders WHERE status = ?", (), "parameter_count_mismatch"),
    ],
)
def test_invalid_statement_facts_are_deterministic(
    resources: tuple[ResourceSchema, ...],
    sql: str,
    parameters: tuple[object, ...],
    expected_code: str,
) -> None:
    result = validate_sqlite_read(
        sql,
        source_id="source:sqlite:test",
        resources=resources,
        parameters=parameters,
    )

    assert result.valid is False
    assert expected_code in result.issue_codes


def test_placeholder_like_text_inside_literal_is_not_a_parameter(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_sqlite_read(
        "SELECT id FROM orders WHERE status = '?'",
        source_id="source:sqlite:test",
        resources=resources,
    )

    assert result.valid is True
    assert result.analysis is not None
    assert result.analysis.positional_parameter_count == 0


def test_catalog_validation_fails_closed_for_missing_source_schema(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_sqlite_read(
        "SELECT id FROM orders",
        source_id="source:sqlite:other",
        resources=resources,
    )

    assert result.valid is False
    assert result.issue_codes == ("catalog_schema_missing", "unknown_resource")


def test_unknown_resource_and_missing_column_return_bounded_candidates(
    resources: tuple[ResourceSchema, ...],
) -> None:
    unknown = validate_sqlite_read(
        "SELECT id FROM orderz",
        source_id="source:sqlite:test",
        resources=resources,
    )
    missing = validate_sqlite_read(
        "SELECT state FROM orders",
        source_id="source:sqlite:test",
        resources=resources,
    )

    assert unknown.issue_codes == ("unknown_resource",)
    assert unknown.issues[0].details["candidates"] == ("orders",)
    assert missing.issue_codes == ("missing_column",)
    assert missing.issues[0].details["resource_id"] == "resource:sqlite:orders"
    assert missing.issues[0].details["candidates"] == ("status",)


def test_explicit_empty_scope_denies_every_catalog_resource(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_sqlite_read(
        "SELECT id FROM orders",
        source_id="source:sqlite:test",
        resources=resources,
        allowed_resource_ids=(),
    )

    assert result.valid is False
    assert result.issue_codes == ("resource_out_of_scope",)
    assert result.resource_ids == ()


def test_table_alias_does_not_bypass_resource_scope(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_sqlite_read(
        "SELECT c.name FROM customers AS c",
        source_id="source:sqlite:test",
        resources=resources,
        allowed_resource_ids=("resource:sqlite:orders",),
    )

    assert result.issue_codes == ("resource_out_of_scope",)


def test_unqualified_column_shared_by_joined_resources_is_rejected_as_ambiguous(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_sqlite_read(
        "SELECT id FROM orders JOIN customers ON orders.customer_id = customers.id",
        source_id="source:sqlite:test",
        resources=resources,
    )

    assert "ambiguous_column" in result.issue_codes


def test_records_are_immutable(resources: tuple[ResourceSchema, ...]) -> None:
    result = validate_sqlite_read(
        "SELECT id FROM orders",
        source_id="source:sqlite:test",
        resources=resources,
    )

    with pytest.raises(FrozenInstanceError):
        result.valid = False  # type: ignore[misc]
