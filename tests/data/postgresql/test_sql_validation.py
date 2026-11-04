import logging
from collections.abc import Mapping

import pytest

from daita.catalog.models import ResourceKind, catalog_resource_id
from daita.domains.data.capabilities import data_query_capability_declarations
from daita.domains.data.export_capabilities import (
    DATA_EXPORT_TABULAR_CAPABILITY_ID,
    exact_export_capability_declarations,
)
from daita.domains.data.sql import (
    MAX_SQL_CHARACTERS,
    MAX_SQL_PARAMETERS,
    SqlAnalysisError,
    analyze_postgresql_sql,
    validate_read,
)

_SOURCE = "source-postgresql"


def _table(name: str) -> str:
    return catalog_resource_id(_SOURCE, ResourceKind.TABLE, f"sales.{name}")


def _view(name: str) -> str:
    return catalog_resource_id(_SOURCE, ResourceKind.VIEW, f"sales.{name}")


def _validate(
    sql: str,
    *,
    parameters: tuple[object, ...] = (),
    allowed: frozenset[str] | None = None,
    schemas: tuple[str, ...] = ("sales",),
):
    return validate_read(
        sql,
        dialect="postgresql",
        source_id=_SOURCE,
        parameters=parameters,
        allowed_resource_ids=allowed,
        schemas=schemas,
    )


@pytest.mark.parametrize(
    ("sql", "expected"),
    (
        ("SELECT id FROM sales.customers", "sales.customers"),
        ("SELECT id FROM SALES.CUSTOMERS", "sales.customers"),
        ('SELECT id FROM sales."customers"', "sales.customers"),
        ('SELECT id FROM sales."Customers"', "sales.Customers"),
        ("SELECT id FROM customers", "sales.customers"),
    ),
)
def test_a_name_folds_as_the_server_does_and_hashes_forward(
    sql: str, expected: str
) -> None:
    result = _validate(sql)

    assert result.valid
    assert result.resource_names == (
        (catalog_resource_id(_SOURCE, ResourceKind.TABLE, expected), expected),
    )


@pytest.mark.parametrize(
    ("sql", "expected_names"),
    (
        (
            "SELECT date_trunc('day', o.ordered_at) FROM sales.orders o",
            ("DATE_TRUNC",),
        ),
        (
            "SELECT bool_and(o.total_amount > 0) FROM sales.orders o",
            ("BOOL_AND",),
        ),
        (
            "SELECT to_char(o.ordered_at, 'YYYY-MM-DD') FROM sales.orders o",
            ("TO_CHAR",),
        ),
        (
            "SELECT json_agg(o.status) FROM sales.orders o",
            ("JSON_AGG",),
        ),
        (
            "SELECT string_agg(o.status, ',') FROM sales.orders o",
            ("STRING_AGG",),
        ),
        (
            "SELECT decode(o.status, 'hex') FROM sales.orders o",
            ("DECODE",),
        ),
    ),
)
def test_analysis_uses_executed_postgresql_callable_names(
    sql: str,
    expected_names: tuple[str, ...],
) -> None:
    analysis = analyze_postgresql_sql(sql)

    assert analysis.function_names == expected_names


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT lower(o.status) FROM sales.orders o",
        "SELECT o.id FROM sales.orders o WHERE lower(o.status) = 'paid'",
        "SELECT o.status FROM sales.orders o GROUP BY o.status "
        "HAVING lower(o.status) <> ''",
        "SELECT o.status FROM sales.orders o ORDER BY lower(o.status)",
        "SELECT o.id FROM sales.orders o JOIN sales.customers c "
        "ON lower(o.customer_id::text) = lower(c.id::text)",
        "SELECT o.id FROM sales.orders o JOIN sales.customers c "
        "ON coalesce(o.customer_id, '') = c.id",
        "SELECT o.id FROM sales.orders o JOIN sales.customers c "
        "ON o.customer_id::text = c.id::text",
    ),
)
def test_safe_scalar_expressions_are_admitted_in_every_scalar_context(sql: str) -> None:
    result = _validate(sql)

    assert result.valid, result.issues


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT date_trunc('day', o.ordered_at), count(*) "
        "FROM sales.orders o GROUP BY 1",
        "SELECT array_agg(o.status), string_agg(o.status, ','), "
        "json_agg(o.status), jsonb_agg(o.status) FROM sales.orders o",
        "SELECT bool_and(o.total_amount > 0), bool_or(o.total_amount > 0), "
        "stddev_pop(o.total_amount), variance(o.total_amount) "
        "FROM sales.orders o",
        "SELECT rank() OVER (ORDER BY o.total_amount), "
        "dense_rank() OVER (ORDER BY o.total_amount), "
        "lag(o.status) OVER (ORDER BY o.ordered_at), "
        "lead(o.status) OVER (ORDER BY o.ordered_at) FROM sales.orders o",
    ),
)
def test_common_analytical_function_families_are_admitted(sql: str) -> None:
    result = _validate(sql)

    assert result.valid, result.issues


def test_latest_order_dates_export_regression_is_valid() -> None:
    result = _validate("""
        WITH latest_dates AS (
          SELECT (ordered_at AT TIME ZONE 'UTC')::date AS order_date
          FROM sales.orders
          GROUP BY 1
          ORDER BY 1 DESC
          LIMIT 30
        )
        SELECT
          (o.ordered_at AT TIME ZONE 'UTC')::date AS order_date,
          COUNT(*) AS order_count,
          SUM(o.total_amount) AS total_order_value,
          COUNT(*) FILTER (WHERE o.status = 'paid') AS completed_order_count,
          COUNT(*) FILTER (WHERE o.status = 'cancelled') AS failed_order_count
        FROM sales.orders AS o
        JOIN latest_dates AS d
          ON (o.ordered_at AT TIME ZONE 'UTC')::date = d.order_date
        GROUP BY 1
        ORDER BY order_date DESC
        """)

    assert result.valid, result.issues
    assert result.analysis is not None
    assert result.analysis.function_names == ("CAST", "COUNT", "SUM")
    assert result.analysis.table_function_names == ()


def test_scalar_function_inside_join_predicate_is_not_a_table_function() -> None:
    analysis = analyze_postgresql_sql(
        "SELECT o.id FROM sales.orders o JOIN sales.customers c "
        "ON lower(o.customer_id::text) = lower(c.id::text)"
    )

    assert analysis.table_function_names == ()


def test_scalar_function_inside_lateral_subquery_is_not_a_table_function() -> None:
    result = _validate(
        "SELECT o.id FROM sales.orders o JOIN LATERAL "
        "(SELECT lower(o.status) AS value) x ON true"
    )

    assert result.valid, result.issues
    assert result.analysis is not None
    assert result.analysis.table_function_names == ()


def test_only_outer_relation_callable_is_classified_as_table_function() -> None:
    analysis = analyze_postgresql_sql(
        "SELECT o.id FROM sales.orders o JOIN LATERAL "
        "pg_catalog.generate_series(length(o.status), 3) g ON true"
    )

    assert analysis.function_names == ("LENGTH", "PG_CATALOG.GENERATE_SERIES")
    assert analysis.table_function_names == ("PG_CATALOG.GENERATE_SERIES",)


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT c.id FROM sales.customers c WHERE EXISTS "
        "(SELECT 1 FROM sales.orders o WHERE o.customer_id = c.id)",
        "WITH recent AS (SELECT o.id, o.status FROM sales.orders o) "
        "SELECT recent.id FROM recent UNION ALL SELECT o.id FROM sales.orders o",
        "SELECT DISTINCT ON (o.customer_id) o.customer_id, o.id "
        "FROM sales.orders o ORDER BY o.customer_id, o.ordered_at DESC",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) "
        "SELECT r.n FROM r, sales.orders o",
        "SELECT o.id FROM sales.orders o JOIN LATERAL "
        "pg_catalog.generate_series(1, 3) AS value ON true",
        "SELECT 'sales.orders'::regclass, o.status::public.custom_type "
        "FROM sales.orders o",
        "SELECT o.total_amount OPERATOR(pg_catalog.+) 1 FROM sales.orders o",
        "SELECT o.id FROM ONLY sales.orders o",
    ),
)
def test_ordinary_postgresql_query_shapes_remain_valid(sql: str) -> None:
    result = _validate(sql)

    assert result.valid, result.issues


@pytest.mark.parametrize(
    ("sql", "issue_code"),
    (
        ("SELECT o.id FROM private.orders o", "schema_not_configured"),
        ("SELECT o.id FROM sales.orders o FOR UPDATE", "mutation_not_allowed"),
        ("SELECT * INTO sales.copy FROM sales.orders", "mutation_not_allowed"),
        ("SELECT o.id FROM sales.orders o; SELECT 1", "multiple_statements"),
        ("/* SELECT 1; */ DROP TABLE sales.orders", "mutation_not_allowed"),
        ("TRUNCATE sales.orders", "mutation_not_allowed"),
        ("COPY sales.orders TO '/tmp/out'", "read_statement_required"),
        ("SET search_path = private", "read_statement_required"),
        ("DO $$ BEGIN DELETE FROM sales.orders; END $$", "read_statement_required"),
        ("COMMIT", "read_statement_required"),
        ("VALUES (1)", "read_statement_required"),
        ("EXPLAIN SELECT o.id FROM sales.orders o", "explain_not_allowed"),
        (
            "SELECT o.total_amount OPERATOR(public.+) 1 FROM sales.orders o",
            "operator_not_allowed",
        ),
        (
            "WITH changed AS (DELETE FROM sales.orders RETURNING id) "
            "SELECT id FROM changed",
            "mutation_not_allowed",
        ),
    ),
)
def test_invalid_statements_fail_closed_before_any_source_io(
    sql: str,
    issue_code: str,
) -> None:
    result = _validate(sql)

    assert not result.valid
    assert issue_code in result.issue_codes


@pytest.mark.parametrize(
    "call",
    (
        "query_to_xml('select * from private.salaries', true, true, '')",
        "pg_catalog.query_to_xml('select 1', true, true, '')",
        "query_to_xmlschema('select 1', true, true, '')",
        "query_to_xml_and_xmlschema('select 1', true, true, '')",
        "table_to_xml('private.salaries', true, true, '')",
        "table_to_xmlschema('private.salaries', true, true, '')",
        "table_to_xml_and_xmlschema('private.salaries', true, true, '')",
        "schema_to_xml('private', true, true, '')",
        "database_to_xml(true, true, '')",
        "cursor_to_xml('c', 1, true, true, '')",
        "cursor_to_xmlschema('c', true, true, '')",
        "ts_stat('select 1')",
        "ts_rewrite('a'::tsquery, 'select 1')",
        "QUERY_TO_XML('select 1', true, true, '')",
        "public.custom_function(o.id)",
        '"pg_read_file"(1)',
    ),
)
def test_functions_refused_in_every_scope_never_reach_the_server(call: str) -> None:
    result = _validate(f"SELECT {call} FROM sales.orders o")

    assert not result.valid
    assert "function_not_allowed" in result.issue_codes


@pytest.mark.parametrize(
    "relation",
    (
        "public.custom_function(1)",
        '"my schema".custom_function(1)',
        "a.b.custom_function(1)",
        '"custom_function"(1)',
    ),
)
def test_a_relation_position_call_outside_pg_catalog_is_refused(relation: str) -> None:
    result = _validate(f"SELECT * FROM sales.orders o, {relation} AS shown")

    assert not result.valid
    assert "function_not_allowed" in result.issue_codes


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT 1 ORDER BY 1 USING <",
        "SELECT pg_read_file('/etc/passwd') FROM sales.orders ORDER BY 1 USING <",
        "SELECT * FROM",
    ),
)
def test_sql_that_fails_to_parse_is_refused_with_its_position(sql: str) -> None:
    result = _validate(sql)

    assert not result.valid and result.analysis is None
    assert result.issue_codes == ("sql_parse_error",)
    assert "line 1, column" in result.issues[0].message
    assert "Rewrite" in result.issues[0].message


def test_a_parse_refusal_is_logged_by_position_without_the_sql_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="daita.data"):
        _validate("SELECT 'private-value' ORDER BY 1 USING <")

    assert "sql_parse_refused" in caplog.text and "column=" in caplog.text
    assert "private-value" not in caplog.text


def test_the_read_transaction_checks_called_functions_except_grammar_forms() -> None:
    result = _validate(
        "SELECT lower(o.status), coalesce(o.status, ''), CAST(o.id AS text), "
        "pg_catalog.upper(o.status), now() FROM sales.orders o"
    )

    assert result.valid, result.issues
    assert result.function_lookups == ("LOWER", "PG_CATALOG.UPPER")


@pytest.mark.parametrize(
    ("sql", "names"),
    (
        ("SELECT o.peek FROM sales.orders o", ("peek",)),
        ("SELECT O.PEEK FROM sales.orders o", ("peek",)),
        ('SELECT o."Peek" FROM sales.orders o', ("Peek",)),
        ("SELECT (o).peek FROM sales.orders o", ("peek",)),
        ("SELECT ((o)).peek FROM sales.orders o", ("peek",)),
        ("SELECT (o.*).peek FROM sales.orders o", ("peek",)),
        ("SELECT sales.orders.peek FROM sales.orders", ("peek",)),
        ("SELECT id FROM sales.orders o WHERE o.peek = 'x'", ("peek",)),
        ("SELECT (SELECT o2 FROM sales.orders o2 LIMIT 1).peek", ("peek",)),
        ("SELECT id FROM sales.orders", ()),
    ),
)
def test_every_attribute_form_reaches_the_server_lookup(
    sql: str, names: tuple[str, ...]
) -> None:
    result = _validate(sql)

    assert result.valid, result.issues
    assert result.attribute_lookups == names


def test_a_table_function_is_a_function_fact_and_not_a_relation() -> None:
    result = _validate("SELECT * FROM sales.orders o, generate_series(1, 3) AS g")

    assert result.valid, result.issues
    assert result.resource_ids == (_table("orders"),)
    assert result.function_lookups == ("GENERATE_SERIES",)


def test_a_read_needs_no_table_at_all() -> None:
    result = _validate("SELECT 1")

    assert result.valid and result.resource_ids == ()


def test_an_unqualified_name_across_schemas_is_left_for_a_live_lookup() -> None:
    result = _validate("SELECT id FROM ORDERS", schemas=("legacy", "sales"))

    assert result.valid
    assert result.unqualified_relations == ("orders",)
    assert result.resource_ids == ()


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT * FROM legacy.orders",
        'SELECT * FROM "SALES".orders',
    ),
)
def test_a_relation_outside_the_configured_schemas_names_how_to_add_it(
    sql: str,
) -> None:
    result = _validate(sql)

    assert result.issue_codes == ("schema_not_configured",)
    assert "Add the schema" in result.issues[0].message


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT * FROM pg_tables",
        "SELECT * FROM pg_catalog.pg_class",
        "SELECT * FROM information_schema.tables",
    ),
)
def test_system_catalogs_need_the_whole_source(sql: str) -> None:
    assert _validate(sql).valid

    selected = _validate(sql, allowed=frozenset({_table("orders")}))
    assert selected.issue_codes == ("resource_out_of_scope",)
    assert "catalog_describe" in selected.issues[0].message


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT * FROM sales.customers",
        "SELECT * FROM customers",
        'SELECT * FROM sales."Orders"',
        'SELECT * FROM sales."ORDERS"',
        "SELECT * FROM db.sales.orders",
        "WITH orders AS (SELECT * FROM sales.customers) SELECT * FROM orders",
        "SELECT * FROM sales.orders UNION ALL SELECT * FROM sales.customers",
    ),
)
def test_selected_scope_refuses_every_other_name(sql: str) -> None:
    result = _validate(sql, allowed=frozenset({_table("orders")}))

    assert "resource_out_of_scope" in result.issue_codes


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT * FROM sales.orders",
        "SELECT * FROM ORDERS",
        'SELECT * FROM sales."orders"',
        "WITH o AS (SELECT * FROM sales.orders) SELECT * FROM o",
    ),
)
def test_selected_scope_admits_its_table_however_it_is_spelled(sql: str) -> None:
    result = _validate(sql, allowed=frozenset({_table("orders")}))

    assert result.valid
    assert result.resource_ids == (_table("orders"),)


def test_selected_scope_admits_a_view_by_its_own_kind() -> None:
    result = _validate(
        "SELECT * FROM sales.recent", allowed=frozenset({_view("recent")})
    )

    assert result.valid and result.resource_ids == (_view("recent"),)


def test_an_unqualified_system_catalog_name_never_means_a_namesake_table() -> None:
    result = _validate(
        "SELECT * FROM pg_tables", allowed=frozenset({_table("pg_tables")})
    )

    assert result.issue_codes == ("resource_out_of_scope",)


@pytest.mark.parametrize(
    ("sql", "parameters", "valid", "issue_code"),
    (
        (
            "SELECT o.id FROM sales.orders o WHERE o.id = $1",
            (1,),
            True,
            None,
        ),
        (
            "SELECT o.id FROM sales.orders o WHERE o.id = $2",
            (1, 2),
            False,
            "parameter_index_mismatch",
        ),
        (
            "SELECT o.id FROM sales.orders o WHERE o.id = ?",
            (1,),
            False,
            "parameter_style_invalid",
        ),
    ),
)
def test_postgresql_parameters_are_contiguous_and_numbered(
    sql: str,
    parameters: tuple[object, ...],
    valid: bool,
    issue_code: str | None,
) -> None:
    result = _validate(sql, parameters=parameters)

    assert result.valid is valid
    if issue_code is not None:
        assert issue_code in result.issue_codes


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT o.id\nFROM sales.orders AS o",
        "select /* why */ o.id from sales.orders o -- trailing",
        r"SELECT id FROM sales.items WHERE name ~ E'\\d+'",
        "SELECT $1,$2",
    ),
)
def test_the_server_receives_the_statement_as_written(sql: str) -> None:
    assert analyze_postgresql_sql(sql).canonical_sql == sql
    assert analyze_postgresql_sql(sql + " ;  ").canonical_sql == sql


@pytest.mark.parametrize(
    ("call", "name"),
    (
        ("ceiling(1)", "ceiling"),
        ("mod(2, 3)", "mod"),
        ("date_part('year', ts)", "date_part"),
        ("now('x')", "now"),
        ("NOW ()", "now"),
    ),
)
def test_a_call_is_checked_as_written_not_as_the_parser_renames_it(
    call: str, name: str
) -> None:
    result = _validate(f"SELECT {call} FROM sales.orders")
    assert name in result.written_calls


def test_a_quoted_call_keeps_its_case_for_the_function_rule() -> None:
    """The tree reads "LOWER"(1) as lower(1), but the server keeps the quotes."""

    result = _validate('SELECT "LOWER"(1), "evil"(2) FROM sales.orders')
    assert {"LOWER", "evil"} <= set(result.written_calls)


def test_grammar_and_qualified_calls_are_not_written_calls() -> None:
    result = _validate(
        "SELECT pg_catalog.lower(a) FROM sales.orders WHERE id IN (1) AND b = ANY ($1)",
        parameters=(1,),
    )
    assert result.written_calls == ()


def test_sql_and_parameter_inputs_are_bounded_before_execution() -> None:
    oversized_sql = "SELECT o.id FROM sales.orders o " + (" " * MAX_SQL_CHARACTERS)
    sql_result = _validate(oversized_sql)
    parameter_result = _validate(
        "SELECT o.id FROM sales.orders o WHERE o.id = $1",
        parameters=tuple(range(MAX_SQL_PARAMETERS + 1)),
    )

    assert not sql_result.valid
    assert sql_result.analysis is None
    assert sql_result.issue_codes == ("sql_too_large",)
    assert not parameter_result.valid
    assert "parameter_count_exceeded" in parameter_result.issue_codes


def test_model_facing_sql_capabilities_publish_the_same_input_bounds() -> None:
    query_capability = data_query_capability_declarations().capabilities[0]
    export_capability = next(
        item
        for item in exact_export_capability_declarations(
            adapter_ids=("sqlite", "postgresql")
        ).capabilities
        if item.id == DATA_EXPORT_TABULAR_CAPABILITY_ID
    )

    for capability in (query_capability, export_capability):
        properties = capability.input_schema["properties"]
        assert isinstance(properties, Mapping)
        sql_schema = properties["sql"]
        parameter_schema = properties["parameters"]
        assert isinstance(sql_schema, Mapping)
        assert isinstance(parameter_schema, Mapping)
        assert sql_schema["maxLength"] == MAX_SQL_CHARACTERS
        assert parameter_schema["maxItems"] == MAX_SQL_PARAMETERS


@pytest.mark.parametrize(
    ("sql", "parameters"),
    (
        ("SELECT $1,$2", (1, 2)),
        ("SELECT $1,$2,$3", (1, 2, 3)),
        ("SELECT $1+$2", (1, 2)),
        ("SELECT $1::int,$2::int", (1, 2)),
        ("SELECT $1,'x',$2", (1, 2)),
        ("SELECT * FROM t WHERE a IN ($1,$2)", (1, 2)),
    ),
)
def test_numbered_parameters_need_no_space_between_them(
    sql: str, parameters: tuple[int, ...]
) -> None:
    analysis = analyze_postgresql_sql(sql)
    assert analysis.parameter_ordinals == tuple(range(1, len(parameters) + 1))


@pytest.mark.parametrize(
    ("sql", "kept"),
    (
        ("SELECT $1,$2, 'cost $1,$2'", "'cost $1,$2'"),
        ("SELECT $1,$2, 'it''s $1,$2'", "'it''s $1,$2'"),
        ("SELECT $1,$2, E'a\\'$1,$2'", "$1,$2"),
        ('SELECT $1,$2 AS "$1,$2"', '"$1,$2"'),
        ("SELECT $1,$2, $tag$ $1,$2 $tag$", " $1,$2 "),
        ("SELECT $1,$2, $$ $1,$2 $$", " $1,$2 "),
    ),
)
def test_spacing_parameters_leaves_literals_exactly_as_written(
    sql: str, kept: str
) -> None:
    assert kept in analyze_postgresql_sql(sql).canonical_sql


def test_a_comment_cannot_hide_a_statement_from_parameter_spacing() -> None:
    analysis = analyze_postgresql_sql("SELECT $1,$2 /* $3,$4 */; DROP TABLE t -- $5,$6")
    assert analysis.statement_count == 2


def test_an_unterminated_dollar_quote_is_still_refused() -> None:
    with pytest.raises(SqlAnalysisError, match="could not be parsed"):
        analyze_postgresql_sql("SELECT $tag$ never closed")


@pytest.mark.parametrize(
    "sql",
    (
        r"SELECT id FROM sales.items WHERE name ~ E'\\d+'",
        r"SELECT E'a\\nb'",
    ),
)
def test_a_backslash_in_an_escape_string_reaches_the_server_unchanged(sql: str) -> None:
    result = _validate(sql)
    assert not result.issues
    assert result.analysis is not None and result.analysis.canonical_sql == sql


@pytest.mark.parametrize(
    "literal", ("E'plain'", r"E'a\nb'", r"E'it\'s'", r"'a\b'", r"$$a\b$$")
)
def test_strings_that_keep_their_meaning_still_read(literal: str) -> None:
    assert not _validate(f"SELECT {literal} AS value").issues
