import logging
from collections.abc import Mapping

import pytest

from daita.domains.data.capabilities import data_query_capability_declarations
from daita.domains.data.export_capabilities import (
    DATA_EXPORT_TABULAR_CAPABILITY_ID,
    exact_export_capability_declarations,
)
from daita.domains.data.sql import (
    MAX_SQL_CHARACTERS,
    MAX_SQL_PARAMETERS,
    ResourceSchema,
    analyze_postgresql_sql,
    validate_read,
)

_REVISION = "sha256:" + "1" * 64
_SOURCE_REVISION = "sha256:" + "2" * 64


def _resource(
    resource_id: str,
    name: str,
    columns: tuple[str, ...],
    *,
    schema: str = "sales",
    resource_kind: str = "table",
) -> ResourceSchema:
    return ResourceSchema(
        resource_id=resource_id,
        source_id="source-postgresql",
        name=name,
        aliases=(f"{schema}.{name}",),
        columns=columns,
        revision=_REVISION,
        source_revision=_SOURCE_REVISION,
        resource_kind=resource_kind,
    )


_ORDERS = _resource(
    "resource-orders",
    "orders",
    (
        "id",
        "customer_id",
        "status",
        "total_amount",
        "ordered_at",
        "payload",
    ),
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
    resources: tuple[ResourceSchema, ...] = _RESOURCES,
    allowed_resource_ids: tuple[str, ...] | None = None,
    schemas: tuple[str, ...] = ("sales",),
):
    return validate_read(
        sql,
        dialect="postgresql",
        source_id="source-postgresql",
        resources=resources,
        parameters=parameters,
        allowed_resource_ids=allowed_resource_ids,
        schemas=schemas,
    )


@pytest.mark.parametrize(
    ("sql", "expected"),
    (
        ("SELECT id FROM sales.customers", "resource-customers"),
        ("SELECT id FROM SALES.CUSTOMERS", "resource-customers"),
        ('SELECT id FROM sales."customers"', "resource-customers"),
        ('SELECT id FROM sales."Customers"', "resource-quoted"),
    ),
)
def test_a_quoted_mixed_case_table_is_not_its_lower_case_namesake(
    sql: str, expected: str
) -> None:
    quoted = _resource("resource-quoted", "Customers", ("id",))
    result = _validate(sql, resources=(_CUSTOMERS, quoted))
    assert result.valid and result.resource_ids == (expected,)


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
        ("SELECT 1", "resource_scope_empty"),
        ("SELECT o.id FROM private.orders o", "resource_out_of_scope"),
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
    assert result.resource_ids == ("resource-orders",)
    assert result.function_lookups == ("GENERATE_SERIES",)


@pytest.mark.parametrize(
    ("sql", "schemas", "valid"),
    (
        ("SELECT id FROM orders", ("sales",), True),
        ("SELECT id FROM ORDERS", ("sales",), True),
        ("SELECT id FROM orders", ("sales", "legacy"), False),
        ("SELECT id FROM sales.orders", ("sales", "legacy"), True),
    ),
)
def test_an_unqualified_name_needs_exactly_one_configured_schema(
    sql: str, schemas: tuple[str, ...], valid: bool
) -> None:
    result = _validate(sql, schemas=schemas)

    assert result.valid is valid
    if not valid:
        assert result.issue_codes == ("unqualified_relation",)
        assert result.issues[0].details["candidates"] == ("sales.orders",)
        assert result.issues[0].details["schemas"] == schemas


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT * FROM pg_tables",
        "SELECT * FROM pg_catalog.pg_class",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM legacy.orders",
        'SELECT * FROM sales."ORDERS"',
        'SELECT * FROM "SALES".orders',
        "SELECT * FROM db.sales.orders",
        "WITH sales AS (SELECT 1) SELECT * FROM private.salaries",
    ),
)
def test_scope_escapes_are_refused(sql: str) -> None:
    result = _validate(sql)

    assert not result.valid
    assert "resource_out_of_scope" in result.issue_codes


def test_an_unqualified_system_catalog_name_never_means_a_namesake_table() -> None:
    namesake = _resource("resource-namesake", "pg_tables", ("id",))

    result = _validate("SELECT * FROM pg_tables", resources=(namesake,))

    assert "resource_out_of_scope" in result.issue_codes


def test_a_cte_shadowing_a_table_name_cannot_hide_its_inner_read() -> None:
    result = _validate(
        "WITH orders AS (SELECT * FROM private.salaries) SELECT * FROM orders"
    )

    assert "resource_out_of_scope" in result.issue_codes


def test_allowed_resource_scope_is_enforced() -> None:
    result = _validate(
        "SELECT c.id FROM sales.customers c",
        allowed_resource_ids=("resource-orders",),
    )

    assert not result.valid
    assert "resource_out_of_scope" in result.issue_codes


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


def test_canonical_sql_keeps_partitions_and_is_format_stable() -> None:
    first = analyze_postgresql_sql("SELECT o.id FROM sales.orders o")
    second = analyze_postgresql_sql("  SELECT o.id\nFROM sales.orders AS o;  ")

    assert "ONLY" not in first.canonical_sql
    assert second.canonical_sql == first.canonical_sql
    assert second.sql_fingerprint == first.sql_fingerprint


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
