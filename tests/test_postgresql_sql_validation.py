from collections.abc import Mapping

import pytest

from daita.domains.data.capabilities import postgresql_query_capability_declarations
from daita.domains.data.export_capabilities import (
    POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
    artifact_capability_declarations,
)
from daita.domains.data.sql import (
    MAX_SQL_CHARACTERS,
    MAX_SQL_PARAMETERS,
    ResourceSchema,
    analyze_postgresql_sql,
    validate_postgresql_read,
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
):
    return validate_postgresql_read(
        sql,
        source_id="source-postgresql",
        resources=resources,
        parameters=parameters,
        allowed_resource_ids=allowed_resource_ids,
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


def test_cte_order_by_output_alias_wins_before_parent_scope_lookup() -> None:
    result = _validate("""
        WITH selected AS (
          SELECT c.name AS customer_id
          FROM sales.customers AS c
          ORDER BY customer_id
        )
        SELECT selected.customer_id
        FROM sales.orders AS o
        CROSS JOIN selected
        """)

    assert result.valid, result.issues


def test_cte_cannot_escape_to_same_named_parent_column_without_local_alias() -> None:
    result = _validate("""
        WITH selected AS (
          SELECT c.name
          FROM sales.customers AS c
          ORDER BY customer_id
        )
        SELECT selected.name
        FROM sales.orders AS o
        CROSS JOIN selected
        """)

    assert not result.valid
    assert "column_scope_escape" in result.issue_codes


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
    ("sql", "function_name"),
    (
        (
            "SELECT o.id FROM sales.orders o "
            "JOIN LATERAL lower(o.status) AS value ON true",
            "LOWER",
        ),
        (
            "SELECT o.id FROM sales.orders o JOIN LATERAL "
            "pg_catalog.generate_series(1, 3) AS value ON true",
            "PG_CATALOG.GENERATE_SERIES",
        ),
    ),
)
def test_relation_producing_function_position_is_rejected(
    sql: str,
    function_name: str,
) -> None:
    result = _validate(sql)

    assert not result.valid
    assert "function_not_allowed" in result.issue_codes
    assert result.analysis is not None
    assert function_name in result.analysis.table_function_names


@pytest.mark.parametrize(
    ("sql", "function_name"),
    (
        (
            "SELECT pg_catalog.pg_read_file('/etc/passwd') FROM sales.orders o",
            "PG_CATALOG.PG_READ_FILE",
        ),
        (
            "SELECT pg_catalog.pg_sleep(1) FROM sales.orders o",
            "PG_CATALOG.PG_SLEEP",
        ),
        (
            "SELECT pg_catalog.lo_export(1, '/tmp/value') FROM sales.orders o",
            "PG_CATALOG.LO_EXPORT",
        ),
        (
            "SELECT pg_catalog.lo_import('/tmp/value') FROM sales.orders o",
            "PG_CATALOG.LO_IMPORT",
        ),
        (
            "SELECT pg_catalog.pg_advisory_lock(1) FROM sales.orders o",
            "PG_CATALOG.PG_ADVISORY_LOCK",
        ),
        (
            "SELECT pg_catalog.pg_terminate_backend(1) FROM sales.orders o",
            "PG_CATALOG.PG_TERMINATE_BACKEND",
        ),
        (
            "SELECT pg_catalog.set_config('search_path', 'public', false) "
            "FROM sales.orders o",
            "PG_CATALOG.SET_CONFIG",
        ),
        (
            "SELECT pg_catalog.nextval('sales.sequence') FROM sales.orders o",
            "PG_CATALOG.NEXTVAL",
        ),
        (
            "SELECT pg_catalog.pg_ls_dir('/tmp') FROM sales.orders o",
            "PG_CATALOG.PG_LS_DIR",
        ),
        (
            "SELECT pg_catalog.current_setting('search_path') FROM sales.orders o",
            "PG_CATALOG.CURRENT_SETTING",
        ),
        (
            "SELECT pg_catalog.query_to_xml('SELECT 1', true, false, '') "
            "FROM sales.orders o",
            "PG_CATALOG.QUERY_TO_XML",
        ),
        ("SELECT random() FROM sales.orders o", "RANDOM"),
        ("SELECT CURRENT_TIMESTAMP FROM sales.orders o", "CURRENT_TIMESTAMP"),
        (
            "SELECT public.custom_function(o.id) FROM sales.orders o",
            "PUBLIC.CUSTOM_FUNCTION",
        ),
    ),
)
def test_external_state_and_untrusted_namespace_functions_fail_closed(
    sql: str,
    function_name: str,
) -> None:
    result = _validate(sql)

    assert not result.valid
    issue = next(item for item in result.issues if item.code == "function_not_allowed")
    denied_functions = issue.details["functions"]
    assert isinstance(denied_functions, tuple)
    assert function_name in denied_functions


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT c.id FROM sales.customers c WHERE EXISTS "
        "(SELECT 1 FROM sales.orders o WHERE o.customer_id = c.id)",
        "WITH recent AS (SELECT o.id, o.status FROM sales.orders o) "
        "SELECT recent.id FROM recent UNION ALL SELECT o.id FROM sales.orders o",
        "SELECT DISTINCT ON (o.customer_id) o.customer_id, o.id "
        "FROM sales.orders o ORDER BY o.customer_id, o.ordered_at DESC",
    ),
)
def test_catalog_scoped_postgresql_query_shapes_remain_valid(sql: str) -> None:
    result = _validate(sql)

    assert result.valid, result.issues


@pytest.mark.parametrize(
    ("sql", "issue_code"),
    (
        ("SELECT 1", "resource_scope_empty"),
        ("SELECT id FROM orders", "schema_qualification_required"),
        ("SELECT o.missing FROM sales.orders o", "missing_column"),
        (
            "SELECT id FROM sales.orders o JOIN sales.customers c "
            "ON o.customer_id = c.id",
            "ambiguous_column",
        ),
        ("SELECT o.id FROM private.orders o", "unknown_resource"),
        ("SELECT o.id FROM sales.orders o FOR UPDATE", "mutation_not_allowed"),
        ("SELECT o.id FROM sales.orders o; SELECT 1", "multiple_statements"),
        ("EXPLAIN SELECT o.id FROM sales.orders o", "explain_not_allowed"),
        (
            "SELECT o.total_amount OPERATOR(public.+) 1 FROM sales.orders o",
            "operator_not_allowed",
        ),
        (
            "SELECT 'sales.orders'::regclass FROM sales.orders o",
            "cast_type_not_allowed",
        ),
        (
            "SELECT o.status::public.custom_type FROM sales.orders o",
            "cast_type_not_allowed",
        ),
        (
            "WITH changed AS (DELETE FROM sales.orders RETURNING id) "
            "SELECT id FROM changed",
            "mutation_not_allowed",
        ),
    ),
)
def test_invalid_statement_and_catalog_shapes_fail_closed(
    sql: str,
    issue_code: str,
) -> None:
    result = _validate(sql)

    assert not result.valid
    assert issue_code in result.issue_codes


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


def test_canonical_sql_is_schema_exact_and_format_stable() -> None:
    first = analyze_postgresql_sql("SELECT o.id FROM sales.orders o")
    second = analyze_postgresql_sql("  SELECT o.id\nFROM sales.orders AS o;  ")

    assert "FROM ONLY sales.orders" in first.canonical_sql
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
    query_capability = postgresql_query_capability_declarations().capabilities[0]
    export_capability = next(
        item
        for item in artifact_capability_declarations().capabilities
        if item.id == POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID
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
