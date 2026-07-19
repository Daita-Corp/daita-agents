from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence

import pytest

from daita.domains.data import (
    ResourceSchema,
    analyze_postgresql_sql,
    validate_postgresql_read,
)


@pytest.fixture
def resources() -> tuple[ResourceSchema, ...]:
    return (
        ResourceSchema(
            resource_id="resource-postgres-orders",
            source_id="source-postgres",
            name="orders",
            aliases=("public.orders",),
            columns=("id", "status", "customer_id"),
            resource_kind="table",
            column_declared_types=(
                ("id", "integer"),
                ("status", "text"),
                ("customer_id", "integer"),
            ),
            revision="sha256:" + "a" * 64,
            source_revision="catalog:sha256:" + "b" * 64,
        ),
        ResourceSchema(
            resource_id="resource-postgres-camel",
            source_id="source-postgres",
            name="CamelCase",
            aliases=("public.CamelCase",),
            columns=("Id", "Value"),
            resource_kind="table",
            column_declared_types=(("Id", "integer"), ("Value", "text")),
            revision="sha256:" + "c" * 64,
            source_revision="catalog:sha256:" + "b" * 64,
        ),
    )


def test_postgresql_analysis_preserves_numbered_parameters_and_dialect() -> None:
    analysis = analyze_postgresql_sql(
        "SELECT id FROM public.orders WHERE status = $1 OR status = $1"
    )

    assert analysis.is_read is True
    assert analysis.positional_parameter_count == 1
    assert analysis.parameter_ordinals == (1,)
    assert analysis.canonical_sql == (
        "SELECT id FROM ONLY public.orders WHERE status = $1 OR status = $1"
    )


@pytest.mark.parametrize(
    ("sql", "parameters", "code"),
    (
        (
            "SELECT id FROM orders WHERE status = $2",
            ("open",),
            "parameter_index_mismatch",
        ),
        ("SELECT id FROM orders WHERE status = $0", (), "parameter_index_mismatch"),
        (
            "SELECT id FROM orders WHERE status = ?",
            ("open",),
            "parameter_style_invalid",
        ),
        ("DELETE FROM orders WHERE id = $1", (1,), "mutation_not_allowed"),
        ("SELECT id INTO copied_orders FROM orders", (), "mutation_not_allowed"),
    ),
)
def test_postgresql_validation_fails_closed_for_parameters_and_mutation(
    resources: tuple[ResourceSchema, ...],
    sql: str,
    parameters: tuple[object, ...],
    code: str,
) -> None:
    result = validate_postgresql_read(
        sql,
        source_id="source-postgres",
        resources=resources,
        parameters=parameters,
    )

    assert result.valid is False
    assert code in result.issue_codes


def test_postgresql_identifier_folding_respects_quotes(
    resources: tuple[ResourceSchema, ...],
) -> None:
    quoted = validate_postgresql_read(
        'SELECT "Id" FROM public."CamelCase"',
        source_id="source-postgres",
        resources=resources,
    )
    unquoted = validate_postgresql_read(
        "SELECT id FROM public.CamelCase",
        source_id="source-postgres",
        resources=resources,
    )

    assert quoted.valid is True
    assert unquoted.valid is False
    assert "unknown_resource" in unquoted.issue_codes


def test_postgresql_resource_schema_preserves_case_distinct_columns() -> None:
    resource = ResourceSchema(
        resource_id="resource-case-columns",
        source_id="source-postgres",
        name="case_columns",
        columns=("id", "Id"),
        column_declared_types=(("id", "text"), ("Id", "integer")),
    )

    assert resource.columns == ("id", "Id")
    assert resource.column_declared_types == (("id", "text"), ("Id", "integer"))


def test_case_distinct_table_reference_cannot_hide_out_of_scope_resource() -> None:
    case_resources = (
        ResourceSchema(
            resource_id="resource-lower",
            source_id="source-postgres",
            name="foo",
            aliases=("public.foo",),
            columns=("id",),
            resource_kind="table",
            column_declared_types=(("id", "integer"),),
        ),
        ResourceSchema(
            resource_id="resource-quoted",
            source_id="source-postgres",
            name="Foo",
            aliases=("public.Foo",),
            columns=("id",),
            resource_kind="table",
            column_declared_types=(("id", "integer"),),
        ),
    )

    result = validate_postgresql_read(
        'SELECT foo.id FROM public.foo JOIN public."Foo" ON foo.id = "Foo".id',
        source_id="source-postgres",
        resources=case_resources,
        allowed_resource_ids=("resource-lower",),
    )

    assert result.valid is False
    assert result.issue_codes == ("resource_out_of_scope",)


def test_quoted_table_name_may_contain_a_dot() -> None:
    result = validate_postgresql_read(
        'SELECT id FROM public."daily.orders"',
        source_id="source-postgres",
        resources=(
            ResourceSchema(
                resource_id="resource-dotted",
                source_id="source-postgres",
                name="daily.orders",
                aliases=("public.daily.orders",),
                columns=("id",),
                resource_kind="table",
                column_declared_types=(("id", "integer"),),
            ),
        ),
    )

    assert result.valid is True
    assert result.resource_ids == ("resource-dotted",)


def test_quoted_cte_name_cannot_hide_an_unquoted_resource_reference(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_postgresql_read(
        'WITH "X" AS (SELECT id FROM orders) SELECT id FROM x',
        source_id="source-postgres",
        resources=resources,
    )

    assert result.valid is False
    assert "unknown_resource" in result.issue_codes


@pytest.mark.parametrize(
    "locking_clause",
    ("FOR UPDATE", "FOR SHARE", "FOR NO KEY UPDATE", "FOR KEY SHARE"),
)
def test_postgresql_row_locking_reads_are_rejected_before_execution(
    resources: tuple[ResourceSchema, ...],
    locking_clause: str,
) -> None:
    result = validate_postgresql_read(
        f"SELECT id FROM orders {locking_clause}",
        source_id="source-postgres",
        resources=resources,
    )

    assert result.valid is False
    assert "mutation_not_allowed" in result.issue_codes
    assert result.analysis is not None
    assert result.analysis.mutation_types == ("LOCK",)


def test_postgresql_explain_is_rejected_from_bounded_tabular_execution(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_postgresql_read(
        "EXPLAIN SELECT id FROM public.orders",
        source_id="source-postgres",
        resources=resources,
    )

    assert result.valid is False
    assert "explain_not_allowed" in result.issue_codes


def test_postgresql_tokenization_errors_are_normalized(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_postgresql_read(
        "SELECT '",
        source_id="source-postgres",
        resources=resources,
    )

    assert result.valid is False
    assert result.analysis is None
    assert result.issue_codes == ("sql_parse_error",)


def test_missing_sqlglot_uses_postgresql_extra_hint(
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

    with pytest.raises(ImportError) as caught:
        analyze_postgresql_sql("SELECT 1")
    assert str(caught.value) == (
        "sqlglot is required for PostgreSQL SQL validation. "
        "Install with: pip install 'daita-agents[postgresql]'"
    )


def _allowed_resource(
    *,
    name: str = "allowed",
    kind: str = "table",
) -> ResourceSchema:
    return ResourceSchema(
        resource_id=f"resource-{name}",
        source_id="source-postgres",
        name=name,
        aliases=(f"public.{name}",),
        columns=(("usename",) if name == "pg_user" else ("id", "status")),
        resource_kind=kind,
        column_declared_types=(
            (("usename", "text"),)
            if name == "pg_user"
            else (("id", "integer"), ("status", "text"))
        ),
    )


def test_nested_cte_name_does_not_hide_unrelated_outer_base_table() -> None:
    result = validate_postgresql_read(
        "SELECT allowed.id FROM secret JOIN allowed ON true "
        "WHERE EXISTS (WITH secret AS (SELECT id FROM allowed) "
        "SELECT id FROM secret)",
        source_id="source-postgres",
        resources=(_allowed_resource(),),
    )

    assert result.valid is False
    assert "unknown_resource" in result.issue_codes
    unknown = next(issue for issue in result.issues if issue.code == "unknown_resource")
    assert unknown.details["resource"] == "secret"
    assert result.analysis is not None
    assert any(
        table.name == "secret" and not table.is_cte for table in result.analysis.tables
    )
    assert any(
        table.name == "secret" and table.is_cte for table in result.analysis.tables
    )

    qualified_base = validate_postgresql_read(
        "WITH secret AS (SELECT id FROM public.allowed) "
        "SELECT id FROM public.secret",
        source_id="source-postgres",
        resources=(_allowed_resource(),),
    )
    assert "unknown_resource" in qualified_base.issue_codes


def test_postgresql_rejects_direct_file_and_table_function_calls(
    resources: tuple[ResourceSchema, ...],
) -> None:
    file_read = validate_postgresql_read(
        "SELECT orders.id, pg_read_file('/etc/passwd') FROM orders",
        source_id="source-postgres",
        resources=resources,
    )
    table_call = validate_postgresql_read(
        "SELECT orders.id FROM orders CROSS JOIN LATERAL "
        "dblink('x','SELECT 1') AS t(x int)",
        source_id="source-postgres",
        resources=resources,
    )

    assert "function_not_allowed" in file_read.issue_codes
    assert file_read.analysis is not None
    assert file_read.analysis.function_names == ("PG_READ_FILE",)
    assert "function_not_allowed" in table_call.issue_codes
    assert table_call.analysis is not None
    assert table_call.analysis.table_function_names == ("DBLINK",)


def test_postgresql_function_resolution_is_pg_catalog_only_and_fail_closed(
    resources: tuple[ResourceSchema, ...],
) -> None:
    unqualified_builtin = validate_postgresql_read(
        "SELECT LOWER(status), CAST(id AS TEXT), "
        "ROW_NUMBER() OVER (ORDER BY id) FROM public.orders",
        source_id="source-postgres",
        resources=resources,
    )
    aggregate = validate_postgresql_read(
        "SELECT COUNT(*) FROM public.orders",
        source_id="source-postgres",
        resources=resources,
    )
    qualified_builtin = validate_postgresql_read(
        "SELECT pg_catalog.lower(status) FROM public.orders",
        source_id="source-postgres",
        resources=resources,
    )
    attached_schema_overload = validate_postgresql_read(
        "SELECT public.lower(status) FROM public.orders",
        source_id="source-postgres",
        resources=resources,
    )
    quoted_namespace_spoof = validate_postgresql_read(
        'SELECT "PG_CATALOG".lower(status) FROM public.orders',
        source_id="source-postgres",
        resources=resources,
    )

    assert unqualified_builtin.valid is True
    assert aggregate.valid is True
    assert qualified_builtin.valid is True
    assert "function_not_allowed" in attached_schema_overload.issue_codes
    assert "function_not_allowed" in quoted_namespace_spoof.issue_codes
    assert attached_schema_overload.analysis is not None
    assert attached_schema_overload.analysis.unresolved_function_names == (
        "PUBLIC.LOWER",
    )


def test_structural_case_and_exists_are_not_misclassified_as_functions(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_postgresql_read(
        "SELECT CASE WHEN orders.id IS NULL THEN 0 ELSE 1 END "
        "FROM public.orders AS orders WHERE EXISTS "
        "(SELECT 1 FROM public.orders AS inner_orders "
        "WHERE inner_orders.id = orders.id)",
        source_id="source-postgres",
        resources=resources,
    )

    assert result.valid is True
    assert result.analysis is not None
    assert result.analysis.function_names == ()


def test_context_sensitive_time_expression_is_not_replay_safe(
    resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_postgresql_read(
        "SELECT id, CURRENT_TIMESTAMP FROM public.orders",
        source_id="source-postgres",
        resources=resources,
    )

    assert "function_not_allowed" in result.issue_codes
    assert result.analysis is not None
    assert result.analysis.function_names == ("CURRENT_TIMESTAMP",)


def test_postgresql_requires_exact_schema_and_enforces_only() -> None:
    resource = _allowed_resource(name="pg_user")
    unqualified = validate_postgresql_read(
        "SELECT usename FROM pg_user",
        source_id="source-postgres",
        resources=(resource,),
    )
    qualified = validate_postgresql_read(
        "SELECT usename FROM public.pg_user",
        source_id="source-postgres",
        resources=(resource,),
    )

    assert "schema_qualification_required" in unqualified.issue_codes
    assert qualified.valid is True
    assert qualified.analysis is not None
    assert qualified.analysis.canonical_sql == "SELECT usename FROM ONLY public.pg_user"


def test_postgresql_rejects_views_custom_casts_and_operators() -> None:
    view = validate_postgresql_read(
        "SELECT id FROM public.allowed",
        source_id="source-postgres",
        resources=(_allowed_resource(kind="view"),),
    )
    unsafe_cast = validate_postgresql_read(
        "SELECT CAST(id AS public.evil_type) FROM public.allowed",
        source_id="source-postgres",
        resources=(_allowed_resource(),),
    )
    unsafe_operator = validate_postgresql_read(
        "SELECT id OPERATOR(public.=) 1 FROM public.allowed",
        source_id="source-postgres",
        resources=(_allowed_resource(),),
    )

    assert "resource_kind_not_allowed" in view.issue_codes
    assert "cast_type_not_allowed" in unsafe_cast.issue_codes
    assert "operator_not_allowed" in unsafe_operator.issue_codes
