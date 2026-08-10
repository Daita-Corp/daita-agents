from __future__ import annotations

from dataclasses import replace

import pytest

from daita.domains.data.sql import (
    POSTGRESQL_UPDATE_MAX_ASSIGNMENTS,
    PostgreSQLUpdateIntent,
    ResourceSchema,
    render_postgresql_update_statement,
    validate_postgresql_update_intent,
)

SOURCE_ID = "source:sha256:" + "1" * 64
RESOURCE_ID = "catalog-resource:sha256:" + "2" * 64
SOURCE_REVISION = "catalog:sha256:" + "3" * 64
RESOURCE_REVISION = "sha256:" + "4" * 64


def _resource(**changes: object) -> ResourceSchema:
    resource = ResourceSchema(
        resource_id=RESOURCE_ID,
        source_id=SOURCE_ID,
        name="accounts",
        aliases=("public.accounts",),
        columns=(
            "account_id",
            "status",
            "closed_at",
            "score",
            "metadata",
            "identity_value",
            "generated_value",
            "blocked_value",
        ),
        revision=RESOURCE_REVISION,
        source_revision=SOURCE_REVISION,
        resource_kind="table",
        writable=True,
        primary_key_columns=("account_id",),
        column_nullability=(
            ("account_id", False),
            ("status", False),
            ("closed_at", True),
            ("score", True),
            ("metadata", True),
            ("identity_value", False),
            ("generated_value", True),
            ("blocked_value", True),
        ),
        column_type_provenance=(
            ("account_id", "pg_catalog", "int8"),
            ("status", "pg_catalog", "text"),
            ("closed_at", "pg_catalog", "timestamptz"),
            ("score", "pg_catalog", "numeric"),
            ("metadata", "pg_catalog", "jsonb"),
            ("identity_value", "pg_catalog", "int8"),
            ("generated_value", "pg_catalog", "text"),
            ("blocked_value", "pg_catalog", "text"),
        ),
        identity_columns=("identity_value",),
        generated_columns=("generated_value",),
        updatable_columns=("status", "closed_at", "score", "metadata"),
    )
    return replace(resource, **changes)  # type: ignore[arg-type]


def _intent(
    *,
    match: object | None = None,
    assignments: object | None = None,
    source_id: str = SOURCE_ID,
    resource_id: str = RESOURCE_ID,
) -> PostgreSQLUpdateIntent:
    return PostgreSQLUpdateIntent.from_mapping(
        {
            "source_id": source_id,
            "resource_id": resource_id,
            "match": (
                [{"column": "account_id", "value": 42}] if match is None else match
            ),
            "assignments": (
                [{"column": "status", "value": "inactive"}]
                if assignments is None
                else assignments
            ),
        }
    )


def _validate(
    intent: PostgreSQLUpdateIntent,
    resource: ResourceSchema | None = None,
):
    return validate_postgresql_update_intent(
        intent,
        resources=((resource or _resource()),),
    )


def test_exact_single_primary_key_intent_is_canonical_and_stable() -> None:
    first = _validate(
        _intent(
            assignments=[
                {"column": "metadata", "value": {"reason": "closed"}},
                {"column": "status", "value": "inactive"},
            ]
        )
    )
    second = _validate(
        _intent(
            assignments=[
                {"column": "status", "value": "inactive"},
                {"column": "metadata", "value": {"reason": "closed"}},
            ]
        )
    )

    assert first.valid is True
    assert first.validated is not None
    assert second.validated is not None
    assert tuple(cell.column for cell in first.validated.assignments) == (
        "status",
        "metadata",
    )
    assert first.validated.intent_sha256 == second.validated.intent_sha256
    assert first.validated == second.validated


@pytest.mark.parametrize(
    ("match", "expected"),
    (
        ([], "write_match_invalid"),
        (
            [
                {"column": "account_id", "value": 42},
                {"column": "status", "value": "inactive"},
            ],
            "write_match_invalid",
        ),
        (
            [
                {"column": "account_id", "value": 42},
                {"column": "account_id", "value": 43},
            ],
            "write_match_invalid",
        ),
        ([{"column": "status", "value": "active"}], "write_match_invalid"),
        ([{"column": "account_id", "value": None}], "write_match_invalid"),
        ([{"column": "account_id", "value": "42"}], "write_match_invalid"),
    ),
)
def test_match_requires_the_exact_complete_ordered_primary_key(
    match: object,
    expected: str,
) -> None:
    result = _validate(_intent(match=match))
    assert result.valid is False
    assert expected in result.issue_codes


def test_missing_or_composite_primary_key_is_rejected() -> None:
    missing = _validate(_intent(), _resource(primary_key_columns=()))
    composite = _validate(
        _intent(),
        _resource(primary_key_columns=("account_id", "status")),
    )
    assert missing.issue_codes == ("write_primary_key_required",)
    assert composite.issue_codes == ("write_primary_key_required",)


@pytest.mark.parametrize(
    ("assignments", "expected"),
    (
        ([], "write_assignment_invalid"),
        (
            [
                {"column": "status", "value": "active"},
                {"column": "status", "value": "inactive"},
            ],
            "write_assignment_invalid",
        ),
        ([{"column": "missing", "value": "x"}], "write_assignment_invalid"),
        ([{"column": "account_id", "value": 43}], "write_assignment_invalid"),
        ([{"column": "identity_value", "value": 1}], "write_assignment_invalid"),
        ([{"column": "generated_value", "value": "x"}], "write_assignment_invalid"),
        ([{"column": "blocked_value", "value": "x"}], "write_assignment_invalid"),
        ([{"column": "status", "value": None}], "write_assignment_invalid"),
        (
            [{"column": "status", "value": {"expression": "now()"}}],
            "write_assignment_invalid",
        ),
    ),
)
def test_assignment_columns_and_values_fail_closed(
    assignments: object,
    expected: str,
) -> None:
    result = _validate(_intent(assignments=assignments))
    assert result.valid is False
    assert expected in result.issue_codes


def test_assignment_count_and_canonical_byte_bounds_are_enforced() -> None:
    oversized_count = [
        {"column": f"field_{index}", "value": index}
        for index in range(POSTGRESQL_UPDATE_MAX_ASSIGNMENTS + 1)
    ]
    count_result = _validate(_intent(assignments=oversized_count))
    byte_result = _validate(
        _intent(assignments=[{"column": "status", "value": "x" * 70_000}])
    )
    assert count_result.issue_codes == ("write_assignment_invalid",)
    assert byte_result.issue_codes == ("write_assignment_invalid",)


@pytest.mark.parametrize(
    ("type_name", "value"),
    (
        ("bool", True),
        ("int2", -32_768),
        ("int4", 2_147_483_647),
        ("int8", 9_223_372_036_854_775_807),
        ("float4", 1.25),
        ("float8", -1.25),
        ("numeric", 42.5),
        ("text", "active"),
        ("varchar", "active"),
        ("bpchar", "A"),
        ("uuid", "123e4567-e89b-12d3-a456-426614174000"),
        ("date", "2026-08-08"),
        ("timestamp", "2026-08-08T18:00:00"),
        ("timestamptz", "2026-08-08T18:00:00Z"),
        ("json", {"state": "active"}),
        ("jsonb", ["active", 1]),
    ),
)
def test_exact_supported_pg_catalog_type_matrix(type_name: str, value: object) -> None:
    resource = _resource(
        column_type_provenance=tuple(
            (
                column,
                namespace,
                (type_name if column == "status" else current_name),
            )
            for column, namespace, current_name in _resource().column_type_provenance
        )
    )
    result = _validate(
        _intent(assignments=[{"column": "status", "value": value}]),
        resource,
    )
    assert result.valid is True


@pytest.mark.parametrize(
    ("namespace", "type_name", "value"),
    (
        ("pg_catalog", "int2", 32_768),
        ("pg_catalog", "int4", True),
        ("pg_catalog", "int8", "42"),
        ("pg_catalog", "bool", 1),
        ("pg_catalog", "float4", 4e38),
        ("pg_catalog", "uuid", "not-a-uuid"),
        ("pg_catalog", "date", "08/08/2026"),
        ("pg_catalog", "timestamp", "2026-08-08T18:00:00Z"),
        ("pg_catalog", "timestamptz", "2026-08-08T18:00:00"),
        ("pg_catalog", "bytea", "abc"),
        ("custom", "text", "abc"),
    ),
)
def test_unsupported_or_incompatible_type_values_are_rejected(
    namespace: str,
    type_name: str,
    value: object,
) -> None:
    resource = _resource(
        column_type_provenance=tuple(
            (
                column,
                (namespace if column == "status" else current_namespace),
                (type_name if column == "status" else current_name),
            )
            for column, current_namespace, current_name in _resource().column_type_provenance
        )
    )
    result = _validate(
        _intent(assignments=[{"column": "status", "value": value}]),
        resource,
    )
    assert result.valid is False
    assert "write_assignment_invalid" in result.issue_codes


def test_resource_identity_revisions_and_structural_facts_are_required() -> None:
    wrong_source = _validate(
        _intent(), _resource(source_id="source:sha256:" + "9" * 64)
    )
    stale_resource = _validate(_intent(), _resource(revision=None))
    stale_source = _validate(_intent(), _resource(source_revision=None))
    non_table = _validate(_intent(), _resource(resource_kind="view", writable=False))
    missing_provenance = _validate(
        _intent(),
        _resource(
            column_type_provenance=tuple(
                item
                for item in _resource().column_type_provenance
                if item[0] != "status"
            )
        ),
    )

    assert wrong_source.issue_codes == ("write_resource_not_writable",)
    assert stale_resource.issue_codes == ("write_resource_not_writable",)
    assert stale_source.issue_codes == ("write_resource_not_writable",)
    assert non_table.issue_codes == ("write_resource_not_writable",)
    assert missing_provenance.issue_codes == ("write_assignment_invalid",)


def test_generated_update_sql_uses_only_quoted_catalog_identifiers_and_parameters() -> (
    None
):
    injected_column = 'status"; DROP TABLE accounts; --'
    resource = replace(
        _resource(),
        columns=("account_id", injected_column),
        column_nullability=(("account_id", False), (injected_column, False)),
        column_type_provenance=(
            ("account_id", "pg_catalog", "int8"),
            (injected_column, "pg_catalog", "text"),
        ),
        identity_columns=(),
        generated_columns=(),
        updatable_columns=(injected_column,),
    )
    value = "inactive'; DELETE FROM audit; --"
    result = _validate(
        _intent(assignments=[{"column": injected_column, "value": value}]),
        resource,
    )
    assert result.validated is not None

    statement = render_postgresql_update_statement(result.validated)

    assert statement.sql == (
        'UPDATE ONLY "public"."accounts" '
        'SET "status""; DROP TABLE accounts; --" = $1 '
        'WHERE "account_id" = $2'
    )
    assert statement.parameters == (value, 42)
    assert value not in statement.sql
    assert "DELETE FROM audit" not in statement.sql
    assert statement.statement_sha256.startswith("sha256:")
