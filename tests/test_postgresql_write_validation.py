from __future__ import annotations

from dataclasses import replace

import pytest

from daita.capabilities import render_approval_arguments
from daita.domains.data.sql import (
    PostgreSQLUpdateCommand,
    PostgreSQLUpdateIntent,
    ResourceSchema,
    render_postgresql_update_statement,
    validate_postgresql_update_intent,
    validate_postgresql_update_scope,
)

SOURCE_ID = "source:sha256:" + "1" * 64
RESOURCE_ID = "catalog-resource:sha256:" + "2" * 64
REVISION = "sha256:" + "3" * 64


def _resource(*, primary_key: tuple[str, ...] = ("id",)) -> ResourceSchema:
    return ResourceSchema(
        resource_id=RESOURCE_ID,
        source_id=SOURCE_ID,
        name="tickets",
        columns=("tenant_id", "id", "status", "priority", "assignee"),
        aliases=("support.tickets",),
        revision=REVISION,
        source_revision=REVISION,
        resource_kind="table",
        writable=True,
        primary_key_columns=primary_key,
        column_nullability=(
            ("tenant_id", False),
            ("id", False),
            ("status", False),
            ("priority", False),
            ("assignee", True),
        ),
        column_type_provenance=(
            ("tenant_id", "pg_catalog", "int8"),
            ("id", "pg_catalog", "int8"),
            ("status", "pg_catalog", "text"),
            ("priority", "pg_catalog", "int4"),
            ("assignee", "pg_catalog", "text"),
        ),
        updatable_columns=("status", "priority", "assignee"),
    )


def _intent(
    *,
    where: tuple[dict[str, object], ...] = (
        {"column": "status", "operator": "eq", "value": "open"},
    ),
    assignments: tuple[dict[str, object], ...] = ({"column": "priority", "value": 4},),
) -> PostgreSQLUpdateIntent:
    return PostgreSQLUpdateIntent.from_mapping(
        {
            "source_id": SOURCE_ID,
            "resource_id": RESOURCE_ID,
            "where": where,
            "assignments": assignments,
        }
    )


def _resource_with_payload(type_name: str) -> ResourceSchema:
    resource = _resource()
    return replace(
        resource,
        columns=(*resource.columns, "payload"),
        column_nullability=(*resource.column_nullability, ("payload", True)),
        column_type_provenance=(
            *resource.column_type_provenance,
            ("payload", "pg_catalog", type_name),
        ),
        updatable_columns=(*resource.updatable_columns, "payload"),
    )


def test_one_contract_validates_single_and_bulk_target_selections():
    single = validate_postgresql_update_intent(
        _intent(
            where=({"column": "id", "operator": "eq", "value": 7},),
        ),
        resources=(_resource(),),
    )
    bulk = validate_postgresql_update_intent(
        _intent(
            where=(
                {"column": "status", "operator": "eq", "value": "open"},
                {"column": "priority", "operator": "lte", "value": 2},
            ),
        ),
        resources=(_resource(),),
    )

    assert single.valid
    assert bulk.valid
    assert single.validated is not None
    assert bulk.validated is not None
    assert single.validated.primary_key_columns == ("id",)
    assert bulk.validated.where[0].column == "status"


def test_composite_primary_keys_are_supported_for_target_fingerprinting():
    result = validate_postgresql_update_intent(
        _intent(
            where=(
                {"column": "tenant_id", "operator": "eq", "value": 9},
                {"column": "status", "operator": "in", "value": ["open", "new"]},
            ),
        ),
        resources=(_resource(primary_key=("tenant_id", "id")),),
    )
    assert result.valid
    assert result.validated is not None
    assert result.validated.primary_key_columns == ("tenant_id", "id")


def test_renderer_uses_only_parameterized_catalog_scoped_sql():
    result = validate_postgresql_update_intent(
        _intent(
            where=(
                {"column": "status", "operator": "in", "value": ["open", "new"]},
                {"column": "assignee", "operator": "is_null", "value": None},
            ),
            assignments=(
                {"column": "priority", "value": 4},
                {"column": "assignee", "value": "ops"},
            ),
        ),
        resources=(_resource(),),
    )
    assert result.validated is not None
    statement = render_postgresql_update_statement(result.validated)
    assert statement.sql == (
        'UPDATE ONLY "support"."tickets" SET "priority" = $1, "assignee" = $2 '
        'WHERE ("status" IN ($3, $4)) AND ("assignee" IS NULL)'
    )
    assert statement.parameters == (4, "ops", "open", "new")
    assert statement.selection_where_sql == (
        '("status" IN ($1, $2)) AND ("assignee" IS NULL)'
    )
    assert "open" not in statement.sql


def test_execution_command_freezes_expected_impact_without_a_row_limit():
    command = PostgreSQLUpdateCommand.from_mapping(
        {
            **_intent().to_payload(),
            "preview_fingerprint": "sha256:" + "4" * 64,
            "expected_affected_rows": 2_000_000,
        }
    )
    assert command.expected_affected_rows == 2_000_000


def test_scope_is_cardinality_independent_and_supports_composite_keys():
    result = validate_postgresql_update_scope(
        SOURCE_ID,
        RESOURCE_ID,
        ("priority", "status"),
        resources=(_resource(primary_key=("tenant_id", "id")),),
    )
    assert result.valid
    assert result.validated is not None
    assert result.validated.primary_key_columns == ("tenant_id", "id")


def test_invalid_filters_and_assignments_fail_closed():
    cases = (
        _intent(where=({"column": "missing", "operator": "eq", "value": 1},)),
        _intent(where=({"column": "status", "operator": "in", "value": []},)),
        _intent(where=({"column": "priority", "operator": "eq", "value": "high"},)),
        _intent(assignments=({"column": "id", "value": 4},)),
    )
    codes = tuple(
        validate_postgresql_update_intent(item, resources=(_resource(),)).issue_codes[0]
        for item in cases
    )
    assert codes == (
        "write_filter_invalid",
        "write_filter_invalid",
        "write_filter_invalid",
        "write_assignment_invalid",
    )


@pytest.mark.parametrize("operator", ("eq", "ne", "in", "not_in"))
def test_json_equality_and_set_filters_fail_before_postgresql_io(operator: str):
    value: object = (
        [{"state": "open"}] if operator in {"in", "not_in"} else {"state": "open"}
    )
    result = validate_postgresql_update_intent(
        _intent(where=({"column": "payload", "operator": operator, "value": value},)),
        resources=(_resource_with_payload("json"),),
    )

    assert result.issue_codes == ("write_filter_invalid",)


@pytest.mark.parametrize("operator", ("eq", "ne", "in", "not_in"))
def test_jsonb_equality_and_set_filters_remain_supported(operator: str):
    value: object = (
        [{"state": "open"}] if operator in {"in", "not_in"} else {"state": "open"}
    )
    result = validate_postgresql_update_intent(
        _intent(where=({"column": "payload", "operator": operator, "value": value},)),
        resources=(_resource_with_payload("jsonb"),),
    )

    assert result.valid


@pytest.mark.parametrize("type_name", ("json", "jsonb"))
def test_json_null_filters_remain_supported(type_name: str):
    result = validate_postgresql_update_intent(
        _intent(where=({"column": "payload", "operator": "is_null", "value": None},)),
        resources=(_resource_with_payload(type_name),),
    )

    assert result.valid


def test_every_accepted_update_intent_has_a_reviewable_exact_command():
    accepted = _intent(
        where=(
            {
                "column": "status",
                "operator": "in",
                "value": [f"v{index:05d}" for index in range(3_000)],
            },
        )
    )
    accepted_result = validate_postgresql_update_intent(
        accepted,
        resources=(_resource(),),
    )
    assert accepted_result.valid
    command = PostgreSQLUpdateCommand(
        accepted,
        preview_fingerprint="sha256:" + "4" * 64,
        expected_affected_rows=9_223_372_036_854_775_807,
    )
    assert render_approval_arguments(command.to_payload()) is not None

    unreviewable = _intent(
        where=(
            {
                "column": "status",
                "operator": "in",
                "value": [f"v{index:05d}" for index in range(5_000)],
            },
        )
    )
    rejected = validate_postgresql_update_intent(
        unreviewable,
        resources=(_resource(),),
    )
    assert rejected.issue_codes == ("write_plan_too_large",)


def test_update_requires_a_primary_key_but_not_a_primary_key_filter():
    missing_key = validate_postgresql_update_intent(
        _intent(), resources=(_resource(primary_key=()),)
    )
    assert missing_key.issue_codes == ("write_primary_key_required",)

    bulk = validate_postgresql_update_intent(_intent(), resources=(_resource(),))
    assert bulk.valid
