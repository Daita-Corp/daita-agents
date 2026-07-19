from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from daita.domains.data import (
    ResourceSchema,
    SqlValidationResult,
    postgresql_query_extension_declarations,
    sqlite_query_extension_declarations,
    validate_postgresql_read,
    validate_sqlite_read,
)

Validator = Callable[..., SqlValidationResult]


@pytest.fixture(params=("sqlite", "postgresql"))
def relational_contract(request: pytest.FixtureRequest):
    if request.param == "postgresql":
        return {
            "validator": validate_postgresql_read,
            "placeholder": "$1",
            "source_id": "source-postgresql",
            "alias": "public.orders",
            "table_sql": "public.orders",
        }
    return {
        "validator": validate_sqlite_read,
        "placeholder": "?",
        "source_id": "source-sqlite",
        "alias": "main.orders",
        "table_sql": "orders",
    }


def _resources(contract: dict[str, object]) -> tuple[ResourceSchema, ...]:
    source_id = str(contract["source_id"])
    alias = str(contract["alias"])
    return (
        ResourceSchema(
            resource_id=f"resource-{source_id}",
            source_id=source_id,
            name="orders",
            aliases=(alias,),
            columns=("id", "status"),
            resource_kind=("table" if source_id == "source-postgresql" else None),
            column_declared_types=(
                (("id", "integer"), ("status", "text"))
                if source_id == "source-postgresql"
                else ()
            ),
            revision="sha256:" + "a" * 64,
            source_revision="catalog:sha256:" + "b" * 64,
        ),
    )


def _validate(
    contract: dict[str, object],
    sql: str,
    *,
    parameters: tuple[object, ...] = (),
    allowed_resource_ids: tuple[str, ...] | None = None,
) -> SqlValidationResult:
    validator = cast(Validator, contract["validator"])
    return validator(
        sql,
        source_id=str(contract["source_id"]),
        resources=_resources(contract),
        parameters=parameters,
        allowed_resource_ids=allowed_resource_ids,
    )


def test_relational_read_contract_accepts_one_catalog_grounded_bounded_select(
    relational_contract: dict[str, object],
) -> None:
    result = _validate(
        relational_contract,
        f"SELECT id FROM {relational_contract['table_sql']} "
        f"WHERE status = {relational_contract['placeholder']}",
        parameters=("open",),
    )

    assert result.valid is True
    assert result.resource_ids == (f"resource-{relational_contract['source_id']}",)
    assert result.source_revision == "catalog:sha256:" + "b" * 64


@pytest.mark.parametrize(
    ("sql", "code"),
    (
        ("DELETE FROM orders", "mutation_not_allowed"),
        ("SELECT missing FROM orders", "missing_column"),
        ("SELECT id FROM absent", "unknown_resource"),
        ("SELECT id FROM orders; SELECT id FROM orders", "multiple_statements"),
    ),
)
def test_relational_read_contract_rejects_the_same_unsafe_or_ungrounded_shapes(
    relational_contract: dict[str, object],
    sql: str,
    code: str,
) -> None:
    result = _validate(relational_contract, sql)

    assert result.valid is False
    assert code in result.issue_codes


def test_relational_read_contract_enforces_explicit_resource_scope(
    relational_contract: dict[str, object],
) -> None:
    result = _validate(
        relational_contract,
        f"SELECT id FROM {relational_contract['table_sql']}",
        allowed_resource_ids=(),
    )

    assert result.issue_codes == ("resource_out_of_scope",)


def test_relational_declarations_share_shape_but_not_replay_claims() -> None:
    sqlite = sqlite_query_extension_declarations().capabilities[0]
    postgresql = postgresql_query_extension_declarations().capabilities[0]

    assert sqlite.input_schema == postgresql.input_schema
    assert sqlite.output_schema == postgresql.output_schema
    assert sqlite.access_mode == postgresql.access_mode
    assert sqlite.risk == postgresql.risk
    assert sqlite.side_effecting == postgresql.side_effecting is False
    assert sqlite.idempotent is True
    assert sqlite.replay_safe is True
    # PostgreSQL can execute server-owned policies, operators, and type support
    # routines that are not represented by the current durable catalog.  It is
    # therefore a read capability, but an ambiguous attempt is not auto-replayed.
    assert postgresql.idempotent is False
    assert postgresql.replay_safe is False
