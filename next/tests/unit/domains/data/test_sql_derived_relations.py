from __future__ import annotations

import pytest

from daita.domains.data import (
    ResourceSchema,
    validate_postgresql_read,
    validate_sqlite_read,
)

_REVISION_A = "sha256:" + "a" * 64
_REVISION_B = "sha256:" + "b" * 64
_SOURCE_REVISION = "catalog:sha256:" + "c" * 64


@pytest.fixture
def sqlite_resources() -> tuple[ResourceSchema, ...]:
    return (
        ResourceSchema(
            resource_id="resource-ledger",
            source_id="source-sqlite",
            name="ledger_entries",
            aliases=("main.ledger_entries",),
            columns=("entry_id", "account_id", "amount_cents", "state"),
            revision=_REVISION_A,
            source_revision=_SOURCE_REVISION,
        ),
        ResourceSchema(
            resource_id="resource-accounts",
            source_id="source-sqlite",
            name="account_directory",
            aliases=("main.account_directory",),
            columns=("account_id", "display_name", "territory"),
            revision=_REVISION_B,
            source_revision=_SOURCE_REVISION,
        ),
    )


@pytest.fixture
def postgresql_resources() -> tuple[ResourceSchema, ...]:
    return (
        ResourceSchema(
            resource_id="resource-events",
            source_id="source-postgresql",
            name="events",
            aliases=("analytics.events",),
            columns=("event_id", "actor_id", "category"),
            resource_kind="table",
            revision=_REVISION_A,
            source_revision=_SOURCE_REVISION,
        ),
        ResourceSchema(
            resource_id="resource-actors",
            source_id="source-postgresql",
            name="ActorDirectory",
            aliases=("analytics.ActorDirectory",),
            columns=("ActorId", "DisplayName"),
            resource_kind="table",
            revision=_REVISION_B,
            source_revision=_SOURCE_REVISION,
        ),
    )


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT slab.key FROM " "(SELECT entry_id AS key FROM ledger_entries) AS slab",
        "SELECT shell.key FROM (SELECT core.key FROM "
        "(SELECT entry_id AS key FROM ledger_entries) AS core) AS shell",
    ),
)
def test_derived_subquery_and_nested_alias_preserve_base_lineage(
    sqlite_resources: tuple[ResourceSchema, ...],
    sql: str,
) -> None:
    result = validate_sqlite_read(
        sql,
        source_id="source-sqlite",
        resources=sqlite_resources,
    )

    assert result.valid is True
    assert result.resource_ids == ("resource-ledger",)
    assert result.resource_revisions == (("resource-ledger", _REVISION_A),)


def test_chained_cte_preserves_every_base_resource_lineage(
    sqlite_resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_sqlite_read(
        """
        WITH filtered_entries AS (
            SELECT entry_id AS key, account_id, amount_cents
            FROM ledger_entries
            WHERE state = ?
        ), enriched_entries AS (
            SELECT f.key, f.amount_cents, d.territory AS area
            FROM filtered_entries AS f
            JOIN account_directory AS d ON d.account_id = f.account_id
        )
        SELECT e.area, SUM(e.amount_cents) AS total_cents
        FROM enriched_entries AS e
        GROUP BY e.area
        """,
        source_id="source-sqlite",
        resources=sqlite_resources,
        parameters=("posted",),
    )

    assert result.valid is True
    assert result.resource_ids == ("resource-ledger", "resource-accounts")
    assert result.resource_revisions == (
        ("resource-accounts", _REVISION_B),
        ("resource-ledger", _REVISION_A),
    )
    assert result.source_revision == _SOURCE_REVISION


def test_set_output_validates_and_rejects_branch_arity_mismatch(
    sqlite_resources: tuple[ResourceSchema, ...],
) -> None:
    valid = validate_sqlite_read(
        "SELECT combined.identity FROM ("
        "SELECT entry_id AS identity FROM ledger_entries "
        "UNION ALL SELECT account_id FROM account_directory"
        ") AS combined",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )
    invalid = validate_sqlite_read(
        "SELECT combined.identity FROM ("
        "SELECT entry_id AS identity FROM ledger_entries "
        "UNION ALL SELECT account_id, territory FROM account_directory"
        ") AS combined",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )

    assert valid.valid is True
    assert valid.resource_ids == ("resource-ledger", "resource-accounts")
    assert invalid.valid is False
    assert "set_projection_arity_mismatch" in invalid.issue_codes


def test_cte_column_list_controls_exposed_names_and_requires_exact_arity(
    sqlite_resources: tuple[ResourceSchema, ...],
) -> None:
    valid = validate_sqlite_read(
        "WITH renamed(item_key, owner_key) AS ("
        "SELECT entry_id, account_id FROM ledger_entries"
        ") SELECT renamed.item_key, renamed.owner_key FROM renamed",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )
    invalid = validate_sqlite_read(
        "WITH renamed(item_key) AS ("
        "SELECT entry_id, account_id FROM ledger_entries"
        ") SELECT renamed.item_key FROM renamed",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )

    assert valid.valid is True
    assert invalid.valid is False
    assert "derived_projection_arity_mismatch" in invalid.issue_codes


@pytest.mark.parametrize(
    "sql",
    (
        "WITH parcel AS (SELECT entry_id AS known FROM ledger_entries) "
        "SELECT parcel.unknown FROM parcel",
        "SELECT parcel.unknown FROM "
        "(SELECT entry_id AS known FROM ledger_entries) AS parcel",
    ),
)
def test_unknown_derived_columns_fail_closed(
    sqlite_resources: tuple[ResourceSchema, ...],
    sql: str,
) -> None:
    result = validate_sqlite_read(
        sql,
        source_id="source-sqlite",
        resources=sqlite_resources,
    )

    assert result.valid is False
    assert result.issue_codes == ("unknown_derived_column",)


def test_duplicate_and_cross_relation_derived_columns_are_ambiguous(
    sqlite_resources: tuple[ResourceSchema, ...],
) -> None:
    duplicate = validate_sqlite_read(
        "SELECT parcel.value FROM ("
        "SELECT entry_id AS value, account_id AS value FROM ledger_entries"
        ") AS parcel",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )
    cross_relation = validate_sqlite_read(
        "WITH left_side AS (SELECT account_id AS key FROM ledger_entries), "
        "right_side AS (SELECT account_id AS key FROM account_directory) "
        "SELECT key FROM left_side JOIN right_side ON true",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )

    assert duplicate.issue_codes == ("ambiguous_column",)
    assert cross_relation.issue_codes == ("ambiguous_column",)


def test_alias_and_cte_shadowing_resolve_only_the_nearest_binding(
    sqlite_resources: tuple[ResourceSchema, ...],
) -> None:
    alias_shadow = validate_sqlite_read(
        "SELECT local_name.entry_id FROM ledger_entries AS local_name "
        "WHERE EXISTS (SELECT 1 FROM account_directory AS local_name "
        "WHERE local_name.entry_id = 1)",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )
    cte_shadow = validate_sqlite_read(
        "WITH visible AS (SELECT entry_id AS outer_key FROM ledger_entries) "
        "SELECT outer_key FROM visible WHERE EXISTS ("
        "WITH visible AS (SELECT display_name AS inner_name FROM account_directory) "
        "SELECT visible.outer_key FROM visible)",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )

    assert alias_shadow.valid is False
    assert alias_shadow.issue_codes == ("missing_column",)
    assert cte_shadow.valid is False
    assert "unknown_derived_column" in cte_shadow.issue_codes


def test_legal_correlation_validates_and_derived_table_scope_escape_fails(
    sqlite_resources: tuple[ResourceSchema, ...],
) -> None:
    correlated = validate_sqlite_read(
        "SELECT entries.entry_id FROM ledger_entries AS entries "
        "WHERE EXISTS (SELECT 1 FROM account_directory AS directory "
        "WHERE directory.account_id = entries.account_id)",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )
    escaped = validate_sqlite_read(
        "SELECT nested.account_id FROM ledger_entries AS outer_entries "
        "JOIN (SELECT directory.account_id FROM account_directory AS directory "
        "WHERE directory.account_id = outer_entries.account_id) AS nested ON true",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )

    assert correlated.valid is True
    assert escaped.valid is False
    assert escaped.issue_codes == ("column_scope_escape",)


def test_recursive_cte_has_one_explicit_fail_closed_code(
    sqlite_resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_sqlite_read(
        "WITH RECURSIVE walking AS ("
        "SELECT entry_id FROM ledger_entries "
        "UNION ALL SELECT entry_id FROM walking"
        ") SELECT * FROM walking",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )

    assert result.valid is False
    assert result.analysis is None
    assert result.issue_codes == ("recursive_cte_not_supported",)


def test_select_alias_positions_follow_sqlite_and_postgresql_rules(
    sqlite_resources: tuple[ResourceSchema, ...],
) -> None:
    sqlite_result = validate_sqlite_read(
        "SELECT state, COUNT(*) AS tally FROM ledger_entries "
        "GROUP BY state HAVING tally > 0 ORDER BY tally",
        source_id="source-sqlite",
        resources=sqlite_resources,
    )
    postgres_order = validate_postgresql_read(
        "SELECT category AS label FROM analytics.events ORDER BY label",
        source_id="source-postgresql",
        resources=(
            ResourceSchema(
                resource_id="resource-events",
                source_id="source-postgresql",
                name="events",
                aliases=("analytics.events",),
                columns=("category",),
                resource_kind="table",
            ),
        ),
    )
    postgres_having = validate_postgresql_read(
        "SELECT category, COUNT(*) AS tally FROM analytics.events "
        "GROUP BY category HAVING tally > 0",
        source_id="source-postgresql",
        resources=(
            ResourceSchema(
                resource_id="resource-events",
                source_id="source-postgresql",
                name="events",
                aliases=("analytics.events",),
                columns=("category",),
                resource_kind="table",
            ),
        ),
    )

    assert sqlite_result.valid is True
    assert postgres_order.valid is True
    assert postgres_having.issue_codes == ("missing_column",)


def test_sqlite_derived_identifiers_fold_ascii_only() -> None:
    resource = ResourceSchema(
        resource_id="resource-unicode",
        source_id="source-sqlite",
        name="glyphs",
        columns=("K", "K"),
    )
    ascii_name = validate_sqlite_read(
        "SELECT projected.k FROM (SELECT K AS k FROM glyphs) AS projected",
        source_id="source-sqlite",
        resources=(resource,),
    )
    unicode_name = validate_sqlite_read(
        'SELECT projected."K" FROM (SELECT "K" FROM glyphs) AS projected',
        source_id="source-sqlite",
        resources=(resource,),
    )

    assert ascii_name.valid is True
    assert unicode_name.valid is True


def test_postgresql_quoted_derived_names_and_nested_aliases_validate(
    postgresql_resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_postgresql_read(
        'WITH "NamedActors"("Key", "Label") AS ('
        'SELECT "ActorId", "DisplayName" FROM analytics."ActorDirectory"'
        ') SELECT wrapper."Label" FROM ('
        'SELECT "Key", "Label" FROM "NamedActors"'
        ") AS wrapper",
        source_id="source-postgresql",
        resources=postgresql_resources,
    )

    assert result.valid is True
    assert result.resource_ids == ("resource-actors",)
    assert result.resource_revisions == (("resource-actors", _REVISION_B),)


def test_postgresql_quoted_derived_name_is_not_unquoted_equivalent(
    postgresql_resources: tuple[ResourceSchema, ...],
) -> None:
    result = validate_postgresql_read(
        'WITH "NamedActors"("Key") AS ('
        'SELECT "ActorId" FROM analytics."ActorDirectory"'
        ') SELECT key FROM "NamedActors"',
        source_id="source-postgresql",
        resources=postgresql_resources,
    )

    assert result.valid is False
    assert result.issue_codes == ("unknown_derived_column",)
