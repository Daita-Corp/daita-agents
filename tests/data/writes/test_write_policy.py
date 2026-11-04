"""What the parser and stored scopes refuse before any source I/O, and what they leave.

What only a server can judge is in ``tests/live/data/test_source_first_writes.py``.
"""

from __future__ import annotations

import pytest

from daita.adapters.postgresql_write_sql import (
    WRITE_MAX_VALUES_ROWS,
    PostgreSQLStatement,
    derived_select,
    describe_write,
)
from daita.catalog.models import ResourceKind, catalog_resource_id
from daita.domains.data.sql import WriteFacts, write_permission_issue
from daita.storage.sqlite_records import RelationalWriteScope

SOURCE_ID = "source:sha256:" + "a" * 64


def _id(table: str) -> str:
    return catalog_resource_id(SOURCE_ID, ResourceKind.TABLE, table)


def _scope(
    table: str,
    operations: tuple[str, ...] = ("insert", "update", "delete"),
    *,
    inserts: tuple[str, ...] = ("id", "name", "qty"),
    updates: tuple[str, ...] = ("name", "qty"),
) -> RelationalWriteScope:
    return RelationalWriteScope(
        agent_id="agent",
        source_id=SOURCE_ID,
        resource_id=_id(table),
        table=table,
        allowed_operations=operations,
        allowed_insert_columns=inserts if "insert" in operations else (),
        allowed_update_columns=updates if "update" in operations else (),
        max_rows=100,
        authorization_fingerprint="sha256:" + "0" * 64,
    )


SCOPES = {_id("sales.items"): _scope("sales.items")}


def _check(
    sql: str,
    parameters: tuple[object, ...] = (),
    *,
    scopes: dict[str, RelationalWriteScope] = SCOPES,
    allowed: frozenset[str] | None = None,
    schemas: tuple[str, ...] = ("sales",),
) -> WriteFacts | tuple[object, ...]:
    facts = describe_write(
        sql,
        source_id=SOURCE_ID,
        parameters=parameters,
        allowed_resource_ids=allowed,
        schemas=schemas,
    )
    if isinstance(facts, tuple):
        return facts
    issue = write_permission_issue(scopes.get(facts.resource_id), facts)
    return facts if issue is None else (issue,)


def _sql(facts: WriteFacts | tuple[object, ...]) -> str:
    assert isinstance(facts, WriteFacts), facts
    return PostgreSQLStatement.of(facts).canonical_sql


def _refusal(*args: object, **kwargs: object) -> str:
    result = _check(*args, **kwargs)  # type: ignore[arg-type]
    assert isinstance(result, tuple), result
    return result[0].code  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("sql", "code"),
    (
        ("CREATE TABLE sales.t (id int)", "write_statement_required"),
        ("DROP TABLE sales.items", "write_statement_required"),
        ("ALTER TABLE sales.items ADD COLUMN x int", "write_statement_required"),
        ("TRUNCATE sales.items", "write_statement_required"),
        ("COPY sales.items FROM '/etc/passwd'", "write_statement_required"),
        (
            "MERGE INTO sales.items USING sales.u ON items.id = u.id "
            "WHEN MATCHED THEN UPDATE SET name = 'a'",
            "write_statement_required",
        ),
        ("SELECT 1", "write_statement_required"),
        ("SET search_path = private", "write_statement_required"),
        ("DO $$ BEGIN END $$", "write_statement_required"),
        ("COMMIT", "write_statement_required"),
        (
            "EXPLAIN ANALYZE DELETE FROM sales.items WHERE id = 1",
            "write_statement_required",
        ),
        (
            "WITH gone AS (DELETE FROM sales.u RETURNING id) "
            "INSERT INTO sales.items (id) SELECT id FROM gone",
            "nested_write_not_allowed",
        ),
        (
            "WITH gone AS (DELETE FROM sales.items WHERE id = 1 RETURNING id) "
            "UPDATE sales.items SET name = 'a' WHERE id IN (SELECT id FROM gone)",
            "nested_write_not_allowed",
        ),
        (
            "UPDATE sales.items SET name = (WITH d AS (DELETE FROM sales.u RETURNING 1) "
            "SELECT 'a') WHERE id = 1",
            "nested_write_not_allowed",
        ),
        (
            "INSERT INTO sales.items (id) SELECT id FROM sales.items FOR UPDATE",
            "nested_write_not_allowed",
        ),
        (
            "UPDATE sales.items SET name = 'a' WHERE id = 1; DELETE FROM sales.items",
            "multiple_statements",
        ),
        (
            "UPDATE sales.items SET name = 'a' WHERE id = 1; DROP TABLE sales.items",
            "multiple_statements",
        ),
        (
            "UPDATE sales.items SET name = 'a' WHERE id = 1 ORDER BY 1 USING <",
            "sql_parse_error",
        ),
        ("UPDATE sales.items SET WHERE", "sql_parse_error"),
        (
            "UPDATE sales.other SET name = 'a' WHERE id = 1",
            "resource_write_not_allowed",
        ),
        ("DELETE FROM private.salaries WHERE id = 1", "schema_not_configured"),
        (
            "UPDATE sales.items SET secret = 'a' WHERE id = 1",
            "update_column_not_allowed",
        ),
        ("UPDATE sales.items SET id = 2 WHERE id = 1", "update_column_not_allowed"),
        ("INSERT INTO sales.items (secret) VALUES (1)", "insert_column_not_allowed"),
        (
            "INSERT INTO sales.items (id) VALUES (1) "
            "ON CONFLICT (id) DO UPDATE SET secret = 'a'",
            "update_column_not_allowed",
        ),
        ("INSERT INTO sales.items VALUES (1, 'a', 2)", "insert_columns_required"),
        ("INSERT INTO sales.items SELECT * FROM sales.u", "insert_columns_required"),
        ("INSERT INTO sales.items DEFAULT VALUES", "insert_columns_required"),
        (
            "INSERT INTO sales.items (id) VALUES (1) ON CONFLICT DO NOTHING",
            "write_preview_failed",
        ),
        (
            "INSERT INTO sales.items (id) VALUES (1) "
            "ON CONFLICT ON CONSTRAINT items_pkey DO NOTHING",
            "write_preview_failed",
        ),
        (
            "INSERT INTO sales.items (id) VALUES (1) "
            "ON CONFLICT (lower(name)) DO NOTHING",
            "write_preview_failed",
        ),
        (
            "INSERT INTO sales.items (id) VALUES (1) "
            "ON CONFLICT (id) WHERE id > 0 DO NOTHING",
            "write_preview_failed",
        ),
        (
            "UPDATE sales.items SET (name, qty) = (SELECT 'a', 1) WHERE id = 1",
            "write_preview_failed",
        ),
        (
            "UPDATE sales.items SET name[1] = 'a' WHERE id = 1",
            "write_preview_failed",
        ),
        (
            "UPDATE sales.items SET name.part = 'a' WHERE id = 1",
            "write_preview_failed",
        ),
        (
            "UPDATE sales.items SET name = pg_read_file('/etc/passwd') WHERE id = 1",
            None,
        ),
        (
            "UPDATE sales.items SET name = sales.evil(1) WHERE id = 1",
            "function_not_allowed",
        ),
        (
            "UPDATE sales.items SET name = query_to_xml('select 1', true, true, '') "
            "WHERE id = 1",
            "function_not_allowed",
        ),
        (
            "DELETE FROM sales.items WHERE id = 1 OPERATOR(sales.+) 1",
            "operator_not_allowed",
        ),
    ),
)
def test_a_statement_outside_the_write_policy_is_refused(
    sql: str, code: str | None
) -> None:
    result = _check(sql)
    if code is None:
        # A volatile built-in is refused live, where the server says what it is.
        assert isinstance(result, WriteFacts)
        assert "PG_READ_FILE" in PostgreSQLStatement.of(result).function_lookups
    else:
        assert isinstance(result, tuple), result
        assert code in {item.code for item in result}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("operation", "sql"),
    (
        ("update", "UPDATE sales.items SET name = 'a' WHERE id = 1"),
        ("delete", "DELETE FROM sales.items WHERE id = 1"),
        ("insert", "INSERT INTO sales.items (id) VALUES (1)"),
    ),
)
def test_each_operation_needs_its_own_permission(operation: str, sql: str) -> None:
    others = tuple(item for item in ("insert", "update", "delete") if item != operation)
    scopes = {_id("sales.items"): _scope("sales.items", others)}
    assert _refusal(sql, scopes=scopes) == "write_operation_not_allowed"


def test_an_upsert_that_updates_needs_update_as_well_as_insert() -> None:
    scopes = {_id("sales.items"): _scope("sales.items", ("insert",))}
    sql = "INSERT INTO sales.items (id, name) VALUES (1, 'a') ON CONFLICT (id) "
    assert (
        _refusal(sql + "DO UPDATE SET name = EXCLUDED.name", scopes=scopes)
        == "write_operation_not_allowed"
    )
    assert isinstance(_check(sql + "DO NOTHING", scopes=scopes), WriteFacts)


@pytest.mark.parametrize(
    "sql",
    (
        "UPDATE sales.items SET name = 'a' FROM sales.hidden WHERE items.id = hidden.id",
        "DELETE FROM sales.items USING sales.hidden WHERE items.id = hidden.id",
        "INSERT INTO sales.items (id) SELECT id FROM sales.hidden",
        "UPDATE sales.items SET name = (SELECT name FROM sales.hidden) WHERE id = 1",
        "UPDATE sales.items SET name = 'a' WHERE id IN (SELECT id FROM SALES.HIDDEN)",
        "INSERT INTO sales.items (id) SELECT oid FROM pg_catalog.pg_class",
        "INSERT INTO sales.items (id) SELECT 1 FROM private.salaries",
    ),
)
def test_every_other_relation_passes_the_read_name_scope(sql: str) -> None:
    allowed = frozenset({_id("sales.items")})
    result = _check(sql, allowed=allowed)
    assert isinstance(result, tuple), result
    assert {item.code for item in result} <= {  # type: ignore[attr-defined]
        "resource_out_of_scope",
        "schema_not_configured",
    }


def test_the_target_itself_must_be_readable_in_this_run() -> None:
    result = _check(
        "WITH items AS (SELECT 1) UPDATE sales.items SET name = 'a' WHERE id = 1",
        allowed=frozenset(),
    )
    assert isinstance(result, tuple)
    assert result[0].code == "resource_out_of_scope"  # type: ignore[attr-defined]


def test_an_unqualified_target_with_several_schemas_must_be_qualified() -> None:
    assert (
        _refusal(
            "UPDATE items SET name = 'a' WHERE id = 1", schemas=("sales", "legacy")
        )
        == "unqualified_relation"
    )
    checked = _check("UPDATE items SET name = 'a' WHERE id = 1")
    assert isinstance(checked, WriteFacts)
    assert checked.table == "sales.items"


@pytest.mark.parametrize(
    "sql",
    (
        "UPDATE SALES.ITEMS SET NAME = 'a' WHERE id = 1",
        'UPDATE "sales"."items" SET "name" = \'a\' WHERE id = 1',
    ),
)
def test_names_fold_as_postgresql_folds_them(sql: str) -> None:
    checked = _check(sql)
    assert isinstance(checked, WriteFacts)
    assert checked.resource_id == _id("sales.items")
    assert checked.update_columns == ("name",)


def test_a_quoted_name_of_another_case_is_another_table() -> None:
    assert (
        _refusal("UPDATE sales.\"Items\" SET name = 'a' WHERE id = 1")
        == "resource_write_not_allowed"
    )


def test_the_statement_runs_as_written() -> None:
    sql = "UPDATE sales.items SET name = 'a' /* ; note */ WHERE id = 1"
    assert _sql(_check(sql + " ;")) == sql


def test_a_batch_keeps_its_row_and_byte_limits() -> None:
    rows = ", ".join(["(1)"] * (WRITE_MAX_VALUES_ROWS + 1))
    assert _refusal(f"INSERT INTO sales.items (id) VALUES {rows}") == "write_too_large"
    assert (
        _refusal("INSERT INTO sales.items (name) VALUES ($1)", ("x" * (64 * 1_024),))
        == "write_too_large"
    )


@pytest.mark.parametrize(
    ("clause", "expected"),
    (("", True), ("WHERE true", True), ("WHERE 1 = 1", True), ("WHERE id = 1", False)),
)
def test_a_where_that_matches_every_row_is_called_out(
    clause: str, expected: bool
) -> None:
    checked = _check(f"DELETE FROM sales.items {clause}")
    assert isinstance(checked, WriteFacts)
    assert PostgreSQLStatement.of(checked).matches_all is expected


def test_a_batch_may_write_its_parameters_without_spaces() -> None:
    checked = _check(
        "INSERT INTO sales.items (id, name) VALUES ($1,$2), ($3,$4) "
        "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
        (1, "a", 2, "b"),
    )
    assert isinstance(checked, WriteFacts)
    assert checked.operation == "insert"
    assert PostgreSQLStatement.of(checked).parameters == (1, "a", 2, "b")


@pytest.mark.parametrize(
    "sql",
    (
        r"UPDATE sales.items SET name = E'a\\nb' WHERE id = 1",
        r"DELETE FROM sales.items WHERE name ~ E'\\d+'",
    ),
)
def test_a_backslash_in_an_escape_string_runs_as_written(sql: str) -> None:
    assert _sql(_check(sql)) == sql


def test_the_preview_keeps_an_escape_string_as_written() -> None:
    """Rendered from the tree it would close early, and the rest would run as SQL."""

    literal = r"E'\\'"
    sql = (
        f"UPDATE sales.items SET name = 'x' WHERE name = {literal} "
        "OR name = ', (SELECT secret FROM private.t) --'"
    )
    scan, lock = derived_select(sql, {"name": "text"}, ("id",))
    assert lock is not None
    for text in (scan, lock):
        assert f"name = {literal} OR" in text
        assert "name = ', (SELECT secret FROM private.t) --'" in text


@pytest.mark.parametrize(
    "literal",
    (
        "E'plain'",
        r"E'a\nb'",
        r"E'it\'s'",
        r"'a\b'",
        r"$$a\b$$",
        "$t$it's$t$",
        r"U&'d\0061t'",
    ),
)
def test_strings_that_keep_their_meaning_still_write(literal: str) -> None:
    sql = f"UPDATE sales.items SET name = {literal} WHERE id = 1"
    assert isinstance(_check(sql), WriteFacts)


@pytest.mark.parametrize(
    "sql",
    (
        "DELETE FROM sales.items WHERE id > 0 LIMIT 1",
        "UPDATE sales.items SET name = 'a' WHERE id > 0 ORDER BY id LIMIT 1",
        "INSERT INTO sales.items (id, name) VALUES (1)",
        "INSERT INTO sales.items (id, name) VALUES (1, DEFAULT)",
        "INSERT INTO sales.items (id, id) VALUES (1, 2)",
        "UPDATE sales.items, sales.items AS u SET name = 'a' WHERE TRUE",
        "UPDATE sales.items SET name = $1 WHERE id = 1",
        "UPDATE sales.items SET name = 'a' WHERE __daita_x = 1",
    ),
)
def test_what_only_the_source_can_judge_is_left_to_the_source(sql: str) -> None:
    """Rule 4: no refusal by shape; each of these fails preview on a real server."""

    assert isinstance(_check(sql), WriteFacts)


@pytest.mark.parametrize(
    "sql",
    (
        "UPDATE sales.items SET name = 'a' WHERE random() < 0.5",
        "DELETE FROM sales.items WHERE id IN "
        "(SELECT id FROM sales.items ORDER BY random() LIMIT 1)",
        "DELETE FROM sales.items WHERE id IN "
        "(SELECT id FROM sales.items TABLESAMPLE SYSTEM (10))",
        "INSERT INTO sales.items (id, name) VALUES ((random() * 2)::bigint, 'k') "
        "ON CONFLICT (id) DO UPDATE SET qty = 9",
        "INSERT INTO sales.items (id, name) VALUES (1, now()::text) "
        "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
        "WITH c AS (SELECT now()::text AS t) "
        "UPDATE sales.items SET name = c.t FROM c WHERE id = 1",
        "WITH items AS (SELECT 1) UPDATE sales.items SET name = 'a' WHERE id = 1",
        "WITH items AS (SELECT 9 AS id) DELETE FROM items WHERE id = 9",
    ),
)
def test_a_statement_that_may_not_match_its_preview_fails_verification_not_policy(
    sql: str,
) -> None:
    """Rule 3 catches these on a real server; no parser list has to name them."""

    assert isinstance(_check(sql), WriteFacts)


@pytest.mark.parametrize(
    ("sql", "expected"),
    (
        ("UPDATE sales.items SET name = now()::text WHERE id = 1", True),
        ("UPDATE sales.items SET name = 'a' WHERE ts < CURRENT_TIMESTAMP", True),
        ("UPDATE sales.items SET name = current_date::text WHERE id = 1", True),
        ("UPDATE sales.items SET name = localtimestamp::text WHERE id = 1", True),
        ("UPDATE sales.items SET name = lower('A') WHERE id = 1", False),
    ),
)
def test_the_sql_time_keywords_come_from_the_parser(sql: str, expected: bool) -> None:
    checked = _check(sql)
    assert isinstance(checked, WriteFacts)
    assert PostgreSQLStatement.of(checked).uses_time is expected
