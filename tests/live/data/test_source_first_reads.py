"""Reads on a real PostgreSQL: the engine resolves names and policy refuses first.

Needs a disposable server whose admin connects without a password:

    DAITA_RUN_POSTGRES_READS=1 DAITA_READS_POSTGRES_PORT=55477 \
    DAITA_READS_POSTGRES_ADMIN=daita_admin pytest tests/live/data/test_source_first_reads.py
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, PostgreSQLSource
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import ModelProfile, ToolCall
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from tests.catalog._schema_support import _OBSERVED_AT
from tests.support.capability_runtime import execute_projected
from tests.support.workspace import workspace_for

pytestmark = [
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_READS") != "1",
        reason="Set DAITA_RUN_POSTGRES_READS=1 with a disposable PostgreSQL server.",
    ),
]

_PORT = int(os.environ.get("DAITA_READS_POSTGRES_PORT", "55477"))
_ADMIN = os.environ.get("DAITA_READS_POSTGRES_ADMIN", "daita_admin")
_DATABASE = "daita_source_first_reads"
_SEED = """
CREATE ROLE daita_reader LOGIN;
CREATE SCHEMA sales;
CREATE SCHEMA legacy;
CREATE SCHEMA events;
CREATE SCHEMA private;
CREATE TABLE sales.customers (id integer PRIMARY KEY, region text);
INSERT INTO sales.customers VALUES (1, 'north'), (2, 'south'), (3, 'north');
CREATE TABLE legacy.customers (id integer PRIMARY KEY, region text);
INSERT INTO legacy.customers VALUES (9, 'old');
CREATE TABLE sales.orders (id integer, customer_id integer, total numeric, note text);
INSERT INTO sales.orders VALUES (1, 1, 10.5, 'a'), (2, 2, 20, 'b');
CREATE TABLE sales.denied (id integer);
INSERT INTO sales.denied VALUES (1);
CREATE TABLE events.page_views (viewed_on date NOT NULL, path text)
    PARTITION BY RANGE (viewed_on);
CREATE TABLE events.page_views_2026 PARTITION OF events.page_views
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
INSERT INTO events.page_views VALUES ('2026-03-15', '/a'), ('2026-03-15', '/b');
CREATE TABLE private.salaries (id integer, amount integer);
INSERT INTO private.salaries VALUES (1, 100);
CREATE FUNCTION sales.evil(integer) RETURNS integer LANGUAGE sql AS 'SELECT $1';
CREATE FUNCTION sales.peek(sales.orders) RETURNS text
    LANGUAGE sql AS $$ SELECT 'ran' $$;
CREATE FUNCTION sales."Peek"(sales.orders) RETURNS text
    LANGUAGE sql AS $$ SELECT 'ran' $$;
CREATE FUNCTION sales.peek_default(sales.orders, integer DEFAULT 1) RETURNS text
    LANGUAGE sql AS $$ SELECT 'ran' $$;
CREATE FUNCTION sales.peek_any(anyelement) RETURNS text
    LANGUAGE sql AS $$ SELECT 'ran' $$;
CREATE FUNCTION sales.note(integer) RETURNS text
    LANGUAGE sql AS $$ SELECT 'ran' $$;
CREATE FUNCTION sales.upper(integer) RETURNS text LANGUAGE sql AS 'SELECT $1::text';
CREATE FUNCTION sales."LOWER"(integer) RETURNS text LANGUAGE sql AS 'SELECT $1::text';
GRANT USAGE ON SCHEMA sales, legacy, events TO daita_reader;
GRANT SELECT ON sales.customers, sales.orders, legacy.customers TO daita_reader;
GRANT SELECT ON events.page_views, events.page_views_2026 TO daita_reader;
"""


async def _admin(database: str = "postgres"):
    import asyncpg  # type: ignore[import-untyped]

    return await asyncpg.connect(
        host="127.0.0.1", port=_PORT, user=_ADMIN, database=database, ssl=False
    )


@pytest.fixture
async def database() -> AsyncIterator[None]:
    connection = await _admin()
    try:
        await connection.execute(f"DROP DATABASE IF EXISTS {_DATABASE} WITH (FORCE)")
        await connection.execute("DROP ROLE IF EXISTS daita_reader")
        await connection.execute(f"CREATE DATABASE {_DATABASE}")
        seeded = await _admin(_DATABASE)
        try:
            await seeded.execute(_SEED)
        finally:
            await seeded.close()
        yield
        await connection.execute(f"DROP DATABASE IF EXISTS {_DATABASE} WITH (FORCE)")
        await connection.execute("DROP ROLE IF EXISTS daita_reader")
    finally:
        await connection.close()


class _Reads:
    def __init__(self, agent: Agent, source_id: str) -> None:
        self.agent = agent
        self.source_id = source_id
        self.calls = 0

    async def query(
        self, sql: str, *, parameters: tuple[object, ...] = ()
    ) -> Mapping[str, object]:
        loop = self.agent._embedded._loop
        assert loop is not None
        self.calls += 1
        outcome = await execute_projected(
            cast(CapabilityRuntime, loop._tools),
            RunInput(
                id=f"reads-{self.calls}",
                agent_id=self.agent.id,
                message="read",
                created_at=_OBSERVED_AT,
                conversation_id="reads",
                source_scope_ids=(self.source_id,),
            ),
            (
                ToolCall(
                    id=f"call-{self.calls}",
                    name="data_query",
                    arguments={
                        "source": "Warehouse",
                        "sql": sql,
                        "parameters": list(parameters),
                    },
                ),
            ),
        )
        return cast(Mapping[str, object], outcome.ordered_results[0].output)


async def _attached(
    tmp_path: Path, schemas: tuple[str, ...], *, username: str = "daita_reader"
) -> _Reads:
    provider = MockModelProvider(())
    agent = await Agent.create(
        "source-first-reads",
        root=tmp_path,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=80_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(
        PostgreSQLSource(
            host="127.0.0.1",
            port=_PORT,
            database=_DATABASE,
            username=username,
            schemas=schemas,
            ssl_mode="disable",
            name="Warehouse",
        )
    )
    return _Reads(agent, source.id)


def _rows(output: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    assert "error" not in output, output
    data = cast(Mapping[str, object], output["data"])
    return tuple(
        dict(row) for row in cast(tuple[Mapping[str, object], ...], data["rows"])
    )


def _error(output: Mapping[str, object]) -> Mapping[str, object]:
    assert "error" in output, output
    return cast(Mapping[str, object], output["error"])


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT pg_read_file('/etc/passwd') FROM sales.orders",
        "SELECT pg_catalog.pg_read_file('/etc/passwd') FROM sales.orders",
        "SELECT * FROM sales.orders, pg_ls_dir('/') AS listing",
        "SELECT pg_sleep(1) FROM sales.orders",
        "SELECT pg_advisory_lock(1) FROM sales.orders",
        "SELECT pg_reload_conf() FROM sales.orders",
        "SELECT pg_stat_file('/etc/passwd') FROM sales.orders",
        "SELECT set_config('search_path', 'private', false) FROM sales.orders",
        "SELECT nextval('sales.none') FROM sales.orders",
        "SELECT query_to_xml('select * from private.salaries', true, true, '') "
        "FROM sales.orders",
        "SELECT query_to_xmlschema('select 1', true, true, '') FROM sales.orders",
        "SELECT query_to_xml_and_xmlschema('select 1', true, true, '') "
        "FROM sales.orders",
        "SELECT table_to_xml('private.salaries', true, true, '') FROM sales.orders",
        "SELECT schema_to_xml('private', true, true, '') FROM sales.orders",
        "SELECT database_to_xml(true, true, '') FROM sales.orders",
        "SELECT cursor_to_xml('c', 1, true, true, '') FROM sales.orders",
        "SELECT ts_stat('select 1') FROM sales.orders",
        "SELECT ts_rewrite('a'::tsquery, 'select 1') FROM sales.orders",
        "SELECT sales.evil(1) FROM sales.orders",
        "SELECT evil(1) FROM sales.orders",
        "SELECT * FROM sales.orders, sales.evil(1) AS shown",
        # A same-named overload outside pg_catalog could be the one PostgreSQL picks.
        "SELECT upper(1) FROM sales.orders",
        'SELECT "evil"(1) FROM sales.orders',
        "SELECT 1 OPERATOR(sales.+) 1 FROM sales.orders",
        # Attribute notation calls peek(o), though it parses as a column.
        "SELECT o.peek FROM sales.orders o",
        "SELECT (o).peek FROM sales.orders o",
        "SELECT ((o)).peek FROM sales.orders o",
        "SELECT (o.*).peek FROM sales.orders o",
        "SELECT sales.orders.peek FROM sales.orders",
        'SELECT o."Peek" FROM sales.orders o',
        "SELECT o.peek_default FROM sales.orders o",
        "SELECT o.peek_any FROM sales.orders o",
        "SELECT id FROM sales.orders o WHERE o.peek = 'ran'",
        "SELECT id FROM sales.orders o ORDER BY o.peek",
        "SELECT (ROW(o.id, o.customer_id, o.total, o.note)::sales.orders).peek "
        "FROM sales.orders o",
        "SELECT (SELECT o2 FROM sales.orders o2 LIMIT 1).peek FROM sales.orders",
        # Unparseable, so it is refused rather than run around the function rule.
        "SELECT pg_read_file('/etc/passwd') FROM sales.orders ORDER BY 1 USING <",
    ),
)
async def test_function_corpus_is_refused(
    database: None, tmp_path: Path, sql: str
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        error = _error(await reads.query(sql))
    finally:
        await reads.agent.close()
    assert error["code"] in {"function_not_allowed", "sql_validation_failed"}, error


async def test_a_quoted_name_cannot_reach_a_function_outside_pg_catalog(
    database: None, tmp_path: Path
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        # Canonical SQL runs lower-cased, so sales."LOWER"(integer) is never called.
        error = _error(await reads.query('SELECT "LOWER"(1) FROM sales.orders'))
    finally:
        await reads.agent.close()
    assert error["code"] == "query_invalid", error
    assert cast(Mapping[str, object], error["details"])["sqlstate"] == "42883"


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT lower(region), coalesce(region, ''), greatest(id, 1) "
        "FROM sales.customers",
        "SELECT count(*), date_trunc('day', now()), random() FROM sales.customers",
        "SELECT * FROM sales.customers, generate_series(1, 2) AS g",
        "SELECT pg_catalog.upper(region) FROM sales.customers",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) "
        "SELECT r.n, c.id FROM r, sales.customers c",
        "SELECT DISTINCT ON (region) region, id FROM sales.customers ORDER BY region",
        # A function that cannot take the row leaves o.note an ordinary column.
        "SELECT o.note, o.id FROM sales.orders o",
        "SELECT o.id, (c).region FROM sales.orders o JOIN sales.customers c "
        "ON c.id = o.customer_id",
        "SELECT to_jsonb(o) ->> 'id', o.to_jsonb FROM sales.orders o",
        "SELECT count(*) FILTER (WHERE id > 1) FROM sales.customers",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY id) FROM sales.customers",
        "SELECT region FROM sales.customers GROUP BY GROUPING SETS ((region), ())",
        "SELECT '{\"a\":1}'::jsonb ->> 'a', id::text FROM sales.customers",
        "SELECT extract(year FROM now()), substring(region FROM 1 FOR 1), "
        "trim(region), position('n' IN region) FROM sales.customers",
    ),
)
async def test_ordinary_analytics_pass_the_function_rule(
    database: None, tmp_path: Path, sql: str
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        assert _rows(await reads.query(sql))
    finally:
        await reads.agent.close()


@pytest.mark.parametrize(
    "sql",
    (
        "WITH d AS (DELETE FROM sales.orders RETURNING id) SELECT id FROM d",
        "SELECT * INTO sales.copy FROM sales.orders",
        "SELECT 1 FROM sales.orders; DROP TABLE sales.orders",
        "/* SELECT */ DROP TABLE sales.orders",
        "COPY sales.orders TO '/tmp/x'",
        "SET search_path = private",
        "DO $$ BEGIN DELETE FROM sales.orders; END $$",
        "COMMIT",
        "SELECT id FROM sales.orders FOR UPDATE",
    ),
)
async def test_write_in_read_corpus_is_refused_before_io(
    database: None, tmp_path: Path, sql: str
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        error = _error(await reads.query(sql))
        assert error["code"] == "sql_validation_failed", error
        assert _rows(await reads.query("SELECT count(*) AS n FROM sales.orders")) == (
            {"n": 2},
        )
    finally:
        await reads.agent.close()


async def test_a_partitioned_parent_returns_its_rows(
    database: None, tmp_path: Path
) -> None:
    reads = await _attached(tmp_path, ("events",))
    try:
        assert _rows(
            await reads.query(
                "SELECT count(*) AS views FROM events.page_views "
                "WHERE viewed_on = '2026-03-15'"
            )
        ) == ({"views": 2},)
    finally:
        await reads.agent.close()


async def test_an_unqualified_name_means_the_one_configured_schema(
    database: None, tmp_path: Path
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        assert _rows(await reads.query("SELECT count(*) AS n FROM customers")) == (
            {"n": 3},
        )
    finally:
        await reads.agent.close()


async def test_an_unqualified_name_across_schemas_lists_its_candidates(
    database: None, tmp_path: Path
) -> None:
    reads = await _attached(tmp_path, ("sales", "legacy"))
    try:
        error = _error(await reads.query("SELECT * FROM customers"))
        assert error["code"] == "sql_validation_failed", error
        text = str(error)
        assert "sales.customers" in text and "legacy.customers" in text
        assert "not available for reading" not in text
    finally:
        await reads.agent.close()


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT * FROM private.salaries",
        "SELECT * FROM legacy.customers",
        "SELECT * FROM pg_tables",
        "SELECT * FROM pg_catalog.pg_class",
        "SELECT * FROM information_schema.tables",
        'SELECT * FROM "SALES"."customers"',
        "WITH sales AS (SELECT 1) SELECT * FROM private.salaries",
    ),
)
async def test_a_name_outside_the_scope_is_refused_naming_it(
    database: None, tmp_path: Path, sql: str
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        error = _error(await reads.query(sql))
    finally:
        await reads.agent.close()
    assert error["code"] == "resource_read_not_allowed", error
    assert sql.split(" FROM ")[1].split()[0].strip('"').split('"')[0] in str(error)


async def test_unrelated_ddl_between_questions_has_no_effect(
    database: None, tmp_path: Path
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        sql = "SELECT count(*) AS n FROM sales.customers"
        assert _rows(await reads.query(sql)) == ({"n": 3},)
        admin = await _admin(_DATABASE)
        try:
            await admin.execute("CREATE TABLE sales.added (id integer)")
            await admin.execute("ALTER TABLE sales.orders ADD COLUMN later text")
        finally:
            await admin.close()
        assert _rows(await reads.query(sql)) == ({"n": 3},)
    finally:
        await reads.agent.close()


@pytest.mark.parametrize(
    ("sql", "code", "sqlstate", "shown"),
    (
        (
            "SELECT nope FROM sales.customers",
            "query_invalid",
            "42703",
            'column "nope" does not exist',
        ),
        (
            "SELECT id, regio FROM sales.customers",
            "query_invalid",
            "42703",
            'column "regio" does not exist',
        ),
        (
            "SELECT id FROM sales.customers c, sales.orders o",
            "query_invalid",
            "42702",
            "ambiguous",
        ),
        (
            "SELECT id::text::integer / 0 FROM sales.customers",
            "query_data_error",
            "22012",
            None,
        ),
        (
            "SELECT 'secret-literal'::integer FROM sales.customers",
            "query_data_error",
            "22P02",
            None,
        ),
    ),
)
async def test_server_errors_are_reported_by_category(
    database: None,
    tmp_path: Path,
    sql: str,
    code: str,
    sqlstate: str,
    shown: str | None,
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        error = _error(await reads.query(sql))
    finally:
        await reads.agent.close()
    assert error["code"] == code, error
    assert cast(Mapping[str, object], error["details"])["sqlstate"] == sqlstate
    if shown is None:
        assert "secret-literal" not in str(error)
    else:
        assert shown in str(error)
    assert "password" not in str(error) and "127.0.0.1" not in str(error)


async def test_a_typo_is_not_reported_as_a_permission_denial(
    database: None, tmp_path: Path
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        hint = _error(await reads.query("SELECT regio FROM sales.customers"))
        admin = await _admin(_DATABASE)
        try:
            await admin.execute("REVOKE SELECT ON sales.orders FROM daita_reader")
        finally:
            await admin.close()
        denied = _error(await reads.query("SELECT id FROM sales.orders"))
    finally:
        await reads.agent.close()
    assert hint["code"] == "query_invalid"
    assert "region" in str(hint)
    assert denied["code"] == "not_permitted"
    assert "orders" in str(denied)


async def test_composite_and_range_values_are_projected(
    database: None, tmp_path: Path
) -> None:
    reads = await _attached(tmp_path, ("sales",))
    try:
        rows = _rows(
            await reads.query(
                "SELECT ROW(id, region) AS pair, int4range(id, 5) AS span, "
                "'{\"a\": 1}'::jsonb AS doc FROM sales.customers WHERE id = 1"
            )
        )
        whole = _rows(
            await reads.query("SELECT c FROM sales.customers c WHERE c.id = 1")
        )
    finally:
        await reads.agent.close()
    assert rows[0]["pair"] == (1, "north")
    assert cast(Mapping[str, object], rows[0]["span"])["lower"] == 1
    assert rows[0]["doc"] == '{"a": 1}'
    assert cast(Mapping[str, object], whole[0]["c"])["region"] == "north"
