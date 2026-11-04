"""Writes on a real PostgreSQL: the four rules, against a real server.

Run what was checked; ask the source; verify the effect; refuse by principle. Preview
runs no write, and apply refuses any change.

Needs a disposable server whose admin connects without a password:

    DAITA_RUN_POSTGRES_WRITES=1 DAITA_WRITES_POSTGRES_PORT=55477 \
    DAITA_WRITES_POSTGRES_ADMIN=daita_admin pytest tests/live/data/test_source_first_writes.py
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from daita import Agent, ApprovalDecision, ApprovalRequest, PostgreSQLSource
from daita.adapters import postgresql_write as native
from daita.adapters.postgresql_write import PostgreSQLWriteBackend, RelationalWriteError
from daita.adapters.registry import SourceOperation
from daita.capabilities import EffectOutcome, ToolExecution
from daita.catalog.protocols import CatalogStoreError
from daita.domains.data.sql import WriteFacts, WritePreview
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
)
from daita.llm.pricing import CostEstimate
from tests.support.workspace import workspace_for

pytestmark = [
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_WRITES") != "1",
        reason="Set DAITA_RUN_POSTGRES_WRITES=1 with a disposable PostgreSQL server.",
    ),
]

_PORT = int(os.environ.get("DAITA_WRITES_POSTGRES_PORT", "55477"))
_ADMIN = os.environ.get("DAITA_WRITES_POSTGRES_ADMIN", "daita_admin")
_DATABASE = "daita_source_first_writes"
_ROLE = "daita_writer"
_SEED = f"""
CREATE ROLE {_ROLE} LOGIN;
CREATE SCHEMA shop;
CREATE TABLE shop.items (
    id bigint PRIMARY KEY,
    name text NOT NULL,
    qty integer NOT NULL DEFAULT 0,
    price numeric(10, 2),
    due date,
    tag text,
    touched_at timestamptz
);
INSERT INTO shop.items (id, name, qty, price) VALUES
    (1, 'one', 1, 1.50), (2, 'two', 2, 2.50), (3, 'three', 3, 3.50);
CREATE TABLE shop.prices (item_id bigint, price numeric(10, 2));
INSERT INTO shop.prices VALUES (1, 9.00), (2, 8.00), (2, 7.00), (3, 6.00), (3, 6.00);
CREATE TABLE shop.incoming (id bigint, name text);
INSERT INTO shop.incoming VALUES (10, 'ten'), (11, 'eleven');
CREATE TABLE shop.audit (note text);
CREATE TABLE shop.audited (id bigint PRIMARY KEY, name text);
INSERT INTO shop.audited VALUES (1, 'a'), (2, 'b');
CREATE FUNCTION shop.record_audit() RETURNS trigger LANGUAGE plpgsql AS
    $$ BEGIN INSERT INTO shop.audit VALUES (TG_OP); RETURN NEW; END $$;
CREATE TRIGGER audited_write BEFORE INSERT OR UPDATE ON shop.audited
    FOR EACH ROW EXECUTE FUNCTION shop.record_audit();
CREATE TABLE shop.skipping (id bigint PRIMARY KEY, name text);
INSERT INTO shop.skipping VALUES (1, 'a'), (2, 'b');
CREATE FUNCTION shop.skip_two() RETURNS trigger LANGUAGE plpgsql AS
    $$ BEGIN IF OLD.id = 2 THEN RETURN NULL; END IF; RETURN NEW; END $$;
CREATE TRIGGER skipping_write BEFORE UPDATE ON shop.skipping
    FOR EACH ROW EXECUTE FUNCTION shop.skip_two();
CREATE TABLE shop.shifting (id bigint PRIMARY KEY, name text);
INSERT INTO shop.shifting VALUES (1, 'a');
CREATE FUNCTION shop.shift_key() RETURNS trigger LANGUAGE plpgsql AS
    $$ BEGIN NEW.id := NEW.id + 100; RETURN NEW; END $$;
CREATE TRIGGER shifting_write BEFORE INSERT OR UPDATE ON shop.shifting
    FOR EACH ROW EXECUTE FUNCTION shop.shift_key();
CREATE TABLE shop.stamped (id bigint PRIMARY KEY, name text, stamped_at timestamptz);
INSERT INTO shop.stamped VALUES (1, 'a', NULL);
CREATE FUNCTION shop.stamp() RETURNS trigger LANGUAGE plpgsql AS
    $$ BEGIN NEW.stamped_at := now(); RETURN NEW; END $$;
CREATE TRIGGER stamped_write BEFORE UPDATE ON shop.stamped
    FOR EACH ROW EXECUTE FUNCTION shop.stamp();
CREATE TABLE shop.keyless (id bigint, name text);
INSERT INTO shop.keyless VALUES (1, 'a'), (2, 'b'), (3, 'c');
CREATE TABLE shop.skus (id bigint PRIMARY KEY, sku text UNIQUE NOT NULL, name text);
INSERT INTO shop.skus VALUES (1, 's1', 'a'), (2, 's2', 'b');
CREATE TABLE shop.generated (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, name text
);
CREATE TABLE shop.parents (id bigint PRIMARY KEY);
INSERT INTO shop.parents VALUES (1), (2);
CREATE TABLE shop.children (
    id bigint PRIMARY KEY,
    parent_id bigint REFERENCES shop.parents (id) ON DELETE CASCADE
);
INSERT INTO shop.children VALUES (1, 1);
CREATE FUNCTION shop.evil(integer) RETURNS text LANGUAGE sql AS 'SELECT ''ran''';
CREATE TABLE shop.secret (id bigint);
CREATE VIEW shop.item_view AS SELECT * FROM shop.items;
GRANT USAGE ON SCHEMA shop TO {_ROLE};
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA shop TO {_ROLE};
GRANT USAGE ON ALL SEQUENCES IN SCHEMA shop TO {_ROLE};
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA shop TO {_ROLE};
"""
_NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)
_ITEM_UPDATES = ("name", "qty", "price", "due", "tag", "touched_at")
_INSERTS = {
    "shop.items": ("id", *_ITEM_UPDATES),
    "shop.parents": ("id",),
    "shop.skus": ("id", "sku", "name"),
    "shop.generated": ("name",),
    "shop.nullable": ("code", "name"),
}
_UPDATES = {
    "shop.items": _ITEM_UPDATES,
    "shop.parents": ("id",),
    "shop.skus": ("id", "sku", "name"),
}


async def _connect(database: str = "postgres") -> Any:
    import asyncpg  # type: ignore[import-untyped]

    return await asyncpg.connect(
        host="127.0.0.1", port=_PORT, user=_ADMIN, database=database, ssl=False
    )


@pytest.fixture
async def admin() -> AsyncIterator[Any]:
    server = await _connect()
    try:
        await server.execute(f"DROP DATABASE IF EXISTS {_DATABASE} WITH (FORCE)")
        await server.execute(f"DROP ROLE IF EXISTS {_ROLE}")
        await server.execute(f"CREATE DATABASE {_DATABASE}")
        connection = await _connect(_DATABASE)
        try:
            await connection.execute(_SEED)
            yield connection
        finally:
            await connection.close()
        await server.execute(f"DROP DATABASE IF EXISTS {_DATABASE} WITH (FORCE)")
        await server.execute(f"DROP ROLE IF EXISTS {_ROLE}")
    finally:
        await server.close()


def _response(*calls: ToolCall, text: str | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=calls,
        text=text,
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
    )


class _Model:
    provider_id = "mock:source-first-writes"
    model_profile = ModelProfile(
        id=provider_id,
        context_window_tokens=64_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )

    def __init__(self) -> None:
        self.steps: list[ModelResponse] = []
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.steps.pop(0)


@dataclass
class _Writes:
    agent: Agent
    model: _Model
    source_id: str
    approvals: list[ApprovalRequest] = field(default_factory=list)
    on_approval: Callable[[], Awaitable[None]] | None = None
    decision: ApprovalDecision = ApprovalDecision.APPROVE
    runs: int = 0

    async def call(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
        *,
        mode: str | None = None,
    ) -> Mapping[str, object]:
        """One scripted run that loads data_write, calls it and reports."""

        self.runs += 1
        arguments: dict[str, object] = {
            "source": "Shop",
            "sql": sql,
            "parameters": list(parameters),
        }
        if mode is not None:
            arguments["mode"] = mode
        self.model.steps = [
            _response(
                ToolCall(
                    f"load-{self.runs}",
                    "toolbox_load",
                    {"tool_names": ["data_write"]},
                )
            ),
            _response(ToolCall(f"write-{self.runs}", "data_write", arguments)),
            _response(text="Done."),
        ]
        await self.agent.run("Write it.")
        request = self.model.requests[-1]
        block = next(
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.call_id == f"write-{self.runs}"
        )
        return cast(Mapping[str, object], block.output)

    def backend(self) -> PostgreSQLWriteBackend:
        return cast(
            PostgreSQLWriteBackend,
            self.agent._embedded._source_backends[SourceOperation.RELATIONAL_WRITE][
                "postgresql"
            ],
        )

    async def checked(
        self, sql: str, parameters: tuple[object, ...] = ()
    ) -> WriteFacts:
        checked = self.backend().describe(
            sql, source_id=self.source_id, parameters=parameters, schemas=("shop",)
        )
        assert isinstance(checked, WriteFacts), checked
        return checked

    async def previewed(self, write: WriteFacts) -> WritePreview:
        return await self.backend().preview(
            agent_id=self.agent.id, source_id=self.source_id, facts=write
        )

    async def read(self, sql: str) -> Mapping[str, object]:
        self.runs += 1
        self.model.steps = [
            _response(
                ToolCall(
                    f"read-{self.runs}", "data_query", {"source": "Shop", "sql": sql}
                )
            ),
            _response(text="Done."),
        ]
        await self.agent.run("Read it.")
        block = next(
            block
            for message in self.model.requests[-1].messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.call_id == f"read-{self.runs}"
        )
        return cast(Mapping[str, object], block.output)


async def _enable(
    writes: _Writes,
    *tables: str,
    operations: tuple[str, ...] = ("insert", "update", "delete"),
) -> None:
    preview = await writes.agent.preview_source_permissions(
        source_id=writes.source_id,
        read_mode="all",
        read_resource_ids=(),
        relational_write_scopes={
            table: {
                "allowed_operations": operations,
                "allowed_insert_columns": (
                    _INSERTS.get(table, ("id", "name"))
                    if "insert" in operations
                    else ()
                ),
                "allowed_update_columns": (
                    _UPDATES.get(table, ("name",)) if "update" in operations else ()
                ),
                "max_rows": 100,
            }
            for table in tables
        },
    )
    await writes.agent.apply_source_permissions(
        source_id=writes.source_id,
        confirmation_fingerprint=preview.confirmation_fingerprint,
    )


@pytest.fixture
async def writes(admin: Any, tmp_path: Path) -> AsyncIterator[_Writes]:
    agent = await _agent(tmp_path, admin)
    yield agent
    await agent.agent.close()


async def _agent(tmp_path: Path, admin: Any, *, enable: bool = True) -> _Writes:
    model = _Model()
    holder: dict[str, _Writes] = {}

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        writes = holder["writes"]
        writes.approvals.append(request)
        if writes.on_approval is not None:
            await writes.on_approval()
        return writes.decision

    agent = await Agent.create(
        "source-first-writes",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        workspace=workspace_for(tmp_path),
        approval_handler=approve,
    )
    source = await agent.attach(
        PostgreSQLSource(
            host="127.0.0.1",
            port=_PORT,
            database=_DATABASE,
            username=_ROLE,
            schemas=("shop",),
            ssl_mode="disable",
            name="Shop",
        )
    )
    writes = _Writes(agent, model, source.id)
    holder["writes"] = writes
    if enable:
        await _enable(
            writes,
            "shop.items",
            "shop.audited",
            "shop.skipping",
            "shop.parents",
            "shop.shifting",
            "shop.stamped",
            "shop.keyless",
            "shop.skus",
            "shop.generated",
        )
    return writes


def _data(output: Mapping[str, object]) -> Mapping[str, object]:
    assert "error" not in output, output
    return cast(Mapping[str, object], output["data"])


def _code(output: Mapping[str, object]) -> str:
    assert "error" in output, output
    return cast(str, cast(Mapping[str, object], output["error"])["code"])


async def test_preview_fires_no_trigger_and_takes_no_row_lock(
    writes: _Writes, admin: Any
) -> None:
    # Another session holds the row, so anything that locks it would time out.
    async with admin.transaction():
        await admin.execute("SELECT 1 FROM shop.audited WHERE id = 1 FOR UPDATE")
        data = _data(
            await writes.call(
                "UPDATE shop.audited SET name = $1 WHERE id = $2",
                ("new", 1),
                mode="preview",
            )
        )
    assert data["affected_rows"] == 1
    assert data["triggers"] == ("audited_write",)
    assert "triggers_not_previewed" in cast(tuple[str, ...], data["warnings"])
    assert await admin.fetchval("SELECT count(*) FROM shop.audit") == 0
    assert await admin.fetchval("SELECT name FROM shop.audited WHERE id = 1") == "a"
    assert not writes.approvals


async def test_planning_a_write_waits_behind_a_table_lock_then_refuses_safely(
    writes: _Writes, admin: Any
) -> None:
    # PostgreSQL takes ROW EXCLUSIVE to plan a write, and EXCLUSIVE blocks that.
    async with admin.transaction():
        await admin.execute("LOCK TABLE shop.audited IN EXCLUSIVE MODE")
        output = await writes.call(
            "UPDATE shop.audited SET name = $1 WHERE id = $2",
            ("new", 1),
            mode="preview",
        )
    assert _code(output) == "write_lock_timeout"
    assert await admin.fetchval("SELECT count(*) FROM shop.audit") == 0


async def test_describing_a_write_holds_no_table_lock_past_describe(
    writes: _Writes, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[str] = []
    backend = writes.backend()
    scan = backend._scan

    async def observed(connection: Any, *args: Any, **kwargs: Any) -> Any:
        held.extend(
            row[0]
            for row in await connection.fetch(
                "SELECT mode FROM pg_catalog.pg_locks WHERE pid = pg_backend_pid() "
                "AND relation = 'shop.items'::regclass"
            )
        )
        return await scan(connection, *args, **kwargs)

    monkeypatch.setattr(backend, "_scan", observed)
    await writes.previewed(
        await writes.checked("UPDATE shop.items SET name = 'z' WHERE id = 1")
    )
    assert "RowExclusiveLock" not in held


async def test_applying_a_trigger_table_fires_the_trigger_after_the_card_warns(
    writes: _Writes, admin: Any
) -> None:
    data = _data(
        await writes.call("UPDATE shop.audited SET name = $1 WHERE id = $2", ("n", 1))
    )
    assert data["outcome"] == "committed" and data["affected_rows"] == 1
    (approval,) = writes.approvals
    preview = cast(Mapping[str, Any], approval.arguments["preview"])
    assert preview["triggers"] == ("audited_write",)
    assert await admin.fetchval("SELECT count(*) FROM shop.audit") == 1


@pytest.mark.parametrize(
    ("sql", "parameters", "counts", "check", "expected"),
    (
        (
            "UPDATE shop.items SET qty = qty + $1, name = $2 WHERE id = $3",
            (10, "uno", 1),
            {"updated_rows": 1},
            "SELECT qty, name FROM shop.items WHERE id = 1",
            (11, "uno"),
        ),
        (
            "DELETE FROM shop.items WHERE qty > $1",
            (1,),
            {"deleted_rows": 2},
            "SELECT count(*), min(id) FROM shop.items",
            (1, 1),
        ),
        (
            "INSERT INTO shop.items (id, name) SELECT id, name FROM shop.incoming",
            (),
            {"inserted_rows": 2},
            "SELECT count(*), max(id) FROM shop.items",
            (5, 11),
        ),
        (
            "INSERT INTO shop.items (id, name, price, due) VALUES "
            "($1, $2, $3, $4), ($5, $6, $7, $8) ON CONFLICT (id) DO UPDATE "
            "SET name = EXCLUDED.name, price = EXCLUDED.price",
            (1, "uno", "9.25", "2026-01-02", 20, "new", 1.5, None),
            {"inserted_rows": 1, "updated_rows": 1},
            "SELECT count(*), (SELECT name FROM shop.items WHERE id = 1), "
            "(SELECT price::text FROM shop.items WHERE id = 1) FROM shop.items",
            (4, "uno", "9.25"),
        ),
        (
            "UPDATE shop.items AS i SET price = p.price FROM shop.prices AS p "
            "WHERE p.item_id = i.id AND i.id = 1",
            (),
            {"updated_rows": 1},
            "SELECT price::text FROM shop.items WHERE id = 1",
            ("9.00",),
        ),
        (
            "DELETE FROM shop.items AS i USING shop.prices AS p "
            "WHERE p.item_id = i.id AND i.id = 2",
            (),
            {"deleted_rows": 1},
            "SELECT count(*) FROM shop.items",
            (2,),
        ),
        (
            "UPDATE shop.items SET touched_at = now(), tag = $1 WHERE id = 1",
            ("stamped",),
            {"updated_rows": 1},
            "SELECT tag, touched_at IS NOT NULL FROM shop.items WHERE id = 1",
            ("stamped", True),
        ),
    ),
)
async def test_an_approved_statement_previews_the_rows_it_then_changes(
    writes: _Writes,
    admin: Any,
    sql: str,
    parameters: tuple[object, ...],
    counts: Mapping[str, int],
    check: str,
    expected: tuple[object, ...],
) -> None:
    before = await admin.fetch("SELECT * FROM shop.items ORDER BY id")
    preview = _data(await writes.call(sql, parameters, mode="preview"))
    assert await admin.fetch("SELECT * FROM shop.items ORDER BY id") == before
    for name, count in counts.items():
        assert preview[name] == count
    assert preview["affected_rows"] == sum(counts.values())
    assert preview["samples"]

    data = _data(await writes.call(sql, parameters))
    assert data["outcome"] == "committed"
    assert data["affected_rows"] == sum(counts.values())
    assert tuple(await admin.fetchrow(check)) == expected
    (approval,) = writes.approvals
    assert approval.capability_id == "data.write"
    (receipt,) = await writes.agent.list_effects()
    assert receipt.outcome is EffectOutcome.SUCCEEDED
    assert receipt.receipt_kind == "data.write"


async def test_parameters_written_without_spaces_bind_in_order(
    writes: _Writes, admin: Any
) -> None:
    data = _data(
        await writes.call(
            "INSERT INTO shop.items (id,name,tag) VALUES ($1,$2,$3),($4,$5,$6)",
            (40, "a", "literal $1,$2", 41, "b", "t"),
        )
    )
    assert data["inserted_rows"] == 2
    assert (
        await admin.fetchval("SELECT tag FROM shop.items WHERE id = 40")
        == "literal $1,$2"
    )


async def test_an_upsert_counts_rows_that_do_nothing_as_skipped(
    writes: _Writes, admin: Any
) -> None:
    data = _data(
        await writes.call(
            "INSERT INTO shop.items (id, name) VALUES (1, 'x'), (30, 'y') "
            "ON CONFLICT (id) DO NOTHING",
            mode="preview",
        )
    )
    assert (data["inserted_rows"], data["skipped_rows"]) == (1, 1)
    assert data["affected_rows"] == 1
    data = _data(
        await writes.call(
            "INSERT INTO shop.items AS i (id, name) VALUES (1, 'x'), (2, 'y') "
            "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name WHERE i.id = 1"
        )
    )
    assert data["affected_rows"] == 1
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 1") == "x"
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 2") == "two"


async def test_update_from_matching_a_row_twice_warns_when_the_value_is_ambiguous(
    writes: _Writes,
) -> None:
    data = _data(
        await writes.call(
            "UPDATE shop.items AS i SET price = p.price FROM shop.prices AS p "
            "WHERE p.item_id = i.id AND i.id = 2"
        )
    )
    assert data["affected_rows"] == 1
    preview = cast(Mapping[str, Any], writes.approvals[0].arguments["preview"])
    assert "row_matched_more_than_once" in preview["warnings"]
    # Two matches with one value are one row, and PostgreSQL updates it once.
    data = _data(
        await writes.call(
            "UPDATE shop.items AS i SET price = p.price FROM shop.prices AS p "
            "WHERE p.item_id = i.id AND i.id = 3"
        )
    )
    assert data["affected_rows"] == 1
    preview = cast(Mapping[str, Any], writes.approvals[1].arguments["preview"])
    assert "row_matched_more_than_once" not in preview["warnings"]


async def test_a_concurrent_change_keeping_the_row_count_refuses_during_approval(
    writes: _Writes, admin: Any
) -> None:
    async def change() -> None:
        await admin.execute("UPDATE shop.items SET tag = 'other' WHERE id = 1")

    writes.on_approval = change
    output = await writes.call("UPDATE shop.items SET qty = 7 WHERE id IN (1, 2)")
    assert _code(output) == "state_changed"
    assert await admin.fetchval("SELECT qty FROM shop.items WHERE id = 1") == 1
    assert await writes.agent.list_effects() == ()


async def _execute(writes: _Writes, write: WriteFacts, preview: WritePreview) -> Any:
    return await writes.backend().apply(
        agent_id=writes.agent.id,
        source_id=writes.source_id,
        execution=_execution(writes, write),
        facts=write,
        preview_fingerprint=preview.preview_fingerprint,
        expected_affected_rows=preview.affected_rows,
    )


def _execution(writes: _Writes, write: WriteFacts) -> ToolExecution:
    return cast(
        ToolExecution,
        type(
            "Execution",
            (),
            {
                "effect_receipt_id": "effect-receipt:sha256:" + "1" * 64,
                "capability_id": "data.write",
                "arguments": {"_write": write.to_mapping()},
            },
        )(),
    )


async def test_apply_refuses_a_same_count_change_after_the_last_recheck(
    writes: _Writes, admin: Any
) -> None:
    write = await writes.checked("UPDATE shop.items SET qty = 7 WHERE id IN (1, 2)")
    preview = await writes.previewed(write)
    await admin.execute("UPDATE shop.items SET tag = 'other' WHERE id = 1")
    with pytest.raises(RelationalWriteError) as refused:
        await _execute(writes, write, preview)
    assert refused.value.error_code == "write_state_changed"
    assert refused.value.effect_observation is not None
    assert refused.value.effect_observation.outcome is EffectOutcome.NOT_APPLIED
    assert await admin.fetchval("SELECT qty FROM shop.items WHERE id = 1") == 1


@pytest.mark.parametrize(
    ("sql", "check", "expected"),
    (
        # A trigger suppresses one row: the count alone differs.
        (
            "UPDATE shop.skipping SET name = 'z' WHERE id IN (1, 2)",
            "SELECT name FROM shop.skipping WHERE id = 1",
            "a",
        ),
        # A trigger sends the row somewhere else: the count stays the same.
        (
            "UPDATE shop.shifting SET name = 'z' WHERE id = 1",
            "SELECT name FROM shop.shifting WHERE id = 1",
            "a",
        ),
        (
            "INSERT INTO shop.shifting (id, name) VALUES (5, 'z')",
            "SELECT count(*)::text FROM shop.shifting",
            "1",
        ),
        # A CTE stands in for the target, so the preview reads the wrong rows.
        (
            "WITH items AS (SELECT 9::bigint AS id, '(0,1)'::tid AS ctid, "
            "'1'::xid AS xmin) INSERT INTO items (id, name) VALUES (2, 'a'), (9, 'b') "
            "ON CONFLICT (id) DO NOTHING",
            "SELECT count(*)::text FROM shop.items",
            "3",
        ),
    ),
)
async def test_rows_that_differ_from_the_card_roll_back(
    writes: _Writes, admin: Any, sql: str, check: str, expected: str
) -> None:
    assert _code(await writes.call(sql)) == "write_affected_rows_mismatch"
    assert await admin.fetchval(check) == expected
    (receipt,) = await writes.agent.list_effects()
    assert receipt.outcome is EffectOutcome.NOT_APPLIED


async def test_an_upsert_the_server_says_was_not_an_insert_rolls_back(
    writes: _Writes, admin: Any
) -> None:
    if int(await admin.fetchval("SHOW server_version_num")) < 150000:
        pytest.skip("NULLS NOT DISTINCT needs PostgreSQL 15")
    await admin.execute(
        "CREATE TABLE shop.nullable (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
        " code text UNIQUE NULLS NOT DISTINCT, name text)"
    )
    await admin.execute(f"GRANT ALL ON shop.nullable TO {_ROLE}")
    await admin.execute("INSERT INTO shop.nullable (code, name) VALUES (NULL, 'old')")
    await _enable(writes, "shop.nullable")
    # NULL = NULL finds no row to the preview's join; the unique index does.
    output = await writes.call(
        "INSERT INTO shop.nullable (code, name) VALUES (NULL, 'new') "
        "ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name"
    )
    assert _code(output) == "write_affected_rows_mismatch"
    assert await admin.fetchval("SELECT name FROM shop.nullable") == "old"


async def test_a_trigger_that_keeps_the_key_does_not_stop_the_write(
    writes: _Writes, admin: Any
) -> None:
    data = _data(await writes.call("UPDATE shop.stamped SET name = 'z' WHERE id = 1"))
    assert data["outcome"] == "committed"
    assert await admin.fetchval(
        "SELECT stamped_at IS NOT NULL FROM shop.stamped WHERE id = 1"
    )


@pytest.mark.parametrize(
    ("sql", "level", "check", "expected"),
    (
        # Each statement names the key it checks, or drops to the count, on the card.
        (
            "UPDATE shop.skus SET name = 'z' WHERE id = 1",
            "rows",
            "SELECT name FROM shop.skus WHERE id = 1",
            "z",
        ),
        (
            "UPDATE shop.skus SET id = id + 100 WHERE id = 1",
            "rows",
            "SELECT id::text FROM shop.skus WHERE sku = 's1'",
            "101",
        ),
        (
            "UPDATE shop.skus SET id = id + 100, sku = sku || 'x' WHERE id = 1",
            "count",
            "SELECT sku FROM shop.skus WHERE id = 101",
            "s1x",
        ),
        (
            "UPDATE shop.keyless SET name = 'z' WHERE id = 1",
            "count",
            "SELECT name FROM shop.keyless WHERE id = 1",
            "z",
        ),
        (
            "DELETE FROM shop.keyless WHERE id <= 2",
            "rows",
            "SELECT count(*)::text FROM shop.keyless",
            "1",
        ),
        (
            "INSERT INTO shop.keyless (id, name) VALUES (9, 'n')",
            "count",
            "SELECT name FROM shop.keyless WHERE id = 9",
            "n",
        ),
        (
            "INSERT INTO shop.generated (name) VALUES ('n')",
            "count",
            "SELECT name FROM shop.generated",
            "n",
        ),
        (
            "INSERT INTO shop.skus (id, sku, name) VALUES (1, 's1', 'u'), (7, 's7', 'u') "
            "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            "rows",
            "SELECT count(*)::text FROM shop.skus WHERE name = 'u'",
            "2",
        ),
    ),
)
async def test_the_card_says_how_far_the_write_is_verified_and_it_commits(
    writes: _Writes,
    admin: Any,
    sql: str,
    level: str,
    check: str,
    expected: str,
) -> None:
    data = _data(await writes.call(sql))
    assert data["outcome"] == "committed"
    (approval,) = writes.approvals
    preview = cast(Mapping[str, Any], approval.arguments["preview"])
    assert preview["verification"] == level
    assert await admin.fetchval(check) == expected


@pytest.mark.parametrize(
    "sql",
    (
        "UPDATE shop.items SET touched_at = now() WHERE id = 1",
        "UPDATE shop.items SET touched_at = clock_timestamp() WHERE id = 1",
        "UPDATE shop.items SET qty = (random() * 100)::int WHERE id = 1",
        "UPDATE shop.items SET touched_at = statement_timestamp() WHERE id = 1",
        "INSERT INTO shop.items (id, name, tag) VALUES (1, 'x', now()::text) "
        "ON CONFLICT (id) DO UPDATE SET tag = EXCLUDED.tag",
    ),
)
async def test_time_values_and_random_values_warn_and_commit(
    writes: _Writes, sql: str
) -> None:
    assert _data(await writes.call(sql))["outcome"] == "committed"
    preview = cast(Mapping[str, Any], writes.approvals[0].arguments["preview"])
    assert "values_can_change" in preview["warnings"]
    assert "(computed when written)" in repr(preview["samples"])


async def test_rows_chosen_at_random_cannot_commit_unless_they_are_the_same_rows(
    writes: _Writes, admin: Any
) -> None:
    await admin.execute(
        "INSERT INTO shop.items (id, name) "
        "SELECT g, 'bulk' FROM generate_series(100, 180) AS g"
    )
    write = await writes.checked(
        "DELETE FROM shop.items WHERE id IN "
        "(SELECT id FROM shop.items ORDER BY random() LIMIT 1)"
    )
    preview = await writes.previewed(write)
    assert "values_can_change" in preview.warnings
    outcomes = [
        await asyncio.gather(_execute(writes, write, preview), return_exceptions=True)
        for _ in range(5)
    ]
    assert any(
        isinstance(item, RelationalWriteError)
        and item.error_code in {"write_state_changed", "write_affected_rows_mismatch"}
        for (item,) in outcomes
    )


async def test_a_temp_table_shadowing_the_target_is_not_the_target(
    writes: _Writes, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only the source can tell: the parser sees the name the scope allows.
    connect = native._connect

    async def shadowing(*args: Any, **kwargs: Any) -> Any:
        connection = await connect(*args, **kwargs)
        await connection.execute("CREATE TEMP TABLE items (id bigint, name text)")
        return connection

    monkeypatch.setattr(native, "_connect", shadowing)
    write = await writes.checked("UPDATE items SET name = 'z' WHERE id = 1")
    with pytest.raises(RelationalWriteError) as refused:
        await writes.previewed(write)
    assert refused.value.error_code == "write_target_invalid"


async def test_a_dropped_column_between_preview_and_apply_refuses(
    writes: _Writes, admin: Any
) -> None:
    write = await writes.checked("UPDATE shop.items SET name = 'z' WHERE id = 1")
    preview = await writes.previewed(write)
    await admin.execute("DROP VIEW shop.item_view")
    await admin.execute("ALTER TABLE shop.items DROP COLUMN tag")
    with pytest.raises(RelationalWriteError) as refused:
        await _execute(writes, write, preview)
    assert refused.value.error_code == "write_state_changed"
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 1") == "one"


async def test_a_held_row_lock_times_out_without_writing(
    writes: _Writes, admin: Any
) -> None:
    write = await writes.checked("UPDATE shop.items SET name = 'z' WHERE id = 1")
    preview = await writes.previewed(write)
    async with admin.transaction():
        await admin.execute("SELECT 1 FROM shop.items WHERE id = 1 FOR UPDATE")
        with pytest.raises(RelationalWriteError) as refused:
            await _execute(writes, write, preview)
    assert refused.value.error_code == "write_lock_timeout"
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 1") == "one"


async def test_a_write_after_unrelated_ddl_succeeds(
    writes: _Writes, admin: Any
) -> None:
    await admin.execute("CREATE TABLE shop.brand_new (id integer)")
    data = _data(await writes.call("UPDATE shop.items SET name = 'z' WHERE id = 1"))
    assert data["outcome"] == "committed"


async def test_a_write_above_the_row_limit_refuses_at_preview(
    writes: _Writes, admin: Any
) -> None:
    await admin.execute(
        "INSERT INTO shop.items (id, name) "
        "SELECT g, 'bulk' FROM generate_series(100, 250) AS g"
    )
    output = await writes.call("DELETE FROM shop.items WHERE name = 'bulk'")
    assert _code(output) == "write_row_limit"
    assert not writes.approvals
    assert await admin.fetchval("SELECT count(*) FROM shop.items") == 154


async def test_where_true_and_cascades_are_called_out_on_the_card(
    writes: _Writes, admin: Any
) -> None:
    data = _data(await writes.call("DELETE FROM shop.parents WHERE true"))
    assert data["affected_rows"] == 2
    preview = cast(Mapping[str, Any], writes.approvals[0].arguments["preview"])
    assert preview["cascades"] == ("shop.children",)
    assert "where_matches_every_row" in preview["warnings"]
    assert "cascades_not_previewed" in preview["warnings"]
    assert await admin.fetchval("SELECT count(*) FROM shop.children") == 0


async def test_a_denied_write_changes_nothing_and_leaves_no_receipt(
    writes: _Writes, admin: Any
) -> None:
    writes.decision = ApprovalDecision.DENY
    output = await writes.call("UPDATE shop.items SET name = 'z' WHERE id = 1")
    assert _code(output) == "approval_denied"
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 1") == "one"
    assert await writes.agent.list_effects() == ()


@pytest.mark.parametrize(
    ("sql", "code"),
    (
        (
            "UPDATE shop.items SET name = pg_read_file('/etc/passwd') WHERE id = 1",
            "function_not_allowed",
        ),
        ("UPDATE shop.items SET name = evil(1) WHERE id = 1", "function_not_allowed"),
        (
            "UPDATE shop.items SET name = 'a' WHERE pg_sleep(1) IS NULL",
            "function_not_allowed",
        ),
        (
            "INSERT INTO shop.items (id, name) SELECT 1, set_config('a.b', 'c', false)",
            "function_not_allowed",
        ),
        ("UPDATE shop.items SET missing = 1 WHERE id = 1", "update_column_not_allowed"),
        (
            "UPDATE shop.item_view SET name = 'a' WHERE id = 1",
            "resource_write_not_allowed",
        ),
        # Every one of these is the source's own answer, through one generic error.
        ("UPDATE shop.items SET name = 'a' WHERE nope = 1", "write_preview_failed"),
        ("DELETE FROM shop.items WHERE id = 1 RETURNING id", "write_preview_failed"),
        ("DELETE FROM shop.items WHERE id > 0 LIMIT 1", "write_preview_failed"),
        ("UPDATE shop.items SET name = 'a' ORDER BY id", "write_preview_failed"),
        ("INSERT INTO shop.items (id, name) VALUES (1)", "write_preview_failed"),
        (
            "INSERT INTO shop.items (id, name) VALUES (70, DEFAULT)",
            "write_preview_failed",
        ),
        ("UPDATE shop.items SET name = $1 WHERE id = 1", "write_preview_failed"),
    ),
)
async def test_a_refused_write_reaches_no_approval_and_changes_nothing(
    writes: _Writes, admin: Any, sql: str, code: str
) -> None:
    before = await admin.fetch("SELECT * FROM shop.items ORDER BY id")
    assert _code(await writes.call(sql)) == code
    assert await admin.fetch("SELECT * FROM shop.items ORDER BY id") == before
    assert not writes.approvals


async def test_an_update_with_no_where_is_called_out_on_the_card(
    writes: _Writes, admin: Any
) -> None:
    data = _data(await writes.call("UPDATE shop.items SET tag = 'all'"))
    assert data["affected_rows"] == 3
    preview = cast(Mapping[str, Any], writes.approvals[0].arguments["preview"])
    assert "where_matches_every_row" in preview["warnings"]
    assert (
        await admin.fetchval("SELECT count(*) FROM shop.items WHERE tag = 'all'") == 3
    )


async def test_results_and_previews_name_tables_never_ids(writes: _Writes) -> None:
    await writes.call("UPDATE shop.items SET name = 'z' WHERE id = 1", mode="preview")
    request = writes.model.requests[-1]
    block = next(
        block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id.startswith("write-")
    )
    rendered = model_tool_result(block, None, request.scope_names)
    assert "catalog-resource:" not in repr(rendered)
    assert "source:sha256:" not in repr(rendered)
    data = cast(Mapping[str, Any], rendered["output"])["data"]
    assert data["table"] == "shop.items"
    assert data["source"] == "Shop"


async def test_writes_need_no_catalog_entry_to_enable_or_run(
    tmp_path: Path, admin: Any
) -> None:
    writes = await _agent(tmp_path, admin, enable=False)
    try:
        await admin.execute(
            "CREATE TABLE shop.after_attach (id bigint PRIMARY KEY, name text)"
        )
        await admin.execute(f"GRANT ALL ON shop.after_attach TO {_ROLE}")
        await admin.execute("INSERT INTO shop.after_attach VALUES (1, 'a')")
        await _enable(writes, "shop.after_attach", operations=("update",))
        preview = await writes.agent.inspect_source_permissions(writes.source_id)
        (scope,) = preview.state.relational_write_scopes
        assert scope.table == "shop.after_attach"
        data = _data(
            await writes.call("UPDATE shop.after_attach SET name = 'b' WHERE id = 1")
        )
        assert data["outcome"] == "committed"
        assert (
            await admin.fetchval("SELECT name FROM shop.after_attach WHERE id = 1")
            == "b"
        )
    finally:
        await writes.agent.close()


async def test_a_wiped_catalog_still_writes(
    writes: _Writes, admin: Any, tmp_path: Path
) -> None:
    with sqlite3.connect(next(tmp_path.rglob("state.db"))) as connection:
        connection.execute("DELETE FROM snapshots")
    data = _data(await writes.call("UPDATE shop.items SET name = 'z' WHERE id = 1"))
    assert data["outcome"] == "committed"
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 1") == "z"


async def test_writes_succeed_with_the_catalog_store_failing(
    writes: _Writes, admin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(*args: object, **kwargs: object) -> object:
        raise CatalogStoreError("catalog store is down")

    store = writes.agent._embedded._store
    for name in (
        "list_resources",
        "load_current_snapshot",
        "list_current_snapshot_refs",
    ):
        if hasattr(type(store), name):
            monkeypatch.setattr(type(store), name, failing)
    data = _data(await writes.call("UPDATE shop.items SET name = 'z' WHERE id = 1"))
    assert data["outcome"] == "committed"
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 1") == "z"


async def test_a_revoked_scope_refuses_at_dispatch(writes: _Writes, admin: Any) -> None:
    write = await writes.checked("UPDATE shop.items SET name = 'z' WHERE id = 1")
    preview = await writes.previewed(write)
    await _enable(writes, "shop.audited")
    with pytest.raises(RelationalWriteError) as refused:
        await _execute(writes, write, preview)
    assert refused.value.error_code == "resource_write_not_allowed"
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 1") == "one"


async def test_concurrent_writes_to_one_row_commit_once(
    writes: _Writes, admin: Any
) -> None:
    write = await writes.checked("UPDATE shop.items SET qty = qty + 1 WHERE id = 1")
    preview = await writes.previewed(write)
    results = await asyncio.gather(
        _execute(writes, write, preview),
        _execute(writes, write, preview),
        return_exceptions=True,
    )
    assert sum(not isinstance(item, BaseException) for item in results) == 1
    assert await admin.fetchval("SELECT qty FROM shop.items WHERE id = 1") == 2


async def test_an_escape_string_stores_exactly_what_was_written(
    writes: _Writes, admin: Any
) -> None:
    for number, literal in enumerate(
        (r"E'a\nb'", r"E'\t'", r"E'it\'s'", "E'plain'", r"'a\b'", r"$$a\b$$")
    ):
        expected = await admin.fetchval(f"SELECT ({literal})::text")
        await writes.call(f"UPDATE shop.items SET tag = {literal} WHERE id = 1")
        stored = await admin.fetchval("SELECT tag FROM shop.items WHERE id = 1")
        assert stored == expected, (number, literal)


async def test_a_disguised_escape_string_stays_a_literal_in_the_preview(
    writes: _Writes,
) -> None:
    # Rendered from a parse tree, the first string closes early and the rest is SQL.
    data = _data(
        await writes.call(
            r"UPDATE shop.items SET name = 'x' WHERE name = E'\\' "
            "OR name = ' OR id > 0 --'",
            mode="preview",
        )
    )
    assert data["affected_rows"] == 0


async def test_an_escape_string_with_a_backslash_applies_as_written(
    writes: _Writes, admin: Any
) -> None:
    sql = r"UPDATE shop.items SET name = E'\\d+' WHERE id = 1"
    assert _data(await writes.call(sql))["sql"] == sql
    assert await admin.fetchval("SELECT name FROM shop.items WHERE id = 1") == r"\d+"


@pytest.mark.parametrize(
    ("sql", "expected"),
    (
        (r"SELECT regexp_replace('a12', E'\\d+', 'N') AS v", "aN"),
        ("SELECT 'x' AS v -- trailing comment", "x"),
        ("SELECT 'x' AS v /* trailing comment */ ;", "x"),
    ),
)
async def test_a_read_runs_as_written_inside_its_bounding_wrapper(
    writes: _Writes, sql: str, expected: str
) -> None:
    data = _data(await writes.read(sql))
    (row,) = cast(tuple[Mapping[str, object], ...], data["rows"])
    assert row["v"] == expected


async def test_a_renamed_call_still_reaches_the_function_rule(
    writes: _Writes, admin: Any
) -> None:
    await admin.execute(
        "CREATE FUNCTION shop.ceiling(text) RETURNS text LANGUAGE sql "
        "AS $f$SELECT 'ran'$f$"
    )
    await admin.execute(f"GRANT EXECUTE ON FUNCTION shop.ceiling(text) TO {_ROLE}")
    assert _code(await writes.read("SELECT ceiling('x') AS v")) == (
        "function_not_allowed"
    )


async def test_a_plain_backslash_stays_literal_when_the_server_default_is_off(
    writes: _Writes, admin: Any
) -> None:
    await admin.execute(f"ALTER ROLE {_ROLE} SET standard_conforming_strings = off")
    await writes.call(r"UPDATE shop.items SET tag = 'a\nb' WHERE id = 1")
    assert await admin.fetchval("SELECT tag FROM shop.items WHERE id = 1") == r"a\nb"
