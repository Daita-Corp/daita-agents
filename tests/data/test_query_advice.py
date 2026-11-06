"""data_query attaches advisory warnings without ever changing a correct result."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from daita.capabilities import ToolExecution
from daita.domains.data import (
    DATA_QUERY_CAPABILITY_ID,
    DataQueryExecutor,
    SQLiteReadResult,
    project_result_rows,
)
from daita.domains.data.sql import CheckedRead
from tests.support.checked_read import checked_read

JOIN_SQL = (
    "SELECT sum(o.fee) AS total FROM orders AS o "
    "JOIN order_items AS i ON i.order_id = o.order_id"
)
EMPTY_SQL = "SELECT id FROM orders WHERE ordered_at >= '2099-01-01'"
PLAIN_SQL = "SELECT id FROM orders"


class _Backend:
    def __init__(self, rows: tuple[dict[str, object], ...], mode: str = "ok") -> None:
        self.rows = rows
        self.mode = mode
        self.reads: list[str] = []
        self.describes = 0

    async def describe_name(self, **arguments: object) -> Mapping[str, object]:
        self.describes += 1
        if self.mode == "hang":
            await asyncio.sleep(30)
        if self.mode == "raise":
            raise RuntimeError("boom")
        name = str(arguments["name"])
        key = ["order_id"] if name == "orders" else ["item_id"]
        return {"primary_key": key, "unique_keys": []}

    async def execute_read(self, **arguments: object) -> SQLiteReadResult:
        read = arguments["read"]
        assert isinstance(read, CheckedRead)
        self.reads.append(read.canonical_sql)
        probe = len(self.reads) > 1
        if probe and self.mode == "hang":
            await asyncio.sleep(30)
        if probe and self.mode == "raise":
            raise RuntimeError("boom")
        rows = (
            ({"highest": "2026-04-30", "lowest": "2026-01-01"},) if probe else self.rows
        )
        return SQLiteReadResult(
            source_id=str(arguments["source_id"]),
            canonical_sql=read.canonical_sql,
            sql_fingerprint="sha256:" + "5" * 64,
            resource_ids=read.resource_ids,
            resource_names=read.resource_names,
            columns=tuple(rows[0]) if rows else ("id",),
            projection=project_result_rows(
                rows,
                max_rows=int(str(arguments["max_rows"])),
                max_bytes=int(str(arguments["max_bytes"])),
            ),
        )


async def _run(backend: _Backend, sql: str) -> Mapping[str, object]:
    read = checked_read(sql, resource_id="r1", name="orders")
    read = replace(
        read,
        resource_ids=("r1", "r2"),
        resource_names=(("r1", "orders"), ("r2", "order_items")),
    )
    output = await DataQueryExecutor("agent", {"sqlite": cast(Any, backend)}).execute(
        ToolExecution(
            run_id="run",
            call_id="call",
            capability_id=DATA_QUERY_CAPABILITY_ID,
            arguments={
                "source_id": "source",
                "resource_ids": ("r1", "r2"),
                "sql": sql,
                "_adapter_id": "sqlite",
                "_reads": (read.to_mapping(),),
            },
        )
    )
    return dict(output.data)


def _without_volatile(data: Mapping[str, object]) -> dict[str, object]:
    return {k: v for k, v in data.items() if k not in {"warnings", "timing", "read_at"}}


async def test_a_fan_out_aggregate_gets_a_warning() -> None:
    data = await _run(_Backend(({"total": 10},)), JOIN_SQL)
    assert data["warnings"]


async def test_an_empty_result_is_explained_with_exactly_one_extra_read() -> None:
    backend = _Backend(())
    data = await _run(backend, EMPTY_SQL)
    assert data["warnings"]
    assert len(backend.reads) == 2


async def test_a_plain_non_empty_read_does_no_extra_work() -> None:
    backend = _Backend(({"id": 1},))
    data = await _run(backend, PLAIN_SQL)
    assert "warnings" not in data
    assert len(backend.reads) == 1 and backend.describes == 0


async def test_failing_or_hanging_checks_leave_the_result_unchanged(
    monkeypatch,
) -> None:
    import daita.domains.data.capabilities as module

    monkeypatch.setattr(module, "DATA_QUERY_ADVICE_SECONDS", 0.2)
    cases: list[tuple[str, tuple[dict[str, object], ...]]] = [
        (JOIN_SQL, ({"total": 10},)),
        (EMPTY_SQL, ()),
    ]
    for sql, rows in cases:
        baseline = _without_volatile(await _run(_Backend(rows, "off"), sql))
        for mode in ("raise", "hang"):
            started = time.monotonic()
            data = await _run(_Backend(rows, mode), sql)
            assert time.monotonic() - started < 2
            assert "warnings" not in data
            assert _without_volatile(data) == baseline
