"""data_query reads many tables and several statements within shared bounds."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, SQLiteSource
from daita.adapters.sqlite_query import SQLiteQueryBackend, SQLiteQueryError
from daita.capabilities import CapabilityRegistry, ToolExecution
from daita.capability_runtime import CapabilityRuntime
from daita.domains.data import DATA_QUERY_CAPABILITY_ID
from daita.llm.models import ModelProfile, ToolCall
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from tests.catalog._schema_support import _OBSERVED_AT
from tests.support.capability_runtime import execute_projected
from tests.support.wide_catalog import MANIFEST, TABLE_COUNT, quote, seed_wide_sqlite
from tests.support.workspace import workspace_for


async def _wide_agent(tmp_path: Path) -> tuple[Agent, str, dict[str, str], Path]:
    database = seed_wide_sqlite(tmp_path / "wide.sqlite")
    provider = MockModelProvider(())
    agent = await Agent.create(
        "wide-query",
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
    source = await agent.attach(SQLiteSource(database))
    resources = await agent.list_catalog_resources(source_id=source.id)
    by_table = {
        resource.native_identity.removeprefix("main."): resource.id
        for resource in resources
    }
    return agent, source.id, by_table, database


async def _call(
    agent: Agent,
    source_id: str,
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    loop = agent._embedded._loop
    assert loop is not None
    runtime = cast(CapabilityRuntime, loop._tools)
    run = RunInput(
        id="wide-query-run",
        agent_id=agent.id,
        message="query the wide fixture",
        created_at=_OBSERVED_AT,
        conversation_id="wide-query-conversation",
        source_scope_ids=(source_id,),
    )
    (result,) = await execute_projected(
        runtime,
        run,
        (ToolCall(id="wide-query", name="data_query", arguments=arguments),),
    )
    return cast(Mapping[str, object], result.output)


def _sqlite_backend(agent: Agent) -> SQLiteQueryBackend:
    registry: CapabilityRegistry = agent._embedded._capabilities
    _capability, executor = registry.resolve_execution(DATA_QUERY_CAPABILITY_ID)
    backend = executor._backends["sqlite"]  # type: ignore[attr-defined]
    assert isinstance(backend, SQLiteQueryBackend)
    return backend


def _union_count_sql() -> str:
    return " UNION ALL ".join(
        f"SELECT {index} AS position, count(*) AS row_count "
        f"FROM {quote(str(item['table']))}"
        for index, item in enumerate(MANIFEST)
    )


async def test_one_statement_counts_all_46_tables_including_empty_and_quoted(
    tmp_path: Path,
):
    agent, source_id, by_table, _database = await _wide_agent(tmp_path)
    try:
        output = await _call(
            agent,
            source_id,
            {
                "source_id": source_id,
                "resource_ids": tuple(by_table.values()),
                "sql": _union_count_sql(),
            },
        )
        assert "error" not in output, output
        data = cast(Mapping[str, object], output["data"])
        rows = cast(tuple[Mapping[str, object], ...], data["rows"])
        assert len(rows) == TABLE_COUNT
        assert {int(cast(int, row["position"])): row["row_count"] for row in rows} == {
            index: item["rows"] for index, item in enumerate(MANIFEST)
        }
        assert rows[0]["row_count"] == 0
        touched = cast(tuple[Mapping[str, object], ...], data["tables_touched"])
        assert len(touched) == TABLE_COUNT
        assert {item["name"] for item in touched} >= {
            "empty_ledger",
            'odd "table',
        }
        timing = cast(Mapping[str, object], data["timing"])
        assert timing["statement_timeout_ms"] == 5_000
        assert timing["total_timeout_ms"] == 15_000
        assert isinstance(timing["elapsed_ms"], int)
        assert data["truncated"] is False
    finally:
        await agent.close()


async def test_several_statements_share_one_row_budget_and_report_each(
    tmp_path: Path,
):
    agent, source_id, by_table, _database = await _wide_agent(tmp_path)
    try:
        output = await _call(
            agent,
            source_id,
            {
                "source_id": source_id,
                "resource_ids": (
                    by_table["empty_ledger"],
                    by_table['odd "table'],
                    by_table["table_02"],
                    by_table["table_03"],
                ),
                "statements": (
                    {"sql": 'SELECT count(*) AS n FROM "empty_ledger"'},
                    {
                        "sql": 'SELECT record_id FROM "odd ""table" '
                        "WHERE record_id < ? ORDER BY record_id",
                        "parameters": (3,),
                    },
                    {
                        "sql": "SELECT a.record_id FROM table_02 AS a "
                        "JOIN table_03 AS b ON b.record_id = a.record_id "
                        "ORDER BY a.record_id",
                    },
                    {"sql": 'SELECT count(*) AS n FROM "empty_ledger"'},
                ),
            },
        )
        assert "error" not in output, output
        data = cast(Mapping[str, object], output["data"])
        statements = cast(tuple[Mapping[str, object], ...], data["statements"])
        assert [item["status"] for item in statements] == [
            "completed",
            "completed",
            "completed",
            "completed",
        ]
        rows = [
            cast(tuple[Mapping[str, object], ...], item["rows"]) for item in statements
        ]
        assert [dict(row) for row in rows[0]] == [{"n": 0}]
        assert tuple(dict(row) for row in rows[1]) == (
            {"record_id": 0},
            {"record_id": 1},
            {"record_id": 2},
        )
        assert {
            item["name"]
            for item in cast(tuple[Mapping[str, object], ...], data["tables_touched"])
        } == {
            "empty_ledger",
            'odd "table',
            "table_02",
            "table_03",
        }
        assert data["returned_rows"] == sum(
            cast(int, item["returned_rows"]) for item in statements
        )
        assert "rows" not in data and "columns" not in data
        assert data["row_limit"] == 100

        budget = await _call(
            agent,
            source_id,
            {
                "source_id": source_id,
                "resource_ids": (by_table["table_12"], by_table["table_25"]),
                "statements": (
                    {"sql": "SELECT record_id FROM table_12"},
                    {"sql": "SELECT record_id FROM table_25"},
                    {"sql": "SELECT record_id FROM table_12"},
                ),
            },
        )
        budget_data = cast(Mapping[str, object], budget["data"])
        budget_statements = cast(
            tuple[Mapping[str, object], ...], budget_data["statements"]
        )
        assert budget_data["returned_rows"] == 100
        assert budget_data["truncated"] is True
        assert "row_limit" in cast(tuple[str, ...], budget_data["truncation_reasons"])
        assert dict(budget_statements[-1]) == {
            "index": 2,
            "status": "skipped_budget_spent",
        }
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("arguments", "code"),
    (
        (
            {
                "sql": "SELECT 1 FROM table_02",
                "statements": ({"sql": "SELECT 1 FROM table_02"},),
            },
            "sql_invalid_input",
        ),
        (
            {
                "statements": (
                    {"sql": "SELECT 1 FROM table_02"},
                    {"sql": "DELETE FROM table_03"},
                )
            },
            "sql_validation_failed",
        ),
    ),
)
async def test_statement_validation_fails_closed_before_any_source_read(
    tmp_path: Path,
    arguments: dict[str, object],
    code: str,
):
    agent, source_id, by_table, _database = await _wide_agent(tmp_path)
    try:
        output = await _call(
            agent,
            source_id,
            {
                "source_id": source_id,
                "resource_ids": (by_table["table_02"], by_table["table_03"]),
                **arguments,
            },
        )
        error = cast(Mapping[str, object], output["error"])
        assert error["code"] == code
    finally:
        await agent.close()


async def test_resource_ceiling_is_64_for_foreground_reads(tmp_path: Path):
    agent = await Agent.create(
        "query-ceiling", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        registry: CapabilityRegistry = agent._embedded._capabilities
        _view, capability = registry.resolve_tool("data_query")
        policy = capability.execution_admission_policy
        assert policy is not None
        assert policy.inline_max_targets == 64
        assert policy.graph_max_targets == 16
        properties = cast(Mapping[str, object], capability.input_schema["properties"])
        resource_rule = cast(Mapping[str, object], properties["resource_ids"])
        assert resource_rule["maxItems"] == 64
        statements = cast(Mapping[str, object], properties["statements"])
        assert statements["maxItems"] == 8
    finally:
        await agent.close()


async def test_sqlite_statement_deadline_interrupts_and_releases_the_source(
    tmp_path: Path,
):
    agent, source_id, by_table, database = await _wide_agent(tmp_path)
    try:
        backend = _sqlite_backend(agent)
        slow = (
            "SELECT count(*) FROM table_10 AS a, table_11 AS b, table_12 AS c, "
            "table_13 AS d, table_14 AS e"
        )
        started = asyncio.get_running_loop().time()
        with pytest.raises(SQLiteQueryError) as timed_out:
            await backend.execute_read(
                agent_id=agent.id,
                source_id=source_id,
                sql=slow,
                parameters=(),
                max_rows=10,
                max_bytes=1_024,
                timeout_seconds=0.2,
            )
        assert timed_out.value.code == "sqlite_query_timeout"
        assert asyncio.get_running_loop().time() - started < 5
        # The interrupted read closed its connection: an exclusive writer lock
        # is immediately available.
        with sqlite3.connect(database, timeout=0) as connection:
            connection.execute("BEGIN EXCLUSIVE")
            connection.execute("ROLLBACK")
    finally:
        await agent.close()


async def test_validated_revisions_are_rechecked_before_source_dispatch(
    tmp_path: Path,
):
    agent, source_id, by_table, _database = await _wide_agent(tmp_path)
    try:
        backend = _sqlite_backend(agent)
        with pytest.raises(SQLiteQueryError) as stale:
            await backend.execute_read(
                agent_id=agent.id,
                source_id=source_id,
                sql="SELECT record_id FROM table_02",
                parameters=(),
                max_rows=10,
                max_bytes=1_024,
                expected_resource_revisions=(
                    (by_table["table_02"], "sha256:" + "0" * 64),
                ),
            )
        assert stale.value.code == "query_binding_stale"
    finally:
        await agent.close()


async def test_total_deadline_stops_remaining_statements(tmp_path: Path):
    agent, source_id, by_table, _database = await _wide_agent(tmp_path)
    try:
        registry: CapabilityRegistry = agent._embedded._capabilities
        _capability, executor = registry.resolve_execution(DATA_QUERY_CAPABILITY_ID)
        ticks = iter((0.0, 0.0, 1.0, 20.0, 20.0))
        executor._clock = lambda: next(ticks)  # type: ignore[attr-defined]
        with pytest.raises(Exception) as limited:
            await executor.execute(
                ToolExecution(
                    run_id="deadline-run",
                    call_id="deadline-call",
                    capability_id=DATA_QUERY_CAPABILITY_ID,
                    arguments={
                        "source_id": source_id,
                        "resource_ids": (by_table["table_02"],),
                        "statements": (
                            {"sql": "SELECT record_id FROM table_02", "parameters": ()},
                            {"sql": "SELECT record_id FROM table_02", "parameters": ()},
                        ),
                        "_adapter_id": "sqlite",
                    },
                )
            )
        assert getattr(limited.value, "code", None) == "query_time_limit"
    finally:
        await agent.close()
