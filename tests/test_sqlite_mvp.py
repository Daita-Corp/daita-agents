import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from daita import Agent, SQLiteSource
from daita.adapters import sqlite_query as sqlite_query_module
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop import LoopExitKind


@pytest.mark.parametrize("execution", ("query", "exact_export"))
async def test_sqlite_execution_fails_closed_when_admitted_path_is_replaced(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    execution: str,
):
    database = tmp_path / f"admitted-{execution}.sqlite"
    replacement = tmp_path / f"replacement-{execution}.sqlite"
    for path, value in ((database, "admitted"), (replacement, "replacement")):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE records (value TEXT)")
            connection.execute("INSERT INTO records VALUES (?)", (value,))
    with sqlite3.connect(database) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    admitted = sqlite_query_module._regular_unaliased_path(str(database))
    original_connect = sqlite_query_module.sqlite3.connect
    replaced = False

    def replace_before_connection(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement.replace(database)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(
        sqlite_query_module.sqlite3,
        "connect",
        replace_before_connection,
    )

    with pytest.raises(
        sqlite_query_module.SQLiteQueryError,
        match="changed after admission",
    ):
        if execution == "query":
            await sqlite_query_module._run_query(
                admitted,
                "SELECT value FROM records",
                (),
                expected_schema_version=schema_version,
                max_rows=10,
                max_bytes=10_000,
            )
        else:
            await sqlite_query_module._run_exact_tabular(
                admitted,
                "SELECT value FROM records",
                (),
                format_name="csv",
                xlsx_provenance=None,
                expected_schema_version=schema_version,
                max_rows=10,
                max_columns=10,
                max_bytes=10_000,
                timeout_seconds=5,
            )


async def test_sqlite_execution_fails_closed_on_symlink_substitution(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "admitted-symlink.sqlite"
    replacement = tmp_path / "replacement-symlink.sqlite"
    for path in (database, replacement):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE records (value TEXT)")
    with sqlite3.connect(database) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    admitted = sqlite_query_module._regular_unaliased_path(str(database))
    original_connect = sqlite_query_module.sqlite3.connect
    substituted = False

    def substitute_before_connection(*args, **kwargs):
        nonlocal substituted
        if not substituted:
            substituted = True
            database.unlink()
            database.symlink_to(replacement)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(
        sqlite_query_module.sqlite3,
        "connect",
        substitute_before_connection,
    )

    with pytest.raises(
        sqlite_query_module.SQLiteQueryError,
        match="changed after admission",
    ):
        await sqlite_query_module._run_query(
            admitted,
            "SELECT value FROM records",
            (),
            expected_schema_version=schema_version,
            max_rows=10,
            max_bytes=10_000,
        )


async def test_sqlite_connect_failure_rechecks_admitted_path_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "admitted-connect-failure.sqlite"
    replacement = tmp_path / "replacement-connect-failure.sqlite"
    for path, value in ((database, "admitted"), (replacement, "replacement")):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE records (value TEXT)")
            connection.execute("INSERT INTO records VALUES (?)", (value,))
    with sqlite3.connect(database) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    admitted = sqlite_query_module._regular_unaliased_path(str(database))

    def replace_then_fail(*args, **kwargs):
        del args, kwargs
        replacement.replace(database)
        raise sqlite3.OperationalError("controlled descriptor open failure")

    monkeypatch.setattr(sqlite_query_module.sqlite3, "connect", replace_then_fail)

    with pytest.raises(
        sqlite_query_module.SQLiteQueryError,
        match="changed after admission",
    ):
        await sqlite_query_module._run_query(
            admitted,
            "SELECT value FROM records",
            (),
            expected_schema_version=schema_version,
            max_rows=10,
            max_bytes=10_000,
        )


@pytest.mark.parametrize("execution", ("query", "exact_export"))
async def test_sqlite_execution_fails_closed_for_live_wal_database(
    tmp_path,
    execution: str,
):
    database = tmp_path / f"live-wal-{execution}.sqlite"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE records (value TEXT)")
        connection.execute("INSERT INTO records VALUES ('visible-in-wal')")
        connection.commit()
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
        assert Path(f"{database}-wal").exists()

        admitted = sqlite_query_module._regular_unaliased_path(str(database))
        with pytest.raises(sqlite_query_module.SQLiteQueryError) as failure:
            if execution == "query":
                await sqlite_query_module._run_query(
                    admitted,
                    "SELECT value FROM records",
                    (),
                    expected_schema_version=schema_version,
                    max_rows=10,
                    max_bytes=10_000,
                )
            else:
                await sqlite_query_module._run_exact_tabular(
                    admitted,
                    "SELECT value FROM records",
                    (),
                    format_name="csv",
                    xlsx_provenance=None,
                    expected_schema_version=schema_version,
                    max_rows=10,
                    max_columns=10,
                    max_bytes=10_000,
                    timeout_seconds=5,
                )
        assert failure.value.code == "source_wal_not_supported"
    finally:
        connection.close()


@pytest.mark.parametrize("execution", ("query", "exact_export"))
async def test_sqlite_execution_remains_bound_during_aba_path_substitution(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    execution: str,
):
    database = tmp_path / f"aba-{execution}.sqlite"
    replacement = tmp_path / f"aba-replacement-{execution}.sqlite"
    admitted_stash = tmp_path / f"aba-admitted-{execution}.sqlite"
    for path, value in ((database, "admitted"), (replacement, "replacement")):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE records (value TEXT)")
            connection.execute("INSERT INTO records VALUES (?)", (value,))
    with sqlite3.connect(database) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    admitted = sqlite_query_module._regular_unaliased_path(str(database))
    original_connect = sqlite_query_module.sqlite3.connect

    def substitute_only_during_connection(*args, **kwargs):
        database.rename(admitted_stash)
        replacement.rename(database)
        try:
            return original_connect(*args, **kwargs)
        finally:
            database.rename(replacement)
            admitted_stash.rename(database)

    monkeypatch.setattr(
        sqlite_query_module.sqlite3,
        "connect",
        substitute_only_during_connection,
    )

    if execution == "query":
        _columns, rows, _revision = await sqlite_query_module._run_query(
            admitted,
            "SELECT value FROM records",
            (),
            expected_schema_version=schema_version,
            max_rows=10,
            max_bytes=10_000,
        )
        assert rows == ({"value": "admitted"},)
    else:
        content, _columns, row_count, _revision = (
            await sqlite_query_module._run_exact_tabular(
                admitted,
                "SELECT value FROM records",
                (),
                format_name="csv",
                xlsx_provenance=None,
                expected_schema_version=schema_version,
                max_rows=10,
                max_columns=10,
                max_bytes=10_000,
                timeout_seconds=5,
            )
        )
        assert row_count == 1
        assert b"admitted" in content
        assert b"replacement" not in content


async def test_default_agent_root_uses_final_daita_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    agent = await Agent.create("default-root")
    try:
        assert agent.home == tmp_path / ".daita" / "agents" / "default-root"
    finally:
        await agent.close()


async def test_public_agent_queries_sqlite_and_reopens_exact_transcript(tmp_path):
    database = tmp_path / "sales.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE sales(product TEXT, revenue INTEGER)")
    connection.executemany(
        "INSERT INTO sales VALUES (?, ?)", (("alpha", 10), ("beta", 20))
    )
    connection.commit()
    connection.close()

    provider = MockModelProvider(())
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=21_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )
    agent = await Agent.create(
        "sales",
        root=tmp_path,
        model=provider,
        model_profile=profile,
    )
    source = await agent.attach(SQLiteSource(database))
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="query-1",
                    name="data_query_sqlite",
                    arguments={
                        "source_id": source.id,
                        "sql": "SELECT product, revenue FROM sales ORDER BY revenue DESC",
                    },
                ),
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            text="beta has the highest revenue.",
        ),
    )

    result = await agent.run("Which product has the highest revenue?")
    transcript = await agent.transcript(result.run_id)
    await agent.close()

    assert result.kind is LoopExitKind.COMPLETED, result.reason
    assert result.final_text == "beta has the highest revenue."
    tool_result = transcript.messages[2].content[0]
    assert isinstance(tool_result, ToolResultBlock)
    data = tool_result.output["data"]
    assert isinstance(data, Mapping)
    rows = data["rows"]
    assert isinstance(rows, tuple)
    assert isinstance(rows[0], Mapping)
    assert rows[0]["product"] == "beta"

    reopened = await Agent.open("sales", root=tmp_path)
    try:
        restored = await reopened.transcript(result.run_id)
        assert restored == transcript
        assert (await reopened.list_sources())[0] == source
        assert len(await reopened.list_catalog_resources()) == 1
    finally:
        await reopened.close()


async def test_invalid_sql_is_a_model_visible_tool_error(tmp_path):
    database = tmp_path / "sales.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE sales(value INTEGER)")
    connection.commit()
    connection.close()
    provider = MockModelProvider(())
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "sales", root=tmp_path, model=provider, model_profile=profile
    )
    source = await agent.attach(SQLiteSource(database))
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="bad-query",
                    name="data_query_sqlite",
                    arguments={
                        "source_id": source.id,
                        "sql": "DELETE FROM sales",
                    },
                ),
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            text="I could not run a write because this agent is read-only.",
        ),
    )
    try:
        result = await agent.run("Delete the rows")
        transcript = await agent.transcript(result.run_id)
        error = transcript.messages[2].content[0]
        assert isinstance(error, ToolResultBlock)
        assert error.is_error is True
        details = error.output["error"]
        assert isinstance(details, Mapping)
        assert details["code"] == "sql_validation_failed"
        assert provider.requests[1].messages[-1] == transcript.messages[2]
    finally:
        await agent.close()
