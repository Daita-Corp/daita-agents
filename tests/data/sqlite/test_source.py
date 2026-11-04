import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from daita import Agent, SQLiteSource
from daita.adapters import sqlite_query as sqlite_query_module
from daita.adapters.models import DiscoveryRequest
from daita.adapters.registry import SourceTypeRegistry
from daita.catalog.models import SourceCatalogSnapshot, TabularFacet
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop import LoopExitKind
from daita.security import EmptySecretProvider
from tests.support.source_contract import (
    assert_source_type_contract,
)
from tests.support.workspace import workspace_for


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
        assert Path(f"{database}-wal").exists()

        admitted = sqlite_query_module._regular_unaliased_path(str(database))
        with pytest.raises(sqlite_query_module.SQLiteQueryError) as failure:
            if execution == "query":
                await sqlite_query_module._run_query(
                    admitted,
                    "SELECT value FROM records",
                    (),
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
        _columns, rows = await sqlite_query_module._run_query(
            admitted,
            "SELECT value FROM records",
            (),
            max_rows=10,
            max_bytes=10_000,
        )
        assert rows == ({"value": "admitted"},)
    else:
        content, _columns, row_count = await sqlite_query_module._run_exact_tabular(
            admitted,
            "SELECT value FROM records",
            (),
            format_name="csv",
            xlsx_provenance=None,
            max_rows=10,
            max_columns=10,
            max_bytes=10_000,
            timeout_seconds=5,
        )
        assert row_count == 1
        assert b"admitted" in content
        assert b"replacement" not in content


async def test_default_agent_root_uses_final_daita_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    agent = await Agent.create("default-root", workspace=workspace_for(None))
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
        context_window_tokens=32_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )
    agent = await Agent.create(
        "sales",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database))
    resource_id = (await agent.list_catalog_resources(source_id=source.id))[0].id
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="query-1",
                    name="data_query",
                    arguments={
                        "source_id": source.id,
                        "resource_ids": (resource_id,),
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

    reopened = await Agent.open(
        "sales", root=tmp_path, workspace=workspace_for(tmp_path)
    )
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
        context_window_tokens=32_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "sales",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database))
    resource_id = (await agent.list_catalog_resources(source_id=source.id))[0].id
    provider._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="bad-query",
                    name="data_query",
                    arguments={
                        "source_id": source.id,
                        "resource_ids": (resource_id,),
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


@pytest.mark.contract
async def test_sqlite_source_type_satisfies_the_source_contract(tmp_path) -> None:
    database = tmp_path / "contract.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, total REAL)")

    await assert_source_type_contract(
        SourceTypeRegistry().require("sqlite"),
        SQLiteSource(database, name="Contract"),
        secret_provider=EmptySecretProvider(),
    )


async def _discover_sqlite(database: Path, **limits: Any) -> SourceCatalogSnapshot:
    now = datetime(2026, 11, 2, tzinfo=UTC)
    adapter = await SQLiteSource(database).open(
        agent_id="agent", attached_at=now, clock=lambda: now
    )
    try:
        result = await adapter.discover(
            DiscoveryRequest(
                agent_id="agent",
                source_id=adapter.registration.id,
                sync_id="sync",
                requested_at=now,
                **limits,
            )
        )
    finally:
        await adapter.close()
    return result.snapshot


def _tabular(snapshot: SourceCatalogSnapshot, name: str) -> TabularFacet:
    (resource,) = [item for item in snapshot.resources if item.name == name]
    (facet,) = [item for item in snapshot.facets if item.resource_id == resource.id]
    return TabularFacet.from_payload(facet.payload)


async def test_sqlite_discovery_beyond_the_resource_limit_keeps_the_first_tables(
    tmp_path: Path,
) -> None:
    database = tmp_path / "many.sqlite"
    with sqlite3.connect(database) as connection:
        for index in range(8):
            connection.execute(f"CREATE TABLE t{index} (id INTEGER)")

    whole = await _discover_sqlite(database, max_resources=8)
    cut = await _discover_sqlite(database, max_resources=5)

    assert whole.sync.truncated is False
    assert cut.sync.truncated is True
    assert [item.name for item in cut.resources] == [f"t{index}" for index in range(5)]


async def test_sqlite_wide_table_keeps_its_first_columns_and_a_whole_key_or_none(
    tmp_path: Path,
) -> None:
    database = tmp_path / "wide.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE whole (a INTEGER PRIMARY KEY, b, c, d);
            CREATE TABLE split_last (a, b, c, d, PRIMARY KEY (a, d));
            CREATE TABLE split_first (a, b, c, d, PRIMARY KEY (d, a));
            CREATE INDEX split_last_by_a ON split_last (a);
            CREATE INDEX split_last_by_d ON split_last (d);
            CREATE TABLE child (x, y, FOREIGN KEY (x) REFERENCES whole (a),
                                FOREIGN KEY (y) REFERENCES whole (d));
            """)

    snapshot = await _discover_sqlite(database, max_columns_per_resource=3)

    assert snapshot.sync.truncated is True
    keys = {
        name: [
            column.name
            for column in _tabular(snapshot, name).columns
            if column.primary_key_ordinal
        ]
        for name in ("whole", "split_last", "split_first")
    }
    assert keys == {"whole": ["a"], "split_last": [], "split_first": []}
    split_last = _tabular(snapshot, "split_last")
    assert [column.name for column in split_last.columns] == ["a", "b", "c"]
    # The index and the key on the cut column d would point outside their table.
    assert [index.name for index in split_last.indexes] == ["split_last_by_a"]
    assert [
        pair.source_field
        for relationship in snapshot.relationships
        for pair in relationship.field_pairs
    ] == ["x"]


async def test_sqlite_more_indexes_and_relationships_than_the_limits_keep_the_first(
    tmp_path: Path,
) -> None:
    database = tmp_path / "linked.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE parent (id INTEGER PRIMARY KEY, a, b, c);
            CREATE INDEX by_a ON parent (a);
            CREATE INDEX by_b ON parent (b);
            CREATE INDEX by_c ON parent (c);
            CREATE TABLE x (parent_id REFERENCES parent (id));
            CREATE TABLE y (parent_id REFERENCES parent (id));
            CREATE TABLE z (parent_id REFERENCES parent (id));
            """)

    whole = await _discover_sqlite(database)
    cut = await _discover_sqlite(
        database, max_indexes_per_resource=2, max_relationships=2
    )

    assert whole.sync.truncated is False
    assert cut.sync.truncated is True
    assert [index.name for index in _tabular(cut, "parent").indexes] == ["by_a", "by_b"]
    assert len(cut.relationships) == 2
