"""A read goes to the source, so catalog state cannot decide its outcome."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from daita import Agent, SQLiteSource
from daita._json import canonical_json
from daita.adapters.sqlite import SQLiteResourceAdapter
from daita.catalog.protocols import CatalogStoreError
from daita.errors import StateCompatibilityError
from daita.hosting import embedded as embedded_module
from daita.llm.models import FinishReason, ModelResponse, ToolCall, ToolResultBlock
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sqlite import SQLiteStateStore
from tests.support.source_refresh import settled, sync_count
from tests.support.workspace import workspace_for

_QUERY = {"source": "CRM", "sql": "SELECT count(*) AS total FROM companies"}


class _BrokenCatalogStore:
    def __getattr__(self, name: str) -> object:
        raise CatalogStoreError(f"catalog store is unavailable: {name}")


def _break_catalog(agent: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    for owner in (agent._embedded._catalog_service, agent._embedded._data_view):
        monkeypatch.setattr(owner, "_store", _BrokenCatalogStore())


def _create_companies(database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE companies (name TEXT NOT NULL)")
    connection.execute("INSERT INTO companies VALUES ('Acme'), ('Globex')")
    connection.commit()
    connection.close()


async def _create_agent(tmp_path: Path) -> Agent:
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("count", "data_query", _QUERY),),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Two companies."),
        )
    )
    return await Agent.create(
        "reads-without-catalog",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )


async def _select_companies(agent: Agent, source: SQLiteSource) -> None:
    attached = await agent.attach(source)
    companies = next(
        item
        for item in await agent.list_catalog_resources(source_id=attached.id)
        if item.name == "companies"
    )
    preview = await agent.preview_source_permissions(
        source_id=attached.id,
        read_mode="selected",
        read_resource_ids=(companies.id,),
        relational_write_scopes={},
    )
    await agent.apply_source_permissions(
        source_id=attached.id,
        confirmation_fingerprint=preview.confirmation_fingerprint,
    )


def _data(result: ToolResultBlock | None) -> Mapping[str, Any]:
    assert result is not None and not result.is_error, result
    data = result.output["data"]
    assert isinstance(data, Mapping)
    return data


async def _assert_counted_two(agent: Agent, run_id: str) -> None:
    ((_, result),) = await agent.tool_evidence(run_id)
    assert [dict(row) for row in _data(result)["rows"]] == [{"total": 2}]


async def test_read_answers_from_a_table_the_catalog_has_never_seen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "crm.sqlite"
    sqlite3.connect(database).close()
    agent = await _create_agent(tmp_path)
    try:
        await agent.attach(SQLiteSource(database, name="CRM"))
        # Created after attach and never refreshed: the catalog stays empty.
        _create_companies(database)
        result = await agent.run("How many companies are there?")
        await _assert_counted_two(agent, result.run_id)
    finally:
        await agent.close()


async def test_describing_an_uncataloged_table_answers_from_the_source(
    tmp_path: Path,
) -> None:
    database = tmp_path / "companies.db"
    sqlite3.connect(database).close()
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "describe",
                        "catalog_describe",
                        {"source": "companies", "name": "companies"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Not cataloged yet."),
        )
    )
    agent = await Agent.create(
        "describe-without-catalog",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="companies"))
        _create_companies(database)
        run = await agent.run("What columns does companies have?")
        ((_, result),) = await agent.tool_evidence(run.run_id)
        assert result is not None and not result.is_error, result
        data = result.output["data"]
        assert isinstance(data, Mapping)
        assert data["name"] == "main.companies"
        assert data["answered_from"] == "live"
        # The miss refreshes the catalog in the background.
        await settled(agent)
        listed = await agent.list_catalog_resources(source_id=source.id)
        assert [item.name for item in listed] == ["companies"]
    finally:
        await agent.close()


async def test_read_answers_while_the_catalog_store_is_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "crm.sqlite"
    _create_companies(database)
    agent = await _create_agent(tmp_path)
    try:
        await agent.attach(SQLiteSource(database, name="CRM"))
        _break_catalog(agent, monkeypatch)
        result = await agent.run("How many companies are there?")
        await _assert_counted_two(agent, result.run_id)
    finally:
        await agent.close()


async def test_a_selected_scope_holds_while_the_catalog_store_is_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "crm.sqlite"
    _create_companies(database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE secrets (value TEXT)")
    agent = await _create_agent(tmp_path)
    try:
        await _select_companies(agent, SQLiteSource(database, name="CRM"))
        _break_catalog(agent, monkeypatch)
        result = await agent.run("How many companies are there?")
        await _assert_counted_two(agent, result.run_id)
    finally:
        await agent.close()


@pytest.mark.parametrize("stored", ("{not json", '{"unexpected": true}'))
@pytest.mark.parametrize("selected", (False, True))
async def test_a_damaged_catalog_row_does_not_stop_a_run(
    tmp_path: Path, stored: str, selected: bool
) -> None:
    database = tmp_path / "crm.sqlite"
    _create_companies(database)
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("count", "data_query", _QUERY),
                    ToolCall("search", "catalog_search", {"query": "companies"}),
                    ToolCall(
                        "describe",
                        "catalog_describe",
                        {"source": "CRM", "name": "companies"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Two companies."),
        )
    )
    agent = await Agent.create(
        "damaged-catalog",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )
    try:
        source = SQLiteSource(database, name="CRM")
        if selected:
            await _select_companies(agent, source)
        else:
            await agent.attach(source)
        with sqlite3.connect(agent.home / "state.db") as connection:
            connection.execute("UPDATE snapshots SET data = ?", (stored,))
        assert isinstance(agent._embedded._store, SQLiteStateStore)
        agent._embedded._store._decoded_catalog_snapshots.clear()
        run = await agent.run("How many companies are there?")
        assert run.reason == "completed"
        counted, searched, described = (
            _data(result) for _, result in await agent.tool_evidence(run.run_id)
        )
        assert [dict(row) for row in counted["rows"]] == [{"total": 2}]
        assert searched["catalog_available"] is False
        assert [hit["name"] for hit in searched["hits"]] == ["main.companies"]
        assert described["answered_from"] == "live"
    finally:
        await agent.close()
    # Opening the home still fails closed on the damaged row.
    with pytest.raises(StateCompatibilityError):
        await Agent.open(
            "damaged-catalog", root=tmp_path, workspace=workspace_for(tmp_path)
        )


async def test_a_read_needs_no_table_and_reports_the_source_classification(
    tmp_path: Path,
) -> None:
    database = tmp_path / "crm.sqlite"
    sqlite3.connect(database).close()
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "one", "data_query", {"source": "CRM", "sql": "SELECT 1 AS n"}
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="One."),
        )
    )
    agent = await Agent.create(
        "reads-without-tables",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="CRM"))
        result = await agent.run("What is one?")
        ((_, block),) = await agent.tool_evidence(result.run_id)
        assert block is not None and not block.is_error, block
        data = block.output["data"]
        assert isinstance(data, Mapping)
        assert data["resource_ids"] == ()
        assert isinstance(data["read_at"], str)
        assert "resource_revisions" not in data and "source_revision" not in data
        assert block.sensitivity is source.presentation_sensitivity
    finally:
        await agent.close()


def _read_call(call_id: str, table: str) -> ToolCall:
    return ToolCall(
        call_id,
        "data_query",
        {"source": "CRM", "sql": f"SELECT count(*) AS total FROM {table}"},
    )


async def _agent_running(tmp_path: Path, *rounds: tuple[ToolCall, ...]) -> Agent:
    """One scripted model; an empty round ends the run, and the last one does too."""

    provider = MockModelProvider(
        (
            *(
                (
                    ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls
                    )
                    if calls
                    else ModelResponse(finish_reason=FinishReason.STOP, text="Done.")
                )
                for calls in rounds
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Done."),
        )
    )
    return await Agent.create(
        "stale-catalog",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
    )


async def test_a_stale_catalog_cannot_fail_a_read(tmp_path: Path) -> None:
    database = tmp_path / "crm.sqlite"
    _create_companies(database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE secrets (value TEXT)")
    agent = await _agent_running(
        tmp_path, (_read_call("kept", "companies"), _read_call("new", "leads"))
    )
    try:
        await agent.attach(SQLiteSource(database, name="CRM"))
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE secrets")
            connection.execute("CREATE TABLE leads (name TEXT)")
            connection.execute("INSERT INTO leads VALUES ('Initech')")
        run = await agent.run("Count companies and leads.")
        kept, new = (
            _data(result) for _, result in await agent.tool_evidence(run.run_id)
        )
        assert [dict(row) for row in kept["rows"]] == [{"total": 2}]
        assert [dict(row) for row in new["rows"]] == [{"total": 1}]
        await settled(agent)
        assert sync_count(agent) == 1
    finally:
        await agent.close()


async def test_reading_a_dropped_table_returns_the_source_error_and_refreshes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "crm.sqlite"
    _create_companies(database)
    agent = await _agent_running(tmp_path, (_read_call("gone", "companies"),))
    try:
        source = await agent.attach(SQLiteSource(database, name="CRM"))
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE companies")
        run = await agent.run("Count companies.")
        ((_, result),) = await agent.tool_evidence(run.run_id)
        assert result is not None and result.is_error
        error = result.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "query_invalid"
        assert "no such table" in str(error["message"])
        assert "undefined_object" not in canonical_json(result.output)
        await settled(agent)
        assert sync_count(agent) == 2
        assert await agent.list_catalog_resources(source_id=source.id) == ()
    finally:
        await agent.close()


async def test_a_refresh_that_fails_changes_nothing_for_the_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    database = tmp_path / "crm.sqlite"
    _create_companies(database)
    agent = await _agent_running(
        tmp_path, (_read_call("kept", "companies"), _read_call("gone", "leads"))
    )

    async def failing(self: object, request: object) -> object:
        raise RuntimeError("password=private-diagnostic")

    try:
        source = await agent.attach(SQLiteSource(database, name="CRM"))
        monkeypatch.setattr(SQLiteResourceAdapter, "discover", failing)
        run = await agent.run("Count companies and leads.")
        kept, gone = [result for _, result in await agent.tool_evidence(run.run_id)]
        assert [dict(row) for row in _data(kept)["rows"]] == [{"total": 2}]
        assert gone is not None and gone.is_error
        await settled(agent)
        listed = await agent.list_catalog_resources(source_id=source.id)
        assert [item.name for item in listed] == ["companies"]
        assert "RuntimeError" in caplog.text
        assert "private-diagnostic" not in caplog.text
    finally:
        await agent.close()


async def test_a_mistyped_query_repeated_starts_one_refresh_until_the_interval_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "crm.sqlite"
    _create_companies(database)
    typos = [(_read_call(f"typo-{index}", f"compnies{index}"),) for index in range(4)]
    agent = await _agent_running(tmp_path, *typos, (), (_read_call("later", "typo9"),))
    try:
        await agent.attach(SQLiteSource(database, name="CRM"))
        await agent.run("Count companies.")
        await settled(agent)
        assert sync_count(agent) == 2
        monkeypatch.setattr(embedded_module, "_SOURCE_REFRESH_MIN_SECONDS", 0)
        await agent.run("Count companies again.")
        await settled(agent)
        assert sync_count(agent) == 3
    finally:
        await agent.close()
