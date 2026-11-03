"""A read goes to the source, so catalog state cannot decide its outcome."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from daita import Agent, SQLiteSource
from daita.catalog.protocols import CatalogStoreError
from daita.llm.models import FinishReason, ModelResponse, ToolCall
from daita.llm.providers.mock import MockModelProvider
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


async def _assert_counted_two(agent: Agent, run_id: str) -> None:
    ((_, result),) = await agent.tool_evidence(run_id)
    assert result is not None and not result.is_error, result
    data = result.output["data"]
    assert isinstance(data, Mapping)
    assert [dict(row) for row in data["rows"]] == [{"total": 2}]


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
        await agent.attach(SQLiteSource(database, name="companies"))
        _create_companies(database)
        run = await agent.run("What columns does companies have?")
        ((_, result),) = await agent.tool_evidence(run.run_id)
        assert result is not None and not result.is_error, result
        data = result.output["data"]
        assert isinstance(data, Mapping)
        assert data["name"] == "main.companies"
        assert data["answered_from"] == "live"
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
        source = await agent.attach(SQLiteSource(database, name="CRM"))
        companies = next(
            item
            for item in await agent.list_catalog_resources(source_id=source.id)
            if item.name == "companies"
        )
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="selected",
            read_resource_ids=(companies.id,),
            relational_write_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )
        _break_catalog(agent, monkeypatch)
        result = await agent.run("How many companies are there?")
        await _assert_counted_two(agent, result.run_id)
    finally:
        await agent.close()


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
