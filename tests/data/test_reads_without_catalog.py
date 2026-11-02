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
    assert data["rows"] == ({"total": 2},)


@pytest.mark.xfail(strict=True, reason="data_query still requires catalog resources")
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


@pytest.mark.xfail(strict=True, reason="a run still needs a working catalog store")
async def test_read_answers_while_the_catalog_store_is_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "crm.sqlite"
    _create_companies(database)
    agent = await _create_agent(tmp_path)
    try:
        await agent.attach(SQLiteSource(database, name="CRM"))
        monkeypatch.setattr(
            agent._embedded._catalog_service, "_store", _BrokenCatalogStore()
        )
        result = await agent.run("How many companies are there?")
        await _assert_counted_two(agent, result.run_id)
    finally:
        await agent.close()
