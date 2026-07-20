"""Joined retained-provider/default-data/source/reopen live acceptance."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import sqlite3

import pytest

from daita import Agent, SQLiteSource
from daita.llm import ModelProfile, OpenAIProvider
from daita.loop.models import LoopExitKind


def _require_live_openai() -> str:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("requires DAITA_RUN_LIVE_LLM=1")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        pytest.skip("requires OPENAI_API_KEY")
    return os.environ.get("OPENAI_TEST_MODEL", "gpt-4.1-mini").strip()


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL
            );
            INSERT INTO customers (status) VALUES ('active'), ('inactive');
            """)


def _request(source_id: str, label: str) -> str:
    return (
        f"For the {label} run, call data_query_sqlite exactly once with source_id "
        f"{source_id!r} and SQL 'SELECT COUNT(*) AS total FROM customers'. Then "
        "answer with the count and cite the accepted tool evidence using the exact "
        "[evidence:<id>] form from the tool result. Do not answer before the tool."
    )


async def _close_provider(provider: object) -> None:
    client = getattr(provider, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_real_retained_openai_runs_default_sqlite_domain_across_cold_reopen(
    tmp_path: Path,
) -> None:
    model = _require_live_openai()
    database = tmp_path / "customers.db"
    _database(database)
    provider = OpenAIProvider(model, max_output_tokens=512)
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=512,
        supports_tools=True,
        supports_parallel_tools=True,
        supports_reasoning=True,
    )
    agent = await Agent.create(
        "retained-live",
        root=tmp_path / "state",
        model=provider,
        model_profile=profile,
    )
    registration = await agent.attach(SQLiteSource(database))
    route = agent.model_route
    assert route is not None
    try:
        first = await agent.run(
            _request(registration.id, "initial"),
            session_id="retained-live-initial",
        )
        first_snapshot = await agent.inspect(first.operation_id)
        assert first.kind is LoopExitKind.COMPLETED
        assert first_snapshot.operation.model_route_revision == route.revision
        assert first_snapshot.operation.model_route_fingerprint == route.fingerprint
        assert first_snapshot.evidence
        assert all(evidence.accepted for evidence in first_snapshot.evidence)
        assert all(
            evidence.validation_facts.source_id == registration.id
            for evidence in first_snapshot.evidence
        )
    finally:
        await agent.close()
        await _close_provider(provider)

    reopened = await Agent.open("retained-live", root=tmp_path / "state")
    try:
        assert reopened.model_route == route
        second = await reopened.run(
            _request(registration.id, "cold-reopen"),
            session_id="retained-live-reopened",
        )
        second_snapshot = await reopened.inspect(second.operation_id)
        assert second.kind is LoopExitKind.COMPLETED
        assert second_snapshot.operation.model_route_revision == route.revision
        assert second_snapshot.operation.model_route_fingerprint == route.fingerprint
        assert second_snapshot.evidence
        assert all(evidence.accepted for evidence in second_snapshot.evidence)
        assert all(
            evidence.validation_facts.source_id == registration.id
            for evidence in second_snapshot.evidence
        )
    finally:
        await reopened.close()
