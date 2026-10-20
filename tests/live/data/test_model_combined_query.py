"""A real model joins a source without SQL to a SQLite table in one data_query.

The provider schema shows data_query's inputs only as a pointer to the full
contract, with one description clause naming the tabular form. Scripted tests
prove the combined path executes; only a real model shows the form is found
and built. The regions source has no SQL at all, so its revenue reaches the
answer only through a tabular input of a combined query. A declared catalog
link states the join, as a deployment would.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import Agent, AgentConfig, LoopLimits, SQLiteSource
from daita.domains.data import DATA_QUERY_COMBINED_EVIDENCE_KIND
from daita.loop.models import LoopExitKind, validate_completed_transcript
from tests.support.live_billing import cost_cap, live_provider
from tests.support.memory_source import MEMORY_NOW, MEMORY_TYPE, MemorySource
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_COMBINED_QUERY"
_MODEL_ID = "DAITA_COMBINED_QUERY_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_COMBINED_QUERY_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_COMBINED_QUERY_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_KEY_ENVIRONMENT = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}
# Regions revenue (from the in-memory source): north 15, east 15, south 7,
# west 3. Ada manages north and east (30); Grace manages south and west (10).
_MANAGERS = (("north", "Ada"), ("east", "Ada"), ("south", "Grace"), ("west", "Grace"))
_QUESTIONS = (
    "What is the total revenue for each manager? Revenue per region is in the "
    "Regions source and each region's manager is in the Managers database.",
    "Combine our regional revenue with the manager assignments and tell me each "
    "manager's total revenue.",
    "Which manager oversees more revenue, and by how much? Give each manager's "
    "total.",
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing three live "
            f"Agent.run interactions, each capped by {_MAX_COST}"
        ),
    ),
]


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.50")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


def _api_key(model_id: str) -> str:
    name = _API_KEY_ENVIRONMENT.get(model_id.partition(":")[0])
    value = os.environ.get(_MODEL_KEY) or (os.environ.get(name) if name else None)
    if not value:
        pytest.fail(f"set {_MODEL_KEY} or the provider's API key for {model_id}")
    return value


@pytest.mark.parametrize("phrasing", range(len(_QUESTIONS)))
async def test_live_model_joins_a_source_without_sql_through_a_tabular_input(
    tmp_path: Path, phrasing: int
) -> None:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    profile, provider = live_provider(
        model_id, api_key=lambda: _api_key(model_id), max_output_tokens=1_536
    )
    database = tmp_path / "managers.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE managers (region TEXT, manager TEXT)")
        connection.executemany("INSERT INTO managers VALUES (?, ?)", _MANAGERS)
    agent = await Agent.create(
        "live-combined-query",
        root=tmp_path / "home",
        config=AgentConfig(source_types=(MEMORY_TYPE,)),
        model=provider,
        model_profile=profile,
        limits=LoopLimits(max_estimated_cost_usd=cost_cap(_cost_limit())),
        # The in-memory source stamps its catalog at this fixed instant.
        clock=lambda: MEMORY_NOW,
        workspace=workspace_for(tmp_path / "home"),
    )
    try:
        regions = await agent.attach(MemorySource("Regions"))
        managers = await agent.attach(SQLiteSource(database, name="Managers"))
        (regions_table,) = await agent.list_catalog_resources(source_id=regions.id)
        (managers_table,) = await agent.list_catalog_resources(source_id=managers.id)
        # The declared link is how a deployment states the join; without it
        # the model rightly asks which fields relate the two tables.
        await agent.declare_catalog_link(
            managers_table.id, regions_table.id, (("region", "region"),)
        )
        result = await agent.run(_QUESTIONS[phrasing])
        transcript = await agent.transcript(result.run_id)
        evidence = await agent.tool_evidence(result.run_id)
    finally:
        try:
            await agent.close()
        finally:
            await provider.close()

    assert result.kind is LoopExitKind.COMPLETED, result
    validate_completed_transcript(transcript, result)
    joined = [
        block.output["data"]
        for call, block in evidence
        if call.name == "data_query"
        and block is not None
        and not block.is_error
        and block.output.get("kind") == DATA_QUERY_COMBINED_EVIDENCE_KIND
    ]
    # The regions source has no SQL, so its revenue must arrive as a tabular
    # input: lineage names its columns, not a canonical statement.
    assert any(
        isinstance(item, Mapping)
        and item.get("source_id") == regions.id
        and "columns" in item
        for data in joined
        if isinstance(data, Mapping)
        for item in data.get("inputs", ())
    ), [call.arguments for call, _ in evidence]
    answer = result.final_text or ""
    for fact in ("Ada", "30", "Grace", "10"):
        assert fact in answer, answer
