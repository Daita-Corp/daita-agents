"""Two sources share a display name: a live model is refused, then asks the person.

A scripted model proves the refusal fires, never how a model answers it.

    DAITA_RUN_LIVE_DUPLICATE_SOURCE_NAMES=1 \\
    DAITA_DUPLICATE_SOURCE_NAMES_LIVE_MODEL_ID=anthropic:claude-sonnet-5 \\
    pytest tests/live/data/test_duplicate_source_names.py
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, LoopLimits, SQLiteSource
from daita.llm.models import ToolResultBlock
from daita.loop.models import LoopExitKind
from tests.support.live_billing import live_provider
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_DUPLICATE_SOURCE_NAMES"
_MODEL_ID = "DAITA_DUPLICATE_SOURCE_NAMES_LIVE_MODEL_ID"
_DEFAULT_MODEL_ID = "anthropic:claude-sonnet-5"

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=f"Set {_AUTHORIZATION}=1 to authorize one live model run.",
    ),
]


def _database(path: Path, total: int) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO orders VALUES (?)", ((row,) for row in range(total))
        )


async def test_duplicate_names_are_refused_and_the_model_asks_without_reading(
    tmp_path: Path,
) -> None:
    profile, provider = live_provider(
        os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID),
        api_key=lambda: os.environ["ANTHROPIC_API_KEY"],
        max_output_tokens=2_000,
    )
    agent = await Agent.create(
        "duplicate-source-names",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=profile,
        # Anthropic routes have no complete preflight price, so a dollar cap refuses
        # every run; these limits hold a Sonnet 5 run under about $0.25.
        limits=LoopLimits(
            max_steps=6,
            max_total_tokens=40_000,
            max_estimated_cost_usd=None,
        ),
    )
    try:
        for name, total in (("first.sqlite", 3), ("second.sqlite", 7)):
            _database(tmp_path / name, total)
            await agent.attach(SQLiteSource(tmp_path / name, name="Sales"))
        result = await agent.run("How many orders are in Sales?")
        print(result.usage, result.final_text)
        assert result.kind is LoopExitKind.COMPLETED, result
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    results = [
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    codes = [
        cast(Mapping[str, object], block.output["error"])["code"]
        for block in results
        if block.is_error
    ]
    assert "source_ambiguous" in codes
    assert all(block.is_error for block in results), "a source was read"
    answer = result.final_text or ""
    assert "?" in answer, answer
    # It asked rather than guessed: neither table's count is claimed as the answer.
    assert "7" not in answer and "3 orders" not in answer, answer
