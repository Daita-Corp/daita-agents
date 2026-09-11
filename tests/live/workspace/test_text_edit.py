"""Authorized live-model acceptance coverage for one bound local text edit."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.support.local_text_edit_live import (
    _AUTHORIZATION,
    _MAX_COST,
    run_live_text_edit_scenario,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing one live "
            f"Agent.run capped by {_MAX_COST}"
        ),
    ),
]


async def test_live_model_reads_edits_approves_and_replaces_exact_bound_file(
    tmp_path: Path,
) -> None:
    await run_live_text_edit_scenario(tmp_path)
