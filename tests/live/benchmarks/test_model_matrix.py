"""Per-model Stage B certification benchmark.

The module authorizes at most one live ``Agent.run`` interaction per model in
``DAITA_STAGE_B_BENCHMARK_MODEL_IDS``. Model IDs must be release-reviewed,
tool-capable API profiles. Each provider uses its provider-specific benchmark
key or the explicitly supplied generic benchmark key.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.support.job_benchmarks import (
    IMMEDIATE_AMOUNT,
    IMMEDIATE_TOKEN,
    assert_completed,
    benchmark_marks,
    capture_run,
    configured_model_ids,
    create_live_agent,
    logical_names,
    record_metrics,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_B_MODEL_MATRIX"
_MODEL_IDS = configured_model_ids()

pytestmark = benchmark_marks(
    _AUTHORIZATION,
    maximum_interactions=1,
    per_model=True,
)


@pytest.mark.parametrize("model_id", _MODEL_IDS, ids=_MODEL_IDS)
async def test_model_certification_immediate_selection(
    tmp_path: Path,
    model_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(tmp_path, "matrix-immediate", model_id)
    try:
        capture = await capture_run(
            fixture,
            "Read the immediate record from stage_b_immediate_probe and tell me "
            "its verification token and amount. This is a quick lookup, so do not "
            "start durable work.",
            source_id=fixture.source_id,
        )
        assert_completed(capture)
        record_metrics(record_property, model_id, capture)
        names = logical_names(capture.transcript)
        assert "data_query" in names
        assert await fixture.agent.list_jobs() == ()
        assert capture.result.final_text is not None
        assert IMMEDIATE_TOKEN in capture.result.final_text
        assert str(IMMEDIATE_AMOUNT) in capture.result.final_text
    finally:
        await fixture.close()


def test_model_matrix_configuration_is_bounded() -> None:
    assert 1 <= len(_MODEL_IDS) <= 6
    assert len(_MODEL_IDS) == len(set(_MODEL_IDS))
