"""Live Stage B resource-selection benchmark with look-alike catalog entries.

The complete module authorizes at most six live ``Agent.run`` interactions for
the default model: one immediate read and one aggregate profile at each of three
catalog sizes. Every interaction retains the shared cost and runtime ceilings.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.support.job_benchmarks import (
    DEFAULT_MODEL_ID,
    IMMEDIATE_AMOUNT,
    IMMEDIATE_TOKEN,
    PROFILE_NULL_VALUES,
    PROFILE_SAMPLE_ROWS,
    TARGET_PROFILE_TABLE,
    assert_completed,
    benchmark_marks,
    capture_run,
    create_live_agent,
    logical_names,
    record_metrics,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_B_CATALOG_BENCHMARK"
_CATALOG_SIZES = (16, 64, 128)

pytestmark = benchmark_marks(_AUTHORIZATION, maximum_interactions=6)


@pytest.mark.parametrize("distractor_tables", _CATALOG_SIZES)
async def test_immediate_target_survives_catalog_distractors(
    tmp_path: Path,
    distractor_tables: int,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(
        tmp_path,
        f"catalog-read-{distractor_tables}",
        DEFAULT_MODEL_ID,
        distractor_tables=distractor_tables,
    )
    try:
        capture = await capture_run(
            fixture,
            "Look up the immediate row in the exact table "
            "stage_b_immediate_probe and report its verification token and amount. "
            "Ignore similarly named backup or archive tables and do not start a "
            "background job.",
            source_id=fixture.source_id,
        )
        assert_completed(capture)
        record_metrics(record_property, DEFAULT_MODEL_ID, capture)
        names = logical_names(capture.transcript)
        assert "data_query" in names
        assert await fixture.agent.list_jobs() == ()
        assert capture.result.final_text is not None
        assert IMMEDIATE_TOKEN in capture.result.final_text
        assert str(IMMEDIATE_AMOUNT) in capture.result.final_text
    finally:
        await fixture.close()


@pytest.mark.parametrize("distractor_tables", _CATALOG_SIZES)
async def test_profile_target_survives_catalog_distractors(
    tmp_path: Path,
    distractor_tables: int,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(
        tmp_path,
        f"catalog-profile-{distractor_tables}",
        DEFAULT_MODEL_ID,
        distractor_tables=distractor_tables,
    )
    try:
        capture = await capture_run(
            fixture,
            "Profile the exact current table stage_b_profile_probe: report its row "
            "count and how many nullable_code values are null. Do not choose any "
            "archive or backup table.",
            source_id=fixture.source_id,
        )
        assert_completed(capture)
        record_metrics(record_property, DEFAULT_MODEL_ID, capture)
        assert "data_query" in logical_names(capture.transcript)
        assert await fixture.agent.list_jobs() == ()
        target = fixture.resource_ids[TARGET_PROFILE_TABLE]
        queried = [
            call.arguments.get("resource_ids")
            for call, _ in capture.evidence
            if call.name == "data_query"
        ]
        assert queried and all(item in ([target], (target,)) for item in queried)
        assert capture.result.final_text is not None
        assert str(PROFILE_SAMPLE_ROWS) in capture.result.final_text
        assert str(PROFILE_NULL_VALUES) in capture.result.final_text
    finally:
        await fixture.close()


def test_catalog_scale_inventory_is_strictly_increasing_and_bounded() -> None:
    assert _CATALOG_SIZES == tuple(sorted(set(_CATALOG_SIZES)))
    assert _CATALOG_SIZES[0] >= 10
    assert _CATALOG_SIZES[-1] <= 512
