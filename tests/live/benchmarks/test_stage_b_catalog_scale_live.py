"""Live Stage B resource-selection benchmark with look-alike catalog entries.

The complete module authorizes at most six live ``Agent.run`` interactions for
the default model: one immediate read and one durable admission at each of three
catalog sizes. Every interaction retains the shared cost and runtime ceilings.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from daita import JobStatus

from ._support import (
    DEFAULT_MODEL_ID,
    IMMEDIATE_AMOUNT,
    IMMEDIATE_TOKEN,
    PROFILE_SAMPLE_ROWS,
    TARGET_PROFILE_TABLE,
    assert_completed,
    assert_on_demand_invocation,
    assert_profile_result,
    benchmark_marks,
    capture_run,
    create_live_agent,
    job_id_from_start,
    logical_names,
    record_metrics,
    wait_for_terminal,
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
        assert "data_query_sqlite" in names
        assert "start_data_profile" not in names
        assert await fixture.agent.list_jobs() == ()
        assert capture.result.usage.total_tokens <= 30_000
        assert capture.result.final_text is not None
        assert IMMEDIATE_TOKEN in capture.result.final_text
        assert str(IMMEDIATE_AMOUNT) in capture.result.final_text
    finally:
        await fixture.agent.close()


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
            "Start a durable background profile for the exact current table "
            "stage_b_profile_probe with a five-row sample. Do not choose any "
            "archive or backup table, and return as soon as the work is accepted.",
            source_id=fixture.source_id,
        )
        assert_completed(capture)
        record_metrics(record_property, DEFAULT_MODEL_ID, capture)
        assert_on_demand_invocation(capture, "start_data_profile")
        job_id = job_id_from_start(capture.transcript)
        terminal = await wait_for_terminal(fixture.agent, job_id)
        assert terminal.summary.status is JobStatus.SUCCEEDED
        assert terminal.summary.resource_ids == (
            fixture.resource_ids[TARGET_PROFILE_TABLE],
        )
        result = await fixture.agent.read_job_result(job_id)
        assert result is not None
        assert result.summary["sampled_rows"] == PROFILE_SAMPLE_ROWS
        await assert_profile_result(fixture.agent, job_id)
        assert capture.result.usage.total_tokens <= 30_000
    finally:
        await fixture.agent.close()


def test_catalog_scale_inventory_is_strictly_increasing_and_bounded() -> None:
    assert _CATALOG_SIZES == tuple(sorted(set(_CATALOG_SIZES)))
    assert _CATALOG_SIZES[0] >= 10
    assert _CATALOG_SIZES[-1] <= 512
