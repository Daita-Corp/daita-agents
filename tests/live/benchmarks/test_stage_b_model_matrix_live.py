"""Per-model Stage B certification benchmark.

The module authorizes at most four live ``Agent.run`` interactions per model in
``DAITA_STAGE_B_BENCHMARK_MODEL_IDS``. Model IDs must be release-reviewed,
tool-capable API profiles. Each provider uses its provider-specific benchmark
key or the explicitly supplied generic benchmark key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from daita import JobStatus
from daita._json import canonical_json
from daita.domains.data.profile_jobs import DATA_PROFILE_EXECUTION_CAPABILITY_ID

from ._support import (
    IMMEDIATE_AMOUNT,
    IMMEDIATE_TOKEN,
    PROFILE_NULL_VALUES,
    PROFILE_SAMPLE_ROWS,
    TARGET_PROFILE_TABLE,
    assert_completed,
    assert_on_demand_invocation,
    assert_profile_result,
    benchmark_marks,
    capture_run,
    configured_model_ids,
    create_live_agent,
    job_id_from_start,
    logical_names,
    record_metrics,
    results_for,
    seed_completed_profile_agent,
    wait_for_running,
    wait_for_terminal,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_B_MODEL_MATRIX"
_MODEL_IDS = configured_model_ids()

pytestmark = benchmark_marks(
    _AUTHORIZATION,
    maximum_interactions=4,
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
        assert "data_query_sqlite" in names
        assert "start_data_profile" not in names
        assert await fixture.agent.list_jobs() == ()
        assert capture.result.final_text is not None
        assert IMMEDIATE_TOKEN in capture.result.final_text
        assert str(IMMEDIATE_AMOUNT) in capture.result.final_text
    finally:
        await fixture.agent.close()


@pytest.mark.parametrize("model_id", _MODEL_IDS, ids=_MODEL_IDS)
async def test_model_certification_cross_conversation_result(
    tmp_path: Path,
    model_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture, job_id = await seed_completed_profile_agent(
        tmp_path,
        "matrix-result",
        model_id,
    )
    try:
        capture = await capture_run(
            fixture,
            "Find my latest completed profile job, read its result and artifact, "
            "then tell me the sampled row count and nullable_code null count.",
        )
        assert_completed(capture)
        record_metrics(record_property, model_id, capture)
        names = logical_names(capture.transcript)
        assert "job_list" in names
        assert "job_read_results" in names
        assert "artifact_read" in names
        assert "data_query_sqlite" not in names
        assert "toolbox_search" not in names
        assert "toolbox_load" not in names
        result_reads = results_for(capture.transcript, "job_read_results")
        assert result_reads and job_id in canonical_json(result_reads[-1].output)
        await assert_profile_result(fixture.agent, job_id)
        assert capture.result.final_text is not None
        assert str(PROFILE_SAMPLE_ROWS) in capture.result.final_text
        assert str(PROFILE_NULL_VALUES) in capture.result.final_text
    finally:
        await fixture.agent.close()


@pytest.mark.parametrize("model_id", _MODEL_IDS, ids=_MODEL_IDS)
async def test_model_certification_start_and_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(tmp_path, "matrix-cancel", model_id)
    _, executor = fixture.agent._embedded._capabilities.resolve_execution(
        DATA_PROFILE_EXECUTION_CAPABILITY_ID
    )
    original_execute = executor.execute
    execution_started = asyncio.Event()
    release = asyncio.Event()

    async def held_execute(request):
        execution_started.set()
        await release.wait()
        return await original_execute(request)

    monkeypatch.setattr(executor, "execute", held_execute)
    try:
        started = await capture_run(
            fixture,
            "Start a durable five-row profile of stage_b_profile_probe in the "
            "background and return once it has been accepted.",
            source_id=fixture.source_id,
        )
        assert_completed(started)
        assert_on_demand_invocation(started, "start_data_profile")
        job_id = job_id_from_start(started.transcript)
        inspection = await fixture.agent.inspect_job(job_id)
        assert inspection is not None
        assert inspection.summary.resource_ids == (
            fixture.resource_ids[TARGET_PROFILE_TABLE],
        )
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        await wait_for_running(fixture.agent, job_id)

        cancelled = await capture_run(
            fixture,
            f"Cancel the exact running job {job_id} now.",
        )
        assert_completed(cancelled)
        record_metrics(record_property, model_id, started)
        record_metrics(record_property, model_id, cancelled)
        assert_on_demand_invocation(cancelled, "job_cancel")
        cancel_results = results_for(cancelled.transcript, "job_cancel")
        assert len(cancel_results) == 1
        assert job_id in canonical_json(cancel_results[0].output)
        terminal = await wait_for_terminal(fixture.agent, job_id)
        assert terminal.summary.status is JobStatus.CANCELLED
    finally:
        release.set()
        await fixture.agent.close()


def test_model_matrix_configuration_is_bounded() -> None:
    assert 1 <= len(_MODEL_IDS) <= 6
    assert len(_MODEL_IDS) == len(set(_MODEL_IDS))
