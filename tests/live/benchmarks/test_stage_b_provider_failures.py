"""Deterministic Stage B provider-failure benchmark.

These tests deliberately use canonical fault-injecting providers. They do not
call a paid API. Their Stage B-specific contract is that a provider failure
after a durable start receipt can interrupt the model run, but cannot duplicate,
cancel, or corrupt the independently admitted job.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent, JobStatus
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import ModelStreamCompleted, ModelTextDelta
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.llm.routing import (
    ModelProviderRegistration,
    ModelRouter,
    RetryPolicy,
)
from daita.loop.models import LoopExitKind

from ._support import (
    OFFLINE_EAGER_LIMITS,
    TARGET_PROFILE_TABLE,
    create_probe_home,
    logical_names,
    start_profile_response,
    stop_response,
    toolbox_load_response,
    transcript_text,
    wait_for_terminal,
)

pytestmark = [pytest.mark.integration, pytest.mark.acceptance]


def _registration(provider):
    return ModelProviderRegistration(
        provider=provider,
        profile=provider.model_profile,
    )


async def _open(
    home,
    provider,
    profile,
) -> Agent:
    return await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=profile,
        limits=OFFLINE_EAGER_LIMITS,
        workspace=workspace_for(home.root),
    )


async def _assert_one_independent_success(agent: Agent, run_id: str) -> str:
    jobs = await agent.list_jobs()
    assert len(jobs) == 1
    job_id = jobs[0].job_id
    terminal = await wait_for_terminal(agent, job_id)
    assert terminal.summary.status is JobStatus.SUCCEEDED
    assert terminal.origin_run_id == run_id
    result = await agent.read_job_result(job_id)
    assert result is not None and result.summary["sampled_rows"] == 5
    return job_id


async def test_transient_retry_after_start_receipt_does_not_reexecute_job_start(
    tmp_path: Path,
) -> None:
    home = await create_probe_home(tmp_path, "provider-transient-retry")
    scripted = MockModelProvider(
        (
            toolbox_load_response("start_data_profile"),
            start_profile_response(home.resource_ids[TARGET_PROFILE_TABLE]),
            ModelProviderError(ProviderErrorCode.TIMEOUT),
            stop_response("The independently admitted job is still running."),
        ),
        provider_id="mock:stage-b-transient",
    )
    router = ModelRouter(
        (_registration(scripted),),
        retry_policy=RetryPolicy(attempts=2, backoff_seconds=0),
    )
    agent = await _open(home, router, router.model_profile)
    try:
        result = await agent.run(
            "Start one profile and survive a later provider retry.",
            source_id=home.source_id,
        )
        assert result.kind is LoopExitKind.COMPLETED
        assert len(scripted.requests) == 4
        transcript = await agent.transcript(result.run_id)
        assert logical_names(transcript).count("start_data_profile") == 1
        await _assert_one_independent_success(agent, result.run_id)
    finally:
        await agent.close()


async def test_permanent_provider_failure_after_receipt_leaves_one_durable_job(
    tmp_path: Path,
) -> None:
    home = await create_probe_home(tmp_path, "provider-permanent-failure")
    scripted = MockModelProvider(
        (
            toolbox_load_response("start_data_profile"),
            start_profile_response(home.resource_ids[TARGET_PROFILE_TABLE]),
            ModelProviderError(ProviderErrorCode.INVALID_REQUEST),
        ),
        provider_id="mock:stage-b-permanent",
    )
    router = ModelRouter(
        (_registration(scripted),),
        retry_policy=RetryPolicy(attempts=5, backoff_seconds=0),
    )
    agent = await _open(home, router, router.model_profile)
    try:
        result = await agent.run(
            "Start one profile before the provider becomes permanently invalid.",
            source_id=home.source_id,
        )
        assert result.kind is LoopExitKind.FAILED
        assert result.reason == ProviderErrorCode.INVALID_REQUEST.value
        assert len(scripted.requests) == 3
        transcript = await agent.transcript(result.run_id)
        assert logical_names(transcript).count("start_data_profile") == 1
        await _assert_one_independent_success(agent, result.run_id)
    finally:
        await agent.close()


async def test_visible_stream_failure_after_receipt_never_persists_partial_text(
    tmp_path: Path,
) -> None:
    home = await create_probe_home(tmp_path, "provider-stream-failure")
    partial = "EPHEMERAL_STAGE_B_TEXT_MUST_NOT_PERSIST"
    scripted = MockStreamingModelProvider(
        (
            (ModelStreamCompleted(toolbox_load_response("start_data_profile")),),
            (
                ModelStreamCompleted(
                    start_profile_response(home.resource_ids[TARGET_PROFILE_TABLE])
                ),
            ),
            (
                ModelTextDelta(partial),
                ModelProviderError(ProviderErrorCode.PROVIDER_UNAVAILABLE),
            ),
        ),
        provider_id="mock:stage-b-stream-failure",
    )
    agent = await _open(home, scripted, scripted.model_profile)
    try:
        result = await agent.run(
            "Start one profile before a visible stream interruption.",
            source_id=home.source_id,
        )
        assert result.kind is LoopExitKind.FAILED
        assert result.reason == ProviderErrorCode.PROVIDER_UNAVAILABLE.value
        transcript = await agent.transcript(result.run_id)
        assert partial not in transcript_text(transcript)
        assert logical_names(transcript).count("start_data_profile") == 1
        await _assert_one_independent_success(agent, result.run_id)
    finally:
        await agent.close()


async def test_malformed_terminal_stream_after_receipt_preserves_job_truth(
    tmp_path: Path,
) -> None:
    home = await create_probe_home(tmp_path, "provider-malformed-stream")
    partial = "MALFORMED_EPHEMERAL_STAGE_B_TEXT"
    scripted = MockStreamingModelProvider(
        (
            (ModelStreamCompleted(toolbox_load_response("start_data_profile")),),
            (
                ModelStreamCompleted(
                    start_profile_response(home.resource_ids[TARGET_PROFILE_TABLE])
                ),
            ),
            (ModelTextDelta(partial),),
        ),
        provider_id="mock:stage-b-malformed-stream",
    )
    agent = await _open(home, scripted, scripted.model_profile)
    try:
        result = await agent.run(
            "Start one profile before a malformed terminal model stream.",
            source_id=home.source_id,
        )
        assert result.kind is LoopExitKind.FAILED
        assert result.reason == ProviderErrorCode.MALFORMED_RESPONSE.value
        transcript = await agent.transcript(result.run_id)
        assert partial not in transcript_text(transcript)
        assert logical_names(transcript).count("start_data_profile") == 1
        await _assert_one_independent_success(agent, result.run_id)
    finally:
        await agent.close()
