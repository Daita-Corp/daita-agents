"""Natural-language Stage B benchmark without model-facing tool instructions.

The complete module authorizes at most sixteen ``Agent.run`` interactions.
Every interaction retains the shared step, token, wall-time, and estimated-cost
ceilings. Deterministic setup runs used to seed completed jobs do not call a
live provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest

from daita import GraphState
from daita._json import canonical_json
from daita.domains.data.profile_jobs import DATA_PROFILE_EXECUTION_CAPABILITY_ID
from tests.support.job_benchmarks import (
    DEFAULT_MODEL_ID,
    IMMEDIATE_AMOUNT,
    IMMEDIATE_TOKEN,
    LIFECYCLE_TOOLS,
    PROFILE_NULL_VALUES,
    PROFILE_SAMPLE_ROWS,
    TARGET_IMMEDIATE_TABLE,
    TARGET_PROFILE_TABLE,
    admit_profile_job,
    assert_completed,
    assert_on_demand_invocation,
    assert_profile_result,
    benchmark_marks,
    capture_run,
    create_live_agent,
    logical_names,
    record_metrics,
    results_for,
    seed_completed_profile_agent,
    wait_for_running,
    wait_for_terminal,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_B_PARAPHRASE_BENCHMARK"

pytestmark = benchmark_marks(_AUTHORIZATION, maximum_interactions=16)

_IMMEDIATE_PROMPTS = (
    (
        "quick-lookup",
        "Could you quickly look up the row named immediate in "
        "stage_b_immediate_probe and tell me its verification token and amount? "
        "I only need the answer now; do not launch background work.",
    ),
    (
        "plain-language",
        "What token and amount are stored on the immediate record in "
        "stage_b_immediate_probe? This is just a small lookup, not a job.",
    ),
    (
        "negative-instruction",
        "Please check stage_b_immediate_probe for probe_name immediate. Answer "
        "from the current data and don't queue or start anything long-running.",
    ),
)

_PROFILE_PROMPTS = (
    (
        "profile-summary",
        "Please profile stage_b_profile_probe: how many rows does it have, and "
        "how many nullable_code values are null?",
    ),
    (
        "data-quality",
        "Give me a quick data-quality check of stage_b_profile_probe: the row "
        "count and the number of missing nullable_code values.",
    ),
    (
        "background-wording",
        "Profile stage_b_profile_probe in the background if you need to, but I "
        "only want its total row count and nullable_code null count.",
    ),
)

_STATUS_PROMPTS = (
    ("all-jobs", "What background jobs have I run here, and what state are they in?"),
    ("latest-job", "Give me the current status of my latest job."),
    (
        "cross-conversation",
        "I'm in a new conversation. Do I have any durable data work, and has it "
        "finished?",
    ),
)

_RESULT_PROMPTS = (
    (
        "latest-profile",
        "How did my most recent data profile turn out? Read its result artifact "
        "and tell me the sampled row count and the null count for nullable_code.",
    ),
    (
        "exact-values",
        "Find my completed profiling job and give me the exact sampled_rows value "
        "and nullable_code null_values value from its artifact.",
    ),
    (
        "new-chat-result",
        "We're in a different chat now. Retrieve the latest profile result and "
        "tell me how many rows it sampled and how many nullable_code values were null.",
    ),
)

# No model tool starts a profile job; each case admits the internal profile
# graph in the conversation of one short setup interaction, then asks the model
# to cancel it.
_CANCEL_CASES = (
    (
        "stop-exact",
        "I am about to ask you about a background job. Reply with the word ready.",
        "Stop the running job {job_id}. Request the cancellation now and tell me "
        "which job received the request.",
    ),
    (
        "cancel-plain-language",
        "Before my next request, just acknowledge this message in one word.",
        "I no longer want {job_id} to run. Cancel that exact background job now.",
    ),
)


@pytest.mark.parametrize(
    ("case_id", "prompt"),
    _IMMEDIATE_PROMPTS,
    ids=tuple(item[0] for item in _IMMEDIATE_PROMPTS),
)
async def test_paraphrased_immediate_reads_do_not_create_jobs(
    tmp_path: Path,
    case_id: str,
    prompt: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(
        tmp_path,
        f"paraphrase-immediate-{case_id}",
        DEFAULT_MODEL_ID,
    )
    try:
        capture = await capture_run(
            fixture,
            prompt,
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


@pytest.mark.parametrize(
    ("case_id", "prompt"),
    _PROFILE_PROMPTS,
    ids=tuple(item[0] for item in _PROFILE_PROMPTS),
)
async def test_paraphrased_profile_requests_use_aggregates_without_jobs(
    tmp_path: Path,
    case_id: str,
    prompt: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(
        tmp_path,
        f"paraphrase-profile-{case_id}",
        DEFAULT_MODEL_ID,
    )
    try:
        capture = await capture_run(
            fixture,
            prompt,
            source_id=fixture.source_id,
        )
        assert_completed(capture)
        record_metrics(record_property, DEFAULT_MODEL_ID, capture)
        names = logical_names(capture.transcript)
        assert "data_query" in names
        assert not (LIFECYCLE_TOOLS & set(names))
        assert await fixture.agent.list_jobs() == ()
        target = fixture.resource_ids[TARGET_PROFILE_TABLE]
        assert any(
            target in cast(Sequence[str], call.arguments.get("resource_ids", ()))
            for call, _ in capture.transcript.tool_pairs
            if call.name == "data_query"
        )
        assert capture.result.final_text is not None
        assert str(PROFILE_SAMPLE_ROWS) in capture.result.final_text
        assert str(PROFILE_NULL_VALUES) in capture.result.final_text
    finally:
        await fixture.close()


@pytest.mark.parametrize(
    ("case_id", "prompt"),
    _STATUS_PROMPTS,
    ids=tuple(item[0] for item in _STATUS_PROMPTS),
)
async def test_paraphrased_status_questions_are_agent_scoped_and_direct(
    tmp_path: Path,
    case_id: str,
    prompt: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture, job_id = await seed_completed_profile_agent(
        tmp_path,
        f"paraphrase-status-{case_id}",
        DEFAULT_MODEL_ID,
    )
    try:
        capture = await capture_run(fixture, prompt)
        assert_completed(capture)
        record_metrics(record_property, DEFAULT_MODEL_ID, capture)
        names = logical_names(capture.transcript)
        assert "job_list" in names
        assert "toolbox_search" not in names
        assert "toolbox_load" not in names
        list_results = results_for(capture.transcript, "job_list")
        assert list_results and job_id in canonical_json(list_results[-1].output)
        direct_names = {tool.name for tool in capture.requests[0].tools}
        assert "job_list" in direct_names
    finally:
        await fixture.close()


@pytest.mark.parametrize(
    ("case_id", "prompt"),
    _RESULT_PROMPTS,
    ids=tuple(item[0] for item in _RESULT_PROMPTS),
)
async def test_paraphrased_result_questions_recover_exact_artifacts(
    tmp_path: Path,
    case_id: str,
    prompt: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture, job_id = await seed_completed_profile_agent(
        tmp_path,
        f"paraphrase-result-{case_id}",
        DEFAULT_MODEL_ID,
    )
    try:
        capture = await capture_run(fixture, prompt)
        assert_completed(capture)
        record_metrics(record_property, DEFAULT_MODEL_ID, capture)
        names = logical_names(capture.transcript)
        assert "job_list" in names
        assert "job_read_results" in names
        assert "artifact_read" in names
        assert "data_query" not in names
        assert "toolbox_search" not in names
        assert "toolbox_load" not in names
        result_reads = results_for(capture.transcript, "job_read_results")
        assert result_reads and job_id in canonical_json(result_reads[-1].output)
        result = await fixture.agent.read_job_result(job_id)
        assert result is not None and len(result.artifact_ids) == 1
        artifact_id = result.artifact_ids[0]
        artifact_reads = results_for(capture.transcript, "artifact_read")
        assert artifact_reads
        assert artifact_id in canonical_json(artifact_reads[-1].output)
        await assert_profile_result(fixture.agent, job_id)
        assert capture.result.final_text is not None
        assert str(PROFILE_SAMPLE_ROWS) in capture.result.final_text
        assert str(PROFILE_NULL_VALUES) in capture.result.final_text
    finally:
        await fixture.close()


@pytest.mark.parametrize(
    ("case_id", "setup_prompt", "cancel_prompt"),
    _CANCEL_CASES,
    ids=tuple(item[0] for item in _CANCEL_CASES),
)
async def test_paraphrased_cancellation_targets_one_running_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    setup_prompt: str,
    cancel_prompt: str,
    record_property: Callable[[str, object], None],
) -> None:
    fixture = await create_live_agent(
        tmp_path,
        f"paraphrase-cancel-{case_id}",
        DEFAULT_MODEL_ID,
    )
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
            setup_prompt,
            source_id=fixture.source_id,
        )
        assert_completed(started)
        job_id = await admit_profile_job(
            fixture.agent, fixture.resource_ids[TARGET_PROFILE_TABLE], started.result
        )
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        await wait_for_running(fixture.agent, job_id)

        cancelled = await capture_run(
            fixture,
            cancel_prompt.format(job_id=job_id),
        )
        assert_completed(cancelled)
        record_metrics(record_property, DEFAULT_MODEL_ID, cancelled)
        assert_on_demand_invocation(cancelled, "job_cancel")
        cancel_results = results_for(cancelled.transcript, "job_cancel")
        assert len(cancel_results) == 1
        assert job_id in canonical_json(cancel_results[0].output)
        terminal = await wait_for_terminal(fixture.agent, job_id)
        assert terminal.job.state is GraphState.CANCELLED
        assert await fixture.agent.read_job_result(job_id) is None
    finally:
        release.set()
        await fixture.close()


def test_paraphrase_case_inventory_is_bounded_and_nonduplicative() -> None:
    cases = (
        *_IMMEDIATE_PROMPTS,
        *_PROFILE_PROMPTS,
        *_STATUS_PROMPTS,
        *_RESULT_PROMPTS,
        *((case_id, start) for case_id, start, _ in _CANCEL_CASES),
        *((case_id + "-cancel", cancel) for case_id, _, cancel in _CANCEL_CASES),
    )
    prompts = tuple(prompt for _, prompt in cases)
    assert len(cases) == 16
    assert len(prompts) == len(set(prompts))
    assert all(20 <= len(prompt) <= 600 for prompt in prompts)
    assert all(
        forbidden not in prompt
        for prompt in prompts
        for forbidden in (
            "toolbox_search",
            "toolbox_load",
            "data_query",
            "job_list",
            "job_read_results",
            "job_cancel",
        )
    )
    assert TARGET_IMMEDIATE_TABLE in " ".join(prompts)
