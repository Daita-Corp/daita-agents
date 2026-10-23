"""A job continues in new runs: answers, a full context, raised limits, restarts."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from daita import LoopLimits, WorkColumn
from daita.jobs import ChecklistStatus, DelegationState, JobState
from daita.jobs.models import (
    INTERRUPTED_BY_RESTART,
    JOB_RUN_LIMIT,
    JOB_TOKEN_LIMIT_RUNS,
)
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from tests.support.jobs import (
    JobModel,
    in_state,
    job_payload,
    open_job_agent,
    reply,
    tool_result,
    wait_for_job,
)

pytestmark = pytest.mark.integration

OBJECTIVE = "Check the two parts and report."
PLAN = {
    "items": [
        {"id": "a", "title": "Part A"},
        {"id": "c", "title": "Combine", "after": ["a"]},
    ]
}


def error_code(request: ModelRequest, call_id: str) -> str:
    result = tool_result(request, call_id)
    assert result.is_error, result.output
    error = result.output["error"]
    assert isinstance(error, Mapping)
    return str(error["code"])


async def test_delegating_an_item_waits_for_its_after_items_and_a_failure_reopens_it(
    tmp_path: Path,
) -> None:
    model = JobModel()
    seen: dict[str, object] = {}

    def after_refusal(request: ModelRequest) -> ModelResponse:
        seen["refusal"] = error_code(request, "early")
        seen["message"] = tool_result(request, "early").output["error"]["message"]  # type: ignore[index]
        return reply(
            ToolCall("try-a", "delegate", {"goal": "A", "context": "", "item": "a"})
        )

    def after_failure(request: ModelRequest) -> ModelResponse:
        seen["failed"] = tool_result(request, "try-a").output["data"]
        return reply(text="Part A could not be read.")

    model.script(
        main=[
            reply(ToolCall("plan", "checklist_write", PLAN)),
            reply(
                ToolCall("early", "delegate", {"goal": "C", "context": "", "item": "c"})
            ),
            after_refusal,
            after_failure,
        ],
        subruns={
            "A": lambda request: ModelResponse(
                finish_reason=FinishReason.LENGTH, text="cut off"
            )
        },
    )
    agent = await open_job_agent(tmp_path, model)
    try:
        job = await agent.start_job(OBJECTIVE)
        done = await wait_for_job(
            agent, job.job_id, in_state(JobState.DONE), model=model
        )
        assert seen["refusal"] == "checklist_item_blocked"
        assert "a" in str(seen["message"])
        failed = seen["failed"]
        assert isinstance(failed, Mapping) and failed["state"] == "failed"
        # The refused delegation never started; the failed one reopened its item.
        (delegation,) = done.delegations
        assert delegation.item == "a" and delegation.state is DelegationState.FAILED
        assert [item.status for item in done.checklist] == [
            ChecklistStatus.PENDING,
            ChecklistStatus.PENDING,
        ]
    finally:
        await agent.close()


async def test_ask_user_waits_for_the_answer_and_a_new_run_sees_it(
    tmp_path: Path,
) -> None:
    model = JobModel()
    model.script(
        main=[
            reply(ToolCall("q", "ask_user", {"question": "Which month?"})),
            lambda request: reply(
                text="Using " + str(job_payload(request)["answers"][0]["answer"])
            ),
        ]
    )
    agent = await open_job_agent(tmp_path, model)
    try:
        job = await agent.start_job(OBJECTIVE)
        waiting = await wait_for_job(
            agent, job.job_id, in_state(JobState.WAITING_FOR_YOU), model=model
        )
        assert waiting.question == "Which month?"
        assert waiting.run_count == 1
        board = await agent.work_board()
        (card,) = board.columns[WorkColumn.NEEDS_YOU]
        assert card.id == job.job_id and card.summary == "Which month?"
        # The question ended the run: nothing else was asked of the model.
        assert model.kinds() == ["job"]

        await agent.answer_job(job.job_id, "September")
        done = await wait_for_job(
            agent, job.job_id, in_state(JobState.DONE), model=model
        )
        assert done.final_summary == "Using September"
        assert done.run_count == 2
        assert [item.text for item in done.answers] == ["September"]
    finally:
        await agent.close()


async def test_a_full_context_continues_the_job_in_a_new_run_with_its_checklist(
    tmp_path: Path,
) -> None:
    model = JobModel()
    window = model.model_profile.maximum_input_tokens
    full = replace(
        reply(ToolCall("plan", "checklist_write", PLAN)),
        request_input_tokens=int(window * 0.61),
    )

    def resumed(request: ModelRequest) -> ModelResponse:
        payload = job_payload(request)
        assert payload["run"] == 2
        assert [item["id"] for item in payload["checklist"]] == ["a", "c"]
        return reply(text="Resumed from the checklist.")

    # A second step would be needed if the run had not ended on its context.
    model.script(main=[full, resumed])
    agent = await open_job_agent(tmp_path, model)
    try:
        job = await agent.start_job(OBJECTIVE)
        done = await wait_for_job(
            agent, job.job_id, in_state(JobState.DONE), model=model
        )
        assert done.run_count == 2
        assert done.final_summary == "Resumed from the checklist."
        first, second = (request for kind, request in model.requests if kind == "job")
        assert job_payload(first)["run"] == 1
    finally:
        await agent.close()


async def test_the_run_limit_fails_the_job_and_raising_it_continues(
    tmp_path: Path,
) -> None:
    model = JobModel()
    window = model.model_profile.maximum_input_tokens
    keep_going = replace(
        reply(ToolCall("note", "checklist_write", PLAN)),
        request_input_tokens=int(window * 0.9),
    )
    model.script(main=[keep_going] * JOB_RUN_LIMIT + [reply(text="Finished.")])
    agent = await open_job_agent(tmp_path, model)
    try:
        job = await agent.start_job(OBJECTIVE)
        failed = await wait_for_job(
            agent, job.job_id, in_state(JobState.FAILED), model=model
        )
        assert failed.failure_code == "job_run_limit_reached"
        assert failed.run_count == JOB_RUN_LIMIT
        board = await agent.work_board()
        assert [card.id for card in board.columns[WorkColumn.NEEDS_YOU]] == [job.job_id]

        await agent.answer_job(job.job_id, "Keep going.")
        done = await wait_for_job(
            agent, job.job_id, in_state(JobState.DONE), model=model
        )
        assert done.run_limit == 2 * JOB_RUN_LIMIT
        assert done.run_count == JOB_RUN_LIMIT + 1
    finally:
        await agent.close()


async def test_the_token_limit_fails_the_job_and_raising_it_continues(
    tmp_path: Path,
) -> None:
    model = JobModel()
    per_run = 2_000
    heavy = ModelUsage(
        input_tokens=per_run * JOB_TOKEN_LIMIT_RUNS // 2,
        output_tokens=0,
        cost_estimate=CostEstimate.complete(Decimal("0")),
    )
    # Each run spends half the job's allowance and stops at its own limit.
    model.script(
        main=[
            replace(reply(ToolCall("n1", "checklist_write", PLAN)), usage=heavy),
            replace(reply(ToolCall("n2", "checklist_write", PLAN)), usage=heavy),
            reply(text="Finished."),
        ]
    )
    agent = await open_job_agent(
        tmp_path, model, limits=LoopLimits(max_total_tokens=per_run)
    )
    try:
        job = await agent.start_job(OBJECTIVE)
        failed = await wait_for_job(
            agent, job.job_id, in_state(JobState.FAILED), model=model
        )
        assert failed.failure_code == "job_token_limit_reached"
        assert failed.tokens_used == per_run * JOB_TOKEN_LIMIT_RUNS
        assert failed.run_count == 2

        await agent.answer_job(job.job_id, "Spend more.")
        done = await wait_for_job(
            agent, job.job_id, in_state(JobState.DONE), model=model
        )
        assert done.token_limit == 2 * per_run * JOB_TOKEN_LIMIT_RUNS
    finally:
        await agent.close()


async def test_cancel_stops_the_main_run_and_its_subruns(tmp_path: Path) -> None:
    model = JobModel()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hang(request: ModelRequest) -> ModelResponse:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    model.script(
        main=[reply(ToolCall("d", "delegate", {"goal": "Slow", "context": ""}))],
        subruns={"Slow": hang},
    )
    agent = await open_job_agent(tmp_path, model)
    try:
        job = await agent.start_job(OBJECTIVE)
        await asyncio.wait_for(started.wait(), timeout=10)
        stopped = await agent.cancel_job(job.job_id)
        assert stopped is not None and stopped.state is JobState.CANCELLED
        await asyncio.wait_for(cancelled.wait(), timeout=10)
        final = await wait_for_job(
            agent,
            job.job_id,
            lambda current: all(
                item.state is DelegationState.CANCELLED for item in current.delegations
            ),
        )
        assert final.state is JobState.CANCELLED
        assert len(final.delegations) == 1
        # Nothing restarts a cancelled job.
        await asyncio.sleep(0.2)
        assert model.kinds() == ["job", "subrun"]
    finally:
        await agent.close()


async def test_a_restart_resumes_from_the_checklist_and_reports_interrupted_subruns(
    tmp_path: Path,
) -> None:
    model = JobModel()
    started = asyncio.Event()

    async def hang(request: ModelRequest) -> ModelResponse:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    model.script(
        main=[
            reply(ToolCall("plan", "checklist_write", PLAN)),
            reply(
                ToolCall("d", "delegate", {"goal": "Slow", "context": "", "item": "a"})
            ),
        ],
        subruns={"Slow": hang},
    )
    agent = await open_job_agent(tmp_path, model)
    job = await agent.start_job(OBJECTIVE)
    await asyncio.wait_for(started.wait(), timeout=10)
    await agent.close()

    resumed_model = JobModel()

    def resumed(request: ModelRequest) -> ModelResponse:
        payload = job_payload(request)
        (delegation,) = payload["delegations"]
        assert delegation["state"] == "failed"
        assert delegation["summary"] == INTERRUPTED_BY_RESTART
        assert [item["status"] for item in payload["checklist"]] == [
            "pending",
            "pending",
        ]
        return reply(
            ToolCall("again", "delegate", {"goal": "Fast", "context": "", "item": "a"})
        )

    resumed_model.script(
        main=[resumed, reply(text="Done after the restart.")],
        subruns={"Fast": lambda request: reply(text="A is fine.")},
    )
    agent = await open_job_agent(tmp_path, resumed_model, reopen=True)
    try:
        done = await wait_for_job(
            agent, job.job_id, in_state(JobState.DONE), model=resumed_model
        )
        assert done.run_count == 2
        assert done.final_summary == "Done after the restart."
        assert [item.state for item in done.delegations] == [
            DelegationState.FAILED,
            DelegationState.DONE,
        ]
        assert done.checklist[0].status is ChecklistStatus.DONE
    finally:
        await agent.close()
