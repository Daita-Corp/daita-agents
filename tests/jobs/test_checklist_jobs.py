"""Background jobs plan with a checklist, delegate sub-runs and finish in the host."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from daita import Agent, SQLiteSource, WorkColumn, WorkKind
from daita.distribution import DeliverySubjectKind, OutcomeState
from daita.jobs import ChecklistStatus, DelegationState, JobState
from daita.llm.models import ModelRequest, ModelResponse, TextBlock, ToolCall
from tests.support.jobs import (
    REQUEST_USAGE,
    JobModel,
    SubrunStep,
    in_state,
    job_payload,
    open_job_agent,
    reply,
    tool_names,
    tool_result,
    wait_for_job,
)
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration

OBJECTIVE = "Compare part A and part B of last month's sales."
PLAN = {
    "items": [
        {"id": "a", "title": "Analyse part A"},
        {"id": "b", "title": "Analyse part B"},
        {"id": "c", "title": "Combine the parts", "after": ["a", "b"]},
    ]
}


def both_started() -> tuple[asyncio.Barrier, Callable[[str], SubrunStep]]:
    """Sub-run responders that each finish only once both sub-runs are running."""

    barrier = asyncio.Barrier(2)

    def answer(text: str):
        async def respond(request: ModelRequest) -> ModelResponse:
            await asyncio.wait_for(barrier.wait(), timeout=10)
            return reply(text=text)

        return respond

    return barrier, answer


async def test_a_job_plans_delegates_two_parallel_subruns_and_delivers(
    tmp_path: Path,
) -> None:
    model = JobModel()
    _barrier, answer = both_started()

    def combine(request: ModelRequest) -> ModelResponse:
        first, second = tool_result(request, "da"), tool_result(request, "db")
        assert first.output["data"]["state"] == "done"  # type: ignore[index]
        assert second.output["data"]["summary"] == "B has 4 rows."  # type: ignore[index]
        return reply(
            ToolCall(
                "finish",
                "checklist_write",
                {"items": [{"id": "c", "status": "done"}], "merge": True},
            )
        )

    model.script(
        user=[
            reply(ToolCall("load", "toolbox_load", {"tool_names": ["job_start"]})),
            reply(ToolCall("start", "job_start", {"objective": OBJECTIVE})),
            reply(text="Started a background job."),
        ],
        main=[
            reply(ToolCall("plan", "checklist_write", PLAN)),
            reply(
                ToolCall(
                    "da",
                    "delegate",
                    {"goal": "Part A", "context": "Sales table", "item": "a"},
                ),
                ToolCall(
                    "db",
                    "delegate",
                    {"goal": "Part B", "context": "Sales table", "item": "b"},
                ),
            ),
            combine,
            reply(text="Part A has 3 rows and part B has 4."),
        ],
        subruns={"Part A": answer("A has 3 rows."), "Part B": answer("B has 4 rows.")},
    )
    agent = await open_job_agent(tmp_path, model)
    try:
        origin = await agent.run("Compare the two parts in the background.")
        assert origin.final_text == "Started a background job."
        (job,) = await agent.list_jobs()
        assert job.conversation_id == origin.conversation_id
        done = await wait_for_job(
            agent, job.job_id, in_state(JobState.DONE), model=model
        )

        assert done.final_summary == "Part A has 3 rows and part B has 4."
        assert [item.status for item in done.checklist] == [ChecklistStatus.DONE] * 3
        assert {item.item: item.state for item in done.delegations} == {
            "a": DelegationState.DONE,
            "b": DelegationState.DONE,
        }
        assert done.run_count == 1
        requests = model.kinds().count("job") + model.kinds().count("subrun")
        assert done.tokens_used == requests * REQUEST_USAGE.total_tokens
        # Only a job's main run has the job-run tools; sub-runs have none.
        main_tools = {
            name
            for kind, request in model.requests
            if kind == "job"
            for name in tool_names(request)
        }
        assert {"checklist_write", "delegate", "ask_user"} <= main_tools
        assert "job_start" not in main_tools
        for kind, request in model.requests:
            if kind == "subrun":
                assert not tool_names(request) & {
                    "checklist_write",
                    "delegate",
                    "ask_user",
                    "job_start",
                }
                assert job_payload(request)["context"] == "Sales table"

        (delivery,) = await agent.list_deliveries(
            conversation_id=origin.conversation_id
        )
        assert delivery.subject_kind is DeliverySubjectKind.JOB
        assert delivery.subject_id == job.job_id
        assert delivery.conclusion_state is OutcomeState.SUCCEEDED
        assert delivery.conclusion_preview == done.final_summary
        board = await agent.work_board()
        (card,) = board.columns[WorkColumn.DONE]
        assert card.kind is WorkKind.JOB and card.id == job.job_id
    finally:
        await agent.close()


async def test_job_runs_find_their_tables_from_the_objective_and_the_goal(
    tmp_path: Path,
) -> None:
    """A job run cannot search the catalog, so its context matches the work."""

    database = tmp_path / "sales.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL);"
            "CREATE TABLE refunds (id INTEGER PRIMARY KEY, amount REAL);"
        )
    created = await Agent.create(
        "jobs", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await created.attach(SQLiteSource(database, name="Sales"))
    await created.close()

    def tables(request: ModelRequest) -> set[str]:
        text = "".join(
            block.text
            for message in request.messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        return {name for name in ("orders", "refunds") if f'"name":"{name}"' in text}

    seen: dict[str, set[str]] = {}

    def main(request: ModelRequest) -> ModelResponse:
        seen["main"] = tables(request)
        return reply(
            ToolCall(
                "d", "delegate", {"goal": "Total the refunds table.", "context": ""}
            )
        )

    def subrun(request: ModelRequest) -> ModelResponse:
        seen["subrun"] = tables(request)
        return reply(text="Refunds total 0.")

    model = JobModel()
    model.script(
        main=[main, reply(text="Compared.")],
        subruns={"Total the refunds table.": subrun},
    )
    agent = await open_job_agent(tmp_path, model, reopen=True)
    try:
        job = await agent.start_job("Compare the orders table with the refunds table.")
        await wait_for_job(agent, job.job_id, in_state(JobState.DONE), model=model)
        assert seen["main"] == {"orders", "refunds"}
        assert "refunds" in seen["subrun"]
    finally:
        await agent.close()
