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
from daita.llm.models import (
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    model_tool_result,
)
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
from tests.support.model_surface import identity_leaks
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


def _rendered(request: ModelRequest, call_id: str) -> dict[str, object]:
    output = model_tool_result(
        tool_result(request, call_id), None, request.scope_names
    )["output"]
    assert isinstance(output, dict)
    return output


def _two_databases(root: Path) -> tuple[Path, Path]:
    shop, staff = root / "shop.db", root / "staff.db"
    with sqlite3.connect(shop) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    with sqlite3.connect(staff) as connection:
        connection.execute("CREATE TABLE people (id INTEGER PRIMARY KEY)")
    return shop, staff


async def test_job_runs_show_only_names_and_stay_inside_their_sources(
    tmp_path: Path,
) -> None:
    shop, staff = _two_databases(tmp_path)
    refusals: dict[str, dict[str, object]] = {}

    def record(request: ModelRequest, *call_ids: str) -> None:
        for call_id in call_ids:
            refusals[call_id] = _rendered(request, call_id)

    def keep_going(call: ToolCall, *call_ids: str):
        def step(request: ModelRequest) -> ModelResponse:
            record(request, *call_ids)
            return reply(call)

        return step

    def finish(request: ModelRequest) -> ModelResponse:
        record(request, "subrun")
        return reply(text="Counted.")

    model = JobModel()
    sql = {"source": "shop", "sql": "SELECT COUNT(*) AS c FROM orders"}
    model.script(
        main=[
            reply(ToolCall("read", "data_query", sql)),
            keep_going(
                ToolCall("other-source", "data_query", {**sql, "source": "staff"}),
                "read",
            ),
            keep_going(
                ToolCall(
                    "other-table",
                    "data_query",
                    {"source": "shop", "sql": "SELECT * FROM people"},
                ),
                "other-source",
            ),
            keep_going(
                ToolCall("describe", "catalog_describe", {"tables": ["main.people"]}),
                "other-table",
            ),
            keep_going(
                ToolCall(
                    "sub-outside",
                    "delegate",
                    {"goal": "Other", "context": "", "sources": ["staff"]},
                ),
                "describe",
            ),
            keep_going(
                ToolCall(
                    "sub-unknown",
                    "delegate",
                    {"goal": "Unknown", "context": "", "sources": ["warehouse"]},
                ),
                "sub-outside",
            ),
            keep_going(
                ToolCall(
                    "subrun",
                    "delegate",
                    {"goal": "Count orders", "context": "", "sources": ["shop"]},
                ),
                "sub-unknown",
            ),
            finish,
        ],
        subruns={"Count orders": lambda request: reply(text="None yet.")},
    )
    agent = await open_job_agent(tmp_path, model)
    try:
        await agent.attach(SQLiteSource(shop, name="shop"))
        await agent.attach(SQLiteSource(staff, name="staff"))
        job = await agent.start_job("Count the orders table.", sources=("shop",))
        await wait_for_job(agent, job.job_id, in_state(JobState.DONE), model=model)

        model.script(
            user=[
                reply(ToolCall("inspect", "job_inspect", {"job_id": job.job_id})),
                reply(text="Done."),
            ]
        )
        await agent.run("How did the job go?")
    finally:
        await agent.close()

    assert refusals["read"]["kind"] == "data.query_result"
    assert refusals["other-source"]["error"]["code"] == "source_unknown"  # type: ignore[index]
    assert refusals["other-table"]["error"]["code"] == "query_invalid"  # type: ignore[index]
    assert refusals["describe"]["error"]["code"] == "table_unknown"  # type: ignore[index]
    for call_id in ("sub-outside", "sub-unknown"):
        assert refusals[call_id]["error"]["code"] == "job_source_unknown"  # type: ignore[index]
    # A refusal lists only what this run may read.
    assert "staff" not in str(refusals["sub-unknown"])
    assert refusals["subrun"]["data"]["summary"] == "None yet."  # type: ignore[index]
    inspected = _rendered(model.requests[-1][1], "inspect")
    assert inspected["data"]["sources"] == ["shop"]  # type: ignore[index]
    assert identity_leaks(request for _, request in model.requests) == []


async def test_a_job_started_from_a_narrowed_run_stays_inside_it(
    tmp_path: Path,
) -> None:
    shop, staff = _two_databases(tmp_path)
    load = {"tool_names": ["job_start"]}
    start = {"objective": "Report."}
    model = JobModel()
    model.script(
        user=[
            reply(ToolCall("load", "toolbox_load", load)),
            reply(
                ToolCall("outside", "job_start", {**start, "sources": ["staff"]}),
            ),
            reply(ToolCall("unnamed", "job_start", start)),
            reply(text="Narrowed."),
            reply(ToolCall("load-all", "toolbox_load", load)),
            reply(ToolCall("every-source", "job_start", start)),
            reply(text="Everything."),
        ],
        main=[reply(text="Reported."), reply(text="Reported.")],
    )
    agent = await open_job_agent(tmp_path, model)
    try:
        first = await agent.attach(SQLiteSource(shop, name="shop"))
        await agent.attach(SQLiteSource(staff, name="staff"))
        await agent.run("Report on the shop.", source_scope_ids=(first.id,))
        (narrowed,) = await agent.list_jobs()
        assert narrowed.source_ids == (first.id,)
        refused = next(
            tool_result(request, "outside")
            for kind, request in model.requests
            if kind == "user" and "outside" in str(request.messages)
        )
        assert refused.output["error"]["code"] == "job_source_unknown"  # type: ignore[index]
        assert refused.output["error"]["details"]["recovery"]["constraint"] == (  # type: ignore[index]
            "Current sources: shop"
        )
        await wait_for_job(agent, narrowed.job_id, in_state(JobState.DONE), model=model)

        # A run over every source keeps a job that reads every current source.
        await agent.run("Report on everything.")
        jobs = await agent.list_jobs()
        assert {job.source_ids for job in jobs} == {(first.id,), ()}
        for job in jobs:
            await wait_for_job(agent, job.job_id, in_state(JobState.DONE), model=model)
    finally:
        await agent.close()


async def test_job_start_refuses_a_name_two_sources_share(tmp_path: Path) -> None:
    shop, staff = _two_databases(tmp_path)
    model = JobModel()
    model.script(
        user=[
            reply(ToolCall("load", "toolbox_load", {"tool_names": ["job_start"]})),
            reply(
                ToolCall(
                    "shared",
                    "job_start",
                    {"objective": "Report.", "sources": ["sales"]},
                )
            ),
            reply(text="Refused."),
        ]
    )
    agent = await open_job_agent(tmp_path, model)
    try:
        await agent.attach(SQLiteSource(shop, name="Sales"))
        await agent.attach(SQLiteSource(staff, name="SALES"))
        await agent.run("Report on sales.")
        assert await agent.list_jobs() == ()
    finally:
        await agent.close()
    refused = tool_result(model.requests[-1][1], "shared")
    assert refused.output["error"]["code"] == "source_ambiguous"  # type: ignore[index]
    assert "SALES (sqlite)" in str(refused.output)


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
