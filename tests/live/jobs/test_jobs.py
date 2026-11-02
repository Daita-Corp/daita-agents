"""A real model runs background jobs: delegating, resuming, asking and writing.

Opt in with ``DAITA_RUN_LIVE_JOBS=1``; use ``DAITA_LIVE_BILLING=subscription``
to run on the ChatGPT subscription. Checklist rules, limits, cancellation and
restart mechanics are deterministic tests; these cases only show a real model
uses the job tools to finish.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from daita import Agent, ApprovalDecision, ApprovalRequest, LoopLimits, SQLiteSource
from daita.adapters.mcp import SDKMCPClientFactory
from daita.distribution import DeliverySubjectKind, OutcomeState
from daita.jobs import ChecklistStatus, DelegationState, Job, JobState
from tests.support.jobs import in_state, wait_for_job
from tests.support.live_billing import cost_cap, live_provider
from tests.support.mcp import mock_transport
from tests.support.native_writes import create_fixture
from tests.support.workspace import workspace_for

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_LIVE_JOBS") != "1",
        reason="requires explicit live background-job authorization",
    ),
]

_MODEL = os.environ.get("DAITA_LIVE_JOBS_MODEL_ID", "openai:gpt-5.6-terra")
_JOB_TIMEOUT = 15 * 60
_LIMITS = LoopLimits(
    max_steps=24,
    max_total_tokens=200_000,
    max_wall_time_seconds=600,
    max_estimated_cost_usd=cost_cap(Decimal("1.00")),
)
_TWO_PARTS = (
    "Compare the orders and the refunds tables. These are two independent "
    "parts: hand each table's summary (row count and total amount) to its own "
    "sub-run, then combine them into one short comparison."
)


def _api_key() -> str:
    key = os.environ.get("DAITA_LIVE_JOBS_LLM_API_KEY", "")
    if not key.strip():
        pytest.fail("DAITA_LIVE_JOBS_LLM_API_KEY is required under API billing")
    return key


def _provider() -> tuple[Any, Any]:
    return live_provider(_MODEL, api_key=_api_key, max_output_tokens=4_096)


def _sales_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL NOT NULL);
            CREATE TABLE refunds (id INTEGER PRIMARY KEY, amount REAL NOT NULL);
            INSERT INTO orders (amount) VALUES (10.0), (20.0), (30.5);
            INSERT INTO refunds (amount) VALUES (5.0), (2.5);
            """)


async def _sales_agent(root: Path, name: str) -> tuple[Agent, Callable[[], Any]]:
    profile, provider = _provider()
    database = root / "sales.db"
    root.mkdir(parents=True, exist_ok=True)
    _sales_database(database)

    def open_agent() -> Any:
        return Agent.open(
            name,
            root=root,
            model=provider,
            model_profile=profile,
            limits=_LIMITS,
            workspace=workspace_for(root),
        )

    created = await Agent.create(
        name,
        root=root,
        model=provider,
        model_profile=profile,
        limits=_LIMITS,
        workspace=workspace_for(root),
    )
    await created.attach(SQLiteSource(database, name="Sales"))
    return created, open_agent


def _assert_compared(job: Job) -> None:
    """The job read both tables: no open items and both real totals reported."""

    assert job.final_summary
    open_items = [
        item.title
        for item in job.checklist
        if item.status not in {ChecklistStatus.DONE, ChecklistStatus.CANCELLED}
    ]
    assert not open_items, open_items
    assert any(item.status is ChecklistStatus.DONE for item in job.checklist)
    # orders total 60.5 over 3 rows; refunds total 7.5 over 2 rows.
    assert "60.5" in job.final_summary, job.final_summary
    assert "7.5" in job.final_summary, job.final_summary


def _record(record_property: Callable[[str, object], None], job: Job) -> None:
    record_property("job_state", job.state.value)
    record_property("run_count", job.run_count)
    record_property("delegations", [item.state.value for item in job.delegations])
    record_property("checklist", [item.status.value for item in job.checklist])


async def test_live_two_part_analysis_job_delegates_and_delivers(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    agent, _open = await _sales_agent(tmp_path / "root", "live-jobs")
    try:
        job = await agent.start_job(_TWO_PARTS)
        done = await wait_for_job(
            agent,
            job.job_id,
            in_state(JobState.DONE, JobState.FAILED, JobState.WAITING_FOR_YOU),
            timeout=_JOB_TIMEOUT,
        )
        _record(record_property, done)
        assert done.state is JobState.DONE, done.failure_code or done.question
        assert done.checklist, "the job planned with checklist_write"
        finished = [
            item for item in done.delegations if item.state is DelegationState.DONE
        ]
        assert len(finished) >= 2, "each part went to its own sub-run"
        _assert_compared(done)
        (delivery,) = await agent.list_deliveries(conversation_id=job.conversation_id)
        assert delivery.subject_kind is DeliverySubjectKind.JOB
        assert delivery.conclusion_state is OutcomeState.SUCCEEDED
    finally:
        await agent.close()


async def test_live_job_interrupted_by_a_restart_resumes_and_finishes(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    agent, open_agent = await _sales_agent(tmp_path / "root", "live-restart")
    try:
        job = await agent.start_job(_TWO_PARTS)
        # Restart as soon as the plan is saved, before the job can finish.
        planned = await wait_for_job(
            agent,
            job.job_id,
            lambda current: bool(current.checklist)
            or current.state is not JobState.RUNNING,
            timeout=_JOB_TIMEOUT,
        )
        assert planned.state is JobState.RUNNING, "the job finished before the restart"
    finally:
        await agent.close()
    agent = await open_agent()
    try:
        done = await wait_for_job(
            agent,
            job.job_id,
            in_state(JobState.DONE, JobState.FAILED, JobState.WAITING_FOR_YOU),
            timeout=_JOB_TIMEOUT,
        )
        _record(record_property, done)
        assert done.state is JobState.DONE, done.failure_code or done.question
        # The restart started a new main run from the saved checklist.
        assert done.run_count >= 2
        _assert_compared(done)
    finally:
        await agent.close()


async def test_live_job_asks_and_uses_the_answer(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    agent, _open = await _sales_agent(tmp_path / "root", "live-ask")
    try:
        job = await agent.start_job(
            "Summarize one of the sales tables for me. I haven't decided which "
            "one yet: ask me which table I want before you query anything, then "
            "report that table's row count and total amount."
        )
        waiting = await wait_for_job(
            agent,
            job.job_id,
            in_state(JobState.DONE, JobState.FAILED, JobState.WAITING_FOR_YOU),
            timeout=_JOB_TIMEOUT,
        )
        assert waiting.state is JobState.WAITING_FOR_YOU, (
            waiting.failure_code or waiting.final_summary
        )
        assert waiting.question
        await agent.answer_job(job.job_id, "The refunds table, please.")
        done = await wait_for_job(
            agent,
            job.job_id,
            in_state(JobState.DONE, JobState.FAILED, JobState.WAITING_FOR_YOU),
            timeout=_JOB_TIMEOUT,
        )
        _record(record_property, done)
        assert done.state is JobState.DONE, done.failure_code or done.question
        # The answer started a new main run, which used it.
        assert done.run_count >= 2
        assert [item.text for item in done.answers] == ["The refunds table, please."]
        assert done.final_summary and "7.5" in done.final_summary, done.final_summary
    finally:
        await agent.close()


async def test_live_job_performs_one_approved_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    fixture, _scripted, db, alpha, *_rest, batch, clock, _approvals = (
        await create_fixture(tmp_path, monkeypatch)
    )
    await fixture.close()
    profile, provider = _provider()
    asked: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.open(
        "native-acceptance",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=profile,
        limits=_LIMITS,
        clock=lambda: clock[0],
        mcp_client_factory=SDKMCPClientFactory(http_transport=mock_transport(alpha)),
        approval_handler=approve,
    )
    (row,) = batch["rows"]
    try:
        job = await agent.start_job(
            "Add one company to the companies table: domain "
            f"{row['domain']}, name {row['name']}, evidence URL "
            f"{row['evidence_url']}. Upsert it on domain, then report what you saved.",
            sources=(str(batch["source_id"]),),
        )
        done = await wait_for_job(
            agent,
            job.job_id,
            in_state(JobState.DONE, JobState.FAILED, JobState.WAITING_FOR_YOU),
            timeout=_JOB_TIMEOUT,
        )
        _record(record_property, done)
        assert done.state is JobState.DONE, done.failure_code or done.question
        writes = [
            request for request in asked if request.capability_id == "data.upsert_rows"
        ]
        assert len(writes) == 1 and writes[0].requester
        inserts = [
            item
            for item in db.log
            if item[0] == "fetch" and item[1].startswith("INSERT")
        ]
        assert len(inserts) == 1
        assert db.rows[row["domain"]]["name"] == row["name"]
        receipts = await agent.list_effects()
        assert len(receipts) == 1
    finally:
        await agent.close()
