"""A job's writes ask a person at the time, commit once, and are never replayed."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from daita import Agent
from daita.adapters.mcp import SDKMCPClientFactory
from daita.capabilities import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    EffectOutcome,
)
from daita.jobs import DelegationState, JobState
from daita.llm.models import ModelRequest, ModelResponse, ToolCall
from tests.support.jobs import (
    JobModel,
    in_state,
    job_payload,
    reply,
    tool_result,
    wait_for_job,
)
from tests.support.mcp import mock_transport
from tests.support.native_writes import create_fixture
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration

OBJECTIVE = "Record New Co in the companies table."


def write_steps(batch: Mapping[str, object], *, then: ModelResponse) -> list[Any]:
    """Load the upsert tool, preview the fixture's one row, apply it, then ``then``."""

    def apply(request: ModelRequest) -> ModelResponse:
        preview = tool_result(request, "preview")
        assert not preview.is_error, preview.output
        data = preview.output["data"]
        assert isinstance(data, Mapping)
        return reply(
            ToolCall(
                "write",
                "data_upsert_rows",
                {
                    **batch,
                    "mode": "apply",
                    "preview_fingerprint": data["preview_fingerprint"],
                },
            )
        )

    def after(request: ModelRequest) -> ModelResponse:
        write = tool_result(request, "write")
        assert not write.is_error, write.output
        return then

    return [
        reply(ToolCall("load", "toolbox_load", {"tool_names": ["data_upsert_rows"]})),
        reply(ToolCall("preview", "data_upsert_rows", {**batch, "mode": "preview"})),
        apply,
        after,
    ]


async def reopen(
    root: Path, model: JobModel, alpha: Any, clock: list[Any], approve: ApprovalHandler
) -> Agent:
    return await Agent.open(
        "native-acceptance",
        root=root,
        workspace=workspace_for(root),
        model=model,
        model_profile=model.model_profile,
        clock=lambda: clock[0],
        mcp_client_factory=SDKMCPClientFactory(http_transport=mock_transport(alpha)),
        approval_handler=approve,
    )


def inserts(db: Any) -> int:
    return len(
        [item for item in db.log if item[0] == "fetch" and item[1].startswith("INSERT")]
    )


async def test_a_job_write_asks_commits_once_and_is_listed_as_committed_after_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _provider, db, alpha, *_rest, batch, clock, _approvals = (
        await create_fixture(tmp_path, monkeypatch)
    )
    await agent.close()
    asked: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request)
        return ApprovalDecision.APPROVE

    model = JobModel()
    seen: dict[str, object] = {}

    def resumed(request: ModelRequest) -> ModelResponse:
        seen["payload"] = job_payload(request)
        return reply(text="New Co is recorded.")

    model.script(
        main=[
            *write_steps(
                batch,
                then=reply(ToolCall("q", "ask_user", {"question": "Anything else?"})),
            ),
            resumed,
        ]
    )
    agent = await reopen(tmp_path, model, alpha, clock, approve)
    try:
        job = await agent.start_job(OBJECTIVE)
        await wait_for_job(
            agent, job.job_id, in_state(JobState.WAITING_FOR_YOU), model=model
        )
        (request,) = asked
        assert request.capability_id == "data.upsert_rows"
        assert request.requester == OBJECTIVE
        assert inserts(db) == 1 and db.rows["new.test"]["name"] == "New Co"
        (receipt,) = await agent.list_effects()
        assert receipt.outcome is EffectOutcome.SUCCEEDED
        assert receipt.routine_id is None and receipt.capability_grant_digest is None

        await agent.answer_job(job.job_id, "No.")
        await wait_for_job(agent, job.job_id, in_state(JobState.DONE), model=model)
        payload = seen["payload"]
        assert isinstance(payload, Mapping)
        (committed,) = payload["committed_writes"]
        assert committed["receipt_id"] == receipt.receipt_id
        # Resuming never replays the write, and asks nothing more.
        assert inserts(db) == 1 and len(asked) == 1
    finally:
        await agent.close()


async def test_a_subrun_write_pending_at_restart_is_never_dispatched_without_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _provider, db, alpha, *_rest, batch, clock, _approvals = (
        await create_fixture(tmp_path, monkeypatch)
    )
    await agent.close()
    pending = asyncio.Event()

    async def never_answered(request: ApprovalRequest) -> ApprovalDecision:
        pending.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    model = JobModel()
    subrun_steps = write_steps(batch, then=reply(text="Recorded."))
    model.script(
        main=[reply(ToolCall("d", "delegate", {"goal": "Write", "context": ""}))],
        subruns={"Write": lambda request: _next(subrun_steps, request)},
    )
    agent = await reopen(tmp_path, model, alpha, clock, never_answered)
    job = await agent.start_job(OBJECTIVE)
    await asyncio.wait_for(pending.wait(), timeout=20)
    assert inserts(db) == 0
    await agent.close()
    assert inserts(db) == 0

    asked: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request)
        return ApprovalDecision.APPROVE

    resumed_model = JobModel()

    def resumed(request: ModelRequest) -> ModelResponse:
        payload = job_payload(request)
        # The pending write was never reserved, so nothing is committed.
        assert payload["committed_writes"] == []
        (delegation,) = payload["delegations"]
        assert delegation["state"] == "failed"
        return reply(ToolCall("again", "delegate", {"goal": "Write", "context": ""}))

    retry_steps = write_steps(batch, then=reply(text="Recorded."))
    resumed_model.script(
        main=[resumed, reply(text="New Co is recorded.")],
        subruns={"Write": lambda request: _next(retry_steps, request)},
    )
    agent = await reopen(tmp_path, resumed_model, alpha, clock, approve)
    try:
        done = await wait_for_job(
            agent, job.job_id, in_state(JobState.DONE), model=resumed_model
        )
        (request,) = asked
        assert request.requester == OBJECTIVE
        assert inserts(db) == 1
        assert [item.state for item in done.delegations] == [
            DelegationState.FAILED,
            DelegationState.DONE,
        ]
        (receipt,) = await agent.list_effects()
        assert receipt.outcome is EffectOutcome.SUCCEEDED
    finally:
        await agent.close()


def _next(steps: list[Any], request: ModelRequest) -> ModelResponse:
    step = steps.pop(0)
    return step(request) if callable(step) else step
