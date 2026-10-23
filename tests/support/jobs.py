"""A scripted model for background jobs and bounded waits on job records."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from daita import Agent, AgentConfig
from daita.config import HostCapacityLimits
from daita.jobs import Job, JobState
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from tests.support.workspace import workspace_for

# Each request reports this measured usage, so job token accounting is exact.
REQUEST_USAGE = ModelUsage(
    input_tokens=120,
    output_tokens=30,
    cost_estimate=CostEstimate.complete(Decimal("0")),
)
Step = ModelResponse | Callable[[ModelRequest], ModelResponse]
SubrunStep = Callable[[ModelRequest], ModelResponse | Awaitable[ModelResponse]]


def reply(*calls: ToolCall, text: str | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=tuple(calls),
        text=text,
        usage=REQUEST_USAGE,
    )


def run_kind(request: ModelRequest) -> str:
    """``job``, ``subrun`` or ``user``, read from the run's code-owned start."""

    for message in request.messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                if "<untrusted_job_subrun_payload>" in block.text:
                    return "subrun"
                if "<untrusted_job_payload>" in block.text:
                    return "job"
    return "user"


def job_payload(request: ModelRequest) -> dict[str, Any]:
    """The untrusted payload a job or sub-run started with."""

    for message in request.messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                for label in ("untrusted_job_payload", "untrusted_job_subrun_payload"):
                    start = f"<{label}>\n"
                    if start in block.text:
                        body = block.text.split(start, 1)[1]
                        return json.loads(body.split(f"\n</{label}>", 1)[0])
    raise AssertionError("request has no job payload")


def tool_result(request: ModelRequest, call_id: str) -> ToolResultBlock:
    return next(
        block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )


def tool_names(request: ModelRequest) -> frozenset[str]:
    return frozenset(tool.name for tool in request.tools)


class JobModel:
    """Answers main runs, sub-runs and foreground runs from separate scripts.

    Sub-runs of one step run in parallel, so they are answered by their goal
    rather than in order.
    """

    provider_id = "mock:jobs"
    model_profile = ModelProfile(
        id=provider_id,
        context_window_tokens=64_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )

    def __init__(self) -> None:
        self.main: list[Step] = []
        self.user: list[Step] = []
        self.subruns: dict[str, SubrunStep] = {}
        self.requests: list[tuple[str, ModelRequest]] = []
        self.errors: list[Exception] = []

    def script(
        self,
        *,
        main: Iterable[Step] = (),
        user: Iterable[Step] = (),
        subruns: dict[str, SubrunStep] | None = None,
    ) -> None:
        self.main.extend(main)
        self.user.extend(user)
        self.subruns.update(subruns or {})

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            return await self._generate(request)
        except Exception as error:
            # The provider boundary reports any failure as malformed output;
            # keep the real one for the test to show.
            self.errors.append(error)
            raise

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        kind = run_kind(request)
        self.requests.append((kind, request))
        if kind == "subrun":
            goal = job_payload(request)["goal"]
            assert isinstance(goal, str)
            answer = self.subruns[goal](request)
            return await answer if inspect.isawaitable(answer) else answer
        steps = self.main if kind == "job" else self.user
        assert steps, f"{kind} script exhausted"
        step = steps.pop(0)
        return step(request) if callable(step) else step

    def kinds(self) -> list[str]:
        return [kind for kind, _request in self.requests]


def stored_job(
    job_id: str = "job-1",
    agent_id: str = "agent-1",
    *,
    at: datetime = datetime(2026, 9, 16, 12, tzinfo=UTC),
) -> Job:
    """One running job record, for storage contracts."""

    return Job(
        job_id=job_id,
        agent_id=agent_id,
        conversation_id="conversation-1",
        objective="Check the stored record.",
        source_ids=(),
        state=JobState.RUNNING,
        created_at=at,
        updated_at=at,
    )


async def open_job_agent(
    root: Path,
    model: JobModel,
    *,
    name: str = "jobs",
    reopen: bool = False,
    **options: Any,
) -> Agent:
    """A library-mode agent whose job supervisor runs in this process."""

    if not reopen:
        created = await Agent.create(name, root=root, workspace=workspace_for(root))
        await created.close()
    options.setdefault(
        # Two background model calls at once, so parallel sub-runs can overlap.
        "config",
        AgentConfig(host_capacity=HostCapacityLimits(provider_capacity=3)),
    )
    return await Agent.open(
        name,
        root=root,
        workspace=workspace_for(root),
        model=model,
        model_profile=model.model_profile,
        **options,
    )


async def wait_for_job(
    agent: Agent,
    job_id: str,
    until: Callable[[Job], bool],
    *,
    model: JobModel | None = None,
    timeout: float = 20.0,
) -> Job:
    """Poll one job record until ``until`` holds, within a bounded wait.

    A scripted step that fails is raised as itself, not as the provider error
    the loop records for it.
    """

    deadline = asyncio.get_running_loop().time() + timeout
    latest: Job | None = None
    while asyncio.get_running_loop().time() < deadline:
        if model is not None and model.errors:
            raise model.errors[0]
        latest = await agent.inspect_job(job_id)
        assert latest is not None
        if until(latest):
            return latest
        await asyncio.sleep(0.02)
    raise AssertionError(f"job did not reach the expected state: {latest!r}")


def in_state(*states: JobState) -> Callable[[Job], bool]:
    return lambda job: job.state in states


__all__ = [
    "REQUEST_USAGE",
    "SubrunStep",
    "JobModel",
    "in_state",
    "open_job_agent",
    "job_payload",
    "reply",
    "run_kind",
    "stored_job",
    "tool_names",
    "tool_result",
    "wait_for_job",
]
