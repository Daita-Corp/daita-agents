"""Authorized live-model acceptance coverage for Stage B durable jobs.

These tests ask a real model to choose between immediate and durable work,
discover the on-demand start/cancel tools, detach after admission, recover a later
result through direct agent-scoped lifecycle reads, and request cancellation from
a new conversation. Supervisor fencing, crash recovery, storage failures,
concurrency limits, and uncertain external outcomes remain deterministic tests.

Running the complete module authorizes at most five ``Agent.run`` interactions.
Each interaction has its own explicit token, wall-time, step, and estimated-cost
ceiling. One interaction can contain several provider requests as the direct
model/tool loop progresses.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent, JobStatus, LoopLimits, SQLiteSource, create_llm_provider
from daita._json import canonical_json
from daita.domains.data.profile_jobs import DATA_PROFILE_EXECUTION_CAPABILITY_ID
from daita.llm.models import (
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    ToolResultBlock,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import (
    ModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
)
from daita.loop.models import (
    LoopExit,
    LoopExitKind,
    Transcript,
    validate_completed_transcript,
)

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_B_JOBS"
_MODEL_ID = "DAITA_STAGE_B_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_STAGE_B_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_STAGE_B_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

_IMMEDIATE_TOKEN = "STAGE_B_IMMEDIATE_SENTINEL_4D8A1C"
_IMMEDIATE_EXPECTATION = f"STAGE_B_IMMEDIATE_CHECK token={_IMMEDIATE_TOKEN} amount=41"
_PROFILE_EXPECTATION = "STAGE_B_PROFILE_CHECK sampled_rows=5 nullable_code_nulls=2"
_LIFECYCLE_TOOLS = frozenset(
    {"job_list", "job_inspect", "job_read_results", "job_cancel"}
)
_TERMINAL_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.NEEDS_ATTENTION,
    }
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing up to "
            "five live Agent.run interactions for the Stage B jobs suite; each "
            f"interaction is capped by {_MAX_COST}"
        ),
    ),
]


class _RecordingProvider:
    """Capture canonical requests and responses around one real provider."""

    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self.requests: list[ModelRequest] = []
        self.responses: list[ModelResponse] = []

    @property
    def provider_id(self) -> str:
        return self._delegate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._delegate.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return provider_has_complete_pricing(self._delegate, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = await self._delegate.generate(request)
        self.responses.append(response)
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(self._delegate, StreamingModelProvider):
            raise TypeError("the live delegate must support canonical streaming")
        self.requests.append(request)
        async for event in self._delegate.stream(request):
            if isinstance(event, ModelStreamCompleted):
                self.responses.append(event.response)
            yield event


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        pytest.fail(f"{name} must be set for the authorized live test")
    return value


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.15")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


def _live_provider() -> tuple[ModelProfile, _RecordingProvider]:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    provider_name = model_id.partition(":")[0]
    if provider_name not in _API_PROVIDERS:
        pytest.fail(
            f"{_MODEL_ID} must name an API-backed provider: "
            + ", ".join(sorted(_API_PROVIDERS))
        )
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")
    if not profile.supports_streaming:
        pytest.fail(f"{_MODEL_ID} must name a model with reviewed streaming support")
    delegate = create_llm_provider(
        model_id,
        api_key=_required_environment(_MODEL_KEY),
        max_output_tokens=min(profile.max_output_tokens, 1_536),
    )
    return profile, _RecordingProvider(delegate)


def _limits() -> LoopLimits:
    return LoopLimits(
        max_steps=12,
        max_total_tokens=30_000,
        max_wall_time_seconds=180,
        max_estimated_cost_usd=_cost_limit(),
    )


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE stage_b_immediate_probe (
                probe_name TEXT PRIMARY KEY,
                verification_token TEXT NOT NULL,
                amount INTEGER NOT NULL
            );
            CREATE TABLE stage_b_profile_probe (
                id INTEGER PRIMARY KEY,
                nullable_code TEXT,
                measure REAL NOT NULL
            );
            """)
        connection.execute(
            "INSERT INTO stage_b_immediate_probe VALUES (?, ?, ?)",
            ("immediate", _IMMEDIATE_TOKEN, 41),
        )
        connection.executemany(
            "INSERT INTO stage_b_profile_probe VALUES (?, ?, ?)",
            (
                (1, "alpha", 10.5),
                (2, None, 11.5),
                (3, "beta", 12.5),
                (4, None, 13.5),
                (5, "alpha", 14.5),
            ),
        )


async def _new_agent(
    tmp_path: Path,
    name: str,
    provider: _RecordingProvider,
    profile: ModelProfile,
) -> tuple[Agent, str]:
    database = tmp_path / f"{name}.sqlite"
    _database(database)
    agent = await Agent.create(
        name,
        root=tmp_path / f"{name}-root",
        model=provider,
        model_profile=profile,
        limits=_limits(),
        workspace=workspace_for(tmp_path / f"{name}-root"),
    )
    source = await agent.attach(SQLiteSource(database, name="Stage B live probes"))
    return agent, source.id


def _successful_logical_results(
    transcript: Transcript,
) -> tuple[tuple[str, ToolResultBlock], ...]:
    outer_names = {
        call.id: call.name
        for message in transcript.messages
        for call in message.tool_calls
    }
    logical: list[tuple[str, ToolResultBlock]] = []
    for message in transcript.messages:
        for block in message.content:
            if not isinstance(block, ToolResultBlock) or block.is_error:
                continue
            name = outer_names.get(block.call_id)
            invocation = block.output.get("invocation")
            if isinstance(invocation, Mapping):
                invoked_name = invocation.get("tool_name")
                if isinstance(invoked_name, str):
                    name = invoked_name
            if name is not None:
                logical.append((name, block))
    return tuple(logical)


def _all_logical_names(transcript: Transcript) -> tuple[str, ...]:
    """Return resolved tool identities, including failed on-demand invocations."""

    outer_names = {
        call.id: call.name
        for message in transcript.messages
        for call in message.tool_calls
    }
    names: list[str] = []
    for message in transcript.messages:
        for block in message.content:
            if not isinstance(block, ToolResultBlock):
                continue
            name = outer_names.get(block.call_id)
            invocation = block.output.get("invocation")
            if isinstance(invocation, Mapping):
                invoked_name = invocation.get("tool_name")
                if isinstance(invoked_name, str):
                    name = invoked_name
            if name is not None:
                names.append(name)
    return tuple(names)


def _requests_for_transcript(
    requests: Sequence[ModelRequest],
    transcript: Transcript,
) -> tuple[ModelRequest, ...]:
    """Select physical requests belonging to one exact run transcript."""

    start = transcript.messages[0]
    selected = tuple(request for request in requests if start in request.messages)
    assert selected
    return selected


def _results_for(
    transcript: Transcript,
    tool_name: str,
) -> tuple[ToolResultBlock, ...]:
    return tuple(
        result
        for name, result in _successful_logical_results(transcript)
        if name == tool_name
    )


def _job_id_from_start(transcript: Transcript) -> str:
    receipts = _results_for(transcript, "start_data_profile")
    assert len(receipts) == 1
    data = receipts[0].output.get("data")
    assert isinstance(data, Mapping)
    assert data.get("status") == JobStatus.QUEUED.value
    job_id = data.get("job_id")
    assert isinstance(job_id, str) and job_id
    return job_id


def _assert_completed(
    result: LoopExit,
    transcript: Transcript,
    requests: Sequence[ModelRequest],
) -> None:
    assert result.kind is LoopExitKind.COMPLETED, (
        result.reason,
        result.usage.cost_estimate.code,
        len(requests),
    )
    assert result.reason == "completed", (
        result.reason,
        result.steps,
        result.usage.total_tokens,
        result.usage.cost_estimate.amount_usd,
    )
    assert result.final_text is not None
    assert 1 <= len(requests) <= 12
    assert result.usage.total_tokens > 0
    validate_completed_transcript(transcript, result)


def _assert_on_demand_invocation(
    transcript: Transcript,
    requests: Sequence[ModelRequest],
    tool_name: str,
) -> None:
    assert requests
    visible_names = {tool.name for tool in requests[0].tools}
    assert {"toolbox_search", "toolbox_load"} <= visible_names
    assert tool_name not in visible_names
    logical_names = _all_logical_names(transcript)
    assert "toolbox_search" in logical_names
    assert "toolbox_load" in logical_names
    assert tool_name in logical_names
    assert _results_for(transcript, tool_name)


async def _terminal(agent: Agent, job_id: str, *, timeout: float = 30.0):
    deadline = asyncio.get_running_loop().time() + timeout
    latest = None
    while asyncio.get_running_loop().time() < deadline:
        latest = await agent.inspect_job(job_id)
        assert latest is not None
        if latest.summary.status in _TERMINAL_STATUSES:
            return latest
        await asyncio.sleep(0.02)
    pytest.fail(f"job did not reach a terminal state: {latest!r}")


async def _running(agent: Agent, job_id: str, *, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    latest = None
    while asyncio.get_running_loop().time() < deadline:
        latest = await agent.inspect_job(job_id)
        assert latest is not None
        if latest.summary.status is JobStatus.RUNNING:
            return latest
        await asyncio.sleep(0.01)
    pytest.fail(f"job did not enter the running state: {latest!r}")


async def test_live_model_chooses_direct_query_instead_of_durable_job(
    tmp_path: Path,
) -> None:
    """Require immediate work to stay in the ordinary synchronous read path."""

    profile, provider = _live_provider()
    agent, source_id = await _new_agent(
        tmp_path,
        "live-stage-b-immediate",
        provider,
        profile,
    )
    try:
        request_start = len(provider.requests)
        result = await agent.run(
            "This is an authorized Stage B live acceptance check. Answer this as "
            "one immediate source read; it does not need durable or background "
            "work. Read the row named immediate from stage_b_immediate_probe. "
            "Do not start a job. End with exactly this format using only tool-returned "
            "values: STAGE_B_IMMEDIATE_CHECK token=<verification_token> "
            "amount=<amount>.",
            source_id=source_id,
        )
        transcript = await agent.transcript(result.run_id)
        requests = _requests_for_transcript(
            provider.requests[request_start:],
            transcript,
        )

        _assert_completed(result, transcript, requests)
        assert result.final_text is not None
        assert _IMMEDIATE_EXPECTATION in result.final_text
        logical_names = _all_logical_names(transcript)
        assert "data_query" in logical_names
        assert "start_data_profile" not in logical_names
        assert await agent.list_jobs() == ()
    finally:
        await agent.close()


async def test_live_model_starts_detaches_and_later_reads_profile_result(
    tmp_path: Path,
) -> None:
    """Exercise on-demand admission and later agent-scoped result recovery."""

    profile, provider = _live_provider()
    agent, source_id = await _new_agent(
        tmp_path,
        "live-stage-b-result",
        provider,
        profile,
    )
    try:
        start_request = len(provider.requests)
        started = await agent.run(
            "This is an authorized Stage B live acceptance check. Start exactly "
            "one durable background data profile for stage_b_profile_probe using a "
            "sample bound of exactly 5 rows. Use toolbox search and load. "
            "Do not use a synchronous SQL query, and do not list, inspect, poll, cancel, "
            "or read the job in this interaction. Once the durable start receipt is "
            "returned, end with: STAGE_B_JOB_STARTED <actual job id>.",
            source_id=source_id,
        )
        start_transcript = await agent.transcript(started.run_id)
        start_requests = _requests_for_transcript(
            provider.requests[start_request:],
            start_transcript,
        )

        _assert_completed(started, start_transcript, start_requests)
        _assert_on_demand_invocation(
            start_transcript,
            start_requests,
            "start_data_profile",
        )
        assert len(_all_logical_names(start_transcript)) >= 3
        assert not (_LIFECYCLE_TOOLS & set(_all_logical_names(start_transcript)))
        assert "data_query" not in _all_logical_names(start_transcript)
        job_id = _job_id_from_start(start_transcript)
        assert started.final_text is not None
        assert "STAGE_B_JOB_STARTED" in started.final_text
        assert job_id in started.final_text
        jobs = await agent.list_jobs()
        assert len(jobs) == 1 and jobs[0].job_id == job_id

        terminal = await _terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.SUCCEEDED
        assert terminal.summary.origin_conversation_id == started.conversation_id

        result_request = len(provider.requests)
        recovered = await agent.run(
            "This is a new conversation for the Stage B acceptance check. List the "
            "jobs owned by this agent to identify its completed data-profile job, "
            "read that job's validated result directly without inspecting it first, "
            "then read the exact referenced JSON artifact. Do not query the source "
            "again and do not load lifecycle reads. From that "
            "artifact, end with exactly: "
            "STAGE_B_PROFILE_CHECK sampled_rows=<resource sampled_rows> "
            "nullable_code_nulls=<nullable_code null_values>.",
        )
        recovered_transcript = await agent.transcript(recovered.run_id)
        recovered_requests = _requests_for_transcript(
            provider.requests[result_request:],
            recovered_transcript,
        )

        _assert_completed(recovered, recovered_transcript, recovered_requests)
        assert recovered.conversation_id != started.conversation_id
        assert recovered.steps <= 4
        recovered_names = _all_logical_names(recovered_transcript)
        assert "job_list" in recovered_names
        assert "job_read_results" in recovered_names
        assert "artifact_read" in recovered_names
        assert "job_inspect" not in recovered_names
        assert "toolbox_search" not in recovered_names
        assert "toolbox_load" not in recovered_names
        assert "data_query" not in recovered_names
        assert (
            recovered_names.index("job_list")
            < recovered_names.index("job_read_results")
            < recovered_names.index("artifact_read")
        )
        direct_names = {tool.name for tool in recovered_requests[0].tools}
        assert {"job_list", "job_read_results", "artifact_read"} <= direct_names
        assert recovered.final_text is not None
        assert _PROFILE_EXPECTATION in recovered.final_text

        result_reads = _results_for(recovered_transcript, "job_read_results")
        assert result_reads
        assert job_id in canonical_json(result_reads[-1].output)
        persisted_result = await agent.read_job_result(job_id)
        assert persisted_result is not None
        assert persisted_result.summary["sampled_rows"] == 5
        assert len(persisted_result.artifact_refs) == 1
        artifact_id = persisted_result.artifact_refs[0].artifact_id
        artifact_reads = _results_for(recovered_transcript, "artifact_read")
        assert artifact_reads and artifact_id in canonical_json(
            artifact_reads[-1].output
        )
        payload = await agent.read_artifact(artifact_id)
        document = json.loads(payload.content)
        resource = document["resources"][0]
        assert resource["sampled_rows"] == 5
        nullable = next(
            item
            for item in resource["column_profiles"]
            if item["column"] == "nullable_code"
        )
        assert nullable["null_values"] == 2
    finally:
        await agent.close()


async def test_live_model_requests_cancellation_of_running_profile_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hold the ordinary Daita executor so a real model can cancel its job."""

    profile, provider = _live_provider()
    agent, source_id = await _new_agent(
        tmp_path,
        "live-stage-b-cancel",
        provider,
        profile,
    )
    _, executor = agent._embedded._capabilities.resolve_execution(
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
        start_request = len(provider.requests)
        started = await agent.run(
            "This is an authorized Stage B cancellation acceptance check. Start "
            "exactly one durable background profile for stage_b_profile_probe with "
            "a sample bound of 5 rows. Use toolbox search and load. Once the start "
            "receipt is returned, end with: STAGE_B_JOB_STARTED <actual job id>.",
            source_id=source_id,
        )
        start_transcript = await agent.transcript(started.run_id)
        start_requests = _requests_for_transcript(
            provider.requests[start_request:],
            start_transcript,
        )
        _assert_completed(started, start_transcript, start_requests)
        _assert_on_demand_invocation(
            start_transcript,
            start_requests,
            "start_data_profile",
        )
        job_id = _job_id_from_start(start_transcript)
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        running = await _running(agent, job_id)
        assert running.summary.status is JobStatus.RUNNING

        cancel_request = len(provider.requests)
        cancelled = await agent.run(
            f"This is a new conversation. The agent-owned durable profile job "
            f"{job_id} is still running. Request cancellation of that exact job now "
            "without listing or inspecting jobs first. Do not claim cancellation is "
            "terminal unless a tool proves that. End with: "
            "STAGE_B_CANCEL_REQUESTED <actual job id>.",
        )
        cancel_transcript = await agent.transcript(cancelled.run_id)
        cancel_requests = _requests_for_transcript(
            provider.requests[cancel_request:],
            cancel_transcript,
        )

        _assert_completed(cancelled, cancel_transcript, cancel_requests)
        assert cancelled.conversation_id != started.conversation_id
        _assert_on_demand_invocation(
            cancel_transcript,
            cancel_requests,
            "job_cancel",
        )
        assert cancelled.final_text is not None
        assert "STAGE_B_CANCEL_REQUESTED" in cancelled.final_text
        assert job_id in cancelled.final_text
        cancel_names = _all_logical_names(cancel_transcript)
        assert "job_list" not in cancel_names
        assert "job_inspect" not in cancel_names
        assert "job_read_results" not in cancel_names
        cancel_results = _results_for(cancel_transcript, "job_cancel")
        assert len(cancel_results) == 1
        assert job_id in canonical_json(cancel_results[0].output)

        terminal = await _terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.CANCELLED
        assert await agent.read_job_result(job_id) is None
    finally:
        release.set()
        await agent.close()
