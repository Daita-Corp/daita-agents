"""Authorized live-model acceptance coverage for Stage C follow-ups.

The suite covers a real Stage B start through Stage C delivery, terminal-run
delivery recovery, a failed terminal job with no result, and a transient route
failure followed by one sticky live fallback.

The complete module authorizes at most five live ``AgentLoop`` runs: one
foreground run and four autonomous follow-ups. Each run has its own explicit
step, token, wall-time, and estimated-cost ceiling. Deterministic tests remain
authoritative for exact crash timing, duplicate observations, fencing, budget
races, scope revocation, and malformed persisted state.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import (
    Agent,
    DeliveryState,
    DeliverySubjectKind,
    JobStatus,
    LoopLimits,
    OutcomeConclusionKind,
    OutcomeState,
    SQLiteSource,
    create_llm_provider,
)
from daita.autonomy import FollowupDisposition
from daita.capabilities import AccessMode, OperationalEffect
from daita.domains.data.profile_jobs import (
    DATA_PROFILE_EXECUTION_CAPABILITY_ID,
    START_DATA_PROFILE_CAPABILITY_ID,
)
from daita.jobs.capabilities import (
    JOB_INSPECT_CAPABILITY_ID,
    JOB_READ_RESULTS_CAPABILITY_ID,
)
from daita.llm._lifecycle import closing_stream
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import (
    ManagedModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from daita.loop.models import (
    LoopExitKind,
    RunOrigin,
    Transcript,
    validate_completed_transcript,
)
from daita.storage.sqlite import SQLiteStateStore

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_C_FOLLOWUPS"
_MODEL_ID = "DAITA_STAGE_C_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_STAGE_C_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_STAGE_C_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

_TABLE = "stage_c_profile_probe"
_PROFILE_ROWS = 5
_TERMINAL_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.NEEDS_ATTENTION,
    }
)
_SEED_LIMITS = LoopLimits()

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing at most "
            "five live AgentLoop runs for the Stage C follow-up suite; each run "
            f"is capped by {_MAX_COST}"
        ),
    ),
]


class _RecordingProvider:
    """Capture canonical requests and responses around one real provider."""

    def __init__(self, delegate: ManagedModelProvider) -> None:
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
        async with closing_stream(self._delegate.stream(request)) as events:
            async for event in events:
                if isinstance(event, ModelStreamCompleted):
                    self.responses.append(event.response)
                yield event

    async def close(self) -> None:
        await self._delegate.close()


class _UnavailableStreamingProvider:
    """Inject one deterministic transient failure before a live fallback."""

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile
        self.requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return self.profile.id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        raise ModelProviderError(
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "injected Stage C failure before the live fallback",
            provider_id=self.provider_id,
            usage=ModelUsage(
                cost_estimate=CostEstimate.complete(Decimal("0")),
            ),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        raise ModelProviderError(
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "injected Stage C failure before the live fallback",
            provider_id=self.provider_id,
            usage=ModelUsage(
                cost_estimate=CostEstimate.complete(Decimal("0")),
            ),
        )
        yield  # pragma: no cover - makes this an async iterator


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


def _limits() -> LoopLimits:
    return LoopLimits(
        max_steps=12,
        max_total_tokens=30_000,
        max_wall_time_seconds=180,
        max_estimated_cost_usd=_cost_limit(),
    )


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


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"CREATE TABLE {_TABLE}("
            "id INTEGER PRIMARY KEY, nullable_code TEXT, measure REAL NOT NULL)"
        )
        connection.executemany(
            f"INSERT INTO {_TABLE} VALUES (?, ?, ?)",
            (
                (1, "alpha", 10.5),
                (2, None, 11.5),
                (3, "beta", 12.5),
                (4, None, 13.5),
                (5, "alpha", 14.5),
            ),
        )


async def _create_home(tmp_path: Path, name: str) -> tuple[Path, str, str]:
    root = tmp_path / f"{name}-root"
    database = tmp_path / f"{name}.sqlite"
    _database(database)
    agent = await Agent.create(name, root=root, workspace=workspace_for(root))
    try:
        source = await agent.attach(SQLiteSource(database, name="Stage C live probe"))
        resources = await agent.list_catalog_resources(source_id=source.id)
        resource = next(item for item in resources if item.name == _TABLE)
        return root, source.id, resource.id
    finally:
        await agent.close()


def _logical_results(
    transcript: Transcript,
    tool_name: str,
    *,
    include_errors: bool = False,
) -> tuple[ToolResultBlock, ...]:
    outer_names = {
        call.id: call.name
        for message in transcript.messages
        for call in message.tool_calls
    }
    selected: list[ToolResultBlock] = []
    for message in transcript.messages:
        for block in message.content:
            if not isinstance(block, ToolResultBlock) or (
                block.is_error and not include_errors
            ):
                continue
            name = outer_names.get(block.call_id)
            invocation = block.output.get("invocation")
            if isinstance(invocation, Mapping):
                invoked_name = invocation.get("tool_name")
                if isinstance(invoked_name, str):
                    name = invoked_name
            if name == tool_name:
                selected.append(block)
    return tuple(selected)


def _job_id_from_start(transcript: Transcript) -> str:
    receipts = _logical_results(transcript, "start_data_profile")
    assert len(receipts) == 1
    assert receipts[0].capability_id == START_DATA_PROFILE_CAPABILITY_ID
    data = receipts[0].output.get("data")
    assert isinstance(data, Mapping)
    job_id = data.get("job_id")
    assert isinstance(job_id, str) and job_id
    return job_id


async def _wait_for_terminal(
    agent: Agent,
    job_id: str,
    *,
    timeout: float = 30.0,
):
    deadline = asyncio.get_running_loop().time() + timeout
    latest = None
    while asyncio.get_running_loop().time() < deadline:
        latest = await agent.inspect_job(job_id)
        assert latest is not None
        if latest.summary.status in _TERMINAL_STATUSES:
            return latest
        await asyncio.sleep(0.02)
    pytest.fail(f"job did not reach a terminal state: {latest!r}")


async def _wait_for_inbox(agent: Agent, *, timeout: float = 180.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        items = await agent.inbox()
        if items:
            return items
        driver = agent._embedded._followup_driver
        if driver is not None and driver.done():
            pytest.fail(f"Stage C follow-up driver stopped: {driver.exception()!r}")
        await asyncio.sleep(0.05)
    followups = await agent._embedded._store.list_autonomous_followups(agent.id)
    pytest.fail(f"live Stage C follow-up did not reach the inbox: {followups!r}")


async def _assert_live_delivery(agent: Agent, job_id: str):
    items = await _wait_for_inbox(agent)
    assert len(items) == 1
    item = items[0]
    assert item.state is DeliveryState.AVAILABLE
    assert item.subject_kind is DeliverySubjectKind.AUTONOMOUS_FOLLOWUP
    assert item.conclusion_kind is OutcomeConclusionKind.TERMINAL_RUN
    assert item.conclusion_state is OutcomeState.SUCCEEDED
    assert item.provenance_digest.startswith("sha256:")
    report = item.conclusion_preview
    assert isinstance(report, str) and report.strip()
    assert str(_PROFILE_ROWS) in report or "five" in report.lower()

    assert item.resulting_run_id is not None
    transcript = await agent.transcript(item.resulting_run_id)
    result = await agent._embedded._store.result(item.resulting_run_id)
    assert result is not None and result.kind is LoopExitKind.COMPLETED
    validate_completed_transcript(transcript, result)
    assert transcript.run.origin is RunOrigin.JOB_EVENT
    assert transcript.run.execution_scope is not None
    scope = transcript.run.execution_scope
    assert scope.job_id == job_id
    assert scope.allowed_access_modes == frozenset({AccessMode.NONE, AccessMode.READ})
    assert scope.allowed_operational_effects == frozenset({OperationalEffect.NONE})
    assert all(message.role is not MessageRole.USER for message in transcript.messages)

    successful_lineage = {
        block.capability_id
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and not block.is_error
    }
    assert {
        JOB_INSPECT_CAPABILITY_ID,
        JOB_READ_RESULTS_CAPABILITY_ID,
    } <= successful_lineage
    assert successful_lineage <= set(scope.allowed_capability_ids)
    assert all(
        block.executor_id is not None
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and not block.is_error
    )
    return item, transcript, result


async def _seed_terminal_job(
    tmp_path: Path,
    name: str,
) -> tuple[Path, str, str]:
    root, source_id, resource_id = await _create_home(tmp_path, name)
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="seed-profile-load",
                        name="toolbox_load",
                        arguments={"tool_names": ["start_data_profile"]},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="seed-profile",
                        name="start_data_profile",
                        arguments={
                            "resource_ids": [resource_id],
                            "sample_rows": _PROFILE_ROWS,
                        },
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Seed accepted."),
        )
    )
    agent = await Agent.open(
        name,
        root=root,
        model=provider,
        model_profile=provider.model_profile,
        limits=_SEED_LIMITS,
        workspace=workspace_for(root),
    )
    try:
        started = await agent.run(
            "Seed one deterministic profile.", source_id=source_id
        )
        transcript = await agent.transcript(started.run_id)
        job_id = _job_id_from_start(transcript)
        terminal = await _wait_for_terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.SUCCEEDED
        result = await agent.read_job_result(job_id)
        assert result is not None
        assert result.summary["sampled_rows"] == _PROFILE_ROWS
    finally:
        try:
            await agent.close()
        finally:
            await provider.close()
    return root, source_id, job_id


async def _seed_failed_terminal_job(
    tmp_path: Path,
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    root, source_id, resource_id = await _create_home(tmp_path, name)
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="seed-failed-profile-load",
                        name="toolbox_load",
                        arguments={"tool_names": ["start_data_profile"]},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="seed-failed-profile",
                        name="start_data_profile",
                        arguments={
                            "resource_ids": [resource_id],
                            "sample_rows": _PROFILE_ROWS,
                        },
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Seed accepted."),
        )
    )
    agent = await Agent.open(
        name,
        root=root,
        model=provider,
        model_profile=provider.model_profile,
        limits=_SEED_LIMITS,
        workspace=workspace_for(root),
    )
    _, executor = agent._embedded._capabilities.resolve_execution(
        DATA_PROFILE_EXECUTION_CAPABILITY_ID
    )

    async def fail_execution(request) -> None:
        del request
        raise RuntimeError("injected_stage_c_live_job_failure")

    monkeypatch.setattr(executor, "execute", fail_execution)
    try:
        started = await agent.run(
            "Seed one deterministically failing profile.", source_id=source_id
        )
        transcript = await agent.transcript(started.run_id)
        job_id = _job_id_from_start(transcript)
        terminal = await _wait_for_terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.FAILED
        assert terminal.failure_code == "job_execution_failed"
        assert await agent.read_job_result(job_id) is None
    finally:
        await agent.close()
    return root, job_id


async def test_live_model_runs_stage_b_start_through_stage_c_inbox(
    tmp_path: Path,
) -> None:
    """Exercise one real foreground start and its real autonomous conclusion."""

    name = "live-stage-c-end-to-end"
    root, source_id, _resource_id = await _create_home(tmp_path, name)
    profile, provider = _live_provider()
    agent = await Agent.open(
        name,
        root=root,
        model=provider,
        model_profile=profile,
        limits=_limits(),
        workspace=workspace_for(root),
    )
    try:
        started = await agent.run(
            f"Start exactly one durable background profile for {_TABLE} with a "
            f"sample bound of exactly {_PROFILE_ROWS} rows.",
            source_id=source_id,
        )
        started_transcript = await agent.transcript(started.run_id)
        assert started.kind is LoopExitKind.COMPLETED, started
        validate_completed_transcript(started_transcript, started)
        job_id = _job_id_from_start(started_transcript)
        assert all(
            not _logical_results(
                started_transcript,
                tool_name,
                include_errors=True,
            )
            for tool_name in (
                "job_list",
                "job_inspect",
                "job_read_results",
                "job_cancel",
            )
        )
        terminal = await _wait_for_terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.SUCCEEDED

        item, _transcript, followup_result = await _assert_live_delivery(agent, job_id)
        assert followup_result.final_text == item.conclusion_preview
        assert len(await agent.list_jobs()) == 1
        acknowledged = await agent.acknowledge_inbox(item.delivery_id)
        assert acknowledged is not None
        assert acknowledged.state is DeliveryState.ACKNOWLEDGED
        assert 2 <= len(provider.requests) <= 24
    finally:
        try:
            await agent.close()
        finally:
            await provider.close()


async def test_live_terminal_run_is_finalized_after_reopen_without_reasoning_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit live reasoning, lose delivery finalization, and recover model-free."""

    name = "live-stage-c-terminal-recovery"
    root, _source_id, job_id = await _seed_terminal_job(tmp_path, name)
    profile, provider = _live_provider()
    finalizer_entered = asyncio.Event()

    async def fail_delivery_finalization(
        store: SQLiteStateStore,
        agent_id: str,
        followup_id: str,
        *,
        delivery_id: str,
        finalized_at: datetime,
    ) -> None:
        del delivery_id, finalized_at
        followup = await store.load_autonomous_followup(agent_id, followup_id)
        assert followup is not None
        assert followup.disposition is (
            FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
        )
        assert followup.reserved_run_id is not None
        assert await store.result(followup.reserved_run_id) is not None
        finalizer_entered.set()
        raise RuntimeError("injected_live_delivery_finalization_failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            SQLiteStateStore,
            "finalize_autonomous_followup",
            fail_delivery_finalization,
        )
        agent = await Agent.open(
            name,
            root=root,
            model=provider,
            model_profile=profile,
            limits=_limits(),
            workspace=workspace_for(root),
        )
        try:
            await asyncio.wait_for(finalizer_entered.wait(), timeout=180)
            followup = (
                await agent._embedded._store.list_autonomous_followups(agent.id)
            )[0]
            assert followup.disposition is (
                FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
            )
            assert followup.reserved_run_id is not None
            committed_run_id = followup.reserved_run_id
            live_request_count = len(provider.requests)
            assert 2 <= live_request_count <= 12
            assert await agent.inbox() == ()
        finally:
            try:
                await agent.close()
            finally:
                await provider.close()

    recovery = MockModelProvider((), complete_pricing=True)
    agent = await Agent.open(
        name,
        root=root,
        model=recovery,
        model_profile=recovery.model_profile,
        limits=_limits(),
        workspace=workspace_for(root),
    )
    try:
        item, _transcript, _result = await _assert_live_delivery(agent, job_id)
        assert item.resulting_run_id == committed_run_id
        assert len(provider.requests) == live_request_count
        assert recovery.requests == ()
        assert len(await agent.inbox()) == 1
        retried = await agent._embedded._store.finalize_autonomous_followup(
            agent.id,
            item.subject_id,
            delivery_id="delivery-live-idempotent-retry",
            finalized_at=datetime.now(UTC),
        )
        assert retried is not None
        assert retried[1].delivery_id == item.delivery_id
        assert len(await agent.inbox()) == 1
        assert recovery.requests == ()
    finally:
        await agent.close()


async def test_live_failed_terminal_job_is_reported_without_inventing_a_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real model must inspect a failed job and tolerate its absent result."""

    name = "live-stage-c-failed-job"
    root, job_id = await _seed_failed_terminal_job(tmp_path, name, monkeypatch)
    profile, provider = _live_provider()
    agent = await Agent.open(
        name,
        root=root,
        model=provider,
        model_profile=profile,
        limits=_limits(),
        workspace=workspace_for(root),
    )
    try:
        items = await _wait_for_inbox(agent)
        assert len(items) == 1
        item = items[0]
        assert item.state is DeliveryState.AVAILABLE
        assert item.subject_kind is DeliverySubjectKind.AUTONOMOUS_FOLLOWUP
        assert item.conclusion_kind is OutcomeConclusionKind.TERMINAL_RUN
        assert item.conclusion_state is OutcomeState.SUCCEEDED
        report = item.conclusion_preview
        assert isinstance(report, str) and "fail" in report.casefold()

        assert item.resulting_run_id is not None
        transcript = await agent.transcript(item.resulting_run_id)
        assert transcript.run.execution_scope is not None
        assert transcript.run.execution_scope.job_id == job_id
        result = await agent._embedded._store.result(item.resulting_run_id)
        assert result is not None and result.kind is LoopExitKind.COMPLETED
        validate_completed_transcript(transcript, result)
        inspections = _logical_results(transcript, "job_inspect")
        reads = _logical_results(
            transcript,
            "job_read_results",
            include_errors=True,
        )
        assert len(inspections) == 1
        assert len(reads) == 1 and reads[0].is_error
        error = reads[0].output.get("error")
        assert isinstance(error, Mapping)
        assert error.get("code") == "job_result_not_ready"
        details = error.get("details")
        assert isinstance(details, Mapping)
        assert details.get("status") == JobStatus.FAILED.value
        assert all(
            message.role is not MessageRole.USER for message in transcript.messages
        )
        assert len(provider.requests) >= 2
    finally:
        try:
            await agent.close()
        finally:
            await provider.close()


async def test_live_followup_uses_sticky_fallback_and_delivers_exactly_once(
    tmp_path: Path,
) -> None:
    """One transient route failure must not duplicate reasoning or delivery."""

    name = "live-stage-c-route-fallback"
    root, _source_id, job_id = await _seed_terminal_job(tmp_path, name)
    profile, live = _live_provider()
    unavailable_profile = replace(profile, id="mock:stage-c-unavailable")
    unavailable = _UnavailableStreamingProvider(unavailable_profile)
    router = ModelRouter(
        (
            ModelProviderRegistration(
                provider=unavailable,
                profile=unavailable_profile,
                allowed_sensitivities=frozenset(ModelSensitivity),
            ),
            ModelProviderRegistration(
                provider=live,
                profile=profile,
                allowed_sensitivities=frozenset(ModelSensitivity),
            ),
        ),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )
    agent = await Agent.open(
        name,
        root=root,
        model=router,
        model_profile=router.model_profile,
        limits=_limits(),
        workspace=workspace_for(root),
    )
    try:
        item, _transcript, _result = await _assert_live_delivery(agent, job_id)
        assert len(unavailable.requests) == 1
        assert 2 <= len(live.requests) <= 12
        assert len(await agent.inbox()) == 1
        followups = await agent._embedded._store.list_autonomous_followups(agent.id)
        assert len(followups) == 1
        assert followups[0].reserved_run_id == item.resulting_run_id
        assert followups[0].attempt_count == 1
    finally:
        try:
            await agent.close()
        finally:
            await live.close()


__all__ = []
