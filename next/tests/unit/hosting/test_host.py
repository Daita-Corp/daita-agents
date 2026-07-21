from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from daita import Agent, AgentConfig, ConfigError
from daita.hosting import AgentHost, AgentHostState
from daita.hosting.embedded import HostActiveError
from daita.hosting.inbox import (
    HostInboxEnqueueConflictError,
    HostInboxKind,
    HostInboxStatus,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopBudgets, LoopExitKind, Readiness, Turn
from daita.monitors import (
    IntervalSchedule,
    MonitorDefinition,
    MonitorRunStatus,
    MonitorScope,
)
from daita.operations.governance import (
    ApprovalRequest,
    ApprovalStatus,
    DefaultPolicyProfile,
)
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TriggerKind,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
HASH = "sha256:" + ("0" * 64)


@dataclass
class MutableClock:
    current: datetime = NOW

    def __call__(self) -> datetime:
        return self.current


class TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class TextDomain:
    def __init__(self, clock: MutableClock) -> None:
        self._clock = clock

    async def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("text-only host test cannot validate an action")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only host test cannot project evidence")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        return Readiness(
            allowed=True,
            code="ready.text",
            message="Text response is ready.",
            evaluated_at=self._clock(),
        )


class BlockingProvider:
    provider_id = "mock:host-blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.operation_id: str | None = None

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.operation_id = request.operation_id
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.finished.set()
        raise AssertionError("blocking provider must be cancelled by the host")


def _response(text: str = "done") -> ModelResponse:
    return ModelResponse(text=text, finish_reason=FinishReason.STOP)


async def _create_configured_host(
    tmp_path,
    provider,
    clock: MutableClock,
    *,
    name: str = "atlas",
) -> AgentHost:
    return await AgentHost.create(
        name,
        root=tmp_path,
        model=provider,
        context_builder=TextContext(),
        domain=TextDomain(clock),
        cadence_seconds=3_600,
        clock=clock,
    )


async def _open_configured_host(
    tmp_path,
    provider,
    clock: MutableClock,
    *,
    name: str = "atlas",
) -> AgentHost:
    return await AgentHost.open(
        name,
        root=tmp_path,
        model=provider,
        context_builder=TextContext(),
        domain=TextDomain(clock),
        cadence_seconds=3_600,
        clock=clock,
    )


def _background_tasks() -> set[asyncio.Task[object]]:
    current = asyncio.current_task()
    return {
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    }


def _pending_trigger(host: AgentHost, key: str, message: str):
    return host._pending_item(
        HostInboxKind.TRIGGER,
        idempotency_key=key,
        payload={
            "message": message,
            "session_id": None,
            "source_id": f"host:{key}",
        },
        trigger_id=f"trigger-{key}",
    )


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=timeout)


async def test_create_open_are_inert_and_start_owns_one_cadence_and_writer(
    tmp_path,
) -> None:
    before = _background_tasks()
    host = await AgentHost.create("atlas", root=tmp_path, cadence_seconds=3_600)
    await asyncio.sleep(0)
    assert _background_tasks() == before

    with pytest.raises(HostActiveError, match="host_active"):
        await AgentHost.open("atlas", root=tmp_path)

    await host.start()
    for _ in range(3):
        await asyncio.sleep(0)
    assert host.state is AgentHostState.RUNNING
    assert host._cadence_task is not None
    assert _background_tasks() - before == {host._cadence_task}

    cadence = host._cadence_task
    await host.start()
    assert host._cadence_task is cadence
    await host.stop()
    assert cadence.done()
    assert host.state is AgentHostState.STOPPED

    with pytest.raises(ValueError, match="cadence_seconds"):
        await AgentHost.open("atlas", root=tmp_path, cadence_seconds=0)
    reopened = await AgentHost.open("atlas", root=tmp_path)
    await asyncio.sleep(0)
    try:
        assert _background_tasks() == before
    finally:
        await reopened.stop()


async def test_host_and_agent_share_the_persisted_runtime_default_binding(
    tmp_path,
) -> None:
    config = AgentConfig(
        budgets=LoopBudgets(max_turns=4, max_actions=7),
        policy_profile=DefaultPolicyProfile(version="host-2"),
    )
    host = await AgentHost.create("shared-config", root=tmp_path, config=config)
    try:
        assert host._embedded.runtime_defaults == config.runtime_defaults
    finally:
        await host.stop()

    agent = await Agent.open("shared-config", root=tmp_path)
    try:
        assert agent._embedded.runtime_defaults == config.runtime_defaults
    finally:
        await agent.close()

    with pytest.raises(ConfigError) as captured:
        await AgentHost.open(
            "shared-config",
            root=tmp_path,
            config=AgentConfig(
                budgets=LoopBudgets(max_turns=5, max_actions=7),
                policy_profile=config.policy_profile,
            ),
        )
    assert captured.value.error_code == "config_conflict"
    assert captured.value.section == "budgets"


async def test_submit_is_durable_idempotent_and_replays_across_restart(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = MockModelProvider((_response("first"),), provider_id="mock:host")
    host = await _create_configured_host(tmp_path, provider, clock)
    await host.start()
    try:
        with pytest.raises(ValueError, match="session_id"):
            await host.submit(
                "Invalid session.",
                idempotency_key="invalid-session",
                session_id="",
            )
        first = await host.submit("Run once.", idempotency_key="request-1")
        replay = await host.submit("Run once.", idempotency_key="request-1")

        assert first.status is HostInboxStatus.COMPLETED
        assert replay == first
        assert first.operation_id is not None
        assert len(provider.requests) == 1
        with pytest.raises(HostInboxEnqueueConflictError):
            await host.submit("Different input.", idempotency_key="request-1")
    finally:
        await host.stop()

    replay_provider = MockModelProvider((), provider_id="mock:host")
    reopened = await _open_configured_host(tmp_path, replay_provider, clock)
    await reopened.start()
    try:
        durable_replay = await reopened.submit(
            "Run once.",
            idempotency_key="request-1",
        )
        assert durable_replay == first
        assert replay_provider.requests == ()
    finally:
        await reopened.stop()


async def test_start_recovers_pending_inbox_and_drain_stop_finishes_fifo(
    tmp_path,
) -> None:
    clock = MutableClock()
    initial = MockModelProvider((), provider_id="mock:host-recovery")
    host = await _create_configured_host(tmp_path, initial, clock)
    pending = _pending_trigger(host, "startup-1", "Recover after restart.")
    await host._store.enqueue_host_inbox(pending)
    await host.stop(drain=False)

    provider = MockModelProvider(
        (_response("recovered"), _response("drained")),
        provider_id="mock:host-recovery",
    )
    reopened = await _open_configured_host(tmp_path, provider, clock)
    await reopened.start()
    assert (
        await reopened._store.list_pending_host_inbox(
            reopened._embedded.identity.id,
            limit=100,
        )
        == ()
    )
    assert len(provider.requests) == 1

    queued_for_drain = _pending_trigger(reopened, "drain-1", "Drain on stop.")
    await reopened._store.enqueue_host_inbox(queued_for_drain)
    await reopened.stop(drain=True)
    assert len(provider.requests) == 2

    audit = await _open_configured_host(
        tmp_path,
        MockModelProvider((), provider_id="mock:host-recovery"),
        clock,
    )
    try:
        assert (
            await audit._store.list_pending_host_inbox(
                audit._embedded.identity.id,
                limit=100,
            )
            == ()
        )
    finally:
        await audit.stop()


async def test_monitor_run_once_uses_the_ordinary_exact_trigger_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MutableClock()
    provider = MockModelProvider(
        (_response("checked"), _response("checked now")),
        provider_id="mock:monitor",
    )
    host = await _create_configured_host(tmp_path, provider, clock)
    await host.start()
    definition = MonitorDefinition(
        name="Backlog check",
        objective="Inspect the backlog.",
        scope=MonitorScope(),
        schedule=IntervalSchedule(interval_seconds=60, anchor_at=NOW),
    )
    proposal = await host.propose_monitor(
        "monitor-backlog",
        definition,
        idempotency_key="monitor-create-1",
    )
    await host.confirm_monitor(
        proposal.id,
        candidate_hash=proposal.candidate_hash,
        actor_id="reviewer-1",
        reason="Enable the monitor.",
    )

    observed: list[AgentTrigger] = []
    ordinary_run_trigger = host._embedded.run_trigger

    async def observe(trigger: AgentTrigger):
        observed.append(trigger)
        return await ordinary_run_trigger(trigger)

    monkeypatch.setattr(host._embedded, "run_trigger", observe)
    try:
        clock.current = NOW + timedelta(seconds=60)
        results = await host.run_once(clock.current)
        run_now = await host.run_monitor_now(
            "monitor-backlog",
            idempotency_key="monitor-now-1",
        )

        assert len(results) == 1
        assert results[0].claimed
        assert results[0].run_status is MonitorRunStatus.SUCCEEDED
        assert run_now.run_status is MonitorRunStatus.SUCCEEDED
        assert len(observed) == 2
        assert observed[0].kind is TriggerKind.MONITOR
        assert observed[0].payload["message"] == definition.objective
        durable = await host._store.load_by_trigger(observed[0].id)
        assert durable is not None
        assert durable.snapshot.trigger == observed[0]
        assert observed[1].kind is TriggerKind.MONITOR
        assert len(provider.requests) == 2
    finally:
        await host.stop()


async def test_control_lane_cancels_a_blocked_submission_without_execution_lock(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = BlockingProvider()
    host = await _create_configured_host(tmp_path, provider, clock)
    await host.start()
    submitted = asyncio.create_task(
        host.submit("Block until cancelled.", idempotency_key="cancel-1")
    )
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        assert provider.operation_id is not None
        assert host._execution_lock.locked()

        interrupted = await asyncio.wait_for(
            host.cancel(provider.operation_id, reason="operator_cancelled"),
            timeout=0.5,
        )
        completed = await submitted
        snapshot = await host._embedded.inspect(provider.operation_id)

        assert interrupted.kind is LoopExitKind.INTERRUPTED
        assert completed.status is HostInboxStatus.COMPLETED
        assert completed.operation_id == provider.operation_id
        assert snapshot.operation.status is OperationStatus.INTERRUPTED
        assert provider.finished.is_set()
    finally:
        if not submitted.done():
            submitted.cancel()
            await asyncio.gather(submitted, return_exceptions=True)
        await host.stop(drain=True)


async def test_stop_without_drain_awaits_active_interruption_and_releases_lock(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = BlockingProvider()
    host = await _create_configured_host(tmp_path, provider, clock)
    await host.start()
    submitted = asyncio.create_task(
        host.submit("Stop without draining.", idempotency_key="stop-1")
    )
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    assert provider.operation_id is not None

    await asyncio.wait_for(host.stop(drain=False), timeout=1)

    assert host.state is AgentHostState.STOPPED
    assert submitted.done()
    assert submitted.cancelled()
    assert provider.finished.is_set()

    reopened = await _open_configured_host(
        tmp_path,
        MockModelProvider((), provider_id=provider.provider_id),
        clock,
    )
    try:
        snapshot = await reopened._embedded.inspect(provider.operation_id)
        status = await reopened.status()
        assert snapshot.operation.status is OperationStatus.INTERRUPTED
        assert status.pending_inbox == 1
    finally:
        await reopened.stop(drain=False)


async def test_monitor_cancellation_does_not_cancel_the_host_cadence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MutableClock()
    provider = BlockingProvider()
    host = await _create_configured_host(tmp_path, provider, clock)
    await host.start()
    definition = MonitorDefinition(
        name="Blocking monitor",
        objective="Wait until the operator cancels this monitor.",
        scope=MonitorScope(),
        schedule=IntervalSchedule(interval_seconds=60, anchor_at=NOW),
    )
    proposal = await host.propose_monitor(
        "monitor-blocking",
        definition,
        idempotency_key="monitor-blocking-create",
    )
    await host.confirm_monitor(
        proposal.id,
        candidate_hash=proposal.candidate_hash,
        actor_id="reviewer-1",
        reason="Exercise cancellation ownership.",
    )

    passes = 0
    run_due = host._scheduler.run_due

    async def count_pass(now, *, limit=100):
        nonlocal passes
        passes += 1
        return await run_due(now, limit=limit)

    monkeypatch.setattr(host._scheduler, "run_due", count_pass)
    clock.current = NOW + timedelta(seconds=60)
    host._wake_hint.set()
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        assert provider.operation_id is not None
        await host.cancel(provider.operation_id, reason="monitor_cancelled")
        await _wait_until(lambda: host._active_task is None and passes >= 1)

        cadence = host._cadence_task
        assert cadence is not None
        assert not cadence.done()
        host._wake_hint.set()
        await _wait_until(lambda: passes >= 2)
        assert not cadence.done()
    finally:
        await host.stop(drain=True)


async def test_cancelled_approval_still_commits_its_durable_wake(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = await AgentHost.create("atlas", root=tmp_path, cadence_seconds=3_600)
    await host.start()
    started = asyncio.Event()
    release = asyncio.Event()
    decision_finished = False
    decision = ApprovalRequest(
        id="approval-1",
        operation_id="operation-1",
        task_id="task-1",
        task_fingerprint=HASH,
        policy_fingerprint=HASH,
        requested_at=NOW,
        status=ApprovalStatus.APPROVED,
        decided_at=NOW,
        decided_by="reviewer-1",
        decision_reason="Approved.",
    )

    async def delayed_decision(*args, **kwargs) -> ApprovalRequest:
        nonlocal decision_finished
        started.set()
        await release.wait()
        decision_finished = True
        return decision

    monkeypatch.setattr(host._embedded, "decide_approval", delayed_decision)
    approving = asyncio.create_task(
        host.approve(
            "approval-1",
            decided_by="reviewer-1",
            reason="Approved.",
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        approving.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await approving

        pending = await host._store.list_pending_host_inbox(
            host._embedded.identity.id,
            limit=100,
        )
        assert decision_finished
        assert len(pending) == 1
        assert pending[0].kind is HostInboxKind.APPROVAL_WAKE
        assert pending[0].payload["operation_id"] == decision.operation_id
    finally:
        await host.stop(drain=False)


async def test_shutdown_failure_still_releases_writer_and_marks_stopped(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MutableClock()
    host = await _create_configured_host(
        tmp_path,
        MockModelProvider((), provider_id="mock:stop-failure"),
        clock,
    )
    await host.start()

    async def fail_drain() -> None:
        raise RuntimeError("injected drain failure")

    monkeypatch.setattr(host, "_drain_inbox_locked", fail_drain)
    with pytest.raises(RuntimeError, match="injected drain failure"):
        await host.stop(drain=True)
    assert host.state is AgentHostState.STOPPED
    await host.stop()

    reopened = await _open_configured_host(
        tmp_path,
        MockModelProvider((), provider_id="mock:stop-failure"),
        clock,
    )
    await reopened.stop()


async def test_drain_reads_until_every_bounded_batch_is_empty(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MutableClock()
    host = await _create_configured_host(
        tmp_path,
        MockModelProvider((), provider_id="mock:drain-batches"),
        clock,
    )
    first = tuple(object() for _ in range(1_000))
    second = (object(),)
    batches = [first, second, ()]
    processed: list[object] = []

    async def list_pending(agent_id, *, limit):
        assert agent_id == host._embedded.identity.id
        assert limit == 1_000
        return batches.pop(0)

    async def process(item):
        processed.append(item)
        return item

    monkeypatch.setattr(host._store, "list_pending_host_inbox", list_pending)
    monkeypatch.setattr(host, "_process_item", process)
    try:
        await host._drain_inbox_locked()
        assert len(processed) == 1_001
        assert batches == []
    finally:
        await host.stop(drain=False)
