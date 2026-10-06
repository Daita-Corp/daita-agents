"""Public agent composition with caller-admitted state, documents and bytes."""

import asyncio
from contextlib import contextmanager
from decimal import Decimal

import pytest

from daita import Agent, AgentConfig
from daita.hosting.embedded import (
    AgentHomeError,
    AgentIdentityMismatchError,
    EmbeddedAgent,
)
from daita.identity import AgentIdentity
from daita.llm.models import ModelSensitivity
from daita.llm.profiles import reviewed_model_profile
from daita.llm.routing import ModelRoute, ModelRouteCandidate, ModelRouter
from daita.storage.errors import StorageCommitUnknownError, StorageOwnershipLostError
from tests.support.approval_learning import (
    ApprovalDecision,
    MockModelProvider,
    _call,
    _memory_call,
    _profile,
    _stop,
)
from tests.support.graph import GRAPH_NOW

pytestmark = [pytest.mark.contract, pytest.mark.integration]


@pytest.mark.parametrize(
    "error_type", [StorageCommitUnknownError, StorageOwnershipLostError]
)
async def test_document_storage_failure_stops_the_run_before_another_model_call(
    state_store,
    advisory_storage,
    artifact_byte_storage,
    error_type,
):
    identity = AgentIdentity(
        id="failed-document", display_name="Failed", created_at=GRAPH_NOW
    )
    await state_store.initialize_identity(identity)
    attempts = []

    class FailingWrites:
        @contextmanager
        def transaction(self, *, write=False):
            with advisory_storage.transaction(write=write) as transaction:
                yield transaction
                if write:
                    attempts.append("write")
                    raise error_type("transport detail must not reach the model")

    async def approve(request) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        (_call(_memory_call(content="New memory.")), _stop("Must not continue."))
    )
    agent = await Agent.from_storage(
        agent_id=identity.id,
        state=state_store,
        advisory_storage=FailingWrites(),
        artifact_storage=artifact_byte_storage,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        with pytest.raises(error_type):
            await agent.run("Remember the new memory.")
        assert len(provider.logical_requests) == 1
        assert attempts == ["write"]
    finally:
        await agent.close()


async def test_supplied_storage_runs_tools_and_reopens_without_a_local_home(
    state_store,
    advisory_storage,
    artifact_byte_storage,
    monkeypatch,
):
    identity = AgentIdentity(
        id="agent-external", display_name="External", created_at=GRAPH_NOW
    )
    await state_store.initialize_identity(identity)
    approvals = []
    state_closes = []
    original_state_close = state_store.close

    async def observe_state_close():
        state_closes.append(True)
        await original_state_close()

    monkeypatch.setattr(state_store, "close", observe_state_close)

    async def approve(request) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    def forbid_home(*args, **kwargs):
        raise AssertionError("external composition attempted local home admission")

    monkeypatch.setattr("daita.hosting.embedded._admit_agent_home", forbid_home)
    provider = MockModelProvider(
        (_call(_memory_call(content="Keep exact totals.")), _stop("Saved."))
    )
    agent = await Agent.from_storage(
        agent_id=identity.id,
        state=state_store,
        advisory_storage=advisory_storage,
        artifact_storage=artifact_byte_storage,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        with pytest.raises(AgentHomeError, match="no local home"):
            _ = agent.home
        await agent.set_user_profile(
            "Answer concisely.", sensitivity=ModelSensitivity.INTERNAL
        )
        await agent.save_skill("report", "Prepare report", "Read the current records.")
        result = await agent.run("Remember: keep exact totals.")
        assert result.final_text == "Saved."
        assert len(approvals) == 1
        assert await agent.read_memory() == "Keep exact totals."
        assert any(
            "Answer concisely." in str(request.messages)
            for request in provider.requests
        )
    finally:
        await agent.close()
    # Supplied state is borrowed and usable after the runtime drains.
    assert await state_store.load_identity() == identity
    reopened = await Agent.from_storage(
        agent_id=identity.id,
        state=state_store,
        advisory_storage=advisory_storage,
        artifact_storage=artifact_byte_storage,
    )
    try:
        assert await reopened.read_memory() == "Keep exact totals."
        assert await reopened.read_user_profile() == "Answer concisely."
        skill = await reopened.read_skill("report")
        assert skill is not None and skill.instructions == "Read the current records."
        assert (await reopened.conversation_runs(result.conversation_id))[
            -1
        ].result == result
    finally:
        await reopened.close()
    assert state_closes == []


async def test_approval_rechecks_document_revision_even_if_bytes_are_restored(
    state_store,
    advisory_storage,
    artifact_byte_storage,
):
    identity = AgentIdentity(
        id="approval-drift", display_name="Drift", created_at=GRAPH_NOW
    )
    await state_store.initialize_identity(identity)

    async def approve(request) -> ApprovalDecision:
        await agent.set_memory("intervening edit")
        await agent.set_memory("original")
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        (_call(_memory_call(content="replacement")), _stop("Stopped."))
    )
    agent = await Agent.from_storage(
        agent_id=identity.id,
        state=state_store,
        advisory_storage=advisory_storage,
        artifact_storage=artifact_byte_storage,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        await agent.set_memory("original")
        await agent.run("Remember replacement.")
        assert await agent.read_memory() == "original"
        assert "state_changed" in str(provider.logical_requests[-1].messages)
    finally:
        await agent.close()


async def test_failed_composition_closes_owned_models_but_not_borrowed_state(
    state_store,
    advisory_storage,
    artifact_byte_storage,
    monkeypatch,
):
    identity = AgentIdentity(
        id="failed-admission", display_name="Failed", created_at=GRAPH_NOW
    )
    await state_store.initialize_identity(identity)
    profile = reviewed_model_profile("openai:gpt-5.6-terra")
    assert profile is not None
    route = ModelRoute((ModelRouteCandidate(provider_id=profile.id, profile=profile),))
    closed = []
    original_close = ModelRouter.close

    async def observe_close(self, **kwargs):
        closed.append(self)
        await original_close(self, **kwargs)

    async def fail_read(*args, **kwargs):
        raise StorageOwnershipLostError("admission lost ownership")

    async def forbid_close():
        raise AssertionError("borrowed state must remain caller-owned")

    with monkeypatch.context() as patch:
        patch.setattr(ModelRouter, "close", observe_close)
        patch.setattr(state_store, "list_mcp_bindings", fail_read)
        patch.setattr(state_store, "close", forbid_close)
        with pytest.raises(StorageOwnershipLostError):
            await Agent.from_storage(
                agent_id=identity.id,
                state=state_store,
                advisory_storage=advisory_storage,
                artifact_storage=artifact_byte_storage,
                config=AgentConfig(model_route=route),
                reviewer_max_estimated_cost_usd=Decimal("1"),
            )
    assert len(closed) == 2 and closed[0] is not closed[1]
    assert await state_store.load_identity() == identity


async def test_identity_mismatch_does_not_recover_or_close_borrowed_state(
    state_store,
    advisory_storage,
    artifact_byte_storage,
    monkeypatch,
):
    identity = AgentIdentity(
        id="actual-agent", display_name="Actual", created_at=GRAPH_NOW
    )
    await state_store.initialize_identity(identity)

    async def forbidden(*args, **kwargs):
        raise AssertionError("admission touched another agent's lifecycle")

    with monkeypatch.context() as patch:
        patch.setattr(state_store, "recover_started_effect_receipts", forbidden)
        patch.setattr(state_store, "close", forbidden)
        with pytest.raises(AgentIdentityMismatchError):
            await Agent.from_storage(
                agent_id="wrong-agent",
                state=state_store,
                advisory_storage=advisory_storage,
                artifact_storage=artifact_byte_storage,
            )
    assert await state_store.load_identity() == identity


async def test_cancelled_admission_settles_recovery_and_drains_started_runtime(
    state_store,
    advisory_storage,
    artifact_byte_storage,
    monkeypatch,
):
    identity = AgentIdentity(
        id="cancel-agent", display_name="Cancel", created_at=GRAPH_NOW
    )
    await state_store.initialize_identity(identity)
    started, release = asyncio.Event(), asyncio.Event()
    recovered = []
    closed = asyncio.Event()
    original_recover = state_store.recover_started_effect_receipts
    original_close = EmbeddedAgent.close

    async def recover(*args, **kwargs):
        started.set()
        await release.wait()
        await original_recover(*args, **kwargs)
        recovered.append("effects")

    async def observe_close(self):
        await original_close(self)
        closed.set()

    monkeypatch.setattr(state_store, "recover_started_effect_receipts", recover)
    monkeypatch.setattr("daita.hosting.embedded.EmbeddedAgent.close", observe_close)
    opening = asyncio.create_task(
        Agent.from_storage(
            agent_id=identity.id,
            state=state_store,
            advisory_storage=advisory_storage,
            artifact_storage=artifact_byte_storage,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        opening.cancel()
        await asyncio.sleep(0)
        assert not opening.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(opening, 3)
        assert recovered == ["effects"]
        assert closed.is_set()
        assert await state_store.load_identity() == identity
    finally:
        release.set()
        if not opening.done():
            await asyncio.gather(opening, return_exceptions=True)
