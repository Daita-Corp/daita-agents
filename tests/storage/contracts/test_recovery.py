"""Persisted identity and transcript behavior shared by durable backends."""

from dataclasses import replace
from datetime import timedelta

import pytest

from daita.identity import AgentIdentity, AgentIdentityConflictError
from daita.llm.models import CanonicalMessage, MessageRole, TextBlock
from daita.loop.analysis import AnalysisEvidence
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.loop.transcripts import ConversationPredecessor
from tests.storage._support import StateStoreFactory
from tests.support.graph import GRAPH_NOW

pytestmark = [pytest.mark.integration, pytest.mark.contract]


async def test_identity_reopen_and_conflict_preserve_original(
    state_store_factory: StateStoreFactory,
) -> None:
    identity = AgentIdentity("agent-1", "Storage contract", GRAPH_NOW)
    async with state_store_factory() as store:
        assert await store.load_identity() is None
        assert await store.initialize_identity(identity) == identity

    async with state_store_factory() as reopened:
        assert await reopened.load_identity() == identity
        assert await reopened.initialize_identity(identity) == identity
        with pytest.raises(AgentIdentityConflictError):
            await reopened.initialize_identity(replace(identity, id="another-agent"))
        assert await reopened.load_identity() == identity


async def test_terminal_transcript_reopens_with_exact_order_and_predecessor(
    state_store_factory: StateStoreFactory,
) -> None:
    run = RunInput(
        id="run-1",
        agent_id="agent-1",
        message="question",
        conversation_id="conversation-1",
        created_at=GRAPH_NOW,
    )
    user = CanonicalMessage(role=MessageRole.USER, content=(TextBlock("question"),))
    final = CanonicalMessage(role=MessageRole.ASSISTANT, content=(TextBlock("answer"),))
    result = LoopExit(
        run_id=run.id,
        conversation_id="conversation-1",
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        final_text="answer",
        created_at=GRAPH_NOW + timedelta(seconds=1),
    )
    async with state_store_factory() as store:
        await store.start(run, predecessor=None)
        with pytest.raises(ValueError, match="out of order"):
            await store.append_at(run.id, 1, user)
        assert (await store.load(run.id)).messages == ()
        await store.append_at(run.id, 0, user)
        await store.complete(result, final)
        with pytest.raises(ValueError, match="terminal"):
            await store.complete(result, final)
        with pytest.raises(ValueError, match="terminal"):
            await store.append_at(run.id, 2, user)

    async with state_store_factory() as reopened:
        assert (await reopened.load(run.id)).messages == (user, final)
        assert await reopened.result(run.id) == result
        history = await reopened.conversation_runs("agent-1", "conversation-1")
        assert len(history) == 1 and history[0].turn_index == 0
        assert await reopened.conversation_runs("another-agent", "conversation-1") == ()
        predecessor = ConversationPredecessor.from_run(history[0])
        next_run = replace(run, id="run-2")
        with pytest.raises(ValueError, match="predecessor"):
            await reopened.start(next_run, predecessor=None)
        await reopened.start(next_run, predecessor=predecessor)
        with pytest.raises(ValueError, match="predecessor"):
            await reopened.start(replace(run, id="stale-run"), predecessor=predecessor)
        history = await reopened.conversation_runs("agent-1", "conversation-1")
        assert [item.transcript.run.id for item in history] == ["run-1", "run-2"]
        assert [item.turn_index for item in history] == [0, 1]


async def test_unfinished_recovery_is_durable_and_does_not_replay(
    state_store_factory: StateStoreFactory,
) -> None:
    run = RunInput(
        id="unfinished",
        agent_id="agent-1",
        message="question",
        conversation_id="conversation-1",
        created_at=GRAPH_NOW,
    )
    user = CanonicalMessage(role=MessageRole.USER, content=(TextBlock("question"),))
    async with state_store_factory() as store:
        await store.start(run, predecessor=None)
        await store.append_at(run.id, 0, user)

    async with state_store_factory() as reopened:
        assert await reopened.result(run.id) is None
        recovered = await reopened.recover_unfinished_runs(
            "agent-1", created_at=GRAPH_NOW
        )
        assert len(recovered) == 1
        assert recovered[0].run_id == run.id
        assert recovered[0].kind is LoopExitKind.INTERRUPTED
        assert recovered[0].reason == "previous_process_terminated"
        assert (await reopened.load(run.id)).messages == (user,)

    async with state_store_factory() as reopened:
        assert await reopened.result(run.id) == recovered[0]
        assert (
            await reopened.recover_unfinished_runs("agent-1", created_at=GRAPH_NOW)
            == ()
        )
        assert (await reopened.load(run.id)).messages == (user,)


async def test_terminal_cleanup_recovery_appends_once_under_exact_cas(
    state_store_factory,
):
    run = RunInput(
        id="cleanup-run",
        agent_id="agent-1",
        message="fixture",
        conversation_id="cleanup-conversation",
        created_at=GRAPH_NOW,
    )
    identity = {
        "scratch": "/private/tmp/daita-analysis-fixture",
        "scratch_device": 1,
        "scratch_inode": 2,
        "closure_path": "/private/tmp/daita-analysis-closure-fixture",
        "closure_device": 1,
        "closure_inode": 3,
        "closure_nonce": "a" * 64,
    }
    original = AnalysisEvidence(
        run.id,
        "parser-1",
        "generation",
        {
            "status": "cleanup_failed",
            **identity,
            "usage": {
                "cpu_complete": False,
                "user_cpu_seconds": None,
                "system_cpu_seconds": None,
            },
            "cleanup": {"scratch_deleted": False, "failure": "PermissionError"},
        },
    )
    native = {
        "status": "recovered_closed",
        **identity,
        "pid": None,
        "usage": {
            "cpu_complete": True,
            "user_cpu_seconds": 0.0,
            "system_cpu_seconds": 0.0,
        },
        "cleanup": {
            "process_reaped": True,
            "process_spawned": False,
            "scratch_deleted": True,
            "descriptors_closed": True,
            "child_io_settled": True,
        },
    }
    result = LoopExit(
        run_id=run.id,
        conversation_id=run.conversation_id or run.id,
        kind=LoopExitKind.FAILED,
        reason="analysis_cleanup_failed",
        created_at=GRAPH_NOW,
    )
    async with state_store_factory() as store:
        await store.initialize_identity(AgentIdentity("agent-1", "Fixture", GRAPH_NOW))
        await store.start(run)
        await store.record_analysis_evidence(
            "agent-1",
            AnalysisEvidence(
                run.id,
                original.evidence_id,
                original.kind,
                {**original.facts, "status": "admitted"},
            ),
        )
        await store.record_analysis_evidence("agent-1", original)
        await store.finish(result)
        with pytest.raises(ValueError, match="live"):
            await store.record_analysis_evidence("agent-1", original)
        with pytest.raises(ValueError, match="identity"):
            await store.recover_analysis_evidence("other-agent", original, native)
        with pytest.raises(ValueError, match="identity"):
            await store.recover_analysis_evidence(
                "agent-1", original, {**native, "closure_nonce": "b" * 64}
            )
        recovered = await store.recover_analysis_evidence("agent-1", original, native)
        assert (
            await store.recover_analysis_evidence("agent-1", original, native)
            == recovered
        )
        assert recovered.facts["status"] == "cleanup_failed"
        assert recovered.facts["cleanup"] == original.facts["cleanup"]
        assert recovered.facts["usage"] == original.facts["usage"]
        with pytest.raises(ValueError, match="once-only"):
            await store.recover_analysis_evidence("agent-1", recovered, native)
        disposed = await store.recover_analysis_evidence(
            "agent-1", recovered, proof_disposed=True
        )
        assert (
            await store.recover_analysis_evidence(
                "agent-1", recovered, proof_disposed=True
            )
            == disposed
        )
        assert await store.result(run.id) == result
    async with state_store_factory() as store:
        assert await store.result(run.id) == result
        assert await store.load_analysis_evidence("agent-1", run.id) == (disposed,)
