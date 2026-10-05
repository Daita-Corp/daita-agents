"""Stable pages and narrow authority reads preserve the existing state contract."""

import pytest

from daita.llm.models import CanonicalMessage, MessageRole, TextBlock
from daita.loop.models import ConversationRun, LoopExit, LoopExitKind, RunInput
from daita.storage.protocols import StateStore
from tests.support.graph import GRAPH_NOW, graph_admission

pytestmark = [pytest.mark.integration, pytest.mark.contract]


async def test_conversation_pages_preserve_exact_turns_and_caller_scope(
    state_store: StateStore,
):
    for index in range(5):
        run = RunInput(
            id=f"run-{index}",
            agent_id="agent-1",
            conversation_id="conversation-1",
            caller_principal_id="reader" if index != 4 else "other",
            message=f"question {index}",
            created_at=GRAPH_NOW,
        )
        await state_store.start(run)
        await state_store.append_at(run.id, 0, run.start_message())
        result = LoopExit(
            run_id=run.id,
            conversation_id="conversation-1",
            kind=LoopExitKind.COMPLETED,
            final_text=f"answer {index}",
            reason="completed",
            created_at=GRAPH_NOW,
        )
        await state_store.complete(
            result,
            CanonicalMessage(
                role=MessageRole.ASSISTANT, content=(TextBlock(f"answer {index}"),)
            ),
        )
    collected: list[ConversationRun] = []
    cursor = -1
    while page := await state_store.conversation_run_page(
        "agent-1", "conversation-1", after_turn_index=cursor, limit=2
    ):
        assert 1 <= len(page) <= 2
        collected.extend(page)
        cursor = page[-1].turn_index
    assert tuple(collected) == await state_store.conversation_runs(
        "agent-1", "conversation-1"
    )
    assert [run.turn_index for run in collected] == list(range(5))
    assert [len(run.transcript.messages) for run in collected] == [2] * 5
    assert await state_store.conversation_run_page("foreign", "conversation-1") == ()
    assert await state_store.conversation_access("agent-1", "conversation-1") == (
        True,
        True,
    )
    assert await state_store.conversation_access(
        "agent-1", "conversation-1", caller_principal_id="reader"
    ) == (True, False)
    assert await state_store.conversation_access(
        "foreign", "conversation-1", caller_principal_id="reader"
    ) == (False, False)
    for limit in (0, 101, True):
        with pytest.raises(ValueError):
            await state_store.conversation_run_page(
                "agent-1", "conversation-1", limit=limit
            )


async def test_attempt_authority_snapshot_is_agent_scoped(state_store: StateStore):
    admission = graph_admission()
    await state_store.admit_graph(admission)
    snapshot = await state_store.read_graph_attempt_authority(
        "agent-1", "job-1", "worker", "missing"
    )
    assert snapshot is not None
    assert snapshot.job == admission.job
    assert snapshot.task == next(
        task for task in admission.tasks if task.task_id == "worker"
    )
    assert snapshot.attempt is None
    assert (
        await state_store.read_graph_attempt_authority(
            "foreign", "job-1", "worker", "missing"
        )
        is None
    )
