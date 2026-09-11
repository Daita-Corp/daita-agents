"""Component-owned tests split from ``test_kernel_contracts.py``."""

from __future__ import annotations

from tests.support.kernel import (
    NOW,
    CanonicalMessage,
    LoopExit,
    LoopExitKind,
    MessageRole,
    SQLiteStateStore,
    TextBlock,
    _run,
    pytest,
)


async def test_sqlite_atomic_completion_rolls_back_both_values_on_encode_failure(
    tmp_path,
    monkeypatch,
):
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    run = _run("run-sqlite-atomic-failure")
    user = CanonicalMessage(role=MessageRole.USER, content=(TextBlock("question"),))
    final = CanonicalMessage(
        role=MessageRole.ASSISTANT,
        content=(TextBlock("answer"),),
    )
    result = LoopExit(
        run_id=run.id,
        conversation_id=run.conversation_id or run.id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        final_text="answer",
        created_at=NOW,
    )
    await store.start(run)
    await store.append(run.id, user)

    def fail_encode(_result):
        raise RuntimeError("injected terminal encoding failure")

    monkeypatch.setattr("daita.storage.sqlite.encode_loop_exit", fail_encode)
    try:
        with pytest.raises(RuntimeError, match="injected terminal encoding failure"):
            await store.complete(result, final)

        assert (await store.load(run.id)).messages == (user,)
        assert await store.result(run.id) is None
    finally:
        await store.close()
