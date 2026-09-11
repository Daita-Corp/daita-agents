"""Component-owned tests split from ``test_conversation_mvp.py``."""

from __future__ import annotations

from tests.support.conversations import (
    NOW,
    Agent,
    AgentLoop,
    InMemoryTranscriptStore,
    LoopExit,
    LoopExitKind,
    MockModelProvider,
    NoTools,
    RunInput,
    TranscriptContext,
    _profile,
    _request_text,
    _stop,
    fields,
    pytest,
    workspace_for,
)


def test_runs_and_terminal_results_carry_explicit_conversation_identity():
    assert "conversation_id" in {field.name for field in fields(RunInput)}
    assert "conversation_id" in {field.name for field in fields(LoopExit)}

    run = RunInput(
        id="run-explicit",
        agent_id="agent-1",
        message="hello",
        created_at=NOW,
        conversation_id="conversation-explicit",
    )
    assert run.conversation_id is not None
    result = LoopExit(
        run_id=run.id,
        conversation_id=run.conversation_id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=NOW,
        final_text="answer",
    )
    assert result.conversation_id == "conversation-explicit"


async def test_direct_loop_defaults_missing_conversation_id_to_run_id():
    provider = MockModelProvider((_stop("answer"),))
    transcripts = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=NoTools(),
        transcripts=transcripts,
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(
            id="run-default",
            agent_id="agent-1",
            message="hello",
            created_at=NOW,
        )
    )
    transcript = await transcripts.load(result.run_id)

    assert result.conversation_id == "run-default"
    assert transcript.run.conversation_id == "run-default"


async def test_conversation_identity_is_agent_scoped(tmp_path):
    first_provider = MockModelProvider((_stop("first answer"),))
    first_agent = await Agent.create(
        "first",
        root=tmp_path,
        model=first_provider,
        model_profile=_profile(first_provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await first_agent.run("first agent sentinel")
    finally:
        await first_agent.close()

    second_provider = MockModelProvider((_stop("must not run"),))
    second_agent = await Agent.create(
        "second",
        root=tmp_path,
        model=second_provider,
        model_profile=_profile(second_provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        with pytest.raises(ValueError):
            await second_agent.run(
                "second agent sentinel",
                conversation_id=first.conversation_id,
            )
        assert second_provider.requests == ()
    finally:
        await second_agent.close()


async def test_public_conversation_continuation_survives_cold_reopen(tmp_path):
    first_provider = MockModelProvider((_stop("cold first answer"),))
    first_agent = await Agent.create(
        "cold",
        root=tmp_path,
        model=first_provider,
        model_profile=_profile(first_provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await first_agent.run("cold first sentinel")
        conversation_id = first.conversation_id
    finally:
        await first_agent.close()

    second_provider = MockModelProvider((_stop("cold follow-up answer"),))
    reopened = await Agent.open(
        "cold",
        root=tmp_path,
        model=second_provider,
        model_profile=_profile(second_provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        follow_up = await reopened.run(
            "cold follow-up sentinel",
            conversation_id=conversation_id,
        )
        assert follow_up.conversation_id == conversation_id
        assert _request_text(second_provider.requests[0]) == (
            "cold first sentinel",
            "cold first answer",
            "cold follow-up sentinel",
        )
    finally:
        await reopened.close()


async def test_public_ids_validate_and_omitted_ids_start_distinct_conversations(
    tmp_path,
):
    provider = MockModelProvider((_stop("one"), _stop("two")))
    agent = await Agent.create(
        "identifiers",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await agent.run("first")
        second = await agent.run("second")
        assert first.conversation_id != second.conversation_id
        assert [
            item.turn_index
            for item in await agent.conversation_runs(first.conversation_id)
        ] == [0]
        assert (await agent.conversation_runs(first.conversation_id))[0].result == first

        for invalid in ("", " space", "slash/id", "a" * 129):
            with pytest.raises(ValueError):
                await agent.run("invalid", conversation_id=invalid)
        with pytest.raises(ValueError):
            await agent.run("unknown", conversation_id="unknown-valid-id")
        with pytest.raises(ValueError):
            await agent.conversation_runs("unknown-valid-id")
        assert len(provider.requests) == 2
    finally:
        await agent.close()
