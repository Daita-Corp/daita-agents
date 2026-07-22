from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from collections.abc import Mapping

import pytest

import daita
from daita import Agent
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop import (
    AgentLoop,
    ConversationRun,
    InMemoryTranscriptStore,
    LoopExit,
    LoopExitKind,
    RunInput,
    Transcript,
)
from daita.domains.data.context import (
    DataContextBuilder,
    _HISTORY_OMISSION_MARKER,
    _project_completed_history,
)
from daita.storage.sqlite import SQLiteStateStore

NOW = datetime(2026, 7, 21, tzinfo=timezone.utc)


class TranscriptContext:
    async def build(self, run, messages, tools, *, step, final=False):
        del run, step, final
        return ModelRequest(messages=messages, tools=tools)


class NoTools:
    async def definitions(self, run):
        del run
        return ()

    async def execute_all(self, run, calls):
        del run
        assert calls == ()
        return ()


class ReplayTools:
    async def definitions(self, run):
        del run
        return tuple(
            ToolDefinition(
                name=name,
                description=name,
                input_schema={"type": "object"},
            )
            for name in ("memory_set", "skill_save", "skill_delete", "skill_view")
        )

    async def execute_all(self, run, calls):
        del run
        return tuple(
            ToolResultBlock(
                call_id=call.id,
                output=(
                    {"name": "secret-skill", "body": "SECRET SKILL BODY"}
                    if call.name == "skill_view"
                    else {"ok": True}
                ),
            )
            for call in calls
        )


class CatalogSpy:
    def __init__(self, resources=()):
        self.queries = []
        self.resources = resources

    async def catalog_context(
        self,
        agent_id,
        query,
        *,
        limit,
        source_ids=(),
        resource_ids=(),
    ):
        del agent_id, limit, source_ids, resource_ids
        from daita._json import FrozenJsonObject

        self.queries.append(query)
        return FrozenJsonObject.from_mapping(
            {
                "resources": self.resources,
                "total_matches": len(self.resources),
                "truncated": False,
                "trust_classification": "untrusted_external_data",
            }
        )


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _tool_response(run_number: int) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(
            ToolCall(
                id=f"shared-{name}",
                name=name,
                arguments={"content": f"SECRET {name} {run_number}"},
                provider_call_id=f"native-{name}",
            )
            for name in ("memory_set", "skill_save", "skill_delete", "skill_view")
        ),
        provider_id="mock:scripted",
        provider_metadata={"continuation": {"run": run_number}},
    )


def _conversation_record(
    index: int,
    messages: tuple[CanonicalMessage, ...],
    *,
    kind: LoopExitKind = LoopExitKind.COMPLETED,
) -> ConversationRun:
    run = RunInput(
        id=f"history-run-{index}",
        agent_id="agent-history",
        message=f"history user {index}",
        created_at=NOW,
        conversation_id="history-conversation",
    )
    result = LoopExit(
        run_id=run.id,
        conversation_id="history-conversation",
        kind=kind,
        reason=kind.value,
        created_at=NOW,
        final_text=(
            f"history answer {index}" if kind is LoopExitKind.COMPLETED else None
        ),
    )
    return ConversationRun(
        turn_index=index,
        transcript=Transcript(run=run, messages=messages),
        result=result,
    )


def _simple_conversation_record(index: int, answer: str | None = None):
    return _conversation_record(
        index,
        (
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(f"history user {index}"),),
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock(answer or f"history answer {index}"),),
            ),
        ),
    )


def _request_text(request: ModelRequest) -> tuple[str, ...]:
    return tuple(
        block.text
        for message in request.messages
        if message.role.value != "system"
        for block in message.content
        if isinstance(block, TextBlock)
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


async def test_follow_up_uses_history_without_copying_it_into_new_transcript(tmp_path):
    provider = MockModelProvider((_stop("first answer"), _stop("follow-up answer")))
    agent = await Agent.create(
        "integrity",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        first = await agent.run("first user sentinel")
        follow_up = await agent.run(
            "follow-up user sentinel",
            conversation_id=first.conversation_id,
        )
        transcript = await agent.transcript(follow_up.run_id)

        assert _request_text(provider.requests[1]) == (
            "first user sentinel",
            "first answer",
            "follow-up user sentinel",
        )
        assert tuple(
            block.text
            for message in transcript.messages
            for block in message.content
            if isinstance(block, TextBlock)
        ) == ("follow-up user sentinel", "follow-up answer")
    finally:
        await agent.close()


async def test_sqlite_groups_runs_with_only_documented_columns_and_index(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = await SQLiteStateStore.open(path)
    await store.close()

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(runs)"))
        index = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'index' AND name = 'runs_conversation_turn'"
        ).fetchone()
        index_columns = tuple(
            row[2]
            for row in connection.execute("PRAGMA index_info(runs_conversation_turn)")
        )

    assert "conversations" not in tables
    assert columns == (
        "id",
        "agent_id",
        "conversation_id",
        "turn_index",
        "input",
        "result",
    )
    assert index is not None
    assert "CREATE UNIQUE INDEX" in index[1].upper()
    assert index_columns == ("agent_id", "conversation_id", "turn_index")


def test_conversation_grouping_adds_no_table_package_or_runtime():
    assert daita.__file__ is not None
    daita_package = Path(daita.__file__).parent
    storage_text = (daita_package / "storage" / "sqlite.py").read_text(encoding="utf-8")
    production_text = "\n".join(
        path.read_text(encoding="utf-8") for path in daita_package.rglob("*.py")
    )

    assert "CREATE TABLE IF NOT EXISTS conversations" not in storage_text
    assert "ConversationRuntime" not in production_text


async def test_public_conversation_continuation_survives_cold_reopen(tmp_path):
    first_provider = MockModelProvider((_stop("cold first answer"),))
    first_agent = await Agent.create(
        "cold",
        root=tmp_path,
        model=first_provider,
        model_profile=_profile(first_provider),
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


async def test_only_completed_runs_are_eligible_for_follow_up_history(tmp_path):
    provider = MockModelProvider(
        (
            _stop("eligible completed answer"),
            ModelProviderError(ProviderErrorCode.TIMEOUT),
            _stop("final answer"),
        )
    )
    agent = await Agent.create(
        "eligibility",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        first = await agent.run("eligible completed user")
        failed = await agent.run(
            "ineligible failed user",
            conversation_id=first.conversation_id,
        )
        assert failed.kind is LoopExitKind.FAILED

        await agent.run(
            "final follow-up user",
            conversation_id=first.conversation_id,
        )
        final_request_text = _request_text(provider.requests[-1])
        assert "eligible completed user" in final_request_text
        assert "eligible completed answer" in final_request_text
        assert "ineligible failed user" not in final_request_text
    finally:
        await agent.close()


async def test_public_ids_validate_and_omitted_ids_start_distinct_conversations(
    tmp_path,
):
    provider = MockModelProvider((_stop("one"), _stop("two")))
    agent = await Agent.create(
        "identifiers",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
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


async def test_historical_tools_are_rewritten_redacted_and_provider_neutral(
    tmp_path,
):
    provider = MockModelProvider(
        (
            _tool_response(1),
            _stop("first answer"),
            _tool_response(2),
            _stop("second answer"),
            _stop("third answer"),
        )
    )
    agent = await Agent.create(
        "historical-tools",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        tools=ReplayTools(),
        context_builder=TranscriptContext(),
    )
    try:
        first = await agent.run("first tool turn")
        second = await agent.run(
            "second tool turn",
            conversation_id=first.conversation_id,
        )
        await agent.run("third turn", conversation_id=first.conversation_id)

        # Provider-native continuation remains intact inside the current run.
        current_assistant = next(
            message
            for message in provider.requests[1].messages
            if message.role is MessageRole.ASSISTANT and message.tool_calls
        )
        assert current_assistant.provider_id == "mock:scripted"
        continuation = current_assistant.provider_metadata["continuation"]
        assert isinstance(continuation, Mapping)
        assert continuation["run"] == 1
        assert all(
            call.provider_call_id is not None for call in current_assistant.tool_calls
        )

        historical_request = provider.requests[-1]
        historical_assistants = [
            message
            for message in historical_request.messages
            if message.role is MessageRole.ASSISTANT and message.tool_calls
        ]
        assert len(historical_assistants) == 2
        all_historical_ids = [
            call.id for message in historical_assistants for call in message.tool_calls
        ]
        assert all(call_id.startswith("hist_") for call_id in all_historical_ids)
        assert len(all_historical_ids) == len(set(all_historical_ids))
        assert all(message.provider_id is None for message in historical_assistants)
        assert all(not message.provider_metadata for message in historical_assistants)
        assert all(
            call.provider_call_id is None
            for message in historical_assistants
            for call in message.tool_calls
        )
        assert all(
            call.arguments.get("redacted") == "[historical knowledge document redacted]"
            for message in historical_assistants
            for call in message.tool_calls
            if call.name != "skill_view"
        )

        historical_results = [
            block
            for message in historical_request.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert {block.call_id for block in historical_results} == set(
            all_historical_ids
        )
        assert "SECRET SKILL BODY" not in repr(historical_results)

        # Durable per-run transcripts retain their exact canonical records.
        first_transcript = await agent.transcript(first.run_id)
        second_transcript = await agent.transcript(second.run_id)
        for transcript in (first_transcript, second_transcript):
            assistant = transcript.messages[1]
            assert assistant.tool_calls[0].id == "shared-memory_set"
            assert assistant.tool_calls[0].provider_call_id == "native-memory_set"
            assert assistant.provider_metadata
            assert "SECRET" in repr(assistant.tool_calls[0].arguments)
            assert "SECRET SKILL BODY" in repr(transcript.messages)
    finally:
        await agent.close()


def test_completed_history_hard_bounds_keep_newest_whole_turns():
    ten_runs = tuple(_simple_conversation_record(index) for index in range(10))
    run_bounded = _project_completed_history(ten_runs)
    run_text = _request_text(ModelRequest(messages=run_bounded))
    assert "history user 0" not in run_text
    assert "history user 1" not in run_text
    assert "history user 2" in run_text
    assert "history user 9" in run_text
    run_marker = run_bounded[0].content[0]
    assert isinstance(run_marker, TextBlock)
    assert run_marker.text == _HISTORY_OMISSION_MARKER

    message_heavy = []
    for index in range(8):
        calls = tuple(
            ToolCall(id=f"{index}-{item}", name="lookup") for item in range(4)
        )
        messages = [
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(f"message-heavy user {index}"),),
            ),
            CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=calls),
            *(
                CanonicalMessage(
                    role=MessageRole.TOOL,
                    content=(ToolResultBlock(call_id=call.id, output={"ok": True}),),
                )
                for call in calls
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock(f"message-heavy answer {index}"),),
            ),
        ]
        message_heavy.append(_conversation_record(index, tuple(messages)))
    message_bounded = _project_completed_history(tuple(message_heavy))
    assert len(message_bounded) == 36  # marker plus five complete seven-message turns
    assert "message-heavy user 2" not in repr(message_bounded)
    assert "message-heavy user 3" in repr(message_bounded)

    byte_heavy = tuple(
        _simple_conversation_record(index, answer=str(index) + ("x" * 12_500))
        for index in range(2)
    )
    byte_bounded = _project_completed_history(byte_heavy)
    assert "history user 0" not in repr(byte_bounded)
    assert "history user 1" in repr(byte_bounded)
    byte_marker = byte_bounded[0].content[0]
    assert isinstance(byte_marker, TextBlock)
    assert byte_marker.text == _HISTORY_OMISSION_MARKER


async def test_final_budget_omits_oldest_turns_and_preserves_current_exchange():
    catalog = CatalogSpy()
    profile = ModelProfile(
        id="mock:budget",
        context_window_tokens=15_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )
    builder = DataContextBuilder(catalog, profile=profile)
    prior = _project_completed_history(
        tuple(
            _simple_conversation_record(index, answer=f"answer-{index}-" + "x" * 1_500)
            for index in range(3)
        )
    )
    current_call = ToolCall(id="current-call", name="lookup")
    current = (
        CanonicalMessage(
            role=MessageRole.USER,
            content=(TextBlock("current user sentinel"),),
        ),
        CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(current_call,)),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(ToolResultBlock(call_id=current_call.id, output={"value": 1}),),
        ),
    )
    request = await builder.build(
        RunInput(
            id="budget-run",
            agent_id="agent-history",
            message="current user sentinel",
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        (*prior, *current),
        (),
        step=2,
    )
    rendered = repr(request.messages)
    assert "history user 0" not in rendered
    assert "history user 2" in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1
    assert "current user sentinel" in rendered
    assert "current-call" in rendered
    roles = [message.role for message in request.messages]
    assert roles[-3:] == [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL]


async def test_catalog_query_uses_current_and_most_recent_prior_user_message():
    catalog = CatalogSpy()
    builder = DataContextBuilder(
        catalog,
        profile=ModelProfile(
            id="mock:catalog-query",
            context_window_tokens=20_000,
            max_output_tokens=1_000,
            supports_tools=True,
        ),
    )
    prior = _project_completed_history(
        (
            _simple_conversation_record(0),
            _simple_conversation_record(1),
        )
    )
    request = await builder.build(
        RunInput(
            id="referential-run",
            agent_id="agent-history",
            message="Now only EMEA",
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        (
            *prior,
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("Now only EMEA"),),
            ),
        ),
        (),
        step=1,
    )
    assert catalog.queries == ["Now only EMEA\n\nPrior user message:\nhistory user 1"]
    system_text = request.messages[0].content[0]
    assert isinstance(system_text, TextBlock)
    assert (
        "fresh source/tool evidence outrank stale historical claims" in system_text.text
    )


async def test_context_overflow_fails_before_provider_and_is_not_replayed(tmp_path):
    provider = MockModelProvider((_stop("must not run"),))
    tiny_profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=2_000,
        max_output_tokens=500,
        supports_tools=True,
    )
    agent = await Agent.create(
        "overflow",
        root=tmp_path,
        model=provider,
        model_profile=tiny_profile,
    )
    try:
        failed = await agent.run("overflow sentinel " + ("x" * 2_000))
        assert failed.kind is LoopExitKind.FAILED
        assert failed.reason == "context_window_exceeded"
        assert provider.requests == ()
        records = await agent.conversation_runs(failed.conversation_id)
        assert records[0].result == failed
    finally:
        await agent.close()

    resumed_provider = MockModelProvider((_stop("recovered"),))
    reopened = await Agent.open(
        "overflow",
        root=tmp_path,
        model=resumed_provider,
        model_profile=_profile(resumed_provider),
    )
    try:
        await reopened.run("clean follow-up", conversation_id=failed.conversation_id)
        assert "overflow sentinel" not in repr(resumed_provider.requests[0].messages)
    finally:
        await reopened.close()


async def test_inspection_keeps_nonreplayable_runs_but_history_excludes_them(tmp_path):
    initial_provider = MockModelProvider((_stop("valid answer"),))
    agent = await Agent.create(
        "inspection",
        root=tmp_path,
        model=initial_provider,
        model_profile=_profile(initial_provider),
    )
    try:
        first = await agent.run("valid completed user")
        agent_id = agent.id
        state_path = agent.home / "state.db"
    finally:
        await agent.close()

    store = await SQLiteStateStore.open(state_path)
    try:
        for offset, (sentinel, kind) in enumerate(
            (
                ("failed sentinel", LoopExitKind.FAILED),
                ("interrupted sentinel", LoopExitKind.INTERRUPTED),
                ("incomplete completed sentinel", LoopExitKind.COMPLETED),
            ),
            start=1,
        ):
            run = RunInput(
                id=f"manual-{offset}",
                agent_id=agent_id,
                message=sentinel,
                created_at=NOW,
                conversation_id=first.conversation_id,
            )
            await store.start(run)
            await store.append(
                run.id,
                CanonicalMessage(
                    role=MessageRole.USER,
                    content=(TextBlock(sentinel),),
                ),
            )
            await store.finish(
                LoopExit(
                    run_id=run.id,
                    conversation_id=first.conversation_id,
                    kind=kind,
                    reason=kind.value,
                    created_at=NOW,
                    final_text=(
                        "claimed complete" if kind is LoopExitKind.COMPLETED else None
                    ),
                )
            )
        unfinished = RunInput(
            id="manual-unfinished",
            agent_id=agent_id,
            message="unfinished sentinel",
            created_at=NOW,
            conversation_id=first.conversation_id,
        )
        await store.start(unfinished)
        await store.append(
            unfinished.id,
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(unfinished.message),),
            ),
        )
    finally:
        await store.close()

    provider = MockModelProvider((_stop("follow-up answer"),))
    reopened = await Agent.open(
        "inspection",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        await reopened.run(
            "inspection follow-up", conversation_id=first.conversation_id
        )
        request = repr(provider.requests[0].messages)
        assert "valid completed user" in request
        for sentinel in (
            "failed sentinel",
            "interrupted sentinel",
            "incomplete completed sentinel",
            "unfinished sentinel",
        ):
            assert sentinel not in request
        records = await reopened.conversation_runs(first.conversation_id)
        assert [record.turn_index for record in records] == list(range(6))
        assert records[4].result is None
        assert records[1].result is not None
        assert records[2].result is not None
        assert records[3].result is not None
        assert records[1].result.kind is LoopExitKind.FAILED
        assert records[2].result.kind is LoopExitKind.INTERRUPTED
        assert records[3].result.kind is LoopExitKind.COMPLETED
    finally:
        await reopened.close()
