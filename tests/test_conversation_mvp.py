from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from collections.abc import Mapping

import pytest

import daita
from daita import Agent
from daita._json import FrozenJsonObject, canonical_json
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
    _MAXIMUM_PRIOR_UTF8_BYTES,
    _neutral_message,
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
                    {
                        "kind": "skill.document",
                        "data": {
                            "name": "secret-skill",
                            "instructions": "SECRET SKILL BODY",
                        },
                    }
                    if call.name == "skill_view"
                    else {
                        "kind": {
                            "memory_set": "memory.replacement",
                            "skill_save": "skill.saved",
                            "skill_delete": "skill.deleted",
                        }[call.name],
                        "data": {"ok": True},
                    }
                ),
            )
            for call in calls
        )


class FreshQueryTools:
    def __init__(self):
        self.calls = []

    async def definitions(self, run):
        del run
        return (
            ToolDefinition(
                name="data_query_postgresql",
                description="Run a fresh read-only PostgreSQL query.",
                input_schema={"type": "object"},
            ),
        )

    async def execute_all(self, run, calls):
        del run
        self.calls.extend(calls)
        return tuple(
            ToolResultBlock(
                call_id=call.id,
                output={
                    "kind": "data.postgresql.query_result",
                    "data": {
                        "columns": ["segment", "paid_revenue"],
                        "rows": [{"segment": "enterprise", "paid_revenue": 321}],
                        "total_rows": 1,
                        "returned_rows": 1,
                        "truncated": False,
                        "resource_revisions": [
                            {
                                "resource_id": "orders",
                                "revision": "sha256:" + ("d" * 64),
                            }
                        ],
                        "source_id": "warehouse",
                        "source_revision": "catalog:current",
                    },
                },
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


def _analytical_conversation_record(
    *,
    oversized: bool,
) -> tuple[ConversationRun, tuple[CanonicalMessage, ...]]:
    user_text = (
        "Analyze captured payments across all dates, grouped by region. Define paid "
        "revenue as tax-exclusive merchandise revenue from captured payments, paid "
        "order count as distinct paid orders, AOV as paid revenue divided by paid "
        "order count, COGS as merchandise unit cost times quantity, and gross margin "
        "as paid revenue minus COGS divided by paid revenue."
    )
    catalog_call = ToolCall(
        id="catalog-native-call",
        name="catalog_inspect",
        arguments={"resource_id": "orders"},
        provider_call_id="provider-catalog-call",
    )
    query_call = ToolCall(
        id="query-native-call",
        name="data_query_postgresql",
        arguments={
            "source_id": "warehouse",
            "sql": (
                "SELECT region, "
                "SUM(quantity * merchandise_unit_price) AS paid_revenue, "
                "COUNT(DISTINCT order_id) AS paid_order_count, "
                "SUM(quantity * merchandise_unit_price) / "
                "COUNT(DISTINCT order_id) AS aov, "
                "SUM(quantity * merchandise_unit_cost) AS cogs, "
                "(SUM(quantity * merchandise_unit_price) - "
                "SUM(quantity * merchandise_unit_cost)) / "
                "SUM(quantity * merchandise_unit_price) AS gross_margin "
                "FROM captured_payments JOIN order_items USING (order_id) "
                "GROUP BY region"
            ),
        },
        provider_call_id="provider-query-call",
    )
    raw_rows = [
        {
            "region": f"region-{index}",
            "paid_revenue": 123_456 + index,
            "paid_order_count": 50 + index,
            "padding": "raw-row-sentinel-" + ("x" * 700),
        }
        for index in range(40 if oversized else 1)
    ]
    messages = (
        CanonicalMessage(role=MessageRole.USER, content=(TextBlock(user_text),)),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(catalog_call,),
            provider_id="mock:history",
            provider_metadata={"native": "catalog"},
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=catalog_call.id,
                    output={
                        "kind": "catalog.resource_snapshot",
                        "data": {
                            "resource_id": "orders",
                            "schema": "catalog-snapshot-sentinel-" + ("z" * 30_000),
                        },
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(query_call,),
            provider_id="mock:history",
            provider_metadata={"native": "query"},
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=query_call.id,
                    output={
                        "kind": "data.postgresql.query_result",
                        "data": {
                            "columns": [
                                "region",
                                "paid_revenue",
                                "paid_order_count",
                                "aov",
                                "cogs",
                                "gross_margin",
                            ],
                            "rows": raw_rows,
                            "total_rows": len(raw_rows),
                            "returned_rows": len(raw_rows),
                            "truncated": False,
                            "resource_revisions": [
                                {
                                    "resource_id": "orders",
                                    "revision": "sha256:" + ("a" * 64),
                                },
                                {
                                    "resource_id": "order_items",
                                    "revision": "sha256:" + ("b" * 64),
                                },
                            ],
                            "source_id": "warehouse",
                            "source_revision": "catalog:history",
                            "trust_classification": "untrusted_external_data",
                        },
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(
                TextBlock(
                    "Overall results use captured-payment scope across all dates with "
                    "regional grouping. Paid revenue is tax-exclusive merchandise "
                    "revenue; paid order count is distinct paid orders; AOV is paid "
                    "revenue divided by paid order count; COGS is merchandise unit "
                    "cost times quantity; gross margin is (paid revenue - COGS) / "
                    "paid revenue. Overall paid revenue was 123456. A segment follow-up "
                    "must compare enterprise customers with the overall results."
                ),
            ),
            provider_id="mock:history",
            provider_metadata={"native": "answer"},
        ),
    )
    return _conversation_record(0, messages), messages


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
        assert [
            call.name
            for message in historical_assistants
            for call in message.tool_calls
        ] == ["skill_view", "skill_view"]
        assert all(call_id.startswith("hist_") for call_id in all_historical_ids)
        assert len(all_historical_ids) == len(set(all_historical_ids))
        assert all(message.provider_id is None for message in historical_assistants)
        assert all(not message.provider_metadata for message in historical_assistants)
        assert all(
            call.provider_call_id is None
            for message in historical_assistants
            for call in message.tool_calls
        )
        assert "memory_set" not in repr(historical_assistants)
        assert "skill_save" not in repr(historical_assistants)
        assert "skill_delete" not in repr(historical_assistants)

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
        assert "[historical skill body redacted]" in repr(historical_results)

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


async def test_oversized_analytical_history_keeps_compact_contract_and_requeries():
    record, durable_messages = _analytical_conversation_record(oversized=True)
    durable_payload = canonical_json(
        [_neutral_message(message) for message in durable_messages]
    )
    assert len(durable_payload.encode("utf-8")) > _MAXIMUM_PRIOR_UTF8_BYTES

    prior = _project_completed_history((record,))
    projected_payload = canonical_json([_neutral_message(message) for message in prior])
    assert len(projected_payload.encode("utf-8")) < (
        len(durable_payload.encode("utf-8")) // 3
    )
    assert len(projected_payload.encode("utf-8")) <= _MAXIMUM_PRIOR_UTF8_BYTES
    assert record.transcript.messages == durable_messages

    rendered = repr(prior)
    for contract in (
        "captured payments",
        "all dates",
        "grouped by region",
        "paid revenue",
        "paid order count",
        "AOV",
        "tax-exclusive merchandise revenue",
        "COGS",
        "gross margin",
        "enterprise customers",
        "overall results",
    ):
        assert contract in rendered
    assert "catalog-snapshot-sentinel" not in rendered
    assert "raw-row-sentinel" not in rendered
    assert "'rows'" not in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1

    historical_calls = [
        call
        for message in prior
        if message.role is MessageRole.ASSISTANT
        for call in message.tool_calls
    ]
    historical_results = [
        block
        for message in prior
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(historical_calls) == len(historical_results) == 1
    assert historical_calls[0].name == "data_query_postgresql"
    assert historical_calls[0].id.startswith("hist_")
    assert historical_calls[0].provider_call_id is None
    assert historical_results[0].call_id == historical_calls[0].id
    receipt = historical_results[0].output
    assert receipt["kind"] == "data.postgresql.query_result"
    assert receipt["historical_projection"] == "continuity"
    assert receipt["state"] == "success"
    data = receipt["data"]
    assert isinstance(data, Mapping)
    assert data["columns"] == (
        "region",
        "paid_revenue",
        "paid_order_count",
        "aov",
        "cogs",
        "gross_margin",
    )
    assert data["total_rows"] == 40
    assert data["returned_rows"] == 40
    assert data["truncated"] is False
    assert data["source_revision"] == "catalog:history"
    assert "rows" not in data

    follow_up = (
        "Now restrict that analysis to enterprise customers and compare it with "
        "the overall results."
    )
    catalog = CatalogSpy()
    profile = ModelProfile(
        id="mock:enterprise-follow-up",
        context_window_tokens=60_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    builder = DataContextBuilder(catalog, profile=profile)
    request = await builder.build(
        RunInput(
            id="enterprise-follow-up",
            agent_id="agent-history",
            message=follow_up,
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        (
            *prior,
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(follow_up),),
            ),
        ),
        (),
        step=1,
    )
    request_rendered = repr(request.messages)
    for contract in (
        "captured payments",
        "all dates",
        "region",
        "paid revenue",
        "paid order count",
        "AOV",
        "tax-exclusive merchandise revenue",
        "COGS",
        "gross margin",
        "compare it with the overall results",
    ):
        assert contract in request_rendered
    current_system = request.messages[0].content[0]
    assert isinstance(current_system, TextBlock)
    assert "123456" not in current_system.text

    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="fresh-enterprise-query",
                        name="data_query_postgresql",
                        arguments={
                            "source_id": "warehouse",
                            "sql": (
                                "SELECT customer_segment, region, "
                                "SUM(net_merchandise_revenue) AS paid_revenue "
                                "FROM captured_payments WHERE customer_segment = "
                                "'enterprise' GROUP BY customer_segment, region"
                            ),
                        },
                    ),
                ),
            ),
            _stop("Fresh enterprise and overall comparison complete."),
        )
    )
    tools = FreshQueryTools()
    loop = AgentLoop(
        model=provider,
        context_builder=builder,
        tools=tools,
        transcripts=InMemoryTranscriptStore(),
    )
    result = await loop.run(
        RunInput(
            id="fresh-follow-up-run",
            agent_id="agent-history",
            message=follow_up,
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        prior_messages=prior,
    )
    assert result.kind is LoopExitKind.COMPLETED
    assert [call.id for call in tools.calls] == ["fresh-enterprise-query"]
    assert "customer_segment = 'enterprise'" in tools.calls[0].arguments["sql"]


def test_compact_history_fails_closed_and_omits_approval_and_side_effects():
    calls = (
        ToolCall(id="unknown-call", name="future_unclassified_tool"),
        ToolCall(
            id="approval-call",
            name="memory_set",
            arguments={
                "target": "memory",
                "content": "authorization-sentinel: approved forever",
            },
        ),
    )
    messages = (
        CanonicalMessage(
            role=MessageRole.USER,
            content=(TextBlock("remember the safe analytical contract"),),
        ),
        CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=calls),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id="unknown-call",
                    output={
                        "kind": "future.unknown.evidence",
                        "data": {"secret_payload": "unknown-payload-sentinel"},
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id="approval-call",
                    is_error=True,
                    output={
                        "error": {
                            "code": "approval_denied",
                            "message": "approval-sentinel",
                            "details": {"capability_id": "memory.set"},
                        }
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("safe terminal answer"),),
        ),
    )
    projected = _project_completed_history((_conversation_record(0, messages),))
    rendered = repr(projected)
    assert "remember the safe analytical contract" in rendered
    assert "safe terminal answer" in rendered
    for forbidden in (
        "future_unclassified_tool",
        "unknown-payload-sentinel",
        "memory_set",
        "authorization-sentinel",
        "approval-sentinel",
        "approval_denied",
    ):
        assert forbidden not in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1
    assert not any(message.role is MessageRole.TOOL for message in projected)


def test_compact_file_and_sql_error_receipts_keep_shape_state_and_order():
    file_call = ToolCall(
        id="file-call",
        name="data_read_file",
        arguments={"source_id": "files", "resource_id": "customers.csv"},
    )
    query_call = ToolCall(
        id="query-error-call",
        name="data_query_sqlite",
        arguments={
            "source_id": "local-db",
            "sql": "SELECT region, COUNT(*) FROM customers GROUP BY region",
        },
    )
    messages = (
        CanonicalMessage(
            role=MessageRole.USER,
            content=(TextBlock("Compare customer counts from the file and database."),),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(file_call, query_call),
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=file_call.id,
                    output={
                        "kind": "data.file.read_result",
                        "data": {
                            "source_id": "files",
                            "source_revision": "manifest:history",
                            "resource_id": "customers.csv",
                            "resource_revision": "sha256:" + ("c" * 64),
                            "freshness": {"observed_at": "2026-07-20T00:00:00Z"},
                            "format": "csv",
                            "encoding": "utf-8",
                            "columns": ["customer_id", "region"],
                            "complete": True,
                            "rows": [
                                {
                                    "customer_id": index,
                                    "region": "raw-file-row-" + ("x" * 1_000),
                                }
                                for index in range(10)
                            ],
                            "total_rows": 10,
                            "returned_rows": 10,
                            "row_limit": 100,
                            "byte_limit": 65_536,
                            "utf8_bytes": 12_000,
                            "truncated": False,
                            "truncation_reasons": [],
                            "trust_classification": "untrusted_external_data",
                        },
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=query_call.id,
                    is_error=True,
                    output={
                        "error": {
                            "code": "sql_unknown_column",
                            "message": "unbounded-error-detail-sentinel",
                            "details": {"candidates": ["region_name"]},
                        }
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("The file shape is known; retry the database query."),),
        ),
    )
    projected = _project_completed_history((_conversation_record(0, messages),))
    calls = [
        call
        for message in projected
        if message.role is MessageRole.ASSISTANT
        for call in message.tool_calls
    ]
    results = [
        block
        for message in projected
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert [call.name for call in calls] == [
        "data_read_file",
        "data_query_sqlite",
    ]
    assert [result.call_id for result in results] == [call.id for call in calls]
    file_receipt = results[0].output
    assert file_receipt["historical_projection"] == "continuity"
    assert file_receipt["state"] == "success"
    file_data = file_receipt["data"]
    assert isinstance(file_data, Mapping)
    assert file_data["resource_id"] == "customers.csv"
    assert file_data["freshness"] == FrozenJsonObject.from_mapping(
        {"observed_at": "2026-07-20T00:00:00Z"}
    )
    assert file_data["columns"] == ("customer_id", "region")
    assert file_data["total_rows"] == 10
    assert "rows" not in file_data
    error_receipt = results[1].output
    assert error_receipt["kind"] == "data.sqlite.query_result"
    assert error_receipt["state"] == "error"
    projected_error = error_receipt["error"]
    assert isinstance(projected_error, Mapping)
    assert projected_error["code"] == "sql_unknown_column"
    assert "unbounded-error-detail-sentinel" not in repr(projected)
    assert repr(projected).count(_HISTORY_OMISSION_MARKER) == 1


def test_small_useful_query_turn_can_use_full_projection():
    record, _ = _analytical_conversation_record(oversized=False)
    projected = _project_completed_history((record,))
    results = [
        block
        for message in projected
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(results) == 1
    assert results[0].output["historical_projection"] == "full"
    data = results[0].output["data"]
    assert isinstance(data, Mapping)
    assert data["rows"][0]["region"] == "region-0"
    assert "catalog-snapshot-sentinel" not in repr(projected)


async def test_whole_request_budget_downgrades_full_turn_before_dropping_it():
    record, _ = _analytical_conversation_record(oversized=False)
    prior = _project_completed_history((record,))
    assert "'historical_projection', 'full'" in repr(prior)
    builder = DataContextBuilder(
        CatalogSpy(),
        profile=ModelProfile(
            id="mock:full-downgrade",
            context_window_tokens=15_500,
            max_output_tokens=1_000,
            supports_tools=True,
        ),
    )
    current = "current-run-message-must-remain-complete"
    request = await builder.build(
        RunInput(
            id="full-downgrade-run",
            agent_id="agent-history",
            message=current,
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        (
            *prior,
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(current),),
            ),
        ),
        (),
        step=1,
    )
    rendered = repr(request.messages)
    assert "'historical_projection', 'continuity'" in rendered
    assert "'historical_projection', 'full'" not in rendered
    assert "Analyze captured payments" in rendered
    assert "current-run-message-must-remain-complete" in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1


def test_oversized_terminal_answer_gets_deterministic_edge_projection():
    answer = "answer-beginning-sentinel-" + ("x" * 40_000) + "-answer-ending-sentinel"
    projected = _project_completed_history(
        (_simple_conversation_record(0, answer=answer),)
    )
    rendered = repr(projected)
    assert "history user 0" in rendered
    assert "answer-beginning-sentinel" in rendered
    assert "answer-ending-sentinel" in rendered
    assert f"original UTF-8 bytes: {len(answer.encode('utf-8'))}" in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1
    assert (
        len(
            canonical_json([_neutral_message(message) for message in projected]).encode(
                "utf-8"
            )
        )
        <= _MAXIMUM_PRIOR_UTF8_BYTES
    )


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
    # Unknown future evidence fails closed, so all eight bounded user/answer
    # continuity pairs fit and the tool payloads are represented by one marker.
    assert len(message_bounded) == 17
    assert "message-heavy user 0" in repr(message_bounded)
    assert "message-heavy user 7" in repr(message_bounded)
    assert "name='lookup'" not in repr(message_bounded)

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


async def test_historical_schema_slice_reuse_requires_current_matching_revisions():
    resource_id = "catalog-resource:sha256:" + ("a" * 64)
    source_id = "source:sha256:" + ("b" * 64)
    revision = "sha256:" + ("c" * 64)
    sync_id = "catalog-sync-current"
    source_revision = "catalog:sha256:" + ("d" * 64)
    schema_call = ToolCall(
        id="schema-history-call",
        name="catalog_schema",
        arguments={"resource_ids": (resource_id,)},
    )
    record = _conversation_record(
        0,
        (
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("Plan the paid revenue query"),),
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(schema_call,),
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id=schema_call.id,
                        output={
                            "kind": "catalog.schema_slice",
                            "data": {
                                "bounds": {"resources": 12},
                                "include_relationships": True,
                                "relationships": (),
                                "resources": (
                                    {
                                        "columns": (
                                            {
                                                "name": "paid_revenue",
                                                "nullable": False,
                                                "type": "NUMERIC",
                                            },
                                        ),
                                        "kind": "table",
                                        "name": "analytics.orders",
                                        "primary_key_fields": ("order_id",),
                                        "resource_id": resource_id,
                                        "revision": revision,
                                        "source_id": source_id,
                                        "structural_facts": {},
                                        "sync_id": sync_id,
                                        "unique_key_fields": (),
                                    },
                                ),
                                "sources": (
                                    {
                                        "source_id": source_id,
                                        "source_revision": source_revision,
                                        "sync_id": sync_id,
                                    },
                                ),
                                "total_matches": 1,
                                "truncation": {"resources": False},
                                "trust_classification": "untrusted_external_data",
                            },
                        },
                    ),
                ),
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock("Use paid_revenue from analytics.orders."),),
            ),
        ),
    )
    prior = _project_completed_history((record,))
    assert "catalog.schema_slice" in repr(prior)
    assert "paid_revenue" in repr(prior)

    current_resource = {
        "kind": "table",
        "name": "orders",
        "resource_id": resource_id,
        "revision": revision,
        "sensitivity": "internal",
        "source_id": source_id,
        "source_revision": source_revision,
        "sync_id": sync_id,
    }
    profile = ModelProfile(
        id="mock:schema-history",
        context_window_tokens=60_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    run = RunInput(
        id="schema-history-follow-up",
        agent_id="agent-history",
        message="Now only EMEA",
        created_at=NOW,
        conversation_id="history-conversation",
    )
    current_user = CanonicalMessage(
        role=MessageRole.USER,
        content=(TextBlock(run.message),),
    )

    unchanged = await DataContextBuilder(
        CatalogSpy((current_resource,)),
        profile=profile,
    ).build(
        run,
        (*prior, current_user),
        (),
        step=1,
    )
    unchanged_text = repr(unchanged.messages)
    assert "catalog.schema_slice" in unchanged_text
    assert "paid_revenue" in unchanged_text

    changed_resource = {
        **current_resource,
        "revision": "sha256:" + ("e" * 64),
        "sync_id": "catalog-sync-refreshed",
        "source_revision": "catalog:sha256:" + ("f" * 64),
    }
    changed = await DataContextBuilder(
        CatalogSpy((changed_resource,)),
        profile=profile,
    ).build(
        run,
        (*prior, current_user),
        (),
        step=1,
    )
    changed_text = repr(changed.messages)
    assert "catalog.schema_slice" not in changed_text
    assert "'name', 'paid_revenue'" not in changed_text
    assert _HISTORY_OMISSION_MARKER in changed_text
