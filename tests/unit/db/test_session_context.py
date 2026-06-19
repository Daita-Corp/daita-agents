import json

from daita.agents.agent import Agent
from daita.agents.conversation import ConversationHistory
from daita.db import DbAgent, DbRequest, DbRuntime
from daita.db.models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
)
from daita.db.session_context import (
    DbSessionContextBuilder,
    db_session_context_from_request,
)
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode, Evidence, Operation, OperationStatus


class SpyRuntime(DbRuntime):
    def __init__(self) -> None:
        super().__init__(runtime_id="db-session-spy")
        self.run_requests: list[DbRequest] = []

    async def run(self, request):
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        self.run_requests.append(db_request)
        return DbOperationResult(
            operation_id=f"spy-{len(self.run_requests)}",
            request=db_request,
            intent=DbIntent(
                kind=DbIntentKind.CONVERSATIONAL,
                confidence=1.0,
                access=AccessMode.NONE,
            ),
            contract=DbOperationContract(operation_type="conversational"),
            status=OperationStatus.SUCCEEDED,
            answer="spy answer",
        )


async def _seed_agent_schema(path):
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script("""
        CREATE TABLE agent_profiles (
            id INTEGER PRIMARY KEY,
            agent_name TEXT NOT NULL
        );
        CREATE TABLE agent_runs (
            id INTEGER PRIMARY KEY,
            profile_id INTEGER REFERENCES agent_profiles(id),
            status TEXT NOT NULL
        );
        CREATE TABLE billing_events (
            id INTEGER PRIMARY KEY,
            amount_cents INTEGER NOT NULL
        );
        INSERT INTO agent_profiles (agent_name) VALUES ('alpha');
        INSERT INTO agent_runs (profile_id, status) VALUES (1, 'ok');
        INSERT INTO billing_events (amount_cents) VALUES (100);
        """)
    await plugin.disconnect()


async def test_session_context_builder_collects_runtime_referents_and_diagnostics():
    runtime = DbRuntime(runtime_id="session-context-builder")
    operation = Operation(
        id="op-1",
        operation_type="schema.query",
        status=OperationStatus.SUCCEEDED,
        request={"prompt": "What columns are in customers?", "session_id": "s1"},
        metadata={"session_id": "s1"},
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_evidence(
        Evidence(
            kind="schema.asset_profile",
            owner="sqlite",
            operation_id="op-1",
            payload={
                "tables": [
                    {
                        "name": "customers",
                        "columns": [{"name": "id"}, {"name": "name"}],
                    }
                ],
                "metadata": {"scope": "asset"},
            },
        )
    )

    context = await DbSessionContextBuilder(runtime).build(
        DbRequest("What columns do those tables have?", session_id="s1"),
        conversation_messages=[
            {"role": "user", "content": "What columns are in customers?"},
            {"role": "assistant", "content": "customers: id, name"},
        ],
    )

    assert context.session_id == "s1"
    assert context.referents.tables == ("customers",)
    assert {"id", "name"} <= set(context.referents.columns)
    assert context.referents.operations == ("op-1",)
    assert (
        "runtime.evidence" in context.diagnostics["referent_sources"]["tables"].values()
    )
    assert "conversation_history" in context.diagnostics["sources"]


async def test_db_agent_history_session_id_flows_and_turn_is_appended():
    history = ConversationHistory(session_id="history-session", workspace="db-test")
    runtime = SpyRuntime()
    agent = DbAgent(runtime=runtime, name="db-history-test")

    result = await agent.run_detailed("Hello there", history=history)

    assert result.request.session_id == "history-session"
    assert runtime.run_requests[0].session_id == "history-session"
    assert history.turn_count == 1
    assert history.messages[-2:] == [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "spy answer"},
    ]


async def test_db_agent_explicit_session_id_overrides_history_for_routing():
    history = ConversationHistory(session_id="history-session", workspace="db-test")
    runtime = SpyRuntime()
    agent = DbAgent(runtime=runtime, name="db-history-test")

    result = await agent.run_detailed(
        "Hello there",
        history=history,
        session_id="explicit-session",
    )

    assert result.request.session_id == "explicit-session"
    assert runtime.run_requests[0].session_id == "explicit-session"
    assert history.turn_count == 1


async def test_db_agent_session_context_payload_compacts_history_and_operations():
    history = ConversationHistory(session_id="compact-session", workspace="db-test")
    await history.add_turn(
        "prior user transcript text that should not be transported",
        "The useful table is `agent_profiles`.",
    )
    runtime = SpyRuntime()
    await runtime.store.save_operation(
        Operation(
            id="op-compact",
            operation_type="schema.query",
            status=OperationStatus.SUCCEEDED,
            request={
                "prompt": "prior operation prompt that should not be transported",
                "session_id": "compact-session",
            },
            metadata={"session_id": "compact-session"},
        )
    )
    agent = DbAgent(runtime=runtime, name="db-history-test")

    result = await agent.run_detailed(
        "What columns do those tables have?", history=history
    )

    payload = result.request.session_context
    assert payload is not None
    dumped = json.dumps(payload)
    assert "conversation_messages" not in payload
    assert "current_prompt" not in payload
    assert "prior user transcript text" not in dumped
    assert "prior operation prompt" not in dumped
    assert payload["recent_operations"][0]["operation_id"] == "op-compact"
    assert "prompt" not in payload["recent_operations"][0]
    assert "prompt_fingerprint" in payload["recent_operations"][0]


async def test_from_db_stateful_true_creates_default_history(tmp_path):
    db_path = tmp_path / "stateful.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path), stateful=True, name="stateful-db-test")

    try:
        result = await agent.run_detailed("How many agent profiles are there?")
    finally:
        await agent.stop()

    assert result.request.session_id is not None
    assert agent._default_history is not None
    assert agent._default_history.turn_count == 1


async def test_stateless_default_does_not_create_session_history(tmp_path):
    db_path = tmp_path / "stateless.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path))

    try:
        result = await agent.run_detailed("How many agent profiles are there?")
    finally:
        await agent.stop()

    assert result.request.session_id is None
    assert agent._default_history is None


async def test_stateful_schema_followup_uses_structured_table_referents(tmp_path):
    db_path = tmp_path / "schema-followup.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path), stateful=True, name="schema-followup")

    try:
        first = await agent.run_detailed(
            "What tables store agent info?",
            mode="schema.query",
        )
        second = await agent.run_detailed(
            "What columns do those tables have?",
            mode="schema.query",
        )
    finally:
        await agent.stop()

    assert first.status is OperationStatus.SUCCEEDED
    assert second.status is OperationStatus.SUCCEEDED
    assert "agent_profiles" in (second.answer or "")
    assert "agent_runs" in (second.answer or "")
    assert "billing_events" not in (second.answer or "")
    assert second.request.session_context is not None
    json.dumps(second.request.session_context)
    assert "conversation_messages" not in second.request.session_context
    assert "current_prompt" not in second.request.session_context
    assert "What tables store agent info?" not in json.dumps(
        second.request.session_context
    )
    context = db_session_context_from_request(second.request)
    assert context is not None
    assert "agent_profiles" in context.referents.tables
    assert "runtime.evidence" in (
        context.diagnostics["referent_sources"]["tables"].values()
    )
    diagnostic_context = second.diagnostics.get("session_context")
    assert isinstance(diagnostic_context, dict)
    dumped_diagnostics = json.dumps(diagnostic_context)
    assert "conversation_messages" not in diagnostic_context
    assert "current_prompt" not in diagnostic_context
    assert "What tables store agent info?" not in dumped_diagnostics


async def test_stateful_schema_followup_does_not_depend_on_key_phrases(tmp_path):
    db_path = tmp_path / "schema-followup-no-key-phrases.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path), stateful=True, name="schema-followup")

    try:
        await agent.run_detailed(
            "What tables store agent info?",
            mode="schema.query",
        )
        result = await agent.run_detailed(
            "Please detail them.",
            mode="schema.query",
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert "agent_profiles" in (result.answer or "")
    assert "agent_runs" in (result.answer or "")
    assert "billing_events" not in (result.answer or "")


async def test_session_referents_are_isolated_by_session(tmp_path):
    db_path = tmp_path / "session-isolation.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path), name="session-isolation")
    history_a = ConversationHistory(session_id="session-a", workspace="db-test")
    history_b = ConversationHistory(session_id="session-b", workspace="db-test")

    try:
        await agent.run_detailed(
            "What columns are in agent_profiles?",
            history=history_a,
            mode="schema.query",
        )
        await agent.run_detailed(
            "What columns are in billing_events?",
            history=history_b,
            mode="schema.query",
        )
        result = await agent.run_detailed(
            "What columns do those tables have?",
            history=history_a,
            mode="schema.query",
        )
    finally:
        await agent.stop()

    assert "agent_profiles" in (result.answer or "")
    assert "billing_events" not in (result.answer or "")


async def test_explicit_source_scope_overrides_session_referents(tmp_path):
    db_path = tmp_path / "source-scope-override.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path), stateful=True, name="source-scope")

    try:
        await agent.run_detailed(
            "What columns are in agent_profiles?",
            mode="schema.query",
        )
        result = await agent.run_detailed(
            "What columns do those tables have?",
            mode="schema.query",
            source_scope=("billing_events",),
        )
    finally:
        await agent.stop()

    assert "billing_events" in (result.answer or "")
    assert "agent_profiles" not in (result.answer or "")


async def test_qualified_source_scope_overrides_session_referents(tmp_path):
    db_path = tmp_path / "qualified-source-scope.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path), stateful=True, name="qualified-source")

    try:
        await agent.run_detailed(
            "What columns are in agent_profiles?",
            mode="schema.query",
        )
        result = await agent.run_detailed(
            "What columns do those tables have?",
            mode="schema.query",
            source_scope=("warehouse.billing_events",),
        )
    finally:
        await agent.stop()

    assert "billing_events" in (result.answer or "")
    assert "amount_cents" in (result.answer or "")
    assert "agent_profiles" not in (result.answer or "")
