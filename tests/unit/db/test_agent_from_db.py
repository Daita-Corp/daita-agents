import asyncio
import json
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from daita.agents.agent import Agent
from daita.db import (
    DbAgent,
    DbLLMConfig,
    DbMemoryConfig,
    DbRequest,
    DbRuntime,
    DbRuntimeOptions,
    DbSourceOptions,
)
from daita.db.context_projection import (
    ProjectionContext,
    ProjectionMode,
    project_runtime_stream_event,
)
from daita.db.models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
)
from daita.db.runtime.extensions import HostedInAppMonitorDeliveryPlugin
from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.catalog.persistence import catalog_profile_key
from daita.plugins import PluginKind, PluginManifest
from daita.plugins.base_db import BaseDatabasePlugin
from daita.plugins.lineage import LineagePlugin
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.memory.memory_plugin import MemoryPlugin
from daita.plugins.postgresql import PostgreSQLPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import (
    AccessMode,
    HostRuntimeContext,
    InMemoryRuntimeStore,
    OperationStatus,
    RuntimeEvent,
    RuntimeEventType,
    SQLiteRuntimeStore,
    Task,
    TaskStatus,
    host_runtime_context,
)
from daita.skills import Skill, SkillRuntimeEffects

import pytest


class FailingRuntimeDbPlugin(BaseDatabasePlugin):
    manifest = PluginManifest(
        id="failing_db",
        display_name="Failing DB",
        version="1.0.0",
        kind=PluginKind.CONNECTOR,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        super().__init__()
        self.setup_called = False
        self.disconnect_called = False

    async def setup(self, context):
        self.setup_called = True
        raise RuntimeError("connection refused")

    async def connect(self):
        pass

    async def disconnect(self):
        self.disconnect_called = True


async def _seed_sqlite(path):
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total REAL NOT NULL
        );
        INSERT INTO customers (name) VALUES ('Ada'), ('Linus');
        INSERT INTO orders (customer_id, total) VALUES (1, 10.0), (2, 20.0);
        """)
    await plugin.disconnect()


async def _seed_sqlite_with_cents(path):
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            total_cents INTEGER NOT NULL
        );
        INSERT INTO orders (total_cents) VALUES (1234);
        """)
    await plugin.disconnect()


def _memory_backend(tmp_path):
    return LocalMemoryBackend(
        workspace="agent-from-db-memory",
        agent_id="agent-from-db-memory-test",
        scope="project",
        base_dir=tmp_path,
        embedder=MockEmbeddingProvider(dim=8),
    )


def _write_catalog_snapshot(profile_key, schema, *, last_seen=None):
    catalog_dir = Path(".daita")
    catalog_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(schema)
    now = datetime.now(timezone.utc).isoformat()
    seen_at = last_seen or now
    payload.setdefault("first_seen", seen_at)
    payload.setdefault("last_seen", seen_at)
    payload.setdefault("profile_key", profile_key)
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("profile_key", profile_key)
    payload["metadata"] = metadata
    (catalog_dir / "catalog.json").write_text(
        json.dumps({f"from_db:{profile_key}": payload}, indent=2)
    )


def _stream_result(
    operation_id,
    prompt,
    *,
    answer="streamed answer",
    status=OperationStatus.SUCCEEDED,
    diagnostics=None,
):
    return DbOperationResult(
        operation_id=operation_id,
        request=DbRequest(prompt),
        intent=DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        contract=DbOperationContract(
            operation_type="db.run",
            access=AccessMode.READ,
        ),
        status=status,
        answer=answer,
        diagnostics=dict(diagnostics or {}),
    )


async def test_agent_from_db_returns_db_agent_backed_by_db_runtime(tmp_path):
    db_path = tmp_path / "phase10.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path), name="phase-10-test")

    try:
        assert isinstance(agent, DbAgent)
        assert isinstance(agent.runtime, DbRuntime)
        assert agent.name == "phase-10-test"
        assert not isinstance(agent, Agent)
        assert not hasattr(agent, "_db_plugin")
        assert not hasattr(agent, "tool_names")
    finally:
        await agent.stop()


async def test_db_agent_streams_concurrent_runs_without_cross_operation_events():
    runtime = DbRuntime()
    agent = DbAgent(runtime=runtime)
    both_created = asyncio.Event()
    created: dict[str, str] = {}

    async def fake_run(prompt, *, history, kwargs, operation_id=None):
        assert operation_id is not None
        await runtime.kernel.create_operation(
            operation_id=operation_id,
            operation_type="db.run",
            request={"prompt": prompt},
            evaluate_governance=False,
        )
        created[prompt] = operation_id
        if len(created) == 2:
            both_created.set()
        await both_created.wait()
        await runtime.kernel.append_event(
            RuntimeEventType.DIAGNOSTIC,
            operation_id=operation_id,
            message=f"diagnostic for {prompt}",
            payload={"diagnostic": f"run.{prompt}"},
        )
        await runtime.kernel.complete_operation(operation_id)
        return _stream_result(operation_id, prompt, answer=f"answer {prompt}")

    agent._run_detailed = fake_run

    async def collect(prompt):
        return [event async for event in agent.stream(prompt)]

    first, second = await asyncio.gather(collect("first"), collect("second"))

    assert created["first"] != created["second"]
    assert {event.operation_id for event in first} == {created["first"]}
    assert {event.operation_id for event in second} == {created["second"]}
    assert first[0].type is RuntimeEventType.OPERATION_CREATED
    assert second[0].type is RuntimeEventType.OPERATION_CREATED
    for events, answer in ((first, "answer first"), (second, "answer second")):
        terminal = [event for event in events if event.payload.get("terminal")]
        assert len(terminal) == 1
        assert terminal[0].payload["answer"] == answer


async def test_db_agent_stream_uses_one_durable_terminal_event():
    runtime = DbRuntime(runtime_id="db-terminal-stream")
    agent = DbAgent(runtime=runtime)

    events = [event async for event in agent.stream("terminal contract")]
    operation_id = events[0].operation_id
    persisted = await runtime.store.list_events(operation_id)
    public_terminals = [
        event for event in events if event.payload.get("terminal") is True
    ]
    durable_terminals = [
        event for event in persisted if event.payload.get("terminal") is True
    ]

    assert len(public_terminals) == 1
    assert len(durable_terminals) == 1
    assert events[-1] == public_terminals[0]
    assert persisted[-1] == durable_terminals[0]
    assert public_terminals[0].payload["result_summary"]["status"] == "blocked"
    assert (
        public_terminals[0].payload["answer"].startswith("DB LLM service is required")
    )
    assert not any(
        event.payload.get("status") in {"succeeded", "failed", "cancelled", "blocked"}
        for event in events[:-1]
    )


async def test_db_runtime_run_accepts_internal_operation_id_without_changing_default():
    runtime = DbRuntime()

    try:
        selected = await runtime.run("selected", operation_id="db-op-selected")
        generated = await runtime.run("generated")
    finally:
        await runtime.teardown()

    assert selected.operation_id == "db-op-selected"
    assert generated.operation_id.startswith("db-op-")
    assert generated.operation_id != selected.operation_id


async def test_db_agent_stream_reports_bounded_slow_consumer_drops():
    runtime = DbRuntime()
    agent = DbAgent(runtime=runtime)
    published = asyncio.Event()

    async def fake_run(prompt, *, history, kwargs, operation_id=None):
        assert operation_id is not None
        await runtime.kernel.create_operation(
            operation_id=operation_id,
            operation_type="db.run",
            request={"prompt": prompt},
            evaluate_governance=False,
        )
        for index in range(300):
            await runtime.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation_id,
                message=f"diagnostic {index}",
                payload={"diagnostic": "test.progress", "index": index},
            )
        await runtime.kernel.complete_operation(operation_id)
        published.set()
        return _stream_result(operation_id, prompt)

    agent._run_detailed = fake_run

    stream = agent.stream("slow")
    first = await anext(stream)
    await published.wait()
    events = [first, *[event async for event in stream]]
    dropped = [
        event
        for event in events
        if event.payload.get("diagnostic") == "runtime_stream.events_dropped"
    ]

    assert len(dropped) == 1
    assert dropped[0].payload["dropped_event_count"] > 0
    assert len([event for event in events if event.payload.get("terminal")]) == 1
    assert runtime.kernel.event_broker.subscriber_count == 0


def test_db_runtime_stream_projection_removes_sensitive_event_content():
    raw = RuntimeEvent(
        type=RuntimeEventType.ERROR,
        operation_id="op-secret",
        message="password=super-secret raw failure customer@example.com",
        policy_id="internal.policy.details",
        payload={
            "prompt": "show secret customers",
            "sql": "SELECT * FROM customers WHERE token = 'sql-secret'",
            "parameters": {"token": "parameter-secret"},
            "rows": [{"password": "row-secret"}],
            "credentials": {"password": "credential-secret"},
            "error": {"message": "exception-secret"},
            "governance": {"reason": "policy-secret"},
            "planner": {"diagnostic": "planner-secret"},
            "executor": {"diagnostic": "executor-secret"},
        },
    )

    projected = project_runtime_stream_event(
        raw,
        ProjectionContext(mode=ProjectionMode.PUBLIC_RESULT),
    )
    encoded = json.dumps(projected.to_dict())

    assert projected.message == "Runtime error."
    assert projected.policy_id is None
    assert projected.payload == {"projected": True}
    for secret in (
        "super-secret",
        "customer@example.com",
        "SELECT *",
        "sql-secret",
        "parameter-secret",
        "row-secret",
        "credential-secret",
        "exception-secret",
        "policy-secret",
        "planner-secret",
        "executor-secret",
    ):
        assert secret not in encoded


async def test_db_agent_stream_close_awaits_run_and_preserves_resumable_state():
    runtime = DbRuntime()
    agent = DbAgent(runtime=runtime)
    run_cancelled = asyncio.Event()

    async def fake_run(prompt, *, history, kwargs, operation_id=None):
        assert operation_id is not None
        operation = await runtime.kernel.create_operation(
            operation_id=operation_id,
            operation_type="db.run",
            request={"prompt": prompt},
            evaluate_governance=False,
        )
        await runtime.store.save_task(
            Task(
                id=f"task-{operation_id}",
                operation_id=operation.id,
                capability_id="db.test.resume",
                executor_id="db.test.resume.executor",
                input={"sql": "SELECT secret"},
                status=TaskStatus.PENDING,
            )
        )
        try:
            await asyncio.Event().wait()
        finally:
            run_cancelled.set()

    agent._run_detailed = fake_run
    stream = agent.stream("cancel me")

    first = await anext(stream)
    await stream.aclose()

    operation = await runtime.store.load_operation(first.operation_id)
    tasks = await runtime.store.list_tasks(first.operation_id)
    assert run_cancelled.is_set()
    assert runtime.kernel.event_broker.subscriber_count == 0
    assert operation.status is OperationStatus.BLOCKED
    assert tasks[0].status is TaskStatus.PENDING


async def test_agent_from_db_defaults_to_in_memory_runtime_store(tmp_path):
    db_path = tmp_path / "default_runtime_store.sqlite"
    runtime_path = tmp_path / "runtime.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        assert isinstance(agent.runtime.store, InMemoryRuntimeStore)
        assert not runtime_path.exists()
    finally:
        await agent.stop()


async def test_agent_from_db_runtime_options_sqlite_store(tmp_path):
    db_path = tmp_path / "runtime_options_source.sqlite"
    runtime_path = tmp_path / "runtime_options.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(
        str(db_path),
        runtime=DbRuntimeOptions(store="sqlite", store_path=runtime_path),
    )

    try:
        assert isinstance(agent.runtime.store, SQLiteRuntimeStore)
        assert agent.runtime.store.path == runtime_path
        assert runtime_path.exists()
    finally:
        await agent.stop()


async def test_agent_from_db_runtime_options_existing_store_instance(tmp_path):
    db_path = tmp_path / "existing_store_source.sqlite"
    runtime_path = tmp_path / "existing_runtime_store.sqlite"
    runtime_store = SQLiteRuntimeStore(runtime_path)
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(
        str(db_path),
        runtime=DbRuntimeOptions(store=runtime_store),
    )

    try:
        assert agent.runtime.store is runtime_store
    finally:
        await agent.stop()


async def test_agent_from_db_registers_skills_before_runtime_setup(tmp_path):
    db_path = tmp_path / "skills.sqlite"
    await _seed_sqlite(db_path)
    skill = Skill(
        name="finance",
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_finance",
            requested_capabilities=("catalog.schema.search",),
            contract_metadata={"planning_hints": {"finance": True}},
        ),
    )

    agent = await Agent.from_db(str(db_path), skills=[skill])

    try:
        inspection = await agent.describe()
        db_request = DbRequest("How many orders are there?")
        intent = agent.runtime.classify_request(db_request)
        skill_resolution = agent.runtime._resolve_skills(db_request)
        contract = agent.runtime.build_contract(
            db_request,
            intent,
            skill_resolution=skill_resolution,
        )
    finally:
        await agent.stop()

    assert "skill_finance" in inspection.plugin_ids
    assert "catalog.schema.search" in contract.required_capabilities


async def test_agent_from_db_consumes_host_runtime_extensions_before_setup(tmp_path):
    db_path = tmp_path / "hosted.sqlite"
    await _seed_sqlite(db_path)
    calls = []

    async def hosted_notification_service(payload):
        calls.append(payload)
        return {"notification_id": "hosted-1"}

    host_context = HostRuntimeContext(
        surface="web_app",
        delivery_defaults=("in_app",),
        services={
            "hosted_in_app_notification_service": hosted_notification_service,
        },
        runtime_extensions=(HostedInAppMonitorDeliveryPlugin(),),
    )

    with host_runtime_context(host_context):
        agent = await Agent.from_db(str(db_path), name="hosted-runtime-test")

    try:
        inspection = await agent.describe()
        evidence = await agent.runtime.execute_capability(
            "monitor.delivery.in_app",
            owner="hosted_monitor_delivery",
            operation_type="monitor.delivery",
            input={
                "delivery_kind": "in_app",
                "target": {"type": "requesting_user"},
                "payload_source": {"type": "monitor.report"},
            },
        )
    finally:
        await agent.stop()

    assert "hosted_monitor_delivery" in inspection.plugin_ids
    assert "hosted_monitor_delivery:monitor.delivery.in_app" in (
        inspection.capability_ids
    )
    assert calls[0]["delivery_kind"] == "in_app"
    assert calls[0]["target"] == {"type": "requesting_user"}
    assert calls[0]["payload_source"] == {"type": "monitor.report"}
    assert evidence[0].payload["result"] == {"notification_id": "hosted-1"}


async def test_agent_from_db_skill_effects_apply_during_run_before_contract_building(
    tmp_path,
):
    db_path = tmp_path / "skills_run.sqlite"
    await _seed_sqlite(db_path)
    skill = Skill(
        name="quality_gate",
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_quality_gate",
            requested_capabilities=("quality.profile",),
            required_evidence=("quality.profile",),
        ),
    )

    agent = await Agent.from_db(str(db_path), skills=[skill])

    try:
        result = await agent.run_detailed("Hello there")
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert result.contract.metadata["missing_capabilities"] == ["quality.profile"]
    assert snapshot is not None
    assert snapshot.tasks == ()
    diagnostics = [event.payload.get("diagnostic") for event in snapshot.events]
    assert "skill.selected" in diagnostics
    assert "skill.contract_modified" in diagnostics


def test_db_runtime_facade_does_not_export_generic_agent_orchestrator():
    import daita.db as db

    wrapper_names = {
        "DbRunOrchestrator",
        "DbRunContract",
        "build_db_run_context",
        "DBAudit",
        "DBContext",
        "attach_db_context",
        "attach_db_describe",
        "build_prompt",
        "build_prompt_result",
        "describe_db_agent",
        "infer_domain",
        "create_db_query_tools",
        "create_db_memory_tools",
        "register_analyst_tools",
        "DbRunState",
        "get_db_run_state",
        "set_db_run_state",
        "attach_db_completeness",
        "evaluate_db_final_answer_readiness",
        "summarize_db_completeness",
        "ToolResultPolicy",
        "compact_tool_result_for_context",
    }
    assert wrapper_names.isdisjoint(db.__all__)
    assert not any(hasattr(db, name) for name in wrapper_names)
    assert importlib.util.find_spec("daita.db.agent_context") is None
    assert importlib.util.find_spec("daita.db.analyst_tools") is None
    assert importlib.util.find_spec("daita.db.describe") is None
    assert importlib.util.find_spec("daita.db.prompt") is None


async def test_agent_from_db_wraps_runtime_setup_failure():
    from daita.core.exceptions import AgentError

    plugin = FailingRuntimeDbPlugin()

    with pytest.raises(AgentError, match="Failed to initialize database runtime"):
        await Agent.from_db(plugin)

    assert plugin.setup_called is True
    assert plugin.disconnect_called is False


async def test_agent_from_db_does_not_expose_legacy_db_facade_tools(tmp_path):
    db_path = tmp_path / "phase13_no_legacy_facade_tools.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert not hasattr(agent, "local_tool_catalog")
    assert not hasattr(agent, "tool_names")
    assert "sqlite:db.sql.execute_read" in inspection.capability_ids
    assert "db_query" not in inspection.tool_view_names
    assert "db_compile_and_query" not in inspection.tool_view_names
    assert "db_count" not in inspection.tool_view_names


async def test_agent_from_db_describe_is_runtime_inspection_not_legacy_db_context(
    tmp_path,
):
    db_path = tmp_path / "phase13_runtime_describe.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(
        str(db_path),
        name="Warehouse",
        mode="governed",
        source_options=DbSourceOptions(
            query_default_limit=25,
            query_max_rows=100,
            query_max_chars=1000,
            query_timeout=12,
            allowed_tables=("orders",),
            blocked_tables=("payments",),
            blocked_columns=("email",),
        ),
    )

    try:
        inspection = await agent.describe()
        sqlite_plugin = agent.runtime.registry.get_plugin("sqlite")
    finally:
        await agent.stop()

    assert inspection.runtime_kind == "db"
    assert inspection.source_type == "str"
    assert inspection.profile == "governed"
    assert inspection.plugin_ids[:2] == ("catalog", "sqlite")
    assert "sqlite:db.sql.execute_read" in inspection.capability_ids
    assert inspection.limits == {
        "max_rows": 100,
        "timeout_seconds": 12,
        "max_tasks": 20,
        "max_evidence_items": 50,
    }
    assert inspection.metadata["from_db_options"]["source_options"] == {
        "include_sample_values": True,
        "redact_pii_columns": True,
        "read_only": True,
        "query_default_limit": 25,
        "query_max_rows": 100,
        "query_max_chars": 1000,
        "query_timeout": 12,
        "allowed_tables": ["orders"],
        "blocked_tables": ["payments"],
        "blocked_columns": ["email"],
    }
    assert inspection.to_dict()["limits"]["max_rows"] == 100
    assert sqlite_plugin.read_only is True
    assert sqlite_plugin.query_default_limit == 25
    assert sqlite_plugin.query_max_rows == 100
    assert sqlite_plugin.query_max_chars == 1000
    assert sqlite_plugin.query_timeout == 12
    assert sqlite_plugin.allowed_tables == {"orders"}
    assert sqlite_plugin.blocked_tables == {"payments"}
    assert sqlite_plugin.blocked_columns == {"email"}
    assert not hasattr(agent, "db")
    assert not hasattr(agent, "_db_summary")
    assert not hasattr(agent, "_watches")


async def test_agent_from_db_resolves_postgresql_sources_to_db_runtime(monkeypatch):
    async def fake_connect(self):
        self._pool = object()

    async def fake_disconnect(self):
        self._pool = None

    monkeypatch.setattr(PostgreSQLPlugin, "connect", fake_connect)
    monkeypatch.setattr(PostgreSQLPlugin, "disconnect", fake_disconnect)

    agent = await Agent.from_db("postgresql://localhost/testdb")

    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert isinstance(agent, DbAgent)
    assert "postgresql" in inspection.plugin_ids
    assert "postgresql:db.sql.execute_read" in inspection.capability_ids


async def test_agent_from_db_rejects_unconverted_source_strings():
    with pytest.raises(ValueError, match="Unsupported scheme"):
        await Agent.from_db("mysql://localhost/testdb")

    with pytest.raises(ValueError, match="Unsupported source"):
        await Agent.from_db("warehouse")


async def test_agent_from_db_memory_config_registers_initialized_memory_plugin(
    tmp_path,
):
    db_path = tmp_path / "phase13_memory_config.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(
        str(db_path),
        name="memory-config-test",
        memory=DbMemoryConfig(enabled=True),
    )

    try:
        inspection = await agent.describe()
        memory = agent.runtime.registry.get_plugin("memory")
    finally:
        await agent.stop()

    assert isinstance(memory, MemoryPlugin)
    assert memory.backend is not None
    assert memory._agent_id == "memory-config-test"
    assert "memory" in inspection.plugin_ids
    assert "memory:memory.semantic.write" in inspection.capability_ids
    assert "memory:memory.context" in inspection.context_provider_ids


async def test_agent_from_db_memory_config_uses_runtime_setup_not_initialize(tmp_path):
    db_path = tmp_path / "phase13_memory_setup.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(
        str(db_path),
        name="memory-setup-test",
        memory=DbMemoryConfig(enabled=True),
    )

    try:
        memory = agent.runtime.registry.get_plugin("memory")
    finally:
        await agent.stop()

    assert isinstance(memory, MemoryPlugin)
    assert memory.backend is not None
    assert memory._agent_id == "memory-setup-test"
    assert not hasattr(MemoryPlugin, "initialize")


async def test_agent_from_db_default_initializes_source_scoped_memory(tmp_path):
    db_path = tmp_path / "phase1_memory_default.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        inspection = await agent.describe()
        memory = agent.runtime.registry.get_plugin("memory")
    finally:
        await agent.stop()

    memory_config = inspection.metadata["from_db_options"]["memory"]
    assert memory_config["enabled"] is True
    assert memory_config["recall"] == "auto"
    assert memory_config["learning"] == "safe"
    assert memory_config["limit"] == 3
    assert memory_config["char_budget"] == 800
    assert memory_config["backend"] == "local"
    assert memory_config["retrieval_mode"] == "structured"
    assert memory_config["embedding_available"] is False
    assert memory_config["structured_index"] == "sqlite_fts5"
    assert memory_config["workspace_scope"] == "source"
    assert memory_config["source_identity"].startswith("sqlite:from_db:")
    assert isinstance(memory, MemoryPlugin)
    assert memory.workspace == memory_config["source_identity"]
    assert memory._embedder is None
    assert "memory" in inspection.plugin_ids
    assert "memory:memory.semantic.write" in inspection.capability_ids


async def test_agent_from_db_default_structured_db_memory_needs_no_embedder(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    db_path = tmp_path / "phase31_structured_memory.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        memory = agent.runtime.registry.get_plugin("memory")
        written = await agent.runtime.execute_capability(
            "memory.semantic.write",
            owner="memory",
            operation_type="memory.update",
            input={
                "db_memory_payload": {
                    "kind": "metric_definition",
                    "key": "metric:revenue",
                    "text": "Revenue excludes refunded orders.",
                    "importance": 0.9,
                    "metadata": {
                        "active": True,
                        "workspace_scope": "source",
                        "confidence": 0.95,
                    },
                },
                "db_memory_prompt": "Revenue excludes refunded orders.",
            },
        )
        recalled = await agent.runtime.execute_capability(
            "memory.semantic.recall",
            owner="memory",
            operation_type="memory.recall",
            input={
                "query": "How should revenue be calculated?",
                "category": "db_semantics",
                "limit": 3,
                "score_threshold": 0.0,
            },
        )
    finally:
        await agent.stop()

    assert memory._embedder is None
    assert getattr(memory.backend, "embedding_available") is False
    assert written[0].payload["success"] is True
    assert written[0].payload["stored"]["structured"] is True
    assert recalled[0].payload["diagnostics"]["retrieval_mode"] == "structured"
    assert recalled[0].payload["diagnostics"]["embedding_available"] is False
    assert recalled[0].payload["diagnostics"]["structured_candidate_count"] == 1
    result = recalled[0].payload["results"][0]
    assert result["metadata"]["db_memory"]["key"] == "metric:revenue"
    assert result["score_breakdown"]["key_overlap"] > 0


async def test_agent_from_db_memory_false_opts_out(tmp_path):
    db_path = tmp_path / "phase1_memory_opt_out.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path), memory=DbMemoryConfig(enabled=False))

    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    memory_config = inspection.metadata["from_db_options"]["memory"]
    assert memory_config["enabled"] is False
    assert memory_config["recall"] == "off"
    assert memory_config["learning"] == "off"
    assert "memory" not in inspection.plugin_ids
    assert "memory:memory.semantic.write" not in inspection.capability_ids


async def test_agent_from_db_embedding_modes_require_explicit_embedder(tmp_path):
    db_path = tmp_path / "phase31_embedding_modes.sqlite"
    await _seed_sqlite(db_path)

    with pytest.raises(ValueError, match="requires an explicit embedding"):
        await Agent.from_db(
            str(db_path), memory=DbMemoryConfig(retrieval_mode="hybrid")
        )

    with pytest.raises(ValueError, match="requires an explicit embedding"):
        await Agent.from_db(
            str(db_path), memory=DbMemoryConfig(retrieval_mode="embedding")
        )


async def test_agent_from_db_rejects_boolean_memory_option(tmp_path):
    db_path = tmp_path / "phase1_memory_true_rejected.sqlite"
    await _seed_sqlite(db_path)

    with pytest.raises(
        TypeError,
        match="memory must be a DbMemoryConfig instance",
    ):
        await Agent.from_db(str(db_path), memory=True)


async def test_agent_from_db_rejects_non_config_memory_option(tmp_path):
    db_path = tmp_path / "phase1_memory_plugin_rejected.sqlite"
    await _seed_sqlite(db_path)

    with pytest.raises(
        TypeError,
        match="memory must be a DbMemoryConfig instance",
    ):
        await Agent.from_db(str(db_path), memory=MemoryPlugin())


async def test_agent_from_db_memory_config_can_provide_backend(tmp_path):
    db_path = tmp_path / "phase1_memory_backend.sqlite"
    await _seed_sqlite(db_path)
    backend = MagicMock()

    agent = await Agent.from_db(str(db_path), memory=DbMemoryConfig(backend=backend))

    try:
        memory = agent.runtime.registry.get_plugin("memory")
    finally:
        await agent.stop()

    assert memory.backend is backend


async def test_agent_from_db_calibrate_memory_stores_structured_unit_conventions(
    tmp_path,
):
    db_path = tmp_path / "phase13_memory_calibration.sqlite"
    await _seed_sqlite_with_cents(db_path)
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(return_value={"status": "success"})

    agent = await Agent.from_db(
        str(db_path),
        memory=DbMemoryConfig(backend=backend, calibrate=True),
    )

    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    categories = [call.kwargs["category"] for call in backend.remember.await_args_list]
    contents = [call.args[0] for call in backend.remember.await_args_list]

    assert "memory" in inspection.plugin_ids
    assert "db_semantics" in categories
    assert "db_cache_marker" in categories
    assert any("orders.total_cents is stored as cents" in item for item in contents)
    assert all("numeric_column_units" not in item for item in contents)


async def test_agent_from_db_calibrate_memory_skips_when_exact_marker_exists(
    tmp_path,
):
    db_path = tmp_path / "phase13_memory_calibration_marker.sqlite"
    await _seed_sqlite_with_cents(db_path)
    source = SQLitePlugin(path=str(db_path))
    original_schema_inspect = source._execute_schema_inspect
    source._execute_schema_inspect = AsyncMock(side_effect=original_schema_inspect)
    backend = MagicMock()
    backend.list_by_category = AsyncMock(
        return_value=[
            {
                "content": (
                    "DB exact cache marker: " f"numeric_unit_calibration:{db_path}"
                )
            }
        ]
    )
    backend.remember = AsyncMock()

    agent = await Agent.from_db(
        source,
        memory=DbMemoryConfig(backend=backend, calibrate=True),
    )

    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert "memory" in inspection.plugin_ids
    source._execute_schema_inspect.assert_not_awaited()
    backend.remember.assert_not_awaited()


async def test_agent_from_db_does_not_register_generic_memory_tools(tmp_path):
    db_path = tmp_path / "phase13_no_generic_memory_tools.sqlite"
    await _seed_sqlite(db_path)
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(return_value={"status": "success"})

    agent = await Agent.from_db(str(db_path), memory=DbMemoryConfig(backend=backend))

    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert not hasattr(agent, "local_tool_catalog")
    assert "remember" not in inspection.tool_view_names
    assert "update_memory" not in inspection.tool_view_names
    assert "memory:memory.semantic.write" in inspection.capability_ids


async def test_agent_from_db_mode_defaults_are_owned_by_db_runtime_factory(tmp_path):
    db_path = tmp_path / "phase13_mode.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path), mode="data_team")

    try:
        inspection = await agent.describe()
        sqlite_plugin = agent.runtime.registry.get_plugin("sqlite")
    finally:
        await agent.stop()

    assert inspection.profile == "data_team"
    assert "data_quality" in inspection.plugin_ids
    assert "lineage" in inspection.plugin_ids
    assert "data_quality:quality.profile" in inspection.capability_ids
    assert sqlite_plugin.query_timeout == 30
    assert sqlite_plugin.query_max_rows == 200


async def test_agent_from_db_simple_mode_applies_connector_limits_without_services(
    tmp_path,
):
    db_path = tmp_path / "phase13_simple_mode.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path), mode="simple")

    try:
        inspection = await agent.describe()
        sqlite_plugin = agent.runtime.registry.get_plugin("sqlite")
    finally:
        await agent.stop()

    assert inspection.profile == "simple"
    assert "data_quality" not in inspection.plugin_ids
    assert "lineage" not in inspection.plugin_ids
    assert "memory" in inspection.plugin_ids
    assert sqlite_plugin.query_max_rows == 100
    assert sqlite_plugin.query_max_chars == 25000


async def test_agent_from_db_mode_quality_override_wins_on_new_path(tmp_path):
    db_path = tmp_path / "phase13_quality_override.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path), mode="data_team", quality=False)

    try:
        inspection = await agent.describe()
        sqlite_plugin = agent.runtime.registry.get_plugin("sqlite")
    finally:
        await agent.stop()

    assert inspection.profile == "data_team"
    assert "data_quality" not in inspection.plugin_ids
    assert "lineage" in inspection.plugin_ids
    assert sqlite_plugin.query_timeout == 30


async def test_agent_from_db_data_team_can_configure_memory_backend(tmp_path):
    db_path = tmp_path / "phase13_custom_memory_backend.sqlite"
    await _seed_sqlite(db_path)
    backend = _memory_backend(tmp_path)

    agent = await Agent.from_db(
        str(db_path),
        mode="data_team",
        memory=DbMemoryConfig(backend=backend),
    )

    try:
        inspection = await agent.describe()
        registered = agent.runtime.registry.get_plugin("memory")
    finally:
        await agent.stop()

    assert isinstance(registered, MemoryPlugin)
    assert registered.backend is backend
    assert "memory" in inspection.plugin_ids
    assert "memory:memory.semantic.write" in inspection.capability_ids
    assert "memory:memory.context" in inspection.context_provider_ids


async def test_agent_from_db_uses_typed_source_and_llm_configuration(tmp_path):
    db_path = tmp_path / "phase13_metadata.sqlite"
    await _seed_sqlite(db_path)
    llm = DbLLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        temperature=0.2,
    )

    agent = await Agent.from_db(
        str(db_path),
        name="metadata-agent",
        llm=llm,
        source_options=DbSourceOptions(
            schema="main",
            include_sample_values=False,
            redact_pii_columns=False,
            cache_ttl=3600,
        ),
    )

    try:
        metadata = agent.runtime.config.metadata["from_db_options"]
        service_config = agent.runtime.db_llm_service.config
        source = agent.runtime.registry.get_plugin("sqlite")
    finally:
        await agent.stop()

    assert metadata["llm"]["provider"] == "openai"
    assert metadata["llm"]["model"] == "gpt-4o-mini"
    assert metadata["llm"]["temperature"] == 0.2
    assert metadata["source_options"]["schema"] == "main"
    assert metadata["source_options"]["include_sample_values"] is False
    assert metadata["source_options"]["redact_pii_columns"] is False
    assert metadata["source_options"]["cache_ttl"] == 3600
    assert source.schema == "main"
    assert source.include_sample_values is False
    assert source.redact_pii_columns is False
    assert "sk-test" not in json.dumps(metadata)
    assert service_config is llm


async def test_agent_from_db_does_not_profile_schema_during_construction(tmp_path):
    db_path = tmp_path / "phase13_no_construction_profile.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    source._execute_schema_inspect = AsyncMock(
        side_effect=AssertionError("schema profiling should be operation-time work")
    )

    agent = await Agent.from_db(source)

    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert "sqlite" in inspection.plugin_ids
    source._execute_schema_inspect.assert_not_awaited()


async def test_agent_from_db_rejects_unknown_mode_on_new_path(tmp_path):
    db_path = tmp_path / "phase13_bad_mode.sqlite"
    await _seed_sqlite(db_path)

    with pytest.raises(ValueError, match="Unknown from_db mode"):
        await Agent.from_db(str(db_path), mode="nope")
