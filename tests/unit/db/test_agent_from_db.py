import json
import importlib.util
import pkgutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from daita.agents.agent import Agent
from daita.db.config.policies import SchemaPromptPolicy, ToolResultPolicy
from daita.db import DbAgent, DbRequest, DbRuntime, DbRuntimeOptions
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
    HostRuntimeContext,
    InMemoryRuntimeStore,
    OperationStatus,
    SQLiteRuntimeStore,
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


async def test_agent_from_db_reopens_persisted_runtime_operation_state(tmp_path):
    db_path = tmp_path / "reopen_source.sqlite"
    runtime_path = tmp_path / "reopen_runtime.sqlite"
    await _seed_sqlite(db_path)

    first = await Agent.from_db(
        str(db_path),
        runtime=DbRuntimeOptions(store="sqlite", store_path=runtime_path),
    )
    try:
        result = await first.run_detailed("How many customers are there?")
    finally:
        await first.stop()

    second = await Agent.from_db(
        str(db_path),
        runtime=DbRuntimeOptions(store="sqlite", store_path=runtime_path),
    )
    try:
        snapshot = await second.runtime.inspect_operation(result.operation_id)
    finally:
        await second.stop()

    assert snapshot is not None
    assert snapshot.operation.id == result.operation_id
    assert snapshot.operation.status is OperationStatus.SUCCEEDED
    assert snapshot.tasks
    assert snapshot.evidence


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


async def test_legacy_agents_db_public_import_routes_to_db_runtime(tmp_path):
    from daita.agents.db import from_db

    db_path = tmp_path / "phase13_public_import.sqlite"
    await _seed_sqlite(db_path)

    agent = await from_db(str(db_path), name="phase-13-public-import")

    try:
        assert isinstance(agent, DbAgent)
        assert isinstance(agent.runtime, DbRuntime)
        assert agent.name == "phase-13-public-import"
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert inspection.plugin_ids[:2] == ("catalog", "sqlite")


def test_legacy_agents_db_package_exposes_only_from_db_bridge():
    import daita.agents.db as legacy_db
    import daita.db as db

    assert legacy_db.from_db is db.from_db
    assert legacy_db.__all__ == ["from_db"]
    assert [module.name for module in pkgutil.iter_modules(legacy_db.__path__)] == []
    assert importlib.util.find_spec("daita.agents.db.builder") is None
    assert importlib.util.find_spec("daita.agents.db.runtime") is None
    assert importlib.util.find_spec("daita.agents.db.query") is None
    assert importlib.util.find_spec("daita.agents.db.config") is None


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


async def test_agent_from_db_executes_schema_query_with_typed_evidence(tmp_path):
    db_path = tmp_path / "phase10.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(f"sqlite://{db_path}")

    try:
        result = await agent.run_detailed(
            "What columns are in the customers table?",
            mode="schema.query",
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.diagnostics["verification"]["passed"] is True
    assert "customers: id, name" in (result.answer or "")
    assert {"schema.asset_profile", "schema.search_result"} <= {
        item.kind for item in result.evidence
    }


async def test_agent_from_db_passes_request_mode_to_runtime(tmp_path):
    db_path = tmp_path / "phase10.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        result = await agent.run_detailed("Customers", mode="schema.query")
    finally:
        await agent.stop()

    assert result.intent.kind.value == "schema.query"
    assert result.request.mode == "schema.query"
    assert result.status is OperationStatus.SUCCEEDED


async def test_agent_from_db_preserves_typed_request_fields_without_prompt_wrapping(
    tmp_path,
):
    db_path = tmp_path / "phase13_typed_request_fields.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        result = await agent.run_detailed(
            "Customers",
            mode="schema.query",
            user_id="user-1",
            session_id="session-1",
            requested_capabilities=("catalog.schema.search",),
            constraints={"audience": "analyst"},
            metadata={"trace_id": "trace-1"},
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.request.prompt == "Customers"
    assert not result.request.prompt.startswith("<db_runtime_context>")
    assert result.request.user_id == "user-1"
    assert result.request.session_id == "session-1"
    assert result.request.requested_capabilities == ("catalog.schema.search",)
    assert result.request.constraints == {"audience": "analyst"}
    assert result.request.metadata == {"trace_id": "trace-1"}


async def test_agent_from_db_executes_count_query_with_typed_evidence(tmp_path):
    db_path = tmp_path / "phase10.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        answer = await agent.run("How many customers are there?")
        result = await agent.run_detailed("How many customers are there?")
    finally:
        await agent.stop()

    assert answer == "The count is 2."
    assert result.status is OperationStatus.SUCCEEDED
    assert {
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "query.result",
    } <= {item.kind for item in result.evidence}
    assert result.diagnostics["verification"]["passed"] is True
    assert result.diagnostics["synthesis"]["cited_evidence_refs"]


async def test_agent_from_db_stream_yields_typed_runtime_result(tmp_path):
    db_path = tmp_path / "phase13_stream_facade.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        streamed = [
            item async for item in agent.stream("How many customers are there?")
        ]
    finally:
        await agent.stop()

    assert len(streamed) == 1
    result = streamed[0]
    assert result.status is OperationStatus.SUCCEEDED
    assert result.answer == "The count is 2."
    assert result.request.prompt == "How many customers are there?"
    assert len(agent.audit_log) == 1
    assert agent.audit_log[0]["prompt"] == "How many customers are there?"
    assert agent.audit_log[0]["status"] == "succeeded"
    assert agent.audit_log[0]["intent_kind"] == "data.query"
    assert not hasattr(agent, "_db_original_stream")
    assert not hasattr(agent, "_db_last_context_metadata")


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
        query_default_limit=25,
        query_max_rows=100,
        query_max_chars=1000,
        query_timeout=12,
        allowed_tables=["orders"],
        blocked_tables=["payments"],
        blocked_columns=["email"],
        prompt="Revenue is net of tax.",
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
    assert inspection.metadata["from_db_options"]["prompt"] == (
        "Revenue is net of tax."
    )
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


async def test_agent_from_db_records_runtime_operation_history(tmp_path):
    db_path = tmp_path / "phase13_operation_history.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        first = await agent.run_detailed("How many customers are there?")
        second = await agent.run_detailed("How many orders are there?")
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert [item.operation_id for item in agent.operations] == [
        first.operation_id,
        second.operation_id,
    ]
    assert inspection.operation_count == 2
    assert inspection.last_operation_id == second.operation_id
    assert len(agent.audit_log) == 2
    assert agent.audit_log[0]["prompt"] == "How many customers are there?"
    assert agent.audit_log[0]["status"] == "succeeded"
    assert agent.audit_log[0]["intent_kind"] == "data.query"
    assert agent.audit_log[0]["evidence_refs"]


async def test_agent_from_db_audit_log_redacts_query_rows(tmp_path):
    db_path = tmp_path / "phase13_audit_redaction.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))

    try:
        result = await agent.run_detailed(
            "Show customer names",
            sql="SELECT name FROM customers ORDER BY id",
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    serialized = json.dumps(agent.audit_log)
    assert "Ada" not in serialized
    assert "Linus" not in serialized
    query_summary = next(
        item
        for item in agent.audit_log[0]["evidence"]
        if item["kind"] == "query.result"
    )
    assert query_summary["row_count"] == 2
    assert query_summary["sql"] == "SELECT name FROM customers ORDER BY id LIMIT 50"
    assert "rows" in query_summary["payload_keys"]


async def test_agent_from_db_guardrails_are_enforced_by_runtime_connectors(tmp_path):
    db_path = tmp_path / "phase10.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path), blocked_columns=["name"])

    try:
        result = await agent.run_detailed(
            "Show customer names",
            sql="SELECT name FROM customers",
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.FAILED
    assert result.warnings == ("db_runtime_execution_failed",)
    assert result.diagnostics["error"]["type"] == "ValidationError"
    assert "blocked column" in result.diagnostics["error"]["message"]


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


async def test_agent_from_db_registers_db_adjacent_extension_plugins(tmp_path):
    db_path = tmp_path / "phase13_services.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(
        str(db_path),
        name="phase-13-services",
        lineage=LineagePlugin(),
        memory={"backend": _memory_backend(tmp_path)},
        quality=True,
    )

    try:
        inspection = await agent.describe()
        memory_result = await agent.run_detailed(
            "Remember that revenue excludes tax",
            mode="memory.update",
            metadata={"category": "db_semantic"},
        )
        quality_result = await agent.run_detailed(
            "Profile the orders table",
            mode="quality.check",
        )
    finally:
        await agent.stop()

    assert inspection.plugin_ids == (
        "catalog",
        "sqlite",
        "data_quality",
        "lineage",
        "memory",
    )
    assert "data_quality:quality.profile" in inspection.capability_ids
    assert "lineage:lineage.trace" in inspection.capability_ids
    assert "memory:memory.semantic.write" in inspection.capability_ids
    assert "memory:memory.context" in inspection.context_provider_ids

    assert memory_result.status is OperationStatus.SUCCEEDED
    assert any(item.kind == "memory.semantic.write" for item in memory_result.evidence)
    assert quality_result.status is OperationStatus.SUCCEEDED
    assert any(item.kind == "quality.profile" for item in quality_result.evidence)


async def test_agent_from_db_lineage_true_registers_runtime_plugin(tmp_path):
    db_path = tmp_path / "phase13_lineage_true.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path), lineage=True)

    try:
        inspection = await agent.describe()
        result = await agent.run_detailed(
            "Trace lineage for orders",
            mode="lineage.trace",
        )
    finally:
        await agent.stop()

    assert "lineage" in inspection.plugin_ids
    assert "lineage:lineage.trace" in inspection.capability_ids
    assert result.status is OperationStatus.SUCCEEDED
    assert any(item.kind == "lineage.trace" for item in result.evidence)
    assert not hasattr(agent, "_db_lineage")


async def test_agent_from_db_memory_config_registers_initialized_memory_plugin(
    tmp_path,
):
    db_path = tmp_path / "phase13_memory_config.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(
        str(db_path),
        name="memory-config-test",
        memory={"enabled": True},
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
        memory={"enabled": True},
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

    agent = await Agent.from_db(str(db_path), memory=False)

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
        await Agent.from_db(str(db_path), memory={"retrieval_mode": "hybrid"})

    with pytest.raises(ValueError, match="requires an explicit embedding"):
        await Agent.from_db(str(db_path), memory={"retrieval_mode": "embedding"})


async def test_agent_from_db_rejects_legacy_memory_true(tmp_path):
    db_path = tmp_path / "phase1_memory_true_rejected.sqlite"
    await _seed_sqlite(db_path)

    with pytest.raises(
        ValueError,
        match="memory must be False, a DbMemoryConfig, a config mapping, or None",
    ):
        await Agent.from_db(str(db_path), memory=True)


async def test_agent_from_db_rejects_memory_plugin_argument(tmp_path):
    db_path = tmp_path / "phase1_memory_plugin_rejected.sqlite"
    await _seed_sqlite(db_path)

    with pytest.raises(
        ValueError,
        match="memory must be False, a DbMemoryConfig, a config mapping, or None",
    ):
        await Agent.from_db(str(db_path), memory=MemoryPlugin())


async def test_agent_from_db_memory_config_can_provide_backend(tmp_path):
    db_path = tmp_path / "phase1_memory_backend.sqlite"
    await _seed_sqlite(db_path)
    backend = MagicMock()

    agent = await Agent.from_db(str(db_path), memory={"backend": backend})

    try:
        memory = agent.runtime.registry.get_plugin("memory")
    finally:
        await agent.stop()

    assert memory.backend is backend


async def test_agent_from_db_memory_update_works_without_extra_enablement(tmp_path):
    db_path = tmp_path / "phase1_memory_update_default.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path))
    try:
        memory = agent.runtime.registry.get_plugin("memory")
        backend = MagicMock()
        backend.list_by_category = AsyncMock(return_value=[])
        backend.remember = AsyncMock(
            return_value={"status": "success", "chunk_id": "mem-1"}
        )
        memory.backend = backend
        result = await agent.run_detailed(
            "Remember the revenue definition",
            mode="memory.update",
            metadata={
                "kind": "metric_definition",
                "key": "metric:revenue",
                "text": "Revenue excludes refunded orders.",
            },
        )
    finally:
        await agent.stop()

    memory_config = agent.runtime.config.metadata["from_db_options"]["memory"]
    assert result.status is OperationStatus.SUCCEEDED
    evidence = next(
        item for item in result.evidence if item.kind == "memory.semantic.write"
    )
    stored = backend.remember.await_args.kwargs["extra_metadata"]["db_memory"]
    assert evidence.payload["success"] is True
    assert stored["metadata"]["source_identity"] == memory_config["source_identity"]
    assert stored["metadata"]["workspace_scope"] == "source"


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
        memory={"backend": backend},
        calibrate_memory=True,
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
        memory={"backend": backend},
        calibrate_memory=True,
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

    agent = await Agent.from_db(str(db_path), memory={"backend": backend})

    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert not hasattr(agent, "local_tool_catalog")
    assert "remember" not in inspection.tool_view_names
    assert "update_memory" not in inspection.tool_view_names
    assert "memory:memory.semantic.write" in inspection.capability_ids


async def test_agent_from_db_memory_update_uses_runtime_capability_not_generic_tools(
    tmp_path,
):
    db_path = tmp_path / "phase13_generic_memory_guardrail.sqlite"
    await _seed_sqlite(db_path)
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )

    agent = await Agent.from_db(str(db_path), memory={"backend": backend})

    try:
        inspection = await agent.describe()
        result = await agent.run_detailed(
            "Please remember that revenue excludes refunds and update_memory if needed."
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind.value == "memory.update"
    assert "remember" not in inspection.tool_view_names
    assert "update_memory" not in inspection.tool_view_names
    backend.remember.assert_awaited_once()
    evidence = next(
        item for item in result.evidence if item.kind == "memory.semantic.write"
    )
    assert evidence.payload["success"] is True
    assert evidence.payload["category"] == "db_semantics"


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
    assert sqlite_plugin.query_timeout == 60
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
    assert sqlite_plugin.query_timeout == 60


async def test_agent_from_db_data_team_can_configure_memory_backend(tmp_path):
    db_path = tmp_path / "phase13_custom_memory_backend.sqlite"
    await _seed_sqlite(db_path)
    backend = _memory_backend(tmp_path)

    agent = await Agent.from_db(
        str(db_path),
        mode="data_team",
        memory={"backend": backend},
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


async def test_agent_from_db_accepts_legacy_construction_metadata_on_new_path(tmp_path):
    db_path = tmp_path / "phase13_metadata.sqlite"
    await _seed_sqlite(db_path)
    schema_prompt_policy = SchemaPromptPolicy()
    tool_result_policy = ToolResultPolicy(max_rows_inline=3)

    agent = await Agent.from_db(
        str(db_path),
        name="metadata-agent",
        model="gpt-4o-mini",
        api_key="sk-test",
        llm_provider="openai",
        temperature=0.2,
        prompt="Revenue is net of tax.",
        db_schema="analytics",
        include_sample_values=False,
        redact_pii_columns=False,
        cache_ttl=3600,
        toolkit=None,
        schema_prompt_policy=schema_prompt_policy,
        tool_result_policy=tool_result_policy,
    )

    try:
        metadata = agent.runtime.config.metadata["from_db_options"]
    finally:
        await agent.stop()

    assert metadata["model"] == "gpt-4o-mini"
    assert metadata["llm_provider"] == "openai"
    assert metadata["temperature"] == 0.2
    assert metadata["prompt"] == "Revenue is net of tax."
    assert metadata["db_schema"] == "analytics"
    assert metadata["include_sample_values"] is False
    assert metadata["redact_pii_columns"] is False
    assert metadata["cache_ttl"] == 3600
    assert metadata["schema_prompt_policy"]["preferred_strategy"] == "auto"
    assert metadata["tool_result_policy"]["max_rows_inline"] == 3
    assert "api_key" not in metadata


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


async def test_agent_from_db_toolkit_override_is_runtime_metadata(tmp_path):
    db_path = tmp_path / "phase13_toolkit_override.sqlite"
    await _seed_sqlite(db_path)

    agent = await Agent.from_db(str(db_path), mode="simple", toolkit="analyst")

    try:
        metadata = agent.runtime.config.metadata["from_db_options"]
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert inspection.profile == "simple"
    assert metadata["toolkit"] == "analyst"


async def test_agent_from_db_reuses_schema_profile_when_cache_ttl_allows(tmp_path):
    db_path = tmp_path / "phase13_schema_cache.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    original = source._execute_schema_inspect
    calls = 0

    async def counted_schema_inspect(payload):
        nonlocal calls
        calls += 1
        return await original(payload)

    source._execute_schema_inspect = counted_schema_inspect
    agent = await Agent.from_db(source, cache_ttl=3600)

    try:
        first = await agent.run_detailed("What columns are in customers?")
        second = await agent.run_detailed("What columns are in orders?")
    finally:
        await agent.stop()

    assert first.status is OperationStatus.SUCCEEDED
    assert second.status is OperationStatus.SUCCEEDED
    assert calls == 1
    cached_schema = next(
        item for item in second.evidence if item.kind == "schema.asset_profile"
    )
    assert cached_schema.metadata["schema_cache"] == "hit"
    assert cached_schema.id


async def test_agent_from_db_cache_hit_schema_evidence_supports_database_wide_followup(
    tmp_path,
):
    db_path = tmp_path / "phase13_schema_cache_database_followup.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    agent = await Agent.from_db(source, cache_ttl=3600)

    try:
        first = await agent.run_detailed("Tell me about the customers table.")
        second = await agent.run_detailed("What tables exist?", mode="schema.query")
    finally:
        await agent.stop()

    assert first.status is OperationStatus.SUCCEEDED
    assert second.status is OperationStatus.SUCCEEDED
    assert "customers: id, name" in first.answer
    assert "orders:" not in first.answer
    assert "Found 2 tables" in second.answer
    assert "customers: id, name" in second.answer
    assert "orders: id, customer_id, total" in second.answer
    cached_schema = next(
        item
        for item in second.evidence
        if item.kind == "schema.asset_profile"
        and item.metadata.get("schema_cache") == "hit"
    )
    assert cached_schema.id
    assert cached_schema.metadata.get("payload_fingerprint")


async def test_agent_from_db_cache_ttl_zero_reprofiles_schema(tmp_path):
    db_path = tmp_path / "phase13_schema_cache_zero.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    original = source._execute_schema_inspect
    calls = 0

    async def counted_schema_inspect(payload):
        nonlocal calls
        calls += 1
        return await original(payload)

    source._execute_schema_inspect = counted_schema_inspect
    agent = await Agent.from_db(source, cache_ttl=0)

    try:
        await agent.run_detailed("What columns are in customers?")
        await agent.run_detailed("What columns are in orders?")
    finally:
        await agent.stop()

    assert calls == 2


async def test_agent_from_db_reuses_persisted_catalog_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "phase13_persisted_catalog.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    profile_key = catalog_profile_key(source)
    _write_catalog_snapshot(
        profile_key,
        {
            "database_type": "sqlite",
            "database_name": str(db_path),
            "tables": [
                {
                    "name": "customers",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "name", "data_type": "TEXT"},
                    ],
                }
            ],
        },
    )
    original = source._execute_schema_inspect
    source._execute_schema_inspect = AsyncMock(side_effect=original)

    agent = await Agent.from_db(source, cache_ttl=None)

    try:
        result = await agent.run_detailed("What columns are in customers?")
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert "customers: id, name" in (result.answer or "")
    source._execute_schema_inspect.assert_not_awaited()
    schema_evidence = next(
        item for item in result.evidence if item.kind == "schema.asset_profile"
    )
    assert schema_evidence.metadata["schema_cache"] == "persistent_hit"
    assert schema_evidence.payload["store_id"] == f"from_db:{profile_key}"


async def test_agent_from_db_cache_ttl_zero_ignores_persisted_catalog_snapshot(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "phase13_expired_catalog.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    profile_key = catalog_profile_key(source)
    _write_catalog_snapshot(
        profile_key,
        {
            "database_type": "sqlite",
            "database_name": str(db_path),
            "tables": [
                {
                    "name": "stale_table",
                    "columns": [{"name": "stale_column", "data_type": "TEXT"}],
                }
            ],
        },
    )
    original = source._execute_schema_inspect
    source._execute_schema_inspect = AsyncMock(side_effect=original)

    agent = await Agent.from_db(source, cache_ttl=0)

    try:
        result = await agent.run_detailed("What columns are in customers?")
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    source._execute_schema_inspect.assert_awaited_once()
    schema_evidence = next(
        item for item in result.evidence if item.kind == "schema.asset_profile"
    )
    assert schema_evidence.metadata != {"schema_cache": "persistent_hit"}
    assert "customers: id, name" in (result.answer or "")


async def test_agent_from_db_unrelated_catalog_snapshot_does_not_shadow_source(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "phase13_unrelated_catalog.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    _write_catalog_snapshot(
        "unrelated-profile",
        {
            "database_type": "sqlite",
            "database_name": "other.sqlite",
            "tables": [
                {
                    "name": "stale_table",
                    "columns": [{"name": "stale_column", "data_type": "TEXT"}],
                }
            ],
        },
    )
    original = source._execute_schema_inspect
    source._execute_schema_inspect = AsyncMock(side_effect=original)

    agent = await Agent.from_db(source, cache_ttl=None)

    try:
        result = await agent.run_detailed("What columns are in customers?")
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    source._execute_schema_inspect.assert_awaited_once()
    schema_evidence = next(
        item for item in result.evidence if item.kind == "schema.asset_profile"
    )
    assert schema_evidence.metadata.get("schema_cache") != "persistent_hit"
    assert "customers: id, name" in (result.answer or "")


async def test_agent_from_db_stale_catalog_snapshot_is_refreshed_and_repersisted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "phase13_refresh_catalog.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    profile_key = catalog_profile_key(source)
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    _write_catalog_snapshot(
        profile_key,
        {
            "database_type": "sqlite",
            "database_name": str(db_path),
            "tables": [
                {
                    "name": "stale_table",
                    "columns": [{"name": "stale_column", "data_type": "TEXT"}],
                }
            ],
        },
        last_seen=old_ts,
    )
    original = source._execute_schema_inspect
    source._execute_schema_inspect = AsyncMock(side_effect=original)

    agent = await Agent.from_db(source, cache_ttl=0)

    try:
        result = await agent.run_detailed("What columns are in customers?")
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    source._execute_schema_inspect.assert_awaited_once()
    catalog = json.loads((Path(".daita") / "catalog.json").read_text())
    refreshed = catalog[f"from_db:{profile_key}"]
    assert [table["name"] for table in refreshed["tables"]] == [
        "customers",
        "orders",
    ]
    assert refreshed["metadata"]["profile_key"] == profile_key


async def test_agent_from_db_uses_stale_catalog_snapshot_when_refresh_fails(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "phase13_stale_fallback.sqlite"
    await _seed_sqlite(db_path)
    source = SQLitePlugin(path=str(db_path))
    profile_key = catalog_profile_key(source)
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    _write_catalog_snapshot(
        profile_key,
        {
            "database_type": "sqlite",
            "database_name": str(db_path),
            "tables": [
                {
                    "name": "orders",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "customer_id", "data_type": "INTEGER"},
                        {"name": "total", "data_type": "REAL"},
                    ],
                }
            ],
        },
        last_seen=old_ts,
    )
    source._execute_schema_inspect = AsyncMock(side_effect=RuntimeError("DB down"))

    agent = await Agent.from_db(source, cache_ttl=0)

    try:
        result = await agent.run_detailed("What columns are in orders?")
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    source._execute_schema_inspect.assert_awaited_once()
    schema_evidence = next(
        item for item in result.evidence if item.kind == "schema.asset_profile"
    )
    assert schema_evidence.metadata["schema_cache"] == "persistent_stale_fallback"
    assert schema_evidence.metadata["refresh_error"] == "DB down"
    assert "orders: id, customer_id, total" in (result.answer or "")


async def test_agent_from_db_rejects_unknown_mode_on_new_path(tmp_path):
    db_path = tmp_path / "phase13_bad_mode.sqlite"
    await _seed_sqlite(db_path)

    with pytest.raises(ValueError, match="Unknown from_db mode"):
        await Agent.from_db(str(db_path), mode="nope")
