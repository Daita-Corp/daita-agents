import importlib.util
import inspect
import json

import pytest

import daita.db as db
from daita.agents.agent import Agent
from daita.core.exceptions import ValidationError
from daita.db import (
    DbLLMConfig,
    DbLimits,
    DbMemoryConfig,
    DbRuntimeConfig,
    DbRuntimeOptions,
    DbSourceOptions,
)
from daita.db.factory import _resolve_runtime_source
from daita.plugins.postgresql import PostgreSQLPlugin

PUBLIC_NAMES = [
    "DbAgent",
    "DbIntent",
    "DbIntentKind",
    "DbLimits",
    "DbOperationContract",
    "DbOperationResult",
    "DbRequest",
    "DbRuntime",
    "DbRuntimeConfig",
    "DbRuntimeInspection",
    "DbSourceOptions",
    "DbLLMConfig",
    "DbMemoryConfig",
    "DbExecutionConfig",
    "DbRuntimeOptions",
    "DbMonitor",
    "DbMonitorInspection",
    "DbMonitorMutation",
    "DbMonitorRun",
    "DbMonitorState",
    "DbMonitorStore",
    "HostedInAppMonitorDeliveryPlugin",
    "from_db",
]

PARAMETER_NAMES = [
    "source",
    "name",
    "mode",
    "config",
    "source_options",
    "llm",
    "runtime",
    "memory",
    "catalog",
    "lineage",
    "quality",
    "history",
    "stateful",
    "plugins",
    "skills",
]

REMOVED_KEYWORDS = [
    "model",
    "api_key",
    "llm_provider",
    "temperature",
    "db_schema",
    "include_sample_values",
    "redact_pii_columns",
    "cache_ttl",
    "calibrate_memory",
    "read_only",
    "query_default_limit",
    "query_max_rows",
    "query_max_chars",
    "query_timeout",
    "allowed_tables",
    "blocked_tables",
    "blocked_columns",
    "prompt",
    "toolkit",
    "budget",
    "schema_prompt_policy",
    "tool_result_policy",
]


def test_public_from_db_signatures_are_exact_and_non_variadic():
    factory_signature = inspect.signature(db.from_db)
    facade_signature = inspect.signature(Agent.from_db)

    assert list(factory_signature.parameters) == PARAMETER_NAMES
    assert list(facade_signature.parameters) == PARAMETER_NAMES
    assert factory_signature.parameters["source"].default is inspect.Parameter.empty
    assert facade_signature.parameters["source"].default is inspect.Parameter.empty
    for name in PARAMETER_NAMES[1:12]:
        assert factory_signature.parameters[name].default is None
        assert facade_signature.parameters[name].default is None
    assert factory_signature.parameters["stateful"].default is False
    assert facade_signature.parameters["stateful"].default is False
    assert factory_signature.parameters["plugins"].default == ()
    assert facade_signature.parameters["plugins"].default == ()
    assert factory_signature.parameters["skills"].default == ()
    assert facade_signature.parameters["skills"].default == ()
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in factory_signature.parameters.values()
    )
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in facade_signature.parameters.values()
    )


@pytest.mark.parametrize("entrypoint", [db.from_db, Agent.from_db])
@pytest.mark.parametrize("keyword", REMOVED_KEYWORDS)
def test_removed_keywords_use_normal_python_type_error(entrypoint, keyword):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        entrypoint(":memory:", **{keyword: None})


def test_db_root_has_exact_public_surface():
    assert db.__all__ == PUBLIC_NAMES
    assert all(getattr(db, name) is not None for name in PUBLIC_NAMES)
    for internal_name in (
        "DbMonitorScheduler",
        "SQLiteDbMonitorStore",
        "DbQueryPlan",
        "DbSynthesizer",
        "DbVerifier",
        "analyze_sql",
        "DBPromptReadModel",
    ):
        assert not hasattr(db, internal_name)


async def test_record_api_merges_restrictively_and_deterministically(tmp_path):
    config = DbRuntimeConfig(
        limits=DbLimits(max_rows=120, timeout_seconds=20),
        source_options=DbSourceOptions(
            read_only=False,
            allowed_tables=("Zeta", "Orders"),
            blocked_tables=("Payments",),
            blocked_columns=("Customers.Email",),
        ),
    )
    explicit = DbSourceOptions(
        read_only=False,
        query_default_limit=7,
        query_max_rows=150,
        query_max_chars=900,
        query_timeout=50,
        allowed_tables=("customers", "orders"),
        blocked_tables=("audit", "payments"),
        blocked_columns=("orders.secret",),
        cache_ttl=0,
    )

    agent = await db.from_db(
        str(tmp_path / "records.sqlite"),
        mode="simple",
        config=config,
        source_options=explicit,
        memory=DbMemoryConfig(enabled=False),
        catalog=False,
    )
    try:
        source = agent.runtime.registry.get_plugin("sqlite")
        inspection = await agent.describe()
    finally:
        await agent.stop()

    effective = agent.runtime.config.source_options
    assert effective is not None
    assert source.read_only is True
    assert source.query_default_limit == 7
    assert source.query_max_rows == 120
    assert source.query_max_chars == 900
    assert source.query_timeout == 20
    assert source.allowed_tables == {"orders"}
    assert source.blocked_tables == {"audit", "payments"}
    assert source.blocked_columns == {"customers.email", "orders.secret"}
    assert effective.allowed_tables == ("orders",)
    assert effective.blocked_tables == ("audit", "payments")
    assert effective.blocked_columns == ("customers.email", "orders.secret")
    assert inspection.limits["max_rows"] == 120
    assert inspection.limits["timeout_seconds"] == 20
    assert (
        inspection.metadata["from_db_options"]["source_options"] == effective.to_dict()
    )


async def test_disjoint_allowed_table_sets_deny_every_table(tmp_path):
    agent = await db.from_db(
        str(tmp_path / "disjoint.sqlite"),
        config=DbRuntimeConfig(
            source_options=DbSourceOptions(allowed_tables=("orders",))
        ),
        source_options=DbSourceOptions(allowed_tables=("customers",)),
        memory=DbMemoryConfig(enabled=False),
        catalog=False,
    )
    try:
        source = agent.runtime.registry.get_plugin("sqlite")
        with pytest.raises(ValidationError, match="outside allowlist"):
            source._validate_sql_policy("select * from orders", operation="query")
    finally:
        await agent.stop()

    assert source.allowed_tables == set()
    assert source._allowed_tables_restricted is True


async def test_mode_defaults_apply_without_record_override(tmp_path):
    agent = await db.from_db(
        str(tmp_path / "mode.sqlite"),
        mode="simple",
        memory=DbMemoryConfig(enabled=False),
        catalog=False,
    )
    try:
        source = agent.runtime.registry.get_plugin("sqlite")
    finally:
        await agent.stop()

    assert source.read_only is True
    assert source.query_default_limit == 50
    assert source.query_max_rows == 100
    assert source.query_max_chars == 25000


def test_source_schema_reaches_postgresql_connector():
    source = _resolve_runtime_source(
        "postgresql://localhost/warehouse",
        options=DbSourceOptions(schema="analytics"),
    )

    assert isinstance(source, PostgreSQLPlugin)
    assert source.schema == "analytics"
    assert source.config["schema"] == "analytics"


async def test_default_analyst_mode_remains_read_only(tmp_path):
    agent = await db.from_db(
        str(tmp_path / "writable.sqlite"),
        source_options=DbSourceOptions(read_only=False),
        memory=DbMemoryConfig(enabled=False),
        catalog=False,
    )
    try:
        source = agent.runtime.registry.get_plugin("sqlite")
    finally:
        await agent.stop()

    assert source.read_only is True


async def test_llm_secrets_stay_at_live_service_boundary(tmp_path):
    secret = "package-l-secret-value"
    store_path = tmp_path / "runtime.sqlite"
    llm = DbLLMConfig(
        provider="mock",
        model="mock-model",
        api_key=secret,
        temperature=0.1,
        options={"credential": secret, "response_format": "json"},
    )
    agent = await db.from_db(
        str(tmp_path / "secret.sqlite"),
        name="secret-test",
        llm=llm,
        runtime=DbRuntimeOptions(store="sqlite", store_path=store_path),
        memory=DbMemoryConfig(enabled=False),
        catalog=False,
    )
    try:

        class _DiagnosticsProvider:
            async def generate(self, messages, *, stream):
                return '{"answer": "safe"}'

            def _get_last_token_usage(self):
                return {"total_tokens": 3, "credential": secret}

            def _estimate_cost(self, usage):
                return secret

            def get_pricing_metadata(self):
                return {
                    "pricing_provider": "mock",
                    "pricing_warning": f"credential={secret}",
                    "api_key": secret,
                }

        agent.runtime.db_llm_service._provider = _DiagnosticsProvider()
        llm_response = await agent.runtime.db_llm_service.generate_json(
            [{"role": "user", "content": "Return JSON."}]
        )
        operation = await agent.runtime.kernel.create_operation(
            operation_type="secret.test",
            request={"value": "safe"},
        )
        await agent.runtime.kernel.complete_operation(
            operation.id, payload={"answer": "safe"}
        )
        inspection = await agent.describe()
        events = await agent.runtime.store.list_events(operation.id)
        assert agent.runtime.db_llm_service.config is llm
        assert agent.runtime.db_llm_service.config.api_key == secret
        assert agent.runtime.db_llm_service.config.options["credential"] == secret
        serialized_live_state = json.dumps(
            {
                "metadata": agent.runtime.config.metadata,
                "inspection": inspection.to_dict(),
                "events": [event.to_dict() for event in events],
                "llm_diagnostics": llm_response.diagnostics,
            },
            sort_keys=True,
        )
    finally:
        await agent.stop()

    assert secret not in serialized_live_state
    assert llm_response.diagnostics["tokens"] == {"total_tokens": 3}
    assert llm_response.diagnostics["pricing"] == {"pricing_provider": "mock"}
    assert "estimated_cost_usd" not in llm_response.diagnostics
    assert secret not in repr(llm)
    assert secret.encode() not in store_path.read_bytes()


async def test_memory_record_owns_enablement_backend_and_calibration(tmp_path):
    backend = object()
    memory = DbMemoryConfig(
        enabled=False,
        recall="auto",
        learning="safe",
        limit=8,
        char_budget=1200,
        score_threshold=0.7,
        backend=backend,
        retrieval_mode="structured",
        calibrate=True,
    )
    agent = await db.from_db(
        str(tmp_path / "memory.sqlite"),
        memory=memory,
        catalog=False,
    )
    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    persisted = inspection.metadata["from_db_options"]["memory"]
    assert "memory" not in inspection.plugin_ids
    assert persisted["enabled"] is False
    assert persisted["recall"] == "off"
    assert persisted["learning"] == "off"
    assert persisted["limit"] == 8
    assert persisted["char_budget"] == 1200
    assert persisted["score_threshold"] == 0.7
    assert persisted["backend"] == "object"
    assert persisted["calibrate"] is True


def test_deleted_module_specs_are_absent():
    for module_name in (
        "daita.agents.db",
        "daita.db.catalog_prompt",
        "daita.db.config",
        "daita.db.config.policies",
    ):
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        assert spec is None
