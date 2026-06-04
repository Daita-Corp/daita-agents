from dataclasses import dataclass

import pytest

from daita.db import (
    DbAgent,
    DbLimits,
    DbRuntime,
    DbRuntimeConfig,
    DbRuntimeInspection,
)
from daita.plugins import RuntimeExtensionPlugin, PluginKind, PluginManifest
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    RiskLevel,
    RuntimeEventType,
    Task,
)


@dataclass(frozen=True)
class SearchExecutor:
    id: str = "catalog.schema.search"
    capability_ids: frozenset[str] = frozenset({"catalog.schema.search"})

    async def execute(self, task: Task, operation: Operation, context):
        return [
            Evidence(
                kind="schema.search_result",
                owner="catalog",
                operation_id=operation.id,
                task_id=task.id,
                payload={"matches": []},
            )
        ]


class CapturingRuntimePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="runtime_probe",
        display_name="Runtime Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.setup_context = None
        self.teardown_called = False

    async def setup(self, context):
        self.setup_context = context

    async def teardown(self):
        self.teardown_called = True

    def declare_capabilities(self):
        return [
            Capability(
                id="runtime_probe.inspect",
                owner="runtime_probe",
                description="Inspect the DB runtime.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"runtime.inspect"}),
                access=AccessMode.NONE,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"runtime_probe.inspection"}),
                executor="runtime_probe.inspect",
                runtime_only=True,
                side_effecting=False,
            )
        ]

    def get_executors(self):
        return [
            SearchExecutor(
                id="runtime_probe.inspect",
                capability_ids=frozenset({"runtime_probe.inspect"}),
            )
        ]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="runtime_probe.inspection",
                owner="runtime_probe",
                json_schema={"type": "object"},
            )
        ]


class CountingExecutor:
    id = "runtime_probe.count"
    capability_ids = frozenset({"runtime_probe.count"})

    def __init__(self):
        self.calls = 0

    async def execute(self, task: Task, operation: Operation, context):
        self.calls += 1
        return [
            Evidence(
                kind="runtime_probe.counted",
                owner="runtime_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload={"calls": self.calls},
            )
        ]


class CountingRuntimePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="runtime_probe_counting",
        display_name="Runtime Probe Counting",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.executor = CountingExecutor()

    def declare_capabilities(self):
        return [
            Capability(
                id="runtime_probe.count",
                owner="runtime_probe_counting",
                description="Count runtime executions.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"runtime.inspect"}),
                access=AccessMode.NONE,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"runtime_probe.counted"}),
                executor="runtime_probe.count",
                runtime_only=True,
                side_effecting=False,
            )
        ]

    def get_executors(self):
        return [self.executor]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="runtime_probe.counted",
                owner="runtime_probe_counting",
                json_schema={"type": "object"},
            )
        ]


def test_db_runtime_owns_registry_and_accepts_config_plugins():
    plugin = CapturingRuntimePlugin()
    runtime = DbRuntime(
        source="sqlite:///tmp/example.db",
        config=DbRuntimeConfig(plugins=(plugin,)),
        runtime_id="db-runtime-test",
    )

    assert runtime.runtime_id == "db-runtime-test"
    assert runtime.registry.plugin_ids == ("runtime_probe",)
    assert runtime.registry.get_capability("runtime_probe.inspect").owner == (
        "runtime_probe"
    )


def test_db_runtime_can_register_plugins_before_setup():
    runtime = DbRuntime()

    runtime.register_plugin(CapturingRuntimePlugin())

    assert runtime.registry.plugin_ids == ("runtime_probe",)
    assert runtime.registry.get_capability("runtime_probe.inspect").executor == (
        "runtime_probe.inspect"
    )


async def test_db_runtime_setup_passes_plugin_context_and_teardown_runs():
    plugin = CapturingRuntimePlugin()
    runtime = DbRuntime(config=DbRuntimeConfig(plugins=(plugin,)))

    await runtime.setup(agent_id="agent-1")
    await runtime.teardown()

    assert plugin.setup_context.runtime_kind == "db"
    assert plugin.setup_context.agent_id == "agent-1"
    assert plugin.setup_context.services.require("db_runtime") is runtime
    assert plugin.setup_context.config["profile"] == "analyst"
    assert plugin.teardown_called is True
    assert runtime.is_setup is False
    assert runtime.setup_context is None


async def test_db_runtime_exposes_runtime_store_to_plugins():
    store = InMemoryRuntimeStore()
    plugin = CapturingRuntimePlugin()
    runtime = DbRuntime(config=DbRuntimeConfig(plugins=(plugin,)), store=store)

    await runtime.setup(agent_id="agent-1")

    assert plugin.setup_context.services.require("runtime_store") is store


async def test_db_runtime_setup_and_teardown_are_idempotent():
    plugin = CapturingRuntimePlugin()
    runtime = DbRuntime(config=DbRuntimeConfig(plugins=(plugin,)))

    await runtime.setup(agent_id="agent-1")
    context = plugin.setup_context
    await runtime.setup(agent_id="agent-2")
    await runtime.teardown()
    await runtime.teardown()

    assert plugin.setup_context is context
    assert plugin.setup_context.agent_id == "agent-1"
    assert plugin.teardown_called is True


async def test_db_runtime_rejects_plugin_registration_after_setup():
    runtime = DbRuntime()

    await runtime.setup()

    with pytest.raises(RuntimeError, match="after DbRuntime.setup"):
        runtime.register_plugin(CapturingRuntimePlugin())


async def test_db_runtime_inspect_reports_registry_diagnostics_and_redacts_source():
    runtime = DbRuntime(
        source="postgresql://user:secret@localhost/sales",
        config=DbRuntimeConfig(plugins=(CapturingRuntimePlugin(),)),
    )

    inspection = await runtime.inspect()

    assert isinstance(inspection, DbRuntimeInspection)
    assert inspection.runtime_kind == "db"
    assert inspection.source_repr == "postgresql://user:***@localhost/sales"
    assert inspection.plugin_ids == ("runtime_probe",)
    assert inspection.capability_count == 1
    assert inspection.executor_count == 1
    assert inspection.evidence_schema_count == 1
    assert inspection.capability_ids == ("runtime_probe:runtime_probe.inspect",)
    assert inspection.executor_ids == ("runtime_probe.inspect",)
    assert inspection.evidence_schema_kinds == (
        "runtime_probe:runtime_probe.inspection",
    )
    assert inspection.diagnostics[0]["plugin_id"] == "runtime_probe"
    assert inspection.to_dict()["plugin_ids"] == ["runtime_probe"]
    assert inspection.to_dict()["capability_ids"] == [
        "runtime_probe:runtime_probe.inspect"
    ]


async def test_db_runtime_persists_operation_state_for_inspection_and_resume():
    store = InMemoryRuntimeStore()
    plugin = CountingRuntimePlugin()
    runtime = DbRuntime(plugins=(plugin,), store=store)

    evidence = await runtime.execute_capability(
        "runtime_probe.count",
        owner="runtime_probe_counting",
        operation_type="runtime.inspect",
        input={"probe": True},
        operation_id="op-count",
    )
    before_resume = await runtime.inspect_operation("op-count")
    resumed = await runtime.resume_operation("op-count")

    assert plugin.executor.calls == 1
    assert evidence[0].payload == {"calls": 1}
    assert before_resume.operation.status is OperationStatus.SUCCEEDED
    assert before_resume.evidence == evidence
    assert len(before_resume.tasks) == 1
    assert before_resume.completed_task_ids == (before_resume.tasks[0].id,)
    assert before_resume.resumable_task_ids == ()
    assert plugin.executor.calls == 1
    assert resumed.completed_task_ids == before_resume.completed_task_ids
    assert [event.type for event in resumed.events][-2:] == [
        RuntimeEventType.OPERATION_RESUMED,
        RuntimeEventType.TASK_SKIPPED,
    ]


async def test_db_agent_facade_uses_runtime_without_generic_agent_patch():
    runtime = DbRuntime()
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed("How many orders are there?")
    answer = await agent.run("How many orders are there?")
    description = await agent.describe()
    streamed = [item async for item in agent.stream("How many orders are there?")]

    assert agent.runtime is runtime
    assert not hasattr(agent, "_db_original_run")
    assert not hasattr(agent, "_db_last_context_metadata")
    assert not hasattr(agent, "local_tool_catalog")
    assert result.status is OperationStatus.BLOCKED
    assert result.request.prompt == "How many orders are there?"
    assert result.warnings == ("db_runtime_missing_capabilities",)
    assert answer == "Required DB capabilities are not registered."
    assert len(streamed) == 1
    assert streamed[0].status is OperationStatus.BLOCKED
    assert streamed[0].request.prompt == "How many orders are there?"
    assert description.runtime_id == runtime.runtime_id


def test_db_limits_validate_positive_values():
    limits = DbLimits(max_rows=10, timeout_seconds=1, max_tasks=2)

    assert limits.to_dict()["max_rows"] == 10
