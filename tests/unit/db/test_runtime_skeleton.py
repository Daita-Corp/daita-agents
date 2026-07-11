import asyncio
from dataclasses import FrozenInstanceError, dataclass, fields
import inspect
from unittest.mock import AsyncMock

import pytest

from daita.db import (
    DbAgent,
    DbLimits,
    DbRuntime,
    DbRuntimeConfig,
    DbRuntimeInspection,
    DbMonitorScheduler,
)
from daita.db.runtime.tasks import DbTaskContext, DbTaskExecutor, DbTaskRuntime
from daita.db.runtime.tasks import execution as task_execution
from daita.db.runtime.tasks.models import DbTaskSpec
from daita.plugins import (
    ExtensionRegistry,
    RuntimeExtensionPlugin,
    PluginKind,
    PluginManifest,
)
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    RiskLevel,
    RuntimeKernel,
    TaskDependency,
    TaskExecutionResult,
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


def test_db_task_context_is_frozen_and_contains_only_approved_fields():
    runtime = DbRuntime(runtime_id="db-runtime-context")

    context = runtime.tasks.context

    assert isinstance(context, DbTaskContext)
    assert tuple(field.name for field in fields(context)) == (
        "registry",
        "store",
        "kernel",
        "config",
        "runtime_id",
    )
    assert context.registry is runtime.registry
    assert context.store is runtime.store
    assert context.kernel is runtime.kernel
    assert context.config is runtime.config
    assert context.runtime_id == runtime.runtime_id
    with pytest.raises(FrozenInstanceError):
        setattr(context, "runtime_id", "changed")


async def test_db_task_executor_runs_without_db_runtime_and_persists_before_kernel(
    monkeypatch,
):
    plugin = CountingRuntimePlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="standalone-task-runtime",
        runtime_kind="db",
        extension_registry=registry,
        runtime_store=store,
    )
    executor = DbTaskExecutor(
        DbTaskContext(
            registry=registry,
            store=store,
            kernel=kernel,
            config=DbRuntimeConfig(),
            runtime_id="standalone-task-runtime",
        )
    )
    operation = Operation(
        id="standalone-operation",
        operation_type="runtime.inspect",
        status=OperationStatus.RUNNING,
    )
    task = Task(
        id="standalone-task",
        operation_id=operation.id,
        capability_id="runtime_probe.count",
        executor_id="runtime_probe.count",
        input={"probe": True},
        metadata={"owner": "runtime_probe_counting", "reason": "test"},
    )
    await store.save_operation(operation)
    original_execute_task = kernel.execute_task
    persisted_at_execution = []

    async def execute_after_persist(task_id, **kwargs):
        persisted_at_execution.append(await store.load_task(task_id))
        return await original_execute_task(task_id, **kwargs)

    monkeypatch.setattr(kernel, "execute_task", execute_after_persist)

    evidence = await executor.execute_task(task, operation)

    assert plugin.executor.calls == 1
    assert evidence[0].payload == {"calls": 1}
    assert persisted_at_execution[0] is not None
    assert persisted_at_execution[0].id == task.id
    assert persisted_at_execution[0].input == task.input
    assert persisted_at_execution[0].dependencies == task.dependencies


async def test_db_task_runtime_composes_without_db_runtime():
    plugin = CountingRuntimePlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="standalone-task-runtime",
        runtime_kind="db",
        extension_registry=registry,
        runtime_store=store,
    )

    tasks = DbTaskRuntime(
        DbTaskContext(
            registry=registry,
            store=store,
            kernel=kernel,
            config=DbRuntimeConfig(),
            runtime_id="standalone-task-runtime",
        )
    )

    operation = Operation(
        id="standalone-runtime-operation",
        operation_type="runtime.inspect",
        status=OperationStatus.RUNNING,
    )
    await store.save_operation(operation)
    plan = await tasks.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id="runtime_probe.count",
                owner="runtime_probe_counting",
                input={"probe": True},
            ),
        ),
    )
    readiness = await tasks.task_readiness(plan.tasks[0], operation)
    executable_input = await tasks.executable_input_for_task(plan.tasks[0], operation)
    evidence = await tasks.execute_task(plan.tasks[0], operation)

    assert isinstance(tasks.executor, DbTaskExecutor)
    assert tasks.context.kernel is kernel
    assert readiness == {
        "ready": True,
        "unsatisfied_dependencies": [],
        "dependency_count": 0,
    }
    assert executable_input == plan.tasks[0].input
    assert evidence[0].payload == {"calls": 1}
    assert (
        await tasks.latest_accepted_evidence(operation.id, "runtime_probe.counted")
        == evidence[0]
    )
    assert tasks.capability_for_task(plan.tasks[0]).owner == "runtime_probe_counting"


def test_task_facades_preserve_explicit_typed_signatures():
    expected_parameters = {
        "execute_task": ("self", "task", "operation", "context"),
        "execute_capability": (
            "self",
            "capability_id",
            "owner",
            "operation_type",
            "input",
            "operation_id",
        ),
        "plan_task_specs": ("self", "operation", "specs", "contract"),
        "task_readiness": ("self", "task", "operation"),
        "executable_input_for_task": ("self", "task", "operation"),
    }
    for owner in (DbTaskRuntime, DbRuntime):
        for method_name, parameter_names in expected_parameters.items():
            signature = inspect.signature(getattr(owner, method_name))
            assert tuple(signature.parameters) == parameter_names
            assert all(
                parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
                for parameter in signature.parameters.values()
            )
            assert signature.return_annotation is not inspect.Signature.empty


def test_db_task_executor_and_task_modules_share_one_execution_implementation():
    class_source = inspect.getsource(DbTaskExecutor.execute_task)
    function_source = inspect.getsource(task_execution.execute_task)

    assert "return await execute_task(self.context" in class_source
    assert "DbTaskExecutor(" not in function_source


async def test_db_runtime_execute_task_delegates_to_composed_task_runtime(monkeypatch):
    runtime = DbRuntime()
    operation = Operation(id="delegate-operation", operation_type="runtime.inspect")
    task = Task(
        id="delegate-task",
        operation_id=operation.id,
        capability_id="runtime_probe.count",
        executor_id="runtime_probe.count",
    )
    delegated = AsyncMock(return_value=())
    monkeypatch.setattr(runtime.tasks, "execute_task", delegated)

    result = await runtime.execute_task(task, operation, {"trace_id": "trace-1"})

    assert result == ()
    delegated.assert_awaited_once_with(
        task,
        operation,
        {"trace_id": "trace-1"},
    )


async def test_db_runtime_execute_capability_owns_setup_before_delegation(monkeypatch):
    runtime = DbRuntime()
    setup = AsyncMock()
    delegated = AsyncMock(return_value=())
    monkeypatch.setattr(runtime, "setup", setup)
    monkeypatch.setattr(runtime.tasks, "execute_capability", delegated)

    result = await runtime.execute_capability(
        "runtime_probe.count",
        owner="runtime_probe_counting",
        operation_type="runtime.inspect",
        input={"probe": True},
        operation_id="setup-delegate-operation",
    )

    assert result == ()
    setup.assert_awaited_once_with()
    delegated.assert_awaited_once_with(
        "runtime_probe.count",
        owner="runtime_probe_counting",
        operation_type="runtime.inspect",
        input={"probe": True},
        operation_id="setup-delegate-operation",
    )


def test_db_runtime_mro_contains_no_task_mixins():
    assert all("Task" not in base.__name__ for base in DbRuntime.__mro__[1:])


async def test_db_task_executor_rejects_executor_mismatch_before_kernel(monkeypatch):
    runtime = DbRuntime(plugins=(CountingRuntimePlugin(),))
    operation = Operation(id="mismatch-operation", operation_type="runtime.inspect")
    task = Task(
        id="mismatch-task",
        operation_id=operation.id,
        capability_id="runtime_probe.count",
        executor_id="wrong.executor",
        metadata={"owner": "runtime_probe_counting"},
    )
    execute_task = AsyncMock()
    monkeypatch.setattr(runtime.kernel, "execute_task", execute_task)

    with pytest.raises(ValueError, match="does not match capability"):
        await runtime.tasks.executor.execute_task(task, operation)

    execute_task.assert_not_awaited()
    assert await runtime.store.load_task(task.id) is None


async def test_db_task_executor_preserves_pending_input_and_dependency_update(
    monkeypatch,
):
    runtime = DbRuntime(plugins=(CountingRuntimePlugin(),))
    capability = runtime.registry.get_capability(
        "runtime_probe.count",
        owner="runtime_probe_counting",
    )
    operation = Operation(
        id="pending-update-operation",
        operation_type="runtime.inspect",
        status=OperationStatus.RUNNING,
    )
    old_dependency = TaskDependency(
        kind="evidence",
        evidence_kind="probe.old",
        operation_id=operation.id,
    )
    new_dependency = TaskDependency(
        kind="evidence",
        evidence_kind="probe.new",
        producer_task_id="producer-new",
        operation_id=operation.id,
    )
    stored_task = Task(
        id="pending-update-task",
        operation_id=operation.id,
        capability_id=capability.id,
        executor_id=capability.executor,
        input={"version": "old"},
        dependencies=(old_dependency,),
        metadata={
            "owner": capability.owner,
            "reason": "old",
            "preserved": True,
        },
    )
    incoming_task = Task(
        id=stored_task.id,
        operation_id=operation.id,
        capability_id=capability.id,
        executor_id=capability.executor,
        input={"version": "new"},
        dependencies=(new_dependency,),
        metadata={
            "owner": capability.owner,
            "reason": "new",
            "discarded": True,
        },
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(stored_task)
    persisted_at_execution = []

    async def execute_updated_task(task_id, **kwargs):
        persisted = await runtime.store.load_task(task_id)
        persisted_at_execution.append(persisted)
        return TaskExecutionResult(operation, persisted, capability)

    monkeypatch.setattr(runtime.kernel, "execute_task", execute_updated_task)

    assert await runtime.tasks.executor.execute_task(incoming_task, operation) == ()

    updated = persisted_at_execution[0]
    assert updated.input == incoming_task.input
    assert [item.to_dict() for item in updated.dependencies] == [
        new_dependency.to_dict()
    ]
    assert updated.metadata == {
        "owner": capability.owner,
        "reason": "new",
        "preserved": True,
    }


def test_db_runtime_owns_registry_and_accepts_config_plugins():
    plugin = CapturingRuntimePlugin()
    runtime = DbRuntime(
        source="sqlite:///tmp/example.db",
        config=DbRuntimeConfig(plugins=(plugin,)),
        runtime_id="db-runtime-test",
    )

    assert runtime.runtime_id == "db-runtime-test"
    assert runtime.registry.plugin_ids == ("db_runtime", "runtime_probe")
    assert runtime.registry.get_capability("runtime_probe.inspect").owner == (
        "runtime_probe"
    )


def test_db_runtime_can_register_plugins_before_setup():
    runtime = DbRuntime()

    runtime.register_plugin(CapturingRuntimePlugin())

    assert runtime.registry.plugin_ids == ("db_runtime", "runtime_probe")
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


async def test_db_runtime_setup_starts_no_monitor_background_work(monkeypatch):
    runtime = DbRuntime()
    created_tasks = []
    original_create_task = asyncio.create_task
    scheduler_pass = AsyncMock()

    def capture_create_task(coro, *args, **kwargs):
        task = original_create_task(coro, *args, **kwargs)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", capture_create_task)
    monkeypatch.setattr(DbMonitorScheduler, "run_once", scheduler_pass)

    try:
        await runtime.setup()
    finally:
        for task in created_tasks:
            task.cancel()
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)

    assert created_tasks == []
    scheduler_pass.assert_not_awaited()
    assert all(
        "monitor" not in worker.id and "monitor" not in worker.role
        for worker in runtime.registry.workers
    )


async def test_db_runtime_exposes_runtime_store_to_plugins():
    store = InMemoryRuntimeStore()
    plugin = CapturingRuntimePlugin()
    runtime = DbRuntime(config=DbRuntimeConfig(plugins=(plugin,)), store=store)

    await runtime.setup(agent_id="agent-1")

    assert plugin.setup_context.services.require("runtime_store") is store


async def test_db_runtime_merges_host_services_before_plugin_setup():
    plugin = CapturingRuntimePlugin()
    hosted_service = object()
    runtime = DbRuntime(
        config=DbRuntimeConfig(plugins=(plugin,)),
        host_services={
            "hosted_in_app_notification_service": hosted_service,
            "db_runtime": "host-value",
        },
    )

    await runtime.setup(agent_id="agent-1")

    assert (
        plugin.setup_context.services.require("hosted_in_app_notification_service")
        is hosted_service
    )
    assert plugin.setup_context.services.require("db_runtime") is runtime


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
    assert inspection.capability_count == len(inspection.capability_ids)
    assert inspection.executor_count == len(inspection.executor_ids)
    assert inspection.evidence_schema_count == len(inspection.evidence_schema_kinds)
    assert "runtime_probe:runtime_probe.inspect" in inspection.capability_ids
    assert "db_runtime:db.analysis.plan" in inspection.capability_ids
    assert "db_runtime:db.analysis.summarize" in inspection.capability_ids
    assert "runtime_probe.inspect" in inspection.executor_ids
    assert "runtime_probe:runtime_probe.inspection" in inspection.evidence_schema_kinds
    assert "db_runtime:analysis.plan" in inspection.evidence_schema_kinds
    assert "db_runtime:analysis.synthesis" in inspection.evidence_schema_kinds
    assert any(item["plugin_id"] == "runtime_probe" for item in inspection.diagnostics)
    assert inspection.to_dict()["plugin_ids"] == ["runtime_probe"]
    assert (
        "runtime_probe:runtime_probe.inspect" in inspection.to_dict()["capability_ids"]
    )


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


def test_db_limits_validate_positive_values():
    limits = DbLimits(max_rows=10, timeout_seconds=1, max_tasks=2)

    assert limits.to_dict()["max_rows"] == 10
