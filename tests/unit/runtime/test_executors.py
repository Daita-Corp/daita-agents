import pytest

from daita.plugins import ExtensionRegistry, PluginKind, PluginManifest
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceWrappingExecutor,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    RiskLevel,
    RuntimeKernel,
    RuntimeKernelExecutorFailed,
    Task,
)


class ExecutorPlugin:
    manifest = PluginManifest(
        id="executor_probe",
        display_name="Executor Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self, executor):
        self.executor = executor

    def declare_capabilities(self):
        return (
            Capability(
                id="probe.execute",
                owner="executor_probe",
                description="Execute probe.",
                domains=frozenset({"probe"}),
                operation_types=frozenset({"probe.execute"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"probe.result"}),
                executor=self.executor.id,
                side_effecting=False,
            ),
        )

    def get_executors(self):
        return (self.executor,)


async def _execute(handler):
    executor = EvidenceWrappingExecutor(
        id="executor_probe.execute",
        owner="executor_probe",
        capability_ids=frozenset({"probe.execute"}),
        evidence_kind="probe.result",
        handler=handler,
        metadata={"adapter": "test"},
    )
    registry = ExtensionRegistry()
    registry.register(ExecutorPlugin(executor))
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="runtime-executor",
        runtime_kind="test",
        extension_registry=registry,
        runtime_store=store,
    )
    operation = Operation(
        id="op-1",
        operation_type="probe.execute",
        status=OperationStatus.RUNNING,
        request={},
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="probe.execute",
        executor_id=executor.id,
        input={"value": "alpha"},
        metadata={"owner": "executor_probe"},
    )
    await store.save_operation(operation)
    await store.save_task(task)
    return await kernel.execute_task(task.id), store


async def test_evidence_wrapping_executor_wraps_dict_payload():
    async def handler(payload):
        return {"echo": payload["value"]}

    result, store = await _execute(handler)
    snapshot = await store.inspect_operation("op-1")

    assert result.evidence[0].kind == "probe.result"
    assert result.evidence[0].owner == "executor_probe"
    assert result.evidence[0].operation_id == "op-1"
    assert result.evidence[0].task_id == "task-1"
    assert result.evidence[0].payload == {"echo": "alpha"}
    assert result.evidence[0].metadata["adapter"] == "test"
    assert snapshot.evidence == result.evidence


async def test_evidence_wrapping_executor_preserves_direct_evidence():
    def handler(_payload):
        return Evidence(kind="probe.result", payload={"direct": True})

    result, _ = await _execute(handler)

    assert result.evidence[0].payload == {"direct": True}
    assert result.evidence[0].owner == "executor_probe"
    assert result.evidence[0].operation_id == "op-1"
    assert result.evidence[0].task_id == "task-1"


async def test_evidence_wrapping_executor_preserves_multiple_evidence_records():
    async def handler(_payload):
        return [
            Evidence(kind="probe.result", payload={"index": 1}),
            Evidence(kind="probe.result", payload={"index": 2}),
        ]

    result, _ = await _execute(handler)

    assert [item.payload["index"] for item in result.evidence] == [1, 2]


async def test_evidence_wrapping_executor_exceptions_flow_through_kernel_failure():
    async def handler(_payload):
        raise ValueError("nope")

    with pytest.raises(RuntimeKernelExecutorFailed) as exc_info:
        await _execute(handler)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_sqlite_no_longer_defines_bespoke_evidence_wrapper():
    import daita.plugins.sqlite_extensions as sqlite_extensions

    assert not hasattr(sqlite_extensions, "SQLiteExecutor")
