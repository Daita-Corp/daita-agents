import pytest

from daita.core.tools import LocalTool
from daita.plugins import ExtensionRegistry
from daita.runtime import (
    AccessMode,
    InMemoryRuntimeStore,
    OperationStatus,
    RiskLevel,
    RuntimeKernel,
    register_local_tool,
)


def _tool(
    *,
    name: str = "lookup",
    output_evidence: tuple[str, ...] = ("lookup.result",),
    handler_value: str = "alpha",
) -> LocalTool:
    async def handler(arguments):
        return {"value": handler_value, "arguments": arguments}

    return LocalTool(
        name=name,
        description="Lookup a value.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        handler=handler,
        capability_ids=("custom.lookup",),
        output_evidence=output_evidence,
        access=AccessMode.METADATA_READ,
        risk=RiskLevel.MEDIUM,
        domains=("cloud",),
        operation_types=("cloud.inspect",),
        timeout_seconds=2,
        retry_safe=True,
        replay_safe=True,
        idempotent=True,
        side_effecting=False,
        metadata={"tier": "test"},
    )


async def test_local_tool_adapter_maps_metadata_to_runtime_declarations():
    registry = ExtensionRegistry()
    registration = register_local_tool(registry, _tool())

    capability = registry.get_capability(
        "custom.lookup",
        owner=registration.plugin_id,
    )

    assert capability.access is AccessMode.METADATA_READ
    assert capability.risk is RiskLevel.MEDIUM
    assert capability.domains == frozenset({"cloud"})
    assert capability.operation_types == frozenset({"cloud.inspect"})
    assert capability.timeout_seconds == 2
    assert capability.retry_safe is True
    assert capability.replay_safe is True
    assert capability.idempotent is True
    assert capability.side_effecting is False
    assert capability.metadata["tier"] == "test"
    assert registry.get_tool_view_owner("lookup") == registration.plugin_id


async def test_non_chat_runtime_executes_local_tool_through_kernel():
    registry = ExtensionRegistry()
    registration = register_local_tool(registry, _tool())
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="runtime-local",
        runtime_kind="cloud",
        extension_registry=registry,
        runtime_store=store,
    )
    operation = await kernel.create_operation(
        operation_type="cloud.inspect",
        request={"prompt": "lookup"},
    )

    result = await kernel.execute_capability(
        registration.capability_id,
        owner=registration.plugin_id,
        operation_id=operation.id,
        input={"query": "alpha"},
    )
    snapshot = await store.inspect_operation(operation.id)

    assert result.evidence[0].kind == "lookup.result"
    assert result.evidence[0].payload["value"] == "alpha"
    assert result.task.status.value == "succeeded"
    assert snapshot.operation.status is OperationStatus.RUNNING


async def test_same_local_tool_declarations_can_replace_handler_without_stale_schema():
    registry = ExtensionRegistry()
    first = register_local_tool(registry, _tool(handler_value="first"))
    second = register_local_tool(registry, _tool(handler_value="second"))
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="runtime-local",
        runtime_kind="cloud",
        extension_registry=registry,
        runtime_store=store,
    )
    operation = await kernel.create_operation(
        operation_type="cloud.inspect",
        request={"prompt": "lookup"},
    )

    result = await kernel.execute_capability(
        first.capability_id,
        owner=second.plugin_id,
        operation_id=operation.id,
        input={"query": "alpha"},
    )

    assert first.plugin_id == second.plugin_id
    assert result.evidence[0].payload["value"] == "second"


async def test_changed_local_tool_declarations_are_rejected():
    registry = ExtensionRegistry()
    register_local_tool(registry, _tool(output_evidence=("lookup.result",)))

    with pytest.raises(ValueError, match="different runtime declarations"):
        register_local_tool(registry, _tool(output_evidence=("lookup.changed",)))


def test_chat_runtime_no_longer_owns_local_tool_adapter_classes():
    import daita.agents.chat.runtime as chat_runtime

    assert not hasattr(chat_runtime, "LocalToolExecutor")
    assert not hasattr(chat_runtime, "LocalToolPlugin")
