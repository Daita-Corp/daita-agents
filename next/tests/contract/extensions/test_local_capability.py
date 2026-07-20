from __future__ import annotations

from collections.abc import Mapping

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
)
from daita.extensions import LocalCapability, tool

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"key": {"type": "string"}},
    "required": ["key"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def _request() -> ExecutionRequest:
    return ExecutionRequest(
        operation_id="operation-1",
        task_id="task-1",
        turn_id="turn-1",
        capability_id="local.lookup",
        executor_id="local.lookup.executor",
        attempt=1,
        fencing_token=1,
        arguments={"key": "alpha"},
    )


async def test_tool_declares_a_capability_executor_and_bounded_tool_view() -> None:
    calls: list[Mapping[str, object]] = []

    @tool(
        id="local.lookup",
        owner="local",
        name="lookup_value",
        description="Look up one local value.",
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
    )
    async def lookup(arguments: Mapping[str, object]) -> Mapping[str, object]:
        calls.append(arguments)
        return {"value": f"value:{arguments['key']}"}

    assert isinstance(lookup, LocalCapability)
    assert calls == []
    declarations = lookup.declarations()
    assert declarations.capabilities == (lookup.capability,)
    assert declarations.executor_ids == (lookup.executor.executor_id,)
    assert declarations.tool_views == (lookup.tool_view,)

    registry = CapabilityRegistry(
        capabilities=(lookup.capability,),
        executors=(lookup.executor,),
        tool_views=(lookup.tool_view,),
    )
    projected = registry.tool_definition("lookup_value")
    assert projected.name == "lookup_value"
    assert projected.description == "Look up one local value."
    assert projected.input_schema == lookup.capability.input_schema
    assert not hasattr(projected, "executor_id")
    assert not hasattr(projected, "handler")
    assert calls == []

    capability, executor = registry.resolve_execution("local.lookup")
    candidate = await executor.execute(_request())
    accepted = registry.validate_evidence(capability.id, candidate)

    assert len(calls) == 1
    assert isinstance(calls[0], FrozenJsonObject)
    assert calls[0].to_dict() == {"key": "alpha"}
    assert accepted == EvidenceCandidate(
        kind="local.lookup.result",
        schema_version=1,
        payload={"value": "value:alpha"},
    )


def test_tool_requires_explicit_contracts_and_preserves_governance_facts() -> None:
    def update(arguments: Mapping[str, object]) -> Mapping[str, object]:
        return {"value": str(arguments["key"])}

    declared = tool(
        update,
        id="local.lookup",
        owner="local",
        name="lookup_value",
        description="Look up one local value.",
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=False,
        replay_safe=False,
    )

    assert isinstance(declared, LocalCapability)
    assert declared.capability.access_mode is AccessMode.WRITE
    assert declared.capability.risk is RiskLevel.HIGH
    assert declared.capability.side_effecting is True
    assert declared.capability.idempotent is False
    assert declared.capability.replay_safe is False

    with pytest.raises(ValueError, match="read capabilities cannot declare"):
        tool(
            update,
            id="local.invalid",
            owner="local",
            name="invalid",
            description="Invalid local declaration.",
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
            side_effecting=True,
        )


async def test_local_executor_rejects_non_mapping_results_without_coercion() -> None:
    @tool(
        id="local.lookup",
        owner="local",
        name="lookup_value",
        description="Look up one local value.",
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
    )
    async def invalid_result(arguments: Mapping[str, object]) -> object:
        del arguments
        return "not-an-evidence-object"

    with pytest.raises(TypeError, match="result must be a mapping"):
        await invalid_result.executor.execute(_request())
