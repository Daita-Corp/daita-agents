from __future__ import annotations

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityInputError,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)


class NoopReadExecutor:
    executor_id = "fake.read.executor"

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        return EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={"key": request.arguments["key"]},
        )


def _capability() -> Capability:
    return Capability(
        id="fake.read",
        owner="loop-lab",
        description="Read one deterministic fake value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="fake.read.result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        executor_id="fake.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


def _registry() -> CapabilityRegistry:
    return CapabilityRegistry(
        capabilities=(_capability(),),
        executors=(NoopReadExecutor(),),
        tool_views=(
            ToolView(
                name="read_fake_value",
                capability_id="fake.read",
                description="Read one fake value by key.",
            ),
        ),
    )


def test_registry_projects_only_model_visible_capability_fields() -> None:
    capability = _capability()
    executor = NoopReadExecutor()
    registry = CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_fake_value",
                capability_id=capability.id,
                description="Read one fake value by key.",
            ),
        ),
    )

    assert registry.capability("fake.read") is capability
    tool = registry.tool_definitions()[0]
    assert tool.name == "read_fake_value"
    assert tool.description == "Read one fake value by key."
    assert isinstance(tool.input_schema, FrozenJsonObject)
    assert not hasattr(tool, "executor_id")
    assert not hasattr(tool, "access_mode")
    assert not hasattr(tool, "risk")


def test_registry_validates_declared_input_shape_without_coercion() -> None:
    registry = _registry()

    validated = registry.validate_arguments("fake.read", {"key": "alpha"})
    assert isinstance(validated, FrozenJsonObject)
    assert validated.to_dict() == {"key": "alpha"}

    with pytest.raises(CapabilityInputError, match="required.*key"):
        registry.validate_arguments("fake.read", {})

    with pytest.raises(CapabilityInputError, match="string"):
        registry.validate_arguments("fake.read", {"key": 42})

    with pytest.raises(CapabilityInputError, match="unexpected"):
        registry.validate_arguments("fake.read", {"key": "alpha", "extra": True})


def test_registry_rejects_duplicate_identity_and_unknown_capabilities() -> None:
    capability = _capability()
    with pytest.raises(ValueError, match="already registered"):
        CapabilityRegistry(
            capabilities=(capability, capability),
            executors=(NoopReadExecutor(),),
        )

    registry = _registry()
    with pytest.raises(KeyError, match="unknown capability"):
        registry.capability("missing")


def test_registry_rejects_undeclared_or_ambiguous_runtime_identity() -> None:
    with pytest.raises(ValueError, match="missing executor"):
        CapabilityRegistry(capabilities=(_capability(),))

    executor = NoopReadExecutor()
    with pytest.raises(ValueError, match="executor already registered"):
        CapabilityRegistry(executors=(executor, executor))

    with pytest.raises(ValueError, match="missing capability"):
        CapabilityRegistry(
            tool_views=(
                ToolView(
                    name="missing",
                    capability_id="missing",
                    description="Missing capability view.",
                ),
            )
        )
