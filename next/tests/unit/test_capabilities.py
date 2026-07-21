from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
import hashlib
from typing import cast

import pytest

from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityInputError,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    ExecutorKnownNoEffectError,
    RiskLevel,
    ToolApplicability,
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
    assert registry.capability_ids == frozenset({"fake.read"})
    tool = registry.tool_definitions()[0]
    assert tool.name == "read_fake_value"
    assert tool.description == "Read one fake value by key."
    assert isinstance(tool.input_schema, FrozenJsonObject)
    assert not hasattr(tool, "executor_id")
    assert not hasattr(tool, "access_mode")
    assert not hasattr(tool, "risk")


def test_tool_applicability_is_immutable_and_defaults_to_a_global_view() -> None:
    first = ToolView(
        name="read_fake_value",
        capability_id="fake.read",
        description="Read one fake value by key.",
    )
    second = ToolView(
        name="read_fake_value_again",
        capability_id="fake.read",
        description="Read one fake value by key.",
    )

    assert first.applicability == ToolApplicability()
    assert first.applicability is not second.applicability
    assert first.applicability.source_adapter_ids == ()
    assert first.applicability.minimum_active_sources == 0
    assert first.applicability.required_configuration_flags == ()

    declared = ToolApplicability(
        source_adapter_ids=cast(
            tuple[str, ...],
            ["custom-adapter", "sqlite"],
        ),
        minimum_active_sources=2,
        required_configuration_flags=cast(
            tuple[str, ...],
            ["write_access", "catalog_access"],
        ),
    )
    assert declared.source_adapter_ids == ("custom-adapter", "sqlite")
    assert declared.required_configuration_flags == (
        "write_access",
        "catalog_access",
    )
    with pytest.raises(FrozenInstanceError):
        declared.minimum_active_sources = 1  # type: ignore[misc]

    with pytest.raises(TypeError, match="must be a ToolApplicability"):
        replace(first, applicability=cast(ToolApplicability, object()))


def test_tool_applicability_rejects_unbounded_or_ambiguous_declarations() -> None:
    with pytest.raises(TypeError, match="source_adapter_ids must be a sequence"):
        ToolApplicability(
            source_adapter_ids=cast(tuple[str, ...], "sqlite"),
        )
    with pytest.raises(ValueError, match="source_adapter_ids exceeds 32 items"):
        ToolApplicability(
            source_adapter_ids=tuple(f"adapter-{index}" for index in range(33)),
        )
    with pytest.raises(ValueError, match="source_adapter_ids must be unique"):
        ToolApplicability(source_adapter_ids=("sqlite", "sqlite"))
    with pytest.raises(ValueError, match="surrounding whitespace"):
        ToolApplicability(source_adapter_ids=(" sqlite",))
    with pytest.raises(ValueError, match="exceed 128 characters"):
        ToolApplicability(required_configuration_flags=("x" * 129,))
    with pytest.raises(ValueError, match="minimum_active_sources"):
        ToolApplicability(minimum_active_sources=cast(int, True))
    with pytest.raises(ValueError, match="minimum_active_sources"):
        ToolApplicability(minimum_active_sources=-1)
    with pytest.raises(ValueError, match="minimum_active_sources"):
        ToolApplicability(minimum_active_sources=65)


def test_registry_validates_declared_input_shape_without_coercion() -> None:
    registry = _registry()

    validated = registry.validate_arguments("fake.read", {"key": "alpha"})
    assert isinstance(validated, FrozenJsonObject)
    assert validated.to_dict() == {"key": "alpha"}

    with pytest.raises(CapabilityInputError) as missing:
        registry.validate_arguments("fake.read", {})
    assert missing.value.code == "capability.input.missing_required_fields"
    assert missing.value.details == FrozenJsonObject.from_mapping(
        {
            "allowed_fields": ["key"],
            "details_truncated": False,
            "missing_fields": ["key"],
        }
    )

    with pytest.raises(CapabilityInputError) as wrong_type:
        registry.validate_arguments("fake.read", {"key": 42, "secret": "not-read"})
    assert wrong_type.value.code == "capability.input.unexpected_fields"
    assert "not-read" not in canonical_json(wrong_type.value.details)

    with pytest.raises(CapabilityInputError) as unexpected:
        registry.validate_arguments("fake.read", {"key": "alpha", "extra": True})
    assert unexpected.value.code == "capability.input.unexpected_fields"
    assert unexpected.value.details == FrozenJsonObject.from_mapping(
        {
            "allowed_fields": ["key"],
            "details_truncated": False,
            "unexpected_fields": ["extra"],
        }
    )

    with pytest.raises(CapabilityInputError) as wrong_type:
        registry.validate_arguments("fake.read", {"key": 42})
    assert wrong_type.value.code == "capability.input.type_mismatch"
    assert wrong_type.value.details == FrozenJsonObject.from_mapping(
        {
            "actual_type": "integer",
            "details_truncated": False,
            "expected_type": "string",
            "field_path": "$.key",
        }
    )


def test_string_enum_is_declared_projected_and_validated_by_the_schema_owner() -> None:
    capability = replace(
        _capability(),
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["strict", "stringify_integral"],
                },
            },
            "required": ["key", "mode"],
            "additionalProperties": False,
        },
    )
    registry = CapabilityRegistry(
        capabilities=(capability,),
        executors=(NoopReadExecutor(),),
        tool_views=(
            ToolView(
                name="read_fake_value",
                capability_id=capability.id,
                description="Read one fake value by key.",
            ),
        ),
    )

    assert (
        registry.validate_arguments(
            capability.id,
            {"key": "alpha", "mode": "strict"},
        )["mode"]
        == "strict"
    )
    input_schema = registry.tool_definition("read_fake_value").input_schema
    assert isinstance(input_schema, FrozenJsonObject)
    projected = input_schema.to_dict()
    properties = projected["properties"]
    assert isinstance(properties, dict)
    mode = properties["mode"]
    assert isinstance(mode, dict)
    assert mode["enum"] == [
        "strict",
        "stringify_integral",
    ]

    with pytest.raises(CapabilityInputError) as mismatch:
        registry.validate_arguments(
            capability.id,
            {"key": "alpha", "mode": "coerce_everything"},
        )
    assert mismatch.value.code == "capability.input.enum_mismatch"
    assert mismatch.value.details == FrozenJsonObject.from_mapping(
        {
            "allowed_values": ["strict", "stringify_integral"],
            "details_truncated": False,
            "field_path": "$.mode",
        }
    )
    assert "coerce_everything" not in canonical_json(mismatch.value.details)


@pytest.mark.parametrize(
    "property_schema",
    (
        {"type": "integer", "enum": ["strict"]},
        {"type": "string", "enum": []},
        {"type": "string", "enum": ["strict", "strict"]},
        {"type": "string", "enum": ["x" * 129]},
        {"type": "string", "enum": [f"mode-{index}" for index in range(33)]},
    ),
)
def test_string_enum_declarations_fail_closed(property_schema: object) -> None:
    with pytest.raises(ValueError, match="enum"):
        replace(
            _capability(),
            input_schema={
                "type": "object",
                "properties": {"key": property_schema},
                "required": ["key"],
                "additionalProperties": False,
            },
        )


def test_capability_input_error_details_are_deterministic_and_bounded() -> None:
    fields = [f"field-{index:02d}-" + ("x" * 120) for index in range(40)]
    error = CapabilityInputError(
        "capability.input.unexpected_fields",
        "Capability input contains unexpected fields.",
        {"allowed_fields": fields, "unexpected_fields": tuple(reversed(fields))},
    )

    assert error.details["details_truncated"] is True
    assert len(canonical_json(error.details)) <= 1_536
    allowed_fields_value = error.details["allowed_fields"]
    assert isinstance(allowed_fields_value, tuple)
    allowed_fields = cast(tuple[str, ...], allowed_fields_value)
    assert allowed_fields == tuple(sorted(allowed_fields))


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


def test_capability_contract_fingerprint_is_stable_and_execution_sensitive() -> None:
    capability = _capability()

    assert capability.contract_fingerprint.startswith("sha256:")
    assert len(capability.contract_fingerprint) == 71
    assert capability.contract_fingerprint == _capability().contract_fingerprint
    assert (
        replace(capability, executor_id="fake.read.executor.v2").contract_fingerprint
        != capability.contract_fingerprint
    )


def test_required_evidence_kinds_are_typed_and_extend_only_opted_in_fingerprints() -> (
    None
):
    capability = _capability()
    legacy_material = {
        "access_mode": capability.access_mode.value,
        "executor_id": capability.executor_id,
        "id": capability.id,
        "idempotent": capability.idempotent,
        "input_schema": capability.input_schema,
        "output_evidence_kind": capability.output_evidence_kind,
        "output_schema": capability.output_schema,
        "output_schema_version": capability.output_schema_version,
        "owner": capability.owner,
        "replay_safe": capability.replay_safe,
        "risk": capability.risk.value,
        "side_effecting": capability.side_effecting,
    }
    legacy_fingerprint = (
        "sha256:"
        + hashlib.sha256(canonical_json(legacy_material).encode("utf-8")).hexdigest()
    )

    assert capability.required_evidence_kinds == ()
    assert capability.contract_fingerprint == legacy_fingerprint
    governed = replace(
        capability,
        required_evidence_kinds=("data.sqlite.update-impact.v1",),
    )
    assert governed.contract_fingerprint != legacy_fingerprint
    assert governed.required_evidence_kinds == ("data.sqlite.update-impact.v1",)

    with pytest.raises(TypeError, match="sequence"):
        invalid_required_evidence: object = "not-a-sequence"
        replace(
            capability,
            required_evidence_kinds=cast(
                tuple[str, ...],
                invalid_required_evidence,
            ),
        )
    with pytest.raises(ValueError, match="unique"):
        replace(capability, required_evidence_kinds=("impact", "impact"))


def _execution_request(
    *,
    executor_id: str = "fake.read.executor",
    fencing_token: int = 7,
    idempotency_key: str | None = "operation-1:task-1",
) -> ExecutionRequest:
    return ExecutionRequest(
        operation_id="operation-1",
        task_id="task-1",
        turn_id="turn-1",
        capability_id="fake.read",
        executor_id=executor_id,
        attempt=2,
        fencing_token=fencing_token,
        idempotency_key=idempotency_key,
        arguments={"key": "alpha"},
    )


def test_execution_request_carries_committed_fence_and_executor_identity() -> None:
    request = _execution_request()

    assert request.executor_id == "fake.read.executor"
    assert request.attempt == 2
    assert request.fencing_token == 7
    assert request.idempotency_key == "operation-1:task-1"
    assert isinstance(request.arguments, FrozenJsonObject)


@pytest.mark.parametrize(
    ("factory", "match"),
    (
        (lambda: _execution_request(executor_id=""), "executor"),
        (lambda: _execution_request(fencing_token=0), "fencing"),
        (lambda: _execution_request(fencing_token=True), "fencing"),
        (lambda: _execution_request(idempotency_key=""), "idempotency"),
    ),
)
def test_execution_request_rejects_invalid_committed_identity(
    factory: Callable[[], ExecutionRequest],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def test_replay_safe_capability_must_also_be_idempotent() -> None:
    with pytest.raises(ValueError, match="replay-safe.*idempotent"):
        replace(_capability(), idempotent=False)


def test_executor_known_no_effect_error_has_a_stable_bounded_code() -> None:
    error = ExecutorKnownNoEffectError(
        "write_precondition_changed",
        "The checked write precondition changed before commit.",
    )

    assert error.code == "write_precondition_changed"
    assert str(error) == "The checked write precondition changed before commit."
    with pytest.raises(ValueError, match="lowercase identifier"):
        ExecutorKnownNoEffectError("Write Failed", "No effect.")
    with pytest.raises(ValueError, match="message.*bounded"):
        ExecutorKnownNoEffectError("write_failed", "x" * 513)
