"""Minimal immutable capability, executor, and model-tool contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Protocol

from ._json import FrozenJsonObject, canonical_json
from .llm.models import ToolDefinition


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CapabilityInputError(ValueError):
    """Raised when proposed arguments violate a capability schema."""


class CapabilityExecutionError(RuntimeError):
    """Raised after a capability task records an execution failure."""


class EvidenceValidationError(RuntimeError):
    """Raised after executor output fails the declared evidence contract."""


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    owner: str
    description: str
    input_schema: Mapping[str, object]
    output_evidence_kind: str
    output_schema_version: int
    output_schema: Mapping[str, object]
    executor_id: str
    access_mode: AccessMode
    risk: RiskLevel
    side_effecting: bool
    idempotent: bool
    replay_safe: bool

    def __post_init__(self) -> None:
        _required_text(self.id, "capability id")
        _required_text(self.owner, "capability owner")
        _required_text(self.description, "capability description")
        _required_text(self.output_evidence_kind, "output evidence kind")
        _required_text(self.executor_id, "executor id")
        if (
            not isinstance(self.output_schema_version, int)
            or isinstance(self.output_schema_version, bool)
            or self.output_schema_version < 1
        ):
            raise ValueError("output_schema_version must be a positive integer")
        if not isinstance(self.access_mode, AccessMode):
            raise TypeError("capability access_mode must be an AccessMode")
        if not isinstance(self.risk, RiskLevel):
            raise TypeError("capability risk must be a RiskLevel")
        for name, value in (
            ("side_effecting", self.side_effecting),
            ("idempotent", self.idempotent),
            ("replay_safe", self.replay_safe),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"capability {name} must be a boolean")
        if self.access_mode is AccessMode.READ and self.side_effecting:
            raise ValueError("read capabilities cannot declare side effects")
        if self.replay_safe and not self.idempotent:
            raise ValueError("replay-safe capabilities must be idempotent")

        input_schema = FrozenJsonObject.from_mapping(self.input_schema)
        output_schema = FrozenJsonObject.from_mapping(self.output_schema)
        _validate_supported_object_schema(input_schema, "input")
        _validate_supported_object_schema(output_schema, "output")
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", output_schema)

    @property
    def contract_fingerprint(self) -> str:
        """Return the stable hash of this capability's execution contract."""

        encoded = canonical_json(
            {
                "access_mode": self.access_mode.value,
                "executor_id": self.executor_id,
                "id": self.id,
                "idempotent": self.idempotent,
                "input_schema": self.input_schema,
                "output_evidence_kind": self.output_evidence_kind,
                "output_schema": self.output_schema,
                "output_schema_version": self.output_schema_version,
                "owner": self.owner,
                "replay_safe": self.replay_safe,
                "risk": self.risk.value,
                "side_effecting": self.side_effecting,
            }
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolView:
    """A bounded model-facing projection over one runtime capability."""

    name: str
    capability_id: str
    description: str

    def __post_init__(self) -> None:
        _required_text(self.name, "tool view name")
        _required_text(self.capability_id, "tool view capability_id")
        _required_text(self.description, "tool view description")


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    operation_id: str
    task_id: str
    turn_id: str
    capability_id: str
    executor_id: str
    attempt: int
    fencing_token: int
    idempotency_key: str | None = None
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "execution operation_id")
        _required_text(self.task_id, "execution task_id")
        _required_text(self.turn_id, "execution turn_id")
        _required_text(self.capability_id, "execution capability_id")
        _required_text(self.executor_id, "execution executor_id")
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt < 1
        ):
            raise ValueError("execution attempt must be a positive integer")
        if (
            not isinstance(self.fencing_token, int)
            or isinstance(self.fencing_token, bool)
            or self.fencing_token < 1
        ):
            raise ValueError("execution fencing_token must be a positive integer")
        if self.idempotency_key is not None:
            _required_text(self.idempotency_key, "execution idempotency_key")
        object.__setattr__(
            self,
            "arguments",
            FrozenJsonObject.from_mapping(self.arguments),
        )


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    """Untrusted immutable bytes for runtime-owned durable materialization."""

    content: bytes
    media_type: str
    sensitivity_class: str
    retention_class: str
    encryption_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("evidence artifact content must be bytes")
        _required_text(self.media_type, "evidence artifact media_type")
        _required_text(
            self.sensitivity_class,
            "evidence artifact sensitivity_class",
        )
        _required_text(self.retention_class, "evidence artifact retention_class")
        object.__setattr__(
            self,
            "encryption_metadata",
            FrozenJsonObject.from_mapping(self.encryption_metadata),
        )


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """Untrusted executor output; the runtime supplies all authoritative IDs."""

    kind: str
    schema_version: int
    payload: Mapping[str, object] = field(default_factory=dict)
    artifact: EvidenceArtifact | None = None

    def __post_init__(self) -> None:
        _required_text(self.kind, "evidence candidate kind")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version < 1
        ):
            raise ValueError("evidence schema_version must be a positive integer")
        if self.artifact is not None and not isinstance(
            self.artifact,
            EvidenceArtifact,
        ):
            raise TypeError("evidence artifact must be an EvidenceArtifact or None")
        object.__setattr__(
            self,
            "payload",
            FrozenJsonObject.from_mapping(self.payload),
        )


class Executor(Protocol):
    @property
    def executor_id(self) -> str: ...

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate: ...


@dataclass(frozen=True, slots=True)
class ExtensionDeclarations:
    """Immutable capability identities advertised by one extension boundary."""

    capabilities: tuple[Capability, ...] = ()
    executor_ids: tuple[str, ...] = ()
    tool_views: tuple[ToolView, ...] = ()

    def __post_init__(self) -> None:
        capabilities = tuple(self.capabilities)
        executor_ids = tuple(self.executor_ids)
        tool_views = tuple(self.tool_views)
        if any(not isinstance(item, Capability) for item in capabilities):
            raise TypeError("extension capabilities must contain Capability records")
        if any(not isinstance(item, ToolView) for item in tool_views):
            raise TypeError("extension tool_views must contain ToolView records")
        for executor_id in executor_ids:
            _required_text(executor_id, "extension executor id")
        capability_ids = tuple(item.id for item in capabilities)
        view_names = tuple(item.name for item in tool_views)
        for values, field_name in (
            (capability_ids, "capability ids"),
            (executor_ids, "executor ids"),
            (view_names, "tool view names"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"extension declarations have duplicate {field_name}")
        if {item.executor_id for item in capabilities} != set(executor_ids):
            raise ValueError(
                "extension executor ids must exactly match declared capabilities"
            )
        if any(item.capability_id not in capability_ids for item in tool_views):
            raise ValueError("extension tool view references an undeclared capability")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "executor_ids", executor_ids)
        object.__setattr__(self, "tool_views", tool_views)


@dataclass(frozen=True, slots=True)
class _Registration:
    capability: Capability
    executor: Executor


class CapabilityRegistry:
    """An immutable declaration registry with bounded model projections."""

    def __init__(
        self,
        *,
        capabilities: Iterable[Capability] = (),
        executors: Iterable[Executor] = (),
        tool_views: Iterable[ToolView] = (),
    ) -> None:
        executor_by_id: dict[str, Executor] = {}
        for executor in executors:
            executor_id = getattr(executor, "executor_id", None)
            if not isinstance(executor_id, str) or not executor_id.strip():
                raise ValueError("executor id must be a non-empty string")
            if not callable(getattr(executor, "execute", None)):
                raise TypeError("executor must provide async execute(request)")
            if executor_id in executor_by_id:
                raise ValueError(f"executor already registered: {executor_id}")
            executor_by_id[executor_id] = executor

        registration_by_id: dict[str, _Registration] = {}
        for capability in capabilities:
            if not isinstance(capability, Capability):
                raise TypeError("capabilities must contain Capability records")
            if capability.id in registration_by_id:
                raise ValueError(f"capability already registered: {capability.id}")
            try:
                executor = executor_by_id[capability.executor_id]
            except KeyError as error:
                raise ValueError(
                    f"capability {capability.id} references missing executor "
                    f"{capability.executor_id}"
                ) from error
            registration_by_id[capability.id] = _Registration(
                capability=capability,
                executor=executor,
            )

        view_by_name: dict[str, ToolView] = {}
        for view in tool_views:
            if not isinstance(view, ToolView):
                raise TypeError("tool_views must contain ToolView records")
            if view.name in view_by_name:
                raise ValueError(f"tool view already registered: {view.name}")
            if view.capability_id not in registration_by_id:
                raise ValueError(
                    f"tool view {view.name} references missing capability "
                    f"{view.capability_id}"
                )
            view_by_name[view.name] = view

        self._registrations = registration_by_id
        self._executors = executor_by_id
        self._tool_views = view_by_name

    @property
    def capability_ids(self) -> frozenset[str]:
        """Return the immutable semantic IDs available to bounded selectors."""

        return frozenset(self._registrations)

    def capability(self, capability_id: str) -> Capability:
        try:
            return self._registrations[capability_id].capability
        except KeyError as error:
            raise KeyError(f"unknown capability: {capability_id}") from error

    def resolve_tool(self, name: str) -> tuple[ToolView, Capability]:
        try:
            view = self._tool_views[name]
        except KeyError as error:
            raise KeyError(f"unknown tool view: {name}") from error
        return view, self.capability(view.capability_id)

    def tool_definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            self.tool_definition(view.name) for view in self._tool_views.values()
        )

    def tool_definition(self, name: str) -> ToolDefinition:
        view, capability = self.resolve_tool(name)
        return ToolDefinition(
            name=view.name,
            description=view.description,
            input_schema=capability.input_schema,
        )

    def validate_arguments(
        self,
        capability_id: str,
        arguments: Mapping[str, object],
    ) -> FrozenJsonObject:
        capability = self.capability(capability_id)
        frozen = FrozenJsonObject.from_mapping(arguments)
        _validate_object(capability.input_schema, frozen, CapabilityInputError)
        return frozen

    def validate_declarations(self, declarations: ExtensionDeclarations) -> None:
        """Admit an extension only when its advertised contracts are registered."""

        if not isinstance(declarations, ExtensionDeclarations):
            raise TypeError("declarations must be ExtensionDeclarations")
        for capability in declarations.capabilities:
            try:
                registered = self.capability(capability.id)
            except KeyError as error:
                raise ValueError(
                    f"extension capability is not registered: {capability.id}"
                ) from error
            if registered != capability:
                raise ValueError(
                    f"extension capability contract differs from registry: "
                    f"{capability.id}"
                )
        for executor_id in declarations.executor_ids:
            if executor_id not in self._executors:
                raise ValueError(f"extension executor is not registered: {executor_id}")
        for view in declarations.tool_views:
            registered_view = self._tool_views.get(view.name)
            if registered_view != view:
                raise ValueError(
                    f"extension tool view differs from registry: {view.name}"
                )

    def validate_evidence(
        self,
        capability_id: str,
        candidate: object,
    ) -> EvidenceCandidate:
        capability = self.capability(capability_id)
        if not isinstance(candidate, EvidenceCandidate):
            raise EvidenceValidationError("executor output is not EvidenceCandidate")
        if candidate.kind != capability.output_evidence_kind:
            raise EvidenceValidationError(
                f"evidence kind {candidate.kind} does not match "
                f"{capability.output_evidence_kind}"
            )
        if candidate.schema_version != capability.output_schema_version:
            raise EvidenceValidationError(
                f"evidence schema_version {candidate.schema_version} does not match "
                f"{capability.output_schema_version}"
            )
        _validate_object(
            capability.output_schema,
            candidate.payload,
            EvidenceValidationError,
        )
        return candidate

    def resolve_execution(self, capability_id: str) -> tuple[Capability, Executor]:
        try:
            registration = self._registrations[capability_id]
        except KeyError as error:
            raise KeyError(f"unknown capability: {capability_id}") from error
        actual_id = getattr(registration.executor, "executor_id", None)
        if actual_id != registration.capability.executor_id:
            raise ValueError(
                f"executor identity changed for capability {capability_id}"
            )
        return registration.capability, registration.executor


_SUPPORTED_PROPERTY_TYPES = {
    "array",
    "boolean",
    "integer",
    "number",
    "object",
    "string",
}
_ROOT_SCHEMA_KEYS = {"type", "properties", "required", "additionalProperties"}
_PROPERTY_SCHEMA_KEYS = {"type"}


def _validate_supported_object_schema(
    schema: FrozenJsonObject,
    direction: str,
) -> None:
    value = schema.to_dict()
    unsupported = sorted(set(value) - _ROOT_SCHEMA_KEYS)
    if unsupported:
        raise ValueError(
            f"unsupported {direction} schema keyword: {', '.join(unsupported)}"
        )
    if value.get("type") != "object":
        raise ValueError(f"capability {direction} schema must have type object")
    properties = value.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(f"capability {direction} schema properties must be an object")
    required = value.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(name, str) for name in required
    ):
        raise ValueError(
            f"capability {direction} schema required must be a string array"
        )
    if len(required) != len(set(required)):
        raise ValueError(f"capability {direction} schema has duplicate requirements")
    if not set(required).issubset(properties):
        raise ValueError(
            f"capability {direction} schema requires an undeclared property"
        )
    additional = value.get("additionalProperties", True)
    if not isinstance(additional, bool):
        raise ValueError("additionalProperties must be a boolean")
    for name, property_schema in properties.items():
        if not isinstance(name, str) or not isinstance(property_schema, dict):
            raise ValueError("capability property schemas must be objects")
        unsupported_property = sorted(set(property_schema) - _PROPERTY_SCHEMA_KEYS)
        if unsupported_property:
            raise ValueError(
                f"unsupported {direction} property schema keyword: "
                f"{', '.join(unsupported_property)}"
            )
        property_type = property_schema.get("type")
        if property_type not in _SUPPORTED_PROPERTY_TYPES:
            raise ValueError(f"unsupported capability property type: {property_type}")


def _validate_object(
    schema: Mapping[str, object],
    value: Mapping[str, object],
    error_type: type[ValueError] | type[RuntimeError],
) -> None:
    schema_value = FrozenJsonObject.from_mapping(schema).to_dict()
    object_value = FrozenJsonObject.from_mapping(value).to_dict()
    properties = schema_value.get("properties", {})
    required = schema_value.get("required", [])
    additional = schema_value.get("additionalProperties", True)
    assert isinstance(properties, dict)
    assert isinstance(required, list)
    assert isinstance(additional, bool)

    missing = [name for name in required if name not in object_value]
    if missing:
        raise error_type(f"required field is missing: {', '.join(missing)}")
    unexpected = sorted(set(object_value) - set(properties))
    if unexpected and not additional:
        raise error_type(f"unexpected field: {', '.join(unexpected)}")
    for name, item in object_value.items():
        property_schema = properties.get(name)
        if property_schema is None:
            continue
        assert isinstance(property_schema, dict)
        expected = property_schema["type"]
        if not _matches_json_type(item, expected):
            raise error_type(f"field {name} must be {expected}")


def _matches_json_type(value: object, expected: object) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False
