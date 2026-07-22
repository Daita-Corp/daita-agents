"""Small tool declaration and execution contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Protocol, TypeVar

from ._json import FrozenJsonObject
from .llm.models import ToolDefinition

_T = TypeVar("_T")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    run_id: str
    call_id: str
    tool_name: str
    capability_id: str
    arguments: FrozenJsonObject
    reason: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.run_id, "approval run_id"),
            (self.call_id, "approval call_id"),
            (self.tool_name, "approval tool_name"),
            (self.capability_id, "approval capability_id"),
            (self.reason, "approval reason"),
        ):
            _text(value, name)
        object.__setattr__(
            self,
            "arguments",
            FrozenJsonObject.from_mapping(self.arguments),
        )


class ApprovalHandler(Protocol):
    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision: ...


class CapabilityInputError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        _text(code, "tool input error code")
        _text(message, "tool input error message")
        self.code = code
        self.details = FrozenJsonObject.from_mapping(details or {})
        super().__init__(message)


class ToolOutputValidationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    description: str
    input_schema: Mapping[str, object]
    output_kind: str
    output_schema: Mapping[str, object]
    executor_id: str
    access_mode: AccessMode = AccessMode.READ
    side_effecting: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.id, "capability id"),
            (self.description, "capability description"),
            (self.output_kind, "capability output kind"),
            (self.executor_id, "executor id"),
        ):
            _text(value, name)
        if not isinstance(self.access_mode, AccessMode):
            raise TypeError("access_mode must be AccessMode")
        if not isinstance(self.side_effecting, bool):
            raise TypeError("side_effecting must be a boolean")
        if (self.access_mode is AccessMode.WRITE) is not self.side_effecting:
            raise ValueError(
                "write tools must be side-effecting and read tools cannot be"
            )
        input_schema = FrozenJsonObject.from_mapping(self.input_schema)
        output_schema = FrozenJsonObject.from_mapping(self.output_schema)
        _check_schema(input_schema)
        _check_schema(output_schema)
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", output_schema)


@dataclass(frozen=True, slots=True)
class ToolApplicability:
    source_adapter_ids: tuple[str, ...] = ()
    minimum_active_sources: int = 0
    required_configuration_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_adapter_ids", "required_configuration_flags"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} cannot contain duplicates")
            object.__setattr__(self, name, values)
        if (
            not isinstance(self.minimum_active_sources, int)
            or isinstance(self.minimum_active_sources, bool)
            or self.minimum_active_sources < 0
        ):
            raise ValueError("minimum_active_sources must be non-negative")


@dataclass(frozen=True, slots=True)
class ToolView:
    name: str
    capability_id: str
    description: str
    applicability: ToolApplicability = field(default_factory=ToolApplicability)

    def __post_init__(self) -> None:
        for value, name in (
            (self.name, "tool name"),
            (self.capability_id, "tool capability_id"),
            (self.description, "tool description"),
        ):
            _text(value, name)
        if not isinstance(self.applicability, ToolApplicability):
            raise TypeError("applicability must be ToolApplicability")


@dataclass(frozen=True, slots=True)
class ToolExecution:
    run_id: str
    capability_id: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.run_id, "tool run_id")
        _text(self.capability_id, "tool capability_id")
        object.__setattr__(
            self, "arguments", FrozenJsonObject.from_mapping(self.arguments)
        )


@dataclass(frozen=True, slots=True)
class ToolArtifact:
    content: bytes
    media_type: str
    sensitivity: str = "internal"
    retention: str = "run"

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("artifact content must be bytes")
        _text(self.media_type, "artifact media_type")
        _text(self.sensitivity, "artifact sensitivity")
        _text(self.retention, "artifact retention")


@dataclass(frozen=True, slots=True)
class ToolOutput:
    kind: str
    data: Mapping[str, object] = field(default_factory=dict)
    artifact: ToolArtifact | None = None

    def __post_init__(self) -> None:
        _text(self.kind, "tool output kind")
        if self.artifact is not None and not isinstance(self.artifact, ToolArtifact):
            raise TypeError("artifact must be ToolArtifact or None")
        object.__setattr__(self, "data", FrozenJsonObject.from_mapping(self.data))


class Executor(Protocol):
    @property
    def executor_id(self) -> str: ...

    async def execute(self, request: ToolExecution) -> ToolOutput: ...


class SideEffectExecutor(Executor, Protocol):
    async def preflight(self, request: ToolExecution) -> FrozenJsonObject: ...


@dataclass(frozen=True, slots=True)
class ExtensionDeclarations:
    capabilities: tuple[Capability, ...] = ()
    executor_ids: tuple[str, ...] = ()
    tool_views: tuple[ToolView, ...] = ()

    def __post_init__(self) -> None:
        capabilities = tuple(self.capabilities)
        executor_ids = tuple(self.executor_ids)
        views = tuple(self.tool_views)
        if {item.executor_id for item in capabilities} != set(executor_ids):
            raise ValueError("executor_ids must match capability executors")
        if any(
            view.capability_id not in {item.id for item in capabilities}
            for view in views
        ):
            raise ValueError("tool view references an undeclared capability")
        for values, name in (
            ((item.id for item in capabilities), "capability"),
            (executor_ids, "executor"),
            ((item.name for item in views), "tool"),
        ):
            items = tuple(values)
            if len(items) != len(set(items)):
                raise ValueError(f"duplicate {name} declaration")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "executor_ids", executor_ids)
        object.__setattr__(self, "tool_views", views)


class CapabilityRegistry:
    """Resolve model-facing tools to their declared executor."""

    def __init__(
        self,
        *,
        capabilities: Iterable[Capability] = (),
        executors: Iterable[Executor] = (),
        tool_views: Iterable[ToolView] = (),
    ) -> None:
        self._capabilities = _unique(capabilities, lambda item: item.id, "capability")
        self._executors = _unique(executors, lambda item: item.executor_id, "executor")
        self._views = _unique(tool_views, lambda item: item.name, "tool")
        for capability in self._capabilities.values():
            if capability.executor_id not in self._executors:
                raise ValueError(f"missing executor: {capability.executor_id}")
        for view in self._views.values():
            if view.capability_id not in self._capabilities:
                raise ValueError(f"missing capability: {view.capability_id}")

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._views)

    def resolve_tool(self, name: str) -> tuple[ToolView, Capability]:
        try:
            view = self._views[name]
            return view, self._capabilities[view.capability_id]
        except KeyError as error:
            raise KeyError(f"unknown tool: {name}") from error

    def tool_definition(self, name: str) -> ToolDefinition:
        view, capability = self.resolve_tool(name)
        return ToolDefinition(
            name=view.name,
            description=view.description,
            input_schema=capability.input_schema,
        )

    def validate_arguments(
        self, capability_id: str, arguments: Mapping[str, object]
    ) -> FrozenJsonObject:
        capability = self._capabilities[capability_id]
        value = FrozenJsonObject.from_mapping(arguments)
        _validate(capability.input_schema, value, CapabilityInputError)
        return value

    def resolve_execution(self, capability_id: str) -> tuple[Capability, Executor]:
        capability = self._capabilities[capability_id]
        return capability, self._executors[capability.executor_id]

    def validate_output(self, capability_id: str, output: object) -> ToolOutput:
        capability = self._capabilities[capability_id]
        if not isinstance(output, ToolOutput):
            raise ToolOutputValidationError("executor did not return ToolOutput")
        if output.kind != capability.output_kind:
            raise ToolOutputValidationError(
                f"output kind {output.kind} does not match {capability.output_kind}"
            )
        _validate(capability.output_schema, output.data, ToolOutputValidationError)
        return output

    def validate_declarations(self, declarations: ExtensionDeclarations) -> None:
        for capability in declarations.capabilities:
            if self._capabilities.get(capability.id) != capability:
                raise ValueError(f"capability declaration differs: {capability.id}")
        for executor_id in declarations.executor_ids:
            if executor_id not in self._executors:
                raise ValueError(f"executor is not registered: {executor_id}")
        for view in declarations.tool_views:
            if self._views.get(view.name) != view:
                raise ValueError(f"tool declaration differs: {view.name}")


def _unique(items: Iterable[_T], key: Callable[[_T], str], name: str) -> dict[str, _T]:
    result: dict[str, _T] = {}
    for item in items:
        item_key = key(item)
        if item_key in result:
            raise ValueError(f"duplicate {name}: {item_key}")
        result[item_key] = item
    return result


def _check_schema(schema: Mapping[str, object]) -> None:
    if schema.get("type") != "object":
        raise ValueError("tool schemas must describe an object")
    if not isinstance(schema.get("properties", {}), Mapping):
        raise ValueError("tool schema properties must be an object")
    properties = schema.get("properties", {})
    assert isinstance(properties, Mapping)
    for name, rule in properties.items():
        if not isinstance(name, str) or not isinstance(rule, Mapping):
            raise ValueError("tool schema properties must contain object rules")
        enum = rule.get("enum")
        if enum is not None and (not isinstance(enum, (tuple, list)) or not enum):
            raise ValueError("tool schema enum must be a non-empty array")
        minimum = rule.get("minLength")
        maximum = rule.get("maxLength")
        for bound, label in ((minimum, "minLength"), (maximum, "maxLength")):
            if bound is not None and (
                not isinstance(bound, int) or isinstance(bound, bool) or bound < 0
            ):
                raise ValueError(f"tool schema {label} must be non-negative")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("tool schema minLength cannot exceed maxLength")
        pattern = rule.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str):
                raise ValueError("tool schema pattern must be a string")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError("tool schema pattern must be valid") from error


def _validate(
    schema: Mapping[str, object],
    value: Mapping[str, object],
    error_type: type[ValueError] | type[RuntimeError],
) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", ())
    additional = schema.get("additionalProperties", True)
    assert isinstance(properties, Mapping)
    if not isinstance(required, (tuple, list)):
        raise ValueError("tool schema required must be an array")
    if any(not isinstance(name, str) for name in required):
        raise ValueError("tool schema required must contain strings")
    names = tuple(name for name in required if isinstance(name, str))
    missing = [name for name in names if name not in value]
    if missing:
        if error_type is CapabilityInputError:
            raise CapabilityInputError(
                "missing_arguments",
                "Missing required tool arguments.",
                {"names": missing},
            )
        raise error_type(f"missing output fields: {', '.join(missing)}")
    if additional is False:
        unexpected = [name for name in value if name not in properties]
        if unexpected:
            if error_type is CapabilityInputError:
                raise CapabilityInputError(
                    "unexpected_arguments",
                    "Unexpected tool arguments.",
                    {"names": unexpected},
                )
            raise error_type(f"unexpected output fields: {', '.join(unexpected)}")
    for name, item in value.items():
        rule = properties.get(name)
        if not isinstance(rule, Mapping):
            continue
        expected = rule.get("type")
        if expected is not None and not _matches_type(item, expected):
            if error_type is CapabilityInputError:
                raise CapabilityInputError(
                    "invalid_argument_type",
                    f"Tool argument {name} must be {expected}.",
                    {"name": name, "expected": expected},
                )
            raise error_type(f"output field {name} must be {expected}")
        enum = rule.get("enum")
        if isinstance(enum, (tuple, list)) and item not in enum:
            _constraint_error(
                error_type,
                name,
                "one of the declared enum values",
                "enum",
            )
        if isinstance(item, str):
            minimum = rule.get("minLength")
            maximum = rule.get("maxLength")
            pattern = rule.get("pattern")
            if isinstance(minimum, int) and len(item) < minimum:
                _constraint_error(
                    error_type, name, f"at least {minimum} characters", "minLength"
                )
            if isinstance(maximum, int) and len(item) > maximum:
                _constraint_error(
                    error_type, name, f"at most {maximum} characters", "maxLength"
                )
            if isinstance(pattern, str) and re.search(pattern, item) is None:
                _constraint_error(error_type, name, "the declared pattern", "pattern")


def _constraint_error(
    error_type: type[ValueError] | type[RuntimeError],
    name: str,
    expected: str,
    constraint: str,
) -> None:
    if error_type is CapabilityInputError:
        raise CapabilityInputError(
            "invalid_argument_value",
            f"Tool argument {name} must satisfy {expected}.",
            {"constraint": constraint, "name": name},
        )
    raise error_type(f"output field {name} must satisfy {expected}")


def _matches_type(value: object, expected: object) -> bool:
    if not isinstance(expected, str):
        return True
    return {
        "array": isinstance(value, (tuple, list)),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": isinstance(value, Mapping),
        "string": isinstance(value, str),
    }.get(expected, True)


__all__ = [
    "AccessMode",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalRequest",
    "Capability",
    "CapabilityInputError",
    "CapabilityRegistry",
    "Executor",
    "ExtensionDeclarations",
    "SideEffectExecutor",
    "ToolApplicability",
    "ToolArtifact",
    "ToolExecution",
    "ToolOutput",
    "ToolOutputValidationError",
    "ToolView",
]
