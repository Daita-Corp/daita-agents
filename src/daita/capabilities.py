"""Define capability metadata, tool schemas, executors, and registry contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from types import MappingProxyType
from typing import Protocol, TypeVar

from ._json import FrozenJsonObject, canonical_json
from .artifacts.models import ArtifactDraft
from .llm.models import ModelSensitivity, ToolDefinition

_T = TypeVar("_T")

MAX_APPROVAL_DOCUMENT_CHARACTERS = 64 * 1_024
MAX_TOOL_DISCOVERY_SUMMARY_CHARACTERS = 256
MAX_TOOL_DISCOVERY_GUIDANCE_CHARACTERS = 512
MAX_TOOL_DISCOVERY_KEYWORDS = 16
MAX_TOOL_DISCOVERY_KEYWORD_CHARACTERS = 64
MAX_TOOL_EAGER_PRIORITY = 1_000
RESERVED_TOOL_NAMES = frozenset({"tool_search", "tool_describe", "tool_call"})
MAX_EXECUTION_SCOPE_IDENTITIES = 256
MAX_EXECUTION_SCOPE_IDENTITY_CHARACTERS = 2_048


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


class AccessMode(str, Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"


class OperationalEffect(str, Enum):
    NONE = "none"
    CHANGE_ADVISORY_CONTEXT = "change_advisory_context"
    START_JOB = "start_job"
    SUBMIT_EXECUTION_GRAPH = "submit_execution_graph"
    EXTEND_EXECUTION_GRAPH = "extend_execution_graph"
    CANCEL_JOB = "cancel_job"
    CANCEL_EXECUTION_GRAPH = "cancel_execution_graph"
    MUTATE_DATA = "mutate_data"
    CHANGE_INFRASTRUCTURE = "change_infrastructure"


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    """One immutable, digestible ceiling carried by a bounded reasoning run."""

    scope_id: str
    revision: int
    agent_id: str
    principal_id: str
    grant_id: str
    job_id: str | None
    job_revision: int | None
    allowed_source_ids: tuple[str, ...]
    allowed_resource_ids: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    allowed_access_modes: frozenset[AccessMode]
    allowed_operational_effects: frozenset[OperationalEffect]
    sensitivity_ceiling: ModelSensitivity
    eligible_model_routes: tuple[str, ...]
    per_run_max_cost_usd: Decimal
    per_run_max_tokens: int
    delivery_destination: str

    def __post_init__(self) -> None:
        for identity_value, identity_name in (
            (self.scope_id, "execution scope_id"),
            (self.agent_id, "execution scope agent_id"),
            (self.principal_id, "execution scope principal_id"),
            (self.grant_id, "execution scope grant_id"),
            (self.delivery_destination, "execution scope delivery_destination"),
        ):
            _text(identity_value, identity_name)
            if len(identity_value) > 512 or any(
                character in "\r\n\x00" for character in identity_value
            ):
                raise ValueError(f"{identity_name} must be bounded single-line text")
        if self.job_id is not None:
            _text(self.job_id, "execution scope job_id")
            if len(self.job_id) > 512 or any(
                character in "\r\n\x00" for character in self.job_id
            ):
                raise ValueError(
                    "execution scope job_id must be bounded single-line text"
                )
        if (self.job_id is None) != (self.job_revision is None):
            raise ValueError(
                "execution scope job identity and revision must be present together"
            )
        revision_values = [(self.revision, "execution scope revision")]
        if self.job_revision is not None:
            revision_values.append((self.job_revision, "execution scope job_revision"))
        for revision_value, revision_name in revision_values:
            if (
                not isinstance(revision_value, int)
                or isinstance(revision_value, bool)
                or revision_value < 1
            ):
                raise ValueError(f"{revision_name} must be positive")
        sources = _scope_identities(self.allowed_source_ids, "allowed_source_ids")
        resources = _scope_identities(
            self.allowed_resource_ids,
            "allowed_resource_ids",
        )
        capabilities = _scope_identities(
            self.allowed_capability_ids,
            "allowed_capability_ids",
        )
        routes = _scope_identities(
            self.eligible_model_routes,
            "eligible_model_routes",
        )
        if not sources or not resources or not capabilities or not routes:
            raise ValueError("execution scope identity ceilings cannot be empty")
        access_modes = frozenset(self.allowed_access_modes)
        effects = frozenset(self.allowed_operational_effects)
        if not access_modes or any(
            not isinstance(item, AccessMode) for item in access_modes
        ):
            raise ValueError("execution scope requires allowed access modes")
        if not effects or any(
            not isinstance(item, OperationalEffect) for item in effects
        ):
            raise ValueError("execution scope requires allowed operational effects")
        if not isinstance(self.sensitivity_ceiling, ModelSensitivity):
            raise TypeError("execution scope sensitivity ceiling is invalid")
        if (
            not isinstance(self.per_run_max_cost_usd, Decimal)
            or not self.per_run_max_cost_usd.is_finite()
            or self.per_run_max_cost_usd < 0
        ):
            raise ValueError(
                "execution scope per-run cost must be a finite non-negative Decimal"
            )
        if (
            not isinstance(self.per_run_max_tokens, int)
            or isinstance(self.per_run_max_tokens, bool)
            or self.per_run_max_tokens < 1
        ):
            raise ValueError("execution scope per-run tokens must be positive")
        object.__setattr__(self, "allowed_source_ids", sources)
        object.__setattr__(self, "allowed_resource_ids", resources)
        object.__setattr__(self, "allowed_capability_ids", capabilities)
        object.__setattr__(self, "eligible_model_routes", routes)
        object.__setattr__(self, "allowed_access_modes", access_modes)
        object.__setattr__(self, "allowed_operational_effects", effects)

    @property
    def digest(self) -> str:
        return (
            "sha256:"
            + sha256(
                canonical_json(
                    {
                        "scope_id": self.scope_id,
                        "revision": self.revision,
                        "agent_id": self.agent_id,
                        "principal_id": self.principal_id,
                        "grant_id": self.grant_id,
                        "job_id": self.job_id,
                        "job_revision": self.job_revision,
                        "allowed_source_ids": self.allowed_source_ids,
                        "allowed_resource_ids": self.allowed_resource_ids,
                        "allowed_capability_ids": self.allowed_capability_ids,
                        "allowed_access_modes": tuple(
                            sorted(item.value for item in self.allowed_access_modes)
                        ),
                        "allowed_operational_effects": tuple(
                            sorted(
                                item.value for item in self.allowed_operational_effects
                            )
                        ),
                        "sensitivity_ceiling": self.sensitivity_ceiling.value,
                        "eligible_model_routes": self.eligible_model_routes,
                        "per_run_max_cost_usd": str(self.per_run_max_cost_usd),
                        "per_run_max_tokens": self.per_run_max_tokens,
                        "delivery_destination": self.delivery_destination,
                    }
                ).encode("utf-8")
            ).hexdigest()
        )

    def allows(self, capability: "Capability") -> bool:
        return (
            capability.id in self.allowed_capability_ids
            and capability.access_mode in self.allowed_access_modes
            and capability.operational_effect in self.allowed_operational_effects
        )


def _scope_identities(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"execution scope {name} must be a sequence")
    items = tuple(values)
    if len(items) > MAX_EXECUTION_SCOPE_IDENTITIES:
        raise ValueError(f"execution scope {name} exceeds its bound")
    if any(
        not isinstance(item, str)
        or not item
        or item != item.strip()
        or len(item) > MAX_EXECUTION_SCOPE_IDENTITY_CHARACTERS
        or any(character in "\r\n\x00" for character in item)
        for item in items
    ):
        raise ValueError(f"execution scope {name} contains invalid text")
    if len(items) != len(set(items)):
        raise ValueError(f"execution scope {name} cannot contain duplicates")
    return tuple(sorted(items))


class ToolExposureClass(str, Enum):
    CORE = "core"
    STANDARD = "standard"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class ToolDiscoveryMetadata:
    """Bounded trusted text used only to present an admitted tool."""

    summary: str
    when_to_use: str
    keywords: tuple[str, ...]
    exposure_class: ToolExposureClass
    eager_priority: int

    def __post_init__(self) -> None:
        for value, name, maximum in (
            (
                self.summary,
                "tool discovery summary",
                MAX_TOOL_DISCOVERY_SUMMARY_CHARACTERS,
            ),
            (
                self.when_to_use,
                "tool discovery when_to_use",
                MAX_TOOL_DISCOVERY_GUIDANCE_CHARACTERS,
            ),
        ):
            _text(value, name)
            if len(value) > maximum:
                raise ValueError(f"{name} exceeds its character bound")
        keywords = tuple(self.keywords)
        if len(keywords) > MAX_TOOL_DISCOVERY_KEYWORDS:
            raise ValueError("tool discovery has too many keywords")
        if len(keywords) != len(set(keywords)):
            raise ValueError("tool discovery keywords must be distinct")
        for keyword in keywords:
            if (
                not isinstance(keyword, str)
                or not keyword
                or keyword != keyword.strip().lower()
                or len(keyword) > MAX_TOOL_DISCOVERY_KEYWORD_CHARACTERS
                or re.fullmatch(r"[a-z0-9][a-z0-9 _.-]*", keyword) is None
            ):
                raise ValueError(
                    "tool discovery keywords must be bounded normalized text"
                )
        if not isinstance(self.exposure_class, ToolExposureClass):
            raise TypeError("tool discovery exposure_class must be ToolExposureClass")
        if (
            not isinstance(self.eager_priority, int)
            or isinstance(self.eager_priority, bool)
            or not 0 <= self.eager_priority <= MAX_TOOL_EAGER_PRIORITY
        ):
            raise ValueError("tool discovery eager_priority is outside its bound")
        object.__setattr__(self, "keywords", keywords)


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

    def render_arguments_for_review(self) -> str | None:
        return render_approval_arguments(self.arguments.to_dict())


class ApprovalHandler(Protocol):
    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision: ...


def render_approval_arguments(arguments: Mapping[str, object]) -> str | None:
    """Render one exact, terminal-safe approval document within its fixed bound."""

    rendered = json.dumps(
        FrozenJsonObject.from_mapping(arguments).to_dict(),
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if len(rendered) > MAX_APPROVAL_DOCUMENT_CHARACTERS:
        return None
    return rendered


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
class ArtifactPolicy:
    """Stable capability metadata for its one optional artifact draft."""

    allowed_media_types: frozenset[str]
    allowed_extensions: tuple[tuple[str, tuple[str, ...]], ...]
    artifact_required: bool
    max_artifact_count: int
    max_bytes_per_artifact: int
    max_total_bytes_per_call: int

    def __post_init__(self) -> None:
        media_types = frozenset(self.allowed_media_types)
        if not media_types or any(
            not isinstance(item, str) or not item.strip() for item in media_types
        ):
            raise ValueError("artifact policy media types must be non-empty text")
        extensions = tuple(
            (media_type, tuple(values))
            for media_type, values in self.allowed_extensions
        )
        if {media_type for media_type, _ in extensions} != media_types:
            raise ValueError("artifact policy extensions must cover each media type")
        if len(extensions) != len(media_types):
            raise ValueError("artifact policy media type declarations cannot duplicate")
        for media_type, values in extensions:
            if not values or len(values) != len(set(values)):
                raise ValueError("artifact policy extensions must be distinct")
            if any(
                not isinstance(value, str) or not re.fullmatch(r"\.[a-z0-9]+", value)
                for value in values
            ):
                raise ValueError("artifact policy extensions must be canonical")
        if not isinstance(self.artifact_required, bool):
            raise TypeError("artifact_required must be a boolean")
        if (
            not isinstance(self.max_artifact_count, int)
            or isinstance(self.max_artifact_count, bool)
            or self.max_artifact_count not in {0, 1}
        ):
            raise ValueError("max_artifact_count must be zero or one")
        if self.artifact_required and self.max_artifact_count != 1:
            raise ValueError("a required artifact needs max_artifact_count one")
        for value, name in (
            (self.max_bytes_per_artifact, "max_bytes_per_artifact"),
            (self.max_total_bytes_per_call, "max_total_bytes_per_call"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_total_bytes_per_call < self.max_bytes_per_artifact:
            raise ValueError("per-call bytes cannot be below per-artifact bytes")
        object.__setattr__(self, "allowed_media_types", media_types)
        object.__setattr__(self, "allowed_extensions", extensions)


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    description: str
    input_schema: Mapping[str, object]
    output_kind: str
    output_schema: Mapping[str, object]
    executor_id: str
    access_mode: AccessMode = AccessMode.NONE
    operational_effect: OperationalEffect = OperationalEffect.NONE
    artifact_policy: ArtifactPolicy | None = None

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
        if not isinstance(self.operational_effect, OperationalEffect):
            raise TypeError("operational_effect must be OperationalEffect")
        if self.artifact_policy is not None and not isinstance(
            self.artifact_policy, ArtifactPolicy
        ):
            raise TypeError("artifact_policy must be ArtifactPolicy or None")
        input_schema = FrozenJsonObject.from_mapping(self.input_schema)
        output_schema = FrozenJsonObject.from_mapping(self.output_schema)
        _check_schema(input_schema)
        _check_schema(output_schema)
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", output_schema)


def capability_contract_digest(
    capability: Capability,
    *,
    domain_owner_id: str,
) -> str:
    """Return the exact immutable execution contract identity for a capability."""

    if not isinstance(capability, Capability):
        raise TypeError("capability contract requires Capability")
    _text(domain_owner_id, "capability contract domain owner")
    material = {
        "domain_owner_id": domain_owner_id,
        "capability_id": capability.id,
        "description": capability.description,
        "input_schema": capability.input_schema,
        "output_kind": capability.output_kind,
        "output_schema": capability.output_schema,
        "executor_id": capability.executor_id,
        "access_mode": capability.access_mode.value,
        "operational_effect": capability.operational_effect.value,
        "artifact_policy": (
            None
            if capability.artifact_policy is None
            else {
                "allowed_media_types": sorted(
                    capability.artifact_policy.allowed_media_types
                ),
                "allowed_extensions": capability.artifact_policy.allowed_extensions,
                "artifact_required": capability.artifact_policy.artifact_required,
                "max_artifact_count": capability.artifact_policy.max_artifact_count,
                "max_bytes_per_artifact": (
                    capability.artifact_policy.max_bytes_per_artifact
                ),
                "max_total_bytes_per_call": (
                    capability.artifact_policy.max_total_bytes_per_call
                ),
            }
        ),
    }
    return "sha256:" + sha256(canonical_json(material).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolView:
    name: str
    capability_id: str
    description: str
    discovery: ToolDiscoveryMetadata
    origin_revision_digest: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.name, "tool name"),
            (self.capability_id, "tool capability_id"),
            (self.description, "tool description"),
        ):
            _text(value, name)
        if self.name in RESERVED_TOOL_NAMES:
            raise ValueError(f"reserved runtime control tool name: {self.name}")
        if not isinstance(self.discovery, ToolDiscoveryMetadata):
            raise TypeError("tool discovery metadata is required")
        if (
            self.origin_revision_digest is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", self.origin_revision_digest)
            is None
        ):
            raise ValueError("tool origin revision digest must use sha256")


@dataclass(frozen=True, slots=True)
class ToolExecution:
    run_id: str
    call_id: str
    capability_id: str
    arguments: Mapping[str, object] = field(default_factory=dict)
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.run_id, "tool run_id")
        _text(self.call_id, "tool call_id")
        _text(self.capability_id, "tool capability_id")
        if self.conversation_id is not None:
            _text(self.conversation_id, "tool conversation_id")
        object.__setattr__(
            self, "arguments", FrozenJsonObject.from_mapping(self.arguments)
        )


@dataclass(frozen=True, slots=True)
class ToolOutput:
    kind: str
    data: Mapping[str, object] = field(default_factory=dict)
    artifact: ArtifactDraft | None = None
    sensitivity: ModelSensitivity | None = None
    sensitivity_provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.kind, "tool output kind")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactDraft):
            raise TypeError("artifact must be ArtifactDraft or None")
        if self.sensitivity is not None and not isinstance(
            self.sensitivity, ModelSensitivity
        ):
            raise TypeError("tool output sensitivity must be ModelSensitivity or None")
        provenance = FrozenJsonObject.from_mapping(self.sensitivity_provenance)
        if (self.sensitivity is None) is bool(provenance):
            raise ValueError(
                "tool output sensitivity and provenance must be present together"
            )
        object.__setattr__(self, "data", FrozenJsonObject.from_mapping(self.data))
        object.__setattr__(self, "sensitivity_provenance", provenance)


class Executor(Protocol):
    @property
    def executor_id(self) -> str: ...

    async def execute(self, request: ToolExecution) -> ToolOutput: ...


class SideEffectExecutor(Executor, Protocol):
    async def preflight(self, request: ToolExecution) -> FrozenJsonObject: ...


@dataclass(frozen=True, slots=True)
class CapabilityDeclarations:
    domain_owner_id: str
    capabilities: tuple[Capability, ...] = ()
    executor_ids: tuple[str, ...] = ()
    tool_views: tuple[ToolView, ...] = ()

    def __post_init__(self) -> None:
        _text(self.domain_owner_id, "domain owner id")
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
    """Resolve statically declared tools, capabilities, owners, and executors."""

    def __init__(
        self,
        *,
        declarations: Iterable[CapabilityDeclarations] = (),
        executors: Iterable[Executor] = (),
    ) -> None:
        declared = _unique(
            declarations,
            lambda item: item.domain_owner_id,
            "domain owner",
        )
        self._declarations = MappingProxyType(declared)
        capabilities = tuple(
            capability
            for declaration in declared.values()
            for capability in declaration.capabilities
        )
        tool_views = tuple(
            view for declaration in declared.values() for view in declaration.tool_views
        )
        self._capabilities = MappingProxyType(
            _unique(capabilities, lambda item: item.id, "capability")
        )
        self._executors = MappingProxyType(
            _unique(executors, lambda item: item.executor_id, "executor")
        )
        self._views = MappingProxyType(
            _unique(tool_views, lambda item: item.name, "tool")
        )
        self._domain_owners = MappingProxyType(
            {
                capability.id: declaration.domain_owner_id
                for declaration in declared.values()
                for capability in declaration.capabilities
            }
        )
        for capability in self._capabilities.values():
            if capability.executor_id not in self._executors:
                raise ValueError(f"missing executor: {capability.executor_id}")
        for view in self._views.values():
            if view.capability_id not in self._capabilities:
                raise ValueError(f"missing capability: {view.capability_id}")
        declared_executor_ids = {
            executor_id
            for declaration in declared.values()
            for executor_id in declaration.executor_ids
        }
        if declared_executor_ids != set(self._executors):
            missing = sorted(declared_executor_ids - set(self._executors))
            unexpected = sorted(set(self._executors) - declared_executor_ids)
            raise ValueError(
                "registered executors must match static declarations: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self._digest = (
            "sha256:"
            + sha256(
                canonical_json(
                    [
                        {
                            "domain_owner_id": declaration.domain_owner_id,
                            "capabilities": [
                                {
                                    "id": capability.id,
                                    "description": capability.description,
                                    "input_schema": capability.input_schema,
                                    "output_kind": capability.output_kind,
                                    "output_schema": capability.output_schema,
                                    "executor_id": capability.executor_id,
                                    "access_mode": capability.access_mode.value,
                                    "operational_effect": (
                                        capability.operational_effect.value
                                    ),
                                    "artifact_policy": (
                                        None
                                        if capability.artifact_policy is None
                                        else {
                                            "allowed_media_types": sorted(
                                                capability.artifact_policy.allowed_media_types
                                            ),
                                            "allowed_extensions": (
                                                capability.artifact_policy.allowed_extensions
                                            ),
                                            "artifact_required": (
                                                capability.artifact_policy.artifact_required
                                            ),
                                            "max_artifact_count": (
                                                capability.artifact_policy.max_artifact_count
                                            ),
                                            "max_bytes_per_artifact": (
                                                capability.artifact_policy.max_bytes_per_artifact
                                            ),
                                            "max_total_bytes_per_call": (
                                                capability.artifact_policy.max_total_bytes_per_call
                                            ),
                                        }
                                    ),
                                }
                                for capability in sorted(
                                    declaration.capabilities, key=lambda item: item.id
                                )
                            ],
                            "tool_views": [
                                {
                                    "name": view.name,
                                    "capability_id": view.capability_id,
                                    "description": view.description,
                                    "discovery": {
                                        "summary": view.discovery.summary,
                                        "when_to_use": view.discovery.when_to_use,
                                        "keywords": view.discovery.keywords,
                                        "exposure_class": (
                                            view.discovery.exposure_class.value
                                        ),
                                        "eager_priority": view.discovery.eager_priority,
                                    },
                                    "origin_revision_digest": (
                                        view.origin_revision_digest
                                    ),
                                }
                                for view in sorted(
                                    declaration.tool_views, key=lambda item: item.name
                                )
                            ],
                        }
                        for declaration in sorted(
                            declared.values(), key=lambda item: item.domain_owner_id
                        )
                    ]
                ).encode("utf-8")
            ).hexdigest()
        )
        self._contract_digests = MappingProxyType(
            {
                capability.id: capability_contract_digest(
                    capability,
                    domain_owner_id=self._domain_owners[capability.id],
                )
                for capability in self._capabilities.values()
            }
        )

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._views)

    @property
    def domain_owner_ids(self) -> frozenset[str]:
        return frozenset(self._declarations)

    @property
    def digest(self) -> str:
        return self._digest

    def resolve_tool(self, name: str) -> tuple[ToolView, Capability]:
        try:
            view = self._views[name]
            return view, self._capabilities[view.capability_id]
        except KeyError as error:
            raise KeyError(f"unknown tool: {name}") from error

    def resolve_domain_owner(self, capability_id: str) -> str:
        try:
            return self._domain_owners[capability_id]
        except KeyError as error:
            raise KeyError(f"unknown capability: {capability_id}") from error

    def resolve_tool_owner(self, name: str) -> tuple[ToolView, Capability, str]:
        view, capability = self.resolve_tool(name)
        return view, capability, self.resolve_domain_owner(capability.id)

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

    def validate_execution_scope_grant(
        self,
        capability_ids: Iterable[str],
        *,
        allowed_access_modes: frozenset[AccessMode],
        allowed_operational_effects: frozenset[OperationalEffect],
    ) -> tuple[str, ...]:
        """Validate immutable grant metadata without creating an execution path."""

        material = tuple(capability_ids)
        if not material or len(material) != len(set(material)):
            raise ValueError("execution scope grant requires distinct capabilities")
        public_capability_ids = {view.capability_id for view in self._views.values()}
        for capability_id in material:
            try:
                capability = self._capabilities[capability_id]
            except KeyError as error:
                raise KeyError(f"unknown capability: {capability_id}") from error
            if capability_id not in public_capability_ids:
                raise ValueError(
                    f"execution scope capability is not model-visible: {capability_id}"
                )
            if (
                capability.access_mode not in allowed_access_modes
                or capability.operational_effect not in allowed_operational_effects
            ):
                raise ValueError(
                    f"execution scope capability metadata is not admitted: {capability_id}"
                )
        return tuple(sorted(material))

    def resolve_execution(self, capability_id: str) -> tuple[Capability, Executor]:
        capability = self._capabilities[capability_id]
        executor = self._executors[capability.executor_id]
        if executor.executor_id != capability.executor_id:
            raise ValueError(f"executor identity changed: {capability.executor_id}")
        return capability, executor

    def contract_digest(self, capability_id: str) -> str:
        try:
            return self._contract_digests[capability_id]
        except KeyError as error:
            raise KeyError(f"unknown capability: {capability_id}") from error

    def resolve_internal_execution(
        self,
        capability_id: str,
        contract_digest: str,
    ) -> tuple[Capability, Executor, str]:
        capability, executor = self.resolve_execution(capability_id)
        if any(view.capability_id == capability_id for view in self._views.values()):
            raise ValueError(
                "trusted internal execution requires an internal-only capability"
            )
        expected = self.contract_digest(capability_id)
        if contract_digest != expected:
            raise ValueError("internal execution capability contract changed")
        return capability, executor, self.resolve_domain_owner(capability_id)

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

    def validate_declarations(self, declarations: CapabilityDeclarations) -> None:
        if self._declarations.get(declarations.domain_owner_id) != declarations:
            raise ValueError(
                f"domain declaration differs: {declarations.domain_owner_id}"
            )
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
        _check_rule(rule)


def _check_rule(rule: Mapping[str, object]) -> None:
    enum = rule.get("enum")
    if enum is not None and (not isinstance(enum, (tuple, list)) or not enum):
        raise ValueError("tool schema enum must be a non-empty array")
    for minimum_name, maximum_name in (
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
    ):
        minimum = rule.get(minimum_name)
        maximum = rule.get(maximum_name)
        for bound, label in ((minimum, minimum_name), (maximum, maximum_name)):
            if bound is not None and (
                not isinstance(bound, int) or isinstance(bound, bool) or bound < 0
            ):
                raise ValueError(f"tool schema {label} must be non-negative")
        if (
            isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and minimum > maximum
        ):
            raise ValueError(f"tool schema {minimum_name} cannot exceed {maximum_name}")
    minimum = rule.get("minimum")
    maximum = rule.get("maximum")
    for bound, label in ((minimum, "minimum"), (maximum, "maximum")):
        if bound is not None and (
            not isinstance(bound, (int, float))
            or isinstance(bound, bool)
            or not math.isfinite(float(bound))
        ):
            raise ValueError(f"tool schema {label} must be a finite number")
    if (
        isinstance(minimum, (int, float))
        and not isinstance(minimum, bool)
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and minimum > maximum
    ):
        raise ValueError("tool schema minimum cannot exceed maximum")
    unique = rule.get("uniqueItems")
    if unique is not None and not isinstance(unique, bool):
        raise ValueError("tool schema uniqueItems must be a boolean")
    pattern = rule.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError("tool schema pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError("tool schema pattern must be valid") from error
    items = rule.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError("tool schema items must be an object rule")
        _check_rule(items)
    properties = rule.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ValueError("tool schema nested properties must be an object")
        for name, nested in properties.items():
            if not isinstance(name, str) or not isinstance(nested, Mapping):
                raise ValueError(
                    "tool schema nested properties must contain object rules"
                )
            _check_rule(nested)


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
        _validate_rule(name, item, rule, error_type)


def validate_tool_schema_value(
    schema: Mapping[str, object],
    value: Mapping[str, object],
) -> FrozenJsonObject:
    """Validate one object against the registry's bounded tool-schema contract."""

    frozen_schema = FrozenJsonObject.from_mapping(schema)
    frozen_value = FrozenJsonObject.from_mapping(value)
    _check_schema(frozen_schema)
    _validate(frozen_schema, frozen_value, ToolOutputValidationError)
    return frozen_value


def _validate_rule(
    name: str,
    item: object,
    rule: Mapping[str, object],
    error_type: type[ValueError] | type[RuntimeError],
) -> None:
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
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        if isinstance(minimum, (int, float)) and item < minimum:
            _constraint_error(error_type, name, f"at least {minimum}", "minimum")
        if isinstance(maximum, (int, float)) and item > maximum:
            _constraint_error(error_type, name, f"at most {maximum}", "maximum")
    if isinstance(item, (tuple, list)):
        minimum = rule.get("minItems")
        maximum = rule.get("maxItems")
        if isinstance(minimum, int) and len(item) < minimum:
            _constraint_error(error_type, name, f"at least {minimum} items", "minItems")
        if isinstance(maximum, int) and len(item) > maximum:
            _constraint_error(error_type, name, f"at most {maximum} items", "maxItems")
        if rule.get("uniqueItems") is True:
            projected = tuple(canonical_json(value) for value in item)
            if len(projected) != len(set(projected)):
                _constraint_error(
                    error_type,
                    name,
                    "unique items",
                    "uniqueItems",
                )
        item_rule = rule.get("items")
        if isinstance(item_rule, Mapping):
            for index, value in enumerate(item):
                _validate_rule(
                    f"{name}[{index}]",
                    value,
                    item_rule,
                    error_type,
                )
    if isinstance(item, Mapping) and expected == "object":
        _validate(rule, item, error_type)


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
    "ArtifactPolicy",
    "Capability",
    "CapabilityInputError",
    "CapabilityRegistry",
    "capability_contract_digest",
    "Executor",
    "OperationalEffect",
    "CapabilityDeclarations",
    "MAX_APPROVAL_DOCUMENT_CHARACTERS",
    "SideEffectExecutor",
    "ToolExecution",
    "ToolDiscoveryMetadata",
    "ToolExposureClass",
    "ToolOutput",
    "ToolOutputValidationError",
    "ToolView",
    "RESERVED_TOOL_NAMES",
    "render_approval_arguments",
    "validate_tool_schema_value",
]
