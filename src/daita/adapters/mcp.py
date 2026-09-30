"""Connect to Streamable HTTP MCP servers and discover or call remote tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from .._installation import repair_guidance
from .._json import FrozenJsonObject, canonical_json
from ..capabilities import (
    AccessMode,
    AutomationEligibility,
    OperationalEffect,
    ToolboxId,
    ToolLoadMode,
    ToolPresentation,
    ToolTextTrust,
    validate_discovery_hints,
)
from ..errors import DaitaError, ErrorRetryability
from ..llm.models import ModelSensitivity
from ..security import SecretProvider, SecretReference, SecretResolutionError

MCP_SUPPORTED_PROTOCOL_VERSIONS = ("2026-07-28", "2025-11-25", "2025-06-18")
MCP_MAX_SCHEMA_BYTES = 64 * 1_024
MCP_MAX_SCHEMA_DEPTH = 12
MCP_MAX_DISCOVERED_TOOLS = 256
MCP_MAX_DISCOVERY_PAGES = 4
MCP_MAX_ADMITTED_TOOLS_PER_BINDING = 128
MCP_MAX_BINDING_CANONICAL_BYTES = 1 * 1_024 * 1_024
MCP_MAX_AGENT_CATALOG_BYTES = 8 * 1_024 * 1_024
MCP_MAX_BINDINGS_PER_AGENT = 32
MCP_MAX_ACTIVE_TOOLS_PER_AGENT = 384
MCP_MAX_REQUEST_BYTES = 256 * 1_024
MCP_MAX_RESPONSE_BYTES = 512 * 1_024
MCP_MAX_RESULT_CONTENT_ITEMS = 32
MCP_MAX_TEXT_CHARACTERS = 256 * 1_024
MCP_REQUEST_TIMEOUT_SECONDS = 15.0

if TYPE_CHECKING:
    import httpx2

_BINDING_ID = re.compile(r"mcp-binding-[0-9a-f]{32}\Z")
_REMOTE_TOOL_NAME = re.compile(r"[^\s\x00-\x1f\x7f]{1,256}\Z")
_LOCAL_ALIAS = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
_SCHEMA_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SERVER_IDENTITY = re.compile(r"[^\r\n\x00]{1,256}\Z")
_JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_ANNOTATION_KEYS = frozenset({"description", "title", "examples"})
_SCHEMA_ROOT_KEYS = (
    frozenset({"$schema", "type", "properties", "required", "additionalProperties"})
    | _SCHEMA_ANNOTATION_KEYS
)
_SCHEMA_RULE_KEYS = (
    frozenset(
        {
            "type",
            "enum",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "items",
            "minItems",
            "maxItems",
            "properties",
            "required",
            "additionalProperties",
        }
    )
    | _SCHEMA_ANNOTATION_KEYS
)


class MCPTransportKind(str, Enum):
    STREAMABLE_HTTP = "streamable_http"


class MCPAuthenticationMode(str, Enum):
    NONE = "none"
    BEARER = "bearer"
    PERSONAL_CONNECTION = "personal_connection"


class MCPBindingState(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    REVOKED = "revoked"


class MCPCompletionSemantics(str, Enum):
    """Locally attested completion contract; never inferred from result prose."""

    DIRECT_RESULT = "direct_result"
    ASYNCHRONOUS_ONLY = "asynchronous_only"


class MCPError(DaitaError):
    """One safe protocol, transport, authentication, or admission failure."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
        *,
        retryability: ErrorRetryability = ErrorRetryability.PERMANENT,
    ) -> None:
        self.code = code
        self.details = FrozenJsonObject.from_mapping(details or {})
        super().__init__(
            message,
            error_code=code,
            retryability=retryability,
        )


class MCPTransportError(MCPError):
    pass


class MCPProtocolError(MCPError):
    pass


class MCPAuthenticationError(MCPError):
    pass


class MCPAdmissionError(MCPError):
    pass


class MCPRemoteToolError(MCPError):
    pass


@dataclass(frozen=True, slots=True)
class MCPAuthentication:
    mode: MCPAuthenticationMode
    secret_reference: SecretReference | None = None
    connection_id: str | None = None
    owner_principal_id: str | None = None
    resource_uri: str | None = None
    required_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, MCPAuthenticationMode):
            raise TypeError("MCP authentication mode is invalid")
        if self.mode is MCPAuthenticationMode.PERSONAL_CONNECTION:
            if self.secret_reference is not None:
                raise ValueError("personal MCP authentication cannot contain a secret")
            for value, label in (
                (self.connection_id, "connection ID"),
                (self.owner_principal_id, "owner principal"),
                (self.resource_uri, "resource URI"),
            ):
                if (
                    not isinstance(value, str)
                    or not value
                    or len(value) > 512
                    or any(character in value for character in "\r\n\x00")
                ):
                    raise ValueError(f"personal MCP {label} is invalid")
            if (
                not isinstance(self.required_scopes, tuple)
                or len(self.required_scopes) > 64
            ):
                raise ValueError("personal MCP scopes are invalid")
            if len(set(self.required_scopes)) != len(self.required_scopes) or any(
                not isinstance(scope, str)
                or not scope
                or len(scope) > 256
                or any(character in scope for character in "\r\n\x00")
                for scope in self.required_scopes
            ):
                raise ValueError("personal MCP scopes are invalid")
            resource = urlsplit(self.resource_uri or "")
            if (
                resource.scheme != "https"
                or not resource.hostname
                or resource.username is not None
                or resource.password is not None
                or resource.query
                or resource.fragment
            ):
                raise ValueError("personal MCP resource URI must be HTTPS")
        else:
            if any(
                (
                    self.connection_id,
                    self.owner_principal_id,
                    self.resource_uri,
                    self.required_scopes,
                )
            ):
                raise ValueError(
                    "non-personal MCP authentication cannot name a connection"
                )
            if self.mode is MCPAuthenticationMode.NONE:
                if self.secret_reference is not None:
                    raise ValueError(
                        "no-auth MCP configuration cannot contain a secret"
                    )
            elif not isinstance(self.secret_reference, SecretReference):
                raise ValueError(
                    "bearer MCP authentication requires a secret reference"
                )

    @classmethod
    def no_auth(cls) -> MCPAuthentication:
        return cls(MCPAuthenticationMode.NONE)

    @classmethod
    def bearer(cls, reference: SecretReference) -> MCPAuthentication:
        return cls(MCPAuthenticationMode.BEARER, reference)

    @classmethod
    def personal_connection(
        cls,
        connection_id: str,
        owner_principal_id: str,
        resource_uri: str,
        required_scopes: tuple[str, ...] = (),
    ) -> MCPAuthentication:
        return cls(
            MCPAuthenticationMode.PERSONAL_CONNECTION,
            connection_id=connection_id,
            owner_principal_id=owner_principal_id,
            resource_uri=resource_uri,
            required_scopes=required_scopes,
        )


class MCPConnectionProvider(Protocol):
    """Host-owned personal connection policy and per-request credential source.

    Implementations must raise MCPAuthenticationError with one of the four
    connection status codes and never include credential material in errors.
    """

    async def check_access(
        self,
        *,
        connection_id: str,
        principal_id: str,
        resource_uri: str,
        required_scopes: tuple[str, ...],
    ) -> None: ...

    async def access_token(
        self,
        *,
        connection_id: str,
        principal_id: str,
        resource_uri: str,
        required_scopes: tuple[str, ...],
    ) -> str: ...


_PERSONAL_STATUS_MESSAGES = {
    "needs_authorization": "The personal MCP connection needs authorization.",
    "needs_scope_upgrade": "The personal MCP connection needs additional scope.",
    "connection_revoked": "The personal MCP connection was revoked.",
    "account_unavailable": "The personal MCP account is unavailable.",
}


async def check_personal_connection(
    authentication: MCPAuthentication,
    provider: MCPConnectionProvider | None,
    principal_id: str | None,
) -> None:
    if authentication.mode is not MCPAuthenticationMode.PERSONAL_CONNECTION:
        return
    if principal_id != authentication.owner_principal_id or provider is None:
        raise MCPAuthenticationError(
            "needs_authorization", _PERSONAL_STATUS_MESSAGES["needs_authorization"]
        )
    assert authentication.connection_id is not None
    assert authentication.resource_uri is not None
    assert principal_id is not None
    try:
        await provider.check_access(
            connection_id=authentication.connection_id,
            principal_id=principal_id,
            resource_uri=authentication.resource_uri,
            required_scopes=authentication.required_scopes,
        )
    except MCPAuthenticationError as error:
        code = (
            error.code
            if error.code in _PERSONAL_STATUS_MESSAGES
            else "account_unavailable"
        )
        raise MCPAuthenticationError(code, _PERSONAL_STATUS_MESSAGES[code]) from None
    except Exception:
        raise MCPAuthenticationError(
            "account_unavailable", _PERSONAL_STATUS_MESSAGES["account_unavailable"]
        ) from None


@dataclass(frozen=True, slots=True)
class MCPToolSelection:
    """Local admission facts for one exact tool, independent of remote hints."""

    remote_name: str
    local_alias: str
    description: str
    summary: str | None = None
    when_to_use: str | None = None
    keywords: tuple[str, ...] = ()
    result_sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL
    access_mode: AccessMode = AccessMode.READ
    operational_effect: OperationalEffect = OperationalEffect.NONE
    automation_eligibility: AutomationEligibility | None = None
    maximum_outbound_sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED
    completion_semantics: MCPCompletionSemantics = MCPCompletionSemantics.DIRECT_RESULT

    def __post_init__(self) -> None:
        _remote_tool_name(self.remote_name)
        if (
            not isinstance(self.local_alias, str)
            or _LOCAL_ALIAS.fullmatch(self.local_alias) is None
        ):
            raise ValueError(
                "MCP local_alias must use lowercase letters, digits, and underscores"
            )
        _bounded_text(self.description, "MCP tool description", maximum=1_024)
        summary = self.description if self.summary is None else self.summary
        when_to_use = self.description if self.when_to_use is None else self.when_to_use
        presentation = ToolPresentation(
            toolbox_id=ToolboxId.SOURCES,
            load_mode=ToolLoadMode.ON_DEMAND,
            text_trust=ToolTextTrust.ADMITTED_UNTRUSTED,
            summary=summary,
            when_to_use=when_to_use,
            keywords=self.keywords,
        )
        object.__setattr__(self, "summary", presentation.summary)
        object.__setattr__(self, "when_to_use", presentation.when_to_use)
        object.__setattr__(self, "keywords", presentation.keywords)
        if not isinstance(self.result_sensitivity, ModelSensitivity):
            raise TypeError("MCP result_sensitivity is invalid")
        if self.automation_eligibility is None:
            object.__setattr__(
                self,
                "automation_eligibility",
                (
                    AutomationEligibility.AUTOMATION_DIRECT
                    if self.operational_effect is OperationalEffect.NONE
                    else AutomationEligibility.INTERACTIVE_ONLY
                ),
            )
        _validate_local_admission(
            self.access_mode,
            self.operational_effect,
            self.automation_eligibility,
            self.maximum_outbound_sensitivity,
            self.completion_semantics,
        )


@dataclass(frozen=True, slots=True)
class MCPInspectedTool:
    remote_name: str
    remote_description: str | None
    input_schema: FrozenJsonObject | None
    input_schema_digest: str | None
    output_schema: FrozenJsonObject | None
    output_schema_digest: str | None
    supported: bool
    unsupported_reason: str | None = None
    task_support: str = "forbidden"

    def __post_init__(self) -> None:
        _remote_tool_name(self.remote_name)
        if self.task_support not in {"forbidden", "optional", "required"}:
            raise ValueError("MCP task support is invalid")
        if self.remote_description is not None:
            _bounded_text(
                self.remote_description,
                "MCP remote description",
                maximum=2_048,
            )
        if not isinstance(self.supported, bool):
            raise TypeError("MCP inspected supported must be a boolean")
        schemas = (
            self.input_schema,
            self.input_schema_digest,
            self.output_schema,
            self.output_schema_digest,
        )
        if self.supported:
            if self.input_schema is None or self.input_schema_digest is None:
                raise ValueError("supported MCP tool requires an input schema")
            if self.unsupported_reason is not None:
                raise ValueError("supported MCP tool cannot have a rejection reason")
        else:
            if any(item is not None for item in schemas):
                raise ValueError("unsupported MCP tool cannot expose accepted schemas")
            _bounded_text(
                cast(str, self.unsupported_reason),
                "MCP unsupported reason",
                maximum=512,
            )
        for digest in (self.input_schema_digest, self.output_schema_digest):
            if digest is not None and _SCHEMA_DIGEST.fullmatch(digest) is None:
                raise ValueError("MCP schema digest is invalid")


@dataclass(frozen=True, slots=True)
class MCPServerInspection:
    endpoint: str
    protocol_version: str
    server_name: str | None
    server_version: str | None
    tools: tuple[MCPInspectedTool, ...]
    observed_at: datetime
    protocol_capabilities_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", normalize_mcp_endpoint(self.endpoint))
        if self.protocol_version not in MCP_SUPPORTED_PROTOCOL_VERSIONS:
            raise ValueError("MCP inspection protocol version is unsupported")
        if self.server_name is not None:
            _server_identity(self.server_name, "MCP server name")
        if self.server_version is not None:
            _server_identity(self.server_version, "MCP server version")
        if (
            self.protocol_capabilities_digest is not None
            and _SCHEMA_DIGEST.fullmatch(self.protocol_capabilities_digest) is None
        ):
            raise ValueError("MCP protocol capabilities digest is invalid")
        tools = tuple(self.tools)
        if len(tools) > MCP_MAX_DISCOVERED_TOOLS:
            raise ValueError("MCP inspection contains too many tools")
        if any(not isinstance(tool, MCPInspectedTool) for tool in tools):
            raise TypeError("MCP inspection tools are invalid")
        if len({tool.remote_name for tool in tools}) != len(tools):
            raise ValueError("MCP inspection repeats a remote tool name")
        _aware(self.observed_at, "MCP observed_at")
        object.__setattr__(
            self,
            "tools",
            tuple(sorted(tools, key=lambda item: item.remote_name)),
        )


@dataclass(frozen=True, slots=True)
class MCPToolBinding:
    capability_id: str
    executor_id: str
    local_name: str
    remote_name: str
    description: str
    presentation: ToolPresentation
    input_schema: FrozenJsonObject
    input_schema_digest: str
    output_schema: FrozenJsonObject | None
    output_schema_digest: str | None
    result_sensitivity: ModelSensitivity
    access_mode: AccessMode
    operational_effect: OperationalEffect
    automation_eligibility: AutomationEligibility
    maximum_outbound_sensitivity: ModelSensitivity
    completion_semantics: MCPCompletionSemantics
    task_support: str

    def __post_init__(self) -> None:
        _validate_local_admission(
            self.access_mode,
            self.operational_effect,
            self.automation_eligibility,
            self.maximum_outbound_sensitivity,
            self.completion_semantics,
        )
        if self.task_support not in {"forbidden", "optional", "required"}:
            raise ValueError("MCP task support is invalid")
        if (
            self.task_support == "required"
            and self.automation_eligibility is AutomationEligibility.AUTOMATION_DIRECT
        ):
            raise ValueError("MCP unattended task completion is unsupported")
        for value, label, maximum in (
            (self.capability_id, "MCP capability id", 512),
            (self.executor_id, "MCP executor id", 512),
            (self.local_name, "MCP local tool name", 64),
            (self.description, "MCP tool description", 1_024),
        ):
            _bounded_text(value, label, maximum=maximum)
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.local_name) is None:
            raise ValueError("MCP local tool name is not provider-safe")
        if not isinstance(self.presentation, ToolPresentation):
            raise TypeError("MCP tool presentation metadata is required")
        if (
            self.presentation.toolbox_id is not ToolboxId.SOURCES
            or self.presentation.load_mode is not ToolLoadMode.ON_DEMAND
            or self.presentation.text_trust is not ToolTextTrust.ADMITTED_UNTRUSTED
        ):
            raise ValueError(
                "MCP tools must use Sources/on-demand/admitted-untrusted presentation"
            )
        _remote_tool_name(self.remote_name)
        if not isinstance(self.input_schema, FrozenJsonObject):
            object.__setattr__(
                self,
                "input_schema",
                FrozenJsonObject.from_mapping(self.input_schema),
            )
        for digest in (self.input_schema_digest, self.output_schema_digest):
            if digest is not None and _SCHEMA_DIGEST.fullmatch(digest) is None:
                raise ValueError("MCP tool schema digest is invalid")
        if (self.output_schema is None) is not (self.output_schema_digest is None):
            raise ValueError("MCP output schema and digest must be present together")
        if not isinstance(self.result_sensitivity, ModelSensitivity):
            raise TypeError("MCP result sensitivity is invalid")


@dataclass(frozen=True, slots=True)
class MCPServerBinding:
    binding_id: str
    agent_id: str
    endpoint: str
    authentication: MCPAuthentication
    protocol_version: str
    server_name: str | None
    server_version: str | None
    local_label: str
    maximum_outbound_sensitivity: ModelSensitivity
    tools: tuple[MCPToolBinding, ...]
    state: MCPBindingState
    revision: int
    admitted_at: datetime
    last_checked_at: datetime
    revoked_at: datetime | None = None
    stale_reason: str | None = None
    summary: str = ""
    when_to_use: str = ""
    keywords: tuple[str, ...] = ()
    owner_principal_id: str | None = None
    protocol_capabilities_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.binding_id, str)
            or _BINDING_ID.fullmatch(self.binding_id) is None
        ):
            raise ValueError("MCP binding_id must use mcp-binding-<32 lowercase hex>")
        _bounded_text(self.agent_id, "MCP agent_id", maximum=256)
        if not isinstance(self.authentication, MCPAuthentication):
            raise TypeError("MCP binding authentication is invalid")
        owner = self.owner_principal_id or self.agent_id
        _bounded_text(owner, "MCP owner principal", maximum=512)
        if self.authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION and (
            self.authentication.owner_principal_id != owner
        ):
            raise ValueError("personal MCP connection owner differs from binding owner")
        object.__setattr__(self, "owner_principal_id", owner)
        object.__setattr__(self, "endpoint", normalize_mcp_endpoint(self.endpoint))
        _require_personal_resource_origin(self.endpoint, self.authentication)
        if self.protocol_version not in MCP_SUPPORTED_PROTOCOL_VERSIONS:
            raise ValueError("MCP binding protocol version is unsupported")
        if self.server_name is not None:
            _server_identity(self.server_name, "MCP server name")
        if self.server_version is not None:
            _server_identity(self.server_version, "MCP server version")
        if (
            self.protocol_capabilities_digest is not None
            and _SCHEMA_DIGEST.fullmatch(self.protocol_capabilities_digest) is None
        ):
            raise ValueError("MCP protocol capabilities digest is invalid")
        _bounded_text(self.local_label, "MCP local server label", maximum=128)
        object.__setattr__(
            self,
            "keywords",
            validate_discovery_hints(
                self.summary, self.when_to_use, self.keywords, allow_empty=True
            ),
        )
        if not isinstance(self.maximum_outbound_sensitivity, ModelSensitivity):
            raise TypeError("MCP outbound sensitivity ceiling is invalid")
        tools = tuple(self.tools)
        if not tools or len(tools) > MCP_MAX_ADMITTED_TOOLS_PER_BINDING:
            raise ValueError("MCP binding requires a bounded admitted tool set")
        for values, label in (
            ((tool.capability_id for tool in tools), "capability"),
            ((tool.local_name for tool in tools), "local tool"),
            ((tool.remote_name for tool in tools), "remote tool"),
        ):
            items = tuple(values)
            if len(items) != len(set(items)):
                raise ValueError(f"MCP binding repeats a {label} identity")
        executor_ids = {tool.executor_id for tool in tools}
        if len(executor_ids) != 1:
            raise ValueError("MCP binding tools must share one binding executor")
        if not isinstance(self.state, MCPBindingState):
            raise TypeError("MCP binding state is invalid")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("MCP binding revision must be positive")
        _aware(self.admitted_at, "MCP admitted_at")
        _aware(self.last_checked_at, "MCP last_checked_at")
        if self.last_checked_at < self.admitted_at:
            raise ValueError("MCP last_checked_at cannot precede admission")
        if self.state is MCPBindingState.REVOKED:
            if self.revoked_at is None:
                raise ValueError("revoked MCP binding requires revoked_at")
            _aware(self.revoked_at, "MCP revoked_at")
        elif self.revoked_at is not None:
            raise ValueError("non-revoked MCP binding cannot contain revoked_at")
        if self.state is MCPBindingState.STALE:
            _bounded_text(
                cast(str, self.stale_reason),
                "MCP stale reason",
                maximum=512,
            )
        elif self.stale_reason is not None:
            raise ValueError("only a stale MCP binding can contain stale_reason")
        object.__setattr__(
            self,
            "tools",
            tuple(sorted(tools, key=lambda item: item.local_name)),
        )

    def checked(
        self,
        *,
        observed_at: datetime,
        stale_reason: str | None,
    ) -> MCPServerBinding:
        return replace(
            self,
            state=(
                MCPBindingState.ACTIVE
                if stale_reason is None
                else MCPBindingState.STALE
            ),
            revision=self.revision + 1,
            last_checked_at=observed_at,
            revoked_at=None,
            stale_reason=stale_reason,
        )

    def revoke(self, *, revoked_at: datetime) -> MCPServerBinding:
        return replace(
            self,
            state=MCPBindingState.REVOKED,
            revision=self.revision + 1,
            last_checked_at=revoked_at,
            revoked_at=revoked_at,
            stale_reason=None,
        )


def mcp_execution_origin_digest(binding: MCPServerBinding, tool: MCPToolBinding) -> str:
    """Identify execution admission independently of editable local hints."""

    material = {
        "agent_id": binding.agent_id,
        "binding_id": binding.binding_id,
        "binding_revision": binding.revision,
        "endpoint": binding.endpoint,
        "protocol_version": binding.protocol_version,
        "server_name": binding.server_name,
        "server_version": binding.server_version,
        "authentication_mode": binding.authentication.mode.value,
        "secret_reference": (
            binding.authentication.secret_reference.to_uri()
            if binding.authentication.secret_reference is not None
            else None
        ),
        "maximum_outbound_sensitivity": binding.maximum_outbound_sensitivity.value,
        "capability_id": tool.capability_id,
        "executor_id": tool.executor_id,
        "local_name": tool.local_name,
        "remote_name": tool.remote_name,
        "input_schema": tool.input_schema,
        "input_schema_digest": tool.input_schema_digest,
        "output_schema": tool.output_schema,
        "output_schema_digest": tool.output_schema_digest,
        "result_sensitivity": tool.result_sensitivity.value,
        "access_mode": tool.access_mode.value,
        "operational_effect": tool.operational_effect.value,
        "automation_eligibility": tool.automation_eligibility.value,
        "tool_maximum_outbound_sensitivity": tool.maximum_outbound_sensitivity.value,
        "completion_semantics": tool.completion_semantics.value,
        "task_support": tool.task_support,
    }
    if binding.authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION:
        material.update(
            owner_principal_id=binding.owner_principal_id,
            connection_id=binding.authentication.connection_id,
            resource_uri=binding.authentication.resource_uri,
            required_scopes=binding.authentication.required_scopes,
        )
    if binding.protocol_capabilities_digest is not None:
        material["protocol_capabilities_digest"] = binding.protocol_capabilities_digest
    return "sha256:" + sha256(canonical_json(material).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MCPBindingStatus:
    binding: MCPServerBinding
    activated_revision: int | None

    @property
    def active_in_runtime(self) -> bool:
        return (
            self.binding.state is MCPBindingState.ACTIVE
            and self.activated_revision == self.binding.revision
        )

    @property
    def reopen_required(self) -> bool:
        return (
            self.binding.state is MCPBindingState.ACTIVE
            and self.activated_revision != self.binding.revision
        )


@dataclass(frozen=True, slots=True)
class MCPToolResult:
    text: tuple[str, ...] = ()
    structured: FrozenJsonObject | None = None
    is_error: bool = False
    accepted_async: bool = False
    operation_handle: str | None = None

    def __post_init__(self) -> None:
        text_items = tuple(self.text)
        if len(text_items) > MCP_MAX_RESULT_CONTENT_ITEMS:
            raise ValueError("MCP result contains too many text items")
        if any(
            not isinstance(item, str) or len(item) > MCP_MAX_TEXT_CHARACTERS
            for item in text_items
        ):
            raise ValueError("MCP result text is invalid or oversized")
        if self.structured is not None and not isinstance(
            self.structured, FrozenJsonObject
        ):
            object.__setattr__(
                self,
                "structured",
                FrozenJsonObject.from_mapping(self.structured),
            )
        if not isinstance(self.is_error, bool):
            raise TypeError("MCP result is_error must be a boolean")
        if not isinstance(self.accepted_async, bool):
            raise TypeError("MCP acceptance must be a boolean")
        if self.operation_handle is not None:
            _bounded_text(self.operation_handle, "MCP operation handle", maximum=256)
            if not self.accepted_async:
                raise ValueError("MCP operation handle requires explicit acceptance")
        object.__setattr__(self, "text", text_items)


class MCPClient(Protocol):
    """One endpoint's protocol client, owned and closed by its factory's caller.

    Inspection must fetch current remote contracts without a tools/list cache.
    A tool call sends at most one tools/call and never continues an input-required
    or asynchronous result. Operations have finite bounds, propagate cancellation
    and use MCPError for safe failures. close is terminal and idempotent.
    """

    async def inspect(self, *, observed_at: datetime) -> MCPServerInspection: ...

    async def call_tool(
        self,
        remote_name: str,
        arguments: Mapping[str, object],
    ) -> MCPToolResult: ...

    async def close(self) -> None: ...


@runtime_checkable
class MCPPersonalConnectionClient(MCPClient, Protocol):
    """Optional client extension for an exact host-owned personal connection."""

    def bind_personal_connection(
        self, provider: MCPConnectionProvider, principal_id: str
    ) -> None:
        """Bind once before first use; resolve credentials only per request."""
        ...


class MCPClientFactory(Protocol):
    """Supported construction seam for Agent.create and Agent.open.

    create performs no network or credential I/O and returns a new independently
    owned client. Agent retains the factory but closes every client it creates,
    including temporary inspection clients and clients rejected before use.
    No-auth and bearer clients need only MCPClient; personal clients additionally
    implement MCPPersonalConnectionClient.
    """

    def create(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        secrets: SecretProvider,
    ) -> MCPClient: ...


class SDKMCPClientFactory:
    """Configure the sole built-in SDK client without importing its integration.

    Agent uses this factory by default. http_transport is an SDK-specific httpx2
    test or transport configuration option, outside MCPClientFactory's contract.
    """

    def __init__(
        self,
        *,
        http_transport: httpx2.AsyncBaseTransport | None = None,
        timeout_seconds: float = MCP_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not 0 < float(timeout_seconds) <= 60
        ):
            raise ValueError("MCP timeout must be positive and at most 60 seconds")
        self._transport = http_transport
        self._timeout = float(timeout_seconds)

    def create(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        secrets: SecretProvider,
    ) -> MCPClient:
        try:
            from .mcp_sdk import SDKMCPClient
        except ImportError as error:
            raise ImportError(
                "The MCP SDK integration is unavailable. " + repair_guidance()
            ) from error
        return SDKMCPClient(
            endpoint=endpoint,
            authentication=authentication,
            secrets=secrets,
            http_transport=self._transport,
            timeout_seconds=self._timeout,
        )


def normalize_mcp_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise ValueError("MCP endpoint must be bounded non-empty text")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("MCP endpoint must use http or https")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("MCP endpoint must have an origin and no user credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("MCP endpoint query strings and fragments are not admitted")
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("non-loopback MCP endpoints must use https")
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _require_personal_resource_origin(
    endpoint: str, authentication: MCPAuthentication
) -> None:
    if authentication.mode is not MCPAuthenticationMode.PERSONAL_CONNECTION:
        return
    assert authentication.resource_uri is not None
    target = urlsplit(endpoint)
    resource = urlsplit(authentication.resource_uri)
    try:
        same_origin = (
            target.scheme == resource.scheme == "https"
            and target.hostname == resource.hostname
            and (target.port or 443) == (resource.port or 443)
        )
    except ValueError:
        same_origin = False
    if not same_origin:
        raise ValueError("personal MCP resource and endpoint origins differ")


def mcp_binding_from_inspection(
    *,
    binding_id: str,
    agent_id: str,
    authentication: MCPAuthentication,
    maximum_outbound_sensitivity: ModelSensitivity,
    selections: tuple[MCPToolSelection, ...],
    inspection: MCPServerInspection,
    local_label: str | None = None,
    prior: MCPServerBinding | None = None,
    owner_principal_id: str | None = None,
) -> MCPServerBinding:
    if prior is not None:
        if prior.binding_id != binding_id or prior.agent_id != agent_id:
            raise MCPAdmissionError(
                "mcp_binding_identity_mismatch",
                "The existing MCP binding belongs to another identity.",
            )
        if prior.owner_principal_id != (owner_principal_id or agent_id):
            raise MCPAdmissionError(
                "mcp_binding_identity_mismatch",
                "The existing MCP binding belongs to another caller.",
            )
        if prior.endpoint != inspection.endpoint:
            raise MCPAdmissionError(
                "mcp_binding_remote_changed",
                "An existing MCP binding cannot be redirected to another remote "
                "server; attach it with a new binding identity.",
                {
                    "binding_id": binding_id,
                    "accepted_endpoint": prior.endpoint,
                    "observed_endpoint": inspection.endpoint,
                    "accepted_server_name": prior.server_name,
                    "observed_server_name": inspection.server_name,
                },
            )
    selections = tuple(selections)
    resolved_local_label = (
        prior.local_label
        if local_label is None and prior is not None
        else (
            _default_mcp_local_label(inspection.endpoint)
            if local_label is None
            else local_label
        )
    )
    _bounded_text(resolved_local_label, "MCP local server label", maximum=128)
    if not selections:
        raise MCPAdmissionError(
            "mcp_allowlist_empty",
            "At least one exact MCP tool must be selected.",
        )
    if len({item.remote_name for item in selections}) != len(selections):
        raise MCPAdmissionError(
            "mcp_allowlist_duplicate",
            "The MCP tool allowlist contains duplicate remote names.",
        )
    if len({item.local_alias for item in selections}) != len(selections):
        raise MCPAdmissionError(
            "mcp_local_name_duplicate",
            "The MCP tool allowlist contains duplicate local aliases.",
        )
    discovered = {tool.remote_name: tool for tool in inspection.tools}
    executor_id = f"mcp.executor:{binding_id}"
    tools: list[MCPToolBinding] = []
    namespace = sha256(binding_id.encode("utf-8")).hexdigest()[:12]
    for selection in selections:
        inspected = discovered.get(selection.remote_name)
        if inspected is None:
            raise MCPAdmissionError(
                "mcp_tool_not_found",
                "A selected MCP tool was not present in the inspected surface.",
                {"remote_name": selection.remote_name},
            )
        if not inspected.supported:
            raise MCPAdmissionError(
                "mcp_schema_unsupported",
                "A selected MCP tool uses an unsupported schema.",
                {
                    "remote_name": selection.remote_name,
                    "reason": inspected.unsupported_reason or "unsupported_schema",
                },
            )
        assert inspected.input_schema is not None
        assert inspected.input_schema_digest is not None
        local_name = f"mcp_{namespace}_{selection.local_alias}"
        capability_hash = sha256(
            f"{binding_id}\x00{selection.remote_name}".encode("utf-8")
        ).hexdigest()
        tools.append(
            MCPToolBinding(
                capability_id=f"mcp.tool:sha256:{capability_hash}",
                executor_id=executor_id,
                local_name=local_name,
                remote_name=selection.remote_name,
                description=selection.description,
                presentation=ToolPresentation(
                    toolbox_id=ToolboxId.SOURCES,
                    load_mode=ToolLoadMode.ON_DEMAND,
                    text_trust=ToolTextTrust.ADMITTED_UNTRUSTED,
                    summary=cast(str, selection.summary),
                    when_to_use=cast(str, selection.when_to_use),
                    keywords=selection.keywords,
                ),
                input_schema=inspected.input_schema,
                input_schema_digest=inspected.input_schema_digest,
                output_schema=inspected.output_schema,
                output_schema_digest=inspected.output_schema_digest,
                result_sensitivity=selection.result_sensitivity,
                access_mode=selection.access_mode,
                operational_effect=selection.operational_effect,
                automation_eligibility=cast(
                    AutomationEligibility, selection.automation_eligibility
                ),
                maximum_outbound_sensitivity=selection.maximum_outbound_sensitivity,
                completion_semantics=selection.completion_semantics,
                task_support=inspected.task_support,
            )
        )
    revision = 1 if prior is None else prior.revision + 1
    admitted_at = inspection.observed_at if prior is None else prior.admitted_at
    return MCPServerBinding(
        binding_id=binding_id,
        agent_id=agent_id,
        endpoint=inspection.endpoint,
        authentication=authentication,
        protocol_version=inspection.protocol_version,
        server_name=inspection.server_name,
        server_version=inspection.server_version,
        protocol_capabilities_digest=inspection.protocol_capabilities_digest,
        local_label=resolved_local_label,
        maximum_outbound_sensitivity=maximum_outbound_sensitivity,
        tools=tuple(tools),
        state=MCPBindingState.ACTIVE,
        revision=revision,
        admitted_at=admitted_at,
        last_checked_at=inspection.observed_at,
        summary="" if prior is None else prior.summary,
        when_to_use="" if prior is None else prior.when_to_use,
        keywords=() if prior is None else prior.keywords,
        owner_principal_id=owner_principal_id or agent_id,
    )


def _default_mcp_local_label(endpoint: str) -> str:
    hostname = urlsplit(endpoint).hostname
    if not hostname:
        raise ValueError("MCP endpoint must have a hostname")
    return f"MCP {hostname}"[:128]


def mcp_binding_drift_reason(
    binding: MCPServerBinding,
    inspection: MCPServerInspection,
) -> str | None:
    if inspection.endpoint != binding.endpoint:
        return "endpoint_changed"
    if inspection.protocol_version != binding.protocol_version:
        return "protocol_version_changed"
    if (
        inspection.server_name is not None
        and binding.server_name is not None
        and inspection.server_name != binding.server_name
    ) or (
        inspection.server_version is not None
        and binding.server_version is not None
        and inspection.server_version != binding.server_version
    ):
        return "server_identity_changed"
    if (
        binding.protocol_capabilities_digest is not None
        and inspection.protocol_capabilities_digest
        != binding.protocol_capabilities_digest
    ):
        return "protocol_capabilities_changed"
    discovered = {tool.remote_name: tool for tool in inspection.tools}
    for accepted in binding.tools:
        current = discovered.get(accepted.remote_name)
        if current is None:
            return f"tool_missing:{accepted.remote_name}"
        if not current.supported:
            return f"tool_schema_unsupported:{accepted.remote_name}"
        if current.task_support != accepted.task_support:
            return f"tool_invocation_changed:{accepted.remote_name}"
        if (
            current.input_schema_digest != accepted.input_schema_digest
            or current.output_schema_digest != accepted.output_schema_digest
        ):
            return f"tool_schema_changed:{accepted.remote_name}"
    return None


def canonical_mcp_schema(
    schema: Mapping[str, object],
) -> tuple[FrozenJsonObject, str]:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]

    raw = FrozenJsonObject.from_mapping(schema)
    encoded = canonical_json(raw).encode("utf-8")
    if len(encoded) > MCP_MAX_SCHEMA_BYTES:
        raise ValueError("schema exceeds the fixed byte bound")
    # The selected model adapters project this deliberately small schema subset.
    # Reject other valid JSON Schema features instead of weakening them on projection.
    pending: list[tuple[Mapping[str, object], int, bool]] = [(raw, 1, True)]
    nodes = 0
    while pending:
        node, depth, root = pending.pop()
        nodes += 1
        if depth > MCP_MAX_SCHEMA_DEPTH or nodes > 1024:
            raise ValueError("schema exceeds the fixed depth or node bound")
        unsupported = sorted(
            set(node) - (_SCHEMA_ROOT_KEYS if root else _SCHEMA_RULE_KEYS)
        )
        if unsupported:
            raise ValueError(f"unsupported schema keyword: {unsupported[0]}")
        if root:
            if "$schema" in node and node["$schema"] != _JSON_SCHEMA_2020_12:
                raise ValueError("schema dialect is unsupported")
            if node.get("type") != "object":
                raise ValueError("schema root must have type object")
        properties = node.get("properties", {})
        if not isinstance(properties, Mapping) or len(properties) > 128:
            raise ValueError("schema properties must be a bounded object")
        for name, rule in properties.items():
            if (
                not isinstance(name, str)
                or not name
                or len(name) > 128
                or not isinstance(rule, Mapping)
            ):
                raise ValueError("schema property declaration is invalid")
            pending.append((rule, depth + 1, False))
        items = node.get("items")
        if items is not None:
            if not isinstance(items, Mapping):
                raise ValueError("schema items must be an object rule")
            pending.append((items, depth + 1, False))
    try:
        Draft202012Validator.check_schema(raw.to_dict())
    except SchemaError:
        raise ValueError("schema is invalid") from None
    projected = FrozenJsonObject.from_mapping(_strip_schema_annotations(raw))
    return projected, f"sha256:{sha256(encoded).hexdigest()}"


def validate_mcp_schema_value(
    schema: FrozenJsonObject, value: Mapping[str, object]
) -> None:
    from jsonschema import Draft202012Validator

    frozen = FrozenJsonObject.from_mapping(value)
    encoded = canonical_json(frozen).encode("utf-8")
    if len(encoded) > MCP_MAX_REQUEST_BYTES:
        raise ValueError("MCP schema value exceeds the fixed byte bound")
    pending: list[tuple[object, int]] = [(frozen, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > MCP_MAX_SCHEMA_DEPTH + 8 or nodes > 4096:
            raise ValueError("MCP schema value exceeds the fixed depth or node bound")
        if isinstance(item, Mapping):
            pending.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend((nested, depth + 1) for nested in item)
    if (
        next(Draft202012Validator(schema.to_dict()).iter_errors(frozen.to_dict()), None)
        is not None
    ):
        raise ValueError("MCP schema value is invalid")


def _inspect_tool(value: object) -> MCPInspectedTool:
    if not isinstance(value, Mapping):
        raise MCPProtocolError(
            "mcp_protocol_invalid",
            "The MCP server returned an invalid tool declaration.",
        )
    name = value.get("name")
    try:
        _remote_tool_name(cast(str, name))
    except (TypeError, ValueError):
        raise MCPProtocolError(
            "mcp_protocol_invalid",
            "The MCP server returned an invalid remote tool identity.",
        ) from None
    description = value.get("description")
    if description is not None:
        if not isinstance(description, str):
            description = None
        else:
            description = description[:2_048]
    input_raw = value.get("inputSchema")
    output_raw = value.get("outputSchema")
    execution = value.get("execution", {})
    task_support = (
        execution.get("taskSupport", "forbidden")
        if isinstance(execution, Mapping)
        else None
    )
    try:
        if task_support not in {"forbidden", "optional", "required"}:
            raise ValueError("unsupported tool invocation semantics")
        if not isinstance(input_raw, Mapping):
            raise ValueError("input schema must be an object")
        input_schema, input_digest = canonical_mcp_schema(input_raw)
        if output_raw is None:
            output_schema = None
            output_digest = None
        else:
            if not isinstance(output_raw, Mapping):
                raise ValueError("output schema must be an object")
            output_schema, output_digest = canonical_mcp_schema(output_raw)
    except (TypeError, ValueError) as error:
        return MCPInspectedTool(
            remote_name=cast(str, name),
            remote_description=description,
            input_schema=None,
            input_schema_digest=None,
            output_schema=None,
            output_schema_digest=None,
            supported=False,
            unsupported_reason=str(error)[:512],
        )
    return MCPInspectedTool(
        remote_name=cast(str, name),
        remote_description=description,
        input_schema=input_schema,
        input_schema_digest=input_digest,
        output_schema=output_schema,
        output_schema_digest=output_digest,
        supported=True,
        task_support=cast(str, task_support),
    )


def _validate_local_admission(
    access: AccessMode,
    effect: OperationalEffect,
    eligibility: AutomationEligibility | None,
    outbound: ModelSensitivity,
    completion: MCPCompletionSemantics,
) -> None:
    if not isinstance(access, AccessMode) or not isinstance(effect, OperationalEffect):
        raise TypeError("MCP local access/effect admission is invalid")
    if effect not in {
        OperationalEffect.NONE,
        OperationalEffect.MUTATE_DATA,
        OperationalEffect.EXTERNAL_ACTION,
    }:
        raise ValueError(
            "MCP operational effect is unsupported; shell, infrastructure and arbitrary execution are not admitted"
        )
    if (effect is OperationalEffect.NONE and access is AccessMode.WRITE) or (
        effect is OperationalEffect.MUTATE_DATA and access is not AccessMode.WRITE
    ):
        raise ValueError("MCP data access must agree with its locally admitted effect")
    if (
        not isinstance(eligibility, AutomationEligibility)
        or not isinstance(outbound, ModelSensitivity)
        or not isinstance(completion, MCPCompletionSemantics)
    ):
        raise TypeError(
            "MCP local eligibility, sensitivity or completion admission is invalid"
        )
    if (
        completion is MCPCompletionSemantics.ASYNCHRONOUS_ONLY
        and eligibility is AutomationEligibility.AUTOMATION_DIRECT
    ):
        raise ValueError("MCP unattended asynchronous completion is unsupported")


def _strip_schema_annotations(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        if key in _SCHEMA_ANNOTATION_KEYS or key == "$schema":
            continue
        if key == "properties" and isinstance(item, Mapping):
            result[key] = {
                name: _strip_schema_annotations(rule)
                for name, rule in item.items()
                if isinstance(name, str) and isinstance(rule, Mapping)
            }
        elif key == "items" and isinstance(item, Mapping):
            result[key] = _strip_schema_annotations(item)
        else:
            result[key] = item
    return result


def _bounded_text(value: str, label: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(character in value for character in "\x00")
    ):
        raise ValueError(f"{label} must be bounded non-empty text")


def _remote_tool_name(value: str) -> None:
    if not isinstance(value, str) or _REMOTE_TOOL_NAME.fullmatch(value) is None:
        raise ValueError("MCP remote tool name is invalid")


def _server_identity(value: str, label: str) -> None:
    if not isinstance(value, str) or _SERVER_IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")


def _aware(value: datetime, label: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{label} must be timezone-aware")


__all__ = [
    "MCPAdmissionError",
    "MCPAuthentication",
    "MCPAuthenticationError",
    "MCPAuthenticationMode",
    "MCPBindingState",
    "MCPBindingStatus",
    "MCPClient",
    "MCPClientFactory",
    "MCPCompletionSemantics",
    "MCPConnectionProvider",
    "MCPError",
    "MCPInspectedTool",
    "MCPProtocolError",
    "MCPPersonalConnectionClient",
    "MCPRemoteToolError",
    "MCPServerBinding",
    "MCPServerInspection",
    "MCPToolBinding",
    "MCPToolResult",
    "MCPToolSelection",
    "MCPTransportError",
    "MCPTransportKind",
    "MCP_SUPPORTED_PROTOCOL_VERSIONS",
    "MCP_MAX_ACTIVE_TOOLS_PER_AGENT",
    "MCP_MAX_BINDINGS_PER_AGENT",
    "MCP_MAX_REQUEST_BYTES",
    "SDKMCPClientFactory",
    "canonical_mcp_schema",
    "mcp_binding_drift_reason",
    "mcp_binding_from_inspection",
    "normalize_mcp_endpoint",
]
