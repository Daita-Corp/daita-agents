"""Connect to Streamable HTTP MCP servers and discover or call remote tools."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from .._installation import repair_guidance
from .._json import FrozenJsonObject, canonical_json
from ..capabilities import (
    ToolDiscoveryMetadata,
    ToolExposureClass,
)
from ..errors import DaitaError, ErrorRetryability
from ..llm.models import ModelSensitivity
from ..security import SecretProvider, SecretReference, SecretResolutionError

if TYPE_CHECKING:
    import httpx

MCP_SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18")
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
            "pattern",
            "items",
            "minItems",
            "maxItems",
            "uniqueItems",
            "properties",
            "required",
            "additionalProperties",
        }
    )
    | _SCHEMA_ANNOTATION_KEYS
)
_SCHEMA_TYPES = frozenset({"array", "boolean", "integer", "number", "object", "string"})


class MCPTransportKind(str, Enum):
    STREAMABLE_HTTP = "streamable_http"


class MCPAuthenticationMode(str, Enum):
    NONE = "none"
    BEARER = "bearer"


class MCPBindingState(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    REVOKED = "revoked"


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

    def __post_init__(self) -> None:
        if not isinstance(self.mode, MCPAuthenticationMode):
            raise TypeError("MCP authentication mode is invalid")
        if self.mode is MCPAuthenticationMode.NONE:
            if self.secret_reference is not None:
                raise ValueError("no-auth MCP configuration cannot contain a secret")
        elif not isinstance(self.secret_reference, SecretReference):
            raise ValueError("bearer MCP authentication requires a secret reference")

    @classmethod
    def no_auth(cls) -> MCPAuthentication:
        return cls(MCPAuthenticationMode.NONE)

    @classmethod
    def bearer(cls, reference: SecretReference) -> MCPAuthentication:
        return cls(MCPAuthenticationMode.BEARER, reference)


@dataclass(frozen=True, slots=True)
class MCPToolSelection:
    """Code-owned admission request for one explicitly selected read tool."""

    remote_name: str
    local_alias: str
    description: str
    summary: str | None = None
    when_to_use: str | None = None
    keywords: tuple[str, ...] = ()
    exposure_class: ToolExposureClass = ToolExposureClass.DEFERRED
    eager_priority: int = 0
    result_sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL
    read_only: bool = True

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
        discovery = ToolDiscoveryMetadata(
            summary=summary,
            when_to_use=when_to_use,
            keywords=self.keywords,
            exposure_class=self.exposure_class,
            eager_priority=self.eager_priority,
        )
        object.__setattr__(self, "summary", discovery.summary)
        object.__setattr__(self, "when_to_use", discovery.when_to_use)
        object.__setattr__(self, "keywords", discovery.keywords)
        if not isinstance(self.result_sensitivity, ModelSensitivity):
            raise TypeError("MCP result_sensitivity is invalid")
        if self.read_only is not True:
            raise ValueError("Stage M2 admits only explicitly attested read-only tools")


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

    def __post_init__(self) -> None:
        _remote_tool_name(self.remote_name)
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
    server_name: str
    server_version: str
    tools: tuple[MCPInspectedTool, ...]
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", normalize_mcp_endpoint(self.endpoint))
        if self.protocol_version not in MCP_SUPPORTED_PROTOCOL_VERSIONS:
            raise ValueError("MCP inspection protocol version is unsupported")
        _server_identity(self.server_name, "MCP server name")
        _server_identity(self.server_version, "MCP server version")
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
    discovery: ToolDiscoveryMetadata
    input_schema: FrozenJsonObject
    input_schema_digest: str
    output_schema: FrozenJsonObject | None
    output_schema_digest: str | None
    result_sensitivity: ModelSensitivity

    def __post_init__(self) -> None:
        for value, label, maximum in (
            (self.capability_id, "MCP capability id", 512),
            (self.executor_id, "MCP executor id", 512),
            (self.local_name, "MCP local tool name", 64),
            (self.description, "MCP tool description", 1_024),
        ):
            _bounded_text(value, label, maximum=maximum)
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.local_name) is None:
            raise ValueError("MCP local tool name is not provider-safe")
        if not isinstance(self.discovery, ToolDiscoveryMetadata):
            raise TypeError("MCP tool discovery metadata is required")
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
    server_name: str
    server_version: str
    local_label: str
    maximum_outbound_sensitivity: ModelSensitivity
    tools: tuple[MCPToolBinding, ...]
    state: MCPBindingState
    revision: int
    admitted_at: datetime
    last_checked_at: datetime
    revoked_at: datetime | None = None
    stale_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.binding_id, str)
            or _BINDING_ID.fullmatch(self.binding_id) is None
        ):
            raise ValueError("MCP binding_id must use mcp-binding-<32 lowercase hex>")
        _bounded_text(self.agent_id, "MCP agent_id", maximum=256)
        object.__setattr__(self, "endpoint", normalize_mcp_endpoint(self.endpoint))
        if not isinstance(self.authentication, MCPAuthentication):
            raise TypeError("MCP binding authentication is invalid")
        if self.protocol_version not in MCP_SUPPORTED_PROTOCOL_VERSIONS:
            raise ValueError("MCP binding protocol version is unsupported")
        _server_identity(self.server_name, "MCP server name")
        _server_identity(self.server_version, "MCP server version")
        _bounded_text(self.local_label, "MCP local server label", maximum=128)
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
        object.__setattr__(self, "text", text_items)


class MCPClient(Protocol):
    async def inspect(self, *, observed_at: datetime) -> MCPServerInspection: ...

    async def call_tool(
        self,
        remote_name: str,
        arguments: Mapping[str, object],
    ) -> MCPToolResult: ...

    async def close(self) -> None: ...


class MCPClientFactory(Protocol):
    def create(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        secrets: SecretProvider,
    ) -> MCPClient: ...


class StreamableHTTPMCPClientFactory:
    """Create clients that all use the same production protocol boundary."""

    def __init__(
        self,
        *,
        http_transport: object | None = None,
        timeout_seconds: float = MCP_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not 0 < float(timeout_seconds) <= 60
        ):
            raise ValueError("MCP timeout must be positive and at most 60 seconds")
        self._http_transport = http_transport
        self._timeout_seconds = float(timeout_seconds)

    def create(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        secrets: SecretProvider,
    ) -> MCPClient:
        return StreamableHTTPMCPClient(
            endpoint=endpoint,
            authentication=authentication,
            secrets=secrets,
            http_transport=self._http_transport,
            timeout_seconds=self._timeout_seconds,
        )


class StreamableHTTPMCPClient:
    """Small JSON-RPC client for the accepted remote Streamable HTTP surface."""

    def __init__(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        secrets: SecretProvider,
        http_transport: object | None = None,
        timeout_seconds: float = MCP_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoint = normalize_mcp_endpoint(endpoint)
        if not isinstance(authentication, MCPAuthentication):
            raise TypeError("MCP authentication is invalid")
        if not isinstance(secrets, SecretProvider):
            raise TypeError("MCP secrets must implement SecretProvider")
        self._authentication = authentication
        self._secrets = secrets
        self._http_transport = http_transport
        self._timeout_seconds = timeout_seconds
        self._http_client: httpx.AsyncClient | None = None
        self._protocol_version: str | None = None
        self._server_name: str | None = None
        self._server_version: str | None = None
        self._session_id: str | None = None
        self._request_id = 0
        self._initialize_lock = asyncio.Lock()
        self._closed = False

    async def inspect(self, *, observed_at: datetime) -> MCPServerInspection:
        _aware(observed_at, "MCP observed_at")
        await self._initialize()
        tools: list[MCPInspectedTool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _page in range(MCP_MAX_DISCOVERY_PAGES):
            params = {} if cursor is None else {"cursor": cursor}
            result = await self._request("tools/list", params)
            raw_tools = result.get("tools")
            if not isinstance(raw_tools, (tuple, list)):
                raise MCPProtocolError(
                    "mcp_protocol_invalid",
                    "The MCP server returned an invalid tool list.",
                )
            for raw_tool in raw_tools:
                tools.append(_inspect_tool(raw_tool))
                if len(tools) > MCP_MAX_DISCOVERED_TOOLS:
                    raise MCPProtocolError(
                        "mcp_discovery_limit",
                        "The MCP server advertised more tools than the fixed bound.",
                    )
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                break
            if (
                not isinstance(next_cursor, str)
                or not next_cursor
                or len(next_cursor) > 1_024
                or next_cursor in seen_cursors
            ):
                raise MCPProtocolError(
                    "mcp_protocol_invalid",
                    "The MCP server returned an invalid pagination cursor.",
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            raise MCPProtocolError(
                "mcp_discovery_limit",
                "The MCP server tool list exceeded the fixed page bound.",
            )
        assert self._protocol_version is not None
        assert self._server_name is not None
        assert self._server_version is not None
        try:
            return MCPServerInspection(
                endpoint=self.endpoint,
                protocol_version=self._protocol_version,
                server_name=self._server_name,
                server_version=self._server_version,
                tools=tuple(tools),
                observed_at=observed_at,
            )
        except ValueError as error:
            raise MCPProtocolError(
                "mcp_protocol_invalid",
                "The MCP server returned invalid bounded identity or tool metadata.",
            ) from error

    async def call_tool(
        self,
        remote_name: str,
        arguments: Mapping[str, object],
    ) -> MCPToolResult:
        _remote_tool_name(remote_name)
        frozen_arguments = FrozenJsonObject.from_mapping(arguments)
        result = await self._request(
            "tools/call",
            {"name": remote_name, "arguments": frozen_arguments.to_dict()},
        )
        content = result.get("content")
        if not isinstance(content, (tuple, list)):
            raise MCPProtocolError(
                "mcp_result_malformed",
                "The MCP server returned a malformed tool result.",
            )
        if len(content) > MCP_MAX_RESULT_CONTENT_ITEMS:
            raise MCPProtocolError(
                "mcp_result_unsupported",
                "The MCP tool result contains too many content items.",
            )
        texts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "text":
                raise MCPProtocolError(
                    "mcp_result_unsupported",
                    "The MCP tool result contains an unsupported content type.",
                )
            text = block.get("text")
            if not isinstance(text, str) or len(text) > MCP_MAX_TEXT_CHARACTERS:
                raise MCPProtocolError(
                    "mcp_result_too_large",
                    "The MCP tool result text exceeded its fixed bound.",
                )
            texts.append(text)
        structured_raw = result.get("structuredContent")
        if structured_raw is not None and not isinstance(structured_raw, Mapping):
            raise MCPProtocolError(
                "mcp_result_malformed",
                "The MCP structured result must be a JSON object.",
            )
        is_error = result.get("isError", False)
        if not isinstance(is_error, bool):
            raise MCPProtocolError(
                "mcp_result_malformed",
                "The MCP tool error indicator is invalid.",
            )
        return MCPToolResult(
            text=tuple(texts),
            structured=(
                None
                if structured_raw is None
                else FrozenJsonObject.from_mapping(structured_raw)
            ),
            is_error=is_error,
        )

    async def close(self) -> None:
        self._closed = True
        client = self._http_client
        self._http_client = None
        if client is not None:
            await client.aclose()

    async def _initialize(self) -> None:
        if self._protocol_version is not None:
            return
        async with self._initialize_lock:
            if self._protocol_version is not None:
                return
            request_id = self._next_request_id()
            result, headers = await self._post_request(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": MCP_SUPPORTED_PROTOCOL_VERSIONS[0],
                        "capabilities": {},
                        "clientInfo": {"name": "daita", "version": "1.0.0"},
                    },
                },
                include_protocol=False,
                expected_id=request_id,
            )
            version = result.get("protocolVersion")
            server_info = result.get("serverInfo")
            capabilities = result.get("capabilities")
            if (
                version not in MCP_SUPPORTED_PROTOCOL_VERSIONS
                or not isinstance(server_info, Mapping)
                or not isinstance(capabilities, Mapping)
                or not isinstance(capabilities.get("tools"), Mapping)
            ):
                raise MCPProtocolError(
                    "mcp_protocol_unsupported",
                    "The MCP server does not fit the supported protocol surface.",
                )
            name = server_info.get("name")
            server_version = server_info.get("version")
            try:
                _server_identity(cast(str, name), "MCP server name")
                _server_identity(
                    cast(str, server_version),
                    "MCP server version",
                )
            except (TypeError, ValueError):
                raise MCPProtocolError(
                    "mcp_protocol_invalid",
                    "The MCP server identity is invalid.",
                ) from None
            session_id = headers.get("mcp-session-id")
            if session_id is not None and (
                not isinstance(session_id, str)
                or not session_id
                or len(session_id) > 1_024
                or any(character in session_id for character in "\r\n\x00")
            ):
                raise MCPProtocolError(
                    "mcp_protocol_invalid",
                    "The MCP server session identity is invalid.",
                )
            self._protocol_version = cast(str, version)
            self._server_name = cast(str, name)
            self._server_version = cast(str, server_version)
            self._session_id = session_id
            try:
                await self._post_notification("notifications/initialized", {})
            except BaseException:
                self._protocol_version = None
                self._server_name = None
                self._server_version = None
                self._session_id = None
                raise

    async def _request(
        self,
        method: str,
        params: Mapping[str, object],
    ) -> FrozenJsonObject:
        await self._initialize()
        request_id = self._next_request_id()
        result, _headers = await self._post_request(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params),
            },
            include_protocol=True,
            expected_id=request_id,
        )
        return FrozenJsonObject.from_mapping(result)

    async def _post_notification(
        self,
        method: str,
        params: Mapping[str, object],
    ) -> None:
        await self._post(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": dict(params),
            },
            include_protocol=True,
            expect_response=False,
        )

    async def _post_request(
        self,
        payload: Mapping[str, object],
        *,
        include_protocol: bool,
        expected_id: int | None = None,
    ) -> tuple[Mapping[str, object], Mapping[str, str]]:
        response, headers = await self._post(
            payload,
            include_protocol=include_protocol,
            expect_response=True,
        )
        if not isinstance(response, Mapping):
            raise MCPProtocolError(
                "mcp_protocol_invalid",
                "The MCP server returned malformed JSON-RPC data.",
            )
        if response.get("jsonrpc") != "2.0":
            raise MCPProtocolError(
                "mcp_protocol_invalid",
                "The MCP server returned an invalid JSON-RPC version.",
            )
        if expected_id is not None and response.get("id") != expected_id:
            raise MCPProtocolError(
                "mcp_protocol_invalid",
                "The MCP server returned a mismatched response identity.",
            )
        error = response.get("error")
        if error is not None:
            raise MCPProtocolError(
                "mcp_remote_protocol_error",
                "The MCP server rejected the protocol request.",
            )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise MCPProtocolError(
                "mcp_protocol_invalid",
                "The MCP server response omitted a result object.",
            )
        return result, headers

    async def _post(
        self,
        payload: Mapping[str, object],
        *,
        include_protocol: bool,
        expect_response: bool,
    ) -> tuple[object, Mapping[str, str]]:
        if self._closed:
            raise MCPTransportError(
                "mcp_client_closed",
                "The MCP client is closed.",
            )
        client = self._client()
        encoded_payload = json.dumps(
            dict(payload),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded_payload) > MCP_MAX_REQUEST_BYTES:
            raise MCPProtocolError(
                "mcp_request_too_large",
                "The MCP request exceeded its fixed byte bound.",
            )
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if include_protocol:
            assert self._protocol_version is not None
            headers["MCP-Protocol-Version"] = self._protocol_version
            if self._session_id is not None:
                headers["Mcp-Session-Id"] = self._session_id
        if self._authentication.mode is MCPAuthenticationMode.BEARER:
            assert self._authentication.secret_reference is not None
            try:
                token = await self._secrets.resolve(
                    self._authentication.secret_reference
                )
            except SecretResolutionError as error:
                raise MCPAuthenticationError(
                    "mcp_authentication_failed",
                    "The MCP bearer credential is unavailable.",
                ) from error
            if (
                not isinstance(token, str)
                or not token
                or len(token.encode("utf-8")) > 64 * 1_024
                or "\r" in token
                or "\n" in token
            ):
                raise MCPAuthenticationError(
                    "mcp_authentication_failed",
                    "The MCP bearer credential is invalid.",
                )
            headers["Authorization"] = f"Bearer {token}"
        body_parts: list[bytes] = []
        response_too_large = False
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with client.stream(
                    "POST",
                    self.endpoint,
                    content=encoded_payload,
                    headers=headers,
                ) as response:
                    status = int(response.status_code)
                    response_headers = {
                        key.lower(): value for key, value in response.headers.items()
                    }
                    response_size = 0
                    async for part in response.aiter_bytes():
                        response_size += len(part)
                        if response_size > MCP_MAX_RESPONSE_BYTES:
                            response_too_large = True
                            break
                        body_parts.append(part)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise MCPTransportError(
                "mcp_timeout",
                "The MCP request exceeded its fixed timeout.",
                retryability=ErrorRetryability.TRANSIENT,
            ) from None
        except Exception:
            raise MCPTransportError(
                "mcp_transport_failed",
                "The MCP endpoint could not be reached.",
                retryability=ErrorRetryability.TRANSIENT,
            ) from None
        if response_too_large:
            raise MCPProtocolError(
                "mcp_response_too_large",
                "The MCP response exceeded its fixed byte bound.",
            )
        if status in {401, 403}:
            raise MCPAuthenticationError(
                "mcp_authentication_failed",
                "The MCP endpoint rejected the configured authentication.",
            )
        if 300 <= status < 400:
            raise MCPTransportError(
                "mcp_redirect_rejected",
                "The MCP endpoint attempted a redirect.",
            )
        if status >= 400:
            raise MCPTransportError(
                "mcp_http_error",
                "The MCP endpoint returned an unsuccessful HTTP status.",
                {"status": status},
                retryability=(
                    ErrorRetryability.TRANSIENT
                    if status >= 500
                    else ErrorRetryability.PERMANENT
                ),
            )
        body = b"".join(body_parts)
        if not expect_response:
            if status != 202 or body:
                raise MCPProtocolError(
                    "mcp_protocol_invalid",
                    "The MCP server returned an invalid notification response.",
                )
            return {}, response_headers
        content_type = response_headers.get("content-type", "").split(";", 1)[0]
        try:
            if content_type == "application/json":
                decoded = json.loads(body)
            elif content_type == "text/event-stream":
                decoded = _decode_sse_response(body)
            else:
                raise MCPProtocolError(
                    "mcp_content_type_unsupported",
                    "The MCP endpoint returned an unsupported content type.",
                )
        except MCPError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            raise MCPProtocolError(
                "mcp_protocol_invalid",
                "The MCP endpoint returned malformed protocol data.",
            ) from None
        if _json_depth(decoded) > MCP_MAX_SCHEMA_DEPTH + 8:
            raise MCPProtocolError(
                "mcp_response_too_deep",
                "The MCP response exceeded its fixed nesting bound.",
            )
        return decoded, response_headers

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "Daita's remote MCP runtime dependency is unavailable. "
                    f"{repair_guidance()}"
                ) from None
            self._http_client = httpx.AsyncClient(
                follow_redirects=False,
                transport=cast(Any, self._http_transport),
                timeout=None,
            )
        return self._http_client

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id


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
) -> MCPServerBinding:
    if prior is not None:
        if prior.binding_id != binding_id or prior.agent_id != agent_id:
            raise MCPAdmissionError(
                "mcp_binding_identity_mismatch",
                "The existing MCP binding belongs to another identity.",
            )
        if (
            prior.endpoint != inspection.endpoint
            or prior.server_name != inspection.server_name
        ):
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
            "At least one exact MCP read tool must be selected.",
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
                capability_id=f"mcp.read:sha256:{capability_hash}",
                executor_id=executor_id,
                local_name=local_name,
                remote_name=selection.remote_name,
                description=selection.description,
                discovery=ToolDiscoveryMetadata(
                    summary=cast(str, selection.summary),
                    when_to_use=cast(str, selection.when_to_use),
                    keywords=selection.keywords,
                    exposure_class=selection.exposure_class,
                    eager_priority=selection.eager_priority,
                ),
                input_schema=inspected.input_schema,
                input_schema_digest=inspected.input_schema_digest,
                output_schema=inspected.output_schema,
                output_schema_digest=inspected.output_schema_digest,
                result_sensitivity=selection.result_sensitivity,
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
        local_label=resolved_local_label,
        maximum_outbound_sensitivity=maximum_outbound_sensitivity,
        tools=tuple(tools),
        state=MCPBindingState.ACTIVE,
        revision=revision,
        admitted_at=admitted_at,
        last_checked_at=inspection.observed_at,
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
        inspection.server_name != binding.server_name
        or inspection.server_version != binding.server_version
    ):
        return "server_identity_changed"
    discovered = {tool.remote_name: tool for tool in inspection.tools}
    for accepted in binding.tools:
        current = discovered.get(accepted.remote_name)
        if current is None:
            return f"tool_missing:{accepted.remote_name}"
        if not current.supported:
            return f"tool_schema_unsupported:{accepted.remote_name}"
        if (
            current.input_schema_digest != accepted.input_schema_digest
            or current.output_schema_digest != accepted.output_schema_digest
        ):
            return f"tool_schema_changed:{accepted.remote_name}"
    return None


def canonical_mcp_schema(
    schema: Mapping[str, object],
) -> tuple[FrozenJsonObject, str]:
    raw = FrozenJsonObject.from_mapping(schema)
    encoded = canonical_json(raw).encode("utf-8")
    if len(encoded) > MCP_MAX_SCHEMA_BYTES:
        raise ValueError("schema exceeds the fixed byte bound")
    if _json_depth(raw) > MCP_MAX_SCHEMA_DEPTH:
        raise ValueError("schema exceeds the fixed depth bound")
    _validate_schema_node(raw, root=True)
    projected = FrozenJsonObject.from_mapping(_strip_schema_annotations(raw))
    return projected, f"sha256:{sha256(encoded).hexdigest()}"


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
    try:
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
    )


def _validate_schema_node(value: Mapping[str, object], *, root: bool) -> None:
    allowed = _SCHEMA_ROOT_KEYS if root else _SCHEMA_RULE_KEYS
    unsupported = sorted(set(value) - allowed)
    if unsupported:
        raise ValueError(f"unsupported schema keyword: {unsupported[0]}")
    if root:
        dialect = value.get("$schema")
        if "$schema" in value and dialect != _JSON_SCHEMA_2020_12:
            raise ValueError("schema dialect is unsupported")
    schema_type = value.get("type")
    if root and schema_type != "object":
        raise ValueError("schema root must have type object")
    if schema_type is not None and schema_type not in _SCHEMA_TYPES:
        raise ValueError("schema type is unsupported")
    for annotation in _SCHEMA_ANNOTATION_KEYS:
        item = value.get(annotation)
        if item is not None and annotation != "examples" and not isinstance(item, str):
            raise ValueError(f"schema {annotation} must be text")
    properties = value.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("schema properties must be an object")
    if len(properties) > 128:
        raise ValueError("schema contains too many properties")
    for name, rule in properties.items():
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 128
            or not isinstance(rule, Mapping)
        ):
            raise ValueError("schema property declaration is invalid")
        _validate_schema_node(rule, root=False)
    required = value.get("required", [])
    if (
        not isinstance(required, (tuple, list))
        or any(not isinstance(name, str) for name in required)
        or len(required) != len(set(required))
        or any(name not in properties for name in required)
    ):
        raise ValueError("schema required entries are invalid")
    additional = value.get("additionalProperties", True)
    if not isinstance(additional, bool):
        raise ValueError("schema additionalProperties must be a boolean")
    items = value.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError("schema items must be an object rule")
        _validate_schema_node(items, root=False)
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        bound = value.get(key)
        if bound is not None and (
            not isinstance(bound, int) or isinstance(bound, bool) or bound < 0
        ):
            raise ValueError(f"schema {key} must be a non-negative integer")
    for minimum, maximum in (
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
        ("minimum", "maximum"),
    ):
        low = value.get(minimum)
        high = value.get(maximum)
        if (
            low is not None
            and high is not None
            and cast(float, low) > cast(float, high)
        ):
            raise ValueError(f"schema {minimum} cannot exceed {maximum}")
    for key in ("minimum", "maximum"):
        bound = value.get(key)
        if bound is not None and (
            not isinstance(bound, (int, float)) or isinstance(bound, bool)
        ):
            raise ValueError(f"schema {key} must be numeric")
    pattern = value.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str) or len(pattern) > 1_024:
            raise ValueError("schema pattern is invalid")
        try:
            re.compile(pattern)
        except re.error:
            raise ValueError("schema pattern is invalid") from None
    unique = value.get("uniqueItems")
    if unique is not None and not isinstance(unique, bool):
        raise ValueError("schema uniqueItems must be a boolean")
    enum = value.get("enum")
    if enum is not None:
        if not isinstance(enum, (tuple, list)) or not enum or len(enum) > 128:
            raise ValueError("schema enum must be a bounded non-empty array")
        try:
            FrozenJsonObject.from_mapping({"enum": enum})
        except (TypeError, ValueError):
            raise ValueError("schema enum contains unsupported JSON") from None


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


def _decode_sse_response(body: bytes) -> object:
    text = body.decode("utf-8")
    events: list[object] = []
    data_lines: list[str] = []
    for line in text.splitlines() + [""]:
        if not line:
            if data_lines:
                events.append(json.loads("\n".join(data_lines)))
                data_lines.clear()
            continue
        if line.startswith(":") or line.startswith("event:") or line.startswith("id:"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if len(events) != 1:
        raise ValueError("MCP SSE response must contain exactly one JSON-RPC result")
    return events[0]


def _json_depth(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, (tuple, list)):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


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
    "MCPError",
    "MCPInspectedTool",
    "MCPProtocolError",
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
    "StreamableHTTPMCPClient",
    "StreamableHTTPMCPClientFactory",
    "canonical_mcp_schema",
    "mcp_binding_drift_reason",
    "mcp_binding_from_inspection",
    "normalize_mcp_endpoint",
]
