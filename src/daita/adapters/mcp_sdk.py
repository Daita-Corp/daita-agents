"""Official MCP SDK transport adapter. Protocol framing belongs to the SDK."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from contextlib import AsyncExitStack
from datetime import datetime
from hashlib import sha256
from typing import Any, cast
from urllib.parse import urlsplit

import httpx2
from mcp import Client, types
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError as SDKError

from .._json import FrozenJsonObject, canonical_json
from .._version import __version__
from ..errors import ErrorRetryability
from ..security import SecretProvider, SecretResolutionError
from .mcp import (
    _PERSONAL_STATUS_MESSAGES,
    MCP_MAX_DISCOVERED_TOOLS,
    MCP_MAX_DISCOVERY_PAGES,
    MCP_MAX_REQUEST_BYTES,
    MCP_MAX_RESPONSE_BYTES,
    MCP_MAX_RESULT_CONTENT_ITEMS,
    MCP_MAX_TEXT_CHARACTERS,
    MCP_REQUEST_TIMEOUT_SECONDS,
    MCP_SUPPORTED_PROTOCOL_VERSIONS,
    MCPAuthentication,
    MCPAuthenticationError,
    MCPAuthenticationMode,
    MCPConnectionProvider,
    MCPProtocolError,
    MCPServerInspection,
    MCPToolResult,
    MCPTransportError,
    _aware,
    _inspect_tool,
    _remote_tool_name,
    _require_personal_resource_origin,
    check_personal_connection,
    normalize_mcp_endpoint,
)

_MODERN_VERSION = "2026-07-28"
_METHOD_NOT_FOUND = -32601
_JSON_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _contains_credential(body: bytes, token: str) -> bool:
    if token.encode("utf-8") in body:
        return True
    if b"\\" not in body:
        return False
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    unescaped = _JSON_UNICODE_ESCAPE.sub(
        lambda match: chr(int(match.group(1), 16)), text
    ).replace("\\/", "/")
    # JSON represents non-BMP characters as a UTF-16 surrogate pair.
    unescaped = unescaped.encode("utf-16", "surrogatepass").decode(
        "utf-16", "surrogatepass"
    )
    return token in unescaped


class _BoundedMCPHTTPClient(httpx2.AsyncClient):
    """One request boundary for SDK traffic, credentials, redirects and bytes."""

    def __init__(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        secrets: SecretProvider,
        timeout_seconds: float,
        transport: httpx2.AsyncBaseTransport | None,
    ) -> None:
        super().__init__(
            transport=transport,
            timeout=httpx2.Timeout(timeout_seconds, connect=min(timeout_seconds, 5.0)),
            max_redirects=0,
            follow_redirects=False,
            trust_env=False,
        )
        self._endpoint = endpoint
        self._authentication = authentication
        self._secrets = secrets
        self._provider: MCPConnectionProvider | None = None
        self._principal_id: str | None = None
        self.wire_fault: (
            MCPProtocolError | MCPTransportError | MCPAuthenticationError | None
        ) = None

    def _reject(
        self,
        request: httpx2.Request,
        error: MCPProtocolError | MCPTransportError | MCPAuthenticationError,
        status: int = 400,
    ) -> httpx2.Response:
        self.wire_fault = error
        return httpx2.Response(status, request=request)

    def bind_personal_connection(
        self, provider: MCPConnectionProvider, principal_id: str
    ) -> None:
        self._provider = provider
        self._principal_id = principal_id

    async def _credential(self) -> str | None:
        auth = self._authentication
        if auth.mode is MCPAuthenticationMode.NONE:
            return None
        if auth.mode is MCPAuthenticationMode.BEARER:
            assert auth.secret_reference is not None
            try:
                token = await self._secrets.resolve(auth.secret_reference)
            except SecretResolutionError:
                raise MCPAuthenticationError(
                    "mcp_authentication_failed",
                    "The MCP bearer credential is unavailable.",
                ) from None
        else:
            await check_personal_connection(auth, self._provider, self._principal_id)
            assert self._provider is not None
            assert self._principal_id is not None
            assert auth.connection_id is not None
            assert auth.resource_uri is not None
            try:
                token = await self._provider.access_token(
                    connection_id=auth.connection_id,
                    principal_id=self._principal_id,
                    resource_uri=auth.resource_uri,
                    required_scopes=auth.required_scopes,
                )
            except MCPAuthenticationError as error:
                code = (
                    error.code
                    if error.code in _PERSONAL_STATUS_MESSAGES
                    else "account_unavailable"
                )
                raise MCPAuthenticationError(
                    code, _PERSONAL_STATUS_MESSAGES[code]
                ) from None
            except Exception:
                raise MCPAuthenticationError(
                    "account_unavailable",
                    _PERSONAL_STATUS_MESSAGES["account_unavailable"],
                ) from None
        if (
            not isinstance(token, str)
            or not token
            or len(token.encode("utf-8")) > 64 * 1024
            or "\r" in token
            or "\n" in token
        ):
            raise MCPAuthenticationError(
                (
                    "account_unavailable"
                    if auth.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
                    else "mcp_authentication_failed"
                ),
                "The MCP credential is invalid.",
            )
        return token

    async def send(self, request: httpx2.Request, **kwargs: Any) -> httpx2.Response:
        target = urlsplit(str(request.url))
        pinned = urlsplit(self._endpoint)
        if (target.scheme, target.hostname, target.port, target.path, target.query) != (
            pinned.scheme,
            pinned.hostname,
            pinned.port,
            pinned.path,
            "",
        ):
            return self._reject(
                request,
                MCPTransportError(
                    "mcp_redirect_rejected", "The MCP endpoint changed origin or path."
                ),
            )
        if request.method == "GET":
            # Daita does not subscribe, resume, poll or accept server initiated
            # requests. The SDK's optional legacy GET stream is unnecessary.
            return httpx2.Response(405, request=request)
        try:
            body = request.content
        except httpx2.RequestNotRead:
            request_parts: list[bytes] = []
            total = 0
            async for part in cast(httpx2.AsyncByteStream, request.stream):
                total += len(part)
                if total > MCP_MAX_REQUEST_BYTES:
                    return self._reject(
                        request,
                        MCPProtocolError(
                            "mcp_request_too_large",
                            "The MCP request exceeded its fixed byte bound.",
                        ),
                    )
                request_parts.append(part)
            body = b"".join(request_parts)
            headers = dict(request.headers)
            headers.pop("content-length", None)
            headers.pop("transfer-encoding", None)
            request = httpx2.Request(
                request.method,
                request.url,
                headers=headers,
                content=body,
                extensions=request.extensions,
            )
        if len(body) > MCP_MAX_REQUEST_BYTES:
            return self._reject(
                request,
                MCPProtocolError(
                    "mcp_request_too_large",
                    "The MCP request exceeded its fixed byte bound.",
                ),
            )
        try:
            token = await self._credential()
        except MCPAuthenticationError as error:
            return self._reject(request, error, 401)
        request.headers.pop("Authorization", None)
        if token is not None:
            request.headers["Authorization"] = f"Bearer {token}"
        # Even the SDK's same-origin redirect helper must not send a second request.
        kwargs["follow_redirects"] = False
        # httpx2 otherwise buffers the full body before our byte limit sees it.
        kwargs["stream"] = True
        try:
            response = await super().send(request, **kwargs)
        except httpx2.TimeoutException:
            return self._reject(
                request,
                MCPTransportError(
                    "mcp_timeout", "The MCP request exceeded its fixed timeout."
                ),
                504,
            )
        except httpx2.TransportError:
            return self._reject(
                request,
                MCPTransportError(
                    "mcp_transport_failed", "The MCP endpoint could not be reached."
                ),
                503,
            )
        if token is not None and any(
            token in value for value in response.headers.values()
        ):
            await response.aclose()
            return self._reject(
                request,
                MCPProtocolError(
                    "mcp_credential_echo_rejected",
                    "The MCP endpoint echoed a credential.",
                ),
            )
        if 300 <= response.status_code < 400:
            await response.aclose()
            return self._reject(
                request,
                MCPTransportError(
                    "mcp_redirect_rejected", "The MCP endpoint attempted a redirect."
                ),
            )
        if response.status_code in {401, 403}:
            await response.aclose()
            return self._reject(
                request,
                MCPAuthenticationError(
                    "mcp_authentication_failed",
                    "The MCP endpoint rejected authentication.",
                ),
                401,
            )
        if response.headers.get("content-encoding", "identity") != "identity":
            await response.aclose()
            return self._reject(
                request,
                MCPProtocolError(
                    "mcp_response_unsupported",
                    "Compressed MCP responses are unsupported.",
                ),
            )
        length = response.headers.get("content-length")
        if (
            length is not None
            and length.isdecimal()
            and int(length) > MCP_MAX_RESPONSE_BYTES
        ):
            await response.aclose()
            return self._reject(
                request,
                MCPProtocolError(
                    "mcp_response_too_large",
                    "The MCP response exceeded its fixed byte bound.",
                ),
            )
        parts: list[bytes] = []
        total = 0
        tail = b""
        token_bytes = None if token is None else token.encode("utf-8")
        try:
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MCP_MAX_RESPONSE_BYTES:
                    return self._reject(
                        request,
                        MCPProtocolError(
                            "mcp_response_too_large",
                            "The MCP response exceeded its fixed byte bound.",
                        ),
                    )
                if token_bytes is not None and token_bytes in tail + chunk:
                    return self._reject(
                        request,
                        MCPProtocolError(
                            "mcp_credential_echo_rejected",
                            "The MCP endpoint echoed a credential.",
                        ),
                    )
                if token_bytes is not None:
                    tail = (
                        (tail + chunk)[-(len(token_bytes) - 1) :]
                        if len(token_bytes) > 1
                        else b""
                    )
                parts.append(chunk)
        except httpx2.TimeoutException:
            return self._reject(
                request,
                MCPTransportError(
                    "mcp_timeout", "The MCP request exceeded its fixed timeout."
                ),
                504,
            )
        except httpx2.TransportError:
            return self._reject(
                request,
                MCPTransportError(
                    "mcp_transport_failed", "The MCP endpoint could not be reached."
                ),
                503,
            )
        finally:
            await response.aclose()
        body = b"".join(parts)
        if token is not None and _contains_credential(body, token):
            return self._reject(
                request,
                MCPProtocolError(
                    "mcp_credential_echo_rejected",
                    "The MCP endpoint echoed a credential.",
                ),
            )
        return httpx2.Response(
            response.status_code,
            headers=response.headers,
            content=body,
            request=request,
        )


def _safe_error(error: BaseException) -> Exception:
    if isinstance(error, (MCPProtocolError, MCPTransportError, MCPAuthenticationError)):
        return error
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            converted = _safe_error(nested)
            if (
                not isinstance(converted, MCPTransportError)
                or converted.code != "mcp_transport_failed"
            ):
                return converted
    if isinstance(error, SDKError):
        if error.code in {-32700, -32600}:
            return MCPProtocolError(
                "mcp_protocol_invalid",
                "The MCP server returned malformed protocol data.",
            )
        return MCPProtocolError(
            "mcp_remote_protocol_error", "The MCP server rejected a protocol request."
        )
    if isinstance(error, (TimeoutError, httpx2.TimeoutException)):
        return MCPTransportError(
            "mcp_timeout",
            "The MCP request exceeded its fixed timeout.",
            retryability=ErrorRetryability.TRANSIENT,
        )
    return MCPTransportError(
        "mcp_transport_failed",
        "The MCP endpoint could not be reached.",
        retryability=ErrorRetryability.TRANSIENT,
    )


class SDKMCPClient:
    """Serialize all SDK client lifecycle and calls in one owning asyncio task."""

    def __init__(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        secrets: SecretProvider,
        http_transport: httpx2.AsyncBaseTransport | None = None,
        timeout_seconds: float = MCP_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoint = normalize_mcp_endpoint(endpoint)
        _require_personal_resource_origin(self.endpoint, authentication)
        self._authentication = authentication
        self._secrets = secrets
        self._transport = http_transport
        self._timeout = float(timeout_seconds)
        self._provider: MCPConnectionProvider | None = None
        self._principal_id: str | None = None
        self._queue: asyncio.Queue[tuple[str, tuple[Any, ...], asyncio.Future[Any]]] = (
            asyncio.Queue()
        )
        self._owner: asyncio.Task[None] | None = None
        self._closed = False

    def bind_personal_connection(
        self, provider: MCPConnectionProvider, principal_id: str
    ) -> None:
        if self._authentication.mode is not MCPAuthenticationMode.PERSONAL_CONNECTION:
            raise ValueError("MCP client has no personal connection")
        if principal_id != self._authentication.owner_principal_id:
            raise MCPAuthenticationError(
                "needs_authorization", "The MCP connection is unavailable."
            )
        if self._provider is not None or self._owner is not None or self._closed:
            raise ValueError("MCP personal connection must be bound once before use")
        self._provider = provider
        self._principal_id = principal_id

    async def _submit(self, kind: str, *args: Any) -> Any:
        if self._closed:
            raise MCPTransportError("mcp_client_closed", "The MCP client is closed.")
        if self._owner is None or self._owner.done():
            self._owner = asyncio.create_task(self._run(), name="daita-mcp-sdk-owner")
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._queue.put((kind, args, future))
        return await future

    async def inspect(self, *, observed_at: datetime) -> MCPServerInspection:
        _aware(observed_at, "MCP observed_at")
        return cast(MCPServerInspection, await self._submit("inspect", observed_at))

    async def call_tool(
        self, remote_name: str, arguments: Mapping[str, object]
    ) -> MCPToolResult:
        _remote_tool_name(remote_name)
        frozen = FrozenJsonObject.from_mapping(arguments)
        if len(canonical_json(frozen).encode("utf-8")) > MCP_MAX_REQUEST_BYTES - 4096:
            raise MCPProtocolError(
                "mcp_request_too_large",
                "The MCP request exceeded its fixed byte bound.",
            )
        return cast(
            MCPToolResult, await self._submit("call", remote_name, frozen.to_dict())
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        owner = self._owner
        if owner is None:
            return
        if owner.done():
            return
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._queue.put(("close", (), future))
        await asyncio.shield(owner)

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                http_client = await stack.enter_async_context(
                    _BoundedMCPHTTPClient(
                        endpoint=self.endpoint,
                        authentication=self._authentication,
                        secrets=self._secrets,
                        timeout_seconds=self._timeout,
                        transport=self._transport,
                    )
                )
                if self._provider is not None and self._principal_id is not None:
                    http_client.bind_personal_connection(
                        self._provider, self._principal_id
                    )
                sdk: Client | None = None
                while True:
                    kind, args, future = await self._queue.get()
                    if kind == "close":
                        if not future.done():
                            future.set_result(None)
                        return
                    if future.cancelled():
                        # A queued command that lost its caller never dispatches.
                        continue
                    result: MCPServerInspection | MCPToolResult
                    try:
                        http_client.wire_fault = None
                        async with asyncio.timeout(self._timeout):
                            if sdk is None:
                                sdk = await self._connect(stack, http_client)
                            if kind == "inspect":
                                result = await self._inspect(
                                    sdk, cast(datetime, args[0])
                                )
                            else:
                                result = await self._call(
                                    sdk,
                                    cast(str, args[0]),
                                    cast(dict[str, Any], args[1]),
                                )
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        if not future.done():
                            future.set_exception(
                                http_client.wire_fault or _safe_error(error)
                            )
                    else:
                        if not future.done():
                            future.set_result(result)
        except BaseException as error:
            while not self._queue.empty():
                _, _, future = self._queue.get_nowait()
                if not future.done():
                    future.set_exception(_safe_error(error))
            if not self._closed:
                # The SDK transport can terminate its task group after a bounded
                # response or network failure. A later command may open a fresh
                # owner; the failed command itself is never replayed.
                return

    async def _connect(
        self, stack: AsyncExitStack, http_client: _BoundedMCPHTTPClient
    ) -> Client:
        modern = Client(
            streamable_http_client(self.endpoint, http_client=http_client),
            mode=_MODERN_VERSION,
            client_info=types.Implementation(name="daita", version=__version__),
            read_timeout_seconds=self._timeout,
            cache=None,
        )
        async with AsyncExitStack() as probe_stack:
            await probe_stack.enter_async_context(modern)
            admitted_legacy_versions: frozenset[str] = frozenset()
            try:
                raw = await modern.session.send_discover(_MODERN_VERSION)
            except SDKError as error:
                if http_client.wire_fault is not None:
                    raise http_client.wire_fault
                if error.code != _METHOD_NOT_FOUND:
                    raise
                admitted_legacy_versions = frozenset(
                    MCP_SUPPORTED_PROTOCOL_VERSIONS[1:]
                )
            else:
                discovered = types.DiscoverResult.model_validate(raw)
                if _MODERN_VERSION in discovered.supported_versions:
                    modern.session.adopt(discovered)
                    stack.push_async_exit(probe_stack.pop_all().__aexit__)
                    return modern
                admitted_legacy_versions = frozenset(
                    discovered.supported_versions
                ).intersection(MCP_SUPPORTED_PROTOCOL_VERSIONS[1:])
                if not admitted_legacy_versions:
                    raise MCPProtocolError(
                        "mcp_protocol_unsupported",
                        "The MCP protocol version is unsupported.",
                    )
        legacy_client = Client(
            streamable_http_client(self.endpoint, http_client=http_client),
            mode="legacy",
            client_info=types.Implementation(name="daita", version=__version__),
            read_timeout_seconds=self._timeout,
            cache=None,
        )
        await stack.enter_async_context(legacy_client)
        if legacy_client.session.protocol_version not in admitted_legacy_versions:
            raise MCPProtocolError(
                "mcp_protocol_unsupported", "The MCP protocol version is unsupported."
            )
        return legacy_client

    async def _inspect(self, sdk: Client, observed_at: datetime) -> MCPServerInspection:
        tools = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(MCP_MAX_DISCOVERY_PAGES):
            page = await sdk.session.list_tools(
                params=types.PaginatedRequestParams(cursor=cursor)
            )
            if page.result_type != "complete":
                raise MCPProtocolError(
                    "mcp_result_unsupported", "The MCP tool list is not final."
                )
            for tool in page.tools:
                tools.append(
                    _inspect_tool(
                        tool.model_dump(by_alias=True, mode="json", exclude_none=True)
                    )
                )
                if len(tools) > MCP_MAX_DISCOVERED_TOOLS:
                    raise MCPProtocolError(
                        "mcp_discovery_limit",
                        "The MCP tool list exceeded its fixed bound.",
                    )
            cursor = page.next_cursor
            if cursor is None:
                break
            if not cursor or len(cursor) > 1024 or cursor in seen:
                raise MCPProtocolError(
                    "mcp_protocol_invalid", "The MCP pagination cursor is invalid."
                )
            seen.add(cursor)
        else:
            raise MCPProtocolError(
                "mcp_discovery_limit",
                "The MCP tool list exceeded its fixed page bound.",
            )
        info = sdk.session.server_info
        capabilities = sdk.session.server_capabilities
        if capabilities is None or capabilities.tools is None:
            raise MCPProtocolError(
                "mcp_protocol_unsupported",
                "The MCP server does not advertise tools capability.",
            )
        relevant_capabilities = {
            "tools": capabilities.tools.model_dump(
                by_alias=True, mode="json", exclude_none=True
            )
        }
        return MCPServerInspection(
            endpoint=self.endpoint,
            protocol_version=cast(str, sdk.session.protocol_version),
            server_name=info.name if info is not None else None,
            server_version=info.version if info is not None else None,
            tools=tuple(tools),
            observed_at=observed_at,
            protocol_capabilities_digest="sha256:"
            + sha256(canonical_json(relevant_capabilities).encode("utf-8")).hexdigest(),
        )

    async def _call(
        self, sdk: Client, remote_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        result = await sdk.session.call_tool(
            remote_name, arguments, allow_input_required=True
        )
        if isinstance(result, types.InputRequiredResult):
            return MCPToolResult(accepted_async=True)
        if (
            not isinstance(result, types.CallToolResult)
            or result.result_type != "complete"
        ):
            return MCPToolResult(accepted_async=True)
        if len(result.content) > MCP_MAX_RESULT_CONTENT_ITEMS:
            raise MCPProtocolError(
                "mcp_result_unsupported",
                "The MCP tool result contains too many content items.",
            )
        texts = []
        for block in result.content:
            if not isinstance(block, types.TextContent):
                raise MCPProtocolError(
                    "mcp_result_unsupported",
                    "The MCP result contains unsupported content.",
                )
            if len(block.text) > MCP_MAX_TEXT_CHARACTERS:
                raise MCPProtocolError(
                    "mcp_result_too_large",
                    "The MCP result text exceeded its fixed bound.",
                )
            texts.append(block.text)
        if result.structured_content is not None and not isinstance(
            result.structured_content, Mapping
        ):
            raise MCPProtocolError(
                "mcp_result_malformed", "The MCP structured result must be an object."
            )
        return MCPToolResult(
            text=tuple(texts),
            structured=(
                None
                if result.structured_content is None
                else FrozenJsonObject.from_mapping(result.structured_content)
            ),
            is_error=result.is_error,
        )
