from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from hashlib import sha256

import httpx2 as httpx
import pytest

from daita import __version__
from daita._json import canonical_json
from daita.adapters.mcp import (
    MCP_MAX_REQUEST_BYTES,
    MCP_MAX_RESPONSE_BYTES,
    MCPAuthentication,
    MCPAuthenticationError,
    MCPProtocolError,
    MCPProtocolPinnedClient,
    MCPTransportError,
    SDKMCPClientFactory,
)
from daita.adapters.mcp_sdk import SDKMCPClient
from daita.security import (
    CredentialSession,
    EmptySecretProvider,
    KeychainSecretProvider,
    SecretReference,
)
from tests.support.mcp import (
    MappingSecretProvider,
    MCPConformanceTransport,
    MCPFixtureIdentity,
    conformance_identities,
    mock_transport,
)

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


async def test_modern_sdk_discovery_optional_server_info_and_uncached_tools_list():
    identity = MCPFixtureIdentity(
        host="modern.fixture.test",
        server_name="display-only",
        server_version="1",
        protocol_version="2026-07-28",
        tools=[{"name": "lookup", "inputSchema": {"type": "object", "properties": {}}}],
        results={"lookup": {"content": [{"type": "text", "text": "modern"}]}},
        omit_server_info=True,
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    assert identity.request_methods == []
    try:
        first = await client.inspect(observed_at=NOW)
        second = await client.inspect(observed_at=NOW)
        assert first.protocol_version == second.protocol_version == "2026-07-28"
        assert first.server_name is None and first.server_version is None
        assert first.protocol_capabilities_digest is not None
        assert identity.request_methods.count("server/discover") == 1
        assert identity.request_methods.count("tools/list") == 2
        assert "initialize" not in identity.request_methods
        assert (await client.call_tool("lookup", {})).text == ("modern",)
    finally:
        await client.close()
        await client.close()
    assert isinstance(client, SDKMCPClient)
    assert client._owner is not None and client._owner.done()


@pytest.mark.parametrize(
    "fault",
    [
        "auth",
        "malformed",
        "network",
        "redirect",
        "empty400",
        "malformed400",
        "modern:-32001",
        "modern:-32020",
        "modern:-32021",
        "modern:-32022",
        "modern:-32042",
        "modern:-32602",
    ],
)
async def test_modern_probe_never_downgrades_after_non_method_failure(fault):
    identity = MCPFixtureIdentity(
        host="probe.fixture.test",
        server_name="probe",
        server_version="1",
        protocol_version="2026-07-28",
        tools=[],
        results={},
        bearer_token="private-probe-token" if fault == "auth" else None,
        malformed_method="server/discover" if fault == "malformed" else None,
    )
    calls: list[str] = []
    base = MCPConformanceTransport(identity)

    async def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if fault == "network" and request.method == "POST":
            raise httpx.ReadError("disconnected", request=request)
        if fault == "redirect" and request.method == "POST":
            return httpx.Response(
                307, headers={"location": identity.endpoint}, request=request
            )
        if fault in {"empty400", "malformed400"} and request.method == "POST":
            return httpx.Response(
                400,
                content=b"" if fault == "empty400" else b"{broken",
                headers={"content-type": "application/json"},
                request=request,
            )
        if fault.startswith("modern:") and request.method == "POST":
            payload = json.loads(request.content)
            assert payload["method"] == "server/discover"
            assert request.headers["mcp-protocol-version"] == "2026-07-28"
            identity.request_methods.append(payload["method"])
            return httpx.Response(
                400,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "error": {
                        "code": int(fault.split(":")[1]),
                        "message": "Rejected request",
                        "data": {"supported": ["2099-01-01"]},
                    },
                },
                request=request,
            )
        return await base(request)

    client = SDKMCPClientFactory(http_transport=httpx.MockTransport(transport)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        with pytest.raises(
            (MCPAuthenticationError, MCPProtocolError, MCPTransportError)
        ):
            await client.inspect(observed_at=NOW)
    finally:
        await client.close()
    assert "initialize" not in identity.request_methods
    assert calls.count("POST") == 1


@pytest.mark.parametrize("advertised", ["2025-11-25", "2025-06-18", "2024-11-05"])
async def test_explicit_legacy_advertisement_requires_an_admitted_handshake(advertised):
    identity = MCPFixtureIdentity(
        host="advertised-legacy.fixture.test",
        server_name="advertised-legacy",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[{"name": "lookup", "inputSchema": {"type": "object", "properties": {}}}],
        results={},
    )
    base = MCPConformanceTransport(identity)

    async def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and b"server/discover" in request.content:
            identity.request_methods.append("server/discover")
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": json.loads(request.content)["id"],
                    "result": {
                        "supportedVersions": [advertised],
                        "capabilities": {"tools": {}},
                        "resultType": "complete",
                        "ttlMs": 0,
                        "cacheScope": "public",
                    },
                },
                request=request,
            )
        return await base(request)

    client = SDKMCPClientFactory(http_transport=httpx.MockTransport(transport)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        if advertised == "2025-11-25":
            assert (
                await client.inspect(observed_at=NOW)
            ).protocol_version == "2025-11-25"
            assert identity.request_methods.count("initialize") == 1
        else:
            with pytest.raises(MCPProtocolError) as raised:
                await client.inspect(observed_at=NOW)
            assert raised.value.code == "mcp_protocol_unsupported"
            assert ("initialize" in identity.request_methods) == (
                advertised == "2025-06-18"
            )
    finally:
        await client.close()


@pytest.mark.parametrize("protocol", ["2025-06-18", "2025-11-25"])
@pytest.mark.parametrize("code", [-32000, -32005])
async def test_initial_http400_rpc_era_signal_uses_one_legacy_handshake(protocol, code):
    identity = MCPFixtureIdentity(
        host="http400-legacy.fixture.test",
        server_name="legacy",
        server_version="1",
        protocol_version=protocol,
        tools=[{"name": "lookup", "inputSchema": {"type": "object", "properties": {}}}],
        results={"lookup": {"content": [{"type": "text", "text": "accepted"}]}},
    )
    base = MCPConformanceTransport(identity)

    async def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and b"server/discover" in request.content:
            payload = json.loads(request.content)
            identity.request_methods.append("server/discover")
            return httpx.Response(
                400,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "error": {
                        "code": code,
                        "message": "Implementation-defined rejection",
                    },
                },
                request=request,
            )
        return await base(request)

    client = SDKMCPClientFactory(http_transport=httpx.MockTransport(transport)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        assert (await client.inspect(observed_at=NOW)).protocol_version == protocol
        await client.inspect(observed_at=NOW)
        assert (await client.call_tool("lookup", {})).text == ("accepted",)
    finally:
        await client.close()
    assert identity.request_methods == [
        "server/discover",
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/list",
        "tools/call",
    ]


@pytest.mark.parametrize("protocol", ["2026-07-28", "2025-11-25", "2025-06-18"])
async def test_admitted_protocol_is_pinned_before_client_entry(protocol):
    identity = MCPFixtureIdentity(
        host="pinned.fixture.test",
        server_name="pinned",
        server_version="1",
        protocol_version=protocol,
        tools=[],
        results={},
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    assert isinstance(client, MCPProtocolPinnedClient)
    client.bind_protocol(protocol)
    try:
        assert (await client.inspect(observed_at=NOW)).protocol_version == protocol
        with pytest.raises(ValueError):
            client.bind_protocol(protocol)
    finally:
        await client.close()
    assert identity.request_methods.count("server/discover") == (
        protocol == "2026-07-28"
    )
    assert identity.request_methods.count("initialize") == (protocol != "2026-07-28")


@pytest.mark.parametrize("protocol", ["2026-07-28", "2025-06-18"])
async def test_admitted_client_refuses_another_protocol_without_tools_dispatch(
    protocol,
):
    identity = MCPFixtureIdentity(
        host="changed-protocol.fixture.test",
        server_name="changed",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[],
        results={},
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    assert isinstance(client, MCPProtocolPinnedClient)
    client.bind_protocol(protocol)
    try:
        with pytest.raises(MCPProtocolError):
            await client.inspect(observed_at=NOW)
    finally:
        await client.close()
    assert "tools/list" not in identity.request_methods
    assert "tools/call" not in identity.request_methods
    assert identity.request_methods.count("initialize") == (protocol != "2026-07-28")


async def test_streamed_body_without_length_is_bounded_before_sdk_parse():
    identity = MCPFixtureIdentity(
        host="streamed.fixture.test",
        server_name="streamed",
        server_version="1",
        protocol_version="2025-06-18",
        tools=[{"name": "large", "inputSchema": {"type": "object", "properties": {}}}],
        results={},
    )
    base = MCPConformanceTransport(identity)

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * (MCP_MAX_RESPONSE_BYTES // 2)
            yield b"y" * (MCP_MAX_RESPONSE_BYTES // 2 + 1)
            raise AssertionError("The byte limit must stop the response stream early")

    async def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and b"tools/call" in request.content:
            identity.request_methods.append("tools/call")
            return httpx.Response(
                200,
                stream=Chunks(),
                headers={"content-type": "application/json"},
                request=request,
            )
        return await base(request)

    client = SDKMCPClientFactory(http_transport=httpx.MockTransport(transport)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        await client.inspect(observed_at=NOW)
        with pytest.raises(MCPProtocolError, match="fixed byte bound") as raised:
            await client.call_tool("large", {})
        assert raised.value.code == "mcp_response_too_large"
    finally:
        await client.close()
    assert identity.request_methods.count("tools/call") == 1


async def test_json_escaped_bearer_echo_is_rejected_before_sdk_parse():
    token = "private-token-echo"
    identity = MCPFixtureIdentity(
        host="escaped-echo.fixture.test",
        server_name="escaped-echo",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[{"name": "echo", "inputSchema": {"type": "object", "properties": {}}}],
        results={"echo": {"content": [{"type": "text", "text": token}]}},
        bearer_token=token,
    )
    base = MCPConformanceTransport(identity)

    async def transport(request: httpx.Request) -> httpx.Response:
        response = await base(request)
        if request.method == "POST" and b"tools/call" in request.content:
            body = response.content.replace(
                token.encode(), b"\\u0070" + token[1:].encode()
            )
            return httpx.Response(
                response.status_code,
                content=body,
                headers=response.headers,
                request=request,
            )
        return response

    client = SDKMCPClientFactory(http_transport=httpx.MockTransport(transport)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.bearer(
            SecretReference.environment("ECHO_TOKEN")
        ),
        secrets=MappingSecretProvider({"env:ECHO_TOKEN": token}),
    )
    try:
        await client.inspect(observed_at=NOW)
        with pytest.raises(MCPProtocolError) as raised:
            await client.call_tool("echo", {})
        assert raised.value.code == "mcp_credential_echo_rejected"
    finally:
        await client.close()


@pytest.mark.parametrize("scheme", ["env", "keychain"])
async def test_default_credential_session_resolves_rotated_bearer_per_request(
    monkeypatch, scheme
):
    class Keyring:
        value = "old-bearer"
        reads = 0

        def get_password(self, service, name):
            self.reads += 1
            return self.value

        def set_password(self, service, name, password):
            raise AssertionError("MCP must not write credentials")

        def delete_password(self, service, name):
            raise AssertionError("MCP must not delete credentials")

    keyring = Keyring()
    session = CredentialSession(KeychainSecretProvider(client=keyring))
    reference = (
        SecretReference.environment("ROTATING_MCP_TOKEN")
        if scheme == "env"
        else SecretReference.keychain("rotating-mcp")
    )
    monkeypatch.setenv("ROTATING_MCP_TOKEN", keyring.value)
    if scheme == "keychain":
        assert await session.resolve(reference) == "old-bearer"
    identity = MCPFixtureIdentity(
        host="rotating.fixture.test",
        server_name="rotating",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[],
        results={},
        bearer_token="old-bearer",
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.bearer(reference),
        secrets=session,
    )
    try:
        await client.inspect(observed_at=NOW)
        keyring.value = identity.bearer_token = "fresh-bearer"
        monkeypatch.setenv("ROTATING_MCP_TOKEN", keyring.value)
        await client.inspect(observed_at=NOW)
        if scheme == "keychain":
            assert keyring.reads >= 5
            assert await session.resolve(reference) == "old-bearer"
    finally:
        await client.close()
        await session.close()
    assert identity.request_methods.count("tools/list") == 2


async def test_two_fixture_identities_use_one_production_streamable_http_boundary():
    alpha, beta = conformance_identities()
    secrets = MappingSecretProvider({"env:BETA_TOKEN": "fixture-beta-secret"})
    factory = SDKMCPClientFactory(http_transport=mock_transport(alpha, beta))
    alpha_client = factory.create(
        endpoint=alpha.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=secrets,
    )
    beta_client = factory.create(
        endpoint=beta.endpoint,
        authentication=MCPAuthentication.bearer(
            SecretReference.environment("BETA_TOKEN")
        ),
        secrets=secrets,
    )
    try:
        alpha_inspection = await alpha_client.inspect(observed_at=NOW)
        beta_inspection = await beta_client.inspect(observed_at=NOW)
        assert alpha_inspection.server_name == "fixture-alpha"
        assert beta_inspection.server_name == "fixture-beta"
        assert alpha_inspection.tools[0].remote_name == "lookup"
        assert beta_inspection.tools[0].remote_name == "lookup"
        accepted_schema = alpha_inspection.tools[0].input_schema
        assert accepted_schema is not None
        assert "$schema" not in accepted_schema
        raw_input_schema = alpha.tool("lookup")["inputSchema"]
        assert isinstance(raw_input_schema, dict)
        assert alpha_inspection.tools[0].input_schema_digest == (
            "sha256:"
            + sha256(canonical_json(raw_input_schema).encode("utf-8")).hexdigest()
        )
        properties = accepted_schema.to_dict()["properties"]
        assert isinstance(properties, dict)
        query_rule = properties["query"]
        assert isinstance(query_rule, dict)
        assert "description" not in query_rule
        accepted_output = alpha_inspection.tools[0].output_schema
        assert accepted_output is not None
        assert "$schema" in accepted_output
        assert beta_inspection.tools[0].supported

        alpha_result = await alpha_client.call_tool("lookup", {"query": "x"})
        beta_result = await beta_client.call_tool("lookup", {"id": 1})
        assert alpha_result.structured is not None
        assert alpha_result.structured.to_dict() == {"answer": "alpha"}
        assert beta_result.text == ("beta",)
        assert alpha.initialize_client_info == {
            "name": "daita",
            "version": __version__,
        }
        assert secrets.resolutions
        assert set(beta.request_methods) >= {
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
        }
    finally:
        await alpha_client.close()
        await beta_client.close()


async def test_inspection_marks_remote_ref_unsupported_without_call_authority():
    identity = MCPFixtureIdentity(
        host="unsupported.fixture.test",
        server_name="unsupported-fixture",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[
            {
                "name": "remote_ref",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"$ref": "https://invalid/schema"}},
                },
            }
        ],
        results={},
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        inspection = await client.inspect(observed_at=NOW)
        assert not inspection.tools[0].supported
        assert inspection.tools[0].unsupported_reason == (
            "only local JSON Pointer schema references are supported"
        )
        assert identity.calls == []
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("input_schema", "reason"),
    (
        (
            {
                "$schema": "http://json-schema.org/draft-04/schema#",
                "type": "object",
                "properties": {},
            },
            "schema dialect is unsupported",
        ),
        (
            {"$schema": None, "type": "object", "properties": {}},
            "schema dialect is unsupported",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "nested": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {},
                    }
                },
            },
            "nested schema dialect change is unsupported",
        ),
    ),
)
async def test_inspection_rejects_other_or_nested_schema_dialects(
    input_schema,
    reason,
):
    identity = MCPFixtureIdentity(
        host="dialect.fixture.test",
        server_name="dialect-fixture",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[{"name": "dialect", "inputSchema": input_schema}],
        results={},
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        inspection = await client.inspect(observed_at=NOW)
        assert not inspection.tools[0].supported
        assert inspection.tools[0].unsupported_reason == reason
    finally:
        await client.close()


async def test_malformed_and_unsupported_media_results_are_typed_and_bounded():
    identity = MCPFixtureIdentity(
        host="malformed.fixture.test",
        server_name="malformed-fixture",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[
            {
                "name": "media",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
        results={
            "media": {
                "content": [
                    {"type": "image", "data": "SECRET-BYTES", "mimeType": "x/y"}
                ]
            }
        },
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        await client.inspect(observed_at=NOW)
        with pytest.raises(MCPProtocolError) as raised:
            await client.call_tool("media", {})
        assert raised.value.code == "mcp_result_unsupported"
        assert "SECRET-BYTES" not in str(raised.value)

        identity.malformed_method = "tools/list"
        with pytest.raises(MCPProtocolError) as malformed:
            await client.inspect(observed_at=NOW)
        assert malformed.value.code == "mcp_protocol_invalid"
    finally:
        await client.close()


async def test_timeout_and_cancellation_do_not_retry_remote_tool_calls():
    identity = MCPFixtureIdentity(
        host="timeout.fixture.test",
        server_name="timeout-fixture",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[
            {
                "name": "wait",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
        results={"wait": {"content": [{"type": "text", "text": "done"}]}},
        block_calls=asyncio.Event(),
    )
    client = SDKMCPClientFactory(
        http_transport=mock_transport(identity),
        timeout_seconds=0.02,
    ).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        await client.inspect(observed_at=NOW)
        with pytest.raises(MCPTransportError) as raised:
            await client.call_tool("wait", {})
        assert raised.value.code == "mcp_timeout"
        assert identity.request_methods.count("tools/call") == 1

        cancelling = asyncio.create_task(client.call_tool("wait", {}))
        while identity.request_methods.count("tools/call") < 2:
            await asyncio.sleep(0)
        cancelling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelling
        assert identity.request_methods.count("tools/call") == 2
    finally:
        assert identity.block_calls is not None
        identity.block_calls.set()
        await client.close()


async def test_cancelled_queued_action_never_reaches_the_sdk():
    gate = asyncio.Event()
    identity = MCPFixtureIdentity(
        host="queued-cancellation.fixture.test",
        server_name="queued-cancellation",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[{"name": "wait", "inputSchema": {"type": "object", "properties": {}}}],
        results={"wait": {"content": [{"type": "text", "text": "done"}]}},
        block_calls=gate,
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        await client.inspect(observed_at=NOW)
        first = asyncio.create_task(client.call_tool("wait", {}))
        while identity.request_methods.count("tools/call") < 1:
            await asyncio.sleep(0)
        queued = asyncio.create_task(client.call_tool("wait", {}))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        gate.set()
        assert (await first).text == ("done",)
        assert identity.request_methods.count("tools/call") == 1
    finally:
        gate.set()
        await client.close()


async def test_sdk_legacy_initialize_notification_is_not_a_second_negotiation():
    identity = MCPFixtureIdentity(
        host="initialize-retry.fixture.test",
        server_name="initialize-retry-fixture",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[],
        results={},
        initialized_notification_failures=1,
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        inspection = await client.inspect(observed_at=NOW)
        assert inspection.server_name == identity.server_name
        assert identity.request_methods.count("initialize") == 1
        assert identity.request_methods.count("notifications/initialized") == 1
        assert identity.request_methods[-1] == "tools/list"
    finally:
        await client.close()


async def test_concurrent_and_cancelled_close_wait_for_the_sdk_owner_to_drain():
    gate = asyncio.Event()
    identity = MCPFixtureIdentity(
        host="close-drain.fixture.test",
        server_name="close-drain",
        server_version="1",
        protocol_version="2026-07-28",
        tools=[{"name": "wait", "inputSchema": {"type": "object", "properties": {}}}],
        results={"wait": {"content": [{"type": "text", "text": "done"}]}},
        block_calls=gate,
    )
    client = SDKMCPClientFactory(
        http_transport=mock_transport(identity), timeout_seconds=1
    ).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        await client.inspect(observed_at=NOW)
        calling = asyncio.create_task(client.call_tool("wait", {}))
        while identity.request_methods.count("tools/call") < 1:
            await asyncio.sleep(0)
        closing = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        also_closing = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        assert not closing.done() and not also_closing.done()
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        gate.set()
        assert (await calling).text == ("done",)
        with pytest.raises(asyncio.CancelledError):
            await closing
        await also_closing
        assert isinstance(client, SDKMCPClient)
        assert client._owner is not None and client._owner.done()
        assert identity.request_methods.count("tools/call") == 1
    finally:
        gate.set()
        await client.close()


async def test_request_and_streamed_response_byte_bounds_precede_tool_results():
    identity = MCPFixtureIdentity(
        host="wire-bounds.fixture.test",
        server_name="wire-bounds-fixture",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[
            {
                "name": "bounded",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            }
        ],
        results={
            "bounded": {
                "content": [{"type": "text", "text": "x" * MCP_MAX_RESPONSE_BYTES}]
            }
        },
    )
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        await client.inspect(observed_at=NOW)
        calls_before = identity.request_methods.count("tools/call")
        with pytest.raises(MCPProtocolError) as oversized_request:
            await client.call_tool(
                "bounded",
                {"value": "x" * MCP_MAX_REQUEST_BYTES},
            )
        assert oversized_request.value.code == "mcp_request_too_large"
        assert identity.request_methods.count("tools/call") == calls_before

        with pytest.raises(MCPProtocolError) as oversized_response:
            await client.call_tool("bounded", {"value": "small"})
        assert oversized_response.value.code == "mcp_response_too_large"
        assert identity.request_methods.count("tools/call") == calls_before + 1
    finally:
        await client.close()


@pytest.mark.parametrize("token", ['private-"token', "private-\\token"])
async def test_standard_json_escaped_credential_echo_never_reaches_sdk_logs(
    token, caplog
):
    import logging

    identity = MCPFixtureIdentity(
        host="simple-escaped-echo.fixture.test",
        server_name="fixture",
        server_version="1",
        protocol_version="2026-07-28",
        bearer_token=token,
        tools=[{"name": "echo", "inputSchema": {"type": "object", "properties": {}}}],
        results={"echo": {"content": [{"type": "text", "text": token}]}},
    )
    caplog.set_level(logging.DEBUG)
    client = SDKMCPClientFactory(http_transport=mock_transport(identity)).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.bearer(
            SecretReference.environment("FIXTURE_TOKEN")
        ),
        secrets=MappingSecretProvider({"env:FIXTURE_TOKEN": token}),
    )
    try:
        await client.inspect(observed_at=NOW)
        with pytest.raises(MCPProtocolError) as raised:
            await client.call_tool("echo", {})
        assert raised.value.code == "mcp_credential_echo_rejected"
        assert token not in str(raised.value)
        assert token not in caplog.text
    finally:
        await client.close()
