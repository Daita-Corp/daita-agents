from __future__ import annotations

import asyncio
import builtins
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from _mcp_fixtures import (
    MappingSecretProvider,
    MCPFixtureIdentity,
    conformance_identities,
    mock_transport,
)

from daita import __version__
from daita._json import canonical_json
from daita.adapters.mcp import (
    MCP_MAX_REQUEST_BYTES,
    MCP_MAX_RESPONSE_BYTES,
    MCPAuthentication,
    MCPProtocolError,
    MCPTransportError,
    StreamableHTTPMCPClient,
    StreamableHTTPMCPClientFactory,
)
from daita.security import EmptySecretProvider, SecretReference

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


async def test_two_fixture_identities_use_one_production_streamable_http_boundary():
    alpha, beta = conformance_identities()
    secrets = MappingSecretProvider({"env:BETA_TOKEN": "fixture-beta-secret"})
    factory = StreamableHTTPMCPClientFactory(http_transport=mock_transport(alpha, beta))
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
        assert "$schema" not in accepted_output
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
    client = StreamableHTTPMCPClientFactory(
        http_transport=mock_transport(identity)
    ).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        inspection = await client.inspect(observed_at=NOW)
        assert not inspection.tools[0].supported
        assert inspection.tools[0].unsupported_reason == (
            "unsupported schema keyword: $ref"
        )
        assert identity.calls == []
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("input_schema", "reason"),
    (
        (
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
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
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {},
                    }
                },
            },
            "unsupported schema keyword: $schema",
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
    client = StreamableHTTPMCPClientFactory(
        http_transport=mock_transport(identity)
    ).create(
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
    client = StreamableHTTPMCPClientFactory(
        http_transport=mock_transport(identity)
    ).create(
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
    client = StreamableHTTPMCPClientFactory(
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


async def test_failed_initialized_notification_resets_negotiated_client_state():
    identity = MCPFixtureIdentity(
        host="initialize-retry.fixture.test",
        server_name="initialize-retry-fixture",
        server_version="1",
        protocol_version="2025-11-25",
        tools=[],
        results={},
        initialized_notification_failures=1,
    )
    client = StreamableHTTPMCPClientFactory(
        http_transport=mock_transport(identity)
    ).create(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    try:
        with pytest.raises(MCPTransportError):
            await client.inspect(observed_at=NOW)
        inspection = await client.inspect(observed_at=NOW)
        assert inspection.server_name == identity.server_name
        assert identity.request_methods.count("initialize") == 2
        assert identity.request_methods.count("notifications/initialized") == 2
        assert identity.request_methods[-1] == "tools/list"
    finally:
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
    client = StreamableHTTPMCPClientFactory(
        http_transport=mock_transport(identity)
    ).create(
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


def test_missing_http_dependency_uses_pipx_repair_guidance(monkeypatch):
    identity, _beta = conformance_identities()
    client = StreamableHTTPMCPClient(
        endpoint=identity.endpoint,
        authentication=MCPAuthentication.no_auth(),
        secrets=EmptySecretProvider(),
    )
    original = builtins.__import__

    def missing_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("controlled missing dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_httpx)
    with pytest.raises(ImportError, match="pipx reinstall daita-agents"):
        client._client()
