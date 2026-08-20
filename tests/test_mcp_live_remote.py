"""Explicitly opt-in interoperability smoke for one real remote MCP server."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pytest

from daita.adapters.mcp import (
    MCPAuthentication,
    StreamableHTTPMCPClientFactory,
)
from daita.security import (
    EmptySecretProvider,
    EnvironmentSecretProvider,
    SecretProvider,
    SecretReference,
)


@pytest.mark.integration
async def test_context7_remote_streamable_http_interoperability_smoke() -> None:
    """Exercise production inspect/call translation only when explicitly enabled."""

    if os.environ.get("DAITA_RUN_LIVE_MCP") != "1":
        pytest.skip("set DAITA_RUN_LIVE_MCP=1 to authorize live MCP network I/O")

    endpoint = os.environ.get(
        "DAITA_MCP_SMOKE_ENDPOINT",
        "https://mcp.context7.com/mcp",
    )
    remote_tool = os.environ.get(
        "DAITA_MCP_SMOKE_TOOL",
        "resolve-library-id",
    )
    arguments = json.loads(
        os.environ.get(
            "DAITA_MCP_SMOKE_ARGUMENTS",
            '{"libraryName":"pytest","query":"Find pytest documentation"}',
        )
    )
    if not isinstance(arguments, dict):
        raise ValueError("DAITA_MCP_SMOKE_ARGUMENTS must be a JSON object")

    if os.environ.get("DAITA_MCP_SMOKE_TOKEN"):
        authentication = MCPAuthentication.bearer(
            SecretReference.environment("DAITA_MCP_SMOKE_TOKEN")
        )
        secrets: SecretProvider = EnvironmentSecretProvider()
    else:
        authentication = MCPAuthentication.no_auth()
        secrets = EmptySecretProvider()

    client = StreamableHTTPMCPClientFactory().create(
        endpoint=endpoint,
        authentication=authentication,
        secrets=secrets,
    )
    try:
        inspection = await client.inspect(observed_at=datetime.now(UTC))
        selected = next(
            (tool for tool in inspection.tools if tool.remote_name == remote_tool),
            None,
        )
        assert selected is not None
        assert selected.supported, selected.unsupported_reason
        result = await client.call_tool(remote_tool, arguments)
        assert not result.is_error
        assert result.text or result.structured is not None
    finally:
        await client.close()
