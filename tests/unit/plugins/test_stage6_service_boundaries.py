"""Regression tests for Stage 6 service-plugin input and lifecycle boundaries."""

import pytest

from daita.plugins.bigquery import BigQueryPlugin
from daita.plugins.exa import ExaSearchPlugin
from daita.plugins.lineage import LineagePlugin
from daita.plugins.mcp import MCPServer
from daita.plugins.rest import RESTPlugin
from daita.plugins.websearch import WebSearchPlugin


async def test_rest_endpoint_is_required() -> None:
    plugin = RESTPlugin(base_url="https://example.test")
    with pytest.raises(ValueError, match="endpoint"):
        await plugin._tool_get({})


async def test_websearch_query_is_required() -> None:
    plugin = WebSearchPlugin(api_key="test-key")
    with pytest.raises(ValueError, match="query"):
        await plugin._tool_search_web({})


async def test_exa_similar_url_is_required() -> None:
    plugin = ExaSearchPlugin(api_key="test-key")
    with pytest.raises(ValueError, match="url"):
        await plugin._tool_find_similar({})


async def test_lineage_flow_source_is_required() -> None:
    plugin = LineagePlugin()
    with pytest.raises(ValueError, match="source_id"):
        await plugin._tool_register_flow({})


async def test_lineage_export_entity_is_required() -> None:
    plugin = LineagePlugin()
    with pytest.raises(ValueError, match="entity_id"):
        await plugin._tool_export_lineage({})


def test_mcp_session_access_before_connect_is_rejected() -> None:
    server = MCPServer(command="test-server")
    with pytest.raises(RuntimeError, match="not connected"):
        _ = server.session


def test_bigquery_client_access_before_connect_is_rejected() -> None:
    plugin = BigQueryPlugin(project="test-project")
    with pytest.raises(RuntimeError, match="not connected"):
        _ = plugin.client
