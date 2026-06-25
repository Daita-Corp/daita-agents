"""Unit tests for WebSearchPlugin extension declarations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daita.plugins.manifest import PluginKind
from daita.plugins.registry import ExtensionRegistry
from daita.plugins.websearch import WebSearchPlugin
from daita.runtime import Operation, Task
from tests.unit.plugins.projection_helpers import projected_tools


def make_plugin(**overrides):
    kwargs = {"api_key": "tvly-test-key"}
    kwargs.update(overrides)
    plugin = WebSearchPlugin(**kwargs)
    plugin._client = MagicMock()
    plugin._session = MagicMock()
    return plugin


def test_websearch_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "websearch"
    assert plugin.manifest.kind is PluginKind.CONNECTOR
    assert registry.plugin_ids == ("websearch",)
    assert {capability.id for capability in registry.capabilities} == {
        "websearch.web.search",
        "websearch.news.search",
        "websearch.page.fetch",
    }
    assert {view.name for view in registry.tool_views} == {
        "search_web",
        "search_news",
        "fetch_page",
    }
    assert {schema.kind for schema in registry.evidence_schemas} == {
        "web.search.results",
        "web.page.content",
    }


def test_websearch_projected_tools_carry_declared_capability_metadata():
    plugin = make_plugin()

    by_name = projected_tools(plugin)

    assert by_name["search_web"].capability_ids == ("websearch.web.search",)
    assert by_name["search_web"].side_effecting is False
    assert by_name["search_web"].idempotent is True
    assert by_name["search_news"].capability_ids == ("websearch.news.search",)
    assert by_name["fetch_page"].capability_ids == ("websearch.page.fetch",)


async def test_websearch_executor_returns_typed_search_evidence():
    plugin = make_plugin()
    plugin.search = AsyncMock(
        return_value={
            "success": True,
            "query": "daita agents",
            "answer": "A data-focused agent framework.",
            "results": [],
            "count": 0,
        }
    )
    registry = ExtensionRegistry()
    registry.register(plugin)

    executor = registry.get_executor("websearch.operations")
    operation = Operation(id="op-1", operation_type="web.search")
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="websearch.web.search",
        executor_id="websearch.operations",
        input={"query": "daita agents"},
        required_evidence=frozenset({"web.search.results"}),
    )

    evidence = await executor.execute(
        task,
        operation,
        {"tool_view": {"name": "search_web"}},
    )

    assert len(evidence) == 1
    assert evidence[0].kind == "web.search.results"
    assert evidence[0].owner == "websearch"
    assert evidence[0].payload["operation"] == "search_web"
    assert evidence[0].payload["request"] == {"query": "daita agents"}
    assert evidence[0].payload["response"]["success"] is True
    assert evidence[0].metadata["capability_id"] == "websearch.web.search"
    assert evidence[0].metadata["tool_view"] == "search_web"


async def test_websearch_executor_returns_typed_page_evidence():
    plugin = make_plugin()
    plugin.fetch_page = AsyncMock(
        return_value={
            "success": True,
            "url": "https://example.com",
            "content": "Example",
            "length": 7,
            "truncated": False,
        }
    )
    registry = ExtensionRegistry()
    registry.register(plugin)

    executor = registry.get_executor("websearch.operations")
    operation = Operation(id="op-1", operation_type="web.page.fetch")
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="websearch.page.fetch",
        executor_id="websearch.operations",
        input={"url": "https://example.com"},
        required_evidence=frozenset({"web.page.content"}),
    )

    evidence = await executor.execute(task, operation, {})

    assert len(evidence) == 1
    assert evidence[0].kind == "web.page.content"
    assert evidence[0].payload["operation"] == "fetch_page"
    assert evidence[0].payload["response"]["content"] == "Example"


def test_missing_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Tavily API key is required"):
            WebSearchPlugin(api_key=None)


def test_invalid_search_depth_raises():
    with pytest.raises(ValueError, match="search_depth must be 'basic' or 'advanced'"):
        WebSearchPlugin(api_key="k", search_depth="deep")


def test_picks_up_api_key_from_env():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "env-key"}, clear=False):
        plugin = WebSearchPlugin()
        assert plugin._api_key == "env-key"
