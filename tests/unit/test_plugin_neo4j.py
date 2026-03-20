"""
Unit tests for Neo4jPlugin.

Tests LIMIT injection, introspection tools, read_only gating,
and error handling — without a real Neo4j connection.
"""

import pytest
from unittest.mock import AsyncMock
from daita.plugins.neo4j_graph import Neo4jPlugin
from daita.core.exceptions import PluginError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin(read_only=False):
    plugin = Neo4jPlugin(uri="bolt://localhost:7687", auth=("neo4j", "test"), read_only=read_only)
    # Inject fake driver so connect() is never needed
    plugin._driver = object()  # truthy sentinel
    return plugin


def _stub_query(plugin, records):
    """Replace plugin.query() with a coroutine that returns the given records."""
    captured = []

    async def fake_query(cypher, parameters=None):
        captured.append(cypher)
        return {"records": records, "count": len(records)}

    plugin.query = fake_query
    return captured


# ---------------------------------------------------------------------------
# Tool names
# ---------------------------------------------------------------------------


def test_tool_names_have_neo4j_prefix():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert "neo4j_query" in names
    assert "neo4j_find_nodes" in names
    assert "neo4j_find_path" in names
    assert "neo4j_get_neighbors" in names


def test_introspection_tools_present():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert "neo4j_list_labels" in names
    assert "neo4j_list_relationship_types" in names
    assert "neo4j_graph_stats" in names


def test_write_tools_absent_when_read_only():
    plugin = make_plugin(read_only=True)
    names = {t.name for t in plugin.get_tools()}
    assert "neo4j_create_node" not in names
    assert "neo4j_create_relationship" not in names
    assert "neo4j_delete_node" not in names


def test_write_tools_present_when_not_read_only():
    plugin = make_plugin(read_only=False)
    names = {t.name for t in plugin.get_tools()}
    assert "neo4j_create_node" in names
    assert "neo4j_create_relationship" in names
    assert "neo4j_delete_node" in names


# ---------------------------------------------------------------------------
# LIMIT injection in _tool_query
# ---------------------------------------------------------------------------


async def test_limit_injected_when_absent():
    plugin = make_plugin()
    captured = _stub_query(plugin, [])

    await plugin._tool_query({"cypher": "MATCH (n:Person) RETURN n"})

    assert "LIMIT" in captured[0].upper()
    assert "LIMIT 200" in captured[0]


async def test_limit_not_duplicated_when_present():
    plugin = make_plugin()
    captured = _stub_query(plugin, [])

    await plugin._tool_query({"cypher": "MATCH (n) RETURN n LIMIT 10"})

    assert captured[0].upper().count("LIMIT") == 1
    assert "LIMIT 10" in captured[0]


async def test_limit_injected_for_uppercase_query():
    plugin = make_plugin()
    captured = _stub_query(plugin, [])

    await plugin._tool_query({"cypher": "MATCH (p:Person)-[:KNOWS]->(q) RETURN p, q"})

    assert "LIMIT 200" in captured[0]


async def test_trailing_semicolon_handled():
    plugin = make_plugin()
    captured = _stub_query(plugin, [])

    await plugin._tool_query({"cypher": "MATCH (n) RETURN n;"})

    # Semicolon stripped before LIMIT appended
    assert captured[0].endswith("LIMIT 200")
    assert ";;" not in captured[0]


# ---------------------------------------------------------------------------
# Introspection tool handlers
# ---------------------------------------------------------------------------


async def test_list_labels_returns_labels():
    plugin = make_plugin()
    _stub_query(plugin, [{"label": "Person"}, {"label": "Company"}])

    result = await plugin._tool_list_labels({})

    assert result["labels"] == ["Person", "Company"]


async def test_list_relationship_types_returns_types():
    plugin = make_plugin()
    _stub_query(plugin, [{"relationshipType": "KNOWS"}, {"relationshipType": "WORKS_AT"}])

    result = await plugin._tool_list_relationship_types({})

    assert result["relationship_types"] == ["KNOWS", "WORKS_AT"]


async def test_graph_stats_returns_counts():
    plugin = make_plugin()
    call_count = 0

    async def fake_query(cypher, parameters=None):
        nonlocal call_count
        call_count += 1
        if "node_count" in cypher:
            return {"records": [{"node_count": 42}], "count": 1}
        return {"records": [{"relationship_count": 17}], "count": 1}

    plugin.query = fake_query

    result = await plugin._tool_graph_stats({})

    assert result["node_count"] == 42
    assert result["relationship_count"] == 17
    assert call_count == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_query_raises_plugin_error_on_failure():
    plugin = Neo4jPlugin(uri="bolt://localhost:7687", auth=("neo4j", "test"))
    plugin._driver = MagicMock = object()  # set to truthy but broken

    async def bad_query(cypher, parameters=None):
        raise PluginError("Query failed")

    plugin.query = bad_query

    with pytest.raises(PluginError, match="Query failed"):
        await plugin._tool_query({"cypher": "MATCH (n) RETURN n"})
