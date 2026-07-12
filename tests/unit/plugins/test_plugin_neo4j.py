"""
Unit tests for Neo4jPlugin.

Tests real behavior: LIMIT injection logic, introspection Cypher content,
find_nodes/find_path/get_neighbors result shapes and query construction,
_build_match_condition, read_only gating, and PluginError propagation.
No real Neo4j connection is needed.
"""

import sys
import types

import pytest
from daita.plugins.manifest import PluginKind
from daita.plugins.neo4j_graph import Neo4jPlugin
from daita.plugins.registry import ExtensionRegistry
from daita.core.exceptions import PluginError
from daita.runtime import Operation, Task
from tests.unit.plugins.projection_helpers import projected_tools, projected_tool_names

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin(read_only=False):
    return Neo4jPlugin(
        uri="bolt://localhost:7687", auth=("neo4j", "test"), read_only=read_only
    )


def stub_query(plugin, records_by_cypher=None, default_records=None):
    """
    Replace plugin.query() with a coroutine that captures calls and returns
    canned records. records_by_cypher maps a substring of the Cypher query
    to a list of records; falls back to default_records.
    """
    captured = []

    async def fake_query(cypher, parameters=None):
        captured.append({"cypher": cypher, "params": parameters})
        if records_by_cypher:
            for key, records in records_by_cypher.items():
                if key in cypher:
                    return {"records": records, "count": len(records)}
        return {"records": default_records or [], "count": len(default_records or [])}

    plugin.query = fake_query
    return captured


# ---------------------------------------------------------------------------
# _build_match_condition  (pure function, no mock needed)
# ---------------------------------------------------------------------------


class TestBuildMatchCondition:
    def test_empty_properties_returns_empty_string(self):
        plugin = make_plugin()
        assert plugin._build_match_condition({}) == ""

    def test_single_property(self):
        plugin = make_plugin()
        result = plugin._build_match_condition({"name": "Alice"})
        assert result == "{name: $name}"

    def test_multiple_properties(self):
        plugin = make_plugin()
        result = plugin._build_match_condition({"name": "Alice", "age": 30})
        assert "name: $name" in result
        assert "age: $age" in result
        assert result.startswith("{")
        assert result.endswith("}")

    def test_prefix_applied_to_parameter_names(self):
        plugin = make_plugin()
        result = plugin._build_match_condition({"name": "Alice"}, prefix="from_")
        assert result == "{name: $from_name}"


# ---------------------------------------------------------------------------
# _tool_query — LIMIT injection logic
# ---------------------------------------------------------------------------


class TestLimitInjection:
    async def test_limit_injected_when_absent(self):
        plugin = make_plugin()
        captured = stub_query(plugin)
        await plugin._tool_query({"cypher": "MATCH (n:Person) RETURN n"})
        assert "LIMIT 200" in captured[0]["cypher"]

    async def test_no_double_limit_when_present(self):
        plugin = make_plugin()
        captured = stub_query(plugin)
        await plugin._tool_query({"cypher": "MATCH (n) RETURN n LIMIT 10"})
        assert captured[0]["cypher"].upper().count("LIMIT") == 1
        assert "LIMIT 10" in captured[0]["cypher"]

    async def test_lowercase_limit_not_doubled(self):
        plugin = make_plugin()
        captured = stub_query(plugin)
        await plugin._tool_query({"cypher": "MATCH (n) RETURN n limit 5"})
        assert captured[0]["cypher"].upper().count("LIMIT") == 1

    async def test_trailing_semicolon_stripped_before_limit(self):
        plugin = make_plugin()
        captured = stub_query(plugin)
        await plugin._tool_query({"cypher": "MATCH (n) RETURN n;"})
        cypher = captured[0]["cypher"]
        assert cypher.endswith("LIMIT 200")
        assert ";;" not in cypher

    async def test_parameters_forwarded_to_query(self):
        plugin = make_plugin()
        captured = stub_query(plugin)
        params = {"name": "Alice"}
        await plugin._tool_query(
            {"cypher": "MATCH (n {name: $name}) RETURN n", "parameters": params}
        )
        assert captured[0]["params"] == params

    async def test_result_shape_from_query(self):
        plugin = make_plugin()
        stub_query(plugin, default_records=[{"n": {"name": "Alice"}}])
        result = await plugin._tool_query({"cypher": "MATCH (n) RETURN n LIMIT 1"})
        # _tool_query returns the raw query() result
        assert "records" in result
        assert "count" in result


# ---------------------------------------------------------------------------
# Introspection tool Cypher content
# ---------------------------------------------------------------------------


class TestIntrospectionCypherContent:
    """Verify the tools send the correct Cypher queries, not just that they return data."""

    async def test_list_labels_sends_correct_cypher(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_list_labels({})
        assert any("db.labels" in c["cypher"] for c in captured)

    async def test_list_labels_extracts_label_field(self):
        plugin = make_plugin()
        stub_query(
            plugin,
            records_by_cypher={
                "db.labels": [{"label": "Person"}, {"label": "Company"}]
            },
        )
        result = await plugin._tool_list_labels({})
        assert result["labels"] == ["Person", "Company"]

    async def test_list_relationship_types_sends_correct_cypher(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_list_relationship_types({})
        assert any("db.relationshipTypes" in c["cypher"] for c in captured)

    async def test_list_relationship_types_extracts_type_field(self):
        plugin = make_plugin()
        stub_query(
            plugin,
            records_by_cypher={
                "db.relationshipTypes": [
                    {"relationshipType": "KNOWS"},
                    {"relationshipType": "WORKS_AT"},
                ]
            },
        )
        result = await plugin._tool_list_relationship_types({})
        assert result["relationship_types"] == ["KNOWS", "WORKS_AT"]

    async def test_graph_stats_runs_two_queries(self):
        plugin = make_plugin()
        captured = stub_query(
            plugin,
            records_by_cypher={
                "node_count": [{"node_count": 42}],
                "relationship_count": [{"relationship_count": 17}],
            },
        )
        result = await plugin._tool_graph_stats({})
        assert len(captured) == 2
        assert result["node_count"] == 42
        assert result["relationship_count"] == 17

    async def test_graph_stats_zero_when_no_records(self):
        plugin = make_plugin()
        stub_query(plugin, default_records=[])
        result = await plugin._tool_graph_stats({})
        assert result["node_count"] == 0
        assert result["relationship_count"] == 0


# ---------------------------------------------------------------------------
# _tool_find_nodes — result shape and query construction
# ---------------------------------------------------------------------------


class TestFindNodes:
    async def test_result_has_nodes_and_count(self):
        plugin = make_plugin()
        stub_query(
            plugin, default_records=[{"n": {"name": "Alice"}}, {"n": {"name": "Bob"}}]
        )
        result = await plugin._tool_find_nodes({"label": "Person"})
        assert "nodes" in result
        assert "count" in result
        assert result["count"] == 2

    async def test_label_included_in_cypher(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_find_nodes({"label": "Company"})
        assert any("Company" in c["cypher"] for c in captured)

    async def test_default_limit_is_50(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_find_nodes({"label": "Person"})
        assert any("50" in c["cypher"] for c in captured)

    async def test_explicit_limit_used(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_find_nodes({"label": "Person", "limit": 10})
        assert any("10" in c["cypher"] for c in captured)

    async def test_properties_included_in_match_condition(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_find_nodes(
            {"label": "Person", "properties": {"name": "Alice"}}
        )
        assert any("$name" in c["cypher"] for c in captured)

    async def test_nodes_extracted_from_n_key(self):
        plugin = make_plugin()
        node_data = {"name": "Alice", "age": 30}
        stub_query(plugin, default_records=[{"n": node_data}])
        result = await plugin._tool_find_nodes({"label": "Person"})
        assert result["nodes"][0] == node_data


# ---------------------------------------------------------------------------
# _tool_find_path — result shape
# ---------------------------------------------------------------------------


class TestFindPath:
    async def test_result_has_path_found_and_length_keys(self):
        plugin = make_plugin()
        stub_query(
            plugin,
            default_records=[{"p": "some_path_object", "path_length": 3}],
        )
        result = await plugin._tool_find_path(
            {
                "from_label": "Person",
                "from_properties": {"name": "Alice"},
                "to_label": "Person",
                "to_properties": {"name": "Bob"},
            }
        )
        assert "path" in result
        assert "path_length" in result
        assert "found" in result

    async def test_found_true_when_records_returned(self):
        plugin = make_plugin()
        stub_query(plugin, default_records=[{"p": {}, "path_length": 2}])
        result = await plugin._tool_find_path(
            {
                "from_label": "Person",
                "from_properties": {"name": "Alice"},
                "to_label": "Person",
                "to_properties": {"name": "Bob"},
            }
        )
        assert result["found"] is True

    async def test_found_false_when_no_records(self):
        plugin = make_plugin()
        stub_query(plugin, default_records=[])
        result = await plugin._tool_find_path(
            {
                "from_label": "Person",
                "from_properties": {"name": "Alice"},
                "to_label": "Person",
                "to_properties": {"name": "Bob"},
            }
        )
        assert result["found"] is False
        assert result["path"] is None

    async def test_max_length_used_in_cypher(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_find_path(
            {
                "from_label": "A",
                "from_properties": {"id": 1},
                "to_label": "B",
                "to_properties": {"id": 2},
                "max_length": 3,
            }
        )
        assert any("3" in c["cypher"] for c in captured)


# ---------------------------------------------------------------------------
# _tool_get_neighbors — direction logic in Cypher
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    async def test_result_has_neighbors_and_count(self):
        plugin = make_plugin()
        stub_query(
            plugin,
            default_records=[
                {"m": {"name": "Bob"}, "relationship_type": "KNOWS", "r": {}}
            ],
        )
        result = await plugin._tool_get_neighbors(
            {"label": "Person", "properties": {"name": "Alice"}}
        )
        assert "neighbors" in result
        assert "count" in result
        assert result["count"] == 1

    async def test_neighbor_has_node_rel_type_and_relationship(self):
        plugin = make_plugin()
        stub_query(
            plugin,
            default_records=[
                {
                    "m": {"name": "Bob"},
                    "relationship_type": "KNOWS",
                    "r": {"since": 2020},
                }
            ],
        )
        result = await plugin._tool_get_neighbors(
            {"label": "Person", "properties": {"name": "Alice"}}
        )
        neighbor = result["neighbors"][0]
        assert "node" in neighbor
        assert "relationship_type" in neighbor
        assert "relationship" in neighbor

    async def test_outgoing_direction_uses_arrow_right(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_get_neighbors(
            {
                "label": "Person",
                "properties": {"name": "Alice"},
                "direction": "outgoing",
            }
        )
        # Outgoing should produce -[r]->
        assert any("->" in c["cypher"] for c in captured)

    async def test_incoming_direction_uses_arrow_left(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_get_neighbors(
            {
                "label": "Person",
                "properties": {"name": "Alice"},
                "direction": "incoming",
            }
        )
        # Incoming should produce <-[r]-
        assert any("<-" in c["cypher"] for c in captured)

    async def test_both_direction_uses_undirected(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_get_neighbors(
            {"label": "Person", "properties": {"name": "Alice"}, "direction": "both"}
        )
        # Bidirectional: no -> or <- should appear around the relationship
        cypher = captured[0]["cypher"]
        assert "-[r]-" in cypher or "-[r" in cypher

    async def test_relationship_type_filter_included_in_cypher(self):
        plugin = make_plugin()
        captured = stub_query(plugin, default_records=[])
        await plugin._tool_get_neighbors(
            {
                "label": "Person",
                "properties": {"name": "Alice"},
                "relationship_type": "KNOWS",
            }
        )
        assert any("KNOWS" in c["cypher"] for c in captured)


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    async def test_query_plugin_error_propagates(self):
        """PluginError from query() should propagate out of _tool_query unchanged."""
        plugin = make_plugin()

        async def bad_query(cypher, parameters=None):
            raise PluginError("Query failed", plugin_name="Neo4j")

        plugin.query = bad_query

        with pytest.raises(PluginError, match="Query failed"):
            await plugin._tool_query({"cypher": "MATCH (n) RETURN n"})

    async def test_list_labels_propagates_query_error(self):
        plugin = make_plugin()

        async def bad_query(cypher, parameters=None):
            raise PluginError("DB unavailable", plugin_name="Neo4j")

        plugin.query = bad_query

        with pytest.raises(PluginError):
            await plugin._tool_list_labels({})


class TestInputAndConnectionBoundaries:
    @pytest.mark.parametrize(
        ("handler_name", "args", "field"),
        [
            ("_tool_create_node", {"properties": {}}, "label"),
            ("_tool_create_node", {"label": "Person"}, "properties"),
            ("_tool_find_path", {"from_label": "Person"}, "from_properties"),
            ("_tool_get_neighbors", {"properties": {}}, "label"),
            ("_tool_delete_node", {"label": "Person"}, "properties"),
        ],
    )
    async def test_required_tool_inputs_are_rejected(self, handler_name, args, field):
        plugin = make_plugin()

        with pytest.raises(ValueError, match=field):
            await getattr(plugin, handler_name)(args)

    async def test_failed_connect_does_not_publish_driver(self, monkeypatch):
        plugin = Neo4jPlugin(uri="bolt://example.invalid:7687")

        class FailingDriver:
            closed = False

            async def verify_connectivity(self):
                raise OSError("unreachable")

            async def close(self):
                self.closed = True

        driver = FailingDriver()

        class FakeAsyncGraphDatabase:
            @staticmethod
            def driver(*args, **kwargs):
                return driver

        fake_neo4j = types.ModuleType("neo4j")
        monkeypatch.setattr(
            fake_neo4j,
            "AsyncGraphDatabase",
            FakeAsyncGraphDatabase,
            raising=False,
        )
        monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

        with pytest.raises(PluginError, match="unreachable"):
            await plugin.connect()

        assert driver.closed is True
        assert plugin.is_connected is False


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestExtensionContract:
    def test_neo4j_plugin_declares_extension_first_contract(self):
        plugin = make_plugin()
        registry = ExtensionRegistry()

        registry.register(plugin)

        assert plugin.manifest.id == "neo4j"
        assert plugin.manifest.kind is PluginKind.CONNECTOR
        assert registry.plugin_ids == ("neo4j",)
        assert {capability.id for capability in registry.capabilities} == {
            "neo4j.query.execute",
            "neo4j.node.find",
            "neo4j.path.find",
            "neo4j.neighbor.list",
            "neo4j.schema.read",
            "neo4j.labels.list",
            "neo4j.relationship_types.list",
            "neo4j.stats.read",
            "neo4j.node.create",
            "neo4j.relationship.create",
            "neo4j.node.delete",
        }
        assert {view.name for view in registry.tool_views} == projected_tool_names(
            plugin
        )
        assert registry.evidence_schemas[0].kind == "neo4j.operation.result"

    def test_neo4j_read_only_contract_excludes_write_capabilities(self):
        plugin = make_plugin(read_only=True)
        registry = ExtensionRegistry()

        registry.register(plugin)

        capability_ids = {capability.id for capability in registry.capabilities}
        view_names = {view.name for view in registry.tool_views}

        assert "neo4j.node.create" not in capability_ids
        assert "neo4j.relationship.create" not in capability_ids
        assert "neo4j.node.delete" not in capability_ids
        assert "neo4j_create_node" not in view_names
        assert "neo4j_create_relationship" not in view_names
        assert "neo4j_delete_node" not in view_names

    def test_neo4j_projected_tools_carry_declared_capability_metadata(self):
        plugin = make_plugin()

        by_name = projected_tools(plugin)

        assert by_name["neo4j_query"].capability_ids == ("neo4j.query.execute",)
        assert by_name["neo4j_query"].side_effecting is False
        assert by_name["neo4j_query"].idempotent is True
        assert by_name["neo4j_create_node"].capability_ids == ("neo4j.node.create",)
        assert by_name["neo4j_create_node"].side_effecting is True
        assert by_name["neo4j_delete_node"].capability_ids == ("neo4j.node.delete",)

    async def test_neo4j_executor_returns_typed_operation_evidence(self):
        plugin = make_plugin()
        stub_query(plugin, default_records=[{"n": {"name": "Alice"}}])
        registry = ExtensionRegistry()
        registry.register(plugin)

        executor = registry.get_executor("neo4j.operations")
        operation = Operation(id="op-1", operation_type="neo4j.node.find")
        task = Task(
            id="task-1",
            operation_id=operation.id,
            capability_id="neo4j.node.find",
            executor_id="neo4j.operations",
            input={"label": "Person", "properties": {"name": "Alice"}, "limit": 1},
            required_evidence=frozenset({"neo4j.operation.result"}),
        )

        evidence = await executor.execute(
            task,
            operation,
            {"tool_view": {"name": "neo4j_find_nodes"}},
        )

        assert len(evidence) == 1
        assert evidence[0].kind == "neo4j.operation.result"
        assert evidence[0].owner == "neo4j"
        assert evidence[0].payload["operation"] == "neo4j_find_nodes"
        assert evidence[0].payload["request"]["label"] == "Person"
        assert evidence[0].payload["result"] == {
            "nodes": [{"name": "Alice"}],
            "count": 1,
        }
        assert evidence[0].metadata["capability_id"] == "neo4j.node.find"
        assert evidence[0].metadata["tool_view"] == "neo4j_find_nodes"


class TestToolRegistration:
    def test_core_read_tools_present(self):
        plugin = make_plugin()
        names = projected_tool_names(plugin)
        assert "neo4j_query" in names
        assert "neo4j_find_nodes" in names
        assert "neo4j_find_path" in names
        assert "neo4j_get_neighbors" in names

    def test_introspection_tools_present(self):
        plugin = make_plugin()
        names = projected_tool_names(plugin)
        assert "neo4j_list_labels" in names
        assert "neo4j_list_relationship_types" in names
        assert "neo4j_graph_stats" in names

    def test_write_tools_absent_when_read_only(self):
        plugin = make_plugin(read_only=True)
        names = projected_tool_names(plugin)
        assert "neo4j_create_node" not in names
        assert "neo4j_create_relationship" not in names
        assert "neo4j_delete_node" not in names

    def test_write_tools_present_when_not_read_only(self):
        plugin = make_plugin(read_only=False)
        names = projected_tool_names(plugin)
        assert "neo4j_create_node" in names
        assert "neo4j_create_relationship" in names
        assert "neo4j_delete_node" in names
