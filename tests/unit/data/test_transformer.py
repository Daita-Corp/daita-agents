"""
Unit tests for daita/plugins/transformer.py.

All tests use mocked database and graph backends — no real database required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from daita.plugins.transformer import TransformerPlugin, transformer, _local_parse_sql

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_backend(existing_node=None):
    """Create a mock graph backend."""
    backend = MagicMock()
    backend.add_node = AsyncMock()
    backend.add_edge = AsyncMock()
    backend.flush = AsyncMock()
    backend.get_node = AsyncMock(return_value=existing_node)
    # resolve_or_placeholder (used by transform_create for source / target
    # table refs) looks up candidates via find_nodes. Default: no matches,
    # so bare names fall through to the __unresolved__ sentinel path.
    backend.find_nodes = AsyncMock(return_value=[])

    mock_graph = MagicMock()
    mock_graph.nodes.return_value = []
    backend.load_graph = AsyncMock(return_value=mock_graph)
    return backend


def make_db(has_execute=True):
    db = MagicMock()
    db.query = AsyncMock(return_value=[])
    if has_execute:
        db.execute = AsyncMock(return_value=5)
    elif hasattr(db, "execute"):
        # Remove execute so hasattr(db, "execute") is False
        del db.execute
    return db


def make_db_no_execute():
    """DB plugin that only has query(), not execute()."""
    db = MagicMock(spec=["query"])
    db.query = AsyncMock(return_value=[])
    return db


def make_tx(db=None, lineage=None, backend=None):
    plugin = TransformerPlugin(db=db, lineage=lineage, backend=backend)
    plugin._agent_id = "test-agent"
    return plugin


# ---------------------------------------------------------------------------
# _local_parse_sql
# ---------------------------------------------------------------------------


def test_local_parse_sql_insert():
    result = _local_parse_sql(
        "INSERT INTO orders_summary SELECT customer_id FROM orders"
    )
    assert "orders_summary" in result["target_tables"]
    assert "orders" in result["source_tables"]


def test_local_parse_sql_create_table_as():
    result = _local_parse_sql("CREATE TABLE summary AS SELECT * FROM raw_data")
    assert "summary" in result["target_tables"]
    assert "raw_data" in result["source_tables"]


def test_local_parse_sql_select_only():
    result = _local_parse_sql("SELECT * FROM customers WHERE id = 1")
    assert result["target_tables"] == []
    assert "customers" in result["source_tables"]


def test_local_parse_sql_update():
    result = _local_parse_sql("UPDATE orders SET status = 'done'")
    assert "orders" in result["target_tables"]


def test_local_parse_sql_delete():
    """TX-05: DELETE FROM produces a target (writes) edge, not a source (reads) edge."""
    result = _local_parse_sql(
        "DELETE FROM stale_records WHERE created_at < '2020-01-01'"
    )
    assert "stale_records" in result["target_tables"]
    assert "stale_records" not in result["source_tables"]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def test_factory_function():
    plugin = transformer()
    assert isinstance(plugin, TransformerPlugin)


def test_factory_passes_db():
    db = make_db()
    plugin = transformer(db=db)
    assert plugin._db is db


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


def test_initialize_sets_agent_id():
    plugin = TransformerPlugin()
    with patch(
        "daita.core.graph.backend.auto_select_backend", return_value=MagicMock()
    ):
        plugin.initialize("agent-xyz")
    assert plugin._agent_id == "agent-xyz"


def test_initialize_skips_backend_if_provided():
    custom_be = MagicMock()
    plugin = TransformerPlugin(backend=custom_be)
    with patch("daita.core.graph.backend.auto_select_backend") as mock_sel:
        plugin.initialize("agent-xyz")
    mock_sel.assert_not_called()


# ---------------------------------------------------------------------------
# _validate_db
# ---------------------------------------------------------------------------


def test_validate_db_raises_without_db():
    plugin = make_tx()
    with pytest.raises(ValueError, match="No database plugin configured"):
        plugin._validate_db()


def test_validate_db_returns_db():
    db = make_db()
    plugin = make_tx(db=db)
    assert plugin._validate_db() is db


# ---------------------------------------------------------------------------
# get_tools
# ---------------------------------------------------------------------------


def test_get_tools_returns_six_tools():
    plugin = make_tx()
    tools = plugin.get_tools()
    assert len(tools) == 6


def test_get_tools_names():
    plugin = make_tx()
    names = {t.name for t in plugin.get_tools()}
    assert names == {
        "transform_create",
        "transform_run",
        "transform_test",
        "transform_version",
        "transform_diff",
        "transform_list",
    }


# ---------------------------------------------------------------------------
# transform_create
# ---------------------------------------------------------------------------


async def test_transform_create_stores_node():
    backend = make_backend()
    plugin = make_tx(backend=backend)

    result = await plugin.transform_create(
        "orders_summary",
        sql="INSERT INTO orders_summary SELECT customer_id FROM orders",
        description="Test",
    )
    assert result["success"] is True
    assert result["name"] == "orders_summary"
    backend.add_node.assert_called()
    backend.add_edge.assert_called()


async def test_transform_create_without_backend():
    """TX-03: create succeeds via in-memory fallback when no graph backend is available."""
    plugin = make_tx()  # no backend
    result = await plugin.transform_create("x", sql="SELECT 1")
    assert result["success"] is True
    assert result["name"] == "x"
    # Confirm in-memory store was populated
    assert "x" in plugin._definitions
    assert plugin._definitions["x"]["sql"] == "SELECT 1"


async def test_transform_create_invalid_name_raises():
    """TX-07: names with special characters are rejected before reaching graph storage."""
    plugin = make_tx()
    with pytest.raises(ValueError, match="Invalid identifier"):
        await plugin.transform_create("bad name!", sql="SELECT 1")


async def test_transform_create_detects_tables():
    backend = make_backend()
    plugin = make_tx(backend=backend)
    result = await plugin.transform_create(
        "test_tx",
        sql="INSERT INTO target_table SELECT * FROM source_table",
    )
    assert "source_table" in result["source_tables"]
    assert "target_table" in result["target_tables"]


async def test_transform_create_uses_lineage_parser():
    backend = make_backend()
    lineage_plugin = MagicMock()
    lineage_plugin.parse_sql_lineage = MagicMock(
        return_value={"source_tables": ["src"], "target_tables": ["tgt"]}
    )
    plugin = make_tx(backend=backend, lineage=lineage_plugin)
    result = await plugin.transform_create("tx", sql="SELECT 1")
    lineage_plugin.parse_sql_lineage.assert_called_once()
    assert result["source_tables"] == ["src"]
    assert result["target_tables"] == ["tgt"]


async def test_transform_create_update_preserves_versions():
    """Updating an existing transformation preserves its version history."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    existing_node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "SELECT 1",
            "versions": [{"sql": "SELECT 0", "created_at": "2024-01-01T00:00:00"}],
            "created_at": "2024-01-01T00:00:00",
        },
    )
    backend = make_backend(existing_node=existing_node)
    plugin = make_tx(backend=backend)

    result = await plugin.transform_create("tx", sql="SELECT 2")
    assert result["is_update"] is True
    # Verify the node was saved with the old versions preserved
    saved_node = backend.add_node.call_args[0][0]
    assert len(saved_node.properties["versions"]) == 1


async def test_transform_create_populates_in_memory_even_with_backend():
    """TX-11: _definitions is always updated so transform_list never needs load_graph."""
    backend = make_backend()
    plugin = make_tx(backend=backend)
    await plugin.transform_create("my_tx", sql="SELECT 42", description="test")
    assert "my_tx" in plugin._definitions
    assert plugin._definitions["my_tx"]["sql"] == "SELECT 42"


# ---------------------------------------------------------------------------
# transform_run
# ---------------------------------------------------------------------------


async def test_transform_run_executes_sql():
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:orders_summary",
        node_type=NodeType.TRANSFORMATION,
        name="orders_summary",
        properties={"sql": "INSERT INTO s SELECT 1", "target_table": "s"},
    )
    backend = make_backend(existing_node=node)
    db = make_db()
    plugin = make_tx(db=db, backend=backend)
    result = await plugin.transform_run(db, "orders_summary")
    assert result["success"] is True


async def test_transform_run_uses_execute_when_available():
    """TX-01: prefers db.execute() over db.query() when it exists."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={"sql": "INSERT INTO s SELECT 1", "target_table": "s"},
    )
    backend = make_backend(existing_node=node)
    db = make_db(has_execute=True)
    plugin = make_tx(db=db, backend=backend)
    await plugin.transform_run(db, "tx")
    db.execute.assert_called_once()
    db.query.assert_not_called()


async def test_transform_run_falls_back_to_query_when_no_execute():
    """TX-01: falls back to db.query() when db lacks execute()."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={"sql": "SELECT 1", "target_table": None},
    )
    backend = make_backend(existing_node=node)
    db = make_db_no_execute()
    plugin = make_tx(db=db, backend=backend)
    result = await plugin.transform_run(db, "tx")
    assert result["success"] is True
    db.query.assert_called_once()


async def test_transform_run_not_found():
    backend = make_backend(existing_node=None)
    db = make_db()
    plugin = make_tx(db=db, backend=backend)
    result = await plugin.transform_run(db, "nonexistent")
    assert result["success"] is False
    assert "not found" in result["error"]


async def test_transform_run_captures_lineage():
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={"sql": "INSERT INTO tgt SELECT * FROM src", "target_table": "tgt"},
    )
    backend = make_backend(existing_node=node)
    db = make_db()
    lineage_plugin = MagicMock()
    lineage_plugin.capture_sql_lineage = AsyncMock(return_value={"success": True})
    plugin = make_tx(db=db, lineage=lineage_plugin, backend=backend)
    await plugin.transform_run(db, "tx")
    lineage_plugin.capture_sql_lineage.assert_called_once()


async def test_transform_run_parameter_substitution():
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "SELECT * FROM orders WHERE status = :status",
            "target_table": None,
        },
    )
    backend = make_backend(existing_node=node)
    db = make_db()
    plugin = make_tx(db=db, backend=backend)
    result = await plugin.transform_run(db, "tx", parameters={"status": "active"})
    assert result["success"] is True
    call_sql = db.execute.call_args[0][0]
    assert ":status" not in call_sql
    assert "active" in call_sql


async def test_transform_run_param_substitution_none_and_bool():
    """None → NULL, True → 1, False → 0 in SQL substitution."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "SELECT * FROM t WHERE a = :a AND b = :b AND c = :c",
            "target_table": None,
        },
    )
    backend = make_backend(existing_node=node)
    db = make_db()
    plugin = make_tx(db=db, backend=backend)
    await plugin.transform_run(db, "tx", parameters={"a": None, "b": True, "c": False})
    call_sql = db.execute.call_args[0][0]
    assert "NULL" in call_sql
    assert " 1" in call_sql
    assert " 0" in call_sql


async def test_transform_run_word_boundary_param_substitution():
    """TX-04: :status should not replace the :status portion of :status_code."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "SELECT * FROM orders WHERE status = :status AND code = :status_code",
            "target_table": None,
        },
    )
    backend = make_backend(existing_node=node)
    db = make_db()
    plugin = make_tx(db=db, backend=backend)
    await plugin.transform_run(
        db, "tx", parameters={"status": "active", "status_code": "X1"}
    )
    call_sql = db.execute.call_args[0][0]
    # :status_code should be replaced with X1, not activeX1 (which would happen with a non-boundary regex)
    assert "X1" in call_sql
    assert ":status" not in call_sql
    assert ":status_code" not in call_sql


async def test_transform_run_updates_run_history():
    """TX-09: successful run updates last_run and run_count in _definitions."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "INSERT INTO s SELECT 1",
            "target_table": "s",
            "run_count": 2,
        },
    )
    backend = make_backend(existing_node=node)
    db = make_db()
    plugin = make_tx(db=db, backend=backend)

    result = await plugin.transform_run(db, "tx")
    assert result["success"] is True

    # In-memory index should reflect new run_count and last_run
    updated = plugin._definitions.get("tx", {})
    assert updated.get("run_count") == 3
    assert updated.get("last_run") is not None
    assert updated["last_run"]["success"] is True


async def test_transform_run_without_backend_in_memory():
    """TX-03: run works with in-memory fallback."""
    plugin = make_tx()
    await plugin.transform_create("tx_mem", sql="SELECT 99")

    db = make_db()
    result = await plugin.transform_run(db, "tx_mem")
    assert result["success"] is True
    db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# transform_test
# ---------------------------------------------------------------------------


async def test_transform_test_valid():
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={"sql": "SELECT 1", "target_table": None},
    )
    backend = make_backend(existing_node=node)
    db = MagicMock()
    db.query = AsyncMock(return_value=[{"QUERY PLAN": "Seq Scan on foo"}])
    plugin = make_tx(db=db, backend=backend)
    result = await plugin.transform_test(db, "tx")
    assert result["success"] is True
    assert result["valid"] is True


async def test_transform_test_invalid_sql():
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={"sql": "INVALID SQL !!!!", "target_table": None},
    )
    backend = make_backend(existing_node=node)
    db = MagicMock()
    db.query = AsyncMock(side_effect=Exception("syntax error"))
    plugin = make_tx(db=db, backend=backend)
    result = await plugin.transform_test(db, "tx")
    assert result["success"] is True
    assert result["valid"] is False
    assert "syntax error" in result["error"]


async def test_transform_test_not_found():
    backend = make_backend(existing_node=None)
    db = make_db()
    plugin = make_tx(db=db, backend=backend)
    result = await plugin.transform_test(db, "ghost")
    assert result["success"] is False


async def test_transform_test_substitutes_dummy_params():
    """TX-10: :param placeholders are replaced with '__test__' before EXPLAIN."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "SELECT * FROM orders WHERE status = :status",
            "target_table": None,
        },
    )
    backend = make_backend(existing_node=node)
    db = MagicMock()
    db.query = AsyncMock(return_value=[{"QUERY PLAN": "Seq Scan"}])
    plugin = make_tx(db=db, backend=backend)
    await plugin.transform_test(db, "tx")

    explain_sql = db.query.call_args[0][0]
    assert explain_sql.startswith("EXPLAIN")
    assert ":status" not in explain_sql
    assert "'__test__'" in explain_sql


# ---------------------------------------------------------------------------
# transform_version
# ---------------------------------------------------------------------------


async def test_transform_version_appends_snapshot():
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={"sql": "SELECT 1", "versions": [], "target_table": None},
    )
    backend = make_backend(existing_node=node)
    plugin = make_tx(backend=backend)
    result = await plugin.transform_version("tx")
    assert result["success"] is True
    assert result["version_index"] == 0
    assert result["total_versions"] == 1
    assert result["snapshot"]["sql"] == "SELECT 1"


async def test_transform_version_without_backend():
    """TX-02: version snapshot works with in-memory fallback."""
    plugin = make_tx()
    await plugin.transform_create("mem_tx", sql="SELECT 42")

    result = await plugin.transform_version("mem_tx")
    assert result["success"] is True
    assert result["version_index"] == 0
    assert result["snapshot"]["sql"] == "SELECT 42"


async def test_transform_version_not_found():
    backend = make_backend(existing_node=None)
    plugin = make_tx(backend=backend)
    result = await plugin.transform_version("nonexistent")
    assert result["success"] is False


# ---------------------------------------------------------------------------
# transform_diff
# ---------------------------------------------------------------------------


async def test_transform_diff_shows_changes():
    """TX-08: version_b='current' refers to live SQL."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "SELECT id, name FROM orders",
            "versions": [
                {"sql": "SELECT id FROM orders", "created_at": "2024-01-01T00:00:00"},
            ],
            "target_table": None,
        },
    )
    backend = make_backend(existing_node=node)
    plugin = make_tx(backend=backend)
    result = await plugin.transform_diff("tx", version_a=0, version_b="current")
    assert result["success"] is True
    assert result["changed"] is True
    assert "name" in result["diff"]


async def test_transform_diff_no_change():
    """TX-08: comparing snapshot to 'current' shows no diff when SQL is identical."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "SELECT 1",
            "versions": [{"sql": "SELECT 1", "created_at": "2024-01-01T00:00:00"}],
            "target_table": None,
        },
    )
    backend = make_backend(existing_node=node)
    plugin = make_tx(backend=backend)
    result = await plugin.transform_diff("tx", version_a=0, version_b="current")
    assert result["changed"] is False


async def test_transform_diff_between_snapshots():
    """Diffing two integer-indexed snapshots works correctly."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={
            "sql": "SELECT id, name, email FROM orders",
            "versions": [
                {"sql": "SELECT id FROM orders", "created_at": "2024-01-01T00:00:00"},
                {
                    "sql": "SELECT id, name FROM orders",
                    "created_at": "2024-01-02T00:00:00",
                },
            ],
            "target_table": None,
        },
    )
    backend = make_backend(existing_node=node)
    plugin = make_tx(backend=backend)
    result = await plugin.transform_diff("tx", version_a=0, version_b=1)
    assert result["success"] is True
    assert result["changed"] is True


async def test_transform_diff_out_of_range():
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:tx",
        node_type=NodeType.TRANSFORMATION,
        name="tx",
        properties={"sql": "SELECT 1", "versions": [], "target_table": None},
    )
    backend = make_backend(existing_node=node)
    plugin = make_tx(backend=backend)
    result = await plugin.transform_diff("tx", version_a=99, version_b=0)
    assert result["success"] is False
    assert "out of range" in result["error"]


# ---------------------------------------------------------------------------
# transform_list
# ---------------------------------------------------------------------------


async def test_transform_list_returns_transformations():
    from daita.core.graph.models import AgentGraphNode, NodeType

    node1 = AgentGraphNode(
        node_id="transformation:tx1",
        node_type=NodeType.TRANSFORMATION,
        name="tx1",
        properties={
            "description": "First",
            "source_tables": [],
            "target_table": None,
            "versions": [],
        },
    )
    node2 = AgentGraphNode(
        node_id="transformation:tx2",
        node_type=NodeType.TRANSFORMATION,
        name="tx2",
        properties={
            "description": "Second",
            "source_tables": [],
            "target_table": None,
            "versions": [],
        },
    )
    # Include a non-transformation node to verify filtering
    table_node = AgentGraphNode(
        node_id="table:orders",
        node_type=NodeType.TABLE,
        name="orders",
        properties={},
    )

    mock_graph = MagicMock()
    mock_graph.nodes.return_value = [
        ("transformation:tx1", {"data": node1}),
        ("transformation:tx2", {"data": node2}),
        ("table:orders", {"data": table_node}),
    ]

    backend = make_backend()
    backend.load_graph = AsyncMock(return_value=mock_graph)
    plugin = make_tx(backend=backend)

    result = await plugin.transform_list()
    assert result["success"] is True
    assert result["count"] == 2
    names = [t["name"] for t in result["transformations"]]
    assert "tx1" in names
    assert "tx2" in names
    assert "orders" not in names


async def test_transform_list_without_backend():
    """Empty _definitions + no backend → success: False."""
    plugin = make_tx()  # no backend, no definitions
    result = await plugin.transform_list()
    assert result["success"] is False


async def test_transform_list_uses_in_memory_definitions():
    """TX-11: transform_list uses _definitions when populated; does not call load_graph."""
    backend = make_backend()
    plugin = make_tx(backend=backend)

    # Populate _definitions without going through the graph path
    await plugin.transform_create("in_mem_tx", sql="SELECT 1", description="inline")

    result = await plugin.transform_list()
    assert result["success"] is True
    assert result["count"] == 1
    assert result["transformations"][0]["name"] == "in_mem_tx"
    # load_graph should NOT have been called — _definitions was used instead
    backend.load_graph.assert_not_called()


async def test_transform_list_cold_start_uses_graph():
    """TX-11: when _definitions is empty, falls back to load_graph (cold-start)."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node = AgentGraphNode(
        node_id="transformation:cold_tx",
        node_type=NodeType.TRANSFORMATION,
        name="cold_tx",
        properties={
            "description": "",
            "source_tables": [],
            "target_table": None,
            "versions": [],
        },
    )
    mock_graph = MagicMock()
    mock_graph.nodes.return_value = [("transformation:cold_tx", {"data": node})]

    backend = make_backend()
    backend.load_graph = AsyncMock(return_value=mock_graph)
    plugin = make_tx(backend=backend)
    # _definitions is empty (no transform_create called)
    result = await plugin.transform_list()
    assert result["success"] is True
    assert result["count"] == 1
    backend.load_graph.assert_called_once()


async def test_transform_list_in_memory_no_backend():
    """TX-03: list works with in-memory fallback after create."""
    plugin = make_tx()  # no backend
    await plugin.transform_create("a", sql="SELECT 1")
    await plugin.transform_create("b", sql="SELECT 2")

    result = await plugin.transform_list()
    assert result["success"] is True
    assert result["count"] == 2
    names = {t["name"] for t in result["transformations"]}
    assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# Tool handler dispatches (smoke tests)
# ---------------------------------------------------------------------------


async def test_tool_create_dispatches():
    backend = make_backend()
    plugin = make_tx(backend=backend)
    result = await plugin._tool_create(
        {
            "name": "test_tx",
            "sql": "SELECT 1",
            "description": "smoke test",
        }
    )
    assert result["success"] is True


async def test_tool_run_no_db_raises():
    plugin = make_tx()
    with pytest.raises(ValueError):
        await plugin._tool_run({"name": "tx"})


async def test_tool_list_dispatches():
    backend = make_backend()
    plugin = make_tx(backend=backend)
    result = await plugin._tool_list({})
    assert result["success"] is True
