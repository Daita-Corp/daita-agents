import pytest

from daita.core.graph.local_backend import LocalGraphBackend
from daita.core.graph.models import AgentGraphEdge, AgentGraphNode, EdgeType, NodeType
from daita.core.graph.tools import build_graph_tools


async def test_graph_tools_project_subgraph_and_shortest_path(tmp_path):
    pytest.importorskip("networkx")
    backend = LocalGraphBackend(graph_type="tool_projection", storage_dir=tmp_path)
    orders = AgentGraphNode(
        node_id="table:orders",
        node_type=NodeType.TABLE,
        name="orders",
    )
    customers = AgentGraphNode(
        node_id="table:customers",
        node_type=NodeType.TABLE,
        name="customers",
    )
    reference = AgentGraphEdge(
        edge_id="orders_references_customers",
        from_node_id=orders.node_id,
        to_node_id=customers.node_id,
        edge_type=EdgeType.REFERENCES,
    )
    await backend.add_node(orders)
    await backend.add_node(customers)
    await backend.add_edge(reference)

    tools = {tool.name: tool for tool in build_graph_tools(backend)}

    subgraph = await tools["graph_subgraph"].execute(
        {"root": orders.node_id, "direction": "downstream", "max_depth": 1}
    )
    shortest_path = await tools["graph_shortest_path"].execute(
        {"from_id": orders.node_id, "to_id": customers.node_id, "max_depth": 2}
    )

    assert set(tools) == {"graph_subgraph", "graph_shortest_path"}
    assert all(tool.category == "graph" for tool in tools.values())
    assert all(tool.source == "core" for tool in tools.values())
    assert all(tool.plugin_name == "GraphQuery" for tool in tools.values())
    assert subgraph["node_count"] == 2
    assert subgraph["edge_count"] == 1
    assert shortest_path["path"] == [orders.node_id, customers.node_id]
    assert shortest_path["reachable"] is True
