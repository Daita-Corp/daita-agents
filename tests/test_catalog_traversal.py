from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast

import pytest

import daita.catalog.service as catalog_service
from daita import Agent, SQLiteSource
from daita.capabilities import ToolExecution
from daita.catalog import (
    CATALOG_TRAVERSE_EVIDENCE_KIND,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTraversalRequest,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    catalog_resource_id,
)
from daita.catalog.models import CatalogRelationship
from daita.catalog.protocols import CatalogResourceNotFoundError, CatalogStoreError

_OBSERVED_AT = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class _EdgeSpec:
    name: str
    source: str
    target: str
    kind: RelationshipKind = RelationshipKind.REFERENCES
    field_pairs: tuple[tuple[str, str], ...] = (("id", "id"),)


@dataclass(frozen=True, slots=True)
class _CommittedGraph:
    resource_ids: dict[str, str]
    relationship_ids: dict[str, str]


class _TraversalFieldPair(TypedDict):
    ordinal: int
    source_field: str
    target_field: str


class _TraversalStep(TypedDict):
    relationship_id: str
    direction: str
    path_from_resource_id: str
    path_to_resource_id: str
    field_pairs: tuple[_TraversalFieldPair, ...]


class _TraversalPath(TypedDict):
    resource_ids: tuple[str, ...]
    steps: tuple[_TraversalStep, ...]


class _TraversalProjection(TypedDict):
    paths: tuple[_TraversalPath, ...]
    reachable: bool
    truncated: bool
    truncation_reason: str | None
    visited_edges: int
    visited_nodes: int


@pytest.fixture
async def graph_agent(tmp_path: Path):
    database = tmp_path / "empty.sqlite"
    database.touch()
    agent = await Agent.create("catalog-traversal", root=tmp_path)
    source = await agent.attach(SQLiteSource(database, name="Traversal"))
    try:
        yield agent, source
    finally:
        await agent.close()


async def _attach_empty_source(
    agent: Agent,
    tmp_path: Path,
    name: str,
):
    database = tmp_path / f"{name}.sqlite"
    database.touch()
    return await agent.attach(SQLiteSource(database, name=name))


async def _commit_graph(
    agent: Agent,
    source_id: str,
    *,
    nodes: tuple[str, ...],
    edges: tuple[_EdgeSpec, ...],
    sync_id: str,
) -> _CommittedGraph:
    resource_ids = {
        name: catalog_resource_id(source_id, ResourceKind.TABLE, f"main.{name}")
        for name in nodes
    }
    relationships = tuple(
        CatalogRelationship.build(
            source_id=source_id,
            from_resource_id=resource_ids[edge.source],
            to_resource_id=resource_ids[edge.target],
            kind=edge.kind,
            provenance=RelationshipProvenance.CONNECTOR,
            confidence=1.0,
            sync_id=sync_id,
            observed_at=_OBSERVED_AT,
            field_pairs=tuple(
                RelationshipFieldPair(
                    source_field=source_field,
                    target_field=target_field,
                    ordinal=ordinal,
                )
                for ordinal, (source_field, target_field) in enumerate(edge.field_pairs)
            ),
        )
        for edge in edges
    )
    incident_revisions: dict[str, list[str]] = {
        resource_id: [] for resource_id in resource_ids.values()
    }
    for relationship in relationships:
        incident_revisions[relationship.from_resource_id].append(relationship.revision)
        incident_revisions[relationship.to_resource_id].append(relationship.revision)
    revisions = tuple(
        CatalogResourceRevision.build(
            resource_id=resource_ids[name],
            sync_id=sync_id,
            observed_at=_OBSERVED_AT,
            relationship_revisions=incident_revisions[resource_ids[name]],
            source_revision=f"test:{sync_id}",
        )
        for name in nodes
    )
    revision_by_resource_id = {revision.resource_id: revision for revision in revisions}
    resources = tuple(
        CatalogResource.build(
            agent_id=agent.id,
            source_id=source_id,
            native_identity=f"main.{name}",
            external_uri=f"test://{source_id}/main/{name}",
            kind=ResourceKind.TABLE,
            name=name,
            sensitivity=Sensitivity.INTERNAL,
            revision=revision_by_resource_id[resource_ids[name]],
            first_observed_at=_OBSERVED_AT,
            last_observed_at=_OBSERVED_AT,
        )
        for name in nodes
    )
    snapshot = SourceCatalogSnapshot(
        sync=CatalogSync(
            id=sync_id,
            agent_id=agent.id,
            source_id=source_id,
            adapter_id="test-graph",
            status=CatalogSyncStatus.SUCCEEDED,
            started_at=_OBSERVED_AT,
            completed_at=_OBSERVED_AT,
            source_revision=f"test:{sync_id}",
            resource_count=len(resources),
            relationship_count=len(relationships),
        ),
        resources=resources,
        revisions=revisions,
        relationships=relationships,
    )
    await agent._embedded._store.commit_snapshot(snapshot)
    return _CommittedGraph(
        resource_ids=resource_ids,
        relationship_ids={
            edge.name: relationship.id
            for edge, relationship in zip(edges, relationships, strict=True)
        },
    )


async def _traverse(
    agent: Agent,
    graph: _CommittedGraph,
    from_names: tuple[str, ...],
    to_names: tuple[str, ...],
    *,
    relationship_kinds: tuple[RelationshipKind, ...] = (),
    max_depth: int = 6,
    max_paths: int = 8,
    max_nodes: int = 256,
    max_edges: int = 1_024,
) -> _TraversalProjection:
    projection = await agent._embedded._catalog_service.traverse(
        CatalogTraversalRequest(
            agent_id=agent.id,
            from_resource_ids=tuple(graph.resource_ids[name] for name in from_names),
            to_resource_ids=tuple(graph.resource_ids[name] for name in to_names),
            relationship_kinds=relationship_kinds,
            max_depth=max_depth,
            max_paths=max_paths,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
    )
    return cast(_TraversalProjection, projection)


def _path_signature(
    projection: _TraversalProjection,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            path["resource_ids"],
            tuple(
                (
                    step["relationship_id"],
                    step["direction"],
                    step["path_from_resource_id"],
                    step["path_to_resource_id"],
                )
                for step in path["steps"]
            ),
        )
        for path in projection["paths"]
    )


async def test_direct_traversal_counts_admitted_nodes_and_examined_adjacency(
    graph_agent,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b"),
        edges=(_EdgeSpec("a-b", "a", "b"),),
        sync_id="direct-1",
    )

    projection = await _traverse(agent, graph, ("a",), ("b",))

    assert projection["reachable"] is True
    assert projection["visited_nodes"] == 2
    assert projection["visited_edges"] == 1
    assert projection["truncated"] is False
    assert projection["truncation_reason"] is None
    assert _path_signature(projection) == (
        (
            (graph.resource_ids["a"], graph.resource_ids["b"]),
            (
                (
                    graph.relationship_ids["a-b"],
                    "forward",
                    graph.resource_ids["a"],
                    graph.resource_ids["b"],
                ),
            ),
        ),
    )


async def test_chain_counts_reverse_entries_when_they_reach_discovered_nodes(
    graph_agent,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b", "c"),
        edges=(
            _EdgeSpec("a-b", "a", "b"),
            _EdgeSpec("b-c", "b", "c"),
        ),
        sync_id="chain-1",
    )

    projection = await _traverse(agent, graph, ("a",), ("c",))

    assert projection["visited_nodes"] == 3
    assert projection["visited_edges"] == 3
    assert projection["truncation_reason"] is None
    assert tuple(
        step["relationship_id"] for step in projection["paths"][0]["steps"]
    ) == (graph.relationship_ids["a-b"], graph.relationship_ids["b-c"])


async def test_diamond_reconstructs_only_both_deterministic_shortest_paths(
    graph_agent,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b", "c", "d"),
        edges=(
            _EdgeSpec("a-b", "a", "b"),
            _EdgeSpec("a-c", "a", "c"),
            _EdgeSpec("b-d", "b", "d"),
            _EdgeSpec("c-d", "c", "d"),
        ),
        sync_id="diamond-1",
    )

    projection = await _traverse(agent, graph, ("a",), ("d",))

    assert projection["visited_nodes"] == 4
    assert projection["visited_edges"] == 6
    assert tuple(path["resource_ids"] for path in projection["paths"]) == (
        (graph.resource_ids["a"], graph.resource_ids["b"], graph.resource_ids["d"]),
        (graph.resource_ids["a"], graph.resource_ids["c"], graph.resource_ids["d"]),
    )


async def test_cycle_returns_the_direct_shortest_simple_path_without_queueing_paths(
    graph_agent,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b", "c"),
        edges=(
            _EdgeSpec("a-b", "a", "b"),
            _EdgeSpec("b-c", "b", "c"),
            _EdgeSpec("a-c", "a", "c", field_pairs=(("direct_id", "id"),)),
        ),
        sync_id="cycle-1",
    )

    projection = await _traverse(agent, graph, ("a",), ("c",))

    assert tuple(path["resource_ids"] for path in projection["paths"]) == (
        (graph.resource_ids["a"], graph.resource_ids["c"]),
    )
    assert projection["paths"][0]["steps"][0]["relationship_id"] == (
        graph.relationship_ids["a-c"]
    )
    assert projection["visited_nodes"] == 3
    assert projection["visited_edges"] == 4


async def test_disconnected_graph_is_honestly_unreachable_without_truncation(
    graph_agent,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b", "c", "d"),
        edges=(
            _EdgeSpec("a-b", "a", "b"),
            _EdgeSpec("c-d", "c", "d"),
        ),
        sync_id="disconnected-1",
    )

    projection = await _traverse(agent, graph, ("a",), ("d",))

    assert projection["paths"] == ()
    assert projection["reachable"] is False
    assert projection["visited_nodes"] == 2
    assert projection["visited_edges"] == 2
    assert projection["truncated"] is False
    assert projection["truncation_reason"] is None


async def test_multi_source_and_target_traversal_stays_source_shard_isolated(
    tmp_path: Path,
):
    first_db = tmp_path / "first.sqlite"
    first_db.touch()
    agent = await Agent.create("catalog-traversal-multi-source", root=tmp_path)
    first = await agent.attach(SQLiteSource(first_db, name="First"))
    second = await _attach_empty_source(agent, tmp_path, "second")
    try:
        first_graph = await _commit_graph(
            agent,
            first.id,
            nodes=("a1", "b1"),
            edges=(_EdgeSpec("a1-b1", "a1", "b1"),),
            sync_id="multi-first-1",
        )
        second_graph = await _commit_graph(
            agent,
            second.id,
            nodes=("a2", "b2"),
            edges=(_EdgeSpec("a2-b2", "a2", "b2"),),
            sync_id="multi-second-1",
        )
        combined = _CommittedGraph(
            resource_ids={
                **first_graph.resource_ids,
                **second_graph.resource_ids,
            },
            relationship_ids={
                **first_graph.relationship_ids,
                **second_graph.relationship_ids,
            },
        )

        projection = await _traverse(
            agent,
            combined,
            ("a2", "a1"),
            ("b2", "b1"),
        )
        assert {path["resource_ids"] for path in projection["paths"]} == {
            (combined.resource_ids["a1"], combined.resource_ids["b1"]),
            (combined.resource_ids["a2"], combined.resource_ids["b2"]),
        }
        assert projection["visited_nodes"] == 4
        assert projection["visited_edges"] == 2

        disconnected = await _traverse(agent, combined, ("a1",), ("b2",))
        assert disconnected["reachable"] is False
        assert disconnected["truncated"] is False
    finally:
        await agent.close()


async def test_relationship_filter_reverse_direction_and_composite_fields(
    graph_agent,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("child", "folder", "parent"),
        edges=(
            _EdgeSpec(
                "folder-child",
                "folder",
                "child",
                kind=RelationshipKind.CONTAINS,
                field_pairs=(),
            ),
            _EdgeSpec(
                "child-parent",
                "child",
                "parent",
                field_pairs=(("tenant_id", "tenant_id"), ("parent_id", "id")),
            ),
        ),
        sync_id="filtered-reverse-composite-1",
    )

    projection = await _traverse(
        agent,
        graph,
        ("parent",),
        ("child", "folder"),
        relationship_kinds=(RelationshipKind.REFERENCES,),
    )

    assert tuple(path["resource_ids"] for path in projection["paths"]) == (
        (graph.resource_ids["parent"], graph.resource_ids["child"]),
    )
    step = projection["paths"][0]["steps"][0]
    assert step["relationship_id"] == graph.relationship_ids["child-parent"]
    assert step["direction"] == "reverse"
    assert step["path_from_resource_id"] == graph.resource_ids["parent"]
    assert step["path_to_resource_id"] == graph.resource_ids["child"]
    assert tuple(
        (pair["ordinal"], pair["source_field"], pair["target_field"])
        for pair in step["field_pairs"]
    ) == (
        (0, "tenant_id", "tenant_id"),
        (1, "parent_id", "id"),
    )
    assert projection["visited_nodes"] == 2
    assert projection["visited_edges"] == 1


async def test_depth_bound_reports_the_first_unadmitted_continuation(graph_agent):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b", "c"),
        edges=(
            _EdgeSpec("a-b", "a", "b"),
            _EdgeSpec("b-c", "b", "c"),
        ),
        sync_id="depth-bound-1",
    )

    projection = await _traverse(
        agent,
        graph,
        ("a",),
        ("c",),
        max_depth=1,
    )

    assert projection["reachable"] is False
    assert projection["visited_nodes"] == 2
    assert projection["visited_edges"] == 3
    assert projection["truncated"] is True
    assert projection["truncation_reason"] == "depth"


async def test_node_bound_counts_the_triggering_examined_adjacency(graph_agent):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b", "c"),
        edges=(
            _EdgeSpec("a-b", "a", "b"),
            _EdgeSpec("a-c", "a", "c"),
        ),
        sync_id="node-bound-1",
    )

    projection = await _traverse(
        agent,
        graph,
        ("a",),
        ("c",),
        max_nodes=2,
    )

    assert projection["reachable"] is False
    assert projection["visited_nodes"] == 2
    assert projection["visited_edges"] == 2
    assert projection["truncated"] is True
    assert projection["truncation_reason"] == "nodes"


async def test_edge_bound_never_counts_or_examines_beyond_the_limit(graph_agent):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b", "c"),
        edges=(
            _EdgeSpec("a-b", "a", "b"),
            _EdgeSpec("a-c", "a", "c"),
        ),
        sync_id="edge-bound-1",
    )

    projection = await _traverse(
        agent,
        graph,
        ("a",),
        ("c",),
        max_edges=1,
    )

    assert projection["reachable"] is False
    assert projection["visited_nodes"] == 2
    assert projection["visited_edges"] == 1
    assert projection["truncated"] is True
    assert projection["truncation_reason"] == "edges"


async def test_path_bound_reconstructs_only_the_bounded_shortest_prefix(graph_agent):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b", "c", "d"),
        edges=(
            _EdgeSpec("a-b", "a", "b"),
            _EdgeSpec("a-c", "a", "c"),
            _EdgeSpec("b-d", "b", "d"),
            _EdgeSpec("c-d", "c", "d"),
        ),
        sync_id="path-bound-1",
    )

    projection = await _traverse(
        agent,
        graph,
        ("a",),
        ("d",),
        max_paths=1,
    )

    assert tuple(path["resource_ids"] for path in projection["paths"]) == (
        (graph.resource_ids["a"], graph.resource_ids["b"], graph.resource_ids["d"]),
    )
    assert projection["visited_nodes"] == 4
    assert projection["visited_edges"] == 6
    assert projection["truncated"] is True
    assert projection["truncation_reason"] == "paths"


async def test_exactly_exhausted_bounds_are_not_reported_as_truncated(graph_agent):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b"),
        edges=(_EdgeSpec("a-b", "a", "b"),),
        sync_id="exact-bounds-1",
    )

    projection = await _traverse(
        agent,
        graph,
        ("a",),
        ("b",),
        max_depth=1,
        max_paths=1,
        max_nodes=2,
        max_edges=1,
    )

    assert projection["reachable"] is True
    assert projection["visited_nodes"] == 2
    assert projection["visited_edges"] == 1
    assert projection["truncated"] is False
    assert projection["truncation_reason"] is None


async def test_dense_frontier_stops_at_unique_node_admission_without_path_growth(
    graph_agent,
):
    agent, source = graph_agent
    connected = tuple(chr(ord("a") + index) for index in range(12))
    edges = tuple(
        _EdgeSpec(
            f"{left}-{right}",
            left,
            right,
            field_pairs=((f"{right}_id", "id"),),
        )
        for left_index, left in enumerate(connected)
        for right in connected[left_index + 1 :]
    )
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=(*connected, "z"),
        edges=edges,
        sync_id="dense-1",
    )

    projection = await _traverse(
        agent,
        graph,
        ("a",),
        ("z",),
        max_nodes=6,
        max_edges=100,
    )

    assert projection["reachable"] is False
    assert projection["visited_nodes"] == 6
    assert projection["visited_edges"] == 6
    assert projection["truncated"] is True
    assert projection["truncation_reason"] == "nodes"


async def test_path_order_is_independent_of_snapshot_insertion_order(graph_agent):
    agent, source = graph_agent
    nodes = ("a", "b", "c", "d")
    edges = (
        _EdgeSpec("a-b", "a", "b"),
        _EdgeSpec("a-c", "a", "c"),
        _EdgeSpec("b-d", "b", "d"),
        _EdgeSpec("c-d", "c", "d"),
    )
    first = await _commit_graph(
        agent,
        source.id,
        nodes=nodes,
        edges=edges,
        sync_id="order-1",
    )
    first_projection = await _traverse(agent, first, ("a",), ("d",))

    second = await _commit_graph(
        agent,
        source.id,
        nodes=tuple(reversed(nodes)),
        edges=tuple(reversed(edges)),
        sync_id="order-2",
    )
    second_projection = await _traverse(agent, second, ("a",), ("d",))

    assert first.resource_ids == second.resource_ids
    assert first.relationship_ids == second.relationship_ids
    assert _path_signature(first_projection) == _path_signature(second_projection)
    assert first_projection["visited_nodes"] == second_projection["visited_nodes"]
    assert first_projection["visited_edges"] == second_projection["visited_edges"]


async def test_refresh_replaces_the_exact_generation_and_removes_stale_edges(
    graph_agent,
):
    agent, source = graph_agent
    old = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b"),
        edges=(_EdgeSpec("a-b", "a", "b"),),
        sync_id="refresh-1",
    )
    assert (await _traverse(agent, old, ("a",), ("b",)))["reachable"] is True

    new = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "c"),
        edges=(_EdgeSpec("a-c", "a", "c"),),
        sync_id="refresh-2",
    )
    projection = await _traverse(agent, new, ("a",), ("c",))
    assert projection["paths"][0]["steps"][0]["relationship_id"] == (
        new.relationship_ids["a-c"]
    )
    assert all(
        key != (agent.id, source.id, "refresh-1")
        for key in agent._embedded._catalog_service._source_indexes
    )
    with pytest.raises(CatalogResourceNotFoundError):
        await agent._embedded._catalog_service.traverse(
            CatalogTraversalRequest(
                agent_id=agent.id,
                from_resource_ids=(new.resource_ids["a"],),
                to_resource_ids=(old.resource_ids["b"],),
            )
        )


async def test_inactive_sources_and_other_agents_are_never_traversable(
    tmp_path: Path,
):
    first_db = tmp_path / "agent-one.sqlite"
    second_db = tmp_path / "agent-two.sqlite"
    first_db.touch()
    second_db.touch()
    first = await Agent.create("catalog-traversal-agent-one", root=tmp_path)
    second = await Agent.create("catalog-traversal-agent-two", root=tmp_path)
    first_source = await first.attach(SQLiteSource(first_db))
    second_source = await second.attach(SQLiteSource(second_db))
    try:
        first_graph = await _commit_graph(
            first,
            first_source.id,
            nodes=("a", "b"),
            edges=(_EdgeSpec("a-b", "a", "b"),),
            sync_id="agent-one-1",
        )
        second_graph = await _commit_graph(
            second,
            second_source.id,
            nodes=("x", "y"),
            edges=(_EdgeSpec("x-y", "x", "y"),),
            sync_id="agent-two-1",
        )

        with pytest.raises(CatalogResourceNotFoundError):
            await first._embedded._catalog_service.traverse(
                CatalogTraversalRequest(
                    agent_id=second.id,
                    from_resource_ids=(second_graph.resource_ids["x"],),
                    to_resource_ids=(second_graph.resource_ids["y"],),
                )
            )

        await first.detach(first_source.id)
        with pytest.raises(CatalogResourceNotFoundError):
            await first._embedded._catalog_service.traverse(
                CatalogTraversalRequest(
                    agent_id=first.id,
                    from_resource_ids=(first_graph.resource_ids["a"],),
                    to_resource_ids=(first_graph.resource_ids["b"],),
                )
            )
        assert first._embedded._catalog_service._source_indexes == {}
    finally:
        await first.close()
        await second.close()


async def test_traversal_reuses_one_compiled_index_per_exact_generation(
    graph_agent,
    monkeypatch: pytest.MonkeyPatch,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b"),
        edges=(_EdgeSpec("a-b", "a", "b"),),
        sync_id="compile-reuse-1",
    )
    original_compile = catalog_service._compile_source_index
    compile_count = 0

    def counting_compile(snapshot):
        nonlocal compile_count
        compile_count += 1
        return original_compile(snapshot)

    monkeypatch.setattr(catalog_service, "_compile_source_index", counting_compile)

    first = await _traverse(agent, graph, ("a",), ("b",))
    second = await _traverse(agent, graph, ("a",), ("b",))

    assert first == second
    assert compile_count == 1


async def test_traversal_capability_validates_the_explicit_truncation_contract(
    graph_agent,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b"),
        edges=(_EdgeSpec("a-b", "a", "b"),),
        sync_id="capability-output-1",
    )
    registry = agent._embedded._capabilities
    _, capability = registry.resolve_tool("catalog_traverse")
    _, executor = registry.resolve_execution(capability.id)

    output = await executor.execute(
        ToolExecution(
            run_id="catalog-traversal-capability",
            call_id="catalog-traversal-call",
            capability_id=capability.id,
            arguments={
                "from_resource_ids": (graph.resource_ids["a"],),
                "to_resource_ids": (graph.resource_ids["b"],),
            },
        )
    )

    assert output.kind == CATALOG_TRAVERSE_EVIDENCE_KIND
    assert output.data["truncated"] is False
    assert output.data["truncation_reason"] is None
    assert registry.validate_output(capability.id, output) == output


async def test_traversal_retries_one_generation_conflict_then_returns_catalog_error(
    graph_agent,
    monkeypatch: pytest.MonkeyPatch,
):
    agent, source = graph_agent
    graph = await _commit_graph(
        agent,
        source.id,
        nodes=("a", "b"),
        edges=(_EdgeSpec("a-b", "a", "b"),),
        sync_id="generation-conflict-1",
    )
    store = agent._embedded._store
    load_count = 0

    async def changing_generation(ref):
        nonlocal load_count
        load_count += 1
        return None

    monkeypatch.setattr(store, "load_current_snapshot", changing_generation)

    with pytest.raises(
        CatalogStoreError,
        match="generation changed repeatedly",
    ):
        await _traverse(agent, graph, ("a",), ("b",))
    assert load_count == 2
