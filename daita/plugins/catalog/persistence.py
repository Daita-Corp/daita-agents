"""
Schema persistence — local JSON and graph backend storage.

Handles writing discovered schemas to .daita/catalog.json (local default)
or a registered catalog backend, plus graph node creation for LineagePlugin.

Graph model — fan-out stores (relational, warehouse, document, NoSQL):

    Table   id: table:<store>.<table>
      │
      ├─:HAS_COLUMN─►  Column   id: column:<store>.<table>.<col>
      │
      └─:INDEXED_BY─►  Index    id: index:<store>.<table>.<idx>
                        │
                        └─:COVERS (position)─►  Column

    Column ─:REFERENCES─►  Column     (foreign keys, intra-store)

Every Table / Column / Index ID carries the store identifier so the graph is
collision-free across multiple data sources in a single CatalogPlugin.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def persist_schema(
    schema: Dict[str, Any],
    catalog_backend: Optional[Any],
    graph_backend: Optional[Any],
    agent_id: Optional[str],
) -> bool:
    """Persist schema to the catalog store and graph backend.

    Storage selection (in priority order):
    1. catalog_backend — set during initialize() when DAITA_CATALOG_BACKEND_CLASS
       is present. Used in cloud deployments. Falls through to local JSON on failure.
    2. Local .daita/catalog.json — default for local development.

    In both cases, if a graph backend is available, schema entities are also
    written as graph nodes so LineagePlugin can reference them by node_id.

    Returns True if schema was persisted, False if the operation was skipped.
    """
    import aiofiles
    from datetime import datetime, timezone
    from pathlib import Path

    persisted = False

    if catalog_backend is not None:
        try:
            persisted = await catalog_backend.persist_schema(schema)
        except Exception as exc:
            logger.warning(
                "Catalog backend failed, falling back to local JSON: %s", exc
            )

    if not persisted:
        catalog_path = Path(".daita") / "catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)

        existing: Dict[str, Any] = {}
        if catalog_path.exists():
            try:
                async with aiofiles.open(catalog_path, "r") as f:
                    existing = json.loads(await f.read())
            except (json.JSONDecodeError, ValueError):
                logger.warning("catalog.json was corrupt, overwriting.")

        key = f"{schema.get('database_type', 'unknown')}:{schema.get('schema', 'default')}"
        now = datetime.now(timezone.utc).isoformat()

        if key in existing:
            # Preserve first_seen from initial discovery; update last_seen
            schema["first_seen"] = existing[key].get("first_seen", now)
            schema["last_seen"] = now
        else:
            schema["first_seen"] = now
            schema["last_seen"] = now

        existing[key] = schema

        async with aiofiles.open(catalog_path, "w") as f:
            await f.write(json.dumps(existing, indent=2, default=str))

        logger.debug("Persisted schema to %s", catalog_path)
        persisted = True

    # Write discovered entities as graph nodes so LineagePlugin can reference
    # them by node_id (e.g. "table:<store>.<name>"). Runs in both local and
    # cloud paths.
    if graph_backend:
        try:
            await persist_schema_to_graph(schema, graph_backend, agent_id)
        except Exception as graph_err:
            logger.error(
                "Failed to persist schema to graph backend (schema data was saved): %s",
                graph_err,
            )
            schema["graph_persist_error"] = str(graph_err)

    return persisted


async def persist_schema_to_graph(
    schema: Dict[str, Any],
    graph_backend: Any,
    agent_id: Optional[str],
) -> None:
    """
    Write discovered schema entities as nodes into the graph backend.

    Every entry in :data:`_NODE_HANDLERS` is a (mode, NodeType) pair where:

      * ``mode="fan_out"`` fans ``schema["tables"]`` out into one node per
        table, plus one Column node per column, one Index node per declared
        index, one ``:HAS_COLUMN`` edge, one ``:INDEXED_BY`` edge, and
        ``:COVERS`` edges with per-position ordinality. Foreign keys emit
        ``:REFERENCES`` edges between Column nodes.
      * ``mode="single"`` writes one parent node for the whole schema.
        Used for bucket/API/stream/queue/topic-shaped sources where the store
        itself is the atomic unit — these IDs are already unique by service
        name and don't need store qualification.

    Unknown ``database_type`` values are silently skipped — the schema has
    already been persisted to local JSON or the catalog backend by the caller.
    """
    db_type = schema.get("database_type") or schema.get("api_type", "unknown")
    schema_name = schema.get("database_name") or schema.get("schema") or "default"

    spec = _NODE_HANDLERS.get(db_type)
    if spec is not None:
        mode, node_type = spec
        if mode == "fan_out":
            await _write_fan_out(schema, graph_backend, agent_id)
        else:
            await _write_single_node(schema, graph_backend, agent_id, node_type)

    # LocalGraphBackend uses lazy writes — flush at the boundary of a logical unit.
    if hasattr(graph_backend, "flush"):
        await graph_backend.flush()

    logger.debug(
        "CatalogPlugin: persisted %s:%s entities to graph backend", db_type, schema_name
    )


# Dispatch table: database_type -> (mode, NodeType).
#   mode = "fan_out" | "single"  — see persist_schema_to_graph() docstring.
_NODE_HANDLERS: Dict[str, tuple] = {
    # Relational / document / warehouse / search sources — one node per
    # table / collection / index.
    "postgresql": ("fan_out", "TABLE"),
    "mysql": ("fan_out", "TABLE"),
    "mongodb": ("fan_out", "TABLE"),
    "documentdb": ("fan_out", "TABLE"),
    "bigquery": ("fan_out", "TABLE"),
    "firestore": ("fan_out", "TABLE"),
    "bigtable": ("fan_out", "TABLE"),
    "opensearch": ("fan_out", "TABLE"),
    # API services — single node for the service.
    "openapi": ("single", "API"),
    "apigateway": ("single", "API"),
    "gcp_apigateway": ("single", "API"),
    # Object stores — single node for the bucket.
    "s3": ("single", "BUCKET"),
    "gcs": ("single", "BUCKET"),
    # Single-entity databases.
    "dynamodb": ("single", "DATABASE"),
    "memorystore": ("single", "DATABASE"),
    # Streaming / messaging — single node for the topic / queue / stream.
    "pubsub_topic": ("single", "SERVICE"),
    "pubsub_subscription": ("single", "SERVICE"),
    "sns": ("single", "SERVICE"),
    "sqs": ("single", "SERVICE"),
    "kinesis": ("single", "SERVICE"),
}


def _build_schema_name(schema: Dict[str, Any]) -> str:
    """Resolve the human-facing name for a schema across varied source shapes."""
    for key in ("database_name", "schema", "bucket", "table_name", "service_name"):
        value = schema.get(key)
        if value:
            return value
    return "unknown"


def _derive_store(schema: Dict[str, Any]) -> str:
    """Return a stable store identifier for this schema's origin.

    The store identifier is the namespace in which table names are unique.
    It always has the shape ``<db_type>:<qualifier>`` so every ``table:`` /
    ``column:`` / ``index:`` ID is globally unique across multiple catalogs
    in the same graph.
    """
    dbt = schema.get("database_type", "unknown")
    meta = schema.get("metadata", {}) or {}
    name = _build_schema_name(schema)

    match dbt:
        case "bigquery":
            project = meta.get("project") or "unknown"
            return f"bigquery:{project}.{name}"
        case "firestore":
            project = meta.get("project") or "unknown"
            return f"firestore:{project}/{name}"
        case "bigtable":
            project = meta.get("project") or "unknown"
            instance = meta.get("instance") or name
            return f"bigtable:{project}/{instance}"
        case "postgresql" | "mysql" | "mongodb" | "documentdb":
            host = meta.get("host")
            return f"{dbt}:{host}/{name}" if host else f"{dbt}:{name}"
        case "opensearch":
            host = meta.get("host")
            return f"opensearch:{host}/{name}" if host else f"opensearch:{name}"
        case "sns" | "sqs" | "kinesis":
            # AWS resource names are region-scoped (a queue named "orders" in
            # us-east-1 is distinct from one in us-west-2). Qualify by region
            # so cross-region catalogs don't collide.
            region = meta.get("region")
            return f"{dbt}:{region}:{name}" if region else f"{dbt}:{name}"
        case _:
            return f"{dbt}:{name}"


async def _write_fan_out(
    schema: Dict[str, Any],
    graph_backend: Any,
    agent_id: Optional[str],
) -> None:
    """Emit Table/Column/Index nodes and HAS_COLUMN/INDEXED_BY/COVERS edges.

    Foreign keys are persisted last so every Column node they reference
    already exists.
    """
    store = _derive_store(schema)
    db_type = schema.get("database_type", "unknown")

    for table in schema.get("tables", []):
        await _emit_table(table, store, db_type, graph_backend, agent_id)
        await _reconcile_unresolved(table["name"], store, graph_backend)

    await _emit_foreign_keys(schema, store, graph_backend, agent_id)


async def _emit_table(
    table: Dict[str, Any],
    store: str,
    db_type: str,
    graph_backend: Any,
    agent_id: Optional[str],
) -> None:
    """Emit one Table node plus its Column and Index nodes + edges."""
    from daita.core.graph.models import (
        AgentGraphEdge,
        AgentGraphNode,
        EdgeType,
        NodeType,
    )

    tname = table["name"]
    table_id = AgentGraphNode.make_id(NodeType.TABLE, f"{store}.{tname}")

    table_props: Dict[str, Any] = {
        "database_type": db_type,
        "store": store,
    }
    if table.get("row_count") is not None:
        table_props["row_count"] = table["row_count"]
    if table.get("metadata"):
        table_props.update(table["metadata"])

    await graph_backend.add_node(
        AgentGraphNode(
            node_id=table_id,
            node_type=NodeType.TABLE,
            name=tname,
            created_by_agent=agent_id,
            properties=table_props,
        )
    )

    for col in table.get("columns", []):
        col_name = col["name"]
        col_id = AgentGraphNode.make_id(
            NodeType.COLUMN, f"{store}.{tname}.{col_name}"
        )
        col_props: Dict[str, Any] = {
            "type": col.get("type", ""),
            "nullable": col.get("nullable", True),
            "is_primary_key": col.get("is_primary_key", False),
            "table": tname,
            "store": store,
        }
        if col.get("column_comment"):
            col_props["comment"] = col["column_comment"]

        await graph_backend.add_node(
            AgentGraphNode(
                node_id=col_id,
                node_type=NodeType.COLUMN,
                name=col_name,
                created_by_agent=agent_id,
                properties=col_props,
            )
        )
        await graph_backend.add_edge(
            AgentGraphEdge(
                edge_id=AgentGraphEdge.make_id(
                    table_id, EdgeType.HAS_COLUMN, col_id
                ),
                from_node_id=table_id,
                to_node_id=col_id,
                edge_type=EdgeType.HAS_COLUMN,
                created_by_agent=agent_id,
            )
        )

    for idx in table.get("indexes", []):
        idx_name = idx.get("name") or ""
        if not idx_name:
            continue
        idx_id = AgentGraphNode.make_id(
            NodeType.INDEX, f"{store}.{tname}.{idx_name}"
        )
        idx_props: Dict[str, Any] = {
            "type": idx.get("type", ""),
            "unique": idx.get("unique", False),
            "table": tname,
            "store": store,
        }
        if idx.get("metadata"):
            idx_props.update(idx["metadata"])

        await graph_backend.add_node(
            AgentGraphNode(
                node_id=idx_id,
                node_type=NodeType.INDEX,
                name=idx_name,
                created_by_agent=agent_id,
                properties=idx_props,
            )
        )
        await graph_backend.add_edge(
            AgentGraphEdge(
                edge_id=AgentGraphEdge.make_id(
                    table_id, EdgeType.INDEXED_BY, idx_id
                ),
                from_node_id=table_id,
                to_node_id=idx_id,
                edge_type=EdgeType.INDEXED_BY,
                created_by_agent=agent_id,
            )
        )

        for position, col_name in enumerate(idx.get("columns", [])):
            if not col_name:
                continue
            col_id = AgentGraphNode.make_id(
                NodeType.COLUMN, f"{store}.{tname}.{col_name}"
            )
            await graph_backend.add_edge(
                AgentGraphEdge(
                    edge_id=AgentGraphEdge.make_id(
                        idx_id, EdgeType.COVERS, col_id
                    ),
                    from_node_id=idx_id,
                    to_node_id=col_id,
                    edge_type=EdgeType.COVERS,
                    created_by_agent=agent_id,
                    properties={"position": position},
                )
            )


async def _emit_foreign_keys(
    schema: Dict[str, Any],
    store: str,
    graph_backend: Any,
    agent_id: Optional[str],
) -> None:
    """Emit ``:REFERENCES`` edges between Column nodes for declared FKs.

    Cross-store FKs are out of scope for this PR; every FK is assumed to be
    intra-store (same ``store`` qualifier on both endpoints).
    """
    from daita.core.graph.models import AgentGraphEdge, AgentGraphNode, EdgeType, NodeType

    for fk in schema.get("foreign_keys", []):
        try:
            src_table = fk["source_table"]
            src_col = fk["source_column"]
            tgt_table = fk["target_table"]
            tgt_col = fk["target_column"]
        except KeyError:
            continue

        src_id = AgentGraphNode.make_id(
            NodeType.COLUMN, f"{store}.{src_table}.{src_col}"
        )
        tgt_id = AgentGraphNode.make_id(
            NodeType.COLUMN, f"{store}.{tgt_table}.{tgt_col}"
        )
        await graph_backend.add_edge(
            AgentGraphEdge(
                edge_id=AgentGraphEdge.make_id(src_id, EdgeType.REFERENCES, tgt_id),
                from_node_id=src_id,
                to_node_id=tgt_id,
                edge_type=EdgeType.REFERENCES,
                created_by_agent=agent_id,
            )
        )


async def _reconcile_unresolved(
    table_name: str,
    store: str,
    graph_backend: Any,
) -> None:
    """Promote any ``table:__unresolved__.<name>`` placeholder into the
    newly-emitted canonical ``table:<store>.<name>`` node.

    Cheap: one `get_node` lookup per emitted table. Runs inside the existing
    flush boundary.
    """
    from daita.core.graph.models import AgentGraphNode, NodeType
    from daita.core.graph.resolution import UNRESOLVED_STORE

    placeholder_id = AgentGraphNode.make_id(
        NodeType.TABLE, f"{UNRESOLVED_STORE}.{table_name}"
    )
    placeholder = await graph_backend.get_node(placeholder_id)
    if placeholder is None:
        return

    canonical_id = AgentGraphNode.make_id(NodeType.TABLE, f"{store}.{table_name}")
    await graph_backend.promote_node(placeholder_id, canonical_id)
    logger.debug(
        "Reconciled unresolved table %r into canonical node %s",
        table_name,
        canonical_id,
    )


async def _write_single_node(
    schema: Dict[str, Any],
    graph_backend: Any,
    agent_id: Optional[str],
    node_type_name: str,
) -> None:
    """Write the whole schema as a single parent node."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node_type = NodeType[node_type_name]
    name = _build_schema_name(schema)
    properties: Dict[str, Any] = {
        "database_type": schema.get("database_type", "unknown"),
        "tables": schema.get("tables", []),
        "metadata": schema.get("metadata", {}),
    }

    # OpenAPI carries service-level fields that don't fit under "metadata".
    if schema.get("database_type") == "openapi" or schema.get("api_type") == "openapi":
        properties.update(
            base_url=schema.get("base_url", ""),
            version=schema.get("version", ""),
            endpoint_count=schema.get("endpoint_count", 0),
        )

    node = AgentGraphNode(
        node_id=AgentGraphNode.make_id(node_type, name),
        node_type=node_type,
        name=name,
        created_by_agent=agent_id,
        properties=properties,
    )
    await graph_backend.add_node(node)


async def prune_stale_catalog(max_age_seconds: int) -> dict:
    """
    Remove catalog entries whose last_seen is older than max_age_seconds.

    Call at the end of a full discovery run to evict schemas for databases
    or services that are no longer reachable or in use.

    Entries with no last_seen (written before this feature) are left untouched.

    Returns {"removed": [list of removed keys]}
    """
    from datetime import datetime, timezone
    from pathlib import Path

    catalog_path = Path(".daita") / "catalog.json"
    if not catalog_path.exists():
        return {"removed": []}

    try:
        with open(catalog_path, "r") as f:
            existing = json.load(f)
    except json.JSONDecodeError:
        return {"removed": []}

    cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
    removed = []

    for key in list(existing.keys()):
        last_seen_raw = existing[key].get("last_seen")
        if last_seen_raw is None:
            continue
        try:
            ts = datetime.fromisoformat(str(last_seen_raw).replace("Z", "+00:00"))
            if ts.timestamp() < cutoff:
                removed.append(key)
                del existing[key]
        except (ValueError, TypeError):
            continue

    if removed:
        with open(catalog_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f"Catalog prune: removed {len(removed)} entries: {removed}")

    return {"removed": removed}
