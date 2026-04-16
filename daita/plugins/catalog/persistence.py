"""
Schema persistence — local JSON and graph backend storage.

Handles writing discovered schemas to .daita/catalog.json (local default)
or a registered catalog backend, plus graph node creation for LineagePlugin.
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
    # them by node_id (e.g. "table:orders"). Runs in both local and cloud paths.
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
        table. Used for relational-shaped sources (Postgres/MySQL/BigQuery/
        MongoDB/Firestore/Bigtable) so individual tables become referenceable
        by ``table:<name>`` for LineagePlugin edges.
      * ``mode="single"`` writes one parent node for the whole schema.
        Used for bucket/API/stream/queue/topic-shaped sources where the store
        itself is the atomic unit.

    Unknown ``database_type`` values are silently skipped — the schema has
    already been persisted to local JSON or the catalog backend by the caller.
    """
    from daita.core.graph.models import NodeType

    db_type = schema.get("database_type") or schema.get("api_type", "unknown")
    schema_name = schema.get("database_name") or schema.get("schema") or "default"

    spec = _NODE_HANDLERS.get(db_type)
    if spec is not None:
        mode, node_type = spec
        if mode == "fan_out":
            await _write_tables_as_nodes(schema, graph_backend, agent_id, node_type)
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
    # Relational / document / warehouse sources — one node per table/collection.
    "postgresql": ("fan_out", "TABLE"),
    "mysql": ("fan_out", "TABLE"),
    "mongodb": ("fan_out", "TABLE"),
    "bigquery": ("fan_out", "TABLE"),
    "firestore": ("fan_out", "TABLE"),
    "bigtable": ("fan_out", "TABLE"),
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
    # Streaming / messaging — single node for the topic or subscription.
    "pubsub_topic": ("single", "SERVICE"),
    "pubsub_subscription": ("single", "SERVICE"),
}


def _build_schema_name(schema: Dict[str, Any]) -> str:
    """Resolve the human-facing name for a schema across varied source shapes."""
    for key in ("database_name", "schema", "bucket", "table_name", "service_name"):
        value = schema.get(key)
        if value:
            return value
    return "unknown"


async def _write_tables_as_nodes(
    schema: Dict[str, Any],
    graph_backend: Any,
    agent_id: Optional[str],
    node_type_name: str,
) -> None:
    """Fan schema['tables'] out into one node per table."""
    from daita.core.graph.models import AgentGraphNode, NodeType

    node_type = NodeType[node_type_name]
    db_type = schema.get("database_type", "unknown")
    parent_name = _build_schema_name(schema)

    for table in schema.get("tables", []):
        tname = table["name"]
        node = AgentGraphNode(
            node_id=AgentGraphNode.make_id(node_type, tname),
            node_type=node_type,
            name=tname,
            created_by_agent=agent_id,
            properties={
                "database_type": db_type,
                "parent": parent_name,
                "row_count": table.get("row_count"),
                "columns": table.get("columns", []),
            },
        )
        await graph_backend.add_node(node)


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
