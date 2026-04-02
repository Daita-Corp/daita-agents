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

    TABLE nodes are written with column metadata stored in properties so
    LineagePlugin flows can reference the same node_ids (e.g. "table:users").
    MongoDB collections are treated as TABLE nodes. OpenAPI services are
    written as API nodes.
    """
    from daita.core.graph.models import AgentGraphNode, NodeType

    db_type = schema.get("database_type") or schema.get("api_type", "unknown")
    schema_name = schema.get("database_name") or schema.get("schema") or "default"

    # --- Relational databases (postgres, mysql) ---
    if db_type in ("postgresql", "mysql"):
        for table in schema.get("tables", []):
            tname = table["name"]
            node = AgentGraphNode(
                node_id=AgentGraphNode.make_id(NodeType.TABLE, tname),
                node_type=NodeType.TABLE,
                name=tname,
                created_by_agent=agent_id,
                properties={
                    "database_type": db_type,
                    "schema": schema_name,
                    "row_count": table.get("row_count"),
                    "columns": table.get("columns", []),
                },
            )
            await graph_backend.add_node(node)

    # --- MongoDB ---
    elif db_type == "mongodb":
        for table in schema.get("tables", []):
            tname = table["name"]
            node = AgentGraphNode(
                node_id=AgentGraphNode.make_id(NodeType.TABLE, tname),
                node_type=NodeType.TABLE,
                name=tname,
                created_by_agent=agent_id,
                properties={
                    "database_type": "mongodb",
                    "database": schema_name,
                    "row_count": table.get("row_count"),
                    "columns": table.get("columns", []),
                },
            )
            await graph_backend.add_node(node)

    # --- OpenAPI services ---
    elif db_type == "openapi":
        svc_name = schema.get("service_name", "unknown_api")
        node = AgentGraphNode(
            node_id=AgentGraphNode.make_id(NodeType.API, svc_name),
            node_type=NodeType.API,
            name=svc_name,
            created_by_agent=agent_id,
            properties={
                "base_url": schema.get("base_url", ""),
                "version": schema.get("version", ""),
                "endpoint_count": schema.get("endpoint_count", 0),
            },
        )
        await graph_backend.add_node(node)

    # --- DynamoDB ---
    elif db_type == "dynamodb":
        table_name = schema.get("database_name") or schema.get("table_name", "unknown")
        node = AgentGraphNode(
            node_id=AgentGraphNode.make_id(NodeType.DATABASE, table_name),
            node_type=NodeType.DATABASE,
            name=table_name,
            created_by_agent=agent_id,
            properties={
                "database_type": "dynamodb",
                "table_count": schema.get("table_count", 1),
                "tables": schema.get("tables", []),
            },
        )
        await graph_backend.add_node(node)

    # --- API Gateway ---
    elif db_type == "apigateway":
        api_name = schema.get("database_name") or "unknown"
        node = AgentGraphNode(
            node_id=AgentGraphNode.make_id(NodeType.API, api_name),
            node_type=NodeType.API,
            name=api_name,
            created_by_agent=agent_id,
            properties={
                "database_type": "apigateway",
                "tables": schema.get("tables", []),
                "metadata": schema.get("metadata", {}),
            },
        )
        await graph_backend.add_node(node)

    # --- S3 ---
    elif db_type == "s3":
        bucket_name = schema.get("database_name") or schema.get("bucket", "unknown")
        node = AgentGraphNode(
            node_id=AgentGraphNode.make_id(NodeType.BUCKET, bucket_name),
            node_type=NodeType.BUCKET,
            name=bucket_name,
            created_by_agent=agent_id,
            properties={
                "database_type": "s3",
                "tables": schema.get("tables", []),
                "metadata": schema.get("metadata", {}),
            },
        )
        await graph_backend.add_node(node)

    # Flush to disk — LocalGraphBackend uses lazy writes
    if hasattr(graph_backend, "flush"):
        await graph_backend.flush()

    logger.debug(
        f"CatalogPlugin: persisted {db_type}:{schema_name} entities to graph backend"
    )


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
