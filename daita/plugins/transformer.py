"""
TransformerPlugin for managing and executing SQL transformations.

Operates ON data through an existing database plugin, storing transformation
definitions in a graph backend as TRANSFORMATION nodes. Falls back to an
in-memory store when no graph backend is available.

Features:
- Named transformation definitions (SQL + description + metadata)
- Execution via database plugin
- Dry-run validation with EXPLAIN
- Version history (snapshots of SQL stored inside node properties)
- SQL diff between versions (integer index for snapshots, "current" for live SQL)
- Optional lineage auto-capture on execution
- Execution history persisted on the node (last_run, run_count)

Usage:
    ```python
    from daita.plugins import postgresql, lineage, transformer

    db = postgresql(host="localhost", database="mydb")
    lin = lineage()
    tx = transformer(db=db, lineage=lin)

    await tx.transform_create("orders_summary",
        sql="INSERT INTO orders_summary SELECT customer_id, SUM(amount) FROM orders GROUP BY 1",
        description="Daily orders summary")
    await tx.transform_run("orders_summary")

    # As agent tools
    agent = Agent(name="transformer_agent", tools=[db, lin, tx])
    ```
"""

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from .base import BasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool
    from .base_db import BaseDatabasePlugin
    from .lineage import LineagePlugin

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_.]+$")
_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([A-Za-z0-9_\.]+)",
    re.IGNORECASE,
)
# Matches :param_name at a word boundary — avoids partial replacements (TX-04)
_PARAM_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)\b")


def _validate_identifier(name: str) -> str:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid identifier {name!r}. Only alphanumeric, underscore, and dot allowed."
        )
    return name


def _parse_version_ref(v: Any) -> Union[int, str]:
    """Parse a version reference: the string 'current' stays as-is, otherwise int."""
    if v == "current":
        return "current"
    return int(v)


def _local_parse_sql(sql: str) -> Dict[str, List[str]]:
    """Minimal regex SQL parser — used when no lineage plugin is available."""
    tables = set()
    for m in _TABLE_RE.finditer(sql):
        t = m.group(1).strip()
        if t.upper() not in {"SELECT", "WITH", "SET"}:
            tables.add(t)

    source_tables: List[str] = []
    target_tables: List[str] = []

    insert_m = re.search(r"\bINSERT\s+INTO\s+([A-Za-z0-9_\.]+)", sql, re.IGNORECASE)
    create_m = re.search(
        r"\bCREATE\s+(?:TEMP\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_\.]+)\s+AS",
        sql,
        re.IGNORECASE,
    )
    update_m = re.search(r"\bUPDATE\s+([A-Za-z0-9_\.]+)", sql, re.IGNORECASE)
    # DELETE FROM produces a WRITES edge, not a READS edge
    delete_m = re.search(r"\bDELETE\s+FROM\s+([A-Za-z0-9_\.]+)", sql, re.IGNORECASE)

    if insert_m:
        target_tables.append(insert_m.group(1))
    if create_m:
        target_tables.append(create_m.group(1))
    if update_m:
        target_tables.append(update_m.group(1))
    if delete_m:
        target_tables.append(delete_m.group(1))

    targets_set = set(target_tables)
    for t in tables:
        if t not in targets_set:
            source_tables.append(t)

    return {"source_tables": source_tables, "target_tables": target_tables}


@dataclass
class _InMemoryNode:
    """
    Lightweight stand-in for AgentGraphNode used when no graph backend is
    available. Exposes the same attributes that TransformerPlugin accesses.
    """

    name: str
    properties: Dict[str, Any]
    node_id: str = ""
    created_by_agent: Optional[str] = None
    tags: list = field(default_factory=list)


def _make_tx_entry(
    name: str, props: Dict[str, Any], node_id: str = ""
) -> Dict[str, Any]:
    """Build the dict returned by transform_list for a single transformation."""
    return {
        "name": name,
        "node_id": node_id or f"transformation:{name}",
        "description": props.get("description", ""),
        "source_tables": props.get("source_tables", []),
        "target_table": props.get("target_table"),
        "version_count": len(props.get("versions", [])),
        "created_at": props.get("created_at"),
        "updated_at": props.get("updated_at"),
        "last_run": props.get("last_run"),
        "run_count": props.get("run_count", 0),
    }


class TransformerPlugin(BasePlugin):
    """
    Plugin for managing, versioning, and executing named SQL transformations.

    Graph backend is the preferred storage. When none is available, falls back
    to an in-memory dict so the plugin remains fully functional within a session.
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        lineage: Optional[Any] = None,
        backend: Optional[Any] = None,
    ):
        """
        Args:
            db: Optional database plugin. Required at execution time.
            lineage: Optional LineagePlugin for auto lineage capture on transform_run.
            backend: Optional graph backend. Auto-selected during initialize() if None.
        """
        self._db = db
        self._lineage = lineage
        self._graph_backend = backend
        self._agent_id: Optional[str] = None
        # in-memory index, always populated by transform_create
        self._definitions: Dict[str, Dict[str, Any]] = {}

    def initialize(self, agent_id: str) -> None:
        self._agent_id = agent_id
        if self._graph_backend is None:
            from daita.core.graph.backend import auto_select_backend

            self._graph_backend = auto_select_backend(graph_type="transformer")
            logger.debug(
                "TransformerPlugin: using graph backend %s",
                type(self._graph_backend).__name__,
            )

    def _validate_db(self) -> Any:
        if self._db is None:
            raise ValueError(
                "No database plugin configured. Pass db=<plugin> to transformer()."
            )
        return self._db

    def get_tools(self) -> List["AgentTool"]:
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="transform_create",
                description=(
                    "Define a named SQL transformation and store it. "
                    "Source and target tables are extracted from the SQL automatically. "
                    "Persists to graph backend when available; falls back to in-memory."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique transformation name (alphanumeric, underscore, dot only)",
                        },
                        "sql": {
                            "type": "string",
                            "description": "SQL to execute for this transformation",
                        },
                        "description": {
                            "type": "string",
                            "description": "Human-readable description of what this transformation does",
                        },
                    },
                    "required": ["name", "sql"],
                },
                handler=self._tool_create,
                category="transformer",
                source="plugin",
                plugin_name="Transformer",
                timeout_seconds=30,
            ),
            AgentTool(
                name="transform_run",
                description=(
                    "Execute a named transformation against the database. "
                    "Automatically captures lineage if a lineage plugin is configured. "
                    "Updates last_run and run_count on the transformation node."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Transformation name to execute",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Optional key-value parameters to substitute into SQL (:param style)",
                        },
                    },
                    "required": ["name"],
                },
                handler=self._tool_run,
                category="transformer",
                source="plugin",
                plugin_name="Transformer",
                timeout_seconds=300,
            ),
            AgentTool(
                name="transform_test",
                description=(
                    "Dry-run validate a transformation SQL using EXPLAIN "
                    "without executing it. Substitutes dummy values for any :param "
                    "placeholders before sending to the DB. Returns the query plan or error."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Transformation name to test",
                        },
                    },
                    "required": ["name"],
                },
                handler=self._tool_test,
                category="transformer",
                source="plugin",
                plugin_name="Transformer",
                timeout_seconds=60,
            ),
            AgentTool(
                name="transform_version",
                description=(
                    "Snapshot the current SQL of a transformation as a new version. "
                    "Snapshots are indexed 0 (oldest) to N-1 (most recent). "
                    "The live, unsaved SQL is always accessible as 'current'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Transformation name to snapshot",
                        },
                    },
                    "required": ["name"],
                },
                handler=self._tool_version,
                category="transformer",
                source="plugin",
                plugin_name="Transformer",
                timeout_seconds=30,
            ),
            AgentTool(
                name="transform_diff",
                description=(
                    "Show a unified diff between two versions of a transformation SQL. "
                    "Use integer indices (0 = oldest snapshot, N-1 = most recent snapshot) "
                    "or the string 'current' to refer to the live unsaved SQL."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Transformation name",
                        },
                        "version_a": {
                            "type": ["integer", "string"],
                            "description": "Snapshot index (0-based) or 'current' for live SQL.",
                        },
                        "version_b": {
                            "type": ["integer", "string"],
                            "description": "Snapshot index (0-based) or 'current' for live SQL.",
                        },
                    },
                    "required": ["name", "version_a", "version_b"],
                },
                handler=self._tool_diff,
                category="transformer",
                source="plugin",
                plugin_name="Transformer",
                timeout_seconds=10,
            ),
            AgentTool(
                name="transform_list",
                description=(
                    "List all defined transformations with their metadata, "
                    "including last_run timestamp and run_count."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "filter_tag": {
                            "type": "string",
                            "description": "Optional tag to filter transformations by",
                        }
                    },
                },
                handler=self._tool_list,
                category="transformer",
                source="plugin",
                plugin_name="Transformer",
                timeout_seconds=30,
            ),
        ]

    # -------------------------------------------------------------------------
    # Tool handlers
    # -------------------------------------------------------------------------

    async def _tool_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = args["name"]
        sql = args["sql"]
        description = args.get("description", "")
        return await self.transform_create(name, sql, description=description)

    async def _tool_run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        db = self._validate_db()
        name = args["name"]
        parameters = args.get("parameters")
        return await self.transform_run(db, name, parameters=parameters)

    async def _tool_test(self, args: Dict[str, Any]) -> Dict[str, Any]:
        db = self._validate_db()
        name = args["name"]
        return await self.transform_test(db, name)

    async def _tool_version(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = args["name"]
        return await self.transform_version(name)

    async def _tool_diff(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = args["name"]
        version_a = _parse_version_ref(args["version_a"])
        version_b = _parse_version_ref(args["version_b"])
        return await self.transform_diff(name, version_a, version_b)

    async def _tool_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        filter_tag = args.get("filter_tag")
        return await self.transform_list(filter_tag=filter_tag)

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    async def transform_create(
        self,
        name: str,
        sql: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Define a named transformation and persist to graph (or in-memory fallback)."""
        _validate_identifier(name)

        # Parse source/target tables
        if self._lineage is not None:
            parsed = self._lineage.parse_sql_lineage(sql)
        else:
            parsed = _local_parse_sql(sql)

        source_tables = parsed.get("source_tables", [])
        target_tables = parsed.get("target_tables", [])

        # Preserve version history and created_at across updates
        existing_props = self._definitions.get(name, {})
        is_update = name in self._definitions

        properties: Dict[str, Any] = {
            "sql": sql,
            "description": description,
            "source_tables": source_tables,
            "target_table": target_tables[0] if target_tables else None,
            "versions": existing_props.get("versions", []),
            "created_at": existing_props.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # always update in-memory index so transform_list never needs load_graph()
        self._definitions[name] = properties
        node_id = f"transformation:{name}"

        if self._graph_backend:
            from daita.core.graph.models import (
                AgentGraphNode,
                AgentGraphEdge,
                NodeType,
                EdgeType,
            )

            node_id = AgentGraphNode.make_id(NodeType.TRANSFORMATION, name)

            # Prefer existing created_at from graph over in-memory for first-run scenarios
            existing_graph = await self._graph_backend.get_node(node_id)
            if existing_graph:
                properties["versions"] = existing_graph.properties.get(
                    "versions", properties["versions"]
                )
                properties["created_at"] = existing_graph.properties.get(
                    "created_at", properties["created_at"]
                )
                is_update = True

            node = AgentGraphNode(
                node_id=node_id,
                node_type=NodeType.TRANSFORMATION,
                name=name,
                created_by_agent=self._agent_id,
                properties=properties,
            )
            await self._graph_backend.add_node(node)

            for src in source_tables:
                src_node_id = AgentGraphNode.make_id(NodeType.TABLE, src)
                await self._graph_backend.add_node(
                    AgentGraphNode(
                        node_id=src_node_id,
                        node_type=NodeType.TABLE,
                        name=src,
                        created_by_agent=self._agent_id,
                    )
                )
                await self._graph_backend.add_edge(
                    AgentGraphEdge(
                        edge_id=AgentGraphEdge.make_id(
                            src_node_id, EdgeType.READS, node_id
                        ),
                        from_node_id=src_node_id,
                        to_node_id=node_id,
                        edge_type=EdgeType.READS,
                        created_by_agent=self._agent_id,
                    )
                )

            for tgt in target_tables:
                tgt_node_id = AgentGraphNode.make_id(NodeType.TABLE, tgt)
                await self._graph_backend.add_node(
                    AgentGraphNode(
                        node_id=tgt_node_id,
                        node_type=NodeType.TABLE,
                        name=tgt,
                        created_by_agent=self._agent_id,
                    )
                )
                await self._graph_backend.add_edge(
                    AgentGraphEdge(
                        edge_id=AgentGraphEdge.make_id(
                            node_id, EdgeType.WRITES, tgt_node_id
                        ),
                        from_node_id=node_id,
                        to_node_id=tgt_node_id,
                        edge_type=EdgeType.WRITES,
                        created_by_agent=self._agent_id,
                    )
                )

        return {
            "success": True,
            "name": name,
            "node_id": node_id,
            "source_tables": source_tables,
            "target_tables": target_tables,
            "is_update": is_update,
        }

    async def transform_run(
        self,
        db: Any,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a named transformation. Captures lineage if lineage plugin is set."""
        node = await self._get_node(name)
        if node is None:
            return {"success": False, "error": f"Transformation '{name}' not found"}

        sql = node.properties["sql"]
        target_table = node.properties.get("target_table")

        # Strings are single-quoted and escaped; numerics/booleans are inlined as-is.
        if parameters:
            for key, val in parameters.items():
                if isinstance(val, str):
                    replacement = "'" + val.replace("'", "''") + "'"
                elif val is None:
                    replacement = "NULL"
                elif isinstance(val, bool):
                    replacement = "1" if val else "0"
                else:
                    replacement = str(val)
                sql = re.sub(r":" + re.escape(key) + r"\b", replacement, sql)

        started_at = datetime.now(timezone.utc)

        if hasattr(db, "execute"):
            result = await db.execute(sql)
        else:
            result = await db.query(sql)

        ended_at = datetime.now(timezone.utc)
        duration_ms = int((ended_at - started_at).total_seconds() * 1000)

        # persist execution metadata to node
        try:
            updated_props = dict(node.properties)
            updated_props["last_run"] = {
                "executed_at": ended_at.isoformat(),
                "success": True,
                "duration_ms": duration_ms,
            }
            updated_props["run_count"] = updated_props.get("run_count", 0) + 1
            self._definitions[name] = updated_props

            if self._graph_backend:
                from daita.core.graph.models import AgentGraphNode, NodeType

                await self._graph_backend.add_node(
                    AgentGraphNode(
                        node_id=node.node_id,
                        node_type=NodeType.TRANSFORMATION,
                        name=node.name,
                        created_by_agent=node.created_by_agent,
                        properties=updated_props,
                    )
                )
        except Exception as exc:
            logger.warning("TransformerPlugin: failed to update run metadata: %s", exc)

        # Auto-capture lineage
        if self._lineage is not None:
            try:
                await self._lineage.capture_sql_lineage(
                    sql, context_table=target_table, transformation=name
                )
            except Exception as exc:
                logger.warning("TransformerPlugin: lineage capture failed: %s", exc)

        return {
            "success": True,
            "name": name,
            "target_table": target_table,
            "duration_ms": duration_ms,
            "rows_affected": result if isinstance(result, int) else None,
        }

    async def transform_test(
        self,
        db: Any,
        name: str,
    ) -> Dict[str, Any]:
        """Dry-run validate SQL using EXPLAIN with dummy parameter substitution."""
        node = await self._get_node(name)
        if node is None:
            return {"success": False, "error": f"Transformation '{name}' not found"}

        sql = node.properties["sql"]

        # substitute dummy values so EXPLAIN sees a syntactically valid query
        test_sql = _PARAM_RE.sub("'__test__'", sql)

        try:
            rows = await db.query(f"EXPLAIN {test_sql}")
            plan_lines = []
            for row in rows:
                if isinstance(row, dict):
                    plan_lines.append(str(list(row.values())[0]))
                else:
                    plan_lines.append(str(row[0]))

            return {
                "success": True,
                "name": name,
                "valid": True,
                "plan": "\n".join(plan_lines),
            }
        except Exception as exc:
            return {
                "success": True,
                "name": name,
                "valid": False,
                "error": str(exc),
            }

    async def transform_version(self, name: str) -> Dict[str, Any]:
        """Snapshot the current SQL as a new version entry."""
        node = await self._get_node(name)
        if node is None:
            return {"success": False, "error": f"Transformation '{name}' not found"}

        current_sql = node.properties["sql"]
        versions = list(node.properties.get("versions", []))
        snapshot = {
            "sql": current_sql,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        versions.append(snapshot)

        updated_props = dict(node.properties)
        updated_props["versions"] = versions
        updated_props["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Always update in-memory index
        self._definitions[name] = updated_props

        # Persist to graph if available (TX-02: only reached when backend exists)
        if self._graph_backend:
            from daita.core.graph.models import AgentGraphNode, NodeType

            await self._graph_backend.add_node(
                AgentGraphNode(
                    node_id=node.node_id,
                    node_type=NodeType.TRANSFORMATION,
                    name=node.name,
                    created_by_agent=node.created_by_agent,
                    properties=updated_props,
                )
            )

        return {
            "success": True,
            "name": name,
            "version_index": len(versions) - 1,
            "total_versions": len(versions),
            "snapshot": snapshot,
        }

    async def transform_diff(
        self,
        name: str,
        version_a: Union[int, str],
        version_b: Union[int, str],
    ) -> Dict[str, Any]:
        """
        Return unified diff between two version references.

        Version references:
        - Integer 0..N-1: snapshot index (0 = oldest)
        - String "current": the live, unsaved SQL
        """
        node = await self._get_node(name)
        if node is None:
            return {"success": False, "error": f"Transformation '{name}' not found"}

        versions = node.properties.get("versions", [])
        current_sql = node.properties["sql"]

        def _get_sql(ref: Union[int, str]) -> Optional[str]:
            # "current" for live SQL, int for snapshots
            if ref == "current":
                return current_sql
            if isinstance(ref, int) and 0 <= ref < len(versions):
                return versions[ref]["sql"]
            return None

        sql_a = _get_sql(version_a)
        sql_b = _get_sql(version_b)

        if sql_a is None:
            return {
                "success": False,
                "error": f"Version index {version_a} out of range",
            }
        if sql_b is None:
            return {
                "success": False,
                "error": f"Version index {version_b} out of range",
            }

        diff_lines = list(
            difflib.unified_diff(
                sql_a.splitlines(keepends=True),
                sql_b.splitlines(keepends=True),
                fromfile=f"{name} v{version_a}",
                tofile=f"{name} v{version_b}",
            )
        )
        diff_text = "".join(diff_lines)

        return {
            "success": True,
            "name": name,
            "version_a": version_a,
            "version_b": version_b,
            "diff": diff_text if diff_text else "(no differences)",
            "changed": bool(diff_text),
        }

    async def transform_list(
        self,
        filter_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all known transformations with their metadata."""
        if not self._definitions and not self._graph_backend:
            return {"success": False, "error": "No graph backend available"}

        transformations = []

        if self._definitions:
            # use in-memory index — avoids full graph load
            for tx_name, props in self._definitions.items():
                if filter_tag and filter_tag not in props.get("tags", []):
                    continue
                transformations.append(_make_tx_entry(tx_name, props))
        else:
            # Cold-start: _definitions empty, fall back to graph
            graph = await self._graph_backend.load_graph()
            from daita.core.graph.models import NodeType

            for node_id, node_data in graph.nodes(data=True):
                node = node_data.get("data")
                if node is None or node.node_type != NodeType.TRANSFORMATION:
                    continue
                self._definitions[node.name] = node.properties  # warm cache
                if filter_tag and filter_tag not in node.tags:
                    continue
                transformations.append(
                    _make_tx_entry(node.name, node.properties, node_id=node.node_id)
                )

        return {
            "success": True,
            "count": len(transformations),
            "transformations": sorted(transformations, key=lambda t: t["name"]),
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    async def _get_node(self, name: str) -> Optional[Any]:
        """
        Retrieve the transformation node for the given name.

        Checks graph backend first; falls back to in-memory definitions.
        Returns None if not found in either.
        """
        if self._graph_backend:
            from daita.core.graph.models import AgentGraphNode, NodeType

            node_id = AgentGraphNode.make_id(NodeType.TRANSFORMATION, name)
            return await self._graph_backend.get_node(node_id)

        if name in self._definitions:
            return _InMemoryNode(
                name=name,
                node_id=f"transformation:{name}",
                properties=self._definitions[name],
                created_by_agent=self._agent_id,
            )
        return None


def transformer(
    db: Optional[Any] = None,
    lineage: Optional[Any] = None,
    backend: Optional[Any] = None,
) -> TransformerPlugin:
    """
    Create a TransformerPlugin.

    Args:
        db: Database plugin for executing transformations.
        lineage: Optional LineagePlugin for auto lineage capture.
        backend: Optional graph backend for persisting transformation definitions.

    Returns:
        TransformerPlugin instance
    """
    return TransformerPlugin(db=db, lineage=lineage, backend=backend)
