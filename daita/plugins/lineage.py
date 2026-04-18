"""
LineagePlugin for data lineage and flow tracking.

Provides tools for tracking data flows, analyzing lineage, and understanding
data dependencies across systems.

Features:
- Automatic SQL parsing to extract table dependencies
- Decorator pattern for Python function lineage tracking
- Manual flow registration for custom integrations
- Database plugin integration for automatic query lineage capture

Usage with Database Plugins:
    ```python
    from daita.plugins import postgresql, lineage

    # Create lineage tracker
    lineage_plugin = lineage()

    # Create database plugin with lineage tracking
    db = postgresql(host="localhost", database="mydb")

    # Execute query with automatic lineage capture
    async with db:
        results = await db.query("INSERT INTO orders SELECT * FROM raw_orders")

        # Manually capture lineage
        await lineage_plugin.capture_sql_lineage(
            "INSERT INTO orders SELECT * FROM raw_orders",
            context_table="orders"
        )

    # Decorator pattern for Python functions
    @lineage_plugin.track(source="table:raw_data", target="table:processed_data")
    async def transform_data(df):
        # Process data
        return processed_df
    ```
"""

import logging
import functools
import re
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timezone

from .base import BasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool
    from ..core.graph.models import EdgeType
    from ..core.graph.resolution import AmbiguousReferencePolicy

logger = logging.getLogger(__name__)


def _parse_edge_types_arg(raw: Optional[List[str]]) -> Optional[List[Any]]:
    """Parse the ``edge_types`` kwarg coming from an agent tool call.

    Returns None when ``raw`` is empty so callers can default to
    LINEAGE_EDGE_TYPES rather than treating the argument as "match nothing".
    """
    if not raw:
        return None
    from ..core.exceptions import ValidationError
    from ..core.graph.models import EdgeType

    out: List[Any] = []
    for value in raw:
        try:
            out.append(EdgeType(value))
        except ValueError:
            raise ValidationError(
                f"Unknown edge_type '{value}'. Valid values: "
                f"{sorted(e.value for e in EdgeType)}"
            )
    return out


class LineagePlugin(BasePlugin):
    """
    Plugin for data lineage tracking and analysis.

    Works standalone (in-memory tracking) or with optional storage backends
    (graph database, relational database) for persistent lineage.

    Supports:
    - Flow registration and tracking
    - Upstream/downstream lineage tracing
    - Impact analysis
    - Pipeline definition
    - Lineage visualization export
    """

    def __init__(
        self,
        storage: Optional[Any] = None,
        organization_id: Optional[int] = None,
        backend: Optional[Any] = None,
        risk_thresholds: Optional[Dict[str, int]] = None,
        ambiguity_policy: Optional["AmbiguousReferencePolicy"] = None,
    ):
        """
        Initialize LineagePlugin.

        Args:
            storage: Optional legacy storage backend (GraphStore, BaseDatabasePlugin, etc.)
            organization_id: Optional organization ID for multi-tenant storage
            backend: Optional graph backend. If None, auto_select_backend() is called
                     during initialize() to pick LocalGraphBackend or DynamoGraphBackend.
            risk_thresholds: Optional dict with "HIGH" and "MEDIUM" integer thresholds
                             used by the in-memory analyze_impact() fallback.
                             Defaults to {"HIGH": 20, "MEDIUM": 5}.
            ambiguity_policy: How to resolve a bare table name that matches
                              multiple stores. Defaults to STRICT — raises on
                              ambiguity so cross-store collisions fail loudly.
                              Set to LENIENT to pick the most-recently-touched
                              candidate with a warning, or UNRESOLVED_SENTINEL
                              to route every reference through a placeholder
                              that reconciles on the next catalog run.
        """
        from daita.core.graph.resolution import AmbiguousReferencePolicy

        self._storage = storage
        self._organization_id = organization_id
        self._graph_backend = backend  # None until initialize() resolves it
        self._agent_id: Optional[str] = None
        self._risk_thresholds = risk_thresholds or {"HIGH": 20, "MEDIUM": 5}
        self._ambiguity_policy = ambiguity_policy or AmbiguousReferencePolicy.STRICT

        # In-memory storage for standalone mode
        self._flows = []  # List of registered flows
        self._pipelines = {}  # Dict of pipeline definitions

        logger.debug(f"LineagePlugin initialized (storage: {storage is not None})")

    def initialize(self, agent_id: str) -> None:
        self._agent_id = agent_id
        if self._graph_backend is None:
            from daita.core.graph.backend import auto_select_backend

            self._graph_backend = auto_select_backend(graph_type="lineage")
            logger.debug(
                f"LineagePlugin: using graph backend {type(self._graph_backend).__name__}"
            )

    def get_tools(self) -> List["AgentTool"]:
        """
        Expose lineage tracking operations as agent tools.

        5-tool surface covering every distinct lineage intent:
          * trace_lineage       — upstream / downstream / both, edge-type filterable
          * analyze_impact      — risk score for downstream breakage
          * find_lineage_paths  — every path between two entities
          * register_flow       — record a data movement
          * export_lineage      — render the lineage graph as a diagram

        register_pipeline and prune_stale_lineage remain on the Python API
        (``self.register_pipeline`` / ``self.prune_stale``) but are not
        exposed as tools — pipelines are declarative config and pruning is
        scheduled maintenance, neither is an agent-driven action.
        """
        from ..core.tools import AgentTool
        from ..core.graph.models import EdgeType

        edge_type_enum = sorted(e.value for e in EdgeType)

        def _lineage_tool(**kw) -> AgentTool:
            return AgentTool(
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                **kw,
            )

        return [
            _lineage_tool(
                name="trace_lineage",
                description=(
                    "Trace the lineage of a data entity upstream, downstream, "
                    "or both. Returns the reachable lineage graph scoped to "
                    "LINEAGE_EDGE_TYPES by default (READS, WRITES, TRANSFORMS, "
                    "SYNCS_TO, DERIVED_FROM, TRIGGERS, CALLS, PRODUCES). Pass "
                    "a custom edge_types list to broaden or narrow the scope."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "ID of the entity to trace (e.g. 'table:orders', 'api:checkout').",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["upstream", "downstream", "both"],
                            "description": "Traversal direction. Default 'both'.",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum traversal depth (default 5).",
                        },
                        "edge_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": edge_type_enum},
                            "description": (
                                "Optional override of the edge-type allowlist. "
                                "Omit for the default lineage set. Pass "
                                "['references'] to follow FK edges only."
                            ),
                        },
                    },
                    "required": ["entity_id"],
                },
                handler=self._tool_trace_lineage,
                timeout_seconds=60,
            ),
            _lineage_tool(
                name="analyze_impact",
                description=(
                    "Score the downstream impact of a change to an entity. "
                    "Returns affected entities ranked by cumulative "
                    "impact_weight along the path, plus a HIGH / MEDIUM / LOW "
                    "risk label. Traversal is bounded by max_depth and edge_types."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity that will change.",
                        },
                        "change_type": {
                            "type": "string",
                            "description": "schema_change | deprecation | deletion | data_quality (default schema_change).",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum impact radius in hops (default 5).",
                        },
                        "edge_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": edge_type_enum},
                            "description": "Optional edge-type allowlist. Default: lineage edge types.",
                        },
                    },
                    "required": ["entity_id"],
                },
                handler=self._tool_analyze_impact,
                timeout_seconds=60,
            ),
            _lineage_tool(
                name="find_lineage_paths",
                description=(
                    "Return every simple path from one entity to another "
                    "(capped by cutoff). Use to answer 'how does data get "
                    "from A to B?' with an explicit list of intermediate hops, "
                    "rather than just a risk score."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "from_entity": {
                            "type": "string",
                            "description": "Source entity ID.",
                        },
                        "to_entity": {
                            "type": "string",
                            "description": "Target entity ID.",
                        },
                        "cutoff": {
                            "type": "integer",
                            "description": "Maximum path length in edges (default 5). Prevents combinatorial explosion on dense graphs.",
                        },
                        "edge_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": edge_type_enum},
                            "description": "Optional edge-type allowlist. Default: lineage edge types.",
                        },
                    },
                    "required": ["from_entity", "to_entity"],
                },
                handler=self._tool_find_lineage_paths,
                timeout_seconds=60,
            ),
            _lineage_tool(
                name="register_flow",
                description=(
                    "Record a data flow from a source entity to a target "
                    "entity. Use for manual lineage capture (ETL jobs, API "
                    "calls, syncs) that isn't covered by automatic SQL / "
                    "function tracking."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "description": "Source entity ID.",
                        },
                        "target_id": {
                            "type": "string",
                            "description": "Target entity ID.",
                        },
                        "flow_type": {
                            "type": "string",
                            "description": "Edge semantic: reads | writes | transforms | syncs_to | derived_from | triggers | calls | produces (default transforms).",
                        },
                        "transformation": {
                            "type": "string",
                            "description": "Optional description of the transformation applied.",
                        },
                        "schedule": {
                            "type": "string",
                            "description": "Optional schedule pattern (e.g. '0 * * * *' for hourly).",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional additional metadata.",
                        },
                    },
                    "required": ["source_id", "target_id"],
                },
                handler=self._tool_register_flow,
                timeout_seconds=30,
            ),
            _lineage_tool(
                name="export_lineage",
                description="Render the lineage graph around an entity as a Mermaid or DOT diagram.",
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Root entity for the diagram.",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["mermaid", "dot"],
                            "description": "Output format (default 'mermaid').",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["upstream", "downstream", "both"],
                            "description": "Direction to include (default 'both').",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum traversal depth (default 5).",
                        },
                    },
                    "required": ["entity_id"],
                },
                handler=self._tool_export_lineage,
                timeout_seconds=30,
            ),
        ]

    async def prune_stale(self, max_age_hours: float = 48) -> Dict[str, Any]:
        """
        Remove stale nodes and edges from the lineage graph.

        Python API only — not exposed as an agent tool. Call this at the end
        of a full pipeline scan to evict entries whose ``last_seen`` wasn't
        refreshed. Agents don't invoke pruning; schedulers do.
        """
        from ..core.exceptions import PluginError

        if self._graph_backend is None:
            raise PluginError("No graph backend available")

        max_age_seconds = int(max_age_hours * 3600)
        summary = await self._graph_backend.prune_stale(max_age_seconds)
        if hasattr(self._graph_backend, "flush"):
            await self._graph_backend.flush()
        return {
            "removed_node_count": len(summary["removed_nodes"]),
            "removed_edge_count": len(summary["removed_edges"]),
            "max_age_hours": max_age_hours,
        }

    async def _tool_trace_lineage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for trace_lineage"""
        entity_id = args.get("entity_id")
        direction = args.get("direction", "both")
        max_depth = args.get("max_depth", 5)
        edge_types = _parse_edge_types_arg(args.get("edge_types"))

        return await self.trace_lineage(
            entity_id,
            direction=direction,
            max_depth=max_depth,
            edge_types=edge_types,
        )

    async def trace_lineage(
        self,
        entity_id: str,
        direction: str = "both",
        max_depth: int = 5,
        edge_types: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Trace the lineage of a data entity.

        Uses the graph backend's ``subgraph`` method (bounded BFS) rather than
        ``load_graph`` + in-memory BFS — only the nodes reachable within
        ``max_depth`` are fetched.

        Args:
            entity_id: ID of the entity to trace
            direction: 'upstream' | 'downstream' | 'both'
            max_depth: Maximum traversal depth
            edge_types: Optional iterable of EdgeType values. Defaults to
                ``LINEAGE_EDGE_TYPES`` so structural edges (HAS_COLUMN,
                INDEXED_BY, REFERENCES, COVERS, PART_OF) are excluded.

        Returns:
            Lineage graph with sources and destinations
        """
        if self._graph_backend:
            from daita.core.graph.algorithms import (
                LINEAGE_EDGE_TYPES,
                traverse,
            )

            effective_edge_types = edge_types or LINEAGE_EDGE_TYPES
            graph = await self._graph_backend.subgraph(
                root=entity_id,
                direction=direction,
                edge_types=effective_edge_types,
                max_depth=max_depth,
            )
            result = traverse(
                graph,
                entity_id,
                direction=direction,
                max_depth=max_depth,
                edge_types=effective_edge_types,
            )
            if direction == "both":
                upstream = result.get("upstream", [])
                downstream = result.get("downstream", [])
            elif direction == "upstream":
                upstream = result
                downstream = []
            else:
                upstream = []
                downstream = result
            lineage = {
                "entity_id": entity_id,
                "upstream": upstream,
                "downstream": downstream,
            }
            return {
                "lineage": lineage,
                "upstream_count": len(upstream),
                "downstream_count": len(downstream),
            }

        # In-memory fallback (no backend initialized)
        lineage = {"entity_id": entity_id, "upstream": [], "downstream": []}

        if direction in ["upstream", "both"]:
            lineage["upstream"] = await self._trace_direction(
                entity_id, "upstream", max_depth
            )

        if direction in ["downstream", "both"]:
            lineage["downstream"] = await self._trace_direction(
                entity_id, "downstream", max_depth
            )

        return {
            "lineage": lineage,
            "upstream_count": len(lineage["upstream"]),
            "downstream_count": len(lineage["downstream"]),
        }

    async def _trace_direction(
        self,
        entity_id: str,
        direction: str,
        max_depth: int,
        current_depth: int = 0,
        visited: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recursively trace lineage in a direction.

        Args:
            entity_id: Current entity
            direction: 'upstream' or 'downstream'
            max_depth: Maximum depth
            current_depth: Current recursion depth
            visited: Set of visited entities to avoid cycles

        Returns:
            List of connected entities
        """
        if visited is None:
            visited = set()

        if current_depth >= max_depth or entity_id in visited:
            return []

        visited.add(entity_id)
        results = []

        # Find connected flows
        for flow in self._flows:
            if direction == "upstream" and flow["target_id"] == entity_id:
                results.append(
                    {
                        "entity_id": flow["source_id"],
                        "flow_type": flow["flow_type"],
                        "transformation": flow.get("transformation"),
                        "depth": current_depth + 1,
                    }
                )
                # Recursively trace upstream
                nested = await self._trace_direction(
                    flow["source_id"], direction, max_depth, current_depth + 1, visited
                )
                results.extend(nested)

            elif direction == "downstream" and flow["source_id"] == entity_id:
                results.append(
                    {
                        "entity_id": flow["target_id"],
                        "flow_type": flow["flow_type"],
                        "transformation": flow.get("transformation"),
                        "depth": current_depth + 1,
                    }
                )
                # Recursively trace downstream
                nested = await self._trace_direction(
                    flow["target_id"], direction, max_depth, current_depth + 1, visited
                )
                results.extend(nested)

        return results

    async def _tool_find_lineage_paths(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for find_lineage_paths.

        Returns every simple path from ``from_entity`` to ``to_entity`` up to
        ``cutoff`` hops, restricted to ``edge_types`` (defaults to
        LINEAGE_EDGE_TYPES).
        """
        from_entity = args.get("from_entity")
        to_entity = args.get("to_entity")
        if not from_entity or not to_entity:
            from ..core.exceptions import ValidationError

            raise ValidationError("from_entity and to_entity are required")

        cutoff = int(args.get("cutoff") or 5)
        edge_types = _parse_edge_types_arg(args.get("edge_types"))

        if self._graph_backend is None:
            from ..core.exceptions import PluginError

            raise PluginError("No graph backend available")

        from daita.core.graph.algorithms import (
            LINEAGE_EDGE_TYPES,
            find_paths,
        )

        effective_edge_types = edge_types or LINEAGE_EDGE_TYPES
        graph = await self._graph_backend.subgraph(
            root=from_entity,
            direction="downstream",
            edge_types=effective_edge_types,
            max_depth=cutoff,
        )
        paths = find_paths(
            graph,
            from_entity,
            to_entity,
            edge_types=effective_edge_types,
            cutoff=cutoff,
        )
        return {
            "from_entity": from_entity,
            "to_entity": to_entity,
            "cutoff": cutoff,
            "paths": paths,
            "path_count": len(paths),
            "reachable": bool(paths),
        }

    async def _tool_register_flow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for register_flow"""
        source_id = args.get("source_id")
        target_id = args.get("target_id")
        flow_type = args.get("flow_type", "FLOWS_TO")
        transformation = args.get("transformation")
        schedule = args.get("schedule")
        metadata = args.get("metadata", {})

        result = await self.register_flow(
            source_id, target_id, flow_type, transformation, schedule, metadata
        )
        return result

    async def _resolve_table_ref(self, value: str, store: Optional[str] = None) -> str:
        """Resolve a bare table name (or ``table:<name>``) to a qualified node ID.

        Non-table entity IDs (``api:...``, ``column:...``, ``pipeline:...``,
        etc.) pass through unchanged. Honours ``self._ambiguity_policy`` for
        multi-store matches; unknown tables fall through to an
        ``__unresolved__`` placeholder that reconciles on the next catalog run.
        """
        from daita.core.graph.resolution import resolve_or_placeholder

        return await resolve_or_placeholder(
            self._graph_backend,
            value,
            store=store,
            agent_id=self._agent_id,
            policy=self._ambiguity_policy,
        )

    async def register_flow(
        self,
        source_id: str,
        target_id: str,
        flow_type: str = "FLOWS_TO",
        transformation: Optional[str] = None,
        schedule: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a data flow between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            flow_type: Type of flow
            transformation: Description of transformation
            schedule: Schedule pattern
            metadata: Additional metadata

        Returns:
            Created flow ID
        """
        registered_at = datetime.now(timezone.utc).isoformat()
        flow_id = f"flow:{source_id}:{target_id}:{registered_at}"

        flow = {
            "flow_id": flow_id,
            "source_id": source_id,
            "target_id": target_id,
            "flow_type": flow_type,
            "transformation": transformation,
            "schedule": schedule,
            "metadata": metadata or {},
            "registered_at": registered_at,
        }

        # Only keep the in-memory list when there is no backend. When a backend
        # is active it owns persistence and trace_lineage uses it exclusively —
        # appending to _flows would grow memory unboundedly with no benefit.
        if not self._graph_backend and not self._storage:
            self._flows.append(flow)

        if self._graph_backend or self._storage:
            await self._persist_flow(flow)

        return {
            "flow_id": flow_id,
            "source_id": source_id,
            "target_id": target_id,
        }

    async def register_pipeline(
        self, name: str, steps: List[Dict[str, str]], schedule: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a data pipeline as a sequence of steps.

        Args:
            name: Pipeline name
            steps: List of pipeline steps
            schedule: Optional schedule

        Returns:
            Pipeline registration result
        """
        pipeline = {
            "name": name,
            "steps": steps,
            "schedule": schedule,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

        self._pipelines[name] = pipeline

        # Register each step as a flow
        for i, step in enumerate(steps):
            await self.register_flow(
                source_id=step["source_id"],
                target_id=step["target_id"],
                flow_type="PIPELINE_STEP",
                transformation=step.get("transformation"),
                metadata={"pipeline": name, "step_index": i},
            )

        return {"pipeline_name": name, "steps_registered": len(steps)}

    async def _tool_analyze_impact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for analyze_impact"""
        entity_id = args.get("entity_id")
        change_type = args.get("change_type", "schema_change")
        max_depth = int(args.get("max_depth") or 5)
        edge_types = _parse_edge_types_arg(args.get("edge_types"))

        return await self.analyze_impact(
            entity_id,
            change_type=change_type,
            max_depth=max_depth,
            edge_types=edge_types,
        )

    async def analyze_impact(
        self,
        entity_id: str,
        change_type: str = "schema_change",
        max_depth: int = 5,
        edge_types: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze impact of a change to an entity.

        When a graph backend is available, delegates to algorithms.impact_analysis()
        which uses cumulative edge impact_weight for risk scoring. This is the
        authoritative path — it honours the impact_weight values set on edges.

        Falls back to a raw downstream count when no backend is initialized
        (standalone / in-memory mode).

        Args:
            entity_id: Entity that will change
            change_type: Type of change

        Returns:
            Impact analysis with affected entities
        """
        if self._graph_backend:
            from daita.core.graph.algorithms import (
                LINEAGE_EDGE_TYPES,
                impact_analysis,
            )

            effective_edge_types = edge_types or LINEAGE_EDGE_TYPES
            graph = await self._graph_backend.subgraph(
                root=entity_id,
                direction="downstream",
                edge_types=effective_edge_types,
                max_depth=max_depth,
            )
            result = impact_analysis(graph, entity_id, edge_types=effective_edge_types)

            # Nodes with path_length == 1 are directly connected
            directly_affected = sum(
                1 for n in result["affected_nodes"] if n["path_length"] == 1
            )

            return {
                "entity_id": entity_id,
                "change_type": change_type,
                "max_depth": max_depth,
                "directly_affected_count": directly_affected,
                "total_affected_count": result["affected_count"],
                "affected_entities": result["affected_nodes"],
                "risk_level": result["risk_level"],
                "recommendation": self._get_impact_recommendation(
                    result["risk_level"], change_type
                ),
            }

        # In-memory fallback — no graph backend available
        downstream_result = await self.trace_lineage(
            entity_id, "downstream", max_depth=10
        )
        downstream = downstream_result["lineage"]["downstream"]

        direct_impact = len([d for d in downstream if d.get("depth") == 1])
        total_impact = len(downstream)

        high_thresh = self._risk_thresholds.get("HIGH", 20)
        medium_thresh = self._risk_thresholds.get("MEDIUM", 5)
        if total_impact > high_thresh:
            risk_level = "HIGH"
        elif total_impact > medium_thresh:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "entity_id": entity_id,
            "change_type": change_type,
            "directly_affected_count": direct_impact,
            "total_affected_count": total_impact,
            "affected_entities": downstream,
            "risk_level": risk_level,
            "recommendation": self._get_impact_recommendation(risk_level, change_type),
        }

    def _get_impact_recommendation(self, risk_level: str, change_type: str) -> str:
        """Get recommendation based on risk level"""
        if risk_level == "HIGH":
            return f"High-risk {change_type}. Review all affected systems, create migration plan, and notify stakeholders."
        elif risk_level == "MEDIUM":
            return f"Medium-risk {change_type}. Test downstream dependencies and coordinate deployment."
        else:
            return f"Low-risk {change_type}. Proceed with standard testing procedures."

    async def _tool_export_lineage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for export_lineage"""
        entity_id = args.get("entity_id")
        format = args.get("format", "mermaid")
        direction = args.get("direction", "both")
        max_depth = int(args.get("max_depth") or 5)

        return await self.export_lineage(
            entity_id, format=format, direction=direction, max_depth=max_depth
        )

    async def export_lineage(
        self,
        entity_id: str,
        format: str = "mermaid",
        direction: str = "both",
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """
        Export lineage diagram for visualization.

        Args:
            entity_id: Root entity
            format: Output format ('mermaid' or 'dot')
            direction: Direction to include

        Returns:
            Diagram in requested format
        """
        # Get lineage
        lineage_result = await self.trace_lineage(
            entity_id, direction, max_depth=max_depth
        )
        lineage = lineage_result["lineage"]

        if format == "mermaid":
            lines = ["graph LR"]

            # Add root node
            lines.append(
                f"    {self._sanitize_id(entity_id)}[{self._mermaid_label(entity_id)}]"
            )

            # Add upstream connections
            for item in lineage["upstream"]:
                item_id = item.get("node_id") or item.get("entity_id", "unknown")
                source = self._sanitize_id(item_id)
                target = self._sanitize_id(entity_id)
                flow_type = item.get("flow_type", "FLOWS_TO")
                lines.append(
                    f"    {source}[{self._mermaid_label(item_id)}] -->|{flow_type}| {target}"
                )

            # Add downstream connections
            for item in lineage["downstream"]:
                item_id = item.get("node_id") or item.get("entity_id", "unknown")
                source = self._sanitize_id(entity_id)
                target = self._sanitize_id(item_id)
                flow_type = item.get("flow_type", "FLOWS_TO")
                lines.append(
                    f"    {source} -->|{flow_type}| {target}[{self._mermaid_label(item_id)}]"
                )

            diagram = "\n".join(lines)

            return {"format": "mermaid", "diagram": diagram}

        elif format == "dot":
            lines = ["digraph lineage {"]
            lines.append("    rankdir=LR;")

            # Add nodes
            lines.append(f'    "{entity_id}" [shape=box];')

            for item in lineage["upstream"]:
                item_id = item.get("node_id") or item.get("entity_id", "unknown")
                lines.append(f'    "{item_id}" [shape=box];')
                lines.append(f'    "{item_id}" -> "{entity_id}";')

            for item in lineage["downstream"]:
                item_id = item.get("node_id") or item.get("entity_id", "unknown")
                lines.append(f'    "{item_id}" [shape=box];')
                lines.append(f'    "{entity_id}" -> "{item_id}";')

            lines.append("}")
            diagram = "\n".join(lines)

            return {"format": "dot", "diagram": diagram}

        else:
            from ..core.exceptions import ValidationError

            raise ValidationError(
                f"Unsupported format: {format}. Use 'mermaid' or 'dot'"
            )

    def _sanitize_id(self, entity_id: str) -> str:
        """Sanitize entity ID for use as a Mermaid node identifier."""
        return entity_id.replace(":", "_").replace(" ", "_").replace("-", "_")

    @staticmethod
    def _mermaid_label(entity_id: str) -> str:
        """Return entity_id as a quoted Mermaid label, escaping inner double-quotes."""
        return '"' + entity_id.replace('"', "'") + '"'

    @staticmethod
    def _resolve_edge_type(flow_type: Optional[str]) -> "EdgeType":
        """
        Normalise a free-text flow_type string into a valid EdgeType.

        Tries (in order):
          1. Exact match against EdgeType values  (e.g. "transforms" -> TRANSFORMS)
          2. Synonym map for common LLM variants  (e.g. "flows_to" -> TRANSFORMS)
          3. Falls back to TRANSFORMS
        """
        from daita.core.graph.models import EdgeType

        if not flow_type:
            return EdgeType.TRANSFORMS

        normalised = flow_type.strip().lower().replace("-", "_").replace(" ", "_")

        # 1. Exact match against enum values
        try:
            return EdgeType(normalised)
        except ValueError:
            pass

        # 2. Synonym map
        _SYNONYMS: Dict[str, "EdgeType"] = {
            "transform": EdgeType.TRANSFORMS,
            "aggregate": EdgeType.TRANSFORMS,
            "aggregation": EdgeType.TRANSFORMS,
            "enrich": EdgeType.TRANSFORMS,
            "enrichment": EdgeType.TRANSFORMS,
            "flow": EdgeType.TRANSFORMS,
            "flows_to": EdgeType.TRANSFORMS,
            "etl": EdgeType.TRANSFORMS,
            "read": EdgeType.READS,
            "select": EdgeType.READS,
            "write": EdgeType.WRITES,
            "insert": EdgeType.WRITES,
            "upsert": EdgeType.WRITES,
            "load": EdgeType.WRITES,
            "sync": EdgeType.SYNCS_TO,
            "replicate": EdgeType.SYNCS_TO,
            "replication": EdgeType.SYNCS_TO,
            "model": EdgeType.DERIVED_FROM,
            "derive": EdgeType.DERIVED_FROM,
            "derived": EdgeType.DERIVED_FROM,
            "trigger": EdgeType.TRIGGERS,
            "call": EdgeType.CALLS,
            "invoke": EdgeType.CALLS,
            "produce": EdgeType.PRODUCES,
            "emit": EdgeType.PRODUCES,
        }
        resolved = _SYNONYMS.get(normalised)
        if resolved:
            return resolved

        logger.warning(
            f"LineagePlugin: unknown flow_type '{flow_type}', defaulting to TRANSFORMS"
        )
        return EdgeType.TRANSFORMS

    async def _persist_flow(self, flow: Dict[str, Any]) -> None:
        """Persist flow to the graph backend."""
        if not self._graph_backend:
            return

        from daita.core.graph.models import AgentGraphNode, AgentGraphEdge, NodeType

        source_id = flow["source_id"]
        target_id = flow["target_id"]
        exec_id = flow.get("metadata", {}).get("execution_id")

        # Resolve edge type from the flow_type string — normalises LLM variants
        edge_type = self._resolve_edge_type(flow.get("flow_type"))

        # Infer node type from id prefix (e.g. "table:orders" -> TABLE)
        def _infer_node_type(node_id: str) -> NodeType:
            prefix = node_id.split(":")[0] if ":" in node_id else ""
            try:
                return NodeType(prefix)
            except ValueError:
                return NodeType.TABLE

        source_node = AgentGraphNode(
            node_id=source_id,
            node_type=_infer_node_type(source_id),
            name=source_id.split(":")[-1],
            created_by_agent=self._agent_id,
            created_at_execution=exec_id,
        )
        target_node = AgentGraphNode(
            node_id=target_id,
            node_type=_infer_node_type(target_id),
            name=target_id.split(":")[-1],
            created_by_agent=self._agent_id,
            created_at_execution=exec_id,
        )
        edge = AgentGraphEdge(
            edge_id=AgentGraphEdge.make_id(source_id, edge_type, target_id),
            from_node_id=source_id,
            to_node_id=target_id,
            edge_type=edge_type,
            created_by_agent=self._agent_id,
            execution_id=exec_id,
            properties={
                "flow_type": edge_type.value,
                "transformation": flow.get("transformation"),
            },
        )

        await self._graph_backend.add_node(source_node)
        await self._graph_backend.add_node(target_node)
        await self._graph_backend.add_edge(edge)
        # Flush once at the end of the logical unit (2 nodes + 1 edge) rather
        # than after each individual mutation — reduces JSON serializations from
        # 3 per flow to 1.
        if hasattr(self._graph_backend, "flush"):
            await self._graph_backend.flush()
        logger.debug(
            f"Persisted lineage flow {flow['flow_id']} "
            f"({source_id} --[{edge_type.value}]--> {target_id})"
        )

    def parse_sql_lineage(
        self, sql: str, context_table: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse SQL query to extract table dependencies automatically.

        Supports SELECT, INSERT, UPDATE, DELETE, CREATE TABLE AS, and WITH (CTE) queries.

        Args:
            sql: SQL query string
            context_table: Optional table being written to (for INSERT/UPDATE)

        Returns:
            Dictionary with source_tables and target_tables
        """
        # Use re.IGNORECASE on the original SQL so quoted case-sensitive identifiers
        # (e.g. FROM "MyTable") are preserved rather than uppercased.
        _RI = re.IGNORECASE

        # Extract source tables from SELECT/FROM/JOIN clauses
        source_tables = set()

        # Find all FROM clauses
        from_pattern = r"\bFROM\s+([a-zA-Z0-9_\.]+)"
        for match in re.finditer(from_pattern, sql, _RI):
            source_tables.add(match.group(1))

        # Find all JOIN clauses
        join_pattern = r"\bJOIN\s+([a-zA-Z0-9_\.]+)"
        for match in re.finditer(join_pattern, sql, _RI):
            source_tables.add(match.group(1))

        # Find tables in WITH (CTE) clauses
        with_pattern = r"\bWITH\s+([a-zA-Z0-9_]+)\s+AS"
        cte_tables = set()
        for match in re.finditer(with_pattern, sql, _RI):
            cte_tables.add(match.group(1))

        # Remove CTEs from source tables (they're temporary)
        source_tables = {
            t for t in source_tables if t.lower() not in {c.lower() for c in cte_tables}
        }

        # Extract target tables
        target_tables = set()

        # INSERT INTO pattern
        insert_pattern = r"\bINSERT\s+INTO\s+([a-zA-Z0-9_\.]+)"
        for match in re.finditer(insert_pattern, sql, _RI):
            target_tables.add(match.group(1))

        # UPDATE pattern
        update_pattern = r"\bUPDATE\s+([a-zA-Z0-9_\.]+)"
        for match in re.finditer(update_pattern, sql, _RI):
            target_tables.add(match.group(1))

        # CREATE TABLE AS pattern
        create_pattern = r"\bCREATE\s+(?:TEMP\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\.]+)\s+AS"
        for match in re.finditer(create_pattern, sql, _RI):
            target_tables.add(match.group(1))

        # If context_table provided and no target found, use it
        if context_table and not target_tables:
            target_tables.add(context_table)

        return {
            "source_tables": list(source_tables),
            "target_tables": list(target_tables),
            "is_read_only": len(target_tables) == 0,
            "cte_tables": list(cte_tables),
        }

    async def capture_sql_lineage(
        self,
        sql: str,
        context_table: Optional[str] = None,
        transformation: Optional[str] = None,
        default_store: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Automatically capture lineage from SQL query execution.

        Args:
            sql: SQL query that was executed
            context_table: Optional table being written to
            transformation: Optional description of the operation
            default_store: Optional store identifier used to qualify bare
                table names parsed out of SQL (e.g. ``postgres:db/schema``).
                When omitted, bare names go through the resolution layer:
                unambiguous matches resolve normally; unknown or ambiguous
                names follow the plugin's ``ambiguity_policy``.

        Returns:
            Result with registered flows
        """
        parsed = self.parse_sql_lineage(sql, context_table)
        flows_registered = []

        for target in parsed["target_tables"]:
            tgt_id = await self._resolve_table_ref(target, store=default_store)
            for source in parsed["source_tables"]:
                src_id = await self._resolve_table_ref(source, store=default_store)
                flow_result = await self.register_flow(
                    source_id=src_id,
                    target_id=tgt_id,
                    flow_type="SQL_TRANSFORM",
                    transformation=transformation or f"SQL: {sql[:100]}",
                    metadata={"sql_query": sql, "is_read_only": parsed["is_read_only"]},
                )
                flows_registered.append(flow_result["flow_id"])

        return {
            "flows_registered": flows_registered,
            "source_tables": parsed["source_tables"],
            "target_tables": parsed["target_tables"],
            "flow_count": len(flows_registered),
        }

    def track(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        transformation: Optional[str] = None,
        store: Optional[str] = None,
    ) -> Callable:
        """
        Decorator for automatic function-based lineage tracking.

        Usage:
            @lineage.track(source="raw_data", target="processed_data")
            async def transform_data(input_df):
                ...

            # Explicit disambiguation:
            @lineage.track(
                source="raw_data",
                target="processed_data",
                store="postgres:prod-host/warehouse",
            )

        Args:
            source: Source entity name or ``table:<name>`` / typed-prefix ID.
                Bare names are routed through the resolution layer.
            target: Target entity name (same rules as ``source``).
            transformation: Description of transformation
            store: Optional store qualifier applied to any bare table
                references in ``source`` / ``target``. Non-table entity IDs
                (``api:...``, ``column:...``) ignore ``store``.

        Returns:
            Decorated function with automatic lineage capture
        """

        def decorator(func: Callable) -> Callable:
            import asyncio

            if not asyncio.iscoroutinefunction(func):
                raise TypeError(
                    f"@lineage.track() requires an async function. "
                    f"{func.__name__!r} is synchronous. "
                    "Convert it to async or call register_flow() manually after execution."
                )

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                actual_source = source or "unknown"
                actual_target = target or func.__name__
                actual_transformation = transformation or f"Function: {func.__name__}"

                # Resolve at call time, not decoration time. Ambiguity
                # errors surface when the function actually runs so imports
                # stay clean.
                resolved_source = await self._resolve_table_ref(
                    actual_source, store=store
                )
                resolved_target = await self._resolve_table_ref(
                    actual_target, store=store
                )

                await self.register_flow(
                    source_id=resolved_source,
                    target_id=resolved_target,
                    flow_type="FUNCTION_TRANSFORM",
                    transformation=actual_transformation,
                    metadata={"function": func.__name__, "module": func.__module__},
                )

                return await func(*args, **kwargs)

            return async_wrapper

        return decorator


def lineage(**kwargs) -> LineagePlugin:
    """Create LineagePlugin with simplified interface."""
    return LineagePlugin(**kwargs)
