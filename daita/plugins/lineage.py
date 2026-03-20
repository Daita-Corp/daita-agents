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

logger = logging.getLogger(__name__)


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
        """
        self._storage = storage
        self._organization_id = organization_id
        self._graph_backend = backend  # None until initialize() resolves it
        self._agent_id: Optional[str] = None
        self._risk_thresholds = risk_thresholds or {"HIGH": 20, "MEDIUM": 5}

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

        Returns:
            List of AgentTool instances for lineage operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="trace_lineage",
                description="Trace the full lineage of a data entity (upstream sources and downstream consumers)",
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "ID of the entity to trace (e.g., 'table:users', 'api:orders')",
                        },
                        "direction": {
                            "type": "string",
                            "description": "Direction to trace: 'upstream', 'downstream', or 'both' (default: 'both')",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum traversal depth (default: 5)",
                        },
                    },
                    "required": ["entity_id"],
                },
                handler=self._tool_trace_lineage,
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                timeout_seconds=60,
            ),
            AgentTool(
                name="trace_upstream",
                description="Get all upstream data sources for an entity (where the data comes from)",
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "ID of the entity",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum traversal depth (default: 3)",
                        },
                    },
                    "required": ["entity_id"],
                },
                handler=self._tool_trace_upstream,
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                timeout_seconds=60,
            ),
            AgentTool(
                name="trace_downstream",
                description="Get all downstream consumers of an entity (where the data flows to)",
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "ID of the entity",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum traversal depth (default: 3)",
                        },
                    },
                    "required": ["entity_id"],
                },
                handler=self._tool_trace_downstream,
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                timeout_seconds=60,
            ),
            AgentTool(
                name="register_flow",
                description="Register a data flow between two entities (e.g., ETL job, API call, data sync)",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "description": "Source entity ID",
                        },
                        "target_id": {
                            "type": "string",
                            "description": "Target entity ID",
                        },
                        "flow_type": {
                            "type": "string",
                            "description": "Type of flow: 'FLOWS_TO', 'SYNCS_TO', 'TRIGGERS', 'TRANSFORMS' (default: 'FLOWS_TO')",
                        },
                        "transformation": {
                            "type": "string",
                            "description": "Optional description of transformation applied",
                        },
                        "schedule": {
                            "type": "string",
                            "description": "Optional schedule pattern (e.g., '0 * * * *' for hourly)",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional additional metadata",
                        },
                    },
                    "required": ["source_id", "target_id"],
                },
                handler=self._tool_register_flow,
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                timeout_seconds=30,
            ),
            AgentTool(
                name="register_pipeline",
                description="Register a data pipeline as a sequence of processing steps",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Pipeline name"},
                        "steps": {
                            "type": "array",
                            "description": "List of pipeline steps with source, target, and transformation",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "target_id": {"type": "string"},
                                    "transformation": {"type": "string"},
                                },
                            },
                        },
                        "schedule": {
                            "type": "string",
                            "description": "Optional pipeline schedule",
                        },
                    },
                    "required": ["name", "steps"],
                },
                handler=self._tool_register_pipeline,
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                timeout_seconds=30,
            ),
            AgentTool(
                name="analyze_impact",
                description="Analyze the impact of a change to an entity (which downstream systems would be affected)",
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity that will change",
                        },
                        "change_type": {
                            "type": "string",
                            "description": "Type of change: 'schema_change', 'deprecation', 'deletion', 'data_quality' (default: 'schema_change')",
                        },
                    },
                    "required": ["entity_id"],
                },
                handler=self._tool_analyze_impact,
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                timeout_seconds=60,
            ),
            AgentTool(
                name="export_lineage",
                description="Export lineage diagram in Mermaid or DOT format for visualization",
                parameters={
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Root entity for lineage diagram",
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format: 'mermaid' or 'dot' (default: 'mermaid')",
                        },
                        "direction": {
                            "type": "string",
                            "description": "Direction: 'upstream', 'downstream', or 'both' (default: 'both')",
                        },
                    },
                    "required": ["entity_id"],
                },
                handler=self._tool_export_lineage,
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                timeout_seconds=30,
            ),
            AgentTool(
                name="prune_stale_lineage",
                description=(
                    "Remove stale nodes and edges from the lineage graph — entries that were not "
                    "refreshed during the current scan run. Call this at the end of a full pipeline "
                    "scan to evict ghost entries (tables or flows that no longer exist in the source "
                    "system). max_age_hours controls how old an entry must be before it is removed."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "max_age_hours": {
                            "type": "number",
                            "description": (
                                "Entries not seen within this many hours are removed. "
                                "Default: 48. Set to 0 to remove anything not touched in this run."
                            ),
                        }
                    },
                    "required": [],
                },
                handler=self._tool_prune_stale,
                category="lineage",
                source="plugin",
                plugin_name="Lineage",
                timeout_seconds=30,
            ),
        ]

    async def _tool_prune_stale(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for prune_stale_lineage"""
        max_age_hours = args.get("max_age_hours", 48)
        max_age_seconds = int(max_age_hours * 3600)

        from ..core.exceptions import PluginError

        if self._graph_backend is None:
            raise PluginError("No graph backend available")

        summary = await self._graph_backend.prune_stale(max_age_seconds)
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

        result = await self.trace_lineage(entity_id, direction, max_depth)
        return result

    async def trace_lineage(
        self, entity_id: str, direction: str = "both", max_depth: int = 5
    ) -> Dict[str, Any]:
        """
        Trace the full lineage of a data entity.

        Uses the graph backend (persisted across runs) when available,
        otherwise falls back to the in-memory flow list.

        Args:
            entity_id: ID of the entity to trace
            direction: Direction to trace ('upstream', 'downstream', 'both')
            max_depth: Maximum traversal depth

        Returns:
            Lineage graph with sources and destinations
        """
        if self._graph_backend:
            from daita.core.graph.algorithms import traverse

            graph = await self._graph_backend.load_graph()
            result = traverse(
                graph, entity_id, direction=direction, max_depth=max_depth
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

    async def _tool_trace_upstream(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for trace_upstream"""
        entity_id = args.get("entity_id")
        max_depth = args.get("max_depth", 3)

        result = await self.trace_lineage(entity_id, "upstream", max_depth)
        return {
            "entity_id": entity_id,
            "upstream_sources": result["lineage"]["upstream"],
            "count": len(result["lineage"]["upstream"]),
        }

    async def _tool_trace_downstream(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for trace_downstream"""
        entity_id = args.get("entity_id")
        max_depth = args.get("max_depth", 3)

        result = await self.trace_lineage(entity_id, "downstream", max_depth)
        return {
            "entity_id": entity_id,
            "downstream_consumers": result["lineage"]["downstream"],
            "count": len(result["lineage"]["downstream"]),
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

    async def _tool_register_pipeline(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for register_pipeline"""
        name = args.get("name")
        steps = args.get("steps")
        schedule = args.get("schedule")

        result = await self.register_pipeline(name, steps, schedule)
        return result

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

        result = await self.analyze_impact(entity_id, change_type)
        return result

    async def analyze_impact(
        self, entity_id: str, change_type: str = "schema_change"
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
            from daita.core.graph.algorithms import impact_analysis

            graph = await self._graph_backend.load_graph()
            result = impact_analysis(graph, entity_id)

            # Nodes with path_length == 1 are directly connected
            directly_affected = sum(
                1 for n in result["affected_nodes"] if n["path_length"] == 1
            )

            return {
                                "entity_id": entity_id,
                "change_type": change_type,
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

        result = await self.export_lineage(entity_id, format, direction)
        return result

    async def export_lineage(
        self, entity_id: str, format: str = "mermaid", direction: str = "both"
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
        lineage_result = await self.trace_lineage(entity_id, direction, max_depth=5)
        lineage = lineage_result["lineage"]

        if format == "mermaid":
            lines = ["graph LR"]

            # Add root node
            lines.append(
                f"    {self._sanitize_id(entity_id)}[{self._mermaid_label(entity_id)}]"
            )

            # Add upstream connections
            for item in lineage["upstream"]:
                source = self._sanitize_id(item["entity_id"])
                target = self._sanitize_id(entity_id)
                flow_type = item.get("flow_type", "FLOWS_TO")
                lines.append(
                    f"    {source}[{self._mermaid_label(item['entity_id'])}] -->|{flow_type}| {target}"
                )

            # Add downstream connections
            for item in lineage["downstream"]:
                source = self._sanitize_id(entity_id)
                target = self._sanitize_id(item["entity_id"])
                flow_type = item.get("flow_type", "FLOWS_TO")
                lines.append(
                    f"    {source} -->|{flow_type}| {target}[{self._mermaid_label(item['entity_id'])}]"
                )

            diagram = "\n".join(lines)

            return {"format": "mermaid", "diagram": diagram}

        elif format == "dot":
            lines = ["digraph lineage {"]
            lines.append("    rankdir=LR;")

            # Add nodes
            lines.append(f'    "{entity_id}" [shape=box];')

            for item in lineage["upstream"]:
                lines.append(f'    "{item["entity_id"]}" [shape=box];')
                lines.append(f'    "{item["entity_id"]}" -> "{entity_id}";')

            for item in lineage["downstream"]:
                lines.append(f'    "{item["entity_id"]}" [shape=box];')
                lines.append(f'    "{entity_id}" -> "{item["entity_id"]}";')

            lines.append("}")
            diagram = "\n".join(lines)

            return {"format": "dot", "diagram": diagram}

        else:
            from ..core.exceptions import ValidationError

            raise ValidationError(f"Unsupported format: {format}. Use 'mermaid' or 'dot'")

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
    ) -> Dict[str, Any]:
        """
        Automatically capture lineage from SQL query execution.

        Args:
            sql: SQL query that was executed
            context_table: Optional table being written to
            transformation: Optional description of the operation

        Returns:
            Result with registered flows
        """
        # Parse SQL to extract dependencies
        parsed = self.parse_sql_lineage(sql, context_table)

        flows_registered = []

        # Create flows for each source -> target combination
        for target in parsed["target_tables"]:
            for source in parsed["source_tables"]:
                flow_result = await self.register_flow(
                    source_id=f"table:{source}",
                    target_id=f"table:{target}",
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
    ) -> Callable:
        """
        Decorator for automatic function-based lineage tracking.

        Usage:
            @lineage.track(source="raw_data", target="processed_data")
            async def transform_data(input_df):
                # Function logic
                return output_df

        Args:
            source: Source entity ID or name
            target: Target entity ID or name
            transformation: Description of transformation

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

                await self.register_flow(
                    source_id=actual_source,
                    target_id=actual_target,
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
