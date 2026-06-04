"""
Neo4jPlugin for graph database operations with Cypher queries.

Provides tools for working with Neo4j graph databases, including node/relationship
creation, Cypher query execution, and graph pattern analysis.

Features:
- Native Cypher query support
- Node and relationship CRUD operations
- Graph pattern matching
- Path finding and traversal
- Community detection and graph analytics
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
    ToolView,
)

from .base import ConnectorPlugin
from .manifest import PluginKind, PluginManifest
from ..core.exceptions import PluginError

if TYPE_CHECKING:
    from ..core.tools import LocalTool

logger = logging.getLogger(__name__)


_NEO4J_OPERATION_DEFINITIONS = (
    {
        "tool_name": "neo4j_query",
        "capability_id": "neo4j.query.execute",
        "operation_type": "neo4j.query.execute",
        "description": "Execute a Cypher query against a Neo4j graph database.",
        "access": AccessMode.READ,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_query",
        "write": False,
    },
    {
        "tool_name": "neo4j_find_nodes",
        "capability_id": "neo4j.node.find",
        "operation_type": "neo4j.node.find",
        "description": "Find Neo4j nodes by label and properties.",
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_find_nodes",
        "write": False,
    },
    {
        "tool_name": "neo4j_find_path",
        "capability_id": "neo4j.path.find",
        "operation_type": "neo4j.path.find",
        "description": "Find the shortest path between two Neo4j nodes.",
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_find_path",
        "write": False,
    },
    {
        "tool_name": "neo4j_get_neighbors",
        "capability_id": "neo4j.neighbor.list",
        "operation_type": "neo4j.neighbor.list",
        "description": "List neighboring nodes connected to a Neo4j node.",
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_get_neighbors",
        "write": False,
    },
    {
        "tool_name": "neo4j_schema",
        "capability_id": "neo4j.schema.read",
        "operation_type": "neo4j.schema.read",
        "description": "Read Neo4j labels and relationship types.",
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_schema",
        "write": False,
    },
    {
        "tool_name": "neo4j_list_labels",
        "capability_id": "neo4j.labels.list",
        "operation_type": "neo4j.labels.list",
        "description": "List Neo4j node labels.",
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_list_labels",
        "write": False,
    },
    {
        "tool_name": "neo4j_list_relationship_types",
        "capability_id": "neo4j.relationship_types.list",
        "operation_type": "neo4j.relationship_types.list",
        "description": "List Neo4j relationship types.",
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_list_relationship_types",
        "write": False,
    },
    {
        "tool_name": "neo4j_graph_stats",
        "capability_id": "neo4j.stats.read",
        "operation_type": "neo4j.stats.read",
        "description": "Read Neo4j node and relationship counts.",
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_graph_stats",
        "write": False,
    },
    {
        "tool_name": "neo4j_create_node",
        "capability_id": "neo4j.node.create",
        "operation_type": "neo4j.node.create",
        "description": "Create a Neo4j node.",
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "handler_name": "_tool_create_node",
        "write": True,
    },
    {
        "tool_name": "neo4j_create_relationship",
        "capability_id": "neo4j.relationship.create",
        "operation_type": "neo4j.relationship.create",
        "description": "Create a Neo4j relationship.",
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "handler_name": "_tool_create_relationship",
        "write": True,
    },
    {
        "tool_name": "neo4j_delete_node",
        "capability_id": "neo4j.node.delete",
        "operation_type": "neo4j.node.delete",
        "description": "Delete a Neo4j node and its relationships.",
        "access": AccessMode.WRITE,
        "risk": RiskLevel.HIGH,
        "retry_safe": False,
        "idempotent": True,
        "side_effecting": True,
        "handler_name": "_tool_delete_node",
        "write": True,
    },
)


class _Neo4jExecutor:
    """Execute Neo4j runtime capabilities and return typed evidence."""

    id = "neo4j.operations"

    def __init__(self, plugin: "Neo4jPlugin") -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"]
            for definition in self._plugin._operation_definitions()
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        tool_view_name = (
            context.get("tool_view", {}).get("name")
            if isinstance(context, dict)
            else None
        ) or definition["tool_name"]
        return [
            Evidence(
                kind="neo4j.operation.result",
                owner=self._plugin.manifest.id,
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "operation": definition["tool_name"],
                    "request": dict(task.input or {}),
                    "result": result,
                },
                metadata={
                    "capability_id": task.capability_id,
                    "tool_view": tool_view_name,
                },
            )
        ]


class Neo4jPlugin(ConnectorPlugin):
    """
    Plugin for Neo4j graph database operations.

    Supports Cypher queries, node/relationship operations, and graph analytics.
    Works with Neo4j instances (local, cloud, or Aura).

    Example:
        ```python
        from daita.plugins import neo4j

        async with neo4j(uri="bolt://localhost:7687", auth=("neo4j", "password")) as graph:
            # Create nodes
            await graph.create_node("Person", {"name": "Alice", "age": 30})

            # Run Cypher query
            result = await graph.query("MATCH (n:Person) RETURN n LIMIT 10")

            # Create relationships
            await graph.create_relationship(
                "Person", {"name": "Alice"},
                "Person", {"name": "Bob"},
                "KNOWS", {"since": "2020"}
            )

            # Find paths
            paths = await graph.find_path("Person", {"name": "Alice"}, "Person", {"name": "Bob"})
        ```
    """

    manifest = PluginManifest(
        id="neo4j",
        display_name="Neo4j",
        version="2.0.0",
        kind=PluginKind.CONNECTOR,
        domains=frozenset({"neo4j", "graph", "database"}),
        provides=frozenset({"graph_query", "graph_mutation", "graph_metadata"}),
        optional_dependencies=frozenset({"neo4j"}),
    )

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: Optional[tuple] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        read_only: bool = False,
        **kwargs,
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j URI (bolt:// or neo4j://)
            auth: Tuple of (username, password)
            username: Username (alternative to auth tuple)
            password: Password (alternative to auth tuple)
            database: Database name (default: "neo4j")
            read_only: If True, only expose read tools (default: False)
            **kwargs: Additional neo4j driver parameters
        """
        self._uri = uri
        self._database = database
        self.read_only = read_only

        # Handle auth parameter
        if auth:
            self._auth = auth
        elif username and password:
            self._auth = (username, password)
        else:
            self._auth = ("neo4j", "password")  # Default

        self._driver = None
        self._driver_config = kwargs
        self._executor = _Neo4jExecutor(self)

        logger.debug(f"Neo4jPlugin initialized for {uri}, database={database}")

    @property
    def is_connected(self) -> bool:
        """Whether the Neo4j driver has been initialized."""
        return self._driver is not None

    async def teardown(self) -> None:
        """Release runtime-owned Neo4j resources."""
        await self.disconnect()

    def _operation_definitions(self) -> tuple[dict[str, Any], ...]:
        """Return operation definitions enabled for this plugin instance."""
        return tuple(
            definition
            for definition in _NEO4J_OPERATION_DEFINITIONS
            if not self.read_only or not definition["write"]
        )

    def _definition_for_capability(self, capability_id: str) -> dict[str, Any]:
        for definition in self._operation_definitions():
            if definition["capability_id"] == capability_id:
                return definition
        raise KeyError(capability_id)

    def _definition_for_tool(self, tool_name: str) -> dict[str, Any]:
        for definition in self._operation_definitions():
            if definition["tool_name"] == tool_name:
                return definition
        raise KeyError(tool_name)

    def declare_capabilities(self) -> tuple[Capability, ...]:
        """Declare Neo4j operations as runtime-plannable capabilities."""
        return tuple(
            Capability(
                id=definition["capability_id"],
                owner=self.manifest.id,
                description=definition["description"],
                domains=frozenset({"neo4j", "graph", "database"}),
                operation_types=frozenset({definition["operation_type"]}),
                access=definition["access"],
                risk=definition["risk"],
                input_schema=self._tool_parameters(definition["tool_name"]),
                output_evidence=frozenset({"neo4j.operation.result"}),
                executor=self._executor.id,
                model_visible=True,
                retry_safe=definition["retry_safe"],
                replay_safe=definition["retry_safe"],
                idempotent=definition["idempotent"],
                side_effecting=definition["side_effecting"],
                timeout_seconds=60,
                metadata={"tool_name": definition["tool_name"]},
            )
            for definition in self._operation_definitions()
        )

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        """Declare typed evidence returned by Neo4j capability execution."""
        return (
            EvidenceSchema(
                kind="neo4j.operation.result",
                owner=self.manifest.id,
                description="Neo4j operation result evidence.",
                json_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "request": {"type": "object"},
                        "result": {"type": "object"},
                    },
                    "required": ["operation", "request", "result"],
                },
            ),
        )

    def get_executors(self) -> tuple[_Neo4jExecutor, ...]:
        """Return the Neo4j runtime executor."""
        return (self._executor,)

    def get_tool_views(self) -> tuple[ToolView, ...]:
        """Return model-visible views over Neo4j capabilities."""
        return tuple(
            ToolView(
                name=definition["tool_name"],
                capability_id=definition["capability_id"],
                description=definition["description"],
                parameters=self._tool_parameters(definition["tool_name"]),
            )
            for definition in self._operation_definitions()
        )

    def _tool_parameters(self, tool_name: str) -> Dict[str, Any]:
        """Return the JSON schema for an existing Neo4j tool view."""
        empty = {"type": "object", "properties": {}, "required": []}
        object_props = {
            "type": "object",
            "description": "Properties to match as key-value pairs",
        }
        schemas: dict[str, dict[str, Any]] = {
            "neo4j_query": {
                "type": "object",
                "properties": {
                    "cypher": {
                        "type": "string",
                        "description": "Cypher query to execute.",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Optional query parameters as key-value pairs.",
                    },
                },
                "required": ["cypher"],
            },
            "neo4j_find_nodes": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Node label to search for.",
                    },
                    "properties": object_props,
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of nodes to return.",
                    },
                },
                "required": ["label"],
            },
            "neo4j_find_path": {
                "type": "object",
                "properties": {
                    "from_label": {
                        "type": "string",
                        "description": "Start node label.",
                    },
                    "from_properties": object_props,
                    "to_label": {"type": "string", "description": "End node label."},
                    "to_properties": object_props,
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum path length.",
                    },
                },
                "required": [
                    "from_label",
                    "from_properties",
                    "to_label",
                    "to_properties",
                ],
            },
            "neo4j_get_neighbors": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Node label."},
                    "properties": object_props,
                    "relationship_type": {
                        "type": "string",
                        "description": "Optional relationship type filter.",
                    },
                    "direction": {
                        "type": "string",
                        "description": "outgoing, incoming, or both.",
                    },
                },
                "required": ["label", "properties"],
            },
            "neo4j_schema": empty,
            "neo4j_list_labels": empty,
            "neo4j_list_relationship_types": empty,
            "neo4j_graph_stats": empty,
            "neo4j_create_node": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Node label."},
                    "properties": object_props,
                },
                "required": ["label", "properties"],
            },
            "neo4j_create_relationship": {
                "type": "object",
                "properties": {
                    "from_label": {
                        "type": "string",
                        "description": "Source node label.",
                    },
                    "from_properties": object_props,
                    "to_label": {"type": "string", "description": "Target node label."},
                    "to_properties": object_props,
                    "relationship_type": {
                        "type": "string",
                        "description": "Relationship type.",
                    },
                    "relationship_properties": {
                        "type": "object",
                        "description": "Optional relationship properties.",
                    },
                },
                "required": [
                    "from_label",
                    "from_properties",
                    "to_label",
                    "to_properties",
                    "relationship_type",
                ],
            },
            "neo4j_delete_node": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Node label."},
                    "properties": object_props,
                },
                "required": ["label", "properties"],
            },
        }
        try:
            return schemas[tool_name]
        except KeyError as exc:
            raise KeyError(tool_name) from exc

    async def connect(self):
        """Connect to Neo4j database."""
        if self._driver is not None:
            return  # Already connected

        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=self._auth, **self._driver_config
            )

            # Verify connectivity
            await self._driver.verify_connectivity()

            logger.info(f"Connected to Neo4j at {self._uri}")
        except ImportError:
            raise ImportError(
                "neo4j driver not installed. Install with: pip install 'daita-agents[neo4j]'"
            )
        except PluginError:
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise PluginError(f"Failed to connect to Neo4j: {e}", plugin_name="Neo4j")

    async def disconnect(self):
        """Disconnect from Neo4j database."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def _tool_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for neo4j_query"""
        import re

        cypher = args.get("cypher", "").rstrip()
        parameters = args.get("parameters", {})

        # Inject LIMIT 200 safety cap if no LIMIT present
        if not re.search(r"\bLIMIT\b", cypher, re.IGNORECASE):
            cypher = cypher.rstrip(";") + " LIMIT 200"

        return await self.query(cypher, parameters)

    async def _tool_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for neo4j_schema — returns labels and relationship types."""
        labels_result = await self.query("CALL db.labels() YIELD label RETURN label")
        rel_result = await self.query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        return {
            "labels": [r["label"] for r in labels_result["records"]],
            "relationship_types": [
                r["relationshipType"] for r in rel_result["records"]
            ],
        }

    async def _tool_list_labels(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for neo4j_list_labels"""
        result = await self.query("CALL db.labels() YIELD label RETURN label")
        return {"labels": [r["label"] for r in result["records"]]}

    async def _tool_list_relationship_types(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tool handler for neo4j_list_relationship_types"""
        result = await self.query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        return {
            "relationship_types": [r["relationshipType"] for r in result["records"]]
        }

    async def _tool_graph_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for neo4j_graph_stats"""
        node_result = await self.query("MATCH (n) RETURN count(n) as node_count")
        rel_result = await self.query(
            "MATCH ()-[r]->() RETURN count(r) as relationship_count"
        )
        return {
            "node_count": (
                node_result["records"][0]["node_count"] if node_result["records"] else 0
            ),
            "relationship_count": (
                rel_result["records"][0]["relationship_count"]
                if rel_result["records"]
                else 0
            ),
        }

    async def query(
        self, cypher: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query.

        Args:
            cypher: Cypher query string
            parameters: Optional query parameters

        Returns:
            Query results as {"records": [...], "count": int}

        Raises:
            PluginError: If the query fails
        """
        if self._driver is None:
            await self.connect()

        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(cypher, parameters or {})
                records = await result.data()

                return {"records": records, "count": len(records)}
        except PluginError:
            raise
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            raise PluginError(f"Neo4j query failed: {e}")

    async def _tool_create_node(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for create_node"""
        label = args.get("label")
        properties = args.get("properties")

        result = await self.create_node(label, properties)
        return result

    async def create_node(
        self, label: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new node.

        Args:
            label: Node label
            properties: Node properties

        Returns:
            Created node information
        """
        # Build property string for Cypher
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])

        cypher = f"CREATE (n:{label} {{{props_str}}}) RETURN n"

        result = await self.query(cypher, properties)

        return {
            "node": result["records"][0]["n"] if result["records"] else None,
            "label": label,
        }

    async def _tool_create_relationship(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for create_relationship"""
        from_label = args.get("from_label")
        from_properties = args.get("from_properties")
        to_label = args.get("to_label")
        to_properties = args.get("to_properties")
        relationship_type = args.get("relationship_type")
        relationship_properties = args.get("relationship_properties", {})

        result = await self.create_relationship(
            from_label,
            from_properties,
            to_label,
            to_properties,
            relationship_type,
            relationship_properties,
        )
        return result

    async def create_relationship(
        self,
        from_label: str,
        from_properties: Dict[str, Any],
        to_label: str,
        to_properties: Dict[str, Any],
        relationship_type: str,
        relationship_properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.

        Args:
            from_label: Source node label
            from_properties: Source node properties to match
            to_label: Target node label
            to_properties: Target node properties to match
            relationship_type: Relationship type
            relationship_properties: Optional relationship properties

        Returns:
            Created relationship information
        """
        # Build match conditions
        from_match = self._build_match_condition(from_properties, "from_")
        to_match = self._build_match_condition(to_properties, "to_")

        # Build relationship properties
        rel_props = relationship_properties or {}
        rel_props_str = ""
        if rel_props:
            rel_props_str = (
                " {" + ", ".join([f"{k}: ${k}" for k in rel_props.keys()]) + "}"
            )

        cypher = f"""
        MATCH (a:{from_label} {from_match})
        MATCH (b:{to_label} {to_match})
        CREATE (a)-[r:{relationship_type}{rel_props_str}]->(b)
        RETURN a, r, b
        """

        # Combine all parameters
        params = {}
        for k, v in from_properties.items():
            params[f"from_{k}"] = v
        for k, v in to_properties.items():
            params[f"to_{k}"] = v
        params.update(rel_props)

        result = await self.query(cypher, params)

        return {
            "relationship_type": relationship_type,
            "from_node": result["records"][0]["a"] if result["records"] else None,
            "to_node": result["records"][0]["b"] if result["records"] else None,
        }

    async def _tool_find_nodes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for find_nodes"""
        label = args.get("label")
        properties = args.get("properties", {})
        limit = args.get("limit", 50)

        result = await self.find_nodes(label, properties, limit)
        return result

    async def find_nodes(
        self, label: str, properties: Optional[Dict[str, Any]] = None, limit: int = 50
    ) -> Dict[str, Any]:
        """
        Find nodes by label and properties.

        Args:
            label: Node label
            properties: Properties to match
            limit: Maximum results

        Returns:
            Matching nodes
        """
        match_condition = self._build_match_condition(properties or {})

        cypher = f"MATCH (n:{label} {match_condition}) RETURN n LIMIT {limit}"

        result = await self.query(cypher, properties or {})

        return {
            "nodes": [record["n"] for record in result["records"]],
            "count": len(result["records"]),
        }

    async def _tool_find_path(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for find_path"""
        from_label = args.get("from_label")
        from_properties = args.get("from_properties")
        to_label = args.get("to_label")
        to_properties = args.get("to_properties")
        max_length = args.get("max_length", 5)

        result = await self.find_path(
            from_label, from_properties, to_label, to_properties, max_length
        )
        return result

    async def find_path(
        self,
        from_label: str,
        from_properties: Dict[str, Any],
        to_label: str,
        to_properties: Dict[str, Any],
        max_length: int = 5,
    ) -> Dict[str, Any]:
        """
        Find shortest path between two nodes.

        Args:
            from_label: Start node label
            from_properties: Start node properties
            to_label: End node label
            to_properties: End node properties
            max_length: Maximum path length

        Returns:
            Shortest path information
        """
        from_match = self._build_match_condition(from_properties, "from_")
        to_match = self._build_match_condition(to_properties, "to_")

        cypher = f"""
        MATCH (a:{from_label} {from_match}), (b:{to_label} {to_match})
        MATCH p = shortestPath((a)-[*..{max_length}]-(b))
        RETURN p, length(p) as path_length
        """

        params = {}
        for k, v in from_properties.items():
            params[f"from_{k}"] = v
        for k, v in to_properties.items():
            params[f"to_{k}"] = v

        result = await self.query(cypher, params)

        return {
            "path": result["records"][0]["p"] if result["records"] else None,
            "path_length": (
                result["records"][0]["path_length"] if result["records"] else None
            ),
            "found": len(result["records"]) > 0,
        }

    async def _tool_get_neighbors(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_neighbors"""
        label = args.get("label")
        properties = args.get("properties")
        relationship_type = args.get("relationship_type")
        direction = args.get("direction", "both")

        result = await self.get_neighbors(
            label, properties, relationship_type, direction
        )
        return result

    async def get_neighbors(
        self,
        label: str,
        properties: Dict[str, Any],
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> Dict[str, Any]:
        """
        Get neighboring nodes.

        Args:
            label: Node label
            properties: Node properties to match
            relationship_type: Optional relationship type filter
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            Neighboring nodes and relationships
        """
        match_condition = self._build_match_condition(properties)

        # Build relationship pattern
        rel_pattern = f":{relationship_type}" if relationship_type else ""

        if direction == "outgoing":
            rel_direction = f"-[r{rel_pattern}]->"
        elif direction == "incoming":
            rel_direction = f"<-[r{rel_pattern}]-"
        else:  # both
            rel_direction = f"-[r{rel_pattern}]-"

        cypher = f"""
        MATCH (n:{label} {match_condition}){rel_direction}(m)
        RETURN m, r, type(r) as relationship_type
        """

        result = await self.query(cypher, properties)

        return {
            "neighbors": [
                {
                    "node": record["m"],
                    "relationship_type": record["relationship_type"],
                    "relationship": record["r"],
                }
                for record in result["records"]
            ],
            "count": len(result["records"]),
        }

    async def _tool_delete_node(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for delete_node"""
        label = args.get("label")
        properties = args.get("properties")

        result = await self.delete_node(label, properties)
        return result

    async def delete_node(
        self, label: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delete a node and all its relationships.

        Args:
            label: Node label
            properties: Properties to match the node

        Returns:
            Deletion result
        """
        match_condition = self._build_match_condition(properties)

        cypher = f"MATCH (n:{label} {match_condition}) DETACH DELETE n"

        await self.query(cypher, properties)
        return {"deleted": True}

    def _build_match_condition(
        self, properties: Dict[str, Any], prefix: str = ""
    ) -> str:
        """
        Build Cypher match condition from properties.

        Args:
            properties: Properties dictionary
            prefix: Optional prefix for parameter names

        Returns:
            Cypher match condition string
        """
        if not properties:
            return ""

        conditions = [f"{k}: ${prefix}{k}" for k in properties.keys()]
        return "{" + ", ".join(conditions) + "}"


def neo4j(**kwargs) -> Neo4jPlugin:
    """Create Neo4jPlugin with simplified interface."""
    return Neo4jPlugin(**kwargs)
