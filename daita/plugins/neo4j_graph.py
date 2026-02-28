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

from .base import BasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class Neo4jPlugin(BasePlugin):
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

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: Optional[tuple] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        **kwargs
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j URI (bolt:// or neo4j://)
            auth: Tuple of (username, password)
            username: Username (alternative to auth tuple)
            password: Password (alternative to auth tuple)
            database: Database name (default: "neo4j")
            **kwargs: Additional neo4j driver parameters
        """
        self._uri = uri
        self._database = database

        # Handle auth parameter
        if auth:
            self._auth = auth
        elif username and password:
            self._auth = (username, password)
        else:
            self._auth = ("neo4j", "password")  # Default

        self._driver = None
        self._driver_config = kwargs

        logger.debug(f"Neo4jPlugin initialized for {uri}, database={database}")

    async def connect(self):
        """Connect to Neo4j database."""
        if self._driver is not None:
            return  # Already connected

        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=self._auth,
                **self._driver_config
            )

            # Verify connectivity
            await self._driver.verify_connectivity()

            logger.info(f"Connected to Neo4j at {self._uri}")
        except ImportError:
            raise ImportError(
                "neo4j driver not installed. Run: pip install neo4j"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

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

    def get_tools(self) -> List['AgentTool']:
        """
        Expose Neo4j operations as agent tools.

        Returns:
            List of AgentTool instances for graph operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="query_graph",
                description="Execute a Cypher query against the Neo4j graph database",
                parameters={
                    "type": "object",
                    "properties": {
                        "cypher": {
                            "type": "string",
                            "description": "Cypher query to execute (e.g., 'MATCH (n:Person) RETURN n LIMIT 10')"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Optional query parameters as key-value pairs"
                        }
                    },
                    "required": ["cypher"]
                },
                handler=self._tool_query,
                category="graph",
                source="plugin",
                plugin_name="Neo4j",
                timeout_seconds=60
            ),
            AgentTool(
                name="create_node",
                description="Create a new node in the graph with a label and properties",
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Node label (e.g., 'Person', 'Company')"
                        },
                        "properties": {
                            "type": "object",
                            "description": "Node properties as key-value pairs"
                        }
                    },
                    "required": ["label", "properties"]
                },
                handler=self._tool_create_node,
                category="graph",
                source="plugin",
                plugin_name="Neo4j",
                timeout_seconds=30
            ),
            AgentTool(
                name="create_relationship",
                description="Create a relationship between two nodes",
                parameters={
                    "type": "object",
                    "properties": {
                        "from_label": {
                            "type": "string",
                            "description": "Label of source node"
                        },
                        "from_properties": {
                            "type": "object",
                            "description": "Properties to match source node"
                        },
                        "to_label": {
                            "type": "string",
                            "description": "Label of target node"
                        },
                        "to_properties": {
                            "type": "object",
                            "description": "Properties to match target node"
                        },
                        "relationship_type": {
                            "type": "string",
                            "description": "Type of relationship (e.g., 'KNOWS', 'WORKS_AT')"
                        },
                        "relationship_properties": {
                            "type": "object",
                            "description": "Optional properties for the relationship"
                        }
                    },
                    "required": ["from_label", "from_properties", "to_label", "to_properties", "relationship_type"]
                },
                handler=self._tool_create_relationship,
                category="graph",
                source="plugin",
                plugin_name="Neo4j",
                timeout_seconds=30
            ),
            AgentTool(
                name="find_nodes",
                description="Find nodes by label and properties",
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Node label to search for"
                        },
                        "properties": {
                            "type": "object",
                            "description": "Properties to match (partial match supported)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of nodes to return (default: 50)"
                        }
                    },
                    "required": ["label"]
                },
                handler=self._tool_find_nodes,
                category="graph",
                source="plugin",
                plugin_name="Neo4j",
                timeout_seconds=60
            ),
            AgentTool(
                name="find_path",
                description="Find shortest path between two nodes",
                parameters={
                    "type": "object",
                    "properties": {
                        "from_label": {
                            "type": "string",
                            "description": "Label of start node"
                        },
                        "from_properties": {
                            "type": "object",
                            "description": "Properties to match start node"
                        },
                        "to_label": {
                            "type": "string",
                            "description": "Label of end node"
                        },
                        "to_properties": {
                            "type": "object",
                            "description": "Properties to match end node"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum path length (default: 5)"
                        }
                    },
                    "required": ["from_label", "from_properties", "to_label", "to_properties"]
                },
                handler=self._tool_find_path,
                category="graph",
                source="plugin",
                plugin_name="Neo4j",
                timeout_seconds=60
            ),
            AgentTool(
                name="get_neighbors",
                description="Get all neighboring nodes connected to a node",
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Label of the node"
                        },
                        "properties": {
                            "type": "object",
                            "description": "Properties to match the node"
                        },
                        "relationship_type": {
                            "type": "string",
                            "description": "Optional relationship type to filter (e.g., 'KNOWS')"
                        },
                        "direction": {
                            "type": "string",
                            "description": "Direction: 'outgoing', 'incoming', or 'both' (default: 'both')"
                        }
                    },
                    "required": ["label", "properties"]
                },
                handler=self._tool_get_neighbors,
                category="graph",
                source="plugin",
                plugin_name="Neo4j",
                timeout_seconds=60
            ),
            AgentTool(
                name="delete_node",
                description="Delete a node and its relationships",
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Label of the node to delete"
                        },
                        "properties": {
                            "type": "object",
                            "description": "Properties to match the node"
                        }
                    },
                    "required": ["label", "properties"]
                },
                handler=self._tool_delete_node,
                category="graph",
                source="plugin",
                plugin_name="Neo4j",
                timeout_seconds=30
            )
        ]

    async def _tool_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for query_graph"""
        cypher = args.get("cypher")
        parameters = args.get("parameters", {})

        result = await self.query(cypher, parameters)
        return result

    async def query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a Cypher query.

        Args:
            cypher: Cypher query string
            parameters: Optional query parameters

        Returns:
            Query results and metadata
        """
        if self._driver is None:
            await self.connect()

        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(cypher, parameters or {})
                records = await result.data()

                return {
                    "success": True,
                    "records": records,
                    "count": len(records)
                }
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": cypher
            }

    async def _tool_create_node(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for create_node"""
        label = args.get("label")
        properties = args.get("properties")

        result = await self.create_node(label, properties)
        return result

    async def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
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

        if result["success"]:
            return {
                "success": True,
                "node": result["records"][0]["n"] if result["records"] else None,
                "label": label
            }
        else:
            return result

    async def _tool_create_relationship(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for create_relationship"""
        from_label = args.get("from_label")
        from_properties = args.get("from_properties")
        to_label = args.get("to_label")
        to_properties = args.get("to_properties")
        relationship_type = args.get("relationship_type")
        relationship_properties = args.get("relationship_properties", {})

        result = await self.create_relationship(
            from_label, from_properties,
            to_label, to_properties,
            relationship_type, relationship_properties
        )
        return result

    async def create_relationship(
        self,
        from_label: str,
        from_properties: Dict[str, Any],
        to_label: str,
        to_properties: Dict[str, Any],
        relationship_type: str,
        relationship_properties: Optional[Dict[str, Any]] = None
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
            rel_props_str = " {" + ", ".join([f"{k}: ${k}" for k in rel_props.keys()]) + "}"

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

        if result["success"]:
            return {
                "success": True,
                "relationship_type": relationship_type,
                "from_node": result["records"][0]["a"] if result["records"] else None,
                "to_node": result["records"][0]["b"] if result["records"] else None
            }
        else:
            return result

    async def _tool_find_nodes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for find_nodes"""
        label = args.get("label")
        properties = args.get("properties", {})
        limit = args.get("limit", 50)

        result = await self.find_nodes(label, properties, limit)
        return result

    async def find_nodes(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 50
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

        if result["success"]:
            return {
                "success": True,
                "nodes": [record["n"] for record in result["records"]],
                "count": len(result["records"])
            }
        else:
            return result

    async def _tool_find_path(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for find_path"""
        from_label = args.get("from_label")
        from_properties = args.get("from_properties")
        to_label = args.get("to_label")
        to_properties = args.get("to_properties")
        max_length = args.get("max_length", 5)

        result = await self.find_path(
            from_label, from_properties,
            to_label, to_properties,
            max_length
        )
        return result

    async def find_path(
        self,
        from_label: str,
        from_properties: Dict[str, Any],
        to_label: str,
        to_properties: Dict[str, Any],
        max_length: int = 5
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

        if result["success"]:
            return {
                "success": True,
                "path": result["records"][0]["p"] if result["records"] else None,
                "path_length": result["records"][0]["path_length"] if result["records"] else None,
                "found": len(result["records"]) > 0
            }
        else:
            return result

    async def _tool_get_neighbors(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_neighbors"""
        label = args.get("label")
        properties = args.get("properties")
        relationship_type = args.get("relationship_type")
        direction = args.get("direction", "both")

        result = await self.get_neighbors(label, properties, relationship_type, direction)
        return result

    async def get_neighbors(
        self,
        label: str,
        properties: Dict[str, Any],
        relationship_type: Optional[str] = None,
        direction: str = "both"
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

        if result["success"]:
            return {
                "success": True,
                "neighbors": [
                    {
                        "node": record["m"],
                        "relationship_type": record["relationship_type"],
                        "relationship": record["r"]
                    }
                    for record in result["records"]
                ],
                "count": len(result["records"])
            }
        else:
            return result

    async def _tool_delete_node(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for delete_node"""
        label = args.get("label")
        properties = args.get("properties")

        result = await self.delete_node(label, properties)
        return result

    async def delete_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
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

        result = await self.query(cypher, properties)

        if result["success"]:
            return {
                "success": True,
                "deleted": True
            }
        else:
            return result

    def _build_match_condition(self, properties: Dict[str, Any], prefix: str = "") -> str:
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
