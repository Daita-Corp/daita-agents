"""
Pinecone vector database plugin for Daita Agents.

Managed cloud vector database with serverless and pod-based deployment options.
"""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base_vector import BaseVectorPlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class PineconePlugin(BaseVectorPlugin):
    """
    Pinecone vector database plugin for managed cloud vector storage.

    Supports Pinecone's native filter syntax and namespaces for multi-tenancy.
    """

    def __init__(
        self,
        api_key: str,
        index: str,
        namespace: str = "",
        host: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Pinecone connection.

        Args:
            api_key: Pinecone API key
            index: Index name to use
            namespace: Optional namespace for multi-tenancy
            host: Optional host URL for serverless Pinecone
            **kwargs: Additional Pinecone configuration
        """
        self.api_key = api_key
        self.index_name = index
        self.namespace = namespace
        self.host = host
        self._index = None

        super().__init__(
            api_key=api_key,
            index=index,
            namespace=namespace,
            host=host,
            **kwargs
        )

        logger.debug(f"Pinecone plugin configured for index '{index}'")

    async def connect(self):
        """Connect to Pinecone."""
        if self._client is not None:
            return

        try:
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self.api_key)
            # Get index - new API automatically resolves host
            self._index = self._client.Index(self.index_name)

            logger.info(f"Connected to Pinecone index '{self.index_name}'")
        except ImportError:
            self._handle_connection_error(
                ImportError("pinecone not installed. Run: pip install pinecone"),
                "connection"
            )
        except Exception as e:
            self._handle_connection_error(e, "connection")

    async def disconnect(self):
        """Disconnect from Pinecone."""
        self._client = None
        self._index = None
        logger.info("Disconnected from Pinecone")

    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Insert or update vectors in Pinecone.

        Args:
            ids: List of unique vector IDs
            vectors: List of vector embeddings
            metadata: Optional list of metadata dictionaries
            namespace: Optional namespace (overrides instance default)

        Returns:
            Dictionary with upsert results
        """
        if self._index is None:
            await self.connect()

        namespace = namespace or self.namespace

        # Build upsert data
        upsert_data = []
        for idx, (id, vector) in enumerate(zip(ids, vectors)):
            item = {"id": id, "values": vector}
            if metadata and idx < len(metadata):
                item["metadata"] = metadata[idx]
            upsert_data.append(item)

        # Upsert vectors
        result = self._index.upsert(vectors=upsert_data, namespace=namespace)

        return {
            "upserted_count": result.get("upserted_count", len(ids)),
            "namespace": namespace
        }

    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Pinecone.

        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Pinecone filter dict (e.g., {"category": {"$eq": "tech"}})
            namespace: Optional namespace (overrides instance default)
            include_metadata: Include metadata in results
            include_values: Include vector values in results

        Returns:
            List of matches with scores and metadata
        """
        if self._index is None:
            await self.connect()

        namespace = namespace or self.namespace

        result = self._index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            namespace=namespace,
            include_metadata=include_metadata,
            include_values=include_values
        )

        # Convert matches to list of dicts
        matches = []
        for match in result.get("matches", []):
            item = {
                "id": match.get("id"),
                "score": match.get("score")
            }
            if include_metadata and "metadata" in match:
                item["metadata"] = match.get("metadata")
            if include_values and "values" in match:
                item["values"] = match.get("values")
            matches.append(item)

        return matches

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """
        Delete vectors from Pinecone.

        Args:
            ids: Optional list of IDs to delete
            filter: Optional Pinecone filter for deletion
            namespace: Optional namespace (overrides instance default)
            delete_all: If True, delete all vectors in namespace

        Returns:
            Dictionary with deletion results
        """
        if self._index is None:
            await self.connect()

        namespace = namespace or self.namespace

        if delete_all:
            self._index.delete(delete_all=True, namespace=namespace)
            return {"success": True, "deleted": "all", "namespace": namespace}
        elif ids:
            self._index.delete(ids=ids, namespace=namespace)
            return {"success": True, "deleted_count": len(ids), "namespace": namespace}
        elif filter:
            self._index.delete(filter=filter, namespace=namespace)
            return {"success": True, "deleted": "by_filter", "namespace": namespace}
        else:
            return {"success": False, "error": "Must provide ids, filter, or delete_all=True"}

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch vectors by ID from Pinecone.

        Args:
            ids: List of IDs to fetch
            namespace: Optional namespace (overrides instance default)

        Returns:
            List of vectors with metadata
        """
        if self._index is None:
            await self.connect()

        namespace = namespace or self.namespace

        result = self._index.fetch(ids=ids, namespace=namespace)

        # Convert to list of dicts
        vectors = []
        for id, data in result.get("vectors", {}).items():
            vectors.append({
                "id": id,
                "values": data.get("values"),
                "metadata": data.get("metadata", {})
            })

        return vectors

    async def describe_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics (Pinecone-specific).

        Returns:
            Dictionary with index stats (dimension, count, etc.)
        """
        if self._index is None:
            await self.connect()

        stats = self._index.describe_index_stats()
        return {
            "dimension": stats.get("dimension"),
            "index_fullness": stats.get("index_fullness"),
            "total_vector_count": stats.get("total_vector_count"),
            "namespaces": stats.get("namespaces", {})
        }

    def get_tools(self) -> List['AgentTool']:
        """
        Expose Pinecone operations as agent tools.

        Returns:
            List of AgentTool instances for vector operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="pinecone_search",
                description="Search for similar vectors in Pinecone using semantic similarity. Returns top matching results with scores.",
                parameters={
                    "type": "object",
                    "properties": {
                        "vector": {
                            "type": "array",
                            "description": "Query vector as array of floats",
                            "items": {"type": "number"}
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10)"
                        },
                        "filter": {
                            "type": "object",
                            "description": "Pinecone filter dict (e.g., {\"category\": {\"$eq\": \"tech\"}})"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace to search within"
                        }
                    },
                    "required": ["vector"]
                },
                handler=self._tool_search,
                category="vector_db",
                source="plugin",
                plugin_name="Pinecone",
                timeout_seconds=60
            ),
            AgentTool(
                name="pinecone_upsert",
                description="Insert or update vectors in Pinecone with metadata. Creates new vectors or updates existing ones.",
                parameters={
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "description": "List of unique vector IDs",
                            "items": {"type": "string"}
                        },
                        "vectors": {
                            "type": "array",
                            "description": "List of vectors (each vector is an array of floats)",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        },
                        "metadata": {
                            "type": "array",
                            "description": "Optional list of metadata objects (one per vector)",
                            "items": {"type": "object"}
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace"
                        }
                    },
                    "required": ["ids", "vectors"]
                },
                handler=self._tool_upsert,
                category="vector_db",
                source="plugin",
                plugin_name="Pinecone",
                timeout_seconds=60
            ),
            AgentTool(
                name="pinecone_delete",
                description="Delete vectors from Pinecone by ID or filter. Can delete specific vectors or all vectors in a namespace.",
                parameters={
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "description": "List of vector IDs to delete",
                            "items": {"type": "string"}
                        },
                        "filter": {
                            "type": "object",
                            "description": "Pinecone filter for deletion"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace"
                        },
                        "delete_all": {
                            "type": "boolean",
                            "description": "If true, delete all vectors in namespace"
                        }
                    },
                    "required": []
                },
                handler=self._tool_delete,
                category="vector_db",
                source="plugin",
                plugin_name="Pinecone",
                timeout_seconds=60
            ),
            AgentTool(
                name="pinecone_stats",
                description="Get Pinecone index statistics including dimension, vector count, and namespace information.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=self._tool_stats,
                category="vector_db",
                source="plugin",
                plugin_name="Pinecone",
                timeout_seconds=30
            )
        ]

    async def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for pinecone_search"""
        vector = args.get("vector")
        top_k = args.get("top_k", 10)
        filter = args.get("filter")
        namespace = args.get("namespace")

        matches = await self.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            namespace=namespace
        )

        return {
            "success": True,
            "matches": matches,
            "count": len(matches)
        }

    async def _tool_upsert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for pinecone_upsert"""
        ids = args.get("ids")
        vectors = args.get("vectors")
        metadata = args.get("metadata")
        namespace = args.get("namespace")

        result = await self.upsert(
            ids=ids,
            vectors=vectors,
            metadata=metadata,
            namespace=namespace
        )

        return {
            "success": True,
            "upserted_count": result.get("upserted_count"),
            "namespace": result.get("namespace")
        }

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for pinecone_delete"""
        ids = args.get("ids")
        filter = args.get("filter")
        namespace = args.get("namespace")
        delete_all = args.get("delete_all", False)

        result = await self.delete(
            ids=ids,
            filter=filter,
            namespace=namespace,
            delete_all=delete_all
        )

        return result

    async def _tool_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for pinecone_stats"""
        stats = await self.describe_index_stats()

        return {
            "success": True,
            "stats": stats
        }


def pinecone(**kwargs) -> PineconePlugin:
    """Create Pinecone plugin with simplified interface."""
    return PineconePlugin(**kwargs)
