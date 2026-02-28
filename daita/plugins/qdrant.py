"""
Qdrant vector database plugin for Daita Agents.

Self-hosted vector database with advanced filtering and high performance.
"""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base_vector import BaseVectorPlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class QdrantPlugin(BaseVectorPlugin):
    """
    Qdrant vector database plugin for self-hosted vector storage.

    Supports advanced payload filtering with automatic conversion
    from simple dict filters to Qdrant Filter objects.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection: str = "default",
        **kwargs
    ):
        """
        Initialize Qdrant connection.

        Args:
            url: Qdrant server URL (default: http://localhost:6333)
            api_key: Optional API key for authentication
            collection: Collection name to use
            **kwargs: Additional Qdrant configuration
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection

        super().__init__(
            url=url,
            api_key=api_key,
            collection=collection,
            **kwargs
        )

        logger.debug(f"Qdrant plugin configured for {url}, collection '{collection}'")

    async def connect(self):
        """Connect to Qdrant."""
        if self._client is not None:
            return

        try:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, api_key=self.api_key)

            # Verify collection exists
            try:
                self._client.get_collection(self.collection_name)
                logger.info(f"Connected to Qdrant collection '{self.collection_name}'")
            except Exception:
                logger.warning(f"Collection '{self.collection_name}' does not exist. Create it with create_collection()")

        except ImportError:
            self._handle_connection_error(
                ImportError("qdrant-client not installed. Run: pip install qdrant-client"),
                "connection"
            )
        except Exception as e:
            self._handle_connection_error(e, "connection")

    async def disconnect(self):
        """Disconnect from Qdrant."""
        if self._client:
            self._client.close()
        self._client = None
        logger.info("Disconnected from Qdrant")

    def _dict_to_filter(self, filter_dict: Dict) -> Any:
        """
        Convert simple dict filter to Qdrant Filter object.

        Args:
            filter_dict: Simple filter like {"category": "tech"}

        Returns:
            Qdrant Filter object
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = []
        for key, value in filter_dict.items():
            if key.startswith("$"):
                continue  # Skip operators for now
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

        return Filter(must=conditions) if conditions else None

    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Insert or update vectors in Qdrant.

        Args:
            ids: List of unique vector IDs (strings will be converted to UUIDs)
            vectors: List of vector embeddings
            metadata: Optional list of metadata (payload) dictionaries

        Returns:
            Dictionary with upsert results
        """
        if self._client is None:
            await self.connect()

        from qdrant_client.models import PointStruct
        import uuid

        # Build points
        points = []
        for idx, (id, vector) in enumerate(zip(ids, vectors)):
            payload = metadata[idx] if metadata and idx < len(metadata) else {}
            # Store original string ID in payload for retrieval
            payload["_original_id"] = id
            # Convert string ID to UUID
            # Create deterministic UUID from string ID
            id_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, id))
            points.append(PointStruct(id=id_uuid, vector=vector, payload=payload))

        # Upsert points
        result = self._client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return {
            "success": True,
            "upserted_count": len(ids),
            "operation_id": result.operation_id if hasattr(result, 'operation_id') else None,
            "collection": self.collection_name
        }

    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant.

        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Simple filter dict (converted to Qdrant Filter internally)
            with_payload: Include payload (metadata) in results
            with_vectors: Include vector values in results

        Returns:
            List of matches with scores and payload
        """
        if self._client is None:
            await self.connect()

        # Convert simple dict filter to Qdrant Filter
        qdrant_filter = self._dict_to_filter(filter) if filter else None

        results = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=with_payload,
            with_vectors=with_vectors
        ).points

        # Convert to list of dicts
        matches = []
        for result in results:
            # Get original ID from payload if it exists
            original_id = result.payload.get("_original_id", result.id) if result.payload else result.id

            match = {
                "id": original_id,
                "score": result.score
            }
            if with_payload and result.payload:
                # Remove internal fields from payload
                payload = {k: v for k, v in result.payload.items() if not k.startswith("_")}
                match["payload"] = payload
            if with_vectors and result.vector:
                match["vector"] = result.vector
            matches.append(match)

        return matches

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Delete vectors from Qdrant.

        Args:
            ids: Optional list of IDs to delete (strings will be converted to UUIDs)
            filter: Optional simple filter dict for deletion

        Returns:
            Dictionary with deletion results
        """
        if self._client is None:
            await self.connect()

        if ids:
            import uuid
            # Convert string IDs to UUIDs
            id_uuids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, id)) for id in ids]

            result = self._client.delete(
                collection_name=self.collection_name,
                points_selector=id_uuids
            )
            return {
                "success": True,
                "deleted_count": len(ids),
                "operation_id": result.operation_id if hasattr(result, 'operation_id') else None,
                "collection": self.collection_name
            }
        elif filter:
            qdrant_filter = self._dict_to_filter(filter)
            result = self._client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_filter
            )
            return {
                "success": True,
                "deleted": "by_filter",
                "operation_id": result.operation_id if hasattr(result, 'operation_id') else None,
                "collection": self.collection_name
            }
        else:
            return {
                "success": False,
                "error": "Must provide ids or filter"
            }

    async def fetch(
        self,
        ids: List[str],
        with_payload: bool = True,
        with_vectors: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch vectors by ID from Qdrant.

        Args:
            ids: List of IDs to fetch (strings will be converted to UUIDs)
            with_payload: Include payload in results
            with_vectors: Include vector values in results

        Returns:
            List of vectors with payload
        """
        if self._client is None:
            await self.connect()

        import uuid
        # Convert string IDs to UUIDs
        id_uuids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, id)) for id in ids]

        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=id_uuids,
            with_payload=with_payload,
            with_vectors=with_vectors
        )

        # Convert to list of dicts
        vectors = []
        for result in results:
            # Get original ID from payload if it exists
            original_id = result.payload.get("_original_id", result.id) if result.payload else result.id

            vector = {"id": original_id}
            if with_payload and result.payload:
                # Remove internal fields from payload
                payload = {k: v for k, v in result.payload.items() if not k.startswith("_")}
                vector["payload"] = payload
            if with_vectors and result.vector:
                vector["vector"] = result.vector
            vectors.append(vector)

        return vectors

    async def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "Cosine"
    ) -> Dict[str, Any]:
        """
        Create a new collection in Qdrant (Qdrant-specific).

        Args:
            name: Collection name
            vector_size: Dimension of vectors
            distance: Distance metric - "Cosine", "Euclid", or "Dot"

        Returns:
            Dictionary with creation results
        """
        if self._client is None:
            await self.connect()

        from qdrant_client.models import VectorParams, Distance

        # Map string to Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT
        }

        distance_metric = distance_map.get(distance, Distance.COSINE)

        try:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=distance_metric)
            )
            return {
                "success": True,
                "collection": name,
                "vector_size": vector_size,
                "distance": distance
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection": name
            }

    async def list_collections(self) -> List[str]:
        """
        List all collections in Qdrant (Qdrant-specific).

        Returns:
            List of collection names
        """
        if self._client is None:
            await self.connect()

        collections = self._client.get_collections()
        return [col.name for col in collections.collections]

    def get_tools(self) -> List['AgentTool']:
        """
        Expose Qdrant operations as agent tools.

        Returns:
            List of AgentTool instances for vector operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="qdrant_search",
                description="Search for similar vectors in Qdrant using semantic similarity. Returns top matching results with scores.",
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
                            "description": "Simple filter dict (e.g., {\"category\": \"tech\"})"
                        }
                    },
                    "required": ["vector"]
                },
                handler=self._tool_search,
                category="vector_db",
                source="plugin",
                plugin_name="Qdrant",
                timeout_seconds=60
            ),
            AgentTool(
                name="qdrant_upsert",
                description="Insert or update vectors in Qdrant with payload (metadata).",
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
                            "description": "Optional list of metadata (payload) objects",
                            "items": {"type": "object"}
                        }
                    },
                    "required": ["ids", "vectors"]
                },
                handler=self._tool_upsert,
                category="vector_db",
                source="plugin",
                plugin_name="Qdrant",
                timeout_seconds=60
            ),
            AgentTool(
                name="qdrant_delete",
                description="Delete vectors from Qdrant by ID or filter.",
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
                            "description": "Simple filter dict for deletion"
                        }
                    },
                    "required": []
                },
                handler=self._tool_delete,
                category="vector_db",
                source="plugin",
                plugin_name="Qdrant",
                timeout_seconds=60
            ),
            AgentTool(
                name="qdrant_create_collection",
                description="Create a new collection in Qdrant with specified vector size and distance metric.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Collection name"
                        },
                        "vector_size": {
                            "type": "integer",
                            "description": "Dimension of vectors"
                        },
                        "distance": {
                            "type": "string",
                            "description": "Distance metric: 'Cosine', 'Euclid', or 'Dot' (default: 'Cosine')"
                        }
                    },
                    "required": ["name", "vector_size"]
                },
                handler=self._tool_create_collection,
                category="vector_db",
                source="plugin",
                plugin_name="Qdrant",
                timeout_seconds=30
            )
        ]

    async def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for qdrant_search"""
        vector = args.get("vector")
        top_k = args.get("top_k", 10)
        filter = args.get("filter")

        matches = await self.query(
            vector=vector,
            top_k=top_k,
            filter=filter
        )

        return {
            "success": True,
            "matches": matches,
            "count": len(matches)
        }

    async def _tool_upsert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for qdrant_upsert"""
        ids = args.get("ids")
        vectors = args.get("vectors")
        metadata = args.get("metadata")

        result = await self.upsert(
            ids=ids,
            vectors=vectors,
            metadata=metadata
        )

        return result

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for qdrant_delete"""
        ids = args.get("ids")
        filter = args.get("filter")

        result = await self.delete(ids=ids, filter=filter)
        return result

    async def _tool_create_collection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for qdrant_create_collection"""
        name = args.get("name")
        vector_size = args.get("vector_size")
        distance = args.get("distance", "Cosine")

        result = await self.create_collection(
            name=name,
            vector_size=vector_size,
            distance=distance
        )

        return result


def qdrant(**kwargs) -> QdrantPlugin:
    """Create Qdrant plugin with simplified interface."""
    return QdrantPlugin(**kwargs)
