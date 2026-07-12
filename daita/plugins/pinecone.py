"""
Pinecone vector database plugin for Daita Agents.

Managed cloud vector database with serverless and pod-based deployment options.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from .base import PluginContext
from .base_vector import BaseVectorPlugin
from .pinecone_extensions import (
    PINECONE_MANIFEST,
    PineconeExecutor,
    pinecone_capabilities,
    pinecone_evidence_schemas,
    pinecone_operation_definitions,
    pinecone_tool_views,
)

if TYPE_CHECKING:
    from pinecone import GrpcIndex, Index, Pinecone

logger = logging.getLogger(__name__)


class PineconePlugin(BaseVectorPlugin):
    """
    Pinecone vector database plugin for managed cloud vector storage.

    Supports Pinecone's native filter syntax and namespaces for multi-tenancy.
    """

    manifest = PINECONE_MANIFEST

    def __init__(
        self,
        api_key: str,
        index: str,
        namespace: str = "",
        host: Optional[str] = None,
        embedding_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize Pinecone connection.

        Args:
            api_key: Pinecone API key
            index: Index name to use
            namespace: Optional namespace for multi-tenancy
            host: Optional host URL for serverless Pinecone
            embedding_fn: Optional callable that converts text to a vector.
                          Enables passing `text` instead of `vector` to search tools.
            **kwargs: Additional Pinecone configuration
        """
        self.api_key = api_key
        self.index_name = index
        self.namespace = namespace
        self.host = host
        self._index: Optional["Index | GrpcIndex"] = None
        self._embedding_fn = embedding_fn
        self._executor = PineconeExecutor(self)

        super().__init__(
            api_key=api_key, index=index, namespace=namespace, host=host, **kwargs
        )
        self._client: Optional["Pinecone"] = None

        logger.debug(f"Pinecone plugin configured for index '{index}'")

    @property
    def client(self) -> "Pinecone":
        """Return the active Pinecone control-plane client."""
        if self._client is None:
            from ..core.exceptions import ValidationError

            raise ValidationError(
                "PineconePlugin is not connected",
                field="connection_state",
            )
        return self._client

    @property
    def pinecone_index(self) -> "Index | GrpcIndex":
        """Return the active Pinecone data-plane index."""
        if self._index is None:
            from ..core.exceptions import ValidationError

            raise ValidationError(
                "PineconePlugin is not connected",
                field="connection_state",
            )
        return self._index

    async def setup(self, context: PluginContext) -> None:
        """Set up the Pinecone connector for a runtime."""
        await self.connect()

    async def teardown(self) -> None:
        """Disconnect the Pinecone connector from a runtime."""
        await self.disconnect()

    # ---------------------------------------------------------------------------
    # Runtime extension declarations
    # ---------------------------------------------------------------------------

    def declare_capabilities(self):
        return pinecone_capabilities()

    def get_executors(self):
        return (self._executor,)

    def declare_evidence_schemas(self):
        return pinecone_evidence_schemas()

    def get_tool_views(self):
        return pinecone_tool_views()

    def _definition_for_capability(self, capability_id: str) -> dict:
        for definition in pinecone_operation_definitions():
            if definition["capability_id"] == capability_id:
                return definition
        raise KeyError(capability_id)

    def _definition_for_tool(self, tool_name: str) -> dict:
        for definition in pinecone_operation_definitions():
            if definition["tool_name"] == tool_name:
                return definition
        raise KeyError(tool_name)

    async def connect(self):
        """Connect to Pinecone."""
        if self._client is not None:
            return

        try:
            from pinecone import Pinecone

            client = Pinecone(api_key=self.api_key)
            # Get index - new API automatically resolves host
            index = client.Index(self.index_name)

            self._client = client
            self._index = index

            logger.info(f"Connected to Pinecone index '{self.index_name}'")
        except ImportError as exc:
            raise ImportError(
                "pinecone is required. Install with: pip install 'daita-agents[pinecone]'"
            ) from exc
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
        namespace: Optional[str] = None,
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
        index = self.pinecone_index

        namespace = namespace or self.namespace

        # Build upsert data
        upsert_data = []
        for idx, (id, vector) in enumerate(zip(ids, vectors)):
            item: Dict[str, object] = {"id": id, "values": vector}
            if metadata and idx < len(metadata):
                item["metadata"] = metadata[idx]
            upsert_data.append(item)

        # Upsert vectors
        result = index.upsert(vectors=upsert_data, namespace=namespace)

        return {
            "upserted_count": result.get("upserted_count", len(ids)),
            "namespace": namespace,
        }

    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_values: bool = False,
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
        index = self.pinecone_index

        namespace = namespace or self.namespace

        result = index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            namespace=namespace,
            include_metadata=include_metadata,
            include_values=include_values,
        )

        # Convert matches to list of dicts
        matches = []
        for match in result.get("matches", []):
            item = {"id": match.get("id"), "score": match.get("score")}
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
        delete_all: bool = False,
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
        index = self.pinecone_index

        namespace = namespace or self.namespace

        if delete_all:
            index.delete(delete_all=True, namespace=namespace)
            return {"success": True, "deleted": "all", "namespace": namespace}
        elif ids:
            index.delete(ids=ids, namespace=namespace)
            return {"success": True, "deleted_count": len(ids), "namespace": namespace}
        elif filter:
            index.delete(filter=filter, namespace=namespace)
            return {"success": True, "deleted": "by_filter", "namespace": namespace}
        else:
            return {
                "success": False,
                "error": "Must provide ids, filter, or delete_all=True",
            }

    async def fetch(
        self, ids: List[str], namespace: Optional[str] = None
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
        index = self.pinecone_index

        namespace = namespace or self.namespace

        result = index.fetch(ids=ids, namespace=namespace)

        # Convert to list of dicts
        vectors = []
        for id, data in result.get("vectors", {}).items():
            vectors.append(
                {
                    "id": id,
                    "values": data.get("values"),
                    "metadata": data.get("metadata", {}),
                }
            )

        return vectors

    async def describe_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics (Pinecone-specific).

        Returns:
            Dictionary with index stats (dimension, count, etc.)
        """
        if self._index is None:
            await self.connect()
        index = self.pinecone_index

        stats = index.describe_index_stats()
        return {
            "dimension": stats.get("dimension"),
            "index_fullness": stats.get("index_fullness"),
            "total_vector_count": stats.get("total_vector_count"),
            "namespaces": stats.get("namespaces", {}),
        }

    async def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for pinecone_search"""
        vector = args.get("vector")
        text = args.get("text")
        top_k = args.get("top_k", 10)
        filter = args.get("filter")
        namespace = args.get("namespace")

        if vector is None:
            if not self._embedding_fn:
                raise ValueError(
                    "Provide 'vector' (float array) or configure embedding_fn in the constructor"
                )
            result = self._embedding_fn(text)
            vector = await result if asyncio.iscoroutine(result) else result

        if not isinstance(vector, list) or not all(
            isinstance(value, (int, float)) for value in vector
        ):
            raise ValueError("vector must be a list of numbers")
        normalized_vector = [float(value) for value in vector]

        matches = await self.query(
            vector=normalized_vector,
            top_k=top_k,
            filter=filter,
            namespace=namespace,
        )

        return {"success": True, "matches": matches, "count": len(matches)}

    async def _tool_upsert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for pinecone_upsert"""
        ids = args.get("ids")
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError("ids must be a list of strings")
        normalized_ids = [item for item in ids if isinstance(item, str)]
        raw_vectors = args.get("vectors")
        if not isinstance(raw_vectors, list) or not all(
            isinstance(vector, list)
            and all(isinstance(value, (int, float)) for value in vector)
            for vector in raw_vectors
        ):
            raise ValueError("vectors must be a list of numeric vectors")
        vectors = [
            [float(value) for value in vector]
            for vector in raw_vectors
            if isinstance(vector, list)
        ]
        metadata = args.get("metadata")
        namespace = args.get("namespace")

        result = await self.upsert(
            ids=normalized_ids,
            vectors=vectors,
            metadata=metadata,
            namespace=namespace,
        )

        return {
            "success": True,
            "upserted_count": result.get("upserted_count"),
            "namespace": result.get("namespace"),
        }

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for pinecone_delete"""
        ids = args.get("ids")
        filter = args.get("filter")
        namespace = args.get("namespace")
        delete_all = args.get("delete_all", False)

        result = await self.delete(
            ids=ids, filter=filter, namespace=namespace, delete_all=delete_all
        )

        return result

    async def _tool_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for pinecone_stats"""
        stats = await self.describe_index_stats()

        return {"success": True, "stats": stats}


def pinecone(**kwargs) -> PineconePlugin:
    """Create Pinecone plugin with simplified interface."""
    return PineconePlugin(**kwargs)
