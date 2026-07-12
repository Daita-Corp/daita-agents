"""
ChromaDB vector database plugin for Daita Agents.

Embeddable vector database supporting local, persistent, and client-server modes.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from .base import PluginContext
from .base_vector import BaseVectorPlugin
from .chroma_extensions import (
    CHROMA_MANIFEST,
    ChromaExecutor,
    chroma_capabilities,
    chroma_evidence_schemas,
    chroma_operation_definitions,
    chroma_tool_views,
)

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import Include, Metadata, PyEmbedding

    from ..core.tools import LocalTool

logger = logging.getLogger(__name__)


class ChromaPlugin(BaseVectorPlugin):
    """
    ChromaDB vector database plugin for local and embedded vector storage.

    Supports three modes:
    - In-memory (ephemeral): No path or host specified
    - Persistent local: path parameter specified
    - Remote client-server: host parameter specified
    """

    manifest = CHROMA_MANIFEST

    def __init__(
        self,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8000,
        collection: str = "default",
        embedding_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize ChromaDB connection.

        Args:
            path: Optional path for persistent local storage
            host: Optional host for remote Chroma server
            port: Port for remote Chroma server (default: 8000)
            collection: Collection name to use
            embedding_fn: Optional callable that converts text to a vector.
                          Enables passing `text` instead of `vector` to search tools.
            **kwargs: Additional Chroma configuration
        """
        self.path = path
        self.host = host
        self.port = port
        self.collection_name = collection
        self._collection: Optional["Collection"] = None
        self._embedding_fn = embedding_fn

        # Determine mode
        if path:
            self.mode = "persistent"
        elif host:
            self.mode = "client"
        else:
            self.mode = "ephemeral"

        super().__init__(
            path=path,
            host=host,
            port=port,
            collection=collection,
            **kwargs,  # embedding_fn is not forwarded
        )
        self._client: Optional["ClientAPI"] = None
        self._executor = ChromaExecutor(self)

        logger.debug(
            f"ChromaDB plugin configured in {self.mode} mode, collection '{collection}'"
        )

    @property
    def client(self) -> "ClientAPI":
        """Return the active Chroma client owned by this plugin."""
        if self._client is None:
            from ..core.exceptions import ValidationError

            raise ValidationError(
                "ChromaPlugin is not connected",
                field="connection_state",
            )
        return self._client

    @property
    def collection(self) -> "Collection":
        """Return the active Chroma collection owned by this plugin."""
        if self._collection is None:
            from ..core.exceptions import ValidationError

            raise ValidationError(
                "ChromaPlugin is not connected",
                field="connection_state",
            )
        return self._collection

    async def setup(self, context: PluginContext) -> None:
        """Set up the Chroma connector for a runtime."""
        await self.connect()

    async def teardown(self) -> None:
        """Disconnect the Chroma connector from a runtime."""
        await self.disconnect()

    # ---------------------------------------------------------------------------
    # Runtime extension declarations
    # ---------------------------------------------------------------------------

    def declare_capabilities(self):
        return chroma_capabilities()

    def get_executors(self):
        return (self._executor,)

    def declare_evidence_schemas(self):
        return chroma_evidence_schemas()

    def get_tool_views(self):
        return chroma_tool_views()

    def _definition_for_capability(self, capability_id: str) -> dict:
        for definition in chroma_operation_definitions():
            if definition["capability_id"] == capability_id:
                return definition
        raise KeyError(capability_id)

    def _definition_for_tool(self, tool_name: str) -> dict:
        for definition in chroma_operation_definitions():
            if definition["tool_name"] == tool_name:
                return definition
        raise KeyError(tool_name)

    async def connect(self):
        """Connect to ChromaDB."""
        if self._client is not None:
            return

        try:
            import chromadb

            # Create client based on mode
            if self.mode == "persistent":
                path = self.path
                if path is None:
                    raise RuntimeError("persistent Chroma mode requires path")
                client = chromadb.PersistentClient(path=path)
            elif self.mode == "client":
                host = self.host
                if host is None:
                    raise RuntimeError("client Chroma mode requires host")
                client = chromadb.HttpClient(host=host, port=self.port)
            else:
                client = chromadb.Client()

            # Get or create collection
            collection = client.get_or_create_collection(name=self.collection_name)

            self._client = client
            self._collection = collection

            logger.info(f"Connected to ChromaDB in {self.mode} mode")
        except ImportError as exc:
            raise ImportError(
                "chromadb is required. Install with: pip install 'daita-agents[chromadb]'"
            ) from exc
        except Exception as e:
            self._handle_connection_error(e, "connection")

    async def disconnect(self):
        """Disconnect from ChromaDB."""
        self._client = None
        self._collection = None
        logger.info("Disconnected from ChromaDB")

    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Insert or update vectors in ChromaDB.

        Args:
            ids: List of unique vector IDs
            vectors: List of vector embeddings
            metadata: Optional list of metadata dictionaries
            documents: Optional list of raw document texts

        Returns:
            Dictionary with upsert results
        """
        if self._collection is None:
            await self.connect()
        collection = self.collection
        embeddings: "List[PyEmbedding]" = [tuple(vector) for vector in vectors]
        metadatas: "List[Metadata] | None" = (
            [dict(item) for item in metadata] if metadata else None
        )

        # ChromaDB's add method handles both insert and update
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        return {
            "success": True,
            "upserted_count": len(ids),
            "collection": self.collection_name,
        }

    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
        include: Optional["Include"] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in ChromaDB.

        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Chroma where filter (e.g., {"category": "tech"})
            include: List of fields to include (metadatas, documents, distances, embeddings)

        Returns:
            List of matches with metadata and scores
        """
        if self._collection is None:
            await self.connect()
        collection = self.collection

        # Default include fields
        if include is None:
            include = ["metadatas", "documents", "distances"]

        result = collection.query(
            query_embeddings=[vector], n_results=top_k, where=filter, include=include
        )

        # Convert to list of dicts
        matches = []
        ids = result["ids"][0] if result["ids"] else []
        distance_groups = result.get("distances") or []
        distances = distance_groups[0] if distance_groups else []
        metadata_groups = result.get("metadatas") or []
        metadatas = metadata_groups[0] if metadata_groups else []
        document_groups = result.get("documents") or []
        documents = document_groups[0] if document_groups else []
        embedding_groups = result.get("embeddings") or []
        embeddings = (
            embedding_groups[0] if "embeddings" in include and embedding_groups else []
        )

        for idx, id in enumerate(ids):
            match: Dict[str, object] = {
                "id": id,
                "distance": distances[idx] if idx < len(distances) else None,
                "score": 1 / (1 + distances[idx]) if idx < len(distances) else None,
            }

            if len(metadatas) > 0 and idx < len(metadatas):
                match["metadata"] = metadatas[idx]
            if len(documents) > 0 and idx < len(documents):
                match["document"] = documents[idx]
            if embeddings is not None and len(embeddings) > 0 and idx < len(embeddings):
                match["embedding"] = embeddings[idx]

            matches.append(match)

        return matches

    async def delete(
        self, ids: Optional[List[str]] = None, filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Delete vectors from ChromaDB.

        Args:
            ids: Optional list of IDs to delete
            filter: Optional Chroma where filter for deletion

        Returns:
            Dictionary with deletion results
        """
        if self._collection is None:
            await self.connect()
        collection = self.collection

        if ids:
            collection.delete(ids=ids)
            return {
                "success": True,
                "deleted_count": len(ids),
                "collection": self.collection_name,
            }
        elif filter:
            collection.delete(where=filter)
            return {
                "success": True,
                "deleted": "by_filter",
                "collection": self.collection_name,
            }
        else:
            return {"success": False, "error": "Must provide ids or filter"}

    async def fetch(
        self, ids: List[str], include: Optional["Include"] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch vectors by ID from ChromaDB.

        Args:
            ids: List of IDs to fetch
            include: List of fields to include (metadatas, documents, embeddings)

        Returns:
            List of vectors with metadata
        """
        if self._collection is None:
            await self.connect()
        collection = self.collection

        # Default include fields
        if include is None:
            include = ["metadatas", "documents", "embeddings"]

        result = collection.get(ids=ids, include=include)

        # Convert to list of dicts
        vectors = []
        ids_result = result.get("ids", [])
        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or []
        embeddings = (result.get("embeddings") or []) if "embeddings" in include else []

        for idx, id in enumerate(ids_result):
            vector: Dict[str, object] = {"id": id}

            if len(metadatas) > 0 and idx < len(metadatas):
                vector["metadata"] = metadatas[idx]
            if len(documents) > 0 and idx < len(documents):
                vector["document"] = documents[idx]
            if len(embeddings) > 0 and idx < len(embeddings):
                vector["embedding"] = embeddings[idx]

            vectors.append(vector)

        return vectors

    async def list_collections(self) -> List[str]:
        """
        List all collections in ChromaDB (Chroma-specific).

        Returns:
            List of collection names
        """
        if self._client is None:
            await self.connect()
        client = self.client

        collections = client.list_collections()
        return [col.name for col in collections]

    async def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for chroma_search"""
        vector = args.get("vector")
        text = args.get("text")
        top_k = args.get("top_k", 10)
        filter = args.get("filter")

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

        matches = await self.query(vector=normalized_vector, top_k=top_k, filter=filter)

        return {"success": True, "matches": matches, "count": len(matches)}

    async def _tool_upsert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for chroma_upsert"""
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
        documents = args.get("documents")

        result = await self.upsert(
            ids=normalized_ids,
            vectors=vectors,
            metadata=metadata,
            documents=documents,
        )

        return result

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for chroma_delete"""
        ids = args.get("ids")
        filter = args.get("filter")

        result = await self.delete(ids=ids, filter=filter)
        return result

    async def _tool_collections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for chroma_collections"""
        collections = await self.list_collections()

        return {"success": True, "collections": collections, "count": len(collections)}


def chroma(**kwargs) -> ChromaPlugin:
    """Create ChromaDB plugin with simplified interface."""
    return ChromaPlugin(**kwargs)
