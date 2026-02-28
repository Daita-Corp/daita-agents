"""
ChromaDB vector database plugin for Daita Agents.

Embeddable vector database supporting local, persistent, and client-server modes.
"""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base_vector import BaseVectorPlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class ChromaPlugin(BaseVectorPlugin):
    """
    ChromaDB vector database plugin for local and embedded vector storage.

    Supports three modes:
    - In-memory (ephemeral): No path or host specified
    - Persistent local: path parameter specified
    - Remote client-server: host parameter specified
    """

    def __init__(
        self,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8000,
        collection: str = "default",
        **kwargs
    ):
        """
        Initialize ChromaDB connection.

        Args:
            path: Optional path for persistent local storage
            host: Optional host for remote Chroma server
            port: Port for remote Chroma server (default: 8000)
            collection: Collection name to use
            **kwargs: Additional Chroma configuration
        """
        self.path = path
        self.host = host
        self.port = port
        self.collection_name = collection
        self._collection = None

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
            **kwargs
        )

        logger.debug(f"ChromaDB plugin configured in {self.mode} mode, collection '{collection}'")

    async def connect(self):
        """Connect to ChromaDB."""
        if self._client is not None:
            return

        try:
            import chromadb

            # Create client based on mode
            if self.mode == "persistent":
                self._client = chromadb.PersistentClient(path=self.path)
            elif self.mode == "client":
                self._client = chromadb.HttpClient(host=self.host, port=self.port)
            else:
                self._client = chromadb.Client()

            # Get or create collection
            self._collection = self._client.get_or_create_collection(name=self.collection_name)

            logger.info(f"Connected to ChromaDB in {self.mode} mode")
        except ImportError:
            self._handle_connection_error(
                ImportError("chromadb not installed. Run: pip install chromadb"),
                "connection"
            )
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
        documents: Optional[List[str]] = None
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

        # ChromaDB's add method handles both insert and update
        kwargs = {
            "ids": ids,
            "embeddings": vectors
        }

        if metadata:
            kwargs["metadatas"] = metadata
        if documents:
            kwargs["documents"] = documents

        self._collection.add(**kwargs)

        return {
            "success": True,
            "upserted_count": len(ids),
            "collection": self.collection_name
        }

    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
        include: Optional[List[str]] = None
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

        # Default include fields
        if include is None:
            include = ["metadatas", "documents", "distances"]

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter,
            include=include
        )

        # Convert to list of dicts
        matches = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0] if "embeddings" in include else None

        for idx, id in enumerate(ids):
            match = {
                "id": id,
                "distance": distances[idx] if idx < len(distances) else None,
                "score": 1 / (1 + distances[idx]) if idx < len(distances) else None
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
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None
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

        if ids:
            self._collection.delete(ids=ids)
            return {
                "success": True,
                "deleted_count": len(ids),
                "collection": self.collection_name
            }
        elif filter:
            self._collection.delete(where=filter)
            return {
                "success": True,
                "deleted": "by_filter",
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
        include: Optional[List[str]] = None
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

        # Default include fields
        if include is None:
            include = ["metadatas", "documents", "embeddings"]

        result = self._collection.get(ids=ids, include=include)

        # Convert to list of dicts
        vectors = []
        ids_result = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        documents = result.get("documents", [])
        embeddings = result.get("embeddings", []) if "embeddings" in include else []

        for idx, id in enumerate(ids_result):
            vector = {"id": id}

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

        collections = self._client.list_collections()
        return [col.name for col in collections]

    def get_tools(self) -> List['AgentTool']:
        """
        Expose ChromaDB operations as agent tools.

        Returns:
            List of AgentTool instances for vector operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="chroma_search",
                description="Search for similar vectors in ChromaDB using semantic similarity. Returns top matching results with scores.",
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
                            "description": "Chroma where filter (e.g., {\"category\": \"tech\"})"
                        }
                    },
                    "required": ["vector"]
                },
                handler=self._tool_search,
                category="vector_db",
                source="plugin",
                plugin_name="ChromaDB",
                timeout_seconds=60
            ),
            AgentTool(
                name="chroma_upsert",
                description="Insert or update vectors in ChromaDB with metadata and optional raw documents.",
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
                        "documents": {
                            "type": "array",
                            "description": "Optional list of raw document texts",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["ids", "vectors"]
                },
                handler=self._tool_upsert,
                category="vector_db",
                source="plugin",
                plugin_name="ChromaDB",
                timeout_seconds=60
            ),
            AgentTool(
                name="chroma_delete",
                description="Delete vectors from ChromaDB by ID or filter.",
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
                            "description": "Chroma where filter for deletion"
                        }
                    },
                    "required": []
                },
                handler=self._tool_delete,
                category="vector_db",
                source="plugin",
                plugin_name="ChromaDB",
                timeout_seconds=60
            ),
            AgentTool(
                name="chroma_collections",
                description="List all collections in the ChromaDB database.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=self._tool_collections,
                category="vector_db",
                source="plugin",
                plugin_name="ChromaDB",
                timeout_seconds=30
            )
        ]

    async def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for chroma_search"""
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
        """Tool handler for chroma_upsert"""
        ids = args.get("ids")
        vectors = args.get("vectors")
        metadata = args.get("metadata")
        documents = args.get("documents")

        result = await self.upsert(
            ids=ids,
            vectors=vectors,
            metadata=metadata,
            documents=documents
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

        return {
            "success": True,
            "collections": collections,
            "count": len(collections)
        }


def chroma(**kwargs) -> ChromaPlugin:
    """Create ChromaDB plugin with simplified interface."""
    return ChromaPlugin(**kwargs)
