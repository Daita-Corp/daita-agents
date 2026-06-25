"""
MongoDB plugin for Daita Agents.

Simple MongoDB connection and querying - no over-engineering.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse, quote
from .base import PluginContext
from .base_db import BaseDatabasePlugin
from .mongodb_extensions import (
    MONGODB_MANIFEST,
    MongoDBExecutor,
    mongodb_capabilities,
    mongodb_evidence_schemas,
    mongodb_operation_definitions,
    mongodb_tool_views,
)

if TYPE_CHECKING:
    from ..core.tools import LocalTool

logger = logging.getLogger(__name__)


class MongoDBPlugin(BaseDatabasePlugin):
    """
    MongoDB plugin for agents with standardized connection management.

    Inherits common database functionality from BaseDatabasePlugin.
    """

    manifest = MONGODB_MANIFEST

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "",
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_string: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize MongoDB connection.

        Args:
            host: MongoDB host
            port: MongoDB port
            database: Database name
            username: Optional username
            password: Optional password
            connection_string: Full connection string (overrides individual params)
            **kwargs: Additional motor configuration
        """
        if connection_string:
            self.connection_string = connection_string
            parsed = urlparse(connection_string)
            self.database_name = parsed.path.lstrip("/") if parsed.path else database
        else:
            self.database_name = database
            if username and password:
                self.connection_string = f"mongodb://{quote(username, safe='')}:{quote(password, safe='')}@{host}:{port}/{database}"
            else:
                self.connection_string = f"mongodb://{host}:{port}/{database}"

        self.client_config = {
            "maxPoolSize": kwargs.get("max_pool_size", 10),
            "minPoolSize": kwargs.get("min_pool_size", 1),
            "serverSelectionTimeoutMS": kwargs.get("server_timeout", 30000),
        }

        # Add TLS/SSL configuration if specified
        if kwargs.get("tls") or kwargs.get("ssl"):
            self.client_config["tls"] = True
        if kwargs.get("tlsAllowInvalidCertificates"):
            self.client_config["tlsAllowInvalidCertificates"] = True
        if kwargs.get("tlsCAFile"):
            self.client_config["tlsCAFile"] = kwargs.get("tlsCAFile")
        if kwargs.get("tlsCertificateKeyFile"):
            self.client_config["tlsCertificateKeyFile"] = kwargs.get(
                "tlsCertificateKeyFile"
            )

        # Initialize base class with all config
        super().__init__(
            host=host,
            port=port,
            database=database,
            username=username,
            connection_string=connection_string,
            **kwargs,
        )
        self._executor = MongoDBExecutor(self)

        logger.debug(f"MongoDB plugin configured for {host}:{port}/{database}")

    async def setup(self, context: PluginContext) -> None:
        """Set up the MongoDB connector for a runtime."""
        await self.connect()

    async def teardown(self) -> None:
        """Disconnect the MongoDB connector from a runtime."""
        await self.disconnect()

    # ------------------------------------------------------------------
    # Runtime extension declarations
    # ------------------------------------------------------------------

    def declare_capabilities(self):
        return mongodb_capabilities(self.read_only)

    def get_executors(self):
        return (self._executor,)

    def declare_evidence_schemas(self):
        return mongodb_evidence_schemas()

    def get_tool_views(self):
        return mongodb_tool_views(self.read_only)

    def _definition_for_capability(self, capability_id: str) -> dict:
        for definition in mongodb_operation_definitions(self.read_only):
            if definition["capability_id"] == capability_id:
                return definition
        raise KeyError(capability_id)

    def _definition_for_tool(self, tool_name: str) -> dict:
        for definition in mongodb_operation_definitions(self.read_only):
            if definition["tool_name"] == tool_name:
                return definition
        raise KeyError(tool_name)

    async def connect(self):
        """Connect to MongoDB."""
        if self._client is not None:
            return  # Already connected

        try:
            import motor.motor_asyncio

            self._client = motor.motor_asyncio.AsyncIOMotorClient(
                self.connection_string, **self.client_config
            )

            # Test connection
            await self._client.admin.command("ping")

            # Get database
            self._db = self._client[self.database_name]

            logger.info(f"Connected to MongoDB database '{self.database_name}'")
        except ImportError:
            self._handle_connection_error(
                ImportError(
                    "motor not installed. Install with: pip install 'daita-agents[mongodb]'"
                ),
                "connection",
            )
        except Exception as e:
            self._handle_connection_error(e, "connection")

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("Disconnected from MongoDB")

    async def find(
        self,
        collection: str,
        filter_doc: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find documents in a collection.

        Args:
            collection: Collection name
            filter_doc: Query filter (defaults to {} for all documents)
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            sort: Sort specification as list of (field, direction) tuples

        Returns:
            List of documents

        Example:
            # Find all users
            users = await db.find("users")

            # Find users in a specific city
            users = await db.find("users", {"city": "New York"})

            # Find with pagination and sorting
            users = await db.find("users",
                                filter_doc={"age": {"$gte": 18}},
                                sort=[("name", 1)],
                                limit=10,
                                skip=20)
        """
        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        filter_doc = filter_doc or {}

        cursor = self._db[collection].find(filter_doc)

        # Apply cursor modifications - these return the cursor object
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        # Convert cursor to list and handle ObjectId
        results = await cursor.to_list(length=None)

        # Convert ObjectId to string for JSON serialization
        for doc in results:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return results

    async def insert(self, collection: str, document: Dict[str, Any]) -> str:
        """
        Insert a single document.

        Args:
            collection: Collection name
            document: Document to insert

        Returns:
            Inserted document ID as string
        """
        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        result = await self._db[collection].insert_one(document)
        return str(result.inserted_id)

    async def insert_many(
        self, collection: str, documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Insert multiple documents.

        Args:
            collection: Collection name
            documents: List of documents to insert

        Returns:
            List of inserted document IDs as strings
        """
        if not documents:
            return []

        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        result = await self._db[collection].insert_many(documents)
        return [str(doc_id) for doc_id in result.inserted_ids]

    async def update(
        self,
        collection: str,
        filter_doc: Dict[str, Any],
        update_doc: Dict[str, Any],
        upsert: bool = False,
    ) -> Dict[str, Any]:
        """
        Update documents.

        Args:
            collection: Collection name
            filter_doc: Filter to match documents
            update_doc: Update operations (use $set, $inc, etc.)
            upsert: Create document if not found

        Returns:
            Update result info

        Example:
            result = await db.update("users",
                                   {"name": "John"},
                                   {"$set": {"age": 30}})
        """
        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        result = await self._db[collection].update_many(
            filter_doc, update_doc, upsert=upsert
        )

        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None,
        }

    async def delete(self, collection: str, filter_doc: Dict[str, Any]) -> int:
        """
        Delete documents.

        Args:
            collection: Collection name
            filter_doc: Filter to match documents to delete

        Returns:
            Number of deleted documents
        """
        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        result = await self._db[collection].delete_many(filter_doc)
        return result.deleted_count

    async def count(
        self, collection: str, filter_doc: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count documents in collection.

        Args:
            collection: Collection name
            filter_doc: Optional filter

        Returns:
            Document count
        """
        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        filter_doc = filter_doc or {}
        return await self._db[collection].count_documents(filter_doc)

    async def collections(self) -> List[str]:
        """List all collections in the database."""
        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        collections = await self._db.list_collection_names()
        return sorted(collections)

    async def aggregate(
        self, collection: str, pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run aggregation pipeline.

        Args:
            collection: Collection name
            pipeline: Aggregation pipeline

        Returns:
            Aggregation results

        Example:
            pipeline = [
                {"$match": {"status": "active"}},
                {"$group": {"_id": "$department", "count": {"$sum": 1}}}
            ]
            results = await db.aggregate("employees", pipeline)
        """
        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        cursor = self._db[collection].aggregate(pipeline)
        results = await cursor.to_list(length=None)

        # Convert ObjectId to string
        for doc in results:
            if "_id" in doc and hasattr(doc["_id"], "binary"):
                doc["_id"] = str(doc["_id"])

        return results

    async def _tool_find(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mongodb_find"""
        collection = args.get("collection")
        filter_doc = args.get("filter", {})
        limit = args.get("limit", 50)
        projection = args.get("projection")

        # Only auto-connect if client/db is None - allows manual mocking
        if self._client is None or self._db is None:
            await self.connect()

        cursor = self._db[collection].find(filter_doc, projection)
        cursor = cursor.limit(limit)
        results = await cursor.to_list(length=None)
        for doc in results:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return {"documents": results}

    async def _tool_insert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mongodb_insert"""
        collection = args.get("collection")
        document = args.get("document")

        inserted_id = await self.insert(collection, document)

        return {"inserted_id": inserted_id, "collection": collection}

    async def _tool_update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mongodb_update"""
        collection = args.get("collection")
        filter_doc = args.get("filter")
        update_doc = args.get("update")

        result = await self.update(collection, filter_doc, update_doc)

        return {
            "matched_count": result["matched_count"],
            "modified_count": result["modified_count"],
            "collection": collection,
        }

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mongodb_delete"""
        collection = args.get("collection")
        filter_doc = args.get("filter")

        deleted_count = await self.delete(collection, filter_doc)

        return {
            "deleted_count": deleted_count,
            "collection": collection,
        }

    async def _tool_list_collections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mongodb_list_collections"""
        collections = await self.collections()
        return {"collections": collections}

    async def _tool_aggregate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mongodb_aggregate"""
        collection = args.get("collection")
        pipeline = args.get("pipeline") or []

        # Inject a $limit safety cap if the pipeline has no $limit stage
        if not any("$limit" in stage for stage in pipeline):
            pipeline = list(pipeline) + [{"$limit": 200}]

        results = await self.aggregate(collection, pipeline)

        return {"results": results}

    async def _tool_count(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for mongodb_count"""
        collection = args.get("collection")
        filter_doc = args.get("filter")

        n = await self.count(collection, filter_doc)
        return {"collection": collection, "count": n}


def mongodb(**kwargs) -> MongoDBPlugin:
    """Create MongoDB plugin with simplified interface."""
    return MongoDBPlugin(**kwargs)
