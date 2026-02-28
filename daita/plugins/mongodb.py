"""
MongoDB plugin for Daita Agents.

Simple MongoDB connection and querying - no over-engineering.
"""
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from .base_db import BaseDatabasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)

class MongoDBPlugin(BaseDatabasePlugin):
    """
    MongoDB plugin for agents with standardized connection management.
    
    Inherits common database functionality from BaseDatabasePlugin.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "",
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_string: Optional[str] = None,
        **kwargs
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
        else:
            if username and password:
                self.connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}"
            else:
                self.connection_string = f"mongodb://{host}:{port}/{database}"
        
        self.database_name = database
        
        self.client_config = {
            'maxPoolSize': kwargs.get('max_pool_size', 10),
            'minPoolSize': kwargs.get('min_pool_size', 1),
            'serverSelectionTimeoutMS': kwargs.get('server_timeout', 30000),
        }

        # Add TLS/SSL configuration if specified
        if kwargs.get('tls') or kwargs.get('ssl'):
            self.client_config['tls'] = True
        if kwargs.get('tlsAllowInvalidCertificates'):
            self.client_config['tlsAllowInvalidCertificates'] = True
        if kwargs.get('tlsCAFile'):
            self.client_config['tlsCAFile'] = kwargs.get('tlsCAFile')
        if kwargs.get('tlsCertificateKeyFile'):
            self.client_config['tlsCertificateKeyFile'] = kwargs.get('tlsCertificateKeyFile')
        
        # Initialize base class with all config
        super().__init__(
            host=host, port=port, database=database,
            username=username, connection_string=connection_string,
            **kwargs
        )
        
        logger.debug(f"MongoDB plugin configured for {host}:{port}/{database}")
    
    async def connect(self):
        """Connect to MongoDB."""
        if self._client is not None:
            return  # Already connected
        
        try:
            import motor.motor_asyncio
            
            self._client = motor.motor_asyncio.AsyncIOMotorClient(
                self.connection_string,
                **self.client_config
            )
            
            # Test connection
            await self._client.admin.command('ping')
            
            # Get database
            self._db = self._client[self.database_name]
            
            logger.info(f"Connected to MongoDB database '{self.database_name}'")
        except ImportError:
            self._handle_connection_error(
                ImportError("motor not installed. Run: pip install motor"),
                "connection"
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
        sort: Optional[List[tuple]] = None
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
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
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
    
    async def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
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
        upsert: bool = False
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
            filter_doc, 
            update_doc, 
            upsert=upsert
        )
        
        return {
            'matched_count': result.matched_count,
            'modified_count': result.modified_count,
            'upserted_id': str(result.upserted_id) if result.upserted_id else None
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
    
    async def count(self, collection: str, filter_doc: Optional[Dict[str, Any]] = None) -> int:
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
    
    async def aggregate(self, collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            if '_id' in doc and hasattr(doc['_id'], 'binary'):
                doc['_id'] = str(doc['_id'])

        return results

    def get_tools(self) -> List['AgentTool']:
        """
        Expose MongoDB operations as agent tools.

        Returns:
            List of AgentTool instances for database operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="find_documents",
                description="Find documents in a MongoDB collection. Returns matching documents as a list.",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name to search in"
                        },
                        "filter": {
                            "type": "object",
                            "description": "MongoDB query filter (e.g., {\"status\": \"active\"}). Empty object {} matches all documents."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return"
                        }
                    },
                    "required": ["collection"]
                },
                handler=self._tool_find,
                category="database",
                source="plugin",
                plugin_name="MongoDB",
                timeout_seconds=60
            ),
            AgentTool(
                name="insert_document",
                description="Insert a single document into a MongoDB collection. Returns the inserted document ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name to insert into"
                        },
                        "document": {
                            "type": "object",
                            "description": "Document data to insert as JSON object"
                        }
                    },
                    "required": ["collection", "document"]
                },
                handler=self._tool_insert,
                category="database",
                source="plugin",
                plugin_name="MongoDB",
                timeout_seconds=30
            ),
            AgentTool(
                name="update_documents",
                description="Update documents in a MongoDB collection. Returns the count of matched and modified documents.",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name"
                        },
                        "filter": {
                            "type": "object",
                            "description": "Query filter to match documents to update"
                        },
                        "update": {
                            "type": "object",
                            "description": "Update operations (e.g., {\"$set\": {\"status\": \"completed\"}})"
                        }
                    },
                    "required": ["collection", "filter", "update"]
                },
                handler=self._tool_update,
                category="database",
                source="plugin",
                plugin_name="MongoDB",
                timeout_seconds=60
            ),
            AgentTool(
                name="delete_documents",
                description="Delete documents from a MongoDB collection. Returns the count of deleted documents.",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name"
                        },
                        "filter": {
                            "type": "object",
                            "description": "Query filter to match documents to delete"
                        }
                    },
                    "required": ["collection", "filter"]
                },
                handler=self._tool_delete,
                category="database",
                source="plugin",
                plugin_name="MongoDB",
                timeout_seconds=60
            ),
            AgentTool(
                name="list_collections",
                description="List all collections in the MongoDB database",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=self._tool_list_collections,
                category="database",
                source="plugin",
                plugin_name="MongoDB",
                timeout_seconds=30
            )
        ]

    async def _tool_find(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for find_documents"""
        collection = args.get("collection")
        filter_doc = args.get("filter", {})
        limit = args.get("limit")

        results = await self.find(
            collection=collection,
            filter_doc=filter_doc,
            limit=limit
        )

        return {
            "success": True,
            "documents": results,
            "count": len(results)
        }

    async def _tool_insert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for insert_document"""
        collection = args.get("collection")
        document = args.get("document")

        inserted_id = await self.insert(collection, document)

        return {
            "success": True,
            "inserted_id": inserted_id,
            "collection": collection
        }

    async def _tool_update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for update_documents"""
        collection = args.get("collection")
        filter_doc = args.get("filter")
        update_doc = args.get("update")

        result = await self.update(collection, filter_doc, update_doc)

        return {
            "success": True,
            "matched_count": result["matched_count"],
            "modified_count": result["modified_count"],
            "collection": collection
        }

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for delete_documents"""
        collection = args.get("collection")
        filter_doc = args.get("filter")

        deleted_count = await self.delete(collection, filter_doc)

        return {
            "success": True,
            "deleted_count": deleted_count,
            "collection": collection
        }

    async def _tool_list_collections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_collections"""
        collections = await self.collections()

        return {
            "success": True,
            "collections": collections,
            "count": len(collections)
        }


def mongodb(**kwargs) -> MongoDBPlugin:
    """Create MongoDB plugin with simplified interface."""
    return MongoDBPlugin(**kwargs)