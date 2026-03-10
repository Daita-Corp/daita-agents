"""
Mock external services for testing integrations.

Provides mock implementations of databases, APIs, and other external services.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncContextManager
from datetime import datetime, timezone
from contextlib import asynccontextmanager


class MockDatabaseConnection:
    """Mock database connection for testing database plugins."""

    def __init__(self, db_type: str = "postgresql"):
        self.db_type = db_type
        self.connected = False
        self.tables = {}
        self.query_history = []
        self.connection_count = 0

    async def connect(self) -> None:
        """Mock database connection."""
        await asyncio.sleep(0.01)  # Simulate connection delay
        self.connected = True
        self.connection_count += 1

    async def disconnect(self) -> None:
        """Mock database disconnection."""
        self.connected = False

    async def execute(self, query: str, params: Optional[List] = None) -> int:
        """Mock query execution."""
        self.query_history.append(
            {
                "query": query,
                "params": params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Simulate different types of queries
        query_lower = query.lower().strip()

        if query_lower.startswith("insert"):
            return 1  # 1 row affected
        elif query_lower.startswith("update"):
            return 2  # 2 rows affected
        elif query_lower.startswith("delete"):
            return 1  # 1 row affected
        else:
            return 0

    async def fetch_all(
        self, query: str, params: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """Mock query that returns results."""
        self.query_history.append(
            {
                "query": query,
                "params": params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Return mock data based on query
        if "users" in query.lower():
            return [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
            ]
        elif "orders" in query.lower():
            return [
                {"id": 101, "user_id": 1, "amount": 99.99, "status": "completed"},
                {"id": 102, "user_id": 2, "amount": 149.99, "status": "pending"},
            ]
        else:
            return [{"result": "mock_data", "count": 42}]

    async def fetch_one(
        self, query: str, params: Optional[List] = None
    ) -> Optional[Dict[str, Any]]:
        """Mock query that returns single result."""
        results = await self.fetch_all(query, params)
        return results[0] if results else None

    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get history of executed queries."""
        return self.query_history.copy()

    def clear_history(self) -> None:
        """Clear query history."""
        self.query_history.clear()


class MockS3Client:
    """Mock S3 client for testing cloud storage operations."""

    def __init__(self):
        self.buckets = {}
        self.objects = {}
        self.operation_history = []

    async def create_bucket(self, bucket: str) -> None:
        """Mock bucket creation."""
        self.buckets[bucket] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "objects": {},
        }
        self._log_operation("create_bucket", {"bucket": bucket})

    async def put_object(self, bucket: str, key: str, data: Any) -> None:
        """Mock object upload."""
        if bucket not in self.buckets:
            await self.create_bucket(bucket)

        self.buckets[bucket]["objects"][key] = {
            "data": data,
            "size": len(str(data)) if data else 0,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._log_operation("put_object", {"bucket": bucket, "key": key})

    async def get_object(self, bucket: str, key: str) -> Any:
        """Mock object download."""
        self._log_operation("get_object", {"bucket": bucket, "key": key})

        if bucket not in self.buckets:
            raise Exception(f"Bucket {bucket} does not exist")

        if key not in self.buckets[bucket]["objects"]:
            raise Exception(f"Object {key} does not exist in bucket {bucket}")

        return self.buckets[bucket]["objects"][key]["data"]

    async def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """Mock object listing."""
        self._log_operation("list_objects", {"bucket": bucket, "prefix": prefix})

        if bucket not in self.buckets:
            return []

        objects = list(self.buckets[bucket]["objects"].keys())
        if prefix:
            objects = [obj for obj in objects if obj.startswith(prefix)]

        return objects

    async def delete_object(self, bucket: str, key: str) -> None:
        """Mock object deletion."""
        if bucket in self.buckets and key in self.buckets[bucket]["objects"]:
            del self.buckets[bucket]["objects"][key]

        self._log_operation("delete_object", {"bucket": bucket, "key": key})

    def _log_operation(self, operation: str, params: Dict[str, Any]) -> None:
        """Log S3 operation."""
        self.operation_history.append(
            {
                "operation": operation,
                "params": params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of S3 operations."""
        return self.operation_history.copy()


class MockSlackClient:
    """Mock Slack client for testing Slack integrations."""

    def __init__(self):
        self.channels = {
            "#general": {"id": "C1234567890", "messages": []},
            "#alerts": {"id": "C0987654321", "messages": []},
        }
        self.message_history = []
        self.connected = False

    async def connect(self) -> None:
        """Mock Slack connection."""
        await asyncio.sleep(0.05)
        self.connected = True

    async def send_message(self, channel: str, text: str, **kwargs) -> Dict[str, Any]:
        """Mock message sending."""
        if not self.connected:
            raise Exception("Not connected to Slack")

        message = {
            "id": f"msg_{len(self.message_history)}",
            "channel": channel,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kwargs": kwargs,
        }

        # Store in channel if it exists
        if channel in self.channels:
            self.channels[channel]["messages"].append(message)

        self.message_history.append(message)

        return {"ok": True, "message": message}

    async def upload_file(
        self, channels: str, file_content: Any, filename: str
    ) -> Dict[str, Any]:
        """Mock file upload."""
        upload_info = {
            "id": f"file_{len(self.message_history)}",
            "channels": channels,
            "filename": filename,
            "size": len(str(file_content)) if file_content else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.message_history.append({"type": "file_upload", **upload_info})

        return {"ok": True, "file": upload_info}

    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get message history."""
        return self.message_history.copy()


class MockElasticsearchClient:
    """Mock Elasticsearch client for testing search operations."""

    def __init__(self):
        self.indices = {}
        self.operation_history = []

    async def create_index(self, index: str, mapping: Optional[Dict] = None) -> None:
        """Mock index creation."""
        self.indices[index] = {
            "mapping": mapping or {},
            "documents": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._log_operation("create_index", {"index": index, "mapping": mapping})

    async def index_document(
        self, index: str, doc_id: str, document: Dict[str, Any]
    ) -> None:
        """Mock document indexing."""
        if index not in self.indices:
            await self.create_index(index)

        self.indices[index]["documents"][doc_id] = {
            "source": document,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._log_operation("index_document", {"index": index, "id": doc_id})

    async def search(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Mock search operation."""
        self._log_operation("search", {"index": index, "query": query})

        if index not in self.indices:
            return {"hits": {"total": {"value": 0}, "hits": []}}

        # Simple mock search - return all documents
        docs = []
        for doc_id, doc_data in self.indices[index]["documents"].items():
            docs.append({"_id": doc_id, "_source": doc_data["source"], "_score": 1.0})

        return {
            "hits": {
                "total": {"value": len(docs)},
                "hits": docs[:10],  # Limit to 10 results
            }
        }

    async def delete_document(self, index: str, doc_id: str) -> None:
        """Mock document deletion."""
        if index in self.indices and doc_id in self.indices[index]["documents"]:
            del self.indices[index]["documents"][doc_id]

        self._log_operation("delete_document", {"index": index, "id": doc_id})

    def _log_operation(self, operation: str, params: Dict[str, Any]) -> None:
        """Log Elasticsearch operation."""
        self.operation_history.append(
            {
                "operation": operation,
                "params": params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


class MockRESTClient:
    """Mock REST client for testing HTTP operations."""

    def __init__(self, base_url: str = "http://mock-api.com"):
        self.base_url = base_url
        self.request_history = []
        self.response_mappings = {}
        self.default_response = {"status": "success", "data": "mock_response"}

    def set_response(self, method: str, path: str, response: Dict[str, Any]) -> None:
        """Set mock response for specific endpoint."""
        key = f"{method.upper()}:{path}"
        self.response_mappings[key] = response

    async def get(self, path: str, **kwargs) -> Dict[str, Any]:
        """Mock GET request."""
        return await self._make_request("GET", path, **kwargs)

    async def post(self, path: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Mock POST request."""
        return await self._make_request("POST", path, data=data, **kwargs)

    async def put(self, path: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Mock PUT request."""
        return await self._make_request("PUT", path, data=data, **kwargs)

    async def delete(self, path: str, **kwargs) -> Dict[str, Any]:
        """Mock DELETE request."""
        return await self._make_request("DELETE", path, **kwargs)

    async def _make_request(
        self, method: str, path: str, data: Any = None, **kwargs
    ) -> Dict[str, Any]:
        """Internal method to make mock request."""
        await asyncio.sleep(0.02)  # Simulate network delay

        request_info = {
            "method": method,
            "path": path,
            "data": data,
            "kwargs": kwargs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.request_history.append(request_info)

        # Check for mapped response
        key = f"{method}:{path}"
        if key in self.response_mappings:
            response = self.response_mappings[key].copy()
            if data:
                response["received_data"] = data
            return response

        # Return default response
        response = self.default_response.copy()
        if data:
            response["received_data"] = data
        return response

    def get_request_history(self) -> List[Dict[str, Any]]:
        """Get request history."""
        return self.request_history.copy()


# Context managers for easy testing
@asynccontextmanager
async def mock_database(
    db_type: str = "postgresql",
) -> AsyncContextManager[MockDatabaseConnection]:
    """Context manager for mock database connection."""
    db = MockDatabaseConnection(db_type)
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()


@asynccontextmanager
async def mock_s3_client() -> AsyncContextManager[MockS3Client]:
    """Context manager for mock S3 client."""
    client = MockS3Client()
    yield client


@asynccontextmanager
async def mock_slack_client() -> AsyncContextManager[MockSlackClient]:
    """Context manager for mock Slack client."""
    client = MockSlackClient()
    await client.connect()
    yield client
