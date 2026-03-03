"""
Elasticsearch plugin for Daita Agents.

Simple Elasticsearch search and indexing - no over-engineering.
"""
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
from .base import BasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)

class ElasticsearchPlugin(BasePlugin):
    """
    Simple Elasticsearch plugin for agents.
    
    Handles search, indexing, and analytics with focus system support and agent-specific features.
    """
    
    def __init__(
        self,
        hosts: Union[str, List[str]] = "localhost:9200",
        auth_method: str = "basic",
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key_id: Optional[str] = None,
        api_key: Optional[str] = None,
        ssl_fingerprint: Optional[str] = None,
        verify_certs: bool = True,
        ca_certs: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize Elasticsearch connection.
        
        Args:
            hosts: Elasticsearch host(s) - string or list of hosts
            auth_method: Authentication method ('basic', 'api_key', 'ssl')
            username: Username for basic auth
            password: Password for basic auth
            api_key_id: API key ID for API key auth
            api_key: API key for API key auth
            ssl_fingerprint: SSL certificate fingerprint
            verify_certs: Whether to verify SSL certificates
            ca_certs: Path to CA certificates file
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            **kwargs: Additional Elasticsearch client parameters
        """
        # Normalize hosts to list
        if isinstance(hosts, str):
            self.hosts = [hosts]
        else:
            self.hosts = hosts
        
        if not self.hosts:
            raise ValueError("At least one Elasticsearch host must be specified")
        
        self.auth_method = auth_method
        self.username = username
        self.password = password
        self.api_key_id = api_key_id
        self.api_key = api_key
        self.ssl_fingerprint = ssl_fingerprint
        self.verify_certs = verify_certs
        self.ca_certs = ca_certs
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Store additional config
        self.config = kwargs
        
        self._client = None
        self._cluster_info = None
        
        logger.debug(f"Elasticsearch plugin configured for hosts: {self.hosts}")
    
    async def connect(self):
        """Initialize Elasticsearch client and test connection."""
        if self._client is not None:
            return  # Already connected
        
        try:
            from elasticsearch import AsyncElasticsearch
            from elasticsearch.exceptions import ConnectionError, AuthenticationException
            
            # Prepare authentication
            auth_config = {}
            
            if self.auth_method == "basic" and self.username and self.password:
                auth_config['basic_auth'] = (self.username, self.password)
            elif self.auth_method == "api_key" and self.api_key_id and self.api_key:
                auth_config['api_key'] = (self.api_key_id, self.api_key)
            elif self.auth_method == "ssl" and self.ssl_fingerprint:
                auth_config['ssl_assert_fingerprint'] = self.ssl_fingerprint
            
            # Prepare SSL config
            ssl_config = {
                'verify_certs': self.verify_certs
            }
            if self.ca_certs:
                ssl_config['ca_certs'] = self.ca_certs
            
            # Create client
            client_config = {
                'hosts': self.hosts,
                'timeout': self.timeout,
                'max_retries': self.max_retries,
                **auth_config,
                **ssl_config,
                **self.config
            }
            
            self._client = AsyncElasticsearch(**client_config)
            
            # Test connection
            try:
                cluster_info = await self._client.info()
                self._cluster_info = {
                    'cluster_name': cluster_info.get('cluster_name'),
                    'version': cluster_info.get('version', {}).get('number'),
                    'lucene_version': cluster_info.get('version', {}).get('lucene_version'),
                    'tagline': cluster_info.get('tagline')
                }
                
                logger.info(f"Connected to Elasticsearch cluster '{self._cluster_info['cluster_name']}' "
                          f"(version {self._cluster_info['version']})")
                
            except AuthenticationException:
                raise RuntimeError("Elasticsearch authentication failed. Please check your credentials.")
            except ConnectionError as e:
                raise RuntimeError(f"Failed to connect to Elasticsearch: {e}")
                
        except ImportError:
            raise ImportError("elasticsearch not installed. Install with: pip install 'daita-agents[elasticsearch]'")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Elasticsearch client: {e}")
    
    async def disconnect(self):
        """Close Elasticsearch connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._cluster_info = None
            logger.info("Disconnected from Elasticsearch")
    
    async def search(
        self,
        index: str,
        query: Optional[Dict[str, Any]] = None,
        focus: Optional[List[str]] = None,
        size: int = 100,
        from_: int = 0,
        sort: Optional[List[Dict[str, Any]]] = None,
        aggregations: Optional[Dict[str, Any]] = None,
        scroll: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search documents in Elasticsearch.
        
        Args:
            index: Index name or pattern
            query: Elasticsearch query DSL (defaults to match_all)
            focus: List of fields to return (source filtering)
            size: Number of documents to return
            from_: Starting offset for pagination
            sort: Sort configuration
            aggregations: Aggregations to compute
            scroll: Scroll context keepalive time
            **kwargs: Additional search parameters
            
        Returns:
            Search results with hits, aggregations, and metadata
            
        Example:
            results = await es.search("logs", {"match": {"level": "ERROR"}}, focus=["timestamp", "message"])
        """
        if self._client is None:
            await self.connect()
        
        try:
            # Default to match_all if no query provided
            if query is None:
                query = {"match_all": {}}
            
            # Prepare search body
            search_body = {
                "query": query,
                "size": size,
                "from": from_
            }
            
            # Apply focus system (source filtering)
            if focus:
                search_body["_source"] = focus
            
            # Add sort if specified
            if sort:
                search_body["sort"] = sort
            
            # Add aggregations if specified
            if aggregations:
                search_body["aggs"] = aggregations
            
            # Prepare search parameters
            search_params = {
                "index": index,
                "body": search_body,
                **kwargs
            }
            
            # Add scroll if specified
            if scroll:
                search_params["scroll"] = scroll
            
            # Execute search
            response = await self._client.search(**search_params)
            
            # Format response
            result = {
                "hits": {
                    "total": response["hits"]["total"],
                    "max_score": response["hits"].get("max_score"),
                    "documents": [hit["_source"] for hit in response["hits"]["hits"]]
                },
                "took": response["took"],
                "timed_out": response["timed_out"],
                "_scroll_id": response.get("_scroll_id")
            }
            
            # Add aggregations if present
            if "aggregations" in response:
                result["aggregations"] = response["aggregations"]
            
            logger.info(f"Search completed: {result['hits']['total']['value']} hits in {result['took']}ms")
            return result
            
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            raise RuntimeError(f"Elasticsearch search failed: {e}")
    
    async def index_document(
        self,
        index: str,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        refresh: Union[bool, str] = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Index a single document.
        
        Args:
            index: Index name
            document: Document to index
            doc_id: Document ID (auto-generated if not provided)
            refresh: Whether to refresh the index
            **kwargs: Additional indexing parameters
            
        Returns:
            Indexing result with document ID and metadata
            
        Example:
            result = await es.index_document("logs", {"level": "INFO", "message": "Test"})
        """
        if self._client is None:
            await self.connect()
        
        try:
            # Prepare index parameters
            index_params = {
                "index": index,
                "body": document,
                "refresh": refresh,
                **kwargs
            }
            
            if doc_id:
                index_params["id"] = doc_id
            
            # Index document
            response = await self._client.index(**index_params)
            
            result = {
                "id": response["_id"],
                "index": response["_index"],
                "version": response["_version"],
                "result": response["result"],
                "created": response["result"] == "created"
            }
            
            logger.debug(f"Indexed document {result['id']} in index {index}")
            return result
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            raise RuntimeError(f"Elasticsearch index_document failed: {e}")
    
    async def bulk_index(
        self,
        index: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 1000,
        refresh: Union[bool, str] = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bulk index multiple documents.
        
        Args:
            index: Index name
            documents: List of documents to index
            batch_size: Number of documents per batch
            refresh: Whether to refresh the index
            **kwargs: Additional bulk parameters
            
        Returns:
            Bulk indexing results with success/error statistics
            
        Example:
            result = await es.bulk_index("logs", log_documents, batch_size=5000)
        """
        if self._client is None:
            await self.connect()
        
        if not documents:
            return {"indexed": 0, "errors": 0, "took": 0}
        
        try:
            from elasticsearch.helpers import async_bulk, BulkIndexError
            
            # Prepare documents for bulk indexing
            def doc_generator():
                for doc in documents:
                    yield {
                        "_index": index,
                        "_source": doc,
                        **kwargs
                    }
            
            # Perform bulk indexing
            start_time = datetime.now()
            
            try:
                success_count, failed_docs = await async_bulk(
                    self._client,
                    doc_generator(),
                    chunk_size=batch_size,
                    refresh=refresh,
                    raise_on_error=False,
                    raise_on_exception=False
                )
                
                end_time = datetime.now()
                duration_ms = (end_time - start_time).total_seconds() * 1000
                
                result = {
                    "indexed": success_count,
                    "errors": len(failed_docs),
                    "total": len(documents),
                    "took": int(duration_ms),
                    "failed_docs": failed_docs if failed_docs else []
                }
                
                logger.info(f"Bulk indexed {success_count}/{len(documents)} documents in {duration_ms:.1f}ms")
                
                if failed_docs:
                    logger.warning(f"Failed to index {len(failed_docs)} documents")
                
                return result
                
            except BulkIndexError as e:
                logger.error(f"Bulk indexing error: {e}")
                raise RuntimeError(f"Bulk indexing failed: {e}")
                
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise RuntimeError(f"Elasticsearch bulk_index failed: {e}")
    
    async def delete_document(
        self,
        index: str,
        doc_id: str,
        refresh: Union[bool, str] = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delete a document by ID.
        
        Args:
            index: Index name
            doc_id: Document ID
            refresh: Whether to refresh the index
            **kwargs: Additional delete parameters
            
        Returns:
            Delete result
            
        Example:
            result = await es.delete_document("logs", "doc123")
        """
        if self._client is None:
            await self.connect()
        
        try:
            response = await self._client.delete(
                index=index,
                id=doc_id,
                refresh=refresh,
                **kwargs
            )
            
            result = {
                "id": response["_id"],
                "index": response["_index"],
                "version": response["_version"],
                "result": response["result"],
                "deleted": response["result"] == "deleted"
            }
            
            logger.debug(f"Deleted document {doc_id} from index {index}")
            return result
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            raise RuntimeError(f"Elasticsearch delete_document failed: {e}")
    
    async def create_index(
        self,
        index: str,
        mapping: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create an index with optional mapping and settings.
        
        Args:
            index: Index name
            mapping: Index mapping definition
            settings: Index settings
            **kwargs: Additional parameters
            
        Returns:
            Index creation result
            
        Example:
            result = await es.create_index("logs", mapping={"properties": {"timestamp": {"type": "date"}}})
        """
        if self._client is None:
            await self.connect()
        
        try:
            # Prepare index body
            body = {}
            if mapping:
                body["mappings"] = mapping
            if settings:
                body["settings"] = settings
            
            # Create index
            response = await self._client.indices.create(
                index=index,
                body=body if body else None,
                **kwargs
            )
            
            result = {
                "acknowledged": response["acknowledged"],
                "shards_acknowledged": response.get("shards_acknowledged", False),
                "index": response.get("index", index)
            }
            
            logger.info(f"Created index {index}")
            return result
            
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            raise RuntimeError(f"Elasticsearch create_index failed: {e}")
    
    async def get_mapping(self, index: str) -> Dict[str, Any]:
        """
        Get index mapping.
        
        Args:
            index: Index name
            
        Returns:
            Index mapping
            
        Example:
            mapping = await es.get_mapping("logs")
        """
        if self._client is None:
            await self.connect()
        
        try:
            response = await self._client.indices.get_mapping(index=index)
            
            # Extract mapping for the index
            if index in response:
                return response[index].get("mappings", {})
            else:
                # Handle index patterns
                return list(response.values())[0].get("mappings", {}) if response else {}
            
        except Exception as e:
            logger.error(f"Get mapping failed: {e}")
            raise RuntimeError(f"Elasticsearch get_mapping failed: {e}")
    
    async def search_agent_logs(
        self,
        index: str,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        time_range: Optional[Dict[str, str]] = None,
        focus: Optional[List[str]] = None,
        size: int = 100
    ) -> Dict[str, Any]:
        """
        Search agent execution logs with intelligent filtering.
        
        Args:
            index: Index name for agent logs
            agent_id: Filter by specific agent ID
            status: Filter by execution status (success, error, etc.)
            time_range: Time range filter {"gte": "now-1h", "lte": "now"}
            focus: Fields to focus on
            size: Number of results to return
            
        Returns:
            Filtered agent logs
            
        Example:
            logs = await es.search_agent_logs("agent_logs", agent_id="data_processor", status="error")
        """
        # Build query
        must_clauses = []
        
        if agent_id:
            must_clauses.append({"term": {"agent_id": agent_id}})
        
        if status:
            must_clauses.append({"term": {"status": status}})
        
        if time_range:
            must_clauses.append({"range": {"timestamp": time_range}})
        
        # Default query if no filters
        if not must_clauses:
            query = {"match_all": {}}
        else:
            query = {"bool": {"must": must_clauses}}
        
        # Default focus for agent logs
        if focus is None:
            focus = ["timestamp", "agent_id", "status", "duration_ms", "message", "error"]
        
        # Sort by timestamp descending
        sort = [{"timestamp": {"order": "desc"}}]
        
        return await self.search(
            index=index,
            query=query,
            focus=focus,
            size=size,
            sort=sort
        )
    
    async def index_agent_results(
        self,
        index: str,
        agent_results: Dict[str, Any],
        agent_id: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index agent execution results with structured schema.
        
        Args:
            index: Index name for agent results
            agent_results: Agent execution results
            agent_id: Agent identifier
            timestamp: Execution timestamp (auto-generated if not provided)
            
        Returns:
            Indexing result
            
        Example:
            result = await es.index_agent_results("agent_outputs", agent_results, "data_processor")
        """
        # Add metadata to document
        from datetime import timezone
        now = datetime.now(timezone.utc).isoformat()
        document = {
            "agent_id": agent_id,
            "timestamp": timestamp or now,
            "indexed_at": now,
            **agent_results
        }
        
        return await self.index_document(index, document)
    
    async def analyze_performance(
        self,
        index: str,
        metric_field: str = "duration_ms",
        group_by: Optional[str] = None,
        time_range: Optional[Dict[str, str]] = None,
        focus: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze agent performance with aggregations.
        
        Args:
            index: Index name
            metric_field: Field to analyze (e.g., duration_ms)
            group_by: Field to group by (e.g., agent_id)
            time_range: Time range filter
            focus: Fields to focus on for individual documents
            
        Returns:
            Performance analytics with aggregations
            
        Example:
            analytics = await es.analyze_performance("agent_metrics", group_by="agent_id")
        """
        # Build query
        query = {"match_all": {}}
        if time_range:
            query = {
                "bool": {
                    "must": [{"range": {"timestamp": time_range}}]
                }
            }
        
        # Build aggregations
        aggregations = {
            "performance_stats": {
                "stats": {"field": metric_field}
            },
            "performance_percentiles": {
                "percentiles": {"field": metric_field, "percents": [50, 95, 99]}
            }
        }
        
        # Group by field if specified
        if group_by:
            aggregations["groups"] = {
                "terms": {"field": group_by},
                "aggs": {
                    "group_stats": {"stats": {"field": metric_field}},
                    "group_percentiles": {"percentiles": {"field": metric_field, "percents": [50, 95, 99]}}
                }
            }
        
        return await self.search(
            index=index,
            query=query,
            focus=focus,
            size=0,  # Only return aggregations
            aggregations=aggregations
        )
    
    async def get_cluster_health(self) -> Dict[str, Any]:
        """
        Get Elasticsearch cluster health information.
        
        Returns:
            Cluster health status and metrics
        """
        if self._client is None:
            await self.connect()
        
        try:
            health = await self._client.cluster.health()
            
            result = {
                "cluster_name": health["cluster_name"],
                "status": health["status"],
                "timed_out": health["timed_out"],
                "number_of_nodes": health["number_of_nodes"],
                "number_of_data_nodes": health["number_of_data_nodes"],
                "active_primary_shards": health["active_primary_shards"],
                "active_shards": health["active_shards"],
                "relocating_shards": health["relocating_shards"],
                "initializing_shards": health["initializing_shards"],
                "unassigned_shards": health["unassigned_shards"],
                "cluster_info": self._cluster_info
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Get cluster health failed: {e}")
            raise RuntimeError(f"Elasticsearch get_cluster_health failed: {e}")

    def get_tools(self) -> List['AgentTool']:
        """
        Expose Elasticsearch operations as agent tools.

        Returns:
            List of AgentTool instances for Elasticsearch operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="search_elasticsearch",
                description="Search an Elasticsearch index using query DSL.",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "description": "Index name or pattern to search in"
                        },
                        "query": {
                            "type": "object",
                            "description": "Elasticsearch query DSL object"
                        },
                        "size": {
                            "type": "integer",
                            "description": "Number of results to return (default: 50)"
                        }
                    },
                    "required": ["index", "query"]
                },
                handler=self._tool_search,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=60
            ),
            AgentTool(
                name="index_document",
                description="Index a single document into an Elasticsearch index.",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "description": "Index name"
                        },
                        "document": {
                            "type": "object",
                            "description": "Document data as JSON object"
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Optional document ID (auto-generated if not provided)"
                        }
                    },
                    "required": ["index", "document"]
                },
                handler=self._tool_index,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=30
            ),
            AgentTool(
                name="get_index_mapping",
                description="Get the field mapping (schema) for an Elasticsearch index.",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "description": "Index name"
                        }
                    },
                    "required": ["index"]
                },
                handler=self._tool_get_mapping,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=30
            ),
            AgentTool(
                name="bulk_index_documents",
                description="Bulk index multiple documents into Elasticsearch. Primary path for batch ingestion.",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "description": "Index name"
                        },
                        "documents": {
                            "type": "array",
                            "description": "List of document objects to index",
                            "items": {"type": "object"}
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "Documents per batch (default: 1000)"
                        }
                    },
                    "required": ["index", "documents"]
                },
                handler=self._tool_bulk_index,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=120
            ),
            AgentTool(
                name="delete_es_document",
                description="Delete a document from Elasticsearch by ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "description": "Index name"
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID to delete"
                        }
                    },
                    "required": ["index", "doc_id"]
                },
                handler=self._tool_delete_document,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=30
            ),
            AgentTool(
                name="create_es_index",
                description="Create an Elasticsearch index with optional mapping and settings.",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "description": "Index name to create"
                        },
                        "mapping": {
                            "type": "object",
                            "description": "Index mapping (e.g., {\"properties\": {\"timestamp\": {\"type\": \"date\"}}})"
                        },
                        "settings": {
                            "type": "object",
                            "description": "Index settings (e.g., {\"number_of_shards\": 1})"
                        }
                    },
                    "required": ["index"]
                },
                handler=self._tool_create_index,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=30
            ),
            AgentTool(
                name="search_agent_logs",
                description="Search agent execution logs in Elasticsearch with filtering by agent ID, status, and time range.",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "description": "Index name for agent logs"
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Filter by agent ID"
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by execution status (e.g., success, error)"
                        },
                        "size": {
                            "type": "integer",
                            "description": "Max results to return (default: 50)"
                        }
                    },
                    "required": ["index"]
                },
                handler=self._tool_search_agent_logs,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=60
            ),
            AgentTool(
                name="analyze_es_performance",
                description="Analyze performance metrics in Elasticsearch using aggregations (stats, percentiles).",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "description": "Index name"
                        },
                        "metric_field": {
                            "type": "string",
                            "description": "Numeric field to analyze (default: duration_ms)"
                        },
                        "group_by": {
                            "type": "string",
                            "description": "Optional field to group results by (e.g., agent_id)"
                        }
                    },
                    "required": ["index"]
                },
                handler=self._tool_analyze_performance,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=60
            ),
            AgentTool(
                name="get_es_cluster_health",
                description="Get Elasticsearch cluster health status and shard metrics.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=self._tool_cluster_health,
                category="search",
                source="plugin",
                plugin_name="Elasticsearch",
                timeout_seconds=30
            )
        ]

    async def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for search_elasticsearch"""
        index = args.get("index")
        query = args.get("query")
        size = args.get("size", 50)

        results = await self.search(index=index, query=query, size=size)

        return {
            "success": True,
            "documents": results["hits"]["documents"],
            "total": results["hits"]["total"]["value"],
            "max_score": results["hits"]["max_score"],
            "took_ms": results["took"]
        }

    async def _tool_index(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for index_document"""
        index = args.get("index")
        document = args.get("document")
        doc_id = args.get("doc_id")

        result = await self.index_document(index=index, document=document, doc_id=doc_id)

        return {
            "success": True,
            "id": result["id"],
            "index": result["index"],
            "created": result["created"],
            "version": result["version"]
        }

    async def _tool_get_mapping(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_index_mapping"""
        index = args.get("index")

        mapping = await self.get_mapping(index)

        return {
            "success": True,
            "index": index,
            "mapping": mapping
        }

    async def _tool_bulk_index(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for bulk_index_documents"""
        index = args.get("index")
        documents = args.get("documents")
        batch_size = args.get("batch_size", 1000)

        result = await self.bulk_index(index=index, documents=documents, batch_size=batch_size)

        return {
            "success": True,
            "indexed": result["indexed"],
            "errors": result["errors"],
            "total": result["total"],
            "took_ms": result["took"]
        }

    async def _tool_delete_document(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for delete_es_document"""
        index = args.get("index")
        doc_id = args.get("doc_id")

        result = await self.delete_document(index=index, doc_id=doc_id)

        return {
            "success": True,
            "id": result["id"],
            "deleted": result["deleted"]
        }

    async def _tool_create_index(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for create_es_index"""
        index = args.get("index")
        mapping = args.get("mapping")
        settings = args.get("settings")

        result = await self.create_index(index=index, mapping=mapping, settings=settings)

        return {
            "success": True,
            "acknowledged": result["acknowledged"],
            "index": result["index"]
        }

    async def _tool_search_agent_logs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for search_agent_logs"""
        index = args.get("index")
        agent_id = args.get("agent_id")
        status = args.get("status")
        size = args.get("size", 50)

        results = await self.search_agent_logs(
            index=index, agent_id=agent_id, status=status, size=size
        )

        return {
            "success": True,
            "documents": results["hits"]["documents"],
            "total": results["hits"]["total"]["value"],
            "took_ms": results["took"]
        }

    async def _tool_analyze_performance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for analyze_es_performance"""
        index = args.get("index")
        metric_field = args.get("metric_field", "duration_ms")
        group_by = args.get("group_by")

        results = await self.analyze_performance(
            index=index, metric_field=metric_field, group_by=group_by
        )

        return {
            "success": True,
            "aggregations": results.get("aggregations", {}),
            "took_ms": results["took"]
        }

    async def _tool_cluster_health(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_es_cluster_health"""
        health = await self.get_cluster_health()

        return {
            "success": True,
            **health
        }

    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


def elasticsearch(**kwargs) -> ElasticsearchPlugin:
    """Create Elasticsearch plugin with simplified interface."""
    return ElasticsearchPlugin(**kwargs)