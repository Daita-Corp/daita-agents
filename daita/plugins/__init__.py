"""
Plugin system for Daita Agents.

This module provides database, vector database, API, cloud storage, search, collaboration, and MCP integrations:

Database Plugins:
- PostgreSQL plugin for async database operations (with pgvector support)
- MySQL plugin for async database operations
- MongoDB plugin for async document database operations
- Snowflake plugin for cloud data warehouse operations
- SQLite plugin for local file-based and in-memory database operations

Vector Database Plugins:
- Pinecone plugin for managed cloud vector database
- ChromaDB plugin for local/embedded vector database
- Qdrant plugin for self-hosted vector database
- PostgreSQL with pgvector extension for vector operations

Integration Plugins:
- REST API plugin for HTTP client functionality
- AWS S3 plugin for cloud object storage operations
- Google Drive plugin for file reading, searching, and organizing across Google Drive
- Slack plugin for team collaboration and notifications
- Email plugin for IMAP/SMTP email operations (Gmail, Outlook, Yahoo, etc.)
- Elasticsearch plugin for search and analytics
- WebSearch plugin for AI-optimized web search with Tavily
- MCP plugin for Model Context Protocol server integration
- Memory plugin for persistent semantic memory (local + cloud)

Knowledge & Orchestration Plugins:
- CatalogPlugin for schema discovery and metadata management
- LineagePlugin for data lineage tracking and impact analysis
- Neo4jPlugin for graph database operations with Cypher queries
- OrchestratorPlugin for multi-agent coordination and task routing

All plugins follow async patterns and provide simple, clean interfaces
without over-engineering.

Usage:
    ```python
    from daita.plugins import postgresql, mysql, mongodb, snowflake
    from daita.plugins import pinecone, chroma, qdrant
    from daita.plugins import rest, s3, slack, email, elasticsearch, websearch, mcp
    from daita import Agent

    # Database plugins
    async with postgresql(host="localhost", database="mydb") as db:
        results = await db.query("SELECT * FROM users")

    async with sqlite(path="./data.db") as db:
        await db.execute_script("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, msg TEXT)")
        results = await db.query("SELECT * FROM logs WHERE id > ?", [0])

    # Vector database plugins
    async with pinecone(api_key="...", index="docs") as db:
        results = await db.query(vector=[0.1, 0.2, ...], top_k=5)

    async with chroma(path="./vectors", collection="docs") as db:
        results = await db.query(vector=[0.1, 0.2, ...], top_k=5)

    async with qdrant(url="http://localhost:6333", collection="docs") as db:
        results = await db.query(vector=[0.1, 0.2, ...], top_k=5)

    # PostgreSQL with pgvector
    async with postgresql(host="localhost", database="mydb") as db:
        results = await db.vector_search(
            table="documents",
            vector_column="embedding",
            query_vector=[0.1, 0.2, ...],
            top_k=5
        )

    # REST API plugin
    async with rest(base_url="https://api.example.com") as api:
        data = await api.get("/users")

    # S3 plugin
    async with s3(bucket="my-bucket", region="us-west-2") as storage:
        data = await storage.get_object("data/file.csv", format="pandas")

    # Google Drive plugin
    async with google_drive(credentials_path="service_account.json") as drive:
        results = await drive.search("Q4 report", file_type="spreadsheet")
        content = await drive.read(results[0]["id"])  # auto-detects format

    # Slack plugin
    async with slack(token="xoxb-token") as slack_client:
        await slack_client.send_agent_summary("#alerts", agent_results)

    # Email plugin
    async with email(
        email_address="user@gmail.com",
        password="app_password",
        provider="gmail"
    ) as email_client:
        emails = await email_client.list_emails(limit=5, unread_only=True)
        await email_client.send_email(to="recipient@example.com", subject="Hello", body="Message")

    # Elasticsearch plugin
    async with elasticsearch(hosts=["localhost:9200"]) as es:
        results = await es.search("logs", {"match": {"level": "ERROR"}}, focus=["timestamp", "message"])

    # WebSearch plugin
    async with websearch(api_key="tvly-xxxxx") as search:
        results = await search.search("Python async best practices", max_results=5)
        print(f"Answer: {results['answer']}")

    # MCP plugin with agent integration
    agent = Agent(
        name="file_analyzer",
        mcp=mcp.server(command="uvx", args=["mcp-server-filesystem", "/data"])
    )
    result = await agent.process("Read report.csv and calculate totals")

    # Catalog plugin - schema discovery
    catalog_plugin = catalog()
    schema = await catalog_plugin.discover_postgres(
        connection_string="postgresql://localhost/mydb",
        persist=False  # Just return data, don't persist
    )
    print(f"Found {schema['schema']['table_count']} tables")

    # Lineage plugin - data flow tracking
    lineage_plugin = lineage()
    await lineage_plugin.register_flow(
        source_id="table:raw_data",
        target_id="table:processed_data",
        transformation="Clean and normalize"
    )
    impact = await lineage_plugin.analyze_impact("table:raw_data")

    # Neo4j plugin - graph database operations
    async with neo4j(uri="bolt://localhost:7687", auth=("neo4j", "password")) as graph:
        result = await graph.query("MATCH (n:Person) RETURN n LIMIT 10")
        await graph.create_node("Person", {"name": "Alice", "age": 30})

    # Orchestrator plugin - multi-agent coordination
    orchestrator_plugin = orchestrator(agents={"analyst": analyst_agent, "reporter": reporter_agent})
    result = await orchestrator_plugin.route_task("Analyze Q4 sales and generate report")
    ```
"""

# Database plugins
from .postgresql import PostgreSQLPlugin, postgresql
from .mysql import MySQLPlugin, mysql
from .mongodb import MongoDBPlugin, mongodb
from .snowflake import SnowflakePlugin, snowflake
from .sqlite import SQLitePlugin, sqlite

# Vector database plugins
from .pinecone import PineconePlugin, pinecone
from .chroma import ChromaPlugin, chroma
from .qdrant import QdrantPlugin, qdrant

# API plugins
from .rest import RESTPlugin, rest

# Cloud storage plugins
from .s3 import S3Plugin, s3
from .google_drive import GoogleDrivePlugin, google_drive

# Collaboration plugins
from .slack import SlackPlugin, slack

# Email plugins
from .email import EmailPlugin, email

# Search and analytics plugins
from .elasticsearch import ElasticsearchPlugin, elasticsearch

# Web search plugins
from .websearch import WebSearchPlugin, websearch

# Messaging plugins
from .redis_messaging import RedisMessagingPlugin, redis_messaging

# MCP plugin
from . import mcp

# Memory plugin
from .memory import MemoryPlugin

# Knowledge & Orchestration plugins
from .catalog import CatalogPlugin, catalog
from .lineage import LineagePlugin, lineage
from .orchestrator import OrchestratorPlugin, orchestrator

# Graph database plugins
from .neo4j_graph import Neo4jPlugin, neo4j

# Data operations plugins
from .data_quality import DataQualityPlugin, data_quality
from .transformer import TransformerPlugin, transformer


# Simple plugin access class for SDK
class PluginAccess:
    """
    Simple plugin access for the SDK.

    Provides clean interface: sdk.plugins.postgresql(...)
    """

    def postgresql(self, **kwargs) -> PostgreSQLPlugin:
        """Create PostgreSQL plugin."""
        return postgresql(**kwargs)

    def mysql(self, **kwargs) -> MySQLPlugin:
        """Create MySQL plugin."""
        return mysql(**kwargs)

    def mongodb(self, **kwargs) -> MongoDBPlugin:
        """Create MongoDB plugin."""
        return mongodb(**kwargs)

    def snowflake(self, **kwargs) -> SnowflakePlugin:
        """Create Snowflake plugin."""
        return snowflake(**kwargs)

    def sqlite(self, **kwargs) -> SQLitePlugin:
        """Create SQLite plugin."""
        return sqlite(**kwargs)

    def pinecone(self, **kwargs) -> PineconePlugin:
        """Create Pinecone plugin."""
        return pinecone(**kwargs)

    def chroma(self, **kwargs) -> ChromaPlugin:
        """Create ChromaDB plugin."""
        return chroma(**kwargs)

    def qdrant(self, **kwargs) -> QdrantPlugin:
        """Create Qdrant plugin."""
        return qdrant(**kwargs)

    def rest(self, **kwargs) -> RESTPlugin:
        """Create REST API plugin."""
        return rest(**kwargs)

    def s3(self, **kwargs) -> S3Plugin:
        """Create S3 plugin."""
        return s3(**kwargs)

    def google_drive(self, **kwargs) -> GoogleDrivePlugin:
        """Create Google Drive plugin."""
        return google_drive(**kwargs)

    def slack(self, **kwargs) -> SlackPlugin:
        """Create Slack plugin."""
        return slack(**kwargs)

    def email(self, **kwargs) -> EmailPlugin:
        """Create Email plugin."""
        return email(**kwargs)

    def elasticsearch(self, **kwargs) -> ElasticsearchPlugin:
        """Create Elasticsearch plugin."""
        return elasticsearch(**kwargs)

    def websearch(self, **kwargs) -> WebSearchPlugin:
        """Create WebSearch plugin."""
        return websearch(**kwargs)

    def redis_messaging(self, **kwargs) -> RedisMessagingPlugin:
        """Create Redis messaging plugin."""
        return redis_messaging(**kwargs)

    def catalog(self, **kwargs) -> CatalogPlugin:
        """Create Catalog plugin for schema discovery."""
        return catalog(**kwargs)

    def lineage(self, **kwargs) -> LineagePlugin:
        """Create Lineage plugin for data flow tracking."""
        return lineage(**kwargs)

    def neo4j(self, **kwargs) -> Neo4jPlugin:
        """Create Neo4j graph database plugin."""
        return neo4j(**kwargs)

    def orchestrator(self, **kwargs) -> OrchestratorPlugin:
        """Create Orchestrator plugin for multi-agent coordination."""
        return orchestrator(**kwargs)

    def data_quality(self, **kwargs) -> DataQualityPlugin:
        """Create DataQuality plugin for profiling and quality checks."""
        return data_quality(**kwargs)

    def transformer(self, **kwargs) -> TransformerPlugin:
        """Create Transformer plugin for managing SQL transformations."""
        return transformer(**kwargs)

    def memory(self, **kwargs) -> MemoryPlugin:
        """Create Memory plugin for persistent semantic memory."""
        return MemoryPlugin(**kwargs)

    def mcp(self, **kwargs):
        """Access MCP plugin module for Model Context Protocol integration."""
        return mcp


# Export everything needed
__all__ = [
    # Plugin classes
    "PostgreSQLPlugin",
    "MySQLPlugin",
    "MongoDBPlugin",
    "SnowflakePlugin",
    "SQLitePlugin",
    "PineconePlugin",
    "ChromaPlugin",
    "QdrantPlugin",
    "RESTPlugin",
    "S3Plugin",
    "GoogleDrivePlugin",
    "SlackPlugin",
    "EmailPlugin",
    "ElasticsearchPlugin",
    "WebSearchPlugin",
    "RedisMessagingPlugin",
    "CatalogPlugin",
    "LineagePlugin",
    "Neo4jPlugin",
    "OrchestratorPlugin",
    "MemoryPlugin",
    "DataQualityPlugin",
    "TransformerPlugin",
    # Factory functions
    "postgresql",
    "mysql",
    "mongodb",
    "snowflake",
    "sqlite",
    "pinecone",
    "chroma",
    "qdrant",
    "rest",
    "s3",
    "google_drive",
    "slack",
    "email",
    "elasticsearch",
    "websearch",
    "redis_messaging",
    "catalog",
    "lineage",
    "neo4j",
    "orchestrator",
    "data_quality",
    "transformer",
    # MCP module
    "mcp",
    # SDK access class
    "PluginAccess",
]
