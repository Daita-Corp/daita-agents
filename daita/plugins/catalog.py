"""
CatalogPlugin for schema discovery and metadata management.

Provides tools for discovering database schemas, API structures, and other
organizational metadata across multiple platforms.
"""

import ipaddress
import json
import logging
import os
import socket
import ssl
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse, unquote

from .base import BasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)

# Module-level registry. Set once at startup via register_catalog_backend_factory().
# None means persist to .daita/catalog.json (local default).
_CATALOG_BACKEND_FACTORY: Optional[Callable[[], Any]] = None


def register_catalog_backend_factory(factory: Optional[Callable[[], Any]]) -> None:
    """
    Register a factory that creates catalog backends.

    Called once at application startup to inject a storage backend for schema
    documents. Pass None to reset to the default (local .daita/catalog.json),
    which is useful in tests.

    Args:
        factory: Callable that takes no arguments and returns a catalog backend
                 with a ``persist_schema(schema: dict) -> bool`` coroutine method.
                 Pass None to clear the registered factory and revert to default.

    Example:
        register_catalog_backend_factory(lambda: MyCatalogBackend())
    """
    global _CATALOG_BACKEND_FACTORY
    _CATALOG_BACKEND_FACTORY = factory


class CatalogPlugin(BasePlugin):
    """
    Plugin for schema discovery and metadata cataloging.

    Works standalone (returns data directly) or with optional graph storage
    for building an organizational knowledge graph.

    Supports:
    - PostgreSQL, MySQL, MongoDB schema discovery
    - GraphQL introspection
    - OpenAPI/Swagger spec parsing
    - Salesforce object metadata
    - Schema comparison and validation
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
        organization_id: Optional[int] = None,
        auto_persist: bool = False,
    ):
        """
        Initialize CatalogPlugin.

        Args:
            backend: Optional graph backend. If None, auto_select_backend() is called
                     during initialize() to pick LocalGraphBackend or DynamoGraphBackend.
            organization_id: Optional organization ID for multi-tenant storage
            auto_persist: If True, automatically persist discoveries to the graph backend
        """
        self._graph_backend = backend
        self._catalog_backend: Optional[Any] = None
        self._organization_id = organization_id
        self._auto_persist = auto_persist
        self._agent_id: Optional[str] = None

        logger.debug(
            "CatalogPlugin initialized (backend: %s, auto_persist: %s)",
            backend is not None,
            auto_persist,
        )

    def initialize(self, agent_id: str) -> None:
        self._agent_id = agent_id
        if self._graph_backend is None:
            from daita.core.graph.backend import auto_select_backend

            self._graph_backend = auto_select_backend(graph_type="catalog")
            logger.debug(
                "CatalogPlugin: using graph backend %s",
                type(self._graph_backend).__name__,
            )

        if _CATALOG_BACKEND_FACTORY is not None and self._catalog_backend is None:
            try:
                self._catalog_backend = _CATALOG_BACKEND_FACTORY()
                logger.debug(
                    "CatalogPlugin: using catalog backend %s",
                    type(self._catalog_backend).__name__,
                )
            except Exception as exc:
                logger.warning(
                    "CatalogPlugin: catalog backend factory failed: %s. "
                    "Schema persistence will use local JSON.",
                    exc,
                )

    @staticmethod
    def _redact_url(connection_string: str) -> str:
        """
        Return a loggable form of the connection string with the password replaced
        by '***'.  Used whenever we need to reference a connection string in a log
        message so that credentials are not stored in log files or traces.
        """
        try:
            parsed = urlparse(connection_string)
            if parsed.password:
                redacted = parsed._replace(
                    netloc=parsed.netloc.replace(f":{parsed.password}@", ":***@")
                )
                return redacted.geturl()
        except Exception:
            pass
        return connection_string

    @staticmethod
    def _parse_conn_url(connection_string: str) -> Dict[str, Any]:
        """
        Parse a database connection URL into explicit credential kwargs.

        Handles URL-encoded passwords (special chars like ! * @ in passwords
        break URL parsing if not encoded, and some drivers choke on the raw
        connection string). Returns a dict safe to splat into any DB driver's
        connect() call after picking the keys that driver needs.
        """
        parsed = urlparse(connection_string)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port,
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "database": (parsed.path or "/").lstrip("/"),
        }

    @staticmethod
    def _validate_openapi_url(url: str) -> Optional[str]:
        """
        Validate that a URL is safe to fetch as an OpenAPI spec.

        Returns an error message string if the URL is unsafe, None if it is safe.

        Blocks:
        - Non-http/https schemes (e.g. file://, ftp://)
        - Hostnames that resolve to private, loopback, or link-local addresses
          (prevents SSRF to AWS IMDSv1 at 169.254.169.254, internal databases, etc.)
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Only http/https URLs are supported, got: {parsed.scheme!r}"

        hostname = parsed.hostname
        if not hostname:
            return "URL has no hostname"

        try:
            addr_infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            return f"Could not resolve hostname {hostname!r}: {exc}"

        for addr_info in addr_infos:
            # Strip IPv6 zone ID (e.g. "fe80::1%eth0" -> "fe80::1")
            ip_str = addr_info[4][0].split("%")[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return (
                    f"Requests to private/internal addresses are not permitted "
                    f"(hostname {hostname!r} resolved to {ip})"
                )

        return None

    @staticmethod
    def _ssl_context(mode: str = "verify-full") -> ssl.SSLContext:
        """Return an SSL context for database connections.

        mode="verify-full"  (default) — full certificate and hostname verification.
                            Use for direct connections to managed cloud DBs (RDS,
                            Cloud SQL, Azure, Supabase direct port 5432).

        mode="require"      — encrypts the connection but skips certificate
                            verification. Use ONLY when connecting through a
                            pgbouncer pooler (e.g. Supabase pooler port 6543)
                            that presents a self-signed or unverifiable cert.
                            The data is still encrypted in transit; this only
                            disables identity verification of the server.
        """
        if mode == "require":
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return ssl.create_default_context()

    def get_tools(self) -> List["AgentTool"]:
        """
        Expose schema discovery operations as agent tools.

        Returns:
            List of AgentTool instances for catalog operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="discover_postgres",
                description="Discover PostgreSQL database schema including tables, columns, foreign keys, and indexes",
                parameters={
                    "type": "object",
                    "properties": {
                        "connection_string": {
                            "type": "string",
                            "description": "PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db)",
                        },
                        "schema": {
                            "type": "string",
                            "description": "Schema name to introspect (default: 'public')",
                        },
                        "persist": {
                            "type": "boolean",
                            "description": "Whether to persist schema to graph storage if available (default: auto_persist setting)",
                        },
                        "ssl_mode": {
                            "type": "string",
                            "description": "SSL mode: 'verify-full' (default, validates cert) or 'require' (encrypt only, for pgbouncer poolers)",
                        },
                    },
                    "required": ["connection_string"],
                },
                handler=self._tool_discover_postgres,
                category="catalog",
                source="plugin",
                plugin_name="Catalog",
                timeout_seconds=120,
            ),
            AgentTool(
                name="discover_mysql",
                description="Discover MySQL/MariaDB database schema including tables, columns, and relationships",
                parameters={
                    "type": "object",
                    "properties": {
                        "connection_string": {
                            "type": "string",
                            "description": "MySQL connection string (e.g., mysql://user:pass@host:port/db)",
                        },
                        "schema": {
                            "type": "string",
                            "description": "Schema/database name to introspect",
                        },
                        "persist": {
                            "type": "boolean",
                            "description": "Whether to persist schema to graph storage if available",
                        },
                    },
                    "required": ["connection_string"],
                },
                handler=self._tool_discover_mysql,
                category="catalog",
                source="plugin",
                plugin_name="Catalog",
                timeout_seconds=120,
            ),
            AgentTool(
                name="discover_mongodb",
                description="Discover MongoDB schema by sampling documents to infer structure",
                parameters={
                    "type": "object",
                    "properties": {
                        "connection_string": {
                            "type": "string",
                            "description": "MongoDB connection string (e.g., mongodb://user:pass@host:port/db)",
                        },
                        "database": {
                            "type": "string",
                            "description": "Database name to introspect",
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of documents to sample per collection (default: 100)",
                        },
                        "persist": {
                            "type": "boolean",
                            "description": "Whether to persist schema to graph storage if available",
                        },
                    },
                    "required": ["connection_string", "database"],
                },
                handler=self._tool_discover_mongodb,
                category="catalog",
                source="plugin",
                plugin_name="Catalog",
                timeout_seconds=120,
            ),
            AgentTool(
                name="discover_openapi",
                description="Discover API structure from OpenAPI/Swagger specification",
                parameters={
                    "type": "object",
                    "properties": {
                        "spec_url": {
                            "type": "string",
                            "description": "URL to OpenAPI spec (JSON or YAML)",
                        },
                        "service_name": {
                            "type": "string",
                            "description": "Optional service name override",
                        },
                        "persist": {
                            "type": "boolean",
                            "description": "Whether to persist schema to graph storage if available",
                        },
                    },
                    "required": ["spec_url"],
                },
                handler=self._tool_discover_openapi,
                category="catalog",
                source="plugin",
                plugin_name="Catalog",
                timeout_seconds=60,
            ),
            AgentTool(
                name="compare_schemas",
                description="Compare two schemas to identify differences for migration planning",
                parameters={
                    "type": "object",
                    "properties": {
                        "schema_a": {
                            "type": "object",
                            "description": "First schema (from discover_* tools)",
                        },
                        "schema_b": {
                            "type": "object",
                            "description": "Second schema to compare against",
                        },
                    },
                    "required": ["schema_a", "schema_b"],
                },
                handler=self._tool_compare_schemas,
                category="catalog",
                source="plugin",
                plugin_name="Catalog",
                timeout_seconds=30,
            ),
            AgentTool(
                name="export_diagram",
                description="Export schema as a visual diagram in Mermaid, DBDiagram, or JSON Schema format",
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {
                            "type": "object",
                            "description": "Schema object (from discover_* tools)",
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format: 'mermaid', 'dbdiagram', or 'json_schema' (default: 'mermaid')",
                        },
                    },
                    "required": ["schema"],
                },
                handler=self._tool_export_diagram,
                category="catalog",
                source="plugin",
                plugin_name="Catalog",
                timeout_seconds=30,
            ),
        ]

    async def _tool_discover_postgres(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for discover_postgres"""
        connection_string = args.get("connection_string")
        if not connection_string:
            return {"success": False, "error": "connection_string is required"}

        schema = args.get("schema", "public")
        persist = args.get("persist", self._auto_persist)
        ssl_mode = args.get("ssl_mode", "verify-full")

        result = await self.discover_postgres(
            connection_string=connection_string,
            schema=schema,
            persist=persist,
            ssl_mode=ssl_mode,
        )

        return result

    async def discover_postgres(
        self,
        connection_string: str,
        schema: str = "public",
        persist: bool = False,
        ssl_mode: str = "verify-full",
    ) -> Dict[str, Any]:
        """
        Discover PostgreSQL database schema.

        Args:
            connection_string: PostgreSQL connection string
            schema: Schema name (default: public)
            persist: Whether to persist to graph storage
            ssl_mode: 'verify-full' (default) or 'require' (for pgbouncer poolers)

        Returns:
            Dictionary with discovered tables, columns, and relationships
        """
        import asyncpg

        logger.debug(
            "discover_postgres: connecting to %s (ssl_mode=%s)",
            self._redact_url(connection_string),
            ssl_mode,
        )
        creds = self._parse_conn_url(connection_string)
        conn = await asyncpg.connect(
            host=creds["host"],
            port=creds["port"] or 5432,
            user=creds["user"],
            password=creds["password"],
            database=creds["database"] or "postgres",
            ssl=self._ssl_context(ssl_mode),
        )

        try:
            # Get tables
            tables = await conn.fetch(
                """
                SELECT table_name,
                       pg_stat_user_tables.n_live_tup as row_count
                FROM information_schema.tables
                LEFT JOIN pg_stat_user_tables
                    ON table_name = relname
                WHERE table_schema = $1
                AND table_type = 'BASE TABLE'
            """,
                schema,
            )

            # Get columns
            columns = await conn.fetch(
                """
                SELECT
                    table_name,
                    column_name,
                    data_type,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = $1
                ORDER BY table_name, ordinal_position
            """,
                schema,
            )

            # Get primary keys
            pkeys = await conn.fetch(
                """
                SELECT
                    tc.table_name,
                    kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = $1
            """,
                schema,
            )

            # Get foreign keys
            fkeys = await conn.fetch(
                """
                SELECT
                    tc.table_name as source_table,
                    kcu.column_name as source_column,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column,
                    tc.constraint_name,
                    rc.delete_rule,
                    rc.update_rule
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                JOIN information_schema.referential_constraints rc
                    ON tc.constraint_name = rc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = $1
            """,
                schema,
            )

            # Get indexes
            indexes = await conn.fetch(
                """
                SELECT
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = $1
            """,
                schema,
            )

            # Build result
            result = {
                "database_type": "postgresql",
                "schema": schema,
                "tables": [dict(row) for row in tables],
                "columns": [dict(row) for row in columns],
                "primary_keys": [dict(row) for row in pkeys],
                "foreign_keys": [dict(row) for row in fkeys],
                "indexes": [dict(row) for row in indexes],
                "table_count": len(tables),
                "column_count": len(columns),
            }

            # Optionally persist to graph storage
            actually_persisted = False
            persist_skipped_reason = None
            if persist:
                actually_persisted = await self._persist_schema(result)
                if not actually_persisted:
                    persist_skipped_reason = "catalog backend not configured"

            response = {
                "success": True,
                "schema": result,
                "persisted": actually_persisted,
            }
            if persist_skipped_reason:
                response["persist_skipped"] = persist_skipped_reason
            return response

        finally:
            await conn.close()

    async def _tool_discover_mysql(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for discover_mysql"""
        connection_string = args.get("connection_string")
        if not connection_string:
            return {"success": False, "error": "connection_string is required"}

        schema = args.get("schema")
        persist = args.get("persist", self._auto_persist)

        result = await self.discover_mysql(
            connection_string=connection_string, schema=schema, persist=persist
        )

        return result

    async def discover_mysql(
        self,
        connection_string: str,
        schema: Optional[str] = None,
        persist: bool = False,
    ) -> Dict[str, Any]:
        """
        Discover MySQL/MariaDB database schema.

        Args:
            connection_string: MySQL connection string
            schema: Schema/database name
            persist: Whether to persist to graph storage

        Returns:
            Dictionary with discovered tables, columns, and relationships
        """
        import aiomysql

        logger.debug(
            "discover_mysql: connecting to %s", self._redact_url(connection_string)
        )
        creds = self._parse_conn_url(connection_string)
        db_name = schema or creds["database"] or "mysql"

        conn = await aiomysql.connect(
            host=creds["host"],
            port=creds["port"] or 3306,
            user=creds["user"],
            password=creds["password"],
            db=db_name,
            ssl=self._ssl_context(),
        )

        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # Get tables
                await cursor.execute(
                    """
                    SELECT TABLE_NAME as table_name,
                           TABLE_ROWS as row_count
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = %s
                    AND TABLE_TYPE = 'BASE TABLE'
                    """,
                    (db_name,),
                )
                tables = await cursor.fetchall()

                # Get columns
                await cursor.execute(
                    """
                    SELECT
                        TABLE_NAME as table_name,
                        COLUMN_NAME as column_name,
                        DATA_TYPE as data_type,
                        CHARACTER_MAXIMUM_LENGTH as character_maximum_length,
                        NUMERIC_PRECISION as numeric_precision,
                        NUMERIC_SCALE as numeric_scale,
                        IS_NULLABLE as is_nullable,
                        COLUMN_DEFAULT as column_default,
                        COLUMN_KEY as column_key
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = %s
                    ORDER BY TABLE_NAME, ORDINAL_POSITION
                    """,
                    (db_name,),
                )
                columns = await cursor.fetchall()

                # Get foreign keys
                await cursor.execute(
                    """
                    SELECT
                        TABLE_NAME as source_table,
                        COLUMN_NAME as source_column,
                        REFERENCED_TABLE_NAME as target_table,
                        REFERENCED_COLUMN_NAME as target_column,
                        CONSTRAINT_NAME as constraint_name
                    FROM information_schema.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = %s
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                    """,
                    (db_name,),
                )
                fkeys = await cursor.fetchall()

            result = {
                "database_type": "mysql",
                "schema": db_name,
                "tables": tables,
                "columns": columns,
                "foreign_keys": fkeys,
                "table_count": len(tables),
                "column_count": len(columns),
            }

            # Optionally persist to graph storage
            actually_persisted = False
            persist_skipped_reason = None
            if persist:
                actually_persisted = await self._persist_schema(result)
                if not actually_persisted:
                    persist_skipped_reason = "catalog backend not configured"

            response = {
                "success": True,
                "schema": result,
                "persisted": actually_persisted,
            }
            if persist_skipped_reason:
                response["persist_skipped"] = persist_skipped_reason
            return response

        finally:
            conn.close()

    async def _tool_discover_mongodb(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for discover_mongodb"""
        connection_string = args.get("connection_string")
        if not connection_string:
            return {"success": False, "error": "connection_string is required"}

        database = args.get("database")
        if not database:
            return {"success": False, "error": "database is required"}

        sample_size = args.get("sample_size", 100)
        persist = args.get("persist", self._auto_persist)

        result = await self.discover_mongodb(
            connection_string=connection_string,
            database=database,
            sample_size=sample_size,
            persist=persist,
        )

        return result

    async def discover_mongodb(
        self,
        connection_string: str,
        database: str,
        sample_size: int = 100,
        persist: bool = False,
    ) -> Dict[str, Any]:
        """
        Discover MongoDB schema by sampling documents.

        Args:
            connection_string: MongoDB connection string
            database: Database name
            sample_size: Number of documents to sample per collection
            persist: Whether to persist to graph storage

        Returns:
            Dictionary with inferred schema from document samples
        """
        from motor.motor_asyncio import AsyncIOMotorClient

        logger.debug(
            "discover_mongodb: connecting to %s", self._redact_url(connection_string)
        )
        client = AsyncIOMotorClient(connection_string)
        db = client[database]

        try:
            # Get collections
            collection_names = await db.list_collection_names()

            collections_schema = []

            for coll_name in collection_names:
                collection = db[coll_name]

                # Sample documents
                cursor = collection.find().limit(sample_size)
                docs = await cursor.to_list(length=sample_size)

                # Infer schema from samples
                fields = {}
                for doc in docs:
                    for key, value in doc.items():
                        if key not in fields:
                            fields[key] = {
                                "field_name": key,
                                "types": set(),
                                "sample_count": 0,
                            }
                        fields[key]["types"].add(type(value).__name__)
                        fields[key]["sample_count"] += 1

                # Convert sets to lists for JSON serialization
                for field in fields.values():
                    field["types"] = list(field["types"])

                collections_schema.append(
                    {
                        "collection_name": coll_name,
                        "document_count": await collection.estimated_document_count(),
                        "sampled_count": len(docs),
                        "fields": list(fields.values()),
                    }
                )

            result = {
                "database_type": "mongodb",
                "database": database,
                "collections": collections_schema,
                "collection_count": len(collections_schema),
                "sample_size": sample_size,
            }

            # Optionally persist to graph storage
            actually_persisted = False
            persist_skipped_reason = None
            if persist:
                actually_persisted = await self._persist_schema(result)
                if not actually_persisted:
                    persist_skipped_reason = "catalog backend not configured"

            response = {
                "success": True,
                "schema": result,
                "persisted": actually_persisted,
            }
            if persist_skipped_reason:
                response["persist_skipped"] = persist_skipped_reason
            return response

        finally:
            client.close()

    async def _tool_discover_openapi(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for discover_openapi"""
        spec_url = args.get("spec_url")
        if not spec_url:
            return {"success": False, "error": "spec_url is required"}

        service_name = args.get("service_name")
        persist = args.get("persist", self._auto_persist)

        result = await self.discover_openapi(
            spec_url=spec_url, service_name=service_name, persist=persist
        )

        return result

    async def discover_openapi(
        self, spec_url: str, service_name: Optional[str] = None, persist: bool = False
    ) -> Dict[str, Any]:
        """
        Discover API structure from OpenAPI/Swagger spec.

        Args:
            spec_url: URL to OpenAPI spec (JSON or YAML)
            service_name: Optional service name override
            persist: Whether to persist to graph storage

        Returns:
            Dictionary with discovered endpoints
        """
        import httpx
        import yaml

        url_error = self._validate_openapi_url(spec_url)
        if url_error:
            return {"success": False, "error": url_error}

        # follow_redirects=False prevents redirect-based SSRF bypasses.
        # timeout guards against slow-loris / hung internal endpoints.
        async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
            resp = await client.get(spec_url)
            resp.raise_for_status()

            if spec_url.endswith(".yaml") or spec_url.endswith(".yml"):
                spec = yaml.safe_load(resp.text)
            else:
                spec = resp.json()

        base_url = ""
        if spec.get("servers"):
            base_url = spec["servers"][0].get("url", "")

        svc_name = service_name or spec.get("info", {}).get("title", "Unknown API")

        # Extract endpoints
        endpoints = []
        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                if method.startswith("x-"):
                    continue

                endpoints.append(
                    {
                        "method": method.upper(),
                        "path": path,
                        "summary": details.get("summary", ""),
                        "description": details.get("description", ""),
                        "parameters": details.get("parameters", []),
                        "request_body": details.get("requestBody", {}),
                        "responses": details.get("responses", {}),
                    }
                )

        result = {
            "api_type": "openapi",
            "service_name": svc_name,
            "base_url": base_url,
            "version": spec.get("info", {}).get("version", "unknown"),
            "endpoints": endpoints,
            "endpoint_count": len(endpoints),
        }

        # Optionally persist to graph storage
        actually_persisted = False
        persist_skipped_reason = None
        if persist:
            actually_persisted = await self._persist_schema(result)
            if not actually_persisted:
                persist_skipped_reason = "catalog backend not configured"

        response = {
            "success": True,
            "schema": result,
            "persisted": actually_persisted,
        }
        if persist_skipped_reason:
            response["persist_skipped"] = persist_skipped_reason
        return response

    async def _tool_compare_schemas(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for compare_schemas"""
        schema_a = args.get("schema_a")
        schema_b = args.get("schema_b")

        result = await self.compare_schemas(schema_a, schema_b)

        return result

    async def compare_schemas(
        self, schema_a: Dict[str, Any], schema_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two schemas to identify differences.

        Args:
            schema_a: First schema
            schema_b: Second schema

        Returns:
            Dictionary with added, removed, and modified elements
        """
        # MongoDB schemas use "collections" instead of "tables"
        db_type_a = schema_a.get("database_type", "")
        db_type_b = schema_b.get("database_type", "")
        if db_type_a == "mongodb" or db_type_b == "mongodb":
            tables_a = {
                c["collection_name"]: c for c in schema_a.get("collections", [])
            }
            tables_b = {
                c["collection_name"]: c for c in schema_b.get("collections", [])
            }
        else:
            tables_a = {t["table_name"]: t for t in schema_a.get("tables", [])}
            tables_b = {t["table_name"]: t for t in schema_b.get("tables", [])}

        added_tables = [name for name in tables_b if name not in tables_a]
        removed_tables = [name for name in tables_a if name not in tables_b]

        # Compare columns
        columns_a = {
            (c["table_name"], c["column_name"]): c for c in schema_a.get("columns", [])
        }
        columns_b = {
            (c["table_name"], c["column_name"]): c for c in schema_b.get("columns", [])
        }

        added_columns = [key for key in columns_b if key not in columns_a]
        removed_columns = [key for key in columns_a if key not in columns_b]

        # Type changes
        modified_columns = []
        for key in set(columns_a.keys()) & set(columns_b.keys()):
            if columns_a[key].get("data_type") != columns_b[key].get("data_type"):
                modified_columns.append(
                    {
                        "table": key[0],
                        "column": key[1],
                        "old_type": columns_a[key].get("data_type"),
                        "new_type": columns_b[key].get("data_type"),
                    }
                )

        return {
            "success": True,
            "comparison": {
                "added_tables": added_tables,
                "removed_tables": removed_tables,
                "added_columns": [
                    {"table": k[0], "column": k[1]} for k in added_columns
                ],
                "removed_columns": [
                    {"table": k[0], "column": k[1]} for k in removed_columns
                ],
                "modified_columns": modified_columns,
                "breaking_changes": len(removed_tables)
                + len(removed_columns)
                + len(modified_columns),
            },
        }

    async def _tool_export_diagram(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for export_diagram"""
        schema = args.get("schema")
        format = args.get("format", "mermaid")

        result = await self.export_diagram(schema, format)

        return result

    async def export_diagram(
        self, schema: Dict[str, Any], format: str = "mermaid"
    ) -> Dict[str, Any]:
        """
        Export schema as a visual diagram.

        Args:
            schema: Schema object
            format: Output format ('mermaid', 'dbdiagram', 'json_schema')

        Returns:
            Dictionary with diagram in requested format
        """
        if format == "mermaid":
            # Generate Mermaid ER diagram
            lines = ["erDiagram"]

            # Group columns by table
            tables = {}
            for col in schema.get("columns", []):
                table_name = col["table_name"]
                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append(col)

            # Add tables and columns
            for table_name, columns in tables.items():
                lines.append(f"    {table_name} {{")
                for col in columns:
                    data_type = col.get("data_type", "unknown")
                    col_name = col["column_name"]
                    lines.append(f"        {data_type} {col_name}")
                lines.append("    }")

            # Add relationships
            for fk in schema.get("foreign_keys", []):
                source = fk["source_table"]
                target = fk["target_table"]
                lines.append(f'    {source} ||--o{{ {target} : ""')

            diagram = "\n".join(lines)

            return {"success": True, "format": "mermaid", "diagram": diagram}

        elif format == "json_schema":
            # Generate JSON Schema representation
            json_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "required": [],
            }

            # Add table schemas
            for table in schema.get("tables", []):
                table_name = table["table_name"]
                table_columns = [
                    c
                    for c in schema.get("columns", [])
                    if c["table_name"] == table_name
                ]

                properties = {}
                for col in table_columns:
                    col_type = self._map_sql_to_json_type(
                        col.get("data_type", "string")
                    )
                    properties[col["column_name"]] = {"type": col_type}

                json_schema["properties"][table_name] = {
                    "type": "object",
                    "properties": properties,
                }

            return {"success": True, "format": "json_schema", "schema": json_schema}

        else:
            return {
                "success": False,
                "error": f"Unsupported format: {format}. Use 'mermaid' or 'json_schema'",
            }

    def _map_sql_to_json_type(self, sql_type: str) -> str:
        """Map SQL data types to JSON Schema types"""
        sql_type = sql_type.lower()

        if any(t in sql_type for t in ["int", "serial", "bigint", "smallint"]):
            return "integer"
        elif any(
            t in sql_type for t in ["float", "double", "decimal", "numeric", "real"]
        ):
            return "number"
        elif any(t in sql_type for t in ["bool", "boolean"]):
            return "boolean"
        elif any(t in sql_type for t in ["json", "jsonb"]):
            return "object"
        elif "array" in sql_type:
            return "array"
        else:
            return "string"

    async def _persist_schema(self, schema: Dict[str, Any]) -> bool:
        """Persist schema to the catalog store and graph backend.

        Storage selection (in priority order):
        1. self._catalog_backend — set during initialize() when DAITA_CATALOG_BACKEND_CLASS
           is present. Used in cloud deployments. Falls through to local JSON on failure.
        2. Local .daita/catalog.json — default for local development.

        In both cases, if a graph backend is available, schema entities are also
        written as graph nodes so LineagePlugin can reference them by node_id.

        Returns True if schema was persisted, False if the operation was skipped.
        """
        import aiofiles
        from datetime import datetime, timezone
        from pathlib import Path

        persisted = False

        if self._catalog_backend is not None:
            try:
                persisted = await self._catalog_backend.persist_schema(schema)
            except Exception as exc:
                logger.warning(
                    "Catalog backend failed, falling back to local JSON: %s", exc
                )

        if not persisted:
            catalog_path = Path(".daita") / "catalog.json"
            catalog_path.parent.mkdir(parents=True, exist_ok=True)

            existing: Dict[str, Any] = {}
            if catalog_path.exists():
                try:
                    async with aiofiles.open(catalog_path, "r") as f:
                        existing = json.loads(await f.read())
                except (json.JSONDecodeError, ValueError):
                    logger.warning("catalog.json was corrupt, overwriting.")

            key = f"{schema.get('database_type', 'unknown')}:{schema.get('schema', 'default')}"
            now = datetime.now(timezone.utc).isoformat()

            if key in existing:
                # Preserve first_seen from initial discovery; update last_seen
                schema["first_seen"] = existing[key].get("first_seen", now)
                schema["last_seen"] = now
            else:
                schema["first_seen"] = now
                schema["last_seen"] = now

            existing[key] = schema

            async with aiofiles.open(catalog_path, "w") as f:
                await f.write(json.dumps(existing, indent=2, default=str))

            logger.debug("Persisted schema to %s", catalog_path)
            persisted = True

        # Write discovered entities as graph nodes so LineagePlugin can reference
        # them by node_id (e.g. "table:orders"). Runs in both local and cloud paths.
        if self._graph_backend:
            try:
                await self._persist_schema_to_graph(schema)
            except Exception as graph_err:
                logger.error(
                    "Failed to persist schema to graph backend (schema data was saved): %s",
                    graph_err,
                )
                schema["graph_persist_error"] = str(graph_err)

        return persisted

    async def _persist_schema_to_graph(self, schema: Dict[str, Any]) -> None:
        """
        Write discovered schema entities as nodes into the graph backend.

        TABLE nodes are written with column metadata stored in properties so
        LineagePlugin flows can reference the same node_ids (e.g. "table:users").
        MongoDB collections are treated as TABLE nodes. OpenAPI services are
        written as API nodes.
        """
        from daita.core.graph.models import AgentGraphNode, NodeType

        db_type = schema.get("database_type") or schema.get("api_type", "unknown")
        schema_name = schema.get("schema") or schema.get("database", "default")

        # --- Relational databases (postgres, mysql) ---
        if db_type in ("postgresql", "mysql"):
            columns_by_table: Dict[str, list] = {}
            for col in schema.get("columns", []):
                columns_by_table.setdefault(col["table_name"], []).append(col)

            for table in schema.get("tables", []):
                tname = table["table_name"]
                node = AgentGraphNode(
                    node_id=AgentGraphNode.make_id(NodeType.TABLE, tname),
                    node_type=NodeType.TABLE,
                    name=tname,
                    created_by_agent=self._agent_id,
                    properties={
                        "database_type": db_type,
                        "schema": schema_name,
                        "row_count": table.get("row_count"),
                        "columns": columns_by_table.get(tname, []),
                    },
                )
                await self._graph_backend.add_node(node)

        # --- MongoDB ---
        elif db_type == "mongodb":
            for coll in schema.get("collections", []):
                cname = coll["collection_name"]
                node = AgentGraphNode(
                    node_id=AgentGraphNode.make_id(NodeType.TABLE, cname),
                    node_type=NodeType.TABLE,
                    name=cname,
                    created_by_agent=self._agent_id,
                    properties={
                        "database_type": "mongodb",
                        "database": schema_name,
                        "document_count": coll.get("document_count"),
                        "fields": coll.get("fields", []),
                    },
                )
                await self._graph_backend.add_node(node)

        # --- OpenAPI services ---
        elif db_type == "openapi":
            svc_name = schema.get("service_name", "unknown_api")
            node = AgentGraphNode(
                node_id=AgentGraphNode.make_id(NodeType.API, svc_name),
                node_type=NodeType.API,
                name=svc_name,
                created_by_agent=self._agent_id,
                properties={
                    "base_url": schema.get("base_url", ""),
                    "version": schema.get("version", ""),
                    "endpoint_count": schema.get("endpoint_count", 0),
                },
            )
            await self._graph_backend.add_node(node)

        logger.debug(
            f"CatalogPlugin: persisted {db_type}:{schema_name} entities to graph backend"
        )

    async def prune_stale_catalog(self, max_age_seconds: int) -> dict:
        """
        Remove catalog entries whose last_seen is older than max_age_seconds.

        Call at the end of a full discovery run to evict schemas for databases
        or services that are no longer reachable or in use.

        Entries with no last_seen (written before this feature) are left untouched.

        Returns {"removed": [list of removed keys]}
        """
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        catalog_path = Path(".daita") / "catalog.json"
        if not catalog_path.exists():
            return {"removed": []}

        try:
            with open(catalog_path, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            return {"removed": []}

        cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
        removed = []

        for key in list(existing.keys()):
            last_seen_raw = existing[key].get("last_seen")
            if last_seen_raw is None:
                continue
            try:
                ts = datetime.fromisoformat(str(last_seen_raw).replace("Z", "+00:00"))
                if ts.timestamp() < cutoff:
                    removed.append(key)
                    del existing[key]
            except (ValueError, TypeError):
                continue

        if removed:
            with open(catalog_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            logger.info(f"Catalog prune: removed {len(removed)} entries: {removed}")

        return {"removed": removed}


def catalog(**kwargs) -> CatalogPlugin:
    """Create CatalogPlugin with simplified interface."""
    return CatalogPlugin(**kwargs)
