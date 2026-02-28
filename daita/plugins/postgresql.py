"""
PostgreSQL plugin for Daita Agents.

Simple database connection and querying - no over-engineering.
"""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base_db import BaseDatabasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)

class PostgreSQLPlugin(BaseDatabasePlugin):
    """
    PostgreSQL plugin for agents with standardized connection management.
    
    Inherits common database functionality from BaseDatabasePlugin.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "",
        username: str = "",
        user: Optional[str] = None,  # Add this
        password: str = "",
        connection_string: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize PostgreSQL connection.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Username
            user: Username (alias for username)
            password: Password
            connection_string: Full connection string (overrides individual params)
            **kwargs: Additional asyncpg parameters
        """
        # Use 'user' parameter as alias for 'username' if provided
        effective_username = user if user is not None else username
        
        # Build connection string
        if connection_string:
            self.connection_string = connection_string
        else:
            # Build connection string - handle empty password case
            if password:
                self.connection_string = f"postgresql://{effective_username}:{password}@{host}:{port}/{database}"
            else:
                self.connection_string = f"postgresql://{effective_username}@{host}:{port}/{database}"
        
        # PostgreSQL-specific pool configuration
        self.pool_config = {
            'min_size': kwargs.get('min_size', 1),
            'max_size': kwargs.get('max_size', 10),
            'command_timeout': kwargs.get('command_timeout', 60),
            'statement_cache_size': kwargs.get('statement_cache_size', 0),  # Set to 0 for pgbouncer compatibility
        }
        
        # Initialize base class with all config
        super().__init__(
            host=host, port=port, database=database, 
            username=effective_username, connection_string=connection_string,
            **kwargs
        )
        
        logger.debug(f"PostgreSQL plugin configured for {host}:{port}/{database}")
    
    async def connect(self):
        """Connect to PostgreSQL database."""
        if self._pool is not None:
            return  # Already connected

        try:
            import asyncpg
            logger.debug(f"Connecting to PostgreSQL with connection string (password masked)")
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                **self.pool_config
            )
            logger.info("Connected to PostgreSQL")
        except ImportError:
            self._handle_connection_error(
                ImportError("asyncpg not installed. Run: pip install asyncpg"),
                "connection"
            )
        except Exception as e:
            # Enhance error message with troubleshooting tips
            error_msg = str(e)
            troubleshooting = []

            if "role" in error_msg and "does not exist" in error_msg:
                troubleshooting.append("Database user may not exist. Check POSTGRES_USER setting.")
                troubleshooting.append("For Docker: ensure container started with correct POSTGRES_USER env var.")

            if "database" in error_msg and "does not exist" in error_msg:
                troubleshooting.append("Database does not exist. Check POSTGRES_DB setting.")
                troubleshooting.append(f"For ankane/pgvector: default database is 'postgres', not custom names.")
                troubleshooting.append("Create database first or connect to existing database.")

            if "password authentication failed" in error_msg:
                troubleshooting.append("Password authentication failed. Check POSTGRES_PASSWORD setting.")
                troubleshooting.append("For Docker: ensure -e POSTGRES_PASSWORD=<pwd> was set when starting container.")

            if troubleshooting:
                enhanced_error = f"{error_msg}\n\nTroubleshooting:\n" + "\n".join(f"  - {tip}" for tip in troubleshooting)
                logger.error(enhanced_error)

            self._handle_connection_error(e, "connection")
    
    async def disconnect(self):
        """Disconnect from PostgreSQL database."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from PostgreSQL")
    
    async def query(self, sql: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        Run a SELECT query and return results.
        
        Args:
            sql: SQL query with $1, $2, etc. placeholders
            params: List of parameters for the query
            
        Returns:
            List of rows as dictionaries
            
        Example:
            results = await db.query("SELECT * FROM users WHERE age > $1", [25])
        """
        # Only auto-connect if pool is None - allows manual mocking
        if self._pool is None:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            if params:
                rows = await conn.fetch(sql, *params)
            else:
                rows = await conn.fetch(sql)
            
            return [dict(row) for row in rows]
    
    async def execute(self, sql: str, params: Optional[List] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE and return affected rows.
        
        Args:
            sql: SQL statement
            params: List of parameters
            
        Returns:
            Number of affected rows
        """
        # Only auto-connect if pool is None - allows manual mocking
        if self._pool is None:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            if params:
                result = await conn.execute(sql, *params)
            else:
                result = await conn.execute(sql)

            # Extract number from result like "INSERT 0 5" or "UPDATE 3"
            # Some commands like "CREATE EXTENSION" don't return a count
            if result:
                try:
                    return int(result.split()[-1])
                except ValueError:
                    # Command succeeded but doesn't return a count (e.g., CREATE EXTENSION)
                    return 0
            return 0
    
    async def insert_many(self, table: str, data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert data into a table.
        
        Args:
            table: Table name
            data: List of dictionaries to insert
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        # Only auto-connect if pool is None - allows manual mocking
        if self._pool is None:
            await self.connect()
        
        # Get columns from first row
        columns = list(data[0].keys())
        placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
        
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Convert to list of tuples for executemany
        rows = [[row[col] for col in columns] for row in data]
        
        async with self._pool.acquire() as conn:
            await conn.executemany(sql, rows)
        
        return len(data)
    
    async def tables(self) -> List[str]:
        """List all tables in the database."""
        sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        results = await self.query(sql)
        return [row['table_name'] for row in results]

    async def vector_search(
        self,
        table: str,
        vector_column: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[str] = None,
        select_columns: str = "*",
        distance_type: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using pgvector extension.

        Args:
            table: Table name containing vectors
            vector_column: Column name with vector data
            query_vector: Query vector as list of floats
            top_k: Number of results to return
            filter: Optional SQL WHERE clause (e.g., "category = 'tech'")
            select_columns: Columns to select (default "*")
            distance_type: Distance metric - "cosine", "l2", or "inner_product"

        Returns:
            List of rows with similarity scores

        Example:
            results = await db.vector_search(
                table="documents",
                vector_column="embedding",
                query_vector=[0.1, 0.2, 0.3, ...],
                top_k=5,
                filter="category = 'tech'",
                distance_type="cosine"
            )
        """
        # Only auto-connect if pool is None
        if self._pool is None:
            await self.connect()

        # Distance operators: <=> (cosine), <-> (L2), <#> (inner product)
        distance_ops = {
            "cosine": "<=>",
            "l2": "<->",
            "inner_product": "<#>"
        }

        if distance_type not in distance_ops:
            raise ValueError(f"Invalid distance_type. Must be one of: {list(distance_ops.keys())}")

        operator = distance_ops[distance_type]

        # Build SQL query
        where_clause = f"WHERE {filter}" if filter else ""
        sql = f"""
        SELECT {select_columns},
               {vector_column} {operator} $1::vector AS distance
        FROM {table}
        {where_clause}
        ORDER BY distance
        LIMIT {top_k}
        """

        # Convert vector to string format for pgvector
        vector_str = f"[{','.join(map(str, query_vector))}]"

        results = await self.query(sql, [vector_str])
        return results

    async def vector_upsert(
        self,
        table: str,
        id_column: str,
        vector_column: str,
        id: str,
        vector: List[float],
        extra_columns: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Insert or update a vector with ON CONFLICT handling.

        Args:
            table: Table name
            id_column: Primary key column name
            vector_column: Vector column name
            id: ID value for the row
            vector: Vector as list of floats
            extra_columns: Optional dictionary of additional columns to upsert

        Returns:
            Dictionary with operation results

        Example:
            result = await db.vector_upsert(
                table="documents",
                id_column="id",
                vector_column="embedding",
                id="doc123",
                vector=[0.1, 0.2, 0.3, ...],
                extra_columns={"title": "My Doc", "category": "tech"}
            )
        """
        # Only auto-connect if pool is None
        if self._pool is None:
            await self.connect()

        # Build column lists
        columns = [id_column, vector_column]
        values_placeholders = ["$1", "$2"]
        params = [id, f"[{','.join(map(str, vector))}]"]

        if extra_columns:
            for idx, (col, val) in enumerate(extra_columns.items(), start=3):
                columns.append(col)
                values_placeholders.append(f"${idx}")
                params.append(val)

        # Build update clause for ON CONFLICT
        update_clauses = [f"{vector_column} = EXCLUDED.{vector_column}"]
        if extra_columns:
            for col in extra_columns.keys():
                update_clauses.append(f"{col} = EXCLUDED.{col}")

        sql = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(values_placeholders)})
        ON CONFLICT ({id_column})
        DO UPDATE SET {', '.join(update_clauses)}
        RETURNING *
        """

        async with self._pool.acquire() as conn:
            result = await conn.fetchrow(sql, *params)
            return dict(result) if result else {}

    async def create_vector_index(
        self,
        table: str,
        vector_column: str,
        index_type: str = "hnsw",
        distance_type: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Create a vector index for faster similarity search.

        Args:
            table: Table name
            vector_column: Vector column name
            index_type: Index type - "hnsw" (faster queries) or "ivfflat" (faster builds)
            distance_type: Distance metric - "cosine", "l2", or "inner_product"

        Returns:
            Dictionary with index creation results

        Example:
            result = await db.create_vector_index(
                table="documents",
                vector_column="embedding",
                index_type="hnsw",
                distance_type="cosine"
            )
        """
        # Only auto-connect if pool is None
        if self._pool is None:
            await self.connect()

        # Distance operators for index: vector_cosine_ops, vector_l2_ops, vector_ip_ops
        ops_map = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops"
        }

        if distance_type not in ops_map:
            raise ValueError(f"Invalid distance_type. Must be one of: {list(ops_map.keys())}")

        if index_type not in ["hnsw", "ivfflat"]:
            raise ValueError("Invalid index_type. Must be 'hnsw' or 'ivfflat'")

        ops = ops_map[distance_type]
        index_name = f"{table}_{vector_column}_{index_type}_idx"

        sql = f"CREATE INDEX {index_name} ON {table} USING {index_type} ({vector_column} {ops})"

        try:
            await self.execute(sql)
            return {
                "success": True,
                "index_name": index_name,
                "table": table,
                "column": vector_column,
                "index_type": index_type,
                "distance_type": distance_type
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "table": table,
                "column": vector_column
            }

    def get_tools(self) -> List['AgentTool']:
        """
        Expose PostgreSQL operations as agent tools.

        Returns:
            List of AgentTool instances for database operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="query_database",
                description="Execute a SQL SELECT query on the PostgreSQL database and return results as a list of dictionaries",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query with $1, $2, etc. placeholders for parameters"
                        },
                        "params": {
                            "type": "array",
                            "description": "Optional list of parameter values for query placeholders",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["sql"]
                },
                handler=self._tool_query,
                category="database",
                source="plugin",
                plugin_name="PostgreSQL",
                timeout_seconds=60
            ),
            AgentTool(
                name="list_tables",
                description="List all tables in the PostgreSQL database",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=self._tool_list_tables,
                category="database",
                source="plugin",
                plugin_name="PostgreSQL",
                timeout_seconds=30
            ),
            AgentTool(
                name="get_table_schema",
                description="Get column information (name, data type, nullable) for a specific table in PostgreSQL",
                parameters={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to inspect"
                        }
                    },
                    "required": ["table_name"]
                },
                handler=self._tool_get_schema,
                category="database",
                source="plugin",
                plugin_name="PostgreSQL",
                timeout_seconds=30
            ),
            AgentTool(
                name="execute_sql",
                description="Execute an INSERT, UPDATE, or DELETE SQL statement on PostgreSQL. Returns the number of affected rows.",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL statement to execute (INSERT, UPDATE, or DELETE)"
                        },
                        "params": {
                            "type": "array",
                            "description": "Optional list of parameter values for statement placeholders",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["sql"]
                },
                handler=self._tool_execute,
                category="database",
                source="plugin",
                plugin_name="PostgreSQL",
                timeout_seconds=60
            ),
            AgentTool(
                name="vector_search",
                description="Search for similar vectors using pgvector extension. Returns matching rows with similarity scores.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Table name containing vector data"
                        },
                        "vector_column": {
                            "type": "string",
                            "description": "Column name with vector embeddings"
                        },
                        "query_vector": {
                            "type": "array",
                            "description": "Query vector as array of floats",
                            "items": {"type": "number"}
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10)"
                        },
                        "filter": {
                            "type": "string",
                            "description": "Optional SQL WHERE clause for filtering (e.g., \"category = 'tech'\")"
                        },
                        "distance_type": {
                            "type": "string",
                            "description": "Distance metric: 'cosine', 'l2', or 'inner_product' (default: 'cosine')"
                        }
                    },
                    "required": ["table", "vector_column", "query_vector"]
                },
                handler=self._tool_vector_search,
                category="database",
                source="plugin",
                plugin_name="PostgreSQL",
                timeout_seconds=60
            ),
            AgentTool(
                name="vector_upsert",
                description="Insert or update a vector with metadata in PostgreSQL using pgvector. Uses ON CONFLICT to handle duplicates.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Table name"
                        },
                        "id_column": {
                            "type": "string",
                            "description": "Primary key column name"
                        },
                        "vector_column": {
                            "type": "string",
                            "description": "Vector column name"
                        },
                        "id": {
                            "type": "string",
                            "description": "ID value for the row"
                        },
                        "vector": {
                            "type": "array",
                            "description": "Vector as array of floats",
                            "items": {"type": "number"}
                        },
                        "extra_columns": {
                            "type": "object",
                            "description": "Optional additional columns to upsert as key-value pairs"
                        }
                    },
                    "required": ["table", "id_column", "vector_column", "id", "vector"]
                },
                handler=self._tool_vector_upsert,
                category="database",
                source="plugin",
                plugin_name="PostgreSQL",
                timeout_seconds=60
            )
        ]

    async def _tool_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for query_database"""
        sql = args.get("sql")
        params = args.get("params")

        results = await self.query(sql, params)

        return {
            "success": True,
            "rows": results,
            "row_count": len(results)
        }

    async def _tool_list_tables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_tables"""
        tables = await self.tables()

        return {
            "success": True,
            "tables": tables,
            "count": len(tables)
        }

    async def _tool_get_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_table_schema"""
        table_name = args.get("table_name")

        schema_query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """

        columns = await self.query(schema_query, [table_name])

        return {
            "success": True,
            "table": table_name,
            "columns": columns,
            "column_count": len(columns)
        }

    async def _tool_execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for execute_sql"""
        sql = args.get("sql")
        params = args.get("params")

        affected_rows = await self.execute(sql, params)

        return {
            "success": True,
            "affected_rows": affected_rows
        }

    async def _tool_vector_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for vector_search"""
        table = args.get("table")
        vector_column = args.get("vector_column")
        query_vector = args.get("query_vector")
        top_k = args.get("top_k", 10)
        filter = args.get("filter")
        distance_type = args.get("distance_type", "cosine")

        results = await self.vector_search(
            table=table,
            vector_column=vector_column,
            query_vector=query_vector,
            top_k=top_k,
            filter=filter,
            distance_type=distance_type
        )

        return {
            "success": True,
            "results": results,
            "count": len(results)
        }

    async def _tool_vector_upsert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for vector_upsert"""
        table = args.get("table")
        id_column = args.get("id_column")
        vector_column = args.get("vector_column")
        id = args.get("id")
        vector = args.get("vector")
        extra_columns = args.get("extra_columns")

        result = await self.vector_upsert(
            table=table,
            id_column=id_column,
            vector_column=vector_column,
            id=id,
            vector=vector,
            extra_columns=extra_columns
        )

        return {
            "success": True,
            "row": result
        }

def postgresql(**kwargs) -> PostgreSQLPlugin:
    """Create PostgreSQL plugin with simplified interface."""
    return PostgreSQLPlugin(**kwargs)