"""
Snowflake plugin for Daita Agents.

Provides Snowflake data warehouse connection and querying capabilities.
Supports key-pair authentication, warehouse management, and stage operations.
"""
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base_db import BaseDatabasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class SnowflakePlugin(BaseDatabasePlugin):
    """
    Snowflake plugin for agents with warehouse management and stage operations.

    Inherits common database functionality from BaseDatabasePlugin and adds
    Snowflake-specific features like warehouse switching and stage data loading.

    Supports password, key-pair, and external browser (SSO) authentication.

    Example:
        from daita.plugins import snowflake

        # Password authentication
        db = snowflake(
            account="xy12345",
            warehouse="COMPUTE_WH",
            database="MYDB",
            user="myuser",
            password="mypass"
        )

        # Key-pair authentication
        db = snowflake(
            account="xy12345",
            warehouse="COMPUTE_WH",
            database="MYDB",
            user="myuser",
            private_key_path="/path/to/key.p8"
        )

        # External browser authentication (SSO)
        db = snowflake(
            account="xy12345",
            warehouse="COMPUTE_WH",
            database="MYDB",
            user="myuser",
            authenticator="externalbrowser"
        )

        # Use with agent
        agent = Agent(
            name="Data Analyst",
            tools=[db]
        )
    """

    def __init__(
        self,
        account: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: str = "PUBLIC",
        user: Optional[str] = None,
        password: Optional[str] = None,
        role: Optional[str] = None,
        authenticator: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        timeout: int = 300,
        **kwargs
    ):
        """
        Initialize Snowflake connection.

        Args:
            account: Snowflake account identifier (e.g., "xy12345")
            warehouse: Compute warehouse name
            database: Database name
            schema: Schema name (default: "PUBLIC")
            user: Username for authentication
            password: Password for authentication (optional if using key-pair or externalbrowser)
            role: Role to use for the session
            authenticator: Authentication method (e.g., "externalbrowser" for SSO)
            private_key_path: Path to private key file for key-pair auth
            private_key_passphrase: Passphrase for encrypted private key
            timeout: Query timeout in seconds (default: 300)
            **kwargs: Additional configuration options

        Environment variables:
            SNOWFLAKE_ACCOUNT: Account identifier
            SNOWFLAKE_WAREHOUSE: Warehouse name
            SNOWFLAKE_DATABASE: Database name
            SNOWFLAKE_SCHEMA: Schema name
            SNOWFLAKE_USER: Username
            SNOWFLAKE_PASSWORD: Password
            SNOWFLAKE_ROLE: Role name
            SNOWFLAKE_AUTHENTICATOR: Authentication method (e.g., "externalbrowser")
            SNOWFLAKE_PRIVATE_KEY_PATH: Path to private key
            SNOWFLAKE_PRIVATE_KEY_PASSPHRASE: Private key passphrase
        """
        # Load from environment variables with fallbacks
        self.account = account if account is not None else os.getenv("SNOWFLAKE_ACCOUNT")
        self.warehouse = warehouse if warehouse is not None else os.getenv("SNOWFLAKE_WAREHOUSE")
        self.database_name = database if database is not None else os.getenv("SNOWFLAKE_DATABASE")
        self.schema = schema if schema != "PUBLIC" else os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        self.user = user if user is not None else os.getenv("SNOWFLAKE_USER")
        self.password = password if password is not None else os.getenv("SNOWFLAKE_PASSWORD")
        self.role = role if role is not None else os.getenv("SNOWFLAKE_ROLE")
        self.authenticator = authenticator if authenticator is not None else os.getenv("SNOWFLAKE_AUTHENTICATOR")
        self.private_key_path = private_key_path if private_key_path is not None else os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        self.private_key_passphrase = private_key_passphrase if private_key_passphrase is not None else os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
        self.timeout = timeout

        # Validate required parameters
        if not self.account:
            raise ValueError("Snowflake account is required. Provide 'account' parameter or set SNOWFLAKE_ACCOUNT environment variable.")
        if not self.user:
            raise ValueError("Snowflake user is required. Provide 'user' parameter or set SNOWFLAKE_USER environment variable.")
        if not self.warehouse:
            raise ValueError("Snowflake warehouse is required. Provide 'warehouse' parameter or set SNOWFLAKE_WAREHOUSE environment variable.")
        if not self.database_name:
            raise ValueError("Snowflake database is required. Provide 'database' parameter or set SNOWFLAKE_DATABASE environment variable.")

        # Validate authentication credentials
        if not self.password and not self.private_key_path and not self.authenticator:
            raise ValueError("Authentication required: provide either 'password', 'private_key_path', or 'authenticator' parameter.")

        # Build connection configuration
        self.connection_config = {
            'account': self.account,
            'warehouse': self.warehouse,
            'database': self.database_name,
            'schema': self.schema,
            'user': self.user,
            'network_timeout': timeout,
            'login_timeout': 60,
        }

        # Add role if specified
        if self.role:
            self.connection_config['role'] = self.role

        # Add authenticator if specified
        if self.authenticator:
            self.connection_config['authenticator'] = self.authenticator

        # Add authentication (handled in connect method)
        self._use_key_pair = bool(self.private_key_path)

        # Call parent constructor
        super().__init__(
            account=self.account,
            warehouse=self.warehouse,
            database=self.database_name,
            schema=self.schema,
            user=self.user,
            role=self.role,
            **kwargs
        )

        # Determine auth method for logging
        if self._use_key_pair:
            auth_method = 'key-pair'
        elif self.authenticator:
            auth_method = self.authenticator
        else:
            auth_method = 'password'

        logger.debug(f"Snowflake plugin configured for {self.account}/{self.database_name} (auth: {auth_method})")

    def _load_private_key(self):
        """Load and decode private key for key-pair authentication."""
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            with open(self.private_key_path, 'rb') as key_file:
                private_key_data = key_file.read()

            # Load private key with optional passphrase
            passphrase = self.private_key_passphrase.encode() if self.private_key_passphrase else None

            private_key = serialization.load_pem_private_key(
                private_key_data,
                password=passphrase,
                backend=default_backend()
            )

            # Get private key bytes in DER format
            private_key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            return private_key_bytes

        except ImportError:
            raise ImportError(
                "cryptography library is required for key-pair authentication. "
                "Install with: pip install 'snowflake-connector-python[secure-local-storage]'"
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Private key file not found: {self.private_key_path}")
        except Exception as e:
            raise ValueError(f"Failed to load private key: {str(e)}")

    async def connect(self):
        """
        Connect to Snowflake.

        Establishes connection using either password or key-pair authentication.
        Connection is idempotent - won't create duplicate connections.
        """
        if self._connection is not None:
            return  # Already connected

        try:
            import snowflake.connector

            # Add authentication credentials
            config = self.connection_config.copy()

            if self._use_key_pair:
                # Use key-pair authentication
                private_key_bytes = self._load_private_key()
                config['private_key'] = private_key_bytes
                logger.debug("Using key-pair authentication")
            elif self.authenticator:
                # Use authenticator (e.g., externalbrowser)
                # Authenticator already added to config in __init__
                logger.debug(f"Using {self.authenticator} authentication")
            else:
                # Use password authentication
                config['password'] = self.password
                logger.debug("Using password authentication")

            # Create connection
            self._connection = snowflake.connector.connect(**config)

            logger.info(f"Connected to Snowflake: {self.account}/{self.database_name}.{self.schema} (warehouse: {self.warehouse})")

        except ImportError:
            self._handle_connection_error(
                ImportError(
                    "snowflake-connector-python not installed. "
                    "Install with: pip install snowflake-connector-python"
                ),
                "connection"
            )
        except Exception as e:
            self._handle_connection_error(e, "connection")

    async def disconnect(self):
        """
        Disconnect from Snowflake.

        Closes the connection and releases resources.
        """
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                logger.info("Disconnected from Snowflake")
            except Exception as e:
                logger.warning(f"Error during disconnect: {str(e)}")
                self._connection = None

    async def query(self, sql: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        Run a SELECT query and return results.

        Args:
            sql: SQL query with %s or %(name)s placeholders
            params: List or dict of parameters for the query

        Returns:
            List of rows as dictionaries

        Example:
            results = await db.query("SELECT * FROM users WHERE age > %s", [25])
            results = await db.query("SELECT * FROM users WHERE age > %(min_age)s", {"min_age": 25})
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            rows = cursor.fetchall()

            # Handle empty result set
            if not rows:
                return []

            # Convert to list of dictionaries
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        finally:
            cursor.close()

    async def execute(self, sql: str, params: Optional[List] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE and return affected rows.

        Args:
            sql: SQL statement with %s or %(name)s placeholders
            params: List or dict of parameters for the statement

        Returns:
            Number of affected rows

        Example:
            affected = await db.execute("INSERT INTO users (name, age) VALUES (%s, %s)", ["Alice", 30])
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            rowcount = cursor.rowcount
            self._connection.commit()
            return rowcount

        finally:
            cursor.close()

    async def tables(self, schema: Optional[str] = None) -> List[str]:
        """
        List all tables in the database or specific schema.

        Args:
            schema: Schema name (defaults to current schema)

        Returns:
            List of table names
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            if schema:
                cursor.execute(f"SHOW TABLES IN SCHEMA {schema}")
            else:
                cursor.execute("SHOW TABLES")

            rows = cursor.fetchall()
            # SHOW TABLES returns multiple columns, table name is in 'name' column (index varies)
            # Fetch as dictionaries to be safe
            columns = [desc[0] for desc in cursor.description]
            name_idx = columns.index('name')
            return [row[name_idx] for row in rows]

        finally:
            cursor.close()

    async def schemas(self) -> List[str]:
        """
        List all schemas in the database.

        Returns:
            List of schema names
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            cursor.execute("SHOW SCHEMAS")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            name_idx = columns.index('name')
            return [row[name_idx] for row in rows]

        finally:
            cursor.close()

    async def describe(self, table: str) -> List[Dict[str, Any]]:
        """
        Get table column information.

        Args:
            table: Table name (optionally schema-qualified: schema.table)

        Returns:
            List of column details with name, type, nullable, default, etc.

        Example:
            columns = await db.describe("users")
            for col in columns:
                print(f"{col['name']}: {col['type']}")
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            cursor.execute(f"DESCRIBE TABLE {table}")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        finally:
            cursor.close()

    async def databases(self) -> List[str]:
        """
        List all accessible databases.

        Returns:
            List of database names
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            cursor.execute("SHOW DATABASES")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            name_idx = columns.index('name')
            return [row[name_idx] for row in rows]

        finally:
            cursor.close()

    async def list_warehouses(self) -> List[Dict[str, Any]]:
        """
        List all available warehouses.

        Returns:
            List of warehouse details with name, state, size, etc.
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            cursor.execute("SHOW WAREHOUSES")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        finally:
            cursor.close()

    async def switch_warehouse(self, warehouse: str) -> None:
        """
        Switch to a different warehouse.

        Args:
            warehouse: Warehouse name to switch to
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            cursor.execute(f"USE WAREHOUSE {warehouse}")
            self.warehouse = warehouse
            logger.info(f"Switched to warehouse: {warehouse}")

        finally:
            cursor.close()

    async def get_current_warehouse(self) -> Dict[str, Any]:
        """
        Get current warehouse information.

        Returns:
            Dictionary with current warehouse details
        """
        result = await self.query("SELECT CURRENT_WAREHOUSE() as warehouse")
        return result[0] if result else {"warehouse": None}

    async def query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent query history.

        Args:
            limit: Maximum number of queries to return (default: 100)

        Returns:
            List of query history records
        """
        sql = f"""
        SELECT
            query_id,
            query_text,
            database_name,
            schema_name,
            query_type,
            warehouse_name,
            user_name,
            role_name,
            execution_status,
            error_message,
            start_time,
            end_time,
            total_elapsed_time,
            bytes_scanned,
            rows_produced
        FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
        ORDER BY start_time DESC
        LIMIT {limit}
        """
        return await self.query(sql)

    async def get_warehouse_usage(self, warehouse: Optional[str] = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get warehouse credit usage.

        Args:
            warehouse: Warehouse name (defaults to current warehouse)
            days: Number of days to look back (default: 7)

        Returns:
            List of usage records
        """
        warehouse_filter = f"AND warehouse_name = '{warehouse}'" if warehouse else ""

        sql = f"""
        SELECT
            warehouse_name,
            DATE(start_time) as usage_date,
            SUM(credits_used) as total_credits
        FROM TABLE(INFORMATION_SCHEMA.WAREHOUSE_METERING_HISTORY(
            DATE_RANGE_START => DATEADD(day, -{days}, CURRENT_DATE())
        ))
        WHERE 1=1 {warehouse_filter}
        GROUP BY warehouse_name, usage_date
        ORDER BY usage_date DESC, warehouse_name
        """
        return await self.query(sql)

    async def list_stages(self) -> List[Dict[str, Any]]:
        """
        List all stages (internal and external).

        Returns:
            List of stage details
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            cursor.execute("SHOW STAGES")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        finally:
            cursor.close()

    async def put_file(self, local_path: str, stage_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Upload file to Snowflake stage.

        Args:
            local_path: Local file path
            stage_path: Stage location (e.g., "@my_stage/path/")
            overwrite: Whether to overwrite existing files

        Returns:
            Dictionary with upload results
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            overwrite_str = "OVERWRITE = TRUE" if overwrite else ""
            sql = f"PUT 'file://{local_path}' {stage_path} {overwrite_str}"
            cursor.execute(sql)

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            return {
                "success": True,
                "files_uploaded": len(results),
                "details": results
            }

        finally:
            cursor.close()

    async def get_file(self, stage_path: str, local_path: str) -> Dict[str, Any]:
        """
        Download file from Snowflake stage.

        Args:
            stage_path: Stage file location (e.g., "@my_stage/path/file.csv")
            local_path: Local directory to download to

        Returns:
            Dictionary with download results
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            sql = f"GET {stage_path} 'file://{local_path}'"
            cursor.execute(sql)

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            return {
                "success": True,
                "files_downloaded": len(results),
                "details": results
            }

        finally:
            cursor.close()

    async def load_from_stage(
        self,
        table: str,
        stage: str,
        file_format: str = "CSV",
        pattern: Optional[str] = None,
        on_error: str = "ABORT_STATEMENT"
    ) -> Dict[str, Any]:
        """
        Load data from stage into table.

        Args:
            table: Target table name
            stage: Stage location (e.g., "@my_stage/path/")
            file_format: File format name or type (default: "CSV")
            pattern: File pattern to match (e.g., r".*\\.csv")
            on_error: Error handling: ABORT_STATEMENT, CONTINUE, SKIP_FILE

        Returns:
            Dictionary with load results
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            pattern_clause = f"PATTERN = '{pattern}'" if pattern else ""

            sql = f"""
            COPY INTO {table}
            FROM {stage}
            FILE_FORMAT = (TYPE = {file_format})
            {pattern_clause}
            ON_ERROR = {on_error}
            """

            cursor.execute(sql)

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            # Calculate summary
            rows_loaded = sum(row.get('rows_loaded', 0) or 0 for row in results)

            return {
                "success": True,
                "rows_loaded": rows_loaded,
                "files_processed": len(results),
                "details": results
            }

        finally:
            cursor.close()

    async def create_stage(
        self,
        name: str,
        url: Optional[str] = None,
        storage_integration: Optional[str] = None,
        credentials: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Create a new stage (internal or external).

        Args:
            name: Stage name
            url: External URL (e.g., "s3://bucket/path/") for external stages
            storage_integration: Storage integration name for cloud storage
            credentials: Cloud credentials (e.g., {"AWS_KEY_ID": "...", "AWS_SECRET_KEY": "..."})

        Example:
            # Internal stage
            await db.create_stage("my_internal_stage")

            # External S3 stage
            await db.create_stage(
                "my_s3_stage",
                url="s3://my-bucket/data/",
                credentials={"AWS_KEY_ID": "...", "AWS_SECRET_KEY": "..."}
            )
        """
        if self._connection is None:
            await self.connect()

        cursor = self._connection.cursor()

        try:
            if url:
                # External stage
                sql = f"CREATE STAGE IF NOT EXISTS {name} URL = '{url}'"

                if storage_integration:
                    sql += f" STORAGE_INTEGRATION = {storage_integration}"
                elif credentials:
                    creds_str = " ".join([f"{k} = '{v}'" for k, v in credentials.items()])
                    sql += f" CREDENTIALS = ({creds_str})"
            else:
                # Internal stage
                sql = f"CREATE STAGE IF NOT EXISTS {name}"

            cursor.execute(sql)
            logger.info(f"Created stage: {name}")

        finally:
            cursor.close()

    def get_tools(self) -> List['AgentTool']:
        """
        Expose Snowflake operations as agent tools.

        Returns:
            List of AgentTool instances for Snowflake operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="query_database",
                description="Execute a SQL SELECT query on Snowflake and return results as a list of dictionaries",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query with %s placeholders for parameters"
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
                plugin_name="Snowflake",
                timeout_seconds=60
            ),
            AgentTool(
                name="execute_sql",
                description="Execute a SQL INSERT, UPDATE, or DELETE statement on Snowflake and return the number of affected rows",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL statement with %s placeholders for parameters"
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
                plugin_name="Snowflake",
                timeout_seconds=60
            ),
            AgentTool(
                name="list_tables",
                description="List all tables in the Snowflake database or a specific schema",
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {
                            "type": "string",
                            "description": "Optional schema name to list tables from (defaults to current schema)"
                        }
                    },
                    "required": []
                },
                handler=self._tool_list_tables,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=30
            ),
            AgentTool(
                name="list_schemas",
                description="List all schemas in the Snowflake database",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=self._tool_list_schemas,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=30
            ),
            AgentTool(
                name="get_table_schema",
                description="Get detailed column information for a Snowflake table including column names, types, and constraints",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Table name (optionally schema-qualified like 'schema.table')"
                        }
                    },
                    "required": ["table"]
                },
                handler=self._tool_get_table_schema,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=30
            ),
            AgentTool(
                name="list_warehouses",
                description="List all available Snowflake compute warehouses with their status and configuration",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=self._tool_list_warehouses,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=30
            ),
            AgentTool(
                name="switch_warehouse",
                description="Switch to a different Snowflake compute warehouse for subsequent queries",
                parameters={
                    "type": "object",
                    "properties": {
                        "warehouse": {
                            "type": "string",
                            "description": "Name of the warehouse to switch to"
                        }
                    },
                    "required": ["warehouse"]
                },
                handler=self._tool_switch_warehouse,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=30
            ),
            AgentTool(
                name="get_query_history",
                description="Get recent query history from Snowflake including execution status, timing, and resource usage",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of queries to return (default: 100)",
                            "default": 100
                        }
                    },
                    "required": []
                },
                handler=self._tool_get_query_history,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=45
            ),
            AgentTool(
                name="list_stages",
                description="List all Snowflake stages (internal and external) for data loading",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=self._tool_list_stages,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=30
            ),
            AgentTool(
                name="upload_to_stage",
                description="Upload a local file to a Snowflake stage for data loading",
                parameters={
                    "type": "object",
                    "properties": {
                        "local_path": {
                            "type": "string",
                            "description": "Local file path to upload"
                        },
                        "stage_path": {
                            "type": "string",
                            "description": "Stage location (e.g., '@my_stage/path/')"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "Whether to overwrite existing files (default: false)",
                            "default": False
                        }
                    },
                    "required": ["local_path", "stage_path"]
                },
                handler=self._tool_upload_to_stage,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=120
            ),
            AgentTool(
                name="load_from_stage",
                description="Load data from a Snowflake stage into a table using COPY INTO command",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Target table name"
                        },
                        "stage": {
                            "type": "string",
                            "description": "Stage location (e.g., '@my_stage/path/')"
                        },
                        "file_format": {
                            "type": "string",
                            "description": "File format type (default: CSV)",
                            "default": "CSV"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Optional file pattern to match (e.g., '.*\\.csv')"
                        }
                    },
                    "required": ["table", "stage"]
                },
                handler=self._tool_load_from_stage,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=180
            ),
            AgentTool(
                name="create_stage",
                description="Create a new Snowflake stage (internal or external) for data loading",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Stage name"
                        },
                        "url": {
                            "type": "string",
                            "description": "External URL for external stages (e.g., 's3://bucket/path/')"
                        },
                        "storage_integration": {
                            "type": "string",
                            "description": "Storage integration name for cloud storage"
                        }
                    },
                    "required": ["name"]
                },
                handler=self._tool_create_stage,
                category="database",
                source="plugin",
                plugin_name="Snowflake",
                timeout_seconds=30
            ),
        ]

    # Tool handler methods

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

    async def _tool_execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for execute_sql"""
        sql = args.get("sql")
        params = args.get("params")

        affected_rows = await self.execute(sql, params)

        return {
            "success": True,
            "affected_rows": affected_rows
        }

    async def _tool_list_tables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_tables"""
        schema = args.get("schema")

        tables = await self.tables(schema)

        return {
            "success": True,
            "tables": tables,
            "count": len(tables)
        }

    async def _tool_list_schemas(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_schemas"""
        schemas = await self.schemas()

        return {
            "success": True,
            "schemas": schemas,
            "count": len(schemas)
        }

    async def _tool_get_table_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_table_schema"""
        table = args.get("table")

        columns = await self.describe(table)

        return {
            "success": True,
            "table": table,
            "columns": columns,
            "column_count": len(columns)
        }

    async def _tool_list_warehouses(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_warehouses"""
        warehouses = await self.list_warehouses()

        return {
            "success": True,
            "warehouses": warehouses,
            "count": len(warehouses)
        }

    async def _tool_switch_warehouse(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for switch_warehouse"""
        warehouse = args.get("warehouse")

        await self.switch_warehouse(warehouse)

        return {
            "success": True,
            "message": f"Switched to warehouse: {warehouse}",
            "warehouse": warehouse
        }

    async def _tool_get_query_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for get_query_history"""
        limit = args.get("limit", 100)

        history = await self.query_history(limit)

        return {
            "success": True,
            "queries": history,
            "count": len(history)
        }

    async def _tool_list_stages(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_stages"""
        stages = await self.list_stages()

        return {
            "success": True,
            "stages": stages,
            "count": len(stages)
        }

    async def _tool_upload_to_stage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for upload_to_stage"""
        local_path = args.get("local_path")
        stage_path = args.get("stage_path")
        overwrite = args.get("overwrite", False)

        result = await self.put_file(local_path, stage_path, overwrite)

        return result

    async def _tool_load_from_stage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for load_from_stage"""
        table = args.get("table")
        stage = args.get("stage")
        file_format = args.get("file_format", "CSV")
        pattern = args.get("pattern")

        result = await self.load_from_stage(table, stage, file_format, pattern)

        return result

    async def _tool_create_stage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for create_stage"""
        name = args.get("name")
        url = args.get("url")
        storage_integration = args.get("storage_integration")

        await self.create_stage(name, url, storage_integration)

        return {
            "success": True,
            "message": f"Created stage: {name}",
            "stage": name
        }


def snowflake(**kwargs) -> SnowflakePlugin:
    """
    Create Snowflake plugin with simplified interface.

    Args:
        **kwargs: Connection parameters (account, warehouse, database, etc.)

    Returns:
        SnowflakePlugin instance

    Example:
        from daita.plugins import snowflake

        db = snowflake(
            account="xy12345",
            warehouse="COMPUTE_WH",
            database="MYDB",
            user="myuser",
            password="mypass"
        )
    """
    return SnowflakePlugin(**kwargs)
