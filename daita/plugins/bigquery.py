"""
BigQuery plugin for Daita Agents.

Provides Google BigQuery data warehouse connection and querying capabilities.
Supports service account and application default credentials.

Example:
    ```python
    from daita.plugins import bigquery

    async with bigquery(project="my-project", dataset="analytics") as bq:
        results = await bq.query("SELECT * FROM users LIMIT 10")
        tables = await bq.tables()
    ```
"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base_db import BaseDatabasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


class BigQueryPlugin(BaseDatabasePlugin):
    """
    BigQuery plugin for agents with query execution and schema inspection.

    Wraps the synchronous google-cloud-bigquery SDK with asyncio executors,
    following the same pattern as the Snowflake plugin.
    """

    sql_dialect = "bigquery"

    def __init__(
        self,
        project: Optional[str] = None,
        dataset: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: Optional[str] = None,
        timeout: int = 300,
        **kwargs,
    ):
        self.project = (
            project
            or os.getenv("BIGQUERY_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        self.dataset = dataset or os.getenv("BIGQUERY_DATASET")
        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.location = location or os.getenv("BIGQUERY_LOCATION", "US")
        self.timeout = timeout

        if not self.project:
            raise ValueError(
                "BigQuery project is required. Provide 'project' parameter "
                "or set BIGQUERY_PROJECT / GOOGLE_CLOUD_PROJECT environment variable."
            )

        super().__init__(timeout=timeout, **kwargs)
        logger.debug(
            f"BigQueryPlugin configured for project={self.project}, dataset={self.dataset}"
        )

    async def connect(self) -> None:
        if self._client is not None:
            return

        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery is required for BigQueryPlugin. "
                "Install with: pip install 'daita-agents[bigquery]'"
            )

        def _create_client():
            client_kwargs = {"project": self.project, "location": self.location}
            if self.credentials_path:
                from google.oauth2 import service_account

                creds = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                client_kwargs["credentials"] = creds
            return bigquery.Client(**client_kwargs)

        loop = asyncio.get_running_loop()
        self._client = await loop.run_in_executor(None, _create_client)
        logger.info(f"Connected to BigQuery project={self.project}")

    async def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            loop = asyncio.get_running_loop()
            client = self._client
            self._client = None
            await loop.run_in_executor(None, client.close)
            logger.info("Disconnected from BigQuery")
        except Exception as e:
            logger.warning(f"Error during BigQuery disconnect: {e}")
            self._client = None

    # ── Sync helpers (called via run_in_executor) ─────────────────────

    def _prepare_job(self, sql: str, params: Optional[List]):
        """Rewrite %s placeholders to @p0, @p1, ... and build a QueryJobConfig."""
        from google.cloud import bigquery as bq_mod

        if params:
            idx = [0]

            def _replacer(match):
                name = f"@p{idx[0]}"
                idx[0] += 1
                return name

            sql = re.sub(r"%s", _replacer, sql)

            # bool must be checked before int (since bool is a subclass of int)
            type_map = {bool: "BOOL", int: "INT64", float: "FLOAT64", str: "STRING"}
            bq_params = [
                bq_mod.ScalarQueryParameter(
                    f"p{i}",
                    next(
                        (v for k, v in type_map.items() if isinstance(val, k)), "STRING"
                    ),
                    val,
                )
                for i, val in enumerate(params)
            ]
            job_config = bq_mod.QueryJobConfig(query_parameters=bq_params)
        else:
            job_config = bq_mod.QueryJobConfig()

        return sql, job_config

    def _run_query(
        self, sql: str, params: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        sql, job_config = self._prepare_job(sql, params)
        job = self._client.query(sql, job_config=job_config, timeout=self.timeout)
        return [dict(row) for row in job.result(timeout=self.timeout)]

    def _run_execute(self, sql: str, params: Optional[List] = None) -> int:
        sql, job_config = self._prepare_job(sql, params)
        job = self._client.query(sql, job_config=job_config, timeout=self.timeout)
        job.result(timeout=self.timeout)
        return job.num_dml_affected_rows or 0

    # ── Core async methods ────────────────────────────────────────────

    async def query(
        self, sql: str, params: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        sql = self._normalize_sql(sql)
        if self._client is None:
            await self.connect()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_query, sql, params)

    async def execute(self, sql: str, params: Optional[List] = None) -> int:
        sql = self._normalize_sql(sql)
        if self._client is None:
            await self.connect()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_execute, sql, params)

    async def tables(self, dataset: Optional[str] = None) -> List[str]:
        ds = dataset or self.dataset
        if not ds:
            raise ValueError(
                "Dataset is required. Provide 'dataset' parameter or set BIGQUERY_DATASET."
            )
        if self._client is None:
            await self.connect()

        def _list():
            return [
                t.table_id for t in self._client.list_tables(f"{self.project}.{ds}")
            ]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _list)

    async def describe(self, table: str) -> List[Dict[str, Any]]:
        if self._client is None:
            await self.connect()
        ref = self._fqn(table)

        def _describe():
            schema = self._client.get_table(ref).schema
            return [
                {
                    "column_name": field.name,
                    "data_type": field.field_type,
                    "is_nullable": "YES" if field.mode != "REQUIRED" else "NO",
                }
                for field in schema
            ]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _describe)

    async def datasets(self) -> List[str]:
        if self._client is None:
            await self.connect()

        def _list():
            return [ds.dataset_id for ds in self._client.list_datasets()]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _list)

    async def count_rows(self, table: str, filter: Optional[str] = None) -> int:
        fqn = self._fqn(table)
        sql = f"SELECT COUNT(*) as cnt FROM `{fqn}`"
        if filter:
            sql += f" WHERE {filter}"
        rows = await self.query(sql)
        return rows[0]["cnt"] if rows else 0

    async def sample_rows(self, table: str, n: int = 5) -> List[Dict[str, Any]]:
        fqn = self._fqn(table)
        sql = f"SELECT * FROM `{fqn}` ORDER BY RAND() LIMIT {int(n)}"
        return await self.query(sql)

    def _fqn(self, table: str) -> str:
        """Build a fully-qualified table name if not already qualified."""
        if "." not in table and self.dataset:
            return f"{self.project}.{self.dataset}.{table}"
        if table.count(".") == 1 and self.project:
            return f"{self.project}.{table}"
        return table

    # ── Tools ─────────────────────────────────────────────────────────

    def get_tools(self) -> List["AgentTool"]:
        from ..core.tools import AgentTool

        _common = dict(category="database", source="plugin", plugin_name="BigQuery")

        tools = [
            AgentTool(
                name="bigquery_query",
                description="Run a SELECT query on BigQuery. Include LIMIT in your SQL to control result size.",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query with %s placeholders",
                        },
                        "params": {
                            "type": "array",
                            "description": "Optional parameter values",
                            "items": {},
                        },
                        "focus": {
                            "type": "string",
                            "description": "Focus DSL to filter/project results",
                        },
                    },
                    "required": ["sql"],
                },
                handler=self._tool_query,
                timeout_seconds=120,
                **_common,
            ),
            AgentTool(
                name="bigquery_inspect",
                description="List tables and their column schemas in one call.",
                parameters={
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "description": "Dataset name (uses default if omitted)",
                        },
                        "tables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to specific tables",
                        },
                    },
                    "required": [],
                },
                handler=self._tool_inspect,
                timeout_seconds=60,
                **_common,
            ),
            AgentTool(
                name="bigquery_count",
                description="Count rows in a BigQuery table, optionally filtered.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"},
                        "filter": {
                            "type": "string",
                            "description": "Optional WHERE clause (without WHERE keyword)",
                        },
                    },
                    "required": ["table"],
                },
                handler=self._tool_count,
                timeout_seconds=60,
                **_common,
            ),
            AgentTool(
                name="bigquery_sample",
                description="Return a random sample of rows from a BigQuery table.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"},
                        "n": {
                            "type": "integer",
                            "description": "Number of rows to sample (default: 5)",
                        },
                    },
                    "required": ["table"],
                },
                handler=self._tool_sample,
                timeout_seconds=60,
                **_common,
            ),
            AgentTool(
                name="bigquery_list_datasets",
                description="List all datasets in the BigQuery project.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=self._tool_list_datasets,
                timeout_seconds=30,
                **_common,
            ),
        ]

        if not self.read_only:
            tools.append(
                AgentTool(
                    name="bigquery_execute",
                    description="Execute INSERT, UPDATE, DELETE, or DDL on BigQuery. Returns affected row count.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL DML/DDL statement",
                            },
                            "params": {
                                "type": "array",
                                "description": "Optional parameter values",
                                "items": {},
                            },
                        },
                        "required": ["sql"],
                    },
                    handler=self._tool_execute,
                    timeout_seconds=120,
                    **_common,
                )
            )

        return tools

    # ── Tool handlers ─────────────────────────────────────────────────

    async def _tool_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sql = self._normalize_sql(args["sql"])
        params = args.get("params") or []
        focus_dsl = args.get("focus")

        if focus_dsl:
            results = await self._run_focus_query(sql, params, focus_dsl)
        else:
            if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
                sql = f"{sql} LIMIT 50"
            results = await self.query(sql, params or None)

        truncated = self._truncate_result(results)
        return {
            "rows": truncated["rows"],
            "total_rows": truncated["total_rows"],
            "truncated": truncated["truncated"],
        }

    async def _tool_execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        affected = await self.execute(args["sql"], args.get("params"))
        return {"affected_rows": affected}

    async def _tool_inspect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        dataset = args.get("dataset")
        filter_tables = args.get("tables")

        all_tables = await self.tables(dataset)
        targets = (
            [t for t in all_tables if t in filter_tables]
            if filter_tables
            else all_tables
        )

        total_tables = len(targets)
        truncated = total_tables > 50
        targets = targets[:50]

        schemas = await asyncio.gather(*[self.describe(t) for t in targets])

        return {
            "tables": [
                {"name": t, "columns": [self._compact_column(c) for c in s]}
                for t, s in zip(targets, schemas)
            ],
            "total_tables": total_tables,
            "truncated": truncated,
        }

    async def _tool_count(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table = args["table"]
        count = await self.count_rows(table, args.get("filter"))
        return {"table": table, "count": count}

    async def _tool_sample(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table = args["table"]
        rows = await self.sample_rows(table, args.get("n", 5))
        return {"table": table, "rows": rows}

    async def _tool_list_datasets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ds_list = await self.datasets()
        return {"datasets": ds_list}


def bigquery(**kwargs) -> BigQueryPlugin:
    """Create a BigQuery plugin instance."""
    return BigQueryPlugin(**kwargs)
