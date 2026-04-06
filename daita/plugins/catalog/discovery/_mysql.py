"""MySQL/MariaDB schema discovery."""

import logging
from typing import Any, Dict, Optional

from ._utils import parse_conn_url, redact_url, ssl_context

logger = logging.getLogger(__name__)


async def discover_mysql(
    connection_string: str,
    schema: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Connect to a MySQL/MariaDB database and extract its schema.

    Returns a raw result dict with tables, columns, and foreign keys.
    """
    try:
        import aiomysql
    except ImportError:
        raise ImportError(
            "aiomysql is required. Install with: pip install 'daita-agents[mysql]'"
        )

    logger.debug("discover_mysql: connecting to %s", redact_url(connection_string))
    creds = parse_conn_url(connection_string)
    db_name = schema or creds["database"] or "mysql"

    conn = await aiomysql.connect(
        host=creds["host"],
        port=creds["port"] or 3306,
        user=creds["user"],
        password=creds["password"],
        db=db_name,
        ssl=ssl_context(),
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
                    COLUMN_KEY as column_key,
                    COLUMN_COMMENT as column_comment
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

        return {
            "database_type": "mysql",
            "schema": db_name,
            "tables": tables,
            "columns": columns,
            "foreign_keys": fkeys,
            "table_count": len(tables),
            "column_count": len(columns),
        }

    finally:
        conn.close()
