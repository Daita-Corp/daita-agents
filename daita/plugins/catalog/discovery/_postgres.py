"""PostgreSQL schema discovery."""

import logging
from typing import Any, Dict

from ._utils import parse_conn_url, redact_url, ssl_context

logger = logging.getLogger(__name__)


async def discover_postgres(
    connection_string: str,
    schema: str = "public",
    ssl_mode: str = "verify-full",
) -> Dict[str, Any]:
    """
    Connect to a PostgreSQL database and extract its schema.

    Returns a raw result dict with tables, columns, primary keys, foreign keys,
    and indexes. Caller is responsible for persistence.
    """
    import asyncpg

    logger.debug(
        "discover_postgres: connecting to %s (ssl_mode=%s)",
        redact_url(connection_string),
        ssl_mode,
    )
    creds = parse_conn_url(connection_string)
    ssl_arg: Any = False if ssl_mode == "disable" else ssl_context(ssl_mode)
    conn = await asyncpg.connect(
        host=creds["host"],
        port=creds["port"] or 5432,
        user=creds["user"],
        password=creds["password"],
        database=creds["database"] or "postgres",
        ssl=ssl_arg,
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

        # Get columns (with optional column-level comments from pg_description)
        columns = await conn.fetch(
            """
            SELECT
                c.table_name,
                c.column_name,
                c.data_type,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                c.is_nullable,
                c.column_default,
                d.description AS column_comment
            FROM information_schema.columns c
            LEFT JOIN pg_catalog.pg_description d
                ON d.objoid = (
                    quote_ident(c.table_schema) || '.' || quote_ident(c.table_name)
                )::regclass
                AND d.objsubid = c.ordinal_position
            WHERE c.table_schema = $1
            ORDER BY c.table_name, c.ordinal_position
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

        return {
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

    finally:
        await conn.close()
