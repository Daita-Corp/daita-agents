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
    try:
        import asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
        )

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
        all_schemas = schema in ("*", "all")
        schema_filter = (
            "t.table_schema NOT IN ('pg_catalog', 'information_schema')"
            if all_schemas
            else "t.table_schema = $1"
        )
        column_schema_filter = (
            "c.table_schema NOT IN ('pg_catalog', 'information_schema')"
            if all_schemas
            else "c.table_schema = $1"
        )
        constraint_schema_filter = (
            "tc.table_schema NOT IN ('pg_catalog', 'information_schema')"
            if all_schemas
            else "tc.table_schema = $1"
        )
        index_schema_filter = (
            "schemaname NOT IN ('pg_catalog', 'information_schema')"
            if all_schemas
            else "schemaname = $1"
        )
        params = [] if all_schemas else [schema]
        table_name_expr = (
            "t.table_schema || '.' || t.table_name" if all_schemas else "t.table_name"
        )
        column_table_expr = (
            "c.table_schema || '.' || c.table_name" if all_schemas else "c.table_name"
        )
        constraint_table_expr = (
            "tc.table_schema || '.' || tc.table_name"
            if all_schemas
            else "tc.table_name"
        )
        fk_target_table_expr = (
            "ccu.table_schema || '.' || ccu.table_name"
            if all_schemas
            else "ccu.table_name"
        )
        index_table_expr = (
            "schemaname || '.' || tablename" if all_schemas else "tablename"
        )

        # Get tables
        tables = await conn.fetch(
            f"""
            SELECT {table_name_expr} AS table_name,
                   pg_stat_user_tables.n_live_tup as row_count
            FROM information_schema.tables t
            LEFT JOIN pg_stat_user_tables
                ON t.table_name = relname AND t.table_schema = schemaname
            WHERE {schema_filter}
            AND table_type = 'BASE TABLE'
            ORDER BY t.table_schema, t.table_name
        """,
            *params,
        )

        # Get columns (with optional column-level comments from pg_description)
        columns = await conn.fetch(
            f"""
            SELECT
                {column_table_expr} AS table_name,
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
            WHERE {column_schema_filter}
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """,
            *params,
        )

        # Get primary keys
        pkeys = await conn.fetch(
            f"""
            SELECT
                {constraint_table_expr} AS table_name,
                kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
            AND {constraint_schema_filter}
        """,
            *params,
        )

        # Get foreign keys
        fkeys = await conn.fetch(
            f"""
            SELECT
                {constraint_table_expr} as source_table,
                kcu.column_name as source_column,
                {fk_target_table_expr} AS target_table,
                ccu.column_name AS target_column,
                tc.constraint_name,
                rc.delete_rule,
                rc.update_rule
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.constraint_schema = tc.constraint_schema
            JOIN information_schema.referential_constraints rc
                ON tc.constraint_name = rc.constraint_name
                AND tc.constraint_schema = rc.constraint_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND {constraint_schema_filter}
        """,
            *params,
        )

        # Get indexes
        indexes = await conn.fetch(
            f"""
            SELECT
                {index_table_expr} AS tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE {index_schema_filter}
        """,
            *params,
        )

        return {
            "database_type": "postgresql",
            "schema": "all" if all_schemas else schema,
            "host": creds["host"],
            "port": creds["port"] or 5432,
            "database": creds["database"] or "postgres",
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
