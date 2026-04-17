"""BigQuery dataset schema discovery."""

import logging
from typing import Any, Optional

from ._gcp_common import gcp_credentials

logger = logging.getLogger(__name__)


async def discover_bigquery(
    project: str,
    dataset: str,
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
    max_tables: int = 1000,
) -> dict[str, Any]:
    """List tables in a BigQuery dataset and extract their schemas.

    Metadata-only — never issues SELECT. Returns a raw result dict with
    tables, columns, and dataset-level metadata.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        raise ImportError(
            "google-cloud-bigquery is required. "
            "Install with: pip install 'daita-agents[gcp]'"
        )

    logger.debug("discover_bigquery: listing %s.%s", project, dataset)

    creds, _ = gcp_credentials(
        credentials_path=credentials_path,
        impersonate_service_account=impersonate_service_account,
    )
    client = bigquery.Client(project=project, credentials=creds)
    dataset_ref = f"{project}.{dataset}"

    try:
        ds_meta = client.get_dataset(dataset_ref)
        location = ds_meta.location or ""
        description = ds_meta.description or ""
    except Exception as exc:
        logger.warning("BigQuery get_dataset failed for %s: %s", dataset_ref, exc)
        location = ""
        description = ""

    tables: list[dict[str, Any]] = []
    columns: list[dict[str, Any]] = []

    try:
        listed = list(client.list_tables(dataset_ref, max_results=max_tables))
    except Exception as exc:
        logger.warning("BigQuery list_tables failed for %s: %s", dataset_ref, exc)
        listed = []

    for table_ref in listed:
        table_name = table_ref.table_id
        try:
            table = client.get_table(table_ref)
        except Exception as exc:
            logger.debug("BigQuery get_table failed for %s: %s", table_name, exc)
            continue

        tables.append(
            {
                "table_name": table_name,
                "row_count": table.num_rows,
                "size_bytes": table.num_bytes,
                "table_type": table.table_type,
            }
        )
        for field in table.schema:
            columns.append(
                {
                    "table_name": table_name,
                    "column_name": field.name,
                    "data_type": field.field_type,
                    "is_nullable": "NO" if field.mode == "REQUIRED" else "YES",
                    "column_comment": field.description or None,
                }
            )

    # BigQuery supports PRIMARY KEY NOT ENFORCED and FOREIGN KEY NOT ENFORCED
    # as declared constraints. INFORMATION_SCHEMA views expose them and are
    # metadata-only — zero bytes scanned, no quota cost. Failures here
    # (permission / legacy dataset) degrade gracefully to empty lists.
    primary_keys, foreign_keys = _discover_bq_constraints(client, project, dataset)

    return {
        "database_type": "bigquery",
        "project": project,
        "dataset": dataset,
        "location": location,
        "description": description,
        "tables": tables,
        "columns": columns,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "table_count": len(tables),
        "column_count": len(columns),
    }


def _discover_bq_constraints(
    client: Any, project: str, dataset: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch declared primary- and foreign-key constraints for a BQ dataset.

    Uses ``INFORMATION_SCHEMA.TABLE_CONSTRAINTS`` + ``KEY_COLUMN_USAGE`` +
    ``CONSTRAINT_COLUMN_USAGE``. Returns ``([], [])`` on any failure so the
    caller never has to handle exceptions.
    """
    sql = f"""
    SELECT
        tc.constraint_type,
        kcu.table_name      AS source_table,
        kcu.column_name     AS source_column,
        kcu.ordinal_position,
        ccu.table_name      AS target_table,
        ccu.column_name     AS target_column
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` tc
    JOIN `{project}.{dataset}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` kcu
      USING (constraint_catalog, constraint_schema, constraint_name)
    LEFT JOIN `{project}.{dataset}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE` ccu
      USING (constraint_catalog, constraint_schema, constraint_name)
    WHERE tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
    ORDER BY tc.constraint_type, kcu.table_name, kcu.ordinal_position
    """
    primary_keys: list[dict[str, Any]] = []
    foreign_keys: list[dict[str, Any]] = []
    try:
        for row in client.query(sql).result():
            if row.constraint_type == "PRIMARY KEY":
                primary_keys.append(
                    {
                        "table_name": row.source_table,
                        "column_name": row.source_column,
                    }
                )
            else:  # FOREIGN KEY
                foreign_keys.append(
                    {
                        "source_table": row.source_table,
                        "source_column": row.source_column,
                        "target_table": row.target_table,
                        "target_column": row.target_column,
                    }
                )
    except Exception as exc:
        logger.debug(
            "BigQuery INFORMATION_SCHEMA constraint query failed for %s.%s: %s",
            project,
            dataset,
            exc,
        )
    return primary_keys, foreign_keys
