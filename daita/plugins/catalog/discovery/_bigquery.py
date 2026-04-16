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

    return {
        "database_type": "bigquery",
        "project": project,
        "dataset": dataset,
        "location": location,
        "description": description,
        "tables": tables,
        "columns": columns,
        "foreign_keys": [],
        "table_count": len(tables),
        "column_count": len(columns),
    }
