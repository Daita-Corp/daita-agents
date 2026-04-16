"""Bigtable instance and table schema discovery."""

import logging
from typing import Any, Optional

from ._gcp_common import gcp_credentials

logger = logging.getLogger(__name__)


async def discover_bigtable(
    project: str,
    instance: str,
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
) -> dict[str, Any]:
    """List tables in a Bigtable instance and their column families.

    Bigtable has no rowkey schema to introspect, so the "columns" returned
    are the column families plus the implicit row_key primary key.
    """
    try:
        from google.cloud import bigtable
    except ImportError:
        raise ImportError(
            "google-cloud-bigtable is required. "
            "Install with: pip install 'daita-agents[gcp]'"
        )

    logger.debug("discover_bigtable: listing %s/%s", project, instance)

    creds, _ = gcp_credentials(
        credentials_path=credentials_path,
        impersonate_service_account=impersonate_service_account,
    )
    client = bigtable.Client(project=project, credentials=creds, admin=True)
    instance_obj = client.instance(instance)

    tables_out: list[dict[str, Any]] = []
    try:
        tables = list(instance_obj.list_tables())
    except Exception as exc:
        logger.warning(
            "Bigtable list_tables failed for %s/%s: %s", project, instance, exc
        )
        tables = []

    for table in tables:
        families: list[str] = []
        try:
            families = sorted(table.list_column_families().keys())
        except Exception as exc:
            logger.debug(
                "Bigtable list_column_families failed for %s: %s", table.name, exc
            )

        tables_out.append(
            {
                "table_name": table.table_id,
                "row_count": None,  # Bigtable doesn't expose this cheaply
                "column_families": families,
            }
        )

    return {
        "database_type": "bigtable",
        "project": project,
        "instance": instance,
        "tables": tables_out,
        "table_count": len(tables_out),
    }
