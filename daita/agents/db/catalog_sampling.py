"""Construction-time sample enrichment for catalog-bound DB profiles."""

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from .catalog_profile import is_numeric_type
from .tools.analyst._helpers import quote_id, quote_path, safe_query

if TYPE_CHECKING:
    from ...plugins.base_db import BaseDatabasePlugin

logger = logging.getLogger(__name__)

PII_COLUMN_PATTERNS: List[str] = [
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "email",
    "phone",
    "mobile",
    "ssn",
    "social_security",
    "credit_card",
    "card_number",
    "cvv",
    "pin",
    "dob",
    "date_of_birth",
    "birth_date",
    "address",
    "street",
    "zip",
    "postal",
    "passport",
    "national_id",
]


async def sample_numeric_columns(
    plugin: "BaseDatabasePlugin",
    schema: Dict[str, Any],
    redact_pii: bool = True,
    sample_size: int = 5,
) -> None:
    """
    Attach up to *sample_size* non-null sample values to numeric columns.

    The profile is mutated before it is registered with CatalogPlugin. Sampling
    uses the existing DB plugin connection and silently skips tables/columns
    that cannot be sampled.
    """
    db_type = schema.get("database_type", "")
    if db_type == "mongodb":
        return

    for table in schema.get("tables", []):
        table_name = table["name"]
        for column in table.get("columns", []):
            column_name = column["name"]
            if not is_numeric_type(column.get("type", "")):
                continue

            if redact_pii:
                lowered = column_name.lower()
                if any(pattern in lowered for pattern in PII_COLUMN_PATTERNS):
                    continue

            try:
                dialect = getattr(plugin, "sql_dialect", db_type)
                quoted_col = quote_id(column_name, dialect)
                sql = (
                    f"SELECT {quoted_col} FROM {quote_path(table_name, dialect)} "
                    f"WHERE {quoted_col} IS NOT NULL LIMIT {sample_size}"
                )

                rows = (await safe_query(plugin, sql)).rows
                if rows:
                    samples = [row[column_name] for row in rows if column_name in row]
                    if samples:
                        column["_samples"] = samples
            except Exception as exc:
                logger.debug(f"Sample query failed for {table_name}.{column_name}: {exc}")
