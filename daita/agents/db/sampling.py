"""
Numeric column sampling and PII-column redaction patterns.
"""

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from .schema import NUMERIC_TYPES, is_numeric_type
from .tools._helpers import quote_id, quote_path, safe_query

if TYPE_CHECKING:
    from ...plugins.base_db import BaseDatabasePlugin

logger = logging.getLogger(__name__)

# PII column name patterns — skip sampling these to avoid leaking sensitive data
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
    Attach up to *sample_size* non-null sample values to numeric columns in *schema*.

    Mutates *schema* in place — adds a ``_samples`` list to qualifying columns.
    Uses the existing plugin connection (no new connection opened).
    Silently skips any table/column that fails to sample.
    """
    db_type = schema.get("database_type", "")
    # MongoDB uses a document model — numeric sampling is handled differently; skip.
    if db_type == "mongodb":
        return

    for table in schema.get("tables", []):
        tname = table["name"]
        for col in table.get("columns", []):
            col_name = col["name"]

            if not is_numeric_type(col.get("type", "")):
                continue

            if redact_pii:
                col_lower = col_name.lower()
                if any(pat in col_lower for pat in PII_COLUMN_PATTERNS):
                    continue

            try:
                dialect = getattr(plugin, "sql_dialect", db_type)
                quoted_col = quote_id(col_name, dialect)
                sql = (
                    f"SELECT {quoted_col} FROM {quote_path(tname, dialect)} "
                    f"WHERE {quoted_col} IS NOT NULL LIMIT {sample_size}"
                )

                rows = (await safe_query(plugin, sql)).rows
                if rows:
                    samples = [row[col_name] for row in rows if col_name in row]
                    if samples:
                        col["_samples"] = samples
            except Exception as exc:
                logger.debug(f"Sample query failed for {tname}.{col_name}: {exc}")
