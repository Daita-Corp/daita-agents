"""
Numeric column sampling and PII-column redaction patterns.
"""

import logging
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...plugins.base_db import BaseDatabasePlugin

logger = logging.getLogger(__name__)

# PII column name patterns — skip sampling these to avoid leaking sensitive data
PII_COLUMN_PATTERNS: List[str] = [
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "email", "phone", "mobile", "ssn", "social_security",
    "credit_card", "card_number", "cvv", "pin",
    "dob", "date_of_birth", "birth_date",
    "address", "street", "zip", "postal",
    "passport", "national_id",
]

# Numeric SQL types eligible for sample-value collection
NUMERIC_TYPES: List[str] = [
    "integer", "int", "bigint", "smallint", "tinyint",
    "numeric", "decimal", "float", "double", "real",
    "number", "money", "smallmoney",
    "int4", "int8", "int2", "float4", "float8",
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
            col_type = col.get("type", "").lower()

            if not any(nt in col_type for nt in NUMERIC_TYPES):
                continue

            if redact_pii:
                col_lower = col_name.lower()
                if any(pat in col_lower for pat in PII_COLUMN_PATTERNS):
                    continue

            try:
                if db_type == "postgresql":
                    sql = (
                        f'SELECT "{col_name}" FROM "{tname}" '
                        f'WHERE "{col_name}" IS NOT NULL LIMIT {sample_size}'
                    )
                elif db_type == "mysql":
                    sql = (
                        f"SELECT `{col_name}` FROM `{tname}` "
                        f"WHERE `{col_name}` IS NOT NULL LIMIT {sample_size}"
                    )
                else:
                    sql = (
                        f'SELECT "{col_name}" FROM "{tname}" '
                        f'WHERE "{col_name}" IS NOT NULL LIMIT {sample_size}'
                    )

                rows = await plugin.query(sql)
                if rows:
                    samples = [row[col_name] for row in rows if col_name in row]
                    if samples:
                        col["_samples"] = samples
            except Exception as exc:
                logger.debug(f"Sample query failed for {tname}.{col_name}: {exc}")
