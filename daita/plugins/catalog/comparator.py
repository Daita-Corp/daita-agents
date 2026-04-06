"""
Schema comparison utilities.

Compares two normalized schema dicts and returns added, removed, and modified
elements useful for migration planning.
"""

from typing import Any, Dict, List, Tuple


def _extract_tables(schema: Dict[str, Any]) -> Dict[str, Dict]:
    """Extract table name → table dict, handling both normalized and MongoDB raw formats."""
    db_type = schema.get("database_type", "")
    if db_type == "mongodb" and "collections" in schema:
        # Raw discover_mongodb output (used directly by compare_schemas tool)
        return {c["collection_name"]: c for c in schema.get("collections", [])}
    return {t["name"]: t for t in schema.get("tables", [])}


def _extract_columns(schema: Dict[str, Any]) -> Dict[Tuple[str, str], Dict]:
    """Flatten nested tables[].columns[] into (table_name, col_name) → column dict."""
    columns: Dict[Tuple[str, str], Dict] = {}
    for table in schema.get("tables", []):
        table_name = table["name"]
        for col in table.get("columns", []):
            columns[(table_name, col["name"])] = col
    return columns


async def compare_schemas(
    schema_a: Dict[str, Any], schema_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two schemas to identify differences.

    Accepts the normalized schema format (nested tables[].columns[] with
    "name" and "type" keys) produced by normalize_*() and NormalizedSchema.to_dict().

    Args:
        schema_a: First schema
        schema_b: Second schema

    Returns:
        Dictionary with added, removed, and modified elements
    """
    tables_a = _extract_tables(schema_a)
    tables_b = _extract_tables(schema_b)

    added_tables = [name for name in tables_b if name not in tables_a]
    removed_tables = [name for name in tables_a if name not in tables_b]

    # Compare columns
    columns_a = _extract_columns(schema_a)
    columns_b = _extract_columns(schema_b)

    added_columns = [key for key in columns_b if key not in columns_a]
    removed_columns = [key for key in columns_a if key not in columns_b]

    # Type changes
    modified_columns = []
    for key in set(columns_a.keys()) & set(columns_b.keys()):
        if columns_a[key].get("type") != columns_b[key].get("type"):
            modified_columns.append(
                {
                    "table": key[0],
                    "column": key[1],
                    "old_type": columns_a[key].get("type"),
                    "new_type": columns_b[key].get("type"),
                }
            )

    return {
        "comparison": {
            "added_tables": added_tables,
            "removed_tables": removed_tables,
            "added_columns": [{"table": k[0], "column": k[1]} for k in added_columns],
            "removed_columns": [
                {"table": k[0], "column": k[1]} for k in removed_columns
            ],
            "modified_columns": modified_columns,
            "breaking_changes": len(removed_tables)
            + len(removed_columns)
            + len(modified_columns),
        },
    }
