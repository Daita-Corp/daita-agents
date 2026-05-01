"""Azure API Management normalizer."""

from typing import Any, Dict


def normalize_azure_apim(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize API Management API discovery output."""
    api_id = raw.get("api_id", "")
    operations = raw.get("operations", [])
    columns = [
        {
            "name": op.get("display_name")
            or op.get("name")
            or op.get("url_template", ""),
            "type": op.get("method", "HTTP"),
            "nullable": False,
            "is_primary_key": False,
            "column_comment": op.get("url_template", ""),
        }
        for op in operations
    ]

    return {
        "database_type": "azure_apim",
        "database_name": api_id,
        "tables": [
            {
                "name": api_id,
                "row_count": len(operations),
                "columns": columns,
                "metadata": {
                    "display_name": raw.get("display_name", ""),
                    "path": raw.get("path", ""),
                    "protocols": raw.get("protocols", []),
                },
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "service": raw.get("service", ""),
            "resource_name": raw.get("resource_name", ""),
            "operation_count": len(operations),
        },
    }
