"""GCP API Gateway normalizer."""

from typing import Any, Dict, List


def normalize_gcp_apigateway(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize GCP API Gateway discover output.

    Each gateway endpoint becomes a column so lineage edges can reference
    individual gateway hostnames.
    """
    api_id = raw.get("api_id", "")

    columns: List[Dict[str, Any]] = []
    for gw in raw.get("gateways", []):
        columns.append(
            {
                "name": f"gateway:{gw.get('name', '')}",
                "type": "endpoint",
                "nullable": False,
                "is_primary_key": False,
                "column_comment": gw.get("default_hostname", ""),
            }
        )

    return {
        "database_type": "gcp_apigateway",
        "database_name": api_id,
        "tables": [
            {
                "name": api_id,
                "row_count": len(columns),
                "columns": columns,
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "project": raw.get("project", ""),
            "location": raw.get("location", ""),
            "resource_name": raw.get("resource_name", ""),
            "display_name": raw.get("display_name", ""),
            "state": raw.get("state", ""),
            "configs": raw.get("configs", []),
        },
    }
