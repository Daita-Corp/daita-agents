"""AWS API Gateway normalizer."""

from typing import Any, Dict


def normalize_apigateway(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize API Gateway discover output.

    Each endpoint becomes a column. The column comment captures the
    authorization type and integration target (Lambda ARN, HTTP URL) —
    this is the key data for lineage edge inference.
    """
    api_name = raw.get("api_name", "")
    endpoints = raw.get("endpoints", [])

    columns = []
    integrations: Dict[str, Dict[str, str]] = {}
    for ep in endpoints:
        method = ep.get("method", "")
        path = ep.get("path", "/")
        col_name = f"{method} {path}"

        auth = ep.get("authorization", "NONE")
        integration_hint = (
            f" -> {ep['integration_uri']}" if ep.get("integration_uri") else ""
        )

        columns.append(
            {
                "name": col_name,
                "type": "endpoint",
                "nullable": False,
                "is_primary_key": False,
                "column_comment": f"{auth}{integration_hint}",
            }
        )

        if ep.get("integration_type") or ep.get("integration_uri"):
            integrations[col_name] = {
                "type": ep.get("integration_type", ""),
                "uri": ep.get("integration_uri", ""),
            }

    return {
        "database_type": "apigateway",
        "database_name": api_name,
        "tables": [
            {
                "name": api_name,
                "row_count": len(endpoints),
                "columns": columns,
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "endpoint": raw.get("endpoint", ""),
            "stage": raw.get("stage", ""),
            "region": raw.get("region", ""),
            "protocol_type": raw.get("protocol_type", ""),
            "authorizers": raw.get("authorizers", []),
            "integrations": integrations,
        },
    }
