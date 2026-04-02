"""
Schema diagram export.

Generates visual schema representations in Mermaid, JSON Schema, or other
formats from normalized schema output.
"""

from typing import Any, Dict


def map_sql_to_json_type(sql_type: str) -> str:
    """Map SQL data types to JSON Schema types."""
    sql_type = sql_type.lower()

    if any(t in sql_type for t in ["int", "serial", "bigint", "smallint"]):
        return "integer"
    elif any(
        t in sql_type for t in ["float", "double", "decimal", "numeric", "real"]
    ):
        return "number"
    elif any(t in sql_type for t in ["bool", "boolean"]):
        return "boolean"
    elif any(t in sql_type for t in ["json", "jsonb"]):
        return "object"
    elif "array" in sql_type:
        return "array"
    else:
        return "string"


async def export_diagram(
    schema: Dict[str, Any], format: str = "mermaid"
) -> Dict[str, Any]:
    """
    Export schema as a visual diagram.

    Accepts the normalized schema format (nested tables[].columns[] with
    "name" and "type" keys) produced by normalize_*() and NormalizedSchema.to_dict().

    Args:
        schema: Normalized schema dict
        format: Output format ('mermaid', 'dbdiagram', 'json_schema')

    Returns:
        Dictionary with diagram in requested format
    """
    if format == "mermaid":
        lines = ["erDiagram"]

        for table in schema.get("tables", []):
            table_name = table["name"]
            lines.append(f"    {table_name} {{")
            for col in table.get("columns", []):
                data_type = col.get("type", "unknown")
                col_name = col["name"]
                lines.append(f"        {data_type} {col_name}")
            lines.append("    }")

        for fk in schema.get("foreign_keys", []):
            source = fk["source_table"]
            target = fk["target_table"]
            lines.append(f'    {source} ||--o{{ {target} : ""')

        diagram = "\n".join(lines)

        return {"format": "mermaid", "diagram": diagram}

    elif format == "json_schema":
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": [],
        }

        for table in schema.get("tables", []):
            table_name = table["name"]
            properties = {}
            for col in table.get("columns", []):
                col_type = map_sql_to_json_type(col.get("type", "string"))
                properties[col["name"]] = {"type": col_type}

            json_schema["properties"][table_name] = {
                "type": "object",
                "properties": properties,
            }

        return {"format": "json_schema", "schema": json_schema}

    else:
        from ...core.exceptions import ValidationError

        raise ValidationError(
            f"Unsupported format: {format}. Use 'mermaid' or 'json_schema'"
        )
