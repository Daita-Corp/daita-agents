"""DynamoDB normalizer."""

from typing import Any, Dict

_DYNAMODB_TYPE_MAP = {
    "S": "string",
    "N": "number",
    "B": "binary",
    "SS": "string_set",
    "NS": "number_set",
    "BS": "binary_set",
    "L": "list",
    "M": "map",
    "BOOL": "boolean",
    "NULL": "null",
}


def normalize_dynamodb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DynamoDB discover output."""
    table_name = raw.get("table_name", "")
    key_schema = raw.get("key_schema", [])
    attribute_defs = raw.get("attribute_definitions", [])
    sampled = raw.get("sampled_attributes", {})

    defined_types = {
        a["AttributeName"]: _DYNAMODB_TYPE_MAP.get(
            a["AttributeType"], a["AttributeType"]
        )
        for a in attribute_defs
    }

    seen: set[str] = set()
    columns = []

    # Key schema attributes first (marked as primary keys)
    for key in key_schema:
        name = key["AttributeName"]
        seen.add(name)
        columns.append(
            {
                "name": name,
                "type": defined_types.get(name, "string"),
                "nullable": False,
                "is_primary_key": True,
                "column_comment": f"{'Partition' if key['KeyType'] == 'HASH' else 'Sort'} key",
            }
        )

    # Remaining defined attributes
    for attr in attribute_defs:
        name = attr["AttributeName"]
        if name in seen:
            continue
        seen.add(name)
        columns.append(
            {
                "name": name,
                "type": _DYNAMODB_TYPE_MAP.get(
                    attr["AttributeType"], attr["AttributeType"]
                ),
                "nullable": True,
                "is_primary_key": False,
            }
        )

    # Sampled attributes not in definitions
    for attr_name, types in sampled.items():
        if attr_name in seen:
            continue
        seen.add(attr_name)
        dtype = types[0] if types else "S"
        columns.append(
            {
                "name": attr_name,
                "type": _DYNAMODB_TYPE_MAP.get(dtype, dtype),
                "nullable": True,
                "is_primary_key": False,
                "column_comment": "inferred from sample",
            }
        )

    return {
        "database_type": "dynamodb",
        "database_name": table_name,
        "tables": [
            {
                "name": table_name,
                "row_count": raw.get("item_count"),
                "columns": columns,
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
    }
