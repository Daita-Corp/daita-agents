"""Shared helpers for profiler implementations."""

from typing import Optional

from ..base_profiler import (
    NormalizedColumn,
    NormalizedForeignKey,
    NormalizedIndex,
    NormalizedSchema,
    NormalizedTable,
)


def _dict_to_normalized_schema(
    d: dict, store_id: Optional[str] = None
) -> NormalizedSchema:
    """Convert a normalize_*() dict to a NormalizedSchema dataclass."""
    tables = []
    for t in d.get("tables", []):
        columns = [
            NormalizedColumn(
                name=c["name"],
                type=c["type"],
                nullable=c.get("nullable", True),
                is_primary_key=c.get("is_primary_key", False),
                comment=c.get("column_comment"),
            )
            for c in t.get("columns", [])
        ]
        indexes = [
            NormalizedIndex(
                name=i["name"],
                type=i.get("type", ""),
                columns=list(i.get("columns", [])),
                unique=bool(i.get("unique", False)),
                metadata=dict(i.get("metadata", {})),
            )
            for i in t.get("indexes", [])
        ]
        tables.append(
            NormalizedTable(
                name=t["name"],
                row_count=t.get("row_count"),
                columns=columns,
                indexes=indexes,
                metadata=dict(t.get("metadata", {})),
            )
        )

    foreign_keys = [
        NormalizedForeignKey(
            source_table=fk["source_table"],
            source_column=fk["source_column"],
            target_table=fk["target_table"],
            target_column=fk["target_column"],
        )
        for fk in d.get("foreign_keys", [])
    ]

    return NormalizedSchema(
        database_type=d.get("database_type", "unknown"),
        database_name=d.get("database_name", ""),
        tables=tables,
        foreign_keys=foreign_keys,
        table_count=d.get("table_count", len(tables)),
        store_id=store_id,
        metadata=d.get("metadata", {}),
    )
