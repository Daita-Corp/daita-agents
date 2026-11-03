"""Discovery keeps every relation kind and every column type it is given."""

from __future__ import annotations

from collections.abc import Mapping

from daita.adapters.models import SourceRegistration, source_registration_id
from daita.adapters.postgresql import _load_structure
from daita.catalog.models import ResourceKind


class _Connection:
    """Answers discovery's tagged queries from fixed rows."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def fetch(self, sql: str, *args: object) -> list[Mapping[str, object]]:
        self.statements.append(sql)
        if "daita:postgresql.resources" in sql:
            return [
                {"schema_name": "sales", "resource_name": name, "resource_kind": kind}
                for name, kind in (
                    ("orders", "table"),
                    ("open_orders", "view"),
                )
            ]
        if "daita:postgresql.columns" in sql:
            return [
                {
                    "column_name": "status",
                    "native_type": "sales.order_status",
                    "type_schema": "sales",
                    "type_name": "order_status",
                    "ordinal": 1,
                    "nullable": False,
                    "default_expression": None,
                    "collation": None,
                    "is_identity": False,
                    "is_generated": False,
                    "is_updatable": True,
                    "primary_key_ordinal": None,
                    "comment": "Where the order is.",
                    "enum_values": ["open", "shipped"],
                },
            ]
        return []


async def test_views_and_custom_types_are_kept_with_their_comments() -> None:
    registration = _registration()
    structure = await _load_structure(
        _Connection(),
        registration,
        max_resources=10,
        max_columns=10,
        max_indexes=10,
        max_relationships=10,
    )
    tables = {table.name: table for table in structure.tables}
    assert tables["orders"].kind is ResourceKind.TABLE
    assert tables["open_orders"].kind is ResourceKind.VIEW
    (column,) = tables["orders"].columns
    assert column.native_type == "sales.order_status"
    assert column.comment == "Where the order is."


def _registration() -> SourceRegistration:
    from datetime import UTC, datetime

    from daita.llm.models import ModelSensitivity

    return SourceRegistration(
        id=source_registration_id("agent", "postgresql", "db.example.test/sales"),
        agent_id="agent",
        adapter_id="postgresql",
        native_identity="db.example.test/sales",
        display_name="Sales",
        configuration={
            "host": "db.example.test",
            "port": 5432,
            "database": "sales",
            "user": "reader",
            "schemas": ("sales",),
        },
        attached_at=datetime(2026, 11, 2, tzinfo=UTC),
        presentation_sensitivity=ModelSensitivity.INTERNAL,
    )
