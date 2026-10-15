"""PostgreSQL discovery estimates stay approximate and never become exact counts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from daita.adapters import postgresql

_ANALYZED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_reltuples_and_size_become_estimates_only_when_postgresql_knows_them():
    known = postgresql._table_estimates(
        {"reltuples": 1249.6, "size_bytes": 8192, "statistics_as_of": _ANALYZED}
    )
    assert known == postgresql._TableEstimates(1250, 8192, _ANALYZED)

    never_analyzed = postgresql._table_estimates(
        {"reltuples": -1.0, "size_bytes": 0, "statistics_as_of": None}
    )
    assert never_analyzed.row_count is None
    assert never_analyzed.size_bytes == 0

    legacy_zero = postgresql._table_estimates(
        {"reltuples": 0.0, "size_bytes": 16384, "statistics_as_of": None}
    )
    assert legacy_zero.row_count is None

    analyzed_empty = postgresql._table_estimates(
        {"reltuples": 0.0, "size_bytes": 0, "statistics_as_of": _ANALYZED}
    )
    assert analyzed_empty.row_count == 0

    malformed = postgresql._table_estimates(
        {
            "reltuples": float("nan"),
            "size_bytes": True,
            "statistics_as_of": datetime(2026, 1, 2),
        }
    )
    assert malformed == postgresql._TableEstimates(None, None, None)


def test_estimates_query_is_bounded_to_discovered_tables():
    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetch(self, sql: str, *arguments: object) -> list[dict[str, object]]:
            self.calls.append((sql, arguments))
            return [
                {
                    "schema_name": "public",
                    "resource_name": "orders",
                    "reltuples": 40.0,
                    "size_bytes": 24576,
                    "statistics_as_of": _ANALYZED,
                },
                {
                    "schema_name": "public",
                    "resource_name": "undiscovered",
                    "reltuples": 9.0,
                    "size_bytes": 1,
                    "statistics_as_of": _ANALYZED,
                },
            ]

    structure = SimpleNamespace(
        tables=(
            SimpleNamespace(schema="public", name="orders"),
            SimpleNamespace(schema="public", name="customers"),
        )
    )
    connection = Connection()
    estimates = asyncio.run(
        postgresql._load_estimates(connection, cast(Any, structure))
    )

    assert estimates == {
        ("public", "orders"): postgresql._TableEstimates(40, 24576, _ANALYZED)
    }
    ((sql, arguments),) = connection.calls
    assert "/* daita:postgresql.estimates */" in sql
    assert "pg_class" in sql and "pg_stat_user_tables" in sql
    assert arguments == (["public", "public"], ["orders", "customers"], 3)
    assert (
        asyncio.run(
            postgresql._load_estimates(
                connection, cast(Any, SimpleNamespace(tables=()))
            )
        )
        == {}
    )
    assert len(connection.calls) == 1
