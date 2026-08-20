"""Add the independently keyed Stage M2 MCP binding aggregate."""

from __future__ import annotations

import sqlite3

from ..sqlite_codecs import decode_mcp_binding
from ..sqlite_schema import (
    CURRENT_TABLES,
    GENERALIZED_UPDATE_TABLES,
    MCP_SERVER_BINDING_TABLE_SQL,
)
from .generalized_postgresql_updates import validate_target as validate_previous
from .models import SQLiteMigration

MIGRATION_ID = "20260819_mcp_server_bindings"
DEFINITION = """20260819_mcp_server_bindings
create one independently keyed MCP server binding aggregate per agent and binding;
persist accepted endpoint, protocol/server identity, secret reference, schema digests,
read-tool declarations, sensitivity ceiling, revision, and lifecycle state only;
store no resolved secret, protocol session, dynamic registry, or execution state
"""


def apply(connection: sqlite3.Connection) -> None:
    connection.execute(MCP_SERVER_BINDING_TABLE_SQL)


def validate_target(connection: sqlite3.Connection) -> None:
    validate_previous(connection)
    counts: dict[str, int] = {}
    for agent_id, binding_id, data in connection.execute(
        "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
    ):
        counts[agent_id] = counts.get(agent_id, 0) + 1
        # Keep this literal inside the checksummed function: historical migration
        # semantics must not follow later runtime binding-policy changes.
        if counts[agent_id] > 32:
            raise ValueError("stored MCP binding count exceeds its fixed bound")
        binding = decode_mcp_binding(
            data,
            agent_id=agent_id,
            binding_id=binding_id,
        )
        if binding.agent_id != agent_id or binding.binding_id != binding_id:
            raise ValueError("stored MCP binding ownership is invalid")


MIGRATION = SQLiteMigration(
    ordinal=5,
    migration_id=MIGRATION_ID,
    definition=DEFINITION,
    source_schema=GENERALIZED_UPDATE_TABLES,
    target_schema=CURRENT_TABLES,
    apply=apply,
    validate_target=validate_target,
)
