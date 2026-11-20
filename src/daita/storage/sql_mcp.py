"""MCP server-binding persistence for the SQL state store.

Every function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns the transaction boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from ..adapters.mcp import (
    MCP_MAX_ACTIVE_TOOLS_PER_AGENT,
    MCP_MAX_AGENT_CATALOG_BYTES,
    MCP_MAX_BINDING_CANONICAL_BYTES,
    MCP_MAX_BINDINGS_PER_AGENT,
    MCPAdmissionError,
    MCPBindingState,
    MCPServerBinding,
)
from .sql_connection import SQLConnection, required_row
from .sqlite_codecs import decode_mcp_binding, encode_mcp_binding


def load_binding(
    connection: SQLConnection, agent_id: str, binding_id: str
) -> MCPServerBinding | None:
    row = connection.execute(
        """SELECT data FROM mcp_server_bindings
           WHERE agent_id = ? AND binding_id = ?""",
        (agent_id, binding_id),
    ).fetchone()
    return (
        None
        if row is None
        else decode_mcp_binding(
            row[0],
            agent_id=agent_id,
            binding_id=binding_id,
        )
    )


def list_bindings(
    connection: SQLConnection, agent_id: str
) -> tuple[MCPServerBinding, ...]:
    rows = tuple(
        connection.execute(
            """SELECT binding_id, data FROM mcp_server_bindings
               WHERE agent_id = ? ORDER BY binding_id""",
            (agent_id,),
        )
    )
    if len(rows) > MCP_MAX_BINDINGS_PER_AGENT:
        raise RuntimeError("stored MCP binding count exceeds its fixed bound")
    encoded_sizes = tuple(len(data.encode("utf-8")) for _, data in rows)
    if any(size > MCP_MAX_BINDING_CANONICAL_BYTES for size in encoded_sizes):
        raise RuntimeError("stored MCP binding exceeds its byte bound")
    if sum(encoded_sizes) > MCP_MAX_AGENT_CATALOG_BYTES:
        raise RuntimeError("stored MCP agent catalog exceeds its byte bound")
    bindings = tuple(
        decode_mcp_binding(
            data,
            agent_id=agent_id,
            binding_id=binding_id,
        )
        for binding_id, data in rows
    )
    if _active_tool_count(bindings) > MCP_MAX_ACTIVE_TOOLS_PER_AGENT:
        raise RuntimeError("stored active MCP tool catalog exceeds its fixed bound")
    return bindings


def store_binding(
    connection: SQLConnection,
    binding: MCPServerBinding,
    encoded: str,
    expected_revision: int | None,
) -> MCPServerBinding:
    row = connection.execute(
        """SELECT data FROM mcp_server_bindings
           WHERE agent_id = ? AND binding_id = ?""",
        (binding.agent_id, binding.binding_id),
    ).fetchone()
    if row is None:
        if expected_revision is not None or binding.revision != 1:
            raise ValueError("MCP binding revision precondition failed")
    else:
        current = decode_mcp_binding(
            row[0],
            agent_id=binding.agent_id,
            binding_id=binding.binding_id,
        )
        if (
            expected_revision is None
            or current.revision != expected_revision
            or binding.revision != expected_revision + 1
        ):
            raise ValueError("MCP binding revision precondition failed")
    other_rows = tuple(
        connection.execute(
            """SELECT binding_id, data FROM mcp_server_bindings
               WHERE agent_id = ? AND binding_id <> ?""",
            (binding.agent_id, binding.binding_id),
        )
    )
    total_bytes = len(encoded.encode("utf-8")) + sum(
        len(data.encode("utf-8")) for _, data in other_rows
    )
    if total_bytes > MCP_MAX_AGENT_CATALOG_BYTES:
        raise ValueError("MCP agent catalog exceeds its byte bound")
    other_bindings = tuple(
        decode_mcp_binding(
            data,
            agent_id=binding.agent_id,
            binding_id=other_binding_id,
        )
        for other_binding_id, data in other_rows
    )
    active_tool_count = _active_tool_count((*other_bindings, binding))
    if active_tool_count > MCP_MAX_ACTIVE_TOOLS_PER_AGENT:
        raise MCPAdmissionError(
            "mcp_agent_tool_limit_exceeded",
            "The active MCP tool catalog exceeds its fixed per-agent bound.",
            {
                "observed_tools": active_tool_count,
                "maximum_tools": MCP_MAX_ACTIVE_TOOLS_PER_AGENT,
            },
        )
    if row is None:
        count = required_row(
            connection.execute(
                "SELECT COUNT(*) FROM mcp_server_bindings WHERE agent_id = ?",
                (binding.agent_id,),
            )
        )[0]
        if count >= MCP_MAX_BINDINGS_PER_AGENT:
            raise ValueError("MCP binding count exceeds its fixed bound")
        connection.execute(
            """INSERT INTO mcp_server_bindings(agent_id, binding_id, data)
               VALUES (?, ?, ?)""",
            (
                binding.agent_id,
                binding.binding_id,
                encoded,
            ),
        )
        return binding
    result = connection.execute(
        """UPDATE mcp_server_bindings SET data = ?
           WHERE agent_id = ? AND binding_id = ? AND data = ?""",
        (
            encoded,
            binding.agent_id,
            binding.binding_id,
            row[0],
        ),
    )
    if result.rowcount != 1:
        raise RuntimeError("MCP binding changed during its transition")
    return binding


def update_discovery(
    connection: SQLConnection,
    agent_id: str,
    binding_id: str,
    *,
    summary: str,
    when_to_use: str,
    keywords: tuple[str, ...],
) -> MCPServerBinding:
    row = connection.execute(
        "SELECT data FROM mcp_server_bindings WHERE agent_id = ? AND binding_id = ?",
        (agent_id, binding_id),
    ).fetchone()
    if row is None:
        raise ValueError("MCP binding does not exist")
    current = decode_mcp_binding(row[0], agent_id=agent_id, binding_id=binding_id)
    updated = replace(
        current, summary=summary, when_to_use=when_to_use, keywords=keywords
    )
    encoded = encode_mcp_binding(updated)
    if len(encoded.encode("utf-8")) > MCP_MAX_BINDING_CANONICAL_BYTES:
        raise ValueError("MCP binding exceeds its byte bound")
    others = connection.execute(
        "SELECT data FROM mcp_server_bindings WHERE agent_id = ? AND binding_id <> ?",
        (agent_id, binding_id),
    )
    if (
        len(encoded.encode("utf-8"))
        + sum(len(item[0].encode("utf-8")) for item in others)
        > MCP_MAX_AGENT_CATALOG_BYTES
    ):
        raise ValueError("MCP agent catalog exceeds its byte bound")
    connection.execute(
        "UPDATE mcp_server_bindings SET data = ? WHERE agent_id = ? AND binding_id = ?",
        (encoded, agent_id, binding_id),
    )
    return updated


def validate_stored_bounds(connection: SQLConnection) -> None:
    totals: dict[str, int] = {}
    binding_counts: dict[str, int] = {}
    active_tool_counts: dict[str, int] = {}
    for agent_id, binding_id, data in connection.execute(
        "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
    ):
        binding_counts[agent_id] = binding_counts.get(agent_id, 0) + 1
        if binding_counts[agent_id] > MCP_MAX_BINDINGS_PER_AGENT:
            raise ValueError("stored MCP binding count exceeds its fixed bound")
        encoded_bytes = len(data.encode("utf-8"))
        if encoded_bytes > MCP_MAX_BINDING_CANONICAL_BYTES:
            raise ValueError("stored MCP binding exceeds its byte bound")
        totals[agent_id] = totals.get(agent_id, 0) + encoded_bytes
        if totals[agent_id] > MCP_MAX_AGENT_CATALOG_BYTES:
            raise ValueError("stored MCP agent catalog exceeds its byte bound")
        binding = decode_mcp_binding(
            data,
            agent_id=agent_id,
            binding_id=binding_id,
        )
        if binding.state is MCPBindingState.ACTIVE:
            active_tool_counts[agent_id] = active_tool_counts.get(agent_id, 0) + len(
                binding.tools
            )
            if active_tool_counts[agent_id] > MCP_MAX_ACTIVE_TOOLS_PER_AGENT:
                raise ValueError(
                    "stored active MCP tool catalog exceeds its fixed bound"
                )


def _active_tool_count(bindings: Iterable[MCPServerBinding]) -> int:
    return sum(
        len(binding.tools)
        for binding in bindings
        if binding.state is MCPBindingState.ACTIVE
    )
