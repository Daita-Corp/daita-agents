"""Convert released revision-2 records to caller-aware framework records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..sqlite_schema import SCHEMA_REVISION_2
from .models import HomeMigration


def _apply_revision_3(home: Path, source_shape: str | None) -> None:
    # A supported revision-1 bridge may have just produced revision 2 in this
    # same staged upgrade. The revision-2 shape remains the only input here.
    del source_shape
    with sqlite3.connect(home / "state.db") as connection:
        for run_id, agent_id, encoded in connection.execute(
            "SELECT id, agent_id, input FROM runs"
        ):
            record = json.loads(encoded)
            fields = record["fields"]
            if record["__record__"] != "RunInput":
                raise ValueError("revision-2 run record is invalid")
            if "caller_principal_id" not in fields:
                scope = fields["start"]["fields"]["execution_scope"]
                principal = (
                    agent_id if scope is None else scope["fields"]["principal_id"]
                )
                fields["caller_principal_id"] = principal
            fields.setdefault("caller_principal_verified", True)
            connection.execute(
                "UPDATE runs SET input = ? WHERE id = ?",
                (json.dumps(record, sort_keys=True, separators=(",", ":")), run_id),
            )
        for agent_id, binding_id, encoded in connection.execute(
            "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
        ):
            record = json.loads(encoded)
            fields = record["fields"]
            if record["__record__"] != "MCPServerBinding":
                raise ValueError("revision-2 MCP binding is invalid")
            if "connection_id" in fields:
                continue
            fields["owner_principal_id"] = agent_id
            fields.update(
                connection_id=None,
                resource_uri=None,
                required_scopes=[],
            )
            connection.execute(
                "UPDATE mcp_server_bindings SET data = ? WHERE agent_id = ? AND binding_id = ?",
                (
                    json.dumps(record, sort_keys=True, separators=(",", ":")),
                    agent_id,
                    binding_id,
                ),
            )
        connection.commit()


REVISION_3 = HomeMigration(
    revision=3,
    migration_id="0003_framework_caller_authority",
    definition=(
        "Add caller provenance and opaque personal MCP connection claims while "
        "preserving released revision-2 identities, grants, receipts and data."
    ),
    affected_paths=("state.db",),
    target_schema=SCHEMA_REVISION_2,
    apply=_apply_revision_3,
)

__all__ = ["REVISION_3"]
