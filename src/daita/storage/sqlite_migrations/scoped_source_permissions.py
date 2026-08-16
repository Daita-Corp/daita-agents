"""Replace source-wide write admission with exact source permission scopes."""

from __future__ import annotations

import sqlite3

from ..sqlite_codecs import (
    decode_postgresql_update_scope,
    decode_source,
    decode_source_read_scope,
    encode_source_read_scope,
)
from ..sqlite_records import SourceReadScope
from ..sqlite_schema import (
    POSTGRESQL_UPDATE_SCOPE_TABLE_SQL,
    SCOPED_PERMISSION_TABLES,
    SOURCE_READ_SCOPE_TABLE_SQL,
    WRITE_ADMISSION_TABLES,
)
from .models import SQLiteMigration

MIGRATION_ID = "20260812_scoped_source_permissions"
DEFINITION = """20260812_scoped_source_permissions
create source_read_scopes and postgresql_update_scopes with owned source foreign keys;
give each active source one explicit all read scope and detached sources no scope;
convert every prior write-admission state to zero exact PostgreSQL update scopes;
drop postgresql_write_admissions from the current physical schema
"""


def apply(connection: sqlite3.Connection) -> None:
    connection.execute(SOURCE_READ_SCOPE_TABLE_SQL)
    connection.execute(POSTGRESQL_UPDATE_SCOPE_TABLE_SQL)
    for agent_id, source_id, data in tuple(
        connection.execute("SELECT agent_id, id, data FROM sources")
    ):
        registration = decode_source(data)
        if registration.agent_id != agent_id or registration.id != source_id:
            raise ValueError("stored source ownership is invalid")
        if not registration.active:
            continue
        scope = SourceReadScope.allow_all(agent_id=agent_id, source_id=source_id)
        connection.execute(
            """INSERT INTO source_read_scopes(agent_id, source_id, data)
               VALUES (?, ?, ?)""",
            (agent_id, source_id, encode_source_read_scope(scope)),
        )
    connection.execute("DROP TABLE postgresql_write_admissions")


def validate_target(connection: sqlite3.Connection) -> None:
    sources = {
        (agent_id, source_id): decode_source(data)
        for agent_id, source_id, data in connection.execute(
            "SELECT agent_id, id, data FROM sources"
        )
    }
    read_scopes: dict[tuple[str, str], SourceReadScope] = {}
    for agent_id, source_id, data in connection.execute(
        "SELECT agent_id, source_id, data FROM source_read_scopes"
    ):
        key = (agent_id, source_id)
        if key in read_scopes:
            raise ValueError("stored source read scope identity is duplicated")
        read_scopes[key] = decode_source_read_scope(
            data,
            agent_id=agent_id,
            source_id=source_id,
        )
    for key, registration in sources.items():
        if registration.agent_id != key[0] or registration.id != key[1]:
            raise ValueError("stored source ownership is invalid")
        scope = read_scopes.pop(key, None)
        if registration.active and scope is None:
            raise ValueError("active source is missing its read scope")
        if not registration.active and scope is not None:
            raise ValueError("detached source retains a read scope")
    if read_scopes:
        raise ValueError("stored source read scope is foreign")

    for agent_id, source_id, resource_id, fingerprint, data in connection.execute(
        """SELECT agent_id, source_id, resource_id,
                  authorization_fingerprint, data
           FROM postgresql_update_scopes"""
    ):
        selected_registration = sources.get((agent_id, source_id))
        if (
            selected_registration is None
            or selected_registration.adapter_id != "postgresql"
            or not selected_registration.active
        ):
            raise ValueError("stored PostgreSQL update scope is foreign")
        decode_postgresql_update_scope(
            data,
            agent_id=agent_id,
            source_id=source_id,
            resource_id=resource_id,
            authorization_fingerprint=fingerprint,
        )


MIGRATION = SQLiteMigration(
    ordinal=3,
    migration_id=MIGRATION_ID,
    definition=DEFINITION,
    source_schema=WRITE_ADMISSION_TABLES,
    target_schema=SCOPED_PERMISSION_TABLES,
    apply=apply,
    validate_target=validate_target,
)
