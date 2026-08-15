"""Move PostgreSQL write admission out of serialized source registration."""

from __future__ import annotations

import sqlite3

from ..sqlite_codecs import decode_preledger_source, decode_source, encode_source
from ..sqlite_schema import (
    ADMISSION_TABLE_SQL,
    JOURNAL_RECEIPT_TABLES,
    WRITE_ADMISSION_TABLES,
)
from .models import SQLiteMigration

MIGRATION_ID = "20260811_postgresql_write_admission"
DEFINITION = """20260811_postgresql_write_admission
create postgresql_write_admissions with owned source foreign key;
move active boolean PostgreSQL admission into presence rows;
remove write_access from serialized PostgreSQL source configuration
"""


def apply(connection: sqlite3.Connection) -> None:
    connection.execute(ADMISSION_TABLE_SQL)
    for agent_id, source_id, data in tuple(
        connection.execute("SELECT agent_id, id, data FROM sources")
    ):
        registration = decode_preledger_source(data)
        if registration.agent_id != agent_id or registration.id != source_id:
            raise ValueError("stored source ownership is invalid")
        if registration.adapter_id != "postgresql":
            continue
        raw_admission = registration.configuration.get("write_access", False)
        if not isinstance(raw_admission, bool):
            raise ValueError("stored PostgreSQL write admission is invalid")
        if raw_admission and registration.active:
            connection.execute(
                """INSERT INTO postgresql_write_admissions(agent_id, source_id)
                   VALUES (?, ?)""",
                (agent_id, source_id),
            )
        connection.execute(
            "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
            (encode_source(registration), agent_id, source_id),
        )


def validate_target(connection: sqlite3.Connection) -> None:
    sources = {
        (agent_id, source_id): decode_source(data)
        for agent_id, source_id, data in connection.execute(
            "SELECT agent_id, id, data FROM sources"
        )
    }
    for (agent_id, source_id), registration in sources.items():
        if registration.agent_id != agent_id or registration.id != source_id:
            raise ValueError("stored source ownership is invalid")
    for agent_id, source_id in connection.execute(
        "SELECT agent_id, source_id FROM postgresql_write_admissions"
    ):
        admission_registration = sources.get((agent_id, source_id))
        if (
            admission_registration is None
            or admission_registration.adapter_id != "postgresql"
            or not admission_registration.active
        ):
            raise ValueError("stored PostgreSQL write admission is invalid")


MIGRATION = SQLiteMigration(
    ordinal=2,
    migration_id=MIGRATION_ID,
    definition=DEFINITION,
    source_schema=JOURNAL_RECEIPT_TABLES,
    target_schema=WRITE_ADMISSION_TABLES,
    apply=apply,
    validate_target=validate_target,
)
