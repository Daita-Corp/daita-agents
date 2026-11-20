"""Source-registry and source-permission persistence for the SQL state store.

Every function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns the transaction boundary.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from ..adapters.models import SourceRegistration
from .sql_connection import SQLConnection
from .sqlite_codecs import (
    decode_relational_write_scope,
    decode_source,
    decode_source_read_scope,
    encode_relational_write_scope,
    encode_source,
    encode_source_read_scope,
)
from .sqlite_records import (
    RelationalWriteScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
    relational_write_authorization_fingerprint,
)

_STATE_SELECT = """SELECT s.agent_id, s.id, s.data, r.data,
          (SELECT COUNT(*) FROM relational_write_scopes AS u
           WHERE u.agent_id = s.agent_id AND u.source_id = s.id)
   FROM sources AS s
   LEFT JOIN source_read_scopes AS r
     ON r.agent_id = s.agent_id AND r.source_id = s.id"""


def register(
    connection: SQLConnection, registration: SourceRegistration
) -> SourceRegistration:
    row = state_row(connection, registration.agent_id, registration.id)
    if row is not None:
        current = decode_state_row(connection, row)
        if current != registration:
            raise ValueError(f"source registration already exists: {registration.id}")
        return current
    connection.execute(
        "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
        (registration.agent_id, registration.id, encode_source(registration)),
    )
    if registration.active:
        _insert_allow_all_scope(connection, registration)
    return registration


def attach_registration(
    connection: SQLConnection, registration: SourceRegistration
) -> None:
    """First attach inserts the source and an allow-all scope; re-attach follows the rules."""

    row = state_row(connection, registration.agent_id, registration.id)
    if row is None:
        connection.execute(
            "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
            (
                registration.agent_id,
                registration.id,
                encode_source(registration),
            ),
        )
        if registration.active:
            _insert_allow_all_scope(connection, registration)
        return
    current = decode_state_row(connection, row)
    if current == registration:
        return
    if (
        current.active
        or not registration.active
        or current.adapter_id != registration.adapter_id
        or current.native_identity != registration.native_identity
    ):
        raise ValueError(f"source registration already exists: {registration.id}")
    connection.execute(
        """UPDATE sources SET data = ?
           WHERE agent_id = ? AND id = ?""",
        (
            encode_source(registration),
            registration.agent_id,
            registration.id,
        ),
    )
    _insert_allow_all_scope(connection, registration)


def load(
    connection: SQLConnection, agent_id: str, source_id: str
) -> SourceRegistration | None:
    row = state_row(connection, agent_id, source_id)
    return None if row is None else decode_state_row(connection, row)


def list_sources(
    connection: SQLConnection, agent_id: str
) -> tuple[SourceRegistration, ...]:
    rows = connection.execute(
        _STATE_SELECT + " WHERE s.agent_id = ? ORDER BY s.id",
        (agent_id,),
    ).fetchall()
    return tuple(decode_state_row(connection, row) for row in rows)


def load_read_scope(
    connection: SQLConnection, agent_id: str, source_id: str
) -> SourceReadScope | None:
    row = state_row(connection, agent_id, source_id)
    if row is None:
        return None
    registration = decode_state_row(connection, row)
    return _decode_owned_read_scope(connection, registration, row[3])


def list_write_scopes(
    connection: SQLConnection, agent_id: str, source_id: str
) -> tuple[RelationalWriteScope, ...]:
    registration = load(connection, agent_id, source_id)
    if registration is None or not registration.active:
        return ()
    rows = connection.execute(
        """SELECT resource_id, authorization_fingerprint, data
           FROM relational_write_scopes
           WHERE agent_id = ? AND source_id = ?
           ORDER BY resource_id""",
        (agent_id, source_id),
    ).fetchall()
    try:
        return tuple(
            decode_relational_write_scope(
                data,
                agent_id=agent_id,
                source_id=source_id,
                resource_id=resource_id,
                authorization_fingerprint=fingerprint,
            )
            for resource_id, fingerprint, data in rows
        )
    except (TypeError, ValueError):
        raise SourcePermissionStateError(
            "stored PostgreSQL update scope is undecodable"
        ) from None


def replace_permission_scopes(
    connection: SQLConnection,
    read_scope: SourceReadScope,
    update_scopes: tuple[RelationalWriteScope, ...],
) -> SourceRegistration:
    registration = load(connection, read_scope.agent_id, read_scope.source_id)
    if registration is None or not registration.active:
        raise ValueError("permission scopes require an active owned source")
    _require_nonforeign_read_resources(connection, read_scope)
    if any(
        scope.agent_id != read_scope.agent_id or scope.source_id != read_scope.source_id
        for scope in update_scopes
    ):
        raise ValueError("update scope belongs to another source")
    update_resource_ids = {scope.resource_id for scope in update_scopes}
    if read_scope.mode is SourceReadMode.NONE and update_resource_ids:
        raise ValueError("PostgreSQL update scope requires read access")
    if read_scope.mode is SourceReadMode.SELECTED and not update_resource_ids <= set(
        read_scope.resource_ids
    ):
        raise ValueError("PostgreSQL update scope must be a read subset")
    for scope in update_scopes:
        expected = relational_write_authorization_fingerprint(
            source=registration, scope=scope
        )
        if scope.authorization_fingerprint != expected:
            raise ValueError("PostgreSQL write authorization fingerprint is stale")
    connection.execute(
        f"""INSERT INTO source_read_scopes(agent_id, source_id, data)
           VALUES (?, ?, ?)
           ON CONFLICT({connection.conflict('agent_id, source_id')})
           DO UPDATE SET data = excluded.data""",
        (
            read_scope.agent_id,
            read_scope.source_id,
            encode_source_read_scope(read_scope),
        ),
    )
    connection.execute(
        """DELETE FROM relational_write_scopes
           WHERE agent_id = ? AND source_id = ?""",
        (read_scope.agent_id, read_scope.source_id),
    )
    for scope in sorted(update_scopes, key=lambda item: item.resource_id):
        connection.execute(
            """INSERT INTO relational_write_scopes(
                   agent_id, source_id, resource_id,
                   authorization_fingerprint, data
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                scope.agent_id,
                scope.source_id,
                scope.resource_id,
                scope.authorization_fingerprint,
                encode_relational_write_scope(scope),
            ),
        )
    return registration


def update_discovery(
    connection: SQLConnection,
    agent_id: str,
    source_id: str,
    *,
    summary: str,
    when_to_use: str,
    keywords: tuple[str, ...],
) -> SourceRegistration:
    current = load(connection, agent_id, source_id)
    if current is None:
        raise ValueError("source does not exist")
    if not current.active:
        raise ValueError("source is not active")
    updated = replace(
        current, summary=summary, when_to_use=when_to_use, keywords=keywords
    )
    connection.execute(
        "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
        (encode_source(updated), agent_id, source_id),
    )
    return updated


def detach(
    connection: SQLConnection, agent_id: str, source_id: str, detached_at: datetime
) -> SourceRegistration:
    row = connection.execute(
        "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
        (agent_id, source_id),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown source: {source_id}")
    current = decode_source(row[0])
    detached = current if not current.active else current.detach(detached_at)
    connection.execute(
        "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
        (encode_source(detached), agent_id, source_id),
    )
    _delete_scopes(connection, agent_id, source_id)
    return detached


def replace_for_edit(
    connection: SQLConnection,
    registration: SourceRegistration,
    replaced_source_id: str,
    replaced_at: datetime,
    read_scope: SourceReadScope,
) -> None:
    """Hand one active source's identity and scopes to its edited registration."""

    replaced = load(connection, registration.agent_id, replaced_source_id)
    if replaced is None or not replaced.active:
        raise ValueError("source edit requires an active owned source")
    if replaced.adapter_id != registration.adapter_id:
        raise ValueError("source edit cannot change source type")
    if registration.id != replaced.id:
        existing_replacement = load(connection, registration.agent_id, registration.id)
        if existing_replacement is None:
            connection.execute(
                "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                (
                    registration.agent_id,
                    registration.id,
                    encode_source(registration),
                ),
            )
        else:
            if existing_replacement.active:
                raise ValueError("replacement connection is already attached")
            if (
                existing_replacement.adapter_id != registration.adapter_id
                or existing_replacement.native_identity != registration.native_identity
            ):
                raise ValueError(
                    "replacement source identity conflicts with stored state"
                )
            _update_source(connection, registration)
        _update_source(connection, replaced.detach(replaced_at))
    else:
        _update_source(connection, registration)
    for source_id in {replaced.id, registration.id}:
        _delete_scopes(connection, registration.agent_id, source_id)
    connection.execute(
        """INSERT INTO source_read_scopes(agent_id, source_id, data)
           VALUES (?, ?, ?)""",
        (
            read_scope.agent_id,
            read_scope.source_id,
            encode_source_read_scope(read_scope),
        ),
    )


def state_row(
    connection: SQLConnection,
    agent_id: str,
    source_id: str,
) -> tuple[object, ...] | None:
    return connection.execute(
        _STATE_SELECT + " WHERE s.agent_id = ? AND s.id = ?",
        (agent_id, source_id),
    ).fetchone()


def decode_state_row(
    connection: SQLConnection, row: tuple[object, ...]
) -> SourceRegistration:
    row_agent_id, row_source_id, source_data, read_scope_data, update_scope_count = row
    if (
        not isinstance(row_agent_id, str)
        or not isinstance(row_source_id, str)
        or not isinstance(source_data, str)
        or not isinstance(update_scope_count, int)
        or isinstance(update_scope_count, bool)
        or update_scope_count < 0
    ):
        raise SourcePermissionStateError("stored source permission state is invalid")
    try:
        registration = decode_source(source_data)
    except (TypeError, ValueError):
        raise SourcePermissionStateError(
            "stored source registration is invalid"
        ) from None
    if registration.agent_id != row_agent_id or registration.id != row_source_id:
        raise SourcePermissionStateError("stored source ownership is invalid")
    _decode_owned_read_scope(connection, registration, read_scope_data)
    if update_scope_count and not registration.active:
        raise SourcePermissionStateError("stored PostgreSQL update scope is foreign")
    return registration


def _decode_owned_read_scope(
    connection: SQLConnection,
    registration: SourceRegistration,
    data: object,
) -> SourceReadScope | None:
    if not registration.active:
        if data is not None:
            raise SourcePermissionStateError(
                "detached source unexpectedly retains a read scope"
            )
        return None
    if not isinstance(data, str):
        raise SourcePermissionStateError("active source is missing its read scope")
    try:
        scope = decode_source_read_scope(
            data,
            agent_id=registration.agent_id,
            source_id=registration.id,
        )
        if scope.mode is SourceReadMode.SELECTED:
            _require_nonforeign_read_resources(connection, scope)
        return scope
    except SourcePermissionStateError:
        raise
    except (TypeError, ValueError):
        raise SourcePermissionStateError(
            "active source read scope is undecodable"
        ) from None


def _require_nonforeign_read_resources(
    connection: SQLConnection,
    scope: SourceReadScope,
) -> None:
    requested = sorted(set(scope.resource_ids))
    # A resource ID hashes its source, so a damaged row cannot make reads foreign.
    for start in range(0, len(requested), 400):
        chunk = requested[start : start + 400]
        marks = ", ".join("?" * len(chunk))
        if connection.execute(
            "SELECT 1 FROM catalog_resources WHERE agent_id = ? "
            f"AND id IN ({marks}) AND source_id <> ?",
            (scope.agent_id, *chunk, scope.source_id),
        ).fetchone():
            raise SourcePermissionStateError(
                "active source read scope contains a foreign resource"
            )


def _insert_allow_all_scope(
    connection: SQLConnection, registration: SourceRegistration
) -> None:
    scope = SourceReadScope.allow_all(
        agent_id=registration.agent_id,
        source_id=registration.id,
    )
    connection.execute(
        """INSERT INTO source_read_scopes(agent_id, source_id, data)
           VALUES (?, ?, ?)""",
        (registration.agent_id, registration.id, encode_source_read_scope(scope)),
    )


def _update_source(connection: SQLConnection, registration: SourceRegistration) -> None:
    connection.execute(
        """UPDATE sources SET data = ?
           WHERE agent_id = ? AND id = ?""",
        (
            encode_source(registration),
            registration.agent_id,
            registration.id,
        ),
    )


def _delete_scopes(connection: SQLConnection, agent_id: str, source_id: str) -> None:
    connection.execute(
        """DELETE FROM source_read_scopes
           WHERE agent_id = ? AND source_id = ?""",
        (agent_id, source_id),
    )
    connection.execute(
        """DELETE FROM relational_write_scopes
           WHERE agent_id = ? AND source_id = ?""",
        (agent_id, source_id),
    )
