"""Semantic-annotation persistence for the SQL state store.

Every function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns the transaction boundary.
"""

from __future__ import annotations

import re
from hashlib import sha256

from .._json import FrozenJsonObject
from ..semantics import (
    SEMANTIC_MAX_ANNOTATIONS,
    SemanticAnnotation,
    SemanticDigestMismatchError,
    SemanticNotFoundError,
    SemanticValidationError,
    semantic_annotation_sha256,
)
from .sql_connection import SQLConnection
from .sqlite_codecs import decode_semantic_annotation, encode_semantic_annotation


def list_annotations(
    connection: SQLConnection, agent_id: str
) -> tuple[SemanticAnnotation, ...]:
    rows = connection.execute(
        """SELECT data FROM semantic_annotations
           WHERE agent_id = ? ORDER BY id""",
        (agent_id,),
    ).fetchall()
    if len(rows) > SEMANTIC_MAX_ANNOTATIONS:
        raise RuntimeError("stored semantic annotation collection exceeds bound")
    return tuple(decode_semantic_annotation(data) for (data,) in rows)


def load_annotation(
    connection: SQLConnection, agent_id: str, annotation_id: str
) -> SemanticAnnotation | None:
    row = connection.execute(
        """SELECT data FROM semantic_annotations
           WHERE agent_id = ? AND id = ?""",
        (agent_id, annotation_id),
    ).fetchone()
    return None if row is None else decode_semantic_annotation(row[0])


def save_annotation(
    connection: SQLConnection,
    agent_id: str,
    annotation: SemanticAnnotation,
    expected_sha256: str | None,
) -> bool:
    save_fingerprint(connection, agent_id, annotation, expected_sha256)
    row = connection.execute(
        """SELECT data FROM semantic_annotations
           WHERE agent_id = ? AND id = ?""",
        (agent_id, annotation.id),
    ).fetchone()
    changed = row is None or decode_semantic_annotation(row[0]) != annotation
    connection.execute(
        f"""INSERT INTO semantic_annotations(agent_id, id, data)
           VALUES (?, ?, ?)
           ON CONFLICT({connection.conflict('agent_id, id')}) DO UPDATE SET data = excluded.data""",
        (agent_id, annotation.id, encode_semantic_annotation(annotation)),
    )
    return changed


def delete_annotation(
    connection: SQLConnection,
    agent_id: str,
    annotation_id: str,
    expected_sha256: str,
) -> bool:
    delete_fingerprint(connection, agent_id, annotation_id, expected_sha256)
    cursor = connection.execute(
        """DELETE FROM semantic_annotations
           WHERE agent_id = ? AND id = ?""",
        (agent_id, annotation_id),
    )
    if cursor.rowcount != 1:
        raise SemanticNotFoundError(annotation_id)
    return True


def validate_owner(
    agent_id: str,
    annotation: SemanticAnnotation,
) -> None:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    if not isinstance(annotation, SemanticAnnotation):
        raise TypeError("annotation must be SemanticAnnotation")
    if annotation.agent_id != agent_id:
        raise ValueError("semantic annotation belongs to another agent")


def _rows(
    connection: SQLConnection,
    agent_id: str,
) -> tuple[tuple[str, SemanticAnnotation, str], ...]:
    rows = connection.execute(
        """SELECT id, data FROM semantic_annotations
           WHERE agent_id = ? ORDER BY id""",
        (agent_id,),
    ).fetchall()
    if len(rows) > SEMANTIC_MAX_ANNOTATIONS:
        raise RuntimeError("stored semantic annotation collection exceeds bound")
    return tuple(
        (annotation_id, decode_semantic_annotation(data), data)
        for annotation_id, data in rows
    )


def _state_sha256(
    rows: tuple[tuple[str, SemanticAnnotation, str], ...],
) -> str:
    payload = "\n".join(f"{annotation_id}:{data}" for annotation_id, _, data in rows)
    return sha256(payload.encode("utf-8")).hexdigest()


def save_fingerprint(
    connection: SQLConnection,
    agent_id: str,
    annotation: SemanticAnnotation,
    expected_sha256: str | None,
) -> FrozenJsonObject:
    validate_owner(agent_id, annotation)
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise SemanticDigestMismatchError(
            "semantic expected_sha256 must be lowercase SHA-256"
        )
    rows = _rows(connection, agent_id)
    by_id = {annotation_id: item for annotation_id, item, _ in rows}
    current = by_id.get(annotation.id)
    expected_target: SemanticAnnotation | None
    expected_target_id: str
    if current is not None:
        expected_target = current
        expected_target_id = current.id
        if annotation.created_at != current.created_at:
            raise SemanticValidationError(
                "semantic replacement must preserve created_at"
            )
    elif annotation.supersedes_id is not None:
        expected_target = by_id.get(annotation.supersedes_id)
        expected_target_id = annotation.supersedes_id
        if expected_target is None:
            raise SemanticNotFoundError(annotation.supersedes_id)
        if (
            expected_target.subject != annotation.subject
            or expected_target.kind is not annotation.kind
        ):
            raise SemanticValidationError(
                "a semantic annotation may supersede only the same subject and kind"
            )
        if annotation.created_at < expected_target.created_at:
            raise SemanticValidationError(
                "semantic supersession cannot predate its target"
            )
    else:
        expected_target = None
        expected_target_id = annotation.id

    if expected_target is None:
        if expected_sha256 is not None:
            raise SemanticDigestMismatchError(
                "new semantic annotations cannot include expected_sha256"
            )
        if len(rows) >= SEMANTIC_MAX_ANNOTATIONS:
            raise SemanticValidationError(
                f"semantic annotation collection is limited to "
                f"{SEMANTIC_MAX_ANNOTATIONS}"
            )
        current_sha256 = sha256(b"").hexdigest()
    else:
        current_sha256 = semantic_annotation_sha256(expected_target)
        if expected_sha256 is None:
            raise SemanticDigestMismatchError(
                "semantic replacement or supersession requires expected_sha256"
            )
        if expected_sha256 != current_sha256:
            raise SemanticDigestMismatchError(
                "semantic annotation changed; load it again with semantic_view"
            )

    return FrozenJsonObject.from_mapping(
        {
            "id": annotation.id,
            "exists": current is not None,
            "expected_target_id": expected_target_id,
            "current_sha256": current_sha256,
            "candidate_sha256": semantic_annotation_sha256(annotation),
            "state_sha256": _state_sha256(rows),
        }
    )


def delete_fingerprint(
    connection: SQLConnection,
    agent_id: str,
    annotation_id: str,
    expected_sha256: str,
) -> FrozenJsonObject:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    if not isinstance(annotation_id, str) or not annotation_id:
        raise ValueError("annotation_id must be a non-empty string")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise SemanticDigestMismatchError(
            "semantic deletion requires lowercase expected_sha256"
        )
    rows = _rows(connection, agent_id)
    current = next(
        (item for item_id, item, _ in rows if item_id == annotation_id),
        None,
    )
    if current is None:
        raise SemanticNotFoundError(annotation_id)
    current_sha256 = semantic_annotation_sha256(current)
    if current_sha256 != expected_sha256:
        raise SemanticDigestMismatchError(
            "semantic annotation changed; load it again with semantic_view"
        )
    return FrozenJsonObject.from_mapping(
        {
            "id": annotation_id,
            "current_sha256": current_sha256,
            "state_sha256": _state_sha256(rows),
        }
    )
