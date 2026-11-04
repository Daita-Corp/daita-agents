"""The approval document a relational write presents before it applies."""

from __future__ import annotations

from typing import Any

from daita import ApprovalRequest

APPROVAL_KEYS = frozenset({"arguments", "target", "preview"})
TARGET_KEYS = frozenset({"source_id", "resource_id", "name", "source_name"})
PREVIEW_KEYS = frozenset(
    {
        "sql",
        "operation",
        "columns_changed",
        "affected_rows",
        "inserted_rows",
        "updated_rows",
        "deleted_rows",
        "skipped_rows",
        "samples",
        "triggers",
        "cascades",
        "warnings",
        "verification",
    }
)


def data_write_approval(request: ApprovalRequest) -> dict[str, Any] | None:
    """Return ``{arguments, target, preview}`` for a data write, or ``None``.

    ``arguments`` are the exact apply arguments, ``target`` names the resource
    and ``preview`` is the fingerprinted review the user approves.
    """

    document = request.arguments.to_dict()
    if set(document) != APPROVAL_KEYS or not all(
        isinstance(document[key], dict) for key in APPROVAL_KEYS
    ):
        return None
    return document
