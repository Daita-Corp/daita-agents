from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from daita.sessions import SessionCompressionCheckpoint

NOW = datetime(2026, 7, 18, 18, 0, tzinfo=timezone.utc)


def _checkpoint() -> SessionCompressionCheckpoint:
    return SessionCompressionCheckpoint(
        id="session-summary-1",
        agent_id="agent-atlas",
        session_id="session-main",
        version=1,
        through_position=1,
        through_operation_id="operation-2",
        source_fingerprint="sha256:" + "a" * 64,
        summary="Earlier requests established customer scope.",
        operation_ids=("operation-1", "operation-2"),
        evidence_ids=("evidence-1",),
        approval_ids=("approval-1",),
        resource_ids=("resource-customers",),
        created_at=NOW,
    )


def test_session_compression_checkpoint_preserves_bounded_references() -> None:
    checkpoint = _checkpoint()

    assert checkpoint.operation_ids == ("operation-1", "operation-2")
    assert checkpoint.through_operation_id == checkpoint.operation_ids[-1]
    assert checkpoint.evidence_ids == ("evidence-1",)
    assert checkpoint.resource_ids == ("resource-customers",)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"version": 0}, "version"),
        ({"through_position": -1}, "position"),
        ({"source_fingerprint": "sha256:bad"}, "fingerprint"),
        ({"summary": "x" * 32_769}, "summary"),
        ({"operation_ids": ("operation-1",)}, "frontier"),
        ({"operation_ids": ("operation-2", "operation-2")}, "duplicates"),
        ({"evidence_ids": ("",)}, "non-empty"),
        ({"created_at": datetime(2026, 7, 18)}, "timezone-aware"),
    ],
)
def test_session_compression_checkpoint_rejects_invalid_state(
    changes: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        replace(_checkpoint(), **changes)
