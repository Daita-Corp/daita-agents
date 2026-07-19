from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from daita._json import FrozenJsonObject
from daita.memory import (
    MemoryCreator,
    MemoryHistory,
    MemoryKind,
    MemoryProvenance,
    MemoryProvenanceKind,
    MemoryRecord,
    MemoryRestoreRequest,
    MemoryScope,
    MemorySensitivity,
    MemorySnapshot,
    MemoryState,
    MemorySupersessionRequest,
    MemoryVersion,
    normalize_memory_logical_key,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
HASH = "sha256:" + "a" * 64


def _provenance() -> MemoryProvenance:
    return MemoryProvenance(
        kind=MemoryProvenanceKind.USER_STATEMENT,
        content_hash=HASH,
        operation_id="operation-1",
        trigger_id="trigger-1",
        session_id="session-origin",
    )


def _version(
    version: int = 1,
    *,
    content: str = "Completed maps to stored value complete.",
) -> MemoryVersion:
    return MemoryVersion(
        memory_id="memory-1",
        version=version,
        content=content,
        creator=MemoryCreator.LEARNING_SERVICE,
        confidence=0.95,
        sensitivity=MemorySensitivity.INTERNAL,
        provenance=_provenance(),
        attributes={"business_term": "completed", "stored_value": "complete"},
        resource_revision="revision-1",
        supersedes_version=None if version == 1 else version - 1,
        created_at=NOW + timedelta(minutes=version),
    )


def _record(version: int = 1) -> MemoryRecord:
    return MemoryRecord(
        id="memory-1",
        scope=MemoryScope(
            agent_id="agent-1",
            source_id="source-1",
            resource_id="resource-1",
        ),
        kind=MemoryKind.RESOURCE_ALIAS,
        logical_key="customers.status:completed",
        current_version=version,
        state=MemoryState.ACTIVE,
        created_at=NOW,
        updated_at=NOW + timedelta(minutes=version),
    )


def test_scope_and_logical_identity_are_normalized_and_stable() -> None:
    assert normalize_memory_logical_key("  CUSTOMERS.STATUS:Completed  ") == (
        "customers.status:completed"
    )
    assert normalize_memory_logical_key("ＣＯＭＰＬＥＴＥＤ") == "completed"

    broad = MemoryScope(agent_id="agent-1")
    scoped = _record().scope
    request = MemoryScope(
        agent_id="agent-1",
        user_id="user-1",
        session_id="later-session",
        source_id="source-1",
        resource_id="resource-1",
    )
    assert broad.contains(request)
    assert scoped.contains(request)
    assert not scoped.contains(MemoryScope(agent_id="agent-1"))
    assert broad.fingerprint != scoped.fingerprint

    first = _record(1)
    corrected = replace(first, current_version=2, updated_at=NOW + timedelta(hours=1))
    assert corrected.logical_identity == first.logical_identity

    with pytest.raises(ValueError, match="requires a source_id"):
        MemoryScope(agent_id="agent-1", resource_id="resource-1")
    with pytest.raises(ValueError, match="already be normalized"):
        replace(first, logical_key="Customers.Status:Completed")


def test_provenance_requires_one_resolvable_origin() -> None:
    assert _provenance().operation_id == "operation-1"
    evidence = MemoryProvenance(
        kind=MemoryProvenanceKind.ACCEPTED_EVIDENCE,
        content_hash=HASH,
        operation_id="operation-1",
        evidence_id="evidence-1",
    )
    imported = MemoryProvenance(
        kind=MemoryProvenanceKind.IMPORT,
        content_hash=HASH,
        external_ref="import://bundle/1",
    )
    assert evidence.evidence_id == "evidence-1"
    assert imported.external_ref == "import://bundle/1"

    with pytest.raises(ValueError, match="operation_id and trigger_id"):
        replace(_provenance(), operation_id=None)
    with pytest.raises(ValueError, match="lowercase sha256"):
        replace(_provenance(), content_hash="not-a-hash")
    with pytest.raises(ValueError, match="cannot reference evidence"):
        replace(_provenance(), evidence_id="evidence-1")


def test_versions_are_bounded_immutable_and_history_preserves_provenance() -> None:
    version_1 = _version()
    version_2 = _version(2, content="Completed now maps to closed.")
    record = _record(2)
    snapshot = MemorySnapshot(record, version_2)
    history = MemoryHistory(record, (version_2, version_1))

    assert isinstance(version_1.attributes, FrozenJsonObject)
    assert history.versions == (version_1, version_2)
    assert history.current is version_2
    assert snapshot.version.provenance.trigger_id == "trigger-1"
    with pytest.raises(TypeError):
        version_1.attributes["new"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        version_1.content = "mutated"  # type: ignore[misc]

    with pytest.raises(ValueError, match="must supersede"):
        replace(version_2, supersedes_version=None)
    with pytest.raises(ValueError, match="follow created_at"):
        replace(version_1, expires_at=NOW)
    with pytest.raises(ValueError, match="resource-scoped"):
        MemorySnapshot(
            replace(record, scope=MemoryScope(agent_id="agent-1")),
            version_2,
        )


def test_supersession_and_restore_requests_are_version_guarded() -> None:
    replacement = _version(2, content="Completed maps to closed.")
    supersession = MemorySupersessionRequest(
        agent_id="agent-1",
        memory_id="memory-1",
        expected_version=1,
        replacement=replacement,
    )
    assert supersession.replacement.supersedes_version == 1

    restore_version = _version(3)
    restore = MemoryRestoreRequest(
        agent_id="agent-1",
        memory_id="memory-1",
        expected_version=2,
        restore_version=1,
        replacement=restore_version,
    )
    assert restore.restore_version == 1

    with pytest.raises(ValueError, match="follow expected_version"):
        replace(supersession, expected_version=2)
    with pytest.raises(ValueError, match="precede the current"):
        replace(restore, restore_version=2)
