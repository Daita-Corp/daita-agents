from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from daita.memory import (
    MemoryCreator,
    MemoryHistory,
    MemoryInspectionRequest,
    MemoryKind,
    MemoryListRequest,
    MemoryNotFoundError,
    MemoryProvenance,
    MemoryProvenanceKind,
    MemoryQualification,
    MemoryRecallRequest,
    MemoryRecord,
    MemoryRestoreRequest,
    MemoryScope,
    MemorySensitivity,
    MemoryService,
    MemoryServiceContractError,
    MemorySnapshot,
    MemoryState,
    MemorySupersessionRequest,
    MemoryVersion,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
HASH = "sha256:" + "b" * 64


class FakeMemoryStore:
    def __init__(self, candidates: tuple[MemorySnapshot, ...] = ()) -> None:
        self.candidates = candidates
        self.histories: dict[str, MemoryHistory] = {}
        self.lifecycle_result: MemoryHistory | None = None
        self.recall_call: dict[str, Any] | None = None
        self.list_call: dict[str, Any] | None = None
        self.supersede_call: MemorySupersessionRequest | None = None
        self.restore_call: MemoryRestoreRequest | None = None

    async def recall_candidates(self, **kwargs: Any) -> tuple[MemorySnapshot, ...]:
        self.recall_call = kwargs
        return self.candidates

    async def list_candidates(self, **kwargs: Any) -> tuple[MemorySnapshot, ...]:
        self.list_call = kwargs
        return self.candidates

    async def load_history(self, agent_id: str, memory_id: str) -> MemoryHistory | None:
        return self.histories.get(memory_id)

    async def supersede(self, request: MemorySupersessionRequest) -> MemoryHistory:
        self.supersede_call = request
        assert self.lifecycle_result is not None
        return self.lifecycle_result

    async def restore(self, request: MemoryRestoreRequest) -> MemoryHistory:
        self.restore_call = request
        assert self.lifecycle_result is not None
        return self.lifecycle_result


def _provenance() -> MemoryProvenance:
    return MemoryProvenance(
        kind=MemoryProvenanceKind.USER_STATEMENT,
        content_hash=HASH,
        operation_id="operation-origin",
        trigger_id="trigger-origin",
        session_id="session-origin",
    )


def _snapshot(
    identifier: str,
    content: str,
    *,
    logical_key: str | None = None,
    attributes: dict[str, object] | None = None,
    scope: MemoryScope | None = None,
    sensitivity: MemorySensitivity = MemorySensitivity.INTERNAL,
    state: MemoryState = MemoryState.ACTIVE,
    expires_at: datetime | None = None,
    revision: str | None = None,
    confidence: float = 0.9,
) -> MemorySnapshot:
    resolved_scope = scope or MemoryScope(
        agent_id="agent-1",
        source_id="source-1",
        resource_id="resource-1",
    )
    record = MemoryRecord(
        id=identifier,
        scope=resolved_scope,
        kind=MemoryKind.RESOURCE_ALIAS,
        logical_key=logical_key or f"customers.status:{identifier}",
        current_version=1,
        state=state,
        superseded_by_id=(
            "replacement-memory" if state is MemoryState.SUPERSEDED else None
        ),
        created_at=NOW - timedelta(days=2),
        updated_at=NOW - timedelta(days=1),
    )
    version = MemoryVersion(
        memory_id=identifier,
        version=1,
        content=content,
        creator=MemoryCreator.LEARNING_SERVICE,
        confidence=confidence,
        sensitivity=sensitivity,
        provenance=_provenance(),
        attributes=attributes or {"term": "completed"},
        expires_at=expires_at,
        resource_revision=revision,
        created_at=NOW - timedelta(days=2),
    )
    return MemorySnapshot(record, version)


def _request(**changes: Any) -> MemoryRecallRequest:
    values: dict[str, Any] = {
        "query": "completed customers status",
        "scope": MemoryScope(
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-new",
            source_id="source-1",
            resource_id="resource-1",
        ),
        "current_resource_revision": "revision-current",
        "limit": 5,
        "character_budget": 4_000,
    }
    values.update(changes)
    return MemoryRecallRequest(**values)


async def test_recall_filters_authority_facts_before_deterministic_ranking() -> None:
    requested = _request().scope
    candidates = (
        _snapshot(
            "specific",
            "For customers, completed status maps to stored value complete.",
            revision="revision-current",
            confidence=0.95,
        ),
        _snapshot(
            "global",
            "Completed is a supported business status.",
            logical_key="business.status:completed",
            scope=MemoryScope(agent_id="agent-1"),
            sensitivity=MemorySensitivity.PUBLIC,
            confidence=0.7,
        ),
        _snapshot(
            "other-source",
            "Completed customers use other source rules.",
            scope=MemoryScope(agent_id="agent-1", source_id="source-2"),
        ),
        _snapshot(
            "other-user",
            "Completed customers use another user's preference.",
            scope=replace(requested, user_id="user-2", session_id=None),
            revision="revision-current",
        ),
        _snapshot(
            "other-session",
            "Completed customers use a private session rule.",
            scope=replace(requested, session_id="session-other"),
            revision="revision-current",
        ),
        _snapshot(
            "restricted",
            "Completed customers contain restricted semantics.",
            sensitivity=MemorySensitivity.RESTRICTED,
            revision="revision-current",
        ),
        _snapshot(
            "expired",
            "Completed customers use an expired mapping.",
            expires_at=NOW - timedelta(hours=1),
            revision="revision-current",
        ),
        _snapshot(
            "stale",
            "Completed customers use a stale mapping.",
            revision="revision-old",
        ),
        _snapshot(
            "superseded",
            "Completed customers use an obsolete mapping.",
            state=MemoryState.SUPERSEDED,
            revision="revision-current",
        ),
        _snapshot(
            "irrelevant",
            "Warehouse fiscal calendars begin Monday.",
            logical_key="warehouse.fiscal-calendar",
            attributes={"topic": "calendar"},
            revision="revision-current",
        ),
    )
    store = FakeMemoryStore(candidates)
    service = MemoryService(store, clock=lambda: NOW)

    result = await service.recall(_request())

    assert [hit.memory.snapshot.record.id for hit in result.hits] == [
        "specific",
        "global",
    ]
    assert [hit.memory.qualification for hit in result.hits] == [
        MemoryQualification.CURRENT,
        MemoryQualification.UNBOUND,
    ]
    assert result.hits[0].lexical_score > result.hits[1].lexical_score
    assert result.omitted_by_scope == 3
    assert result.omitted_by_sensitivity == 1
    assert result.omitted_by_lifecycle == 2
    assert result.omitted_by_revision == 1
    assert result.omitted_by_relevance == 1
    assert result.used_characters <= 4_000
    assert store.recall_call is not None
    assert store.recall_call["states"] == (MemoryState.ACTIVE,)
    assert store.recall_call["unexpired_at"] == NOW
    assert store.recall_call["limit"] == _request().candidate_limit


async def test_recall_enforces_limit_budget_and_store_contract() -> None:
    candidates = (
        _snapshot("a", "Completed customers status mapping alpha."),
        _snapshot("b", "Completed customers status mapping beta."),
    )
    store = FakeMemoryStore(candidates)
    service = MemoryService(store, clock=lambda: NOW)

    limited = await service.recall(_request(limit=1))
    budgeted = await service.recall(_request(character_budget=50))

    assert len(limited.hits) == 1
    assert limited.omitted_by_limit == 1
    assert limited.truncated is True
    assert budgeted.hits == ()
    assert budgeted.omitted_by_budget == 2
    assert budgeted.truncated is True

    store.candidates = (candidates[0], candidates[0])
    with pytest.raises(MemoryServiceContractError, match="duplicate"):
        await service.recall(_request())


async def test_list_and_inspect_keep_stale_expired_and_history_visible() -> None:
    active = _snapshot(
        "active", "Completed customers mapping.", revision="revision-current"
    )
    stale = _snapshot(
        "stale", "Completed customers stale mapping.", revision="revision-old"
    )
    expired = _snapshot(
        "expired",
        "Completed customers expired mapping.",
        revision="revision-current",
        expires_at=NOW - timedelta(hours=1),
    )
    superseded = _snapshot(
        "superseded",
        "Completed customers old mapping.",
        state=MemoryState.SUPERSEDED,
        revision="revision-current",
    )
    store = FakeMemoryStore((superseded, stale, expired, active))
    store.histories["stale"] = MemoryHistory(stale.record, (stale.version,))
    service = MemoryService(store, clock=lambda: NOW)
    scope = _request().scope

    default = await service.list(
        MemoryListRequest(
            scope=scope,
            current_resource_revision="revision-current",
        )
    )
    with_history = await service.list(
        MemoryListRequest(
            scope=scope,
            current_resource_revision="revision-current",
            include_superseded=True,
        )
    )
    inspected = await service.inspect(
        MemoryInspectionRequest(
            agent_id="agent-1",
            memory_id="stale",
            current_resource_revision="revision-current",
        )
    )

    assert {item.snapshot.record.id for item in default.items} == {
        "active",
        "expired",
        "stale",
    }
    qualifications = {
        item.snapshot.record.id: item.qualification for item in default.items
    }
    assert qualifications["expired"] is MemoryQualification.EXPIRED
    assert qualifications["stale"] is MemoryQualification.STALE_REVISION
    assert {item.snapshot.record.id for item in with_history.items} == {
        "active",
        "expired",
        "stale",
        "superseded",
    }
    assert inspected.qualification is MemoryQualification.STALE_REVISION
    assert inspected.history.current.provenance.operation_id == "operation-origin"

    with pytest.raises(MemoryNotFoundError):
        await service.inspect(
            MemoryInspectionRequest(agent_id="agent-1", memory_id="missing")
        )


def _version(
    version: int,
    content: str,
    *,
    supersedes: int | None,
) -> MemoryVersion:
    return MemoryVersion(
        memory_id="memory-lifecycle",
        version=version,
        content=content,
        creator=MemoryCreator.USER,
        confidence=1.0,
        sensitivity=MemorySensitivity.INTERNAL,
        provenance=_provenance(),
        attributes={"term": "completed", "stored_value": content.rsplit(" ", 1)[-1]},
        resource_revision="revision-current",
        supersedes_version=supersedes,
        created_at=NOW + timedelta(minutes=version),
    )


def _history(current: int, versions: tuple[MemoryVersion, ...]) -> MemoryHistory:
    record = MemoryRecord(
        id="memory-lifecycle",
        scope=MemoryScope(
            agent_id="agent-1",
            source_id="source-1",
            resource_id="resource-1",
        ),
        kind=MemoryKind.RESOURCE_ALIAS,
        logical_key="customers.status:completed",
        current_version=current,
        state=MemoryState.ACTIVE,
        created_at=NOW,
        updated_at=NOW + timedelta(minutes=current),
    )
    return MemoryHistory(record, versions)


async def test_supersession_and_restore_validate_history_before_store_mutation() -> (
    None
):
    version_1 = _version(1, "Completed maps to complete", supersedes=None)
    version_2 = _version(2, "Completed maps to closed", supersedes=1)
    correction = _version(3, "Completed maps to finalized", supersedes=2)
    current = _history(2, (version_1, version_2))
    corrected = _history(3, (version_1, version_2, correction))
    store = FakeMemoryStore()
    store.histories["memory-lifecycle"] = current
    store.lifecycle_result = corrected
    service = MemoryService(store, clock=lambda: NOW + timedelta(hours=1))

    correction_request = MemorySupersessionRequest(
        agent_id="agent-1",
        memory_id="memory-lifecycle",
        expected_version=2,
        replacement=correction,
    )
    result = await service.supersede(correction_request)

    assert result.history.current is correction
    assert store.supersede_call is correction_request
    assert result.history.record.logical_identity == current.record.logical_identity

    restored_copy = replace(
        version_1,
        version=3,
        supersedes_version=2,
        created_at=NOW + timedelta(minutes=3),
    )
    restored = _history(3, (version_1, version_2, restored_copy))
    store.lifecycle_result = restored
    restore_request = MemoryRestoreRequest(
        agent_id="agent-1",
        memory_id="memory-lifecycle",
        expected_version=2,
        restore_version=1,
        replacement=restored_copy,
    )
    restore_result = await service.restore(restore_request)

    assert restore_result.history.current.content == version_1.content
    assert store.restore_call is restore_request

    bad_restore = replace(
        restore_request,
        replacement=replace(restored_copy, content="Different semantic payload"),
    )
    with pytest.raises(ValueError, match="historical content"):
        await service.restore(bad_restore)
