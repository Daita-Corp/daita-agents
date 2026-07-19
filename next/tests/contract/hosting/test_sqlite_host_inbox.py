from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita.events.models import RuntimeEvent
from daita.hosting.inbox import (
    HostInboxEnqueueConflictError,
    HostInboxItem,
    HostInboxKind,
    HostInboxRevisionConflict,
    HostInboxStatus,
    HostMutationAdmission,
    HostMutationConflictError,
    host_inbox_request_hash,
    host_mutation_request_hash,
)
from daita.identity import AgentIdentity
from daita.loop.models import LoopBudgets, LoopState
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
AGENT_ID = "agent-host-inbox"


def _trigger_item(
    item_id: str,
    *,
    trigger_id: str,
    idempotency_key: str,
    created_at: datetime = NOW,
    payload: dict[str, object] | None = None,
) -> HostInboxItem:
    request_payload = (
        {"message": f"run {trigger_id}", "options": {"bounded": True}}
        if payload is None
        else payload
    )
    return HostInboxItem(
        id=item_id,
        agent_id=AGENT_ID,
        kind=HostInboxKind.TRIGGER,
        idempotency_key=idempotency_key,
        request_hash=host_inbox_request_hash(
            kind=HostInboxKind.TRIGGER,
            payload=request_payload,
            trigger_id=trigger_id,
        ),
        payload=request_payload,
        revision=1,
        status=HostInboxStatus.PENDING,
        created_at=created_at,
        updated_at=created_at,
        trigger_id=trigger_id,
    )


def _operation(
    item: HostInboxItem,
    *,
    operation_id: str,
    trigger_id: str | None = None,
) -> OperationSnapshot:
    operation_trigger_id = item.trigger_id if trigger_id is None else trigger_id
    assert operation_trigger_id is not None
    trigger = AgentTrigger(
        id=operation_trigger_id,
        agent_id=AGENT_ID,
        kind=TriggerKind.INTERNAL,
        source_id="host:inbox",
        payload=item.payload,
        created_at=item.created_at,
    )
    operation = Operation(
        id=operation_id,
        agent_id=AGENT_ID,
        trigger_id=trigger.id,
        status=OperationStatus.RUNNING,
        created_at=item.created_at,
        updated_at=item.created_at,
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(),
        budgets=LoopBudgets(),
        turns=(),
        model_calls=(),
        readiness=(),
        tasks=(),
        evidence=(),
        observations=(),
        events=(
            RuntimeEvent(
                id=f"{operation_id}:created",
                type="operation.created",
                agent_id=AGENT_ID,
                operation_id=operation_id,
                payload={"trigger_id": trigger.id},
                created_at=item.created_at,
            ),
        ),
    )


async def _open_store(path: Path) -> SQLiteOperationStore:
    store = await SQLiteOperationStore.open(path)
    await store.initialize_identity(
        AgentIdentity(
            id=AGENT_ID,
            display_name="Host Inbox Contract",
            created_at=NOW,
        )
    )
    return store


def test_host_inbox_request_identity_is_canonical_and_tamper_evident() -> None:
    left: dict[str, object] = {
        "message": "run",
        "options": {"a": 1, "b": 2},
    }
    right: dict[str, object] = {
        "options": {"b": 2, "a": 1},
        "message": "run",
    }

    assert host_inbox_request_hash(
        kind=HostInboxKind.TRIGGER,
        payload=left,
        trigger_id="trigger-canonical",
    ) == host_inbox_request_hash(
        kind=HostInboxKind.TRIGGER,
        payload=right,
        trigger_id="trigger-canonical",
    )

    item = _trigger_item(
        "inbox-canonical",
        trigger_id="trigger-canonical",
        idempotency_key="request-canonical",
        payload=left,
    )
    with pytest.raises(ValueError, match="request_hash"):
        replace(item, payload={"message": "tampered"})


async def test_enqueue_replays_exact_request_and_lists_bounded_fifo(
    tmp_path: Path,
) -> None:
    store = await _open_store(tmp_path / "state.db")
    try:
        first = _trigger_item(
            "inbox-z",
            trigger_id="trigger-z",
            idempotency_key="request-z",
        )
        tie_breaker = _trigger_item(
            "inbox-a",
            trigger_id="trigger-a",
            idempotency_key="request-a",
        )
        later = _trigger_item(
            "inbox-later",
            trigger_id="trigger-later",
            idempotency_key="request-later",
            created_at=NOW + timedelta(seconds=1),
        )
        for item in (first, tie_breaker, later):
            assert await store.enqueue_host_inbox(item) == item

        replay = replace(
            first,
            id="inbox-retry",
            created_at=NOW + timedelta(seconds=2),
            updated_at=NOW + timedelta(seconds=2),
        )
        assert await store.enqueue_host_inbox(replay) == first
        assert await store.list_pending_host_inbox(
            AGENT_ID,
            limit=2,
        ) == (tie_breaker, first)

        changed_payload = {"message": "different"}
        conflicting = replace(
            replay,
            request_hash=host_inbox_request_hash(
                kind=HostInboxKind.TRIGGER,
                payload=changed_payload,
                trigger_id=first.trigger_id,
            ),
            payload=changed_payload,
        )
        with pytest.raises(HostInboxEnqueueConflictError):
            await store.enqueue_host_inbox(conflicting)
    finally:
        await store.close()


async def test_completion_is_trigger_linked_cas_and_survives_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    item = _trigger_item(
        "inbox-complete",
        trigger_id="trigger-complete",
        idempotency_key="request-complete",
    )
    completion = replace(
        item,
        revision=2,
        status=HostInboxStatus.COMPLETED,
        updated_at=NOW + timedelta(seconds=1),
        operation_id="operation-complete",
    )

    store = await _open_store(path)
    try:
        await store.enqueue_host_inbox(item)
        await store.create(_operation(item, operation_id="operation-complete"))
        assert (
            await store.complete_host_inbox(
                completion,
                expected_revision=1,
            )
            == completion
        )
        assert await store.list_pending_host_inbox(AGENT_ID, limit=10) == ()
        with pytest.raises(HostInboxRevisionConflict):
            await store.complete_host_inbox(completion, expected_revision=1)
    finally:
        await store.close()

    reopened = await _open_store(path)
    try:
        replay = replace(
            item,
            id="inbox-complete-retry",
            created_at=NOW + timedelta(minutes=1),
            updated_at=NOW + timedelta(minutes=1),
        )
        assert await reopened.enqueue_host_inbox(replay) == completion
        assert await reopened.list_pending_host_inbox(AGENT_ID, limit=10) == ()
        assert sqlite_owner._MIGRATIONS[-1].version == 13
        assert sqlite_owner._MIGRATIONS[-1].name == "persist_task_validation_facts"
    finally:
        await reopened.close()


async def test_invalid_operation_link_rolls_back_completion(
    tmp_path: Path,
) -> None:
    item = _trigger_item(
        "inbox-wrong-link",
        trigger_id="trigger-expected",
        idempotency_key="request-wrong-link",
    )
    store = await _open_store(tmp_path / "state.db")
    try:
        await store.enqueue_host_inbox(item)
        await store.create(
            _operation(
                item,
                operation_id="operation-other-trigger",
                trigger_id="trigger-other",
            )
        )
        invalid = replace(
            item,
            revision=2,
            status=HostInboxStatus.COMPLETED,
            updated_at=NOW + timedelta(seconds=1),
            operation_id="operation-other-trigger",
        )
        with pytest.raises(ValueError, match="another trigger"):
            await store.complete_host_inbox(invalid, expected_revision=1)

        assert await store.list_pending_host_inbox(AGENT_ID, limit=10) == (item,)
    finally:
        await store.close()


async def test_mutation_admission_binds_key_to_exact_request_across_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    params: dict[str, object] = {
        "operation_id": "operation-a",
        "reason": "operator_cancelled",
    }
    request = HostMutationAdmission(
        agent_id=AGENT_ID,
        idempotency_key="cancel-request-1",
        method="operation.cancel",
        request_hash=host_mutation_request_hash(
            method="operation.cancel",
            params=params,
        ),
        created_at=NOW,
    )

    store = await _open_store(path)
    try:
        assert await store.admit_host_mutation(request) == request
        replay = replace(request, created_at=NOW + timedelta(minutes=1))
        assert await store.admit_host_mutation(replay) == request

        conflict = replace(
            replay,
            request_hash=host_mutation_request_hash(
                method="operation.cancel",
                params={**params, "operation_id": "operation-b"},
            ),
        )
        with pytest.raises(HostMutationConflictError):
            await store.admit_host_mutation(conflict)
    finally:
        await store.close()

    reopened = await _open_store(path)
    try:
        assert (
            await reopened.admit_host_mutation(
                replace(request, created_at=NOW + timedelta(hours=1))
            )
            == request
        )
    finally:
        await reopened.close()
