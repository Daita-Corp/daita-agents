from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from daita.adapters.models import SourceRegistration
from daita.adapters.protocols import SourceStore
from daita.identity import AgentIdentity, AgentIdentityConflictError
from daita.storage.sqlite import SQLiteOperationStore, SQLiteStoreError

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _registration(
    native_identity: str,
    *,
    attached_offset: int = 0,
    agent_id: str = "agent-1",
) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id=agent_id,
        adapter_id="sqlite",
        native_identity=native_identity,
        display_name=native_identity.rsplit("/", 1)[-1],
        configuration={"path": native_identity, "read_only": True},
        attached_at=NOW + timedelta(seconds=attached_offset),
    )


async def _open_store(path) -> SQLiteOperationStore:
    store = await SQLiteOperationStore.open(path)
    await store.initialize_identity(
        AgentIdentity(
            id="agent-1",
            display_name="Source test agent",
            created_at=NOW,
        )
    )
    return store


async def test_source_store_registers_lists_detaches_and_reopens(tmp_path) -> None:
    path = tmp_path / "agent.sqlite3"
    store = await _open_store(path)
    first = _registration("/data/first.sqlite")
    second = _registration("/data/second.sqlite", attached_offset=1)

    assert isinstance(store, SourceStore)
    assert await store.register_source(first) == first
    assert await store.register_source(first) == first
    assert await store.register_source(second) == second
    assert await store.load_source("agent-1", first.id) == first
    assert await store.load_source("other-agent", first.id) is None
    assert await store.list_sources("agent-1") == (first, second)

    detached = await store.detach_source(
        "agent-1",
        first.id,
        NOW + timedelta(seconds=2),
    )
    assert detached.active is False
    assert await store.list_sources("agent-1") == (detached, second)
    await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        assert await reopened.load_source("agent-1", first.id) == detached
        assert await reopened.list_sources("agent-1") == (detached, second)
    finally:
        await reopened.close()


async def test_source_registration_conflicts_and_detach_are_fail_closed(
    tmp_path,
) -> None:
    store = await _open_store(tmp_path / "agent.sqlite3")
    registration = _registration("/data/source.sqlite")
    await store.register_source(registration)
    try:
        with pytest.raises(SQLiteStoreError, match="registration conflict"):
            await store.register_source(
                replace(registration, display_name="changed without a new identity")
            )
        with pytest.raises(AgentIdentityConflictError, match="authoritative"):
            await store.register_source(
                _registration("/data/other.sqlite", agent_id="agent-2")
            )
        with pytest.raises(SQLiteStoreError, match="unknown source"):
            await store.detach_source(
                "other-agent",
                registration.id,
                NOW + timedelta(seconds=1),
            )

        detached = await store.detach_source(
            "agent-1",
            registration.id,
            NOW + timedelta(seconds=1),
        )
        with pytest.raises(SQLiteStoreError, match="already detached"):
            await store.detach_source(
                "agent-1",
                registration.id,
                NOW + timedelta(seconds=2),
            )
        with pytest.raises(ValueError, match="already detached"):
            await store.register_source(detached)
        assert await store.load_source("agent-1", registration.id) == detached
    finally:
        await store.close()
