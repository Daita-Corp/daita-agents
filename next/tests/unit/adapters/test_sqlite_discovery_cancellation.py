from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading

import pytest

import daita.adapters.sqlite as sqlite_adapter
from daita.adapters import DiscoveryRequest, SQLiteSource

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _create_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE durable (id INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()


async def test_discovery_cancellation_waits_for_worker_rollback_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.sqlite3"
    _create_database(path)
    adapter = await SQLiteSource(path).open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW,
    )
    request = DiscoveryRequest(
        agent_id="agent-1",
        source_id=adapter.registration.id,
        sync_id="sync-cancel",
        requested_at=NOW,
    )
    original_discover = sqlite_adapter._discover
    worker_started = threading.Event()
    allow_rollback = threading.Event()
    rollback_finished = threading.Event()

    def blocking_discover(*args, **kwargs):
        connection = args[0]
        connection.execute("BEGIN")
        worker_started.set()
        if not allow_rollback.wait(timeout=5):
            raise TimeoutError("test did not release discovery worker")
        connection.execute("ROLLBACK")
        rollback_finished.set()
        return original_discover(*args, **kwargs)

    monkeypatch.setattr(sqlite_adapter, "_discover", blocking_discover)
    discovery = asyncio.create_task(adapter.discover(request))
    close = None
    try:
        assert await asyncio.to_thread(worker_started.wait, 1)
        discovery.cancel()
        await asyncio.sleep(0)
        assert not discovery.done()

        close = asyncio.create_task(adapter.close())
        await asyncio.sleep(0)
        assert not close.done()

        allow_rollback.set()
        with pytest.raises(asyncio.CancelledError):
            await discovery
        assert rollback_finished.is_set()
        await close
        assert adapter._latest is None
    finally:
        allow_rollback.set()
        await asyncio.gather(discovery, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await adapter.close()
