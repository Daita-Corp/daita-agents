"""Shared helpers extracted from ``test_effect_receipts.py``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from daita.identity import AgentIdentity
from daita.loop.models import RunInput
from daita.storage.sqlite import SQLiteStateStore

STARTED_AT = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


async def _store(path: Path) -> SQLiteStateStore:
    store = await SQLiteStateStore.open(path, clock=lambda: STARTED_AT)
    await store.initialize_identity(
        AgentIdentity("agent-effect", "Effects", STARTED_AT)
    )
    await store.start(
        RunInput(
            id="run-effect",
            agent_id="agent-effect",
            message="exact effect",
            created_at=STARTED_AT,
            conversation_id="conversation-effect",
        )
    )
    return store
