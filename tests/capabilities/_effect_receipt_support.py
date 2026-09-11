"""Shared helpers extracted from ``test_effect_receipts.py``."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import EffectEvidenceBasis, EffectObservation, EffectOutcome
from daita.identity import AgentIdentity
from daita.llm.models import ModelSensitivity
from daita.loop.models import RunInput
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_codecs import decode_receipt, encode_receipt
from daita.storage.sqlite_records import (
    EffectReceipt,
    EffectReceiptConflictError,
    EffectResolution,
    EffectResolutionDecision,
    EffectUnresolvedError,
    effect_receipt_id,
)

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
