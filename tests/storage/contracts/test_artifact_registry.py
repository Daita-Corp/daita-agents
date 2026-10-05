"""Metadata lifecycle and stable artifact pagination across durable backends."""

from dataclasses import replace

import pytest

from daita.artifacts.models import ArtifactRecord, ArtifactState
from daita.identity import AgentIdentity
from daita.loop.models import RunInput
from daita.storage.protocols import StateStore
from tests.artifacts._public_surface_support import _surface_records

pytestmark = [pytest.mark.integration, pytest.mark.contract]


async def test_artifact_keyset_survives_cursor_deletion_and_preserves_scope(
    state_store: StateStore,
):
    ref, _, _ = _surface_records()
    await state_store.initialize_identity(
        AgentIdentity("agent-1", "Artifacts", ref.created_at)
    )
    await state_store.start(
        RunInput(
            id=ref.run_id,
            agent_id="agent-1",
            caller_principal_id="reader",
            conversation_id=ref.conversation_id,
            message="test",
            created_at=ref.created_at,
        )
    )
    records = []
    for index in range(1, 4):
        record = ArtifactRecord(
            ref=replace(ref, artifact_id=f"artifact-{index:032x}"),
            agent_id="agent-1",
            caller_principal_id="reader",
            state=ArtifactState.CREATING,
        )
        await state_store.begin_artifact_creation(record)
        records.append(
            await state_store.transition_artifact(record, ArtifactState.READY)
        )
    first = await state_store.list_artifact_records("agent-1", limit=2)
    assert first == tuple(records[:2])
    cursor = (first[-1].ref.created_at, first[-1].ref.artifact_id)
    deleting = await state_store.transition_artifact(first[-1], ArtifactState.DELETING)
    await state_store.finish_artifact_deletion(deleting)
    assert await state_store.list_artifact_records(
        "agent-1", limit=2, after=cursor
    ) == (records[2],)
    assert await state_store.list_artifact_refs("agent-1", limit=2, after=cursor) == (
        records[2].ref,
    )
    assert (
        await state_store.list_artifact_records("another-agent", limit=2, after=cursor)
        == ()
    )
    assert (
        await state_store.list_artifact_records(
            "agent-1", caller_principal_id="another-reader", limit=2
        )
        == ()
    )
    with pytest.raises(ValueError):
        await state_store.list_artifact_records("agent-1", after=cursor)
    with pytest.raises(ValueError):
        await state_store.list_artifact_records(
            "agent-1", limit=2, offset=1, after=cursor
        )
