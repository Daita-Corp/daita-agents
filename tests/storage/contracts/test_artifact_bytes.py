"""One artifact lifecycle with remote bytes and either real state database."""

import asyncio
import threading

import pytest

from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
    ArtifactProvenance,
    ArtifactState,
)
from daita.artifacts.store import AgentHomeArtifactStore
from daita.capabilities import ArtifactPolicy
from daita.catalog.models import Sensitivity
from daita.identity import AgentIdentity
from daita.storage.errors import StorageCommitUnknownError, StorageOwnershipLostError
from tests.artifacts.byte_storage_support import ObjectClient, ObjectServiceError
from tests.support.graph import GRAPH_NOW

pytestmark = [pytest.mark.integration, pytest.mark.contract]

RUN_ID = "run-" + "1" * 32
ARTIFACT_ID = "artifact-" + "2" * 32


async def _open(registry, client):
    if await registry.load_identity() is None:
        await registry.initialize_identity(
            AgentIdentity(id="agent-one", display_name="Example", created_at=GRAPH_NOW)
        )
    return await AgentHomeArtifactStore.open(
        agent_id="agent-one",
        byte_storage=client.storage(),
        registry=registry,
        clock=lambda: GRAPH_NOW,
        id_factory=lambda _: ARTIFACT_ID,
    )


async def _commit(store):
    return await store.commit(
        ArtifactDraft(
            content=b"durable result",
            suggested_filename="result.txt",
            media_type="text/plain",
            sensitivity=Sensitivity.INTERNAL,
            provenance=ArtifactProvenance(
                authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
            ),
        ),
        ArtifactPolicy(
            allowed_media_types=frozenset({"text/plain"}),
            allowed_authorships=frozenset({ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS}),
            allowed_extensions=(("text/plain", (".txt",)),),
            artifact_required=True,
            max_bytes_per_artifact=1024,
            max_total_bytes_per_call=1024,
            max_artifact_count=1,
        ),
        run_id=RUN_ID,
        conversation_id="conversation-one",
        call_id="create",
        capability_id="artifact.create_document",
        caller_principal_id="member-one",
    )


async def test_remote_bytes_reopen_and_delete_without_a_home(state_store_factory):
    client = ObjectClient()
    async with state_store_factory() as registry:
        store = await _open(registry, client)
        ref = await _commit(store)
        assert (await store.read(ref.artifact_id)).content == b"durable result"
        assert await store.list_refs(caller_principal_id="member-two") == ()
        assert await store.list_refs(caller_principal_id="member-one") == (ref,)
        await store.close()
    async with state_store_factory() as registry:
        store = await _open(registry, client)
        assert await store.list_refs() == (ref,)
        assert (await store.read_ref(ref)).content == b"durable result"
        assert await store.delete(ref.artifact_id)
        assert not await store.delete(ref.artifact_id)
        assert await registry.get_artifact_record(ref.artifact_id) is None
        assert not client.objects
    assert len(client.publications) == 1
    assert all(body.closed for body in client.bodies)


async def test_unknown_state_failure_during_admission_stops_open(
    state_store_factory, monkeypatch
):
    client = ObjectClient()

    async def unavailable(*args, **kwargs):
        raise StorageCommitUnknownError("state owner has an unresolved commit")

    async with state_store_factory() as registry:
        await registry.initialize_identity(
            AgentIdentity(id="agent-one", display_name="Example", created_at=GRAPH_NOW)
        )
        monkeypatch.setattr(registry, "list_artifact_records", unavailable)
        with pytest.raises(StorageCommitUnknownError):
            await _open(registry, client)
    assert not client.reads and not client.publications and not client.deleted


async def test_lost_publication_ack_reconciles_exact_bytes_without_replay(
    state_store_factory, monkeypatch
):
    client = ObjectClient()
    put = client.put_object

    def lose_ack(**kwargs):
        put(**kwargs)
        raise TimeoutError("response lost after object publication")

    monkeypatch.setattr(client, "put_object", lose_ack)
    async with state_store_factory() as registry:
        store = await _open(registry, client)
        with pytest.raises(ArtifactError):
            await _commit(store)
        record = await registry.get_artifact_record(ARTIFACT_ID)
        assert record is not None and record.state is ArtifactState.READY
    async with state_store_factory() as registry:
        store = await _open(registry, client)
        assert (await store.read(ARTIFACT_ID)).content == b"durable result"
    assert len(client.publications) == 1


async def test_unavailable_readback_preserves_creation_for_recovery(
    state_store_factory, monkeypatch
):
    client = ObjectClient()

    def unavailable(**kwargs):
        raise ObjectServiceError("AccessDenied")

    async with state_store_factory() as registry:
        store = await _open(registry, client)
        with monkeypatch.context() as patch:
            patch.setattr(client, "get_object", unavailable)
            with pytest.raises(ArtifactError):
                await _commit(store)
            record = await registry.get_artifact_record(ARTIFACT_ID)
            assert record is not None and record.state is ArtifactState.CREATING
            assert not (await _open(registry, client)).available
        assert not client.deleted
    async with state_store_factory() as registry:
        store = await _open(registry, client)
        assert store.available
        assert (await store.read(ARTIFACT_ID)).content == b"durable result"
    assert len(client.publications) == 1


async def test_failed_deletion_stays_hidden_and_reopen_finishes_cleanup(
    state_store_factory, monkeypatch
):
    client = ObjectClient()

    def unavailable(**kwargs):
        raise ObjectServiceError("AccessDenied")

    async with state_store_factory() as registry:
        store = await _open(registry, client)
        ref = await _commit(store)
        with monkeypatch.context() as patch:
            patch.setattr(client, "delete_object", unavailable)
            with pytest.raises(ArtifactError):
                await store.delete(ref.artifact_id)
        assert await store.list_refs() == ()
        record = await registry.get_artifact_record(ref.artifact_id)
        assert record is not None and record.state is ArtifactState.DELETING
        assert client.objects
    async with state_store_factory() as registry:
        assert (await _open(registry, client)).available
        assert await registry.get_artifact_record(ref.artifact_id) is None
    assert not client.objects


@pytest.mark.parametrize(
    "failure", [StorageCommitUnknownError, StorageOwnershipLostError]
)
async def test_state_failure_during_cleanup_is_not_hidden(
    state_store_factory, monkeypatch, failure
):
    client = ObjectClient()

    async def state_failed(record):
        raise failure("state owner cannot confirm cleanup")

    async with state_store_factory() as registry:
        store = await _open(registry, client)
        ref = await _commit(store)
        with monkeypatch.context() as patch:
            patch.setattr(registry, "finish_artifact_deletion", state_failed)
            with pytest.raises(failure):
                await store.delete(ref.artifact_id)
        record = await registry.get_artifact_record(ref.artifact_id)
        assert record is not None and record.state is ArtifactState.DELETING
        assert not client.objects
    async with state_store_factory() as registry:
        assert (await _open(registry, client)).available
        assert await registry.get_artifact_record(ref.artifact_id) is None


@pytest.mark.parametrize("replacement", [None, b"tampered bytes"])
async def test_missing_or_corrupt_ready_bytes_do_not_erase_registry(
    state_store_factory, replacement
):
    client = ObjectClient()
    async with state_store_factory() as registry:
        store = await _open(registry, client)
        ref = await _commit(store)
        key = next(iter(client.objects))
        if replacement is None:
            del client.objects[key]
        else:
            client.objects[key] = replacement
        with pytest.raises(ArtifactError) as error:
            await store.read(ref.artifact_id)
        assert error.value.code == (
            "artifact_missing" if replacement is None else "artifact_corrupt"
        )
        record = await registry.get_artifact_record(ref.artifact_id)
        assert record is not None and record.state is ArtifactState.READY
        assert not client.deleted


async def test_cancelled_remote_publication_drains_without_blocking_event_loop(
    state_store_factory, monkeypatch
):
    client = ObjectClient()
    started, release = threading.Event(), threading.Event()
    put = client.put_object

    def blocked(**kwargs):
        started.set()
        assert release.wait(5)
        return put(**kwargs)

    monkeypatch.setattr(client, "put_object", blocked)
    async with state_store_factory() as registry:
        store = await _open(registry, client)
        task = asyncio.create_task(_commit(store))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (await store.read(ARTIFACT_ID)).content == b"durable result"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    assert len(client.publications) == 1
