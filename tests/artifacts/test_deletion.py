"""Durable deletion through the real home, storage and public API boundaries."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import daita.artifacts.delivery as artifact_delivery
import daita.artifacts.store as artifact_store
from daita import Agent
from daita.artifacts.models import (
    ArtifactDraft,
    ArtifactError,
    ArtifactRef,
    ArtifactState,
)
from daita.capabilities import ArtifactPolicy
from daita.errors import StateCompatibilityError
from daita.hosting import home_upgrade
from daita.hosting.embedded import _validate_agent_home_target
from daita.llm.models import FinishReason, ModelResponse
from daita.storage.home_migrations.revision_0003 import PRE_REGISTRY_REVISION_3_CHECKSUM
from tests.artifacts._public_surface_support import (
    MockModelProvider,
    _ids,
    _profile,
    _tool,
    workspace_for,
)
from tests.support.job_benchmarks import (
    OFFLINE_EAGER_LIMITS,
    TARGET_PROFILE_TABLE,
    create_probe_home,
    start_profile_response,
    stop_response,
    toolbox_load_response,
    wait_for_terminal,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def artifact_agent(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    provider = MockModelProvider(
        tuple(
            _tool(
                f"create-{index}",
                "artifact_create_document",
                {
                    "format": "txt",
                    "filename": f"result-{index}.txt",
                    "content": f"payload {index}",
                },
            )
            for index in range(2)
        )
        + (ModelResponse(finish_reason=FinishReason.STOP, text="created"),)
    )
    agent = await Agent.create(
        "deletion",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Create two text files.")
        assert len(result.artifacts) == 2
        yield agent, result, downloads
    finally:
        await agent.close()


@pytest.mark.acceptance
async def test_delete_hides_one_artifact_preserves_evidence_and_exports(
    artifact_agent, tmp_path
):
    agent, result, downloads = artifact_agent
    ref, other = result.artifacts
    assert await agent.list_artifacts(limit=1) == (ref,)
    assert await agent.list_artifacts(limit=1, offset=1) == (other,)
    assert await agent.list_artifacts(limit=1, offset=2) == ()
    original = await agent.transcript(result.run_id)
    exported = await agent.save_artifact(ref.artifact_id)
    assert await agent.delete_artifact(ref.artifact_id) is True
    assert await agent.delete_artifact(ref.artifact_id) is False
    assert await agent.transcript(result.run_id) == original
    assert (await agent.read_artifact(other.artifact_id)).content == b"payload 1"
    assert Path(exported.saved_path).read_bytes() == b"payload 0"
    assert await agent._embedded._artifact_store.list_refs() == (other,)
    assert await agent.list_artifacts() == (other,)
    for call in (agent.read_artifact, agent.save_artifact):
        with pytest.raises(ArtifactError) as error:
            await call(ref.artifact_id)
        assert error.value.code == "artifact_missing"
    home = tmp_path / "agents/deletion"
    assert not (home / "artifacts" / ref.run_id / ref.artifact_id).exists()
    await agent.close()
    reopened = await Agent.open(
        "deletion", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert await reopened.delete_artifact(ref.artifact_id) is False
        assert await reopened.transcript(result.run_id) == original
        assert (await reopened.read_artifact(other.artifact_id)).content == b"payload 1"
        # Completed deletion leaves no registry row, even after history is cleared.
        await reopened.clear_conversations()
        assert await reopened.list_artifacts() == (other,)
        assert await reopened.delete_artifact(ref.artifact_id) is False
    finally:
        await reopened.close()


@pytest.mark.parametrize("removed", [None, "payload", "manifest.json"])
@pytest.mark.parametrize("reopen", [False, True])
async def test_failed_cleanup_is_hidden_and_retryable(
    artifact_agent, tmp_path, monkeypatch, removed, reopen
):
    agent, result, _ = artifact_agent
    ref, other = result.artifacts
    directory = tmp_path / "agents/deletion/artifacts" / ref.run_id / ref.artifact_id

    def fail_cleanup(path):
        if removed:
            (path / removed).unlink()
        raise OSError("injected unlink failure")

    with monkeypatch.context() as patch:
        patch.setattr(artifact_store, "_remove_artifact_directory", fail_cleanup)
        with pytest.raises(ArtifactError) as error:
            await agent.delete_artifact(ref.artifact_id)
        assert error.value.details["stage"] == "delete_cleanup"
        assert directory.exists()
        with pytest.raises(ArtifactError) as missing:
            await agent.read_artifact(ref.artifact_id)
        assert missing.value.code == "artifact_missing"
        assert await agent._embedded._artifact_store.list_refs() == (other,)
    if reopen:
        await agent.close()
        agent = await Agent.open(
            "deletion", root=tmp_path, workspace=workspace_for(tmp_path)
        )
    try:
        assert await agent.delete_artifact(ref.artifact_id) is False
        assert not directory.exists()
        assert (await agent.read_artifact(other.artifact_id)).content == b"payload 1"
    finally:
        if reopen:
            await agent.close()


@pytest.mark.parametrize("unsafe_entry", ["run", "artifact", "payload", "extra"])
async def test_delete_rejects_symlinks_and_unexpected_files(
    artifact_agent, tmp_path, unsafe_entry
):
    agent, result, _ = artifact_agent
    ref = result.artifacts[0]
    run_path = tmp_path / "agents/deletion/artifacts" / ref.run_id
    directory = run_path / ref.artifact_id
    external = tmp_path / "external.txt"
    external.write_text("preserve outside bytes")
    original = tmp_path / "original"
    if unsafe_entry in {"run", "artifact"}:
        target = run_path if unsafe_entry == "run" else directory
        target.rename(original)
        target.symlink_to(original, target_is_directory=True)
    elif unsafe_entry == "payload":
        (directory / "payload").unlink()
        (directory / "payload").symlink_to(external)
    else:
        (directory / "unexpected.txt").write_text("preserve unexpected bytes")
    with pytest.raises(ArtifactError) as error:
        await agent.delete_artifact(ref.artifact_id)
    assert error.value.details["stage"] == "delete_cleanup"
    assert external.read_text() == "preserve outside bytes"
    assert await agent._embedded._store.get_artifact_record(ref.artifact_id) is not None
    if unsafe_entry in {"run", "artifact"}:
        assert target.is_symlink()
        preserved = original / ref.artifact_id if unsafe_entry == "run" else original
        assert (preserved / "payload").read_bytes() == b"payload 0"
    elif unsafe_entry == "payload":
        assert (directory / "manifest.json").is_file()
    else:
        assert (directory / "unexpected.txt").read_text() == "preserve unexpected bytes"
        assert (directory / "payload").read_bytes() == b"payload 0"


async def test_cancelled_cleanup_drains_before_publication_can_continue(
    artifact_agent, monkeypatch
):
    agent, result, _ = artifact_agent
    ref = result.artifacts[0]
    entered, release = threading.Event(), threading.Event()
    original = artifact_store._remove_artifact_directory

    def block_cleanup(path):
        entered.set()
        assert release.wait(5)
        original(path)

    monkeypatch.setattr(artifact_store, "_remove_artifact_directory", block_cleanup)
    deletion = asyncio.create_task(agent.delete_artifact(ref.artifact_id))
    pending = []
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        deletion.cancel()
        pending = [
            asyncio.create_task(call(ref.artifact_id))
            for call in (agent.read_artifact, agent.save_artifact)
        ]
        await asyncio.sleep(0)
        assert not deletion.done()
        assert not any(task.done() for task in pending)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await deletion
    for task in pending:
        with pytest.raises(ArtifactError) as error:
            await task
        assert error.value.code == "artifact_missing"
    assert await agent.delete_artifact(ref.artifact_id) is False


@pytest.mark.parametrize("model_delivery", [False, True])
async def test_delete_waits_for_inflight_public_or_model_delivery(
    artifact_agent, monkeypatch, model_delivery
):
    agent, result, _ = artifact_agent
    ref = result.artifacts[0]
    entered, release = threading.Event(), threading.Event()
    original = artifact_delivery.os.link

    def block_publish(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(artifact_delivery.os, "link", block_publish)
    delivery = agent._embedded._require_artifact_delivery()
    save = asyncio.create_task(
        delivery.save_committed(
            run_id=ref.run_id,
            artifact_id=ref.artifact_id,
            mode="create_new",
            destination_id="default",
        )
        if model_delivery
        else agent.save_artifact(ref.artifact_id)
    )
    deletion = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        deletion = asyncio.create_task(agent.delete_artifact(ref.artifact_id))
        await asyncio.sleep(0)
        assert not deletion.done()
        assert (
            await agent._embedded._store.get_artifact_record(ref.artifact_id)
        ).state is ArtifactState.READY
    finally:
        release.set()
    receipt = await save
    assert deletion is not None and await deletion
    assert Path(receipt.saved_path).read_bytes() == b"payload 0"
    with pytest.raises(ArtifactError) as error:
        await delivery.save_committed(
            run_id=ref.run_id,
            artifact_id=ref.artifact_id,
            mode="create_new",
            destination_id="default",
        )
    assert error.value.code == "artifact_missing"


async def test_deletion_cannot_mark_forged_or_foreign_refs_or_reuse_ids(
    artifact_agent, tmp_path
):
    agent, result, _ = artifact_agent
    ref = result.artifacts[0]
    state = agent._embedded._store
    with pytest.raises(ArtifactError):
        await state.transition_artifact(
            replace(
                (await state.get_artifact_record(ref.artifact_id)),
                ref=replace(ref, filename="forged.txt"),
            ),
            ArtifactState.DELETING,
        )
    assert (
        await state.get_artifact_record(ref.artifact_id)
    ).state is ArtifactState.READY
    other = await Agent.create(
        "other", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert await other.delete_artifact(ref.artifact_id) is False
    finally:
        await other.close()
    assert await agent.delete_artifact(ref.artifact_id)
    assert await state.get_artifact_record(ref.artifact_id) is None
    store = agent._embedded._artifact_store
    assert await store.recover_reserved(ref.run_id, ref.artifact_id) is None
    # A persisted reservation or a colliding ID factory cannot resurrect it.
    with pytest.raises(ArtifactError, match="reservation is no longer active"):
        await store.commit(
            ArtifactDraft(
                content=b"new bytes",
                suggested_filename=ref.filename,
                media_type=ref.media_type,
                sensitivity=ref.sensitivity,
                provenance=ref.provenance,
            ),
            ArtifactPolicy(
                allowed_media_types=frozenset({ref.media_type}),
                allowed_authorships=frozenset({ref.provenance.authorship}),
                allowed_extensions=((ref.media_type, (".txt",)),),
                artifact_required=True,
                max_artifact_count=1,
                max_bytes_per_artifact=1024,
                max_total_bytes_per_call=1024,
            ),
            run_id=ref.run_id,
            conversation_id=ref.conversation_id,
            call_id=ref.call_id,
            capability_id=ref.capability_id,
            reserved_artifact_id=ref.artifact_id,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_id", "another-agent"),
        ("ref", {}),
        ("caller_principal_id", ""),
        ("state", "invalid"),
        ("agent_id", ""),
        ("unexpected", True),
    ],
)
async def test_malformed_registry_records_fail_home_admission(
    artifact_agent, tmp_path, field, value
):
    agent, result, _ = artifact_agent
    ref = result.artifacts[0]
    await agent.close()
    path = tmp_path / "agents/deletion/state.db"
    with sqlite3.connect(path) as connection:
        record = json.loads(
            connection.execute(
                "SELECT data FROM artifacts WHERE artifact_id = ?", (ref.artifact_id,)
            ).fetchone()[0]
        )
        record["fields"][field] = value
        connection.execute(
            "UPDATE artifacts SET data = ? WHERE artifact_id = ?",
            (json.dumps(record), ref.artifact_id),
        )
    with pytest.raises(StateCompatibilityError):
        await Agent.open("deletion", root=tmp_path, workspace=workspace_for(tmp_path))


async def test_active_job_reservation_prevents_deletion_until_result_is_accepted(
    tmp_path, monkeypatch
):
    home = await create_probe_home(tmp_path, "active-artifact")
    provider = MockModelProvider(
        (
            toolbox_load_response("start_data_profile"),
            start_profile_response(home.resource_ids[TARGET_PROFILE_TABLE]),
            stop_response(),
        )
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=provider.model_profile,
        limits=OFFLINE_EAGER_LIMITS,
        workspace=workspace_for(home.root),
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    observed: list[ArtifactRef] = []
    supervisor = agent._embedded._job_supervisor
    promote = supervisor._promote_graph_artifact

    async def pause_promotion(inspection, task, attempt, ref, content):
        if not observed:
            observed.append(ref)
            entered.set()
            await release.wait()
        return await promote(inspection, task, attempt, ref, content)

    monkeypatch.setattr(supervisor, "_promote_graph_artifact", pause_promotion)
    try:
        await agent.run("Profile this table.", source_scope_ids=(home.source_id,))
        await asyncio.wait_for(entered.wait(), timeout=10)
        (ref,) = observed
        with pytest.raises(ArtifactError) as busy:
            await agent.delete_artifact(ref.artifact_id)
        assert busy.value.code == "artifact_busy"
        record = await agent._embedded._store.get_artifact_record(ref.artifact_id)
        assert record is not None and record.state is ArtifactState.READY
        assert (await agent.read_artifact(ref.artifact_id)).content
        release.set()
        (job,) = await agent.list_jobs()
        await wait_for_terminal(agent, job.job_id)
        assert await agent.delete_artifact(ref.artifact_id)
        assert await agent._embedded._store.get_artifact_record(ref.artifact_id) is None
    finally:
        release.set()
        await agent.close()


@pytest.mark.parametrize("clear_history", [False, True])
@pytest.mark.parametrize("source_revision", [2, "pre_registry_3", 3])
async def test_graph_and_delivery_artifact_deletion_survives_reopen(
    tmp_path, clear_history, source_revision
):
    home = await create_probe_home(tmp_path, "graph-deletion")
    provider = MockModelProvider(
        (
            toolbox_load_response("start_data_profile"),
            start_profile_response(home.resource_ids[TARGET_PROFILE_TABLE]),
            stop_response(),
        )
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=provider.model_profile,
        limits=OFFLINE_EAGER_LIMITS,
        workspace=workspace_for(home.root),
    )
    try:
        await agent.run("Profile this table.", source_scope_ids=(home.source_id,))
        (job,) = await agent.list_jobs()
        await wait_for_terminal(agent, job.job_id)
        result = await agent.read_job_result(job.job_id)
        assert result is not None
        (artifact_id,) = result.artifact_ids
        task_artifacts = await agent.list_task_artifacts(job.job_id)
        assert artifact_id in task_artifacts
        if clear_history:
            await agent.clear_conversations()
        if source_revision != 3:
            await agent.close()
            with sqlite3.connect(agent.home / "state.db") as connection:
                connection.execute("DROP TABLE artifacts")
                if source_revision == 2:
                    connection.execute(
                        "DELETE FROM agent_home_migrations WHERE revision = 3"
                    )
                else:
                    connection.execute(
                        "UPDATE agent_home_migrations SET checksum = ? WHERE revision = 3",
                        (PRE_REGISTRY_REVISION_3_CHECKSUM,),
                    )
            agent = await Agent.open(
                home.name, root=home.root, workspace=workspace_for(home.root)
            )
            assert (await agent.read_artifact(artifact_id)).content
        assert await agent.delete_artifact(artifact_id)
        assert await agent.list_task_artifacts(job.job_id) == tuple(
            item for item in task_artifacts if item != artifact_id
        )
        assert await agent._embedded._store.get_artifact_record(artifact_id) is None
        assert await agent.read_job_result(job.job_id) == result
    finally:
        await agent.close()
    reopened = await Agent.open(
        home.name, root=home.root, workspace=workspace_for(home.root)
    )
    try:
        assert await reopened.read_job_result(job.job_id) == result
        assert await reopened.delete_artifact(artifact_id) is False
        with pytest.raises(ArtifactError) as error:
            await reopened.read_artifact(artifact_id)
        assert error.value.code == "artifact_missing"
    finally:
        await reopened.close()


@pytest.mark.parametrize("published", [False, True])
async def test_startup_reconciles_creation_before_or_after_file_publication(
    artifact_agent, tmp_path, published
):
    """A crash leaves a creating row; only a verified final directory survives."""
    import shutil

    from daita.storage.sqlite_codecs.artifacts import encode_artifact_record

    agent, result, _ = artifact_agent
    ref = result.artifacts[0]
    record = await agent._embedded._store.get_artifact_record(ref.artifact_id)
    assert record is not None
    directory = agent.home / "artifacts" / ref.run_id / ref.artifact_id
    await agent.close()
    with sqlite3.connect(agent.home / "state.db") as connection:
        connection.execute(
            "UPDATE artifacts SET state = 'creating', data = ? WHERE artifact_id = ?",
            (
                encode_artifact_record(replace(record, state=ArtifactState.CREATING)),
                ref.artifact_id,
            ),
        )
    if not published:
        shutil.rmtree(directory)
    reopened = await Agent.open(
        "deletion", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        recovered = await reopened._embedded._store.get_artifact_record(ref.artifact_id)
        if published:
            assert recovered is not None and recovered.state is ArtifactState.READY
            assert (
                await reopened.read_artifact(ref.artifact_id)
            ).content == b"payload 0"
        else:
            assert recovered is None
            with pytest.raises(ArtifactError):
                await reopened.read_artifact(ref.artifact_id)
        assert (
            await reopened.read_artifact(result.artifacts[1].artifact_id)
        ).content == b"payload 1"
    finally:
        await reopened.close()


async def test_startup_finishes_deletion_after_files_removed_before_row_removed(
    artifact_agent, tmp_path, monkeypatch
):
    agent, result, _ = artifact_agent
    ref = result.artifacts[0]

    async def fail_finalization(record):
        raise OSError("injected database finalization failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            agent._embedded._store, "finish_artifact_deletion", fail_finalization
        )
        with pytest.raises(ArtifactError) as failure:
            await agent.delete_artifact(ref.artifact_id)
    assert failure.value.details["stage"] == "delete_cleanup"
    assert not (agent.home / "artifacts" / ref.run_id / ref.artifact_id).exists()
    pending = await agent._embedded._store.get_artifact_record(ref.artifact_id)
    assert pending is not None and pending.state is ArtifactState.DELETING
    await agent.close()
    reopened = await Agent.open(
        "deletion", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert (
            await reopened._embedded._store.get_artifact_record(ref.artifact_id) is None
        )
        assert await reopened.delete_artifact(ref.artifact_id) is False
    finally:
        await reopened.close()


async def test_pending_deletion_at_payload_quota_can_recover(
    artifact_agent, tmp_path, monkeypatch
):
    agent, result, _ = artifact_agent
    ref = result.artifacts[0]
    record = await agent._embedded._store.get_artifact_record(ref.artifact_id)
    assert record is not None
    await agent._embedded._store.transition_artifact(record, ArtifactState.DELETING)
    await agent.close()
    byte_limit = sum(item.byte_size for item in result.artifacts)
    monkeypatch.setattr(artifact_store, "MAX_ARTIFACT_BYTES_PER_RUN", byte_limit)
    monkeypatch.setattr(artifact_store, "MAX_ARTIFACT_BYTES_PER_AGENT", byte_limit)
    reopened = await Agent.open(
        "deletion", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert (
            await reopened._embedded._store.get_artifact_record(ref.artifact_id) is None
        )
        assert (
            await reopened.read_artifact(result.artifacts[1].artifact_id)
        ).content == b"payload 1"
    finally:
        await reopened.close()


async def test_registry_reads_and_lists_do_not_decode_historical_results(
    artifact_agent, monkeypatch
):
    import daita.storage.sqlite as sqlite_store

    agent, result, _ = artifact_agent

    def reject_history(*args, **kwargs):
        raise AssertionError("live artifact queries must not decode history")

    monkeypatch.setattr(sqlite_store, "decode_message", reject_history)
    monkeypatch.setattr(
        sqlite_store, "decode_task_result", reject_history, raising=False
    )
    assert await agent._embedded._artifact_store.list_refs() == result.artifacts
    assert (
        await agent.read_artifact(result.artifacts[0].artifact_id)
    ).content == b"payload 0"
    assert await agent.delete_artifact(result.artifacts[0].artifact_id)
    assert await agent._embedded._artifact_store.list_refs() == (result.artifacts[1],)


@pytest.mark.parametrize("source_revision", [2, "pre_registry_3"])
@pytest.mark.parametrize("phase", [None, "prepared", "committing", "committed"])
async def test_prior_artifact_roots_are_backfilled_once_without_changing_bytes(
    artifact_agent, tmp_path, source_revision, phase
):
    agent, result, _ = artifact_agent
    home = agent.home
    await agent.close()
    with sqlite3.connect(home / "state.db") as connection:
        connection.execute("DROP TABLE artifacts")
        if source_revision == 2:
            connection.execute("DELETE FROM agent_home_migrations WHERE revision = 3")
        else:
            connection.execute(
                "UPDATE agent_home_migrations SET checksum = ? WHERE revision = 3",
                (PRE_REGISTRY_REVISION_3_CHECKSUM,),
            )
        for run_id, encoded in tuple(connection.execute("SELECT id, input FROM runs")):
            payload = json.loads(encoded)
            if source_revision == 2:
                payload["fields"].pop("caller_principal_id")
                payload["fields"].pop("caller_principal_verified")
            connection.execute(
                "UPDATE runs SET input = ? WHERE id = ?", (json.dumps(payload), run_id)
            )
    before = {
        ref.artifact_id: (
            home / "artifacts" / ref.run_id / ref.artifact_id / "payload"
        ).read_bytes()
        for ref in result.artifacts
    }
    if phase is not None:

        def crash(observed):
            if observed == phase:
                raise RuntimeError(f"injected interruption at {phase}")

        with pytest.raises(StateCompatibilityError):
            home_upgrade.upgrade_agent_home(
                home,
                validate_home=lambda source, candidate, affected: _validate_agent_home_target(
                    agent._embedded.identity, source, candidate, affected
                ),
                phase_hook=crash,
            )
        assert (
            json.loads((home / ".home-upgrade/journal.json").read_text())["phase"]
            == phase
        )
        assert (await Agent.inspect_home("deletion", root=tmp_path)).recovery_required
    reopened = await Agent.open(
        "deletion", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert await reopened._embedded._artifact_store.list_refs() == result.artifacts
        for ref in result.artifacts:
            assert (await reopened.read_artifact(ref.artifact_id)).content == before[
                ref.artifact_id
            ]
        assert await reopened.delete_artifact(result.artifacts[0].artifact_id)
    finally:
        await reopened.close()


async def test_old_development_checksum_cannot_admit_an_unexpected_schema(
    artifact_agent, tmp_path
):
    agent, result, _ = artifact_agent
    home = agent.home
    await agent.close()
    with sqlite3.connect(home / "state.db") as connection:
        connection.execute(
            "UPDATE agent_home_migrations SET checksum = ? WHERE revision = 3",
            (PRE_REGISTRY_REVISION_3_CHECKSUM,),
        )
    before = (home / "state.db").read_bytes()
    with pytest.raises(StateCompatibilityError):
        await Agent.open("deletion", root=tmp_path, workspace=workspace_for(tmp_path))
    assert (home / "state.db").read_bytes() == before
    assert not (home / ".home-upgrade").exists()
    for ref in result.artifacts:
        assert (home / "artifacts" / ref.run_id / ref.artifact_id / "payload").is_file()
