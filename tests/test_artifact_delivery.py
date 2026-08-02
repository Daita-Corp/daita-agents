from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
import sqlite3
import threading

import pytest

from daita import Agent, ArtifactError, LocalDirectorySource
import daita.artifacts.delivery as delivery_module
from daita.artifacts.models import (
    DestinationAuthorization,
    DestinationAvailability,
    SYSTEM_DOWNLOADS_DESTINATION_ID,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider


def _ids():
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        if prefix in {"run", "conversation", "artifact", "destination"}:
            return f"{prefix}-{counts[prefix]:032x}"
        return f"{prefix}-{counts[prefix]}"

    return create


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _document_script(*, final: str = "created") -> tuple[ModelResponse, ...]:
    return (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="create",
                    name="artifact_create_document",
                    arguments={
                        "format": "txt",
                        "filename": "notes.txt",
                        "content": "artifact delivery payload\n",
                    },
                ),
            ),
        ),
        ModelResponse(finish_reason=FinishReason.STOP, text=final),
    )


async def _agent_with_artifact(
    tmp_path: Path,
    name: str,
    downloads: Path,
    *,
    provider: MockModelProvider | None = None,
) -> tuple[Agent, str, MockModelProvider]:
    selected = provider or MockModelProvider(
        _document_script(), provider_id=f"mock:{name}"
    )
    agent = await Agent.create(
        name,
        root=tmp_path,
        model=selected,
        model_profile=_profile(selected),
        id_factory=_ids(),
        downloads_directory=downloads,
    )
    result = await agent.run("Create a TXT file.")
    return agent, result.artifacts[0].artifact_id, selected


async def test_injected_host_downloads_is_default_without_hardcoded_home_or_shell(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "localized-folder"
    downloads.mkdir()
    agent, _, _ = await _agent_with_artifact(tmp_path, "downloads-default", downloads)
    try:
        destination = await agent.export_destination()
        assert destination.destination_id == SYSTEM_DOWNLOADS_DESTINATION_ID
        assert destination.authorization is DestinationAuthorization.SYSTEM
        assert destination.availability is DestinationAvailability.AVAILABLE
        source = Path(delivery_module.__file__).read_text(encoding="utf-8")
        assert "subprocess" not in source
        assert "~/Downloads" not in source
    finally:
        await agent.close()


async def test_default_selector_prefers_explicit_persistent_destination_then_system_downloads(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    persistent = tmp_path / "persistent"
    downloads.mkdir()
    persistent.mkdir()
    agent, artifact_id, _ = await _agent_with_artifact(
        tmp_path, "precedence", downloads
    )
    try:
        selected = await agent.set_export_destination(persistent)
        assert selected.is_default
        receipt = await agent.save_artifact(artifact_id)
        assert Path(receipt.saved_path).parent == persistent
        reset = await agent.reset_export_destination()
        assert reset.destination_id == SYSTEM_DOWNLOADS_DESTINATION_ID
        receipt = await agent.save_artifact(artifact_id)
        assert Path(receipt.saved_path).parent == downloads
    finally:
        await agent.close()


async def test_unavailable_explicit_destination_does_not_fall_back_to_downloads(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    persistent = tmp_path / "persistent"
    downloads.mkdir()
    persistent.mkdir()
    agent, artifact_id, _ = await _agent_with_artifact(
        tmp_path, "no-security-fallback", downloads
    )
    try:
        await agent.set_export_destination(persistent)
        persistent.rmdir()
        with pytest.raises(ArtifactError) as failure:
            await agent.save_artifact(artifact_id)
        assert failure.value.code == "artifact_destination_unavailable"
        assert not tuple(downloads.iterdir())
    finally:
        await agent.close()


async def test_authorized_one_time_destination_changes_one_copy_and_not_the_default(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    one_time = tmp_path / "one-time"
    downloads.mkdir()
    one_time.mkdir()
    agent, artifact_id, _ = await _agent_with_artifact(tmp_path, "one-time", downloads)
    try:
        first = await agent.save_artifact(artifact_id, one_time)
        assert Path(first.saved_path).parent == one_time
        assert (await agent.export_destination()).destination_id == (
            SYSTEM_DOWNLOADS_DESTINATION_ID
        )
        second = await agent.save_artifact(artifact_id)
        assert Path(second.saved_path).parent == downloads
    finally:
        await agent.close()


async def test_persistent_destination_survives_restart_and_conversation_clear(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    persistent = tmp_path / "persistent"
    downloads.mkdir()
    persistent.mkdir()
    agent, _, _ = await _agent_with_artifact(tmp_path, "persistent-restart", downloads)
    try:
        selected = await agent.set_export_destination(persistent)
        await agent.clear_conversations()
    finally:
        await agent.close()
    reopened = await Agent.open(
        "persistent-restart", root=tmp_path, downloads_directory=downloads
    )
    try:
        assert await reopened.export_destination() == selected
    finally:
        await reopened.close()


async def test_destination_config_is_operational_state_not_memory_skill_or_model_context(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    persistent = tmp_path / "private-export-root"
    downloads.mkdir()
    persistent.mkdir()
    follow = ModelResponse(finish_reason=FinishReason.STOP, text="follow-up")
    provider = MockModelProvider(
        (*_document_script(), follow), provider_id="mock:operational-destination"
    )
    agent, _, _ = await _agent_with_artifact(
        tmp_path,
        "operational-destination",
        downloads,
        provider=provider,
    )
    try:
        await agent.set_export_destination(persistent)
        await agent.run("Analyze this without creating a file.")
        config = agent.home / "artifacts" / "delivery-config.json"
        assert str(persistent) in config.read_text(encoding="utf-8")
        with sqlite3.connect(agent.home / "state.db") as connection:
            database_text = "\n".join(
                str(value)
                for table in ("messages", "metadata")
                for row in connection.execute(f"SELECT * FROM {table}")
                for value in row
            )
        assert str(persistent) not in database_text
        request_text = "\n".join(
            block.text
            for message in provider.requests[-1].messages
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert str(persistent) not in request_text
    finally:
        await agent.close()


async def test_destination_authorization_rejects_unknown_raw_path_source_root_agent_home_and_symlink_swap(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    source_root = tmp_path / "source"
    persistent = tmp_path / "persistent"
    replacement = tmp_path / "replacement"
    for directory in (downloads, source_root, persistent, replacement):
        directory.mkdir()
    (source_root / "rows.csv").write_text("id\n1\n", encoding="utf-8")
    agent, artifact_id, _ = await _agent_with_artifact(
        tmp_path, "destination-authorization", downloads
    )
    try:
        await agent.attach(LocalDirectorySource(source_root))
        for directory in (agent.home, source_root):
            with pytest.raises(ArtifactError) as failure:
                await agent.save_artifact(artifact_id, directory)
            assert failure.value.code == "artifact_destination_unauthorized"
        selected = await agent.set_export_destination(persistent)
        moved = tmp_path / "moved-persistent"
        persistent.rename(moved)
        persistent.symlink_to(replacement, target_is_directory=True)
        with pytest.raises(ArtifactError) as swapped:
            await agent.save_artifact(artifact_id)
        assert swapped.value.code == "artifact_destination_revoked"
        assert swapped.value.details["destination_id"] == selected.destination_id
    finally:
        await agent.close()


async def test_revoked_unavailable_and_downloads_unavailable_errors_preserve_internal_artifact(
    tmp_path: Path,
) -> None:
    unavailable_downloads = tmp_path / "missing-downloads"
    provider = MockModelProvider(
        _document_script(), provider_id="mock:unavailable-downloads"
    )
    agent = await Agent.create(
        "unavailable-downloads",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=unavailable_downloads,
    )
    try:
        result = await agent.run("Create a TXT file.")
        artifact_id = result.artifacts[0].artifact_id
        with pytest.raises(ArtifactError) as failure:
            await agent.save_artifact(artifact_id)
        assert failure.value.code == "artifact_downloads_unavailable"
        assert failure.value.details["artifact_id"] == artifact_id
        assert failure.value.details["artifact_retained"] is True
        assert (await agent.read_artifact(artifact_id)).content
    finally:
        await agent.close()


async def test_delivery_writes_verifies_and_atomically_publishes_without_overwrite(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "notes.txt").write_text("existing", encoding="utf-8")
    agent, artifact_id, _ = await _agent_with_artifact(
        tmp_path, "atomic-copy", downloads
    )
    try:
        receipt = await agent.save_artifact(artifact_id)
        assert (downloads / "notes.txt").read_text(encoding="utf-8") == "existing"
        assert receipt.filename == "notes (1).txt"
        assert (
            Path(receipt.saved_path).read_bytes()
            == (await agent.read_artifact(artifact_id)).content
        )
        assert not tuple(downloads.glob(".daita-artifact-*.tmp"))
    finally:
        await agent.close()


async def test_delivery_failure_and_cancellation_before_publish_remove_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    agent, artifact_id, _ = await _agent_with_artifact(
        tmp_path, "delivery-failure-cleanup", downloads
    )
    try:

        def fail_link(*args, **kwargs):
            del args, kwargs
            raise OSError("atomic publication unavailable")

        monkeypatch.setattr(delivery_module.os, "link", fail_link)
        with pytest.raises(ArtifactError) as failure:
            await agent.save_artifact(artifact_id)
        assert failure.value.code == "artifact_delivery_failed"
        assert not tuple(downloads.iterdir())
    finally:
        await agent.close()


async def test_delivery_late_cancellation_may_leave_only_a_complete_verified_user_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    agent, artifact_id, _ = await _agent_with_artifact(
        tmp_path, "delivery-late-cancel", downloads
    )
    published = threading.Event()
    original_link = delivery_module.os.link

    def observed_link(*args, **kwargs):
        result = original_link(*args, **kwargs)
        published.set()
        return result

    monkeypatch.setattr(delivery_module.os, "link", observed_link)
    try:
        saving = asyncio.create_task(agent.save_artifact(artifact_id))
        await asyncio.to_thread(published.wait, 2)
        saving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await saving
        copies = tuple(downloads.iterdir())
        assert len(copies) == 1
        assert (
            copies[0].read_bytes() == (await agent.read_artifact(artifact_id)).content
        )
        assert not tuple(downloads.glob(".daita-artifact-*.tmp"))
    finally:
        await agent.close()


async def test_name_collision_uses_first_available_parenthesized_suffix(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for filename in ("notes.txt", "notes (1).txt", "notes (3).txt"):
        (downloads / filename).write_text("existing", encoding="utf-8")
    agent, artifact_id, _ = await _agent_with_artifact(tmp_path, "collision", downloads)
    try:
        receipt = await agent.save_artifact(artifact_id)
        assert receipt.filename == "notes (2).txt"
        assert receipt.renamed_for_collision
    finally:
        await agent.close()


async def test_collision_exhaustion_returns_structured_error_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for filename in ("notes.txt", "notes (1).txt", "notes (2).txt"):
        (downloads / filename).write_text("existing", encoding="utf-8")
    monkeypatch.setattr(delivery_module, "MAX_COLLISION_SUFFIX", 2)
    agent, artifact_id, _ = await _agent_with_artifact(
        tmp_path, "collision-exhausted", downloads
    )
    try:
        with pytest.raises(ArtifactError) as failure:
            await agent.save_artifact(artifact_id)
        assert failure.value.code == "artifact_name_collision"
        assert all(
            path.read_text(encoding="utf-8") == "existing"
            for path in downloads.iterdir()
        )
    finally:
        await agent.close()


async def test_delivery_rejects_internal_digest_change_before_external_copy(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    agent, artifact_id, _ = await _agent_with_artifact(
        tmp_path, "digest-before-copy", downloads
    )
    try:
        ref = (await agent.list_artifacts())[0]
        payload = agent.home / "artifacts" / ref.run_id / artifact_id / "payload"
        payload.write_bytes(b"x" * ref.byte_size)
        with pytest.raises(ArtifactError) as failure:
            await agent.save_artifact(artifact_id)
        assert failure.value.code == "artifact_corrupt"
        assert not tuple(downloads.iterdir())
    finally:
        await agent.close()


async def test_verified_saved_path_is_only_path_allowed_in_current_receipt_and_is_redacted_from_history(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    artifact_id = "artifact-00000000000000000000000000000001"
    provider = MockModelProvider(
        (
            _document_script()[0],
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="save",
                        name="artifact_save_local",
                        arguments={
                            "artifact_id": artifact_id,
                            "destination_id": "default",
                        },
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="saved"),
            ModelResponse(finish_reason=FinishReason.STOP, text="continued"),
        ),
        provider_id="mock:receipt-history",
    )
    agent = await Agent.create(
        "receipt-history",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
    )
    try:
        first = await agent.run("Create and save a TXT file.")
        receipt = first.artifact_deliveries[0]
        assert Path(receipt.saved_path).is_absolute()
        await agent.run("Continue.", conversation_id=first.conversation_id)
        historical = provider.requests[-1]
        text = str(historical)
        assert receipt.saved_path not in text
        assert artifact_id in text
        assert all(
            message.role is not MessageRole.SYSTEM or str(downloads) not in str(message)
            for message in historical.messages
        )
    finally:
        await agent.close()
