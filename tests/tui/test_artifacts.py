"""Component-owned tests split from ``test_public_surfaces.py``."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button, OptionList, Static

from daita import Agent
from daita.artifacts import ArtifactError, ArtifactPayload, ArtifactRef
from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactProvenance,
    ArtifactResourceBinding,
)
from daita.artifacts.renderers import (
    XLSX_MEDIA_TYPE,
    ExactXlsxProvenance,
    render_exact_xlsx,
)
from daita.tui.app import DaitaApp
from daita.tui.screens.artifacts import (
    MAX_PREVIEW_BYTES,
    ArtifactsScreen,
    render_artifact_preview,
)
from daita.tui.screens.confirm import ConfirmScreen
from tests.artifacts._public_surface_support import (
    CanonicalMessage,
    MessageRole,
    Path,
    RunInput,
    ToolCall,
    ToolResultBlock,
    Transcript,
    _create_artifact_agent,
    _surface_records,
    artifact_delivery_messages,
    delivery_module,
)
from tests.support.workspace import workspace_for


async def _wait_for_actions(manager: ArtifactsScreen) -> None:
    workers = tuple(worker for worker in manager.workers if worker.node is manager)
    if workers:
        await asyncio.wait_for(manager.workers.wait_for_complete(workers), timeout=5)


def test_terminal_renders_authoritative_saved_path_and_truthful_delivery_failure() -> (
    None
):
    ref, receipt, result = _surface_records()
    assert any(
        receipt.filename in getattr(item, "filename", "")
        or getattr(item, "saved_path", "") == receipt.saved_path
        for item in result.artifact_deliveries
    )
    run = RunInput(
        id=ref.run_id,
        agent_id="agent-one",
        message="save a file",
        created_at=ref.created_at,
        conversation_id=ref.conversation_id,
    )
    failed = Transcript(
        run=run,
        messages=(
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="save",
                        name="artifact_save_local",
                        arguments={
                            "artifact_id": ref.artifact_id,
                            "mode": "create_new",
                            "destination_id": "default",
                        },
                    ),
                ),
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id="save",
                        is_error=True,
                        output={
                            "error": {
                                "code": "artifact_downloads_unavailable",
                                "message": "Downloads is unavailable.",
                            }
                        },
                    ),
                ),
            ),
        ),
    )
    failed_messages = artifact_delivery_messages(failed.tool_pairs)
    assert any(
        "remains available; local delivery failed" in text for text in failed_messages
    )

    uncertain_edit = Transcript(
        run=run,
        messages=(
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="replace",
                        name="artifact_save_local",
                        arguments={
                            "artifact_id": ref.artifact_id,
                            "mode": "replace_bound_file",
                        },
                    ),
                ),
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id="replace",
                        output={
                            "kind": "artifact.delivery_receipt",
                            "data": {
                                "artifact_id": ref.artifact_id,
                                "mode": "replace_bound_file",
                                "outcome": "uncertain",
                                "relative_path": "config.yaml",
                            },
                        },
                    ),
                ),
            ),
        ),
    )
    uncertain_messages = artifact_delivery_messages(uncertain_edit.tool_pairs)
    assert any(
        "update outcome for local file config.yaml is uncertain" in text
        for text in uncertain_messages
    )

    not_delivered = Transcript(
        run=run,
        messages=(
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="create",
                        name="artifact_create_document",
                        arguments={"format": "txt", "content": "report"},
                    ),
                ),
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id="create",
                        output={
                            "kind": "artifact.document",
                            "data": {"format": "txt", "character_count": 6},
                            "artifact": {"artifact_id": ref.artifact_id},
                            "delivery_status": "not_delivered",
                        },
                    ),
                ),
            ),
        ),
    )
    not_delivered_messages = artifact_delivery_messages(not_delivered.tool_pairs)
    assert any(
        "was created internally but was not saved locally" in text
        for text in not_delivered_messages
    )


@pytest.mark.integration
async def test_artifact_command_manages_registered_files_after_history_clear(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    agent, ref = await _create_artifact_agent(tmp_path, "tui-artifacts", downloads)
    await agent.clear_conversations()
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = agent
    try:
        invalid = await app.controller.dispatch_command("/artifacts delete")
        assert invalid.kind == "notice"
        assert "Usage: /artifacts" in invalid.message
        async with app.run_test(size=(80, 24)) as pilot:
            await app._show_chat()
            await pilot.press(
                "/", "a", "r", "t", "i", "f", "a", "c", "t", "s", "enter", "enter"
            )
            await pilot.pause()
            assert isinstance(app.screen, ArtifactsScreen)
            manager = app.screen
            await _wait_for_actions(manager)
            await pilot.pause()
            assert manager.query_one("#artifacts-list", OptionList).option_count == 1
            assert ref.artifact_id in str(
                manager.query_one("#artifacts-detail", Static).content
            )
            assert (
                manager.query_one("#artifacts-navigation").region.bottom
                <= manager.query_one("#artifacts-manager").region.bottom
            )
            assert (
                manager.query_one("#artifacts-close", Button).region.right
                <= manager.query_one("#artifacts-navigation").region.right
            )
            await pilot.press("v")
            await _wait_for_actions(manager)
            await pilot.pause()
            assert "surface payload" in str(
                manager.query_one("#artifacts-detail", Static).content
            )

            await pilot.press("s")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await _wait_for_actions(manager)
            await pilot.pause()
            assert (downloads / "result.txt").read_bytes() == b"surface payload\n"
            assert str(downloads / "result.txt") in str(
                manager.query_one("#artifacts-notice", Static).content
            )

            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("escape")
            await _wait_for_actions(manager)
            assert await agent.list_artifacts() == (ref,)
            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await _wait_for_actions(manager)
            await pilot.pause()
            assert manager.query_one("#artifacts-list", OptionList).option_count == 0
            assert manager.query_one("#artifacts-delete", Button).disabled
            assert "Artifact deleted" in str(
                manager.query_one("#artifacts-notice", Static).content
            )
            assert await agent.list_artifacts() == ()
            assert not (
                agent.home / "artifacts" / ref.run_id / ref.artifact_id
            ).exists()
            assert (downloads / "result.txt").exists()
            await pilot.press("escape")
            assert app.screen is not manager
            app.exit(0)
    finally:
        await agent.close()
    reopened = await Agent.open(
        "tui-artifacts", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        assert await reopened.list_artifacts() == ()
    finally:
        await reopened.close()


async def test_artifact_manager_browses_all_pages_and_returns_from_deleted_last_page(
    monkeypatch,
):
    ref, _, _ = _surface_records()
    items = tuple(
        replace(ref, artifact_id=f"artifact-{index:032x}") for index in range(51)
    )
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    async def listing(*, limit, offset):
        return items[offset : offset + limit]

    async def delete(artifact_id):
        nonlocal items
        items = tuple(item for item in items if item.artifact_id != artifact_id)
        return True

    monkeypatch.setattr(app.controller, "list_artifacts", listing)
    monkeypatch.setattr(app.controller, "delete_artifact", delete)
    async with app.run_test(size=(100, 35)) as pilot:
        manager = ArtifactsScreen()
        await app.push_screen(manager)
        await _wait_for_actions(manager)
        await pilot.pause()
        assert manager.query_one("#artifacts-list", OptionList).option_count == 50
        assert manager.query_one("#artifacts-previous", Button).disabled
        await pilot.press("right")
        await _wait_for_actions(manager)
        await pilot.pause()
        assert manager.query_one("#artifacts-list", OptionList).option_count == 1
        assert manager.query_one("#artifacts-next", Button).disabled
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        await _wait_for_actions(manager)
        await pilot.pause()
        assert manager.query_one("#artifacts-list", OptionList).option_count == 50
        assert manager.query_one("#artifacts-previous", Button).disabled
        assert manager.query_one("#artifacts-next", Button).disabled
        app.exit(0)


@pytest.mark.parametrize("code", ["artifact_busy", "artifact_storage_failed"])
async def test_artifact_manager_shows_deletion_failures_and_reloads_inventory(
    monkeypatch, code
):
    ref, _, _ = _surface_records()
    items: tuple[ArtifactRef, ...] = (ref,)
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    async def listing(*, limit, offset):
        return items[offset : offset + limit]

    async def delete(artifact_id):
        nonlocal items
        if code == "artifact_storage_failed":
            items = ()
        raise ArtifactError(
            code,
            "Cannot finish deletion.",
            {"stage": "delete_cleanup"} if code == "artifact_storage_failed" else {},
        )

    monkeypatch.setattr(app.controller, "list_artifacts", listing)
    monkeypatch.setattr(app.controller, "delete_artifact", delete)
    async with app.run_test(size=(100, 35)) as pilot:
        manager = ArtifactsScreen()
        await app.push_screen(manager)
        await _wait_for_actions(manager)
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("y")
        await _wait_for_actions(manager)
        await pilot.pause()
        error = str(manager.query_one("#artifacts-error", Static).content)
        assert code in error
        assert "Artifact deleted" not in str(
            manager.query_one("#artifacts-notice", Static).content
        )
        assert manager.query_one("#artifacts-list", OptionList).option_count == len(
            items
        )
        assert not manager.query_one("#artifacts-close", Button).disabled
        app.exit(0)


def test_artifact_preview_bounds_text_and_preserves_literal_markup():
    ref, _, _ = _surface_records()
    content = ("[bold]literal[/]\x1b[31m\n" + "é" * MAX_PREVIEW_BYTES).encode("utf-8")
    payload = ArtifactPayload(replace(ref, byte_size=len(content)), content)
    preview = render_artifact_preview(payload)
    assert "[bold]literal[/]" in preview
    assert "\x1b" not in preview
    assert "Preview truncated" in preview
    assert len(preview) < MAX_PREVIEW_BYTES + 100


def test_artifact_preview_reads_verified_xlsx_rows_and_marks_truncation():
    ref, _, _ = _surface_records()
    content = render_exact_xlsx(
        ("value",),
        ((f"row-{index}",) for index in range(21)),
        provenance=ExactXlsxProvenance(
            source_id="source-test",
            source_revision="schema:1",
            resource_revisions=(("resource-test", "sha256:" + "1" * 64),),
            sql_fingerprint="sha256:" + "2" * 64,
            parameters_sha256="sha256:" + "3" * 64,
            sensitivity=ref.sensitivity,
            created_at=ref.created_at,
        ),
    )
    ref = replace(
        ref,
        media_type=XLSX_MEDIA_TYPE,
        filename="table.xlsx",
        byte_size=len(content),
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.EXACT_SOURCE_DATA,
            resource_bindings=(
                ArtifactResourceBinding(
                    "source-test", "schema:1", "resource-test", "sha256:" + "1" * 64
                ),
            ),
            sql_fingerprint="sha256:" + "2" * 64,
            parameters_sha256="sha256:" + "3" * 64,
            columns=("value",),
            row_count=21,
        ),
    )
    preview = render_artifact_preview(ArtifactPayload(ref, content))
    assert "value\nrow-0" in preview
    assert "row-19" in preview
    assert "row-20" not in preview
    assert "Preview truncated" in preview


def test_open_reveal_and_folder_picker_are_user_actions_not_model_or_shell_tools() -> (
    None
):
    from daita.domains.data.export_capabilities import artifact_capability_declarations

    tool_names = {item.name for item in artifact_capability_declarations().tool_views}
    assert tool_names.isdisjoint(
        {"artifact_open", "artifact_reveal", "artifact_pick_folder"}
    )
    assert delivery_module.__file__ is not None
    delivery_source = Path(delivery_module.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in delivery_source
    assert "shell=True" not in delivery_source
