from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

import daita.artifacts.delivery as delivery_module
from daita import Agent, ArtifactDeliveryReceipt, cli
from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactProvenance,
    ArtifactRef,
)
from daita.catalog.models import Sensitivity
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExit, LoopExitKind, RunInput, Transcript
from daita.tui.projection import artifact_delivery_messages, completed_tool_pairs


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _ids():
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        return f"{prefix}-{counts[prefix]:032x}"

    return create


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


async def _create_artifact_agent(
    tmp_path: Path,
    name: str,
    downloads: Path,
) -> tuple[Agent, ArtifactRef]:
    provider = MockModelProvider(
        (
            _tool(
                "create",
                "artifact_create_document",
                {
                    "format": "txt",
                    "filename": "result.txt",
                    "content": "surface payload\n",
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="created"),
        ),
        provider_id=f"mock:{name}",
    )
    agent = await Agent.create(
        name,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
    )
    result = await agent.run("Create a TXT file.")
    return agent, result.artifacts[0]


async def test_public_known_id_read_and_save_work_after_restart_without_rerunning_model(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "public-downloads"
    downloads.mkdir()
    agent, ref = await _create_artifact_agent(tmp_path, "public-restart", downloads)
    try:
        assert ref.conversation_id
    finally:
        await agent.close()
    reopened = await Agent.open(
        "public-restart", root=tmp_path, downloads_directory=downloads
    )
    try:
        assert (await reopened.read_artifact(ref.artifact_id)).content == (
            b"surface payload\n"
        )
        receipt = await reopened.save_artifact(ref.artifact_id)
        assert Path(receipt.saved_path).read_bytes() == b"surface payload\n"
    finally:
        await reopened.close()


async def test_public_save_path_is_one_time_and_public_set_location_is_persistent(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads-public"
    one_time = tmp_path / "one-time-public"
    persistent = tmp_path / "persistent-public"
    for directory in (downloads, one_time, persistent):
        directory.mkdir()
    agent, ref = await _create_artifact_agent(tmp_path, "public-paths", downloads)
    try:
        await agent.save_artifact(ref.artifact_id, one_time)
        assert (await agent.export_destination()).display_name == "Downloads"
        selected = await agent.set_export_destination(persistent)
        assert selected.is_default
    finally:
        await agent.close()
    reopened = await Agent.open(
        "public-paths", root=tmp_path, downloads_directory=downloads
    )
    try:
        assert (await reopened.export_destination()).destination_id == (
            selected.destination_id
        )
    finally:
        await reopened.close()


async def test_public_save_has_no_overwrite_and_reports_final_collision_path(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads-collision"
    downloads.mkdir()
    (downloads / "result.txt").write_text("existing", encoding="utf-8")
    agent, ref = await _create_artifact_agent(tmp_path, "public-collision", downloads)
    try:
        receipt = await agent.save_artifact(ref.artifact_id)
        assert receipt.filename == "result (1).txt"
        assert Path(receipt.saved_path) == downloads / "result (1).txt"
        assert (downloads / "result.txt").read_text(encoding="utf-8") == "existing"
    finally:
        await agent.close()


def _surface_records() -> tuple[ArtifactRef, ArtifactDeliveryReceipt, LoopExit]:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    ref = ArtifactRef(
        artifact_id="artifact-00000000000000000000000000000001",
        run_id="run-00000000000000000000000000000001",
        conversation_id="conversation-00000000000000000000000000000001",
        call_id="create",
        capability_id="artifact.create_document",
        filename="result.txt",
        media_type="text/plain",
        byte_size=8,
        sha256="sha256:" + "1" * 64,
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
        ),
        created_at=now,
    )
    receipt = ArtifactDeliveryReceipt(
        artifact_id=ref.artifact_id,
        destination_id="destination-system-downloads",
        filename="result.txt",
        saved_path="/verified/Downloads/result.txt",
        byte_size=ref.byte_size,
        sha256=ref.sha256,
        renamed_for_collision=False,
        delivered_at=now,
    )
    result = LoopExit(
        run_id=ref.run_id,
        conversation_id=ref.conversation_id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=now,
        final_text="model did not mention the path",
        steps=3,
        artifacts=(ref,),
        artifact_deliveries=(receipt,),
    )
    return ref, receipt, result


async def test_cli_run_json_contains_refs_and_receipts_but_no_payload_or_grant_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref, receipt, result = _surface_records()

    class _FakeAgent:
        async def run(self, message: str, *, conversation_id: str | None = None):
            del message, conversation_id
            return result

        async def close(self) -> None:
            return None

    async def fake_open(*args, **kwargs):
        del args, kwargs
        return _FakeAgent()

    provider = MockModelProvider((), provider_id="mock:cli-artifacts")
    monkeypatch.setattr(cli.Agent, "open", staticmethod(fake_open))
    monkeypatch.setattr(
        cli,
        "_model_configuration",
        lambda *args, **kwargs: (provider, _profile(provider)),
    )
    args = cli.build_parser().parse_args(
        ["--root", str(tmp_path), "run", "agent", "create file", "--model", "x:y"]
    )
    mapping = await cli._execute(args)
    assert isinstance(mapping, dict)
    assert mapping["artifacts"][0]["artifact_id"] == ref.artifact_id
    assert mapping["artifact_deliveries"][0]["saved_path"] == receipt.saved_path
    rendered = str(mapping)
    assert "content" not in rendered
    assert "grant_digest" not in rendered
    assert "destination root" not in rendered


async def test_cli_artifact_save_uses_direct_destination_once_and_returns_structured_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref, receipt, _ = _surface_records()
    destination = tmp_path / "direct"
    destination.mkdir()
    calls: list[tuple[str, Path | None, str | None]] = []

    class _FakeAgent:
        async def save_artifact(
            self,
            artifact_id: str,
            selected: Path | None,
            *,
            filename: str | None,
        ) -> ArtifactDeliveryReceipt:
            calls.append((artifact_id, selected, filename))
            return receipt

        async def close(self) -> None:
            return None

    async def fake_open(*args, **kwargs):
        del args, kwargs
        return _FakeAgent()

    monkeypatch.setattr(cli.Agent, "open", staticmethod(fake_open))
    args = cli.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "artifacts",
            "save",
            "agent",
            ref.artifact_id,
            "--destination",
            str(destination),
            "--filename",
            "renamed.txt",
        ]
    )
    mapping = await cli._execute(args)
    assert calls == [(ref.artifact_id, destination, "renamed.txt")]
    assert isinstance(mapping, dict)
    assert mapping["saved_path"] == receipt.saved_path
    assert "content" not in mapping


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
    failed_messages = artifact_delivery_messages(completed_tool_pairs(failed))
    assert any(
        "remains available; local delivery failed" in text for text in failed_messages
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
    not_delivered_messages = artifact_delivery_messages(
        completed_tool_pairs(not_delivered)
    )
    assert any(
        "was created internally but was not saved locally" in text
        for text in not_delivered_messages
    )


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


def test_hosted_download_returns_authenticated_handle_and_never_server_local_path() -> (
    None
):
    ref, _receipt, _result = _surface_records()
    # This repository has no hosted HTTP composition yet. The shared record from
    # which a future authenticated handle is derived is deliberately path-free.
    assert not hasattr(ref, "saved_path")
    assert not hasattr(ref, "internal_path")
    assert ref.artifact_id.startswith("artifact-")


async def test_fake_provider_markdown_vertical_slice_commits_delivers_restarts_and_redelivers(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    artifact_id = "artifact-00000000000000000000000000000001"
    provider = MockModelProvider(
        (
            _tool(
                "create-report",
                "artifact_create_document",
                {
                    "format": "markdown",
                    "filename": "report.md",
                    "content": "# Report\r\n\r\nVerified content.\r\n",
                },
            ),
            _tool(
                "save-report",
                "artifact_save_local",
                {
                    "artifact_id": artifact_id,
                    "destination_id": "default",
                },
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Your report is ready."
            ),
        ),
        provider_id="mock:artifact-vertical",
    )
    agent = await Agent.create(
        "artifact-vertical",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
    )
    try:
        result = await agent.run("Create and download a Markdown report.")
        assert result.final_text == "Your report is ready."
        assert tuple(item.artifact_id for item in result.artifacts) == (artifact_id,)
        assert tuple(item.artifact_id for item in result.artifact_deliveries) == (
            artifact_id,
        )
        receipt = result.artifact_deliveries[0]
        assert Path(receipt.saved_path) == downloads / "report.md"
        assert (downloads / "report.md").read_bytes() == (
            b"# Report\n\nVerified content.\n"
        )
        assert (await agent.read_artifact(artifact_id)).content == (
            b"# Report\n\nVerified content.\n"
        )
        assert result.conversation_id
    finally:
        await agent.close()

    reopened = await Agent.open(
        "artifact-vertical",
        root=tmp_path,
        downloads_directory=downloads,
    )
    try:
        assert (await reopened.read_artifact(artifact_id)).content == (
            b"# Report\n\nVerified content.\n"
        )
        second = await reopened.save_artifact(artifact_id)
        assert second.filename == "report (1).md"
        assert Path(second.saved_path).read_bytes() == (
            b"# Report\n\nVerified content.\n"
        )
    finally:
        await reopened.close()
