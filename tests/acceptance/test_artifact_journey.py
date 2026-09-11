"""Component-owned tests split from ``test_public_surfaces.py``."""

from __future__ import annotations

from tests.artifacts._public_surface_support import (
    Agent,
    FinishReason,
    MockModelProvider,
    ModelResponse,
    Path,
    _ids,
    _profile,
    _tool,
    workspace_for,
)


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
                    "mode": "create_new",
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
        workspace=workspace_for(tmp_path),
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
        workspace=workspace_for(tmp_path),
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
