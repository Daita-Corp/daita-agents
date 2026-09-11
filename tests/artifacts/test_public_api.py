"""Component-owned tests split from ``test_public_surfaces.py``."""

from __future__ import annotations

from tests.artifacts._public_surface_support import (
    Agent,
    Path,
    _create_artifact_agent,
    workspace_for,
)


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
        "public-restart",
        root=tmp_path,
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
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
        "public-paths",
        root=tmp_path,
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
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
