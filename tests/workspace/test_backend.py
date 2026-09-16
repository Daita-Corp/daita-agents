from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from daita._json import FrozenJsonObject
from daita.adapters.local_workspace import (
    LocalWorkspaceBackend,
    LocalWorkspaceError,
    LocalWorkspaceLimits,
)
from daita.llm.models import ModelSensitivity
from daita.workspace import LocalFileAccess, LocalWorkspace

_NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


async def _backend(
    tmp_path: Path,
    *,
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
    limits: LocalWorkspaceLimits | None = None,
) -> tuple[Path, LocalWorkspaceBackend]:
    workspace = tmp_path / "workspace"
    state_root = tmp_path / "state"
    agent_home = state_root / "agent"
    workspace.mkdir()
    agent_home.mkdir(parents=True)
    backend = await LocalWorkspaceBackend.open(
        LocalWorkspace(workspace, sensitivity),
        agent_root=state_root,
        agent_home=agent_home,
        limits=limits,
        clock=lambda: _NOW,
    )
    return workspace, backend


def test_workspace_intent_is_canonical_immutable_and_internal_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    workspace = LocalWorkspace(root / ".")

    assert workspace.root == root.resolve(strict=True)
    assert workspace.sensitivity is ModelSensitivity.INTERNAL
    with pytest.raises(AttributeError):
        workspace.root = tmp_path  # type: ignore[misc]
    with pytest.raises(ValueError, match="internal or stricter"):
        LocalWorkspace(root, ModelSensitivity.PUBLIC)
    with pytest.raises(ValueError, match="existing directory"):
        LocalWorkspace(tmp_path / "missing")
    with pytest.raises(ValueError, match="filesystem root"):
        LocalWorkspace(Path(root.anchor), access=LocalFileAccess.COMPUTER)


@pytest.mark.parametrize("relationship", ["equal", "workspace_parent", "state_parent"])
async def test_workspace_admission_rejects_state_overlap(
    tmp_path: Path,
    relationship: str,
) -> None:
    if relationship == "equal":
        state_root = tmp_path / "shared"
        workspace_root = state_root
    elif relationship == "workspace_parent":
        workspace_root = tmp_path / "workspace"
        state_root = workspace_root / "state"
    else:
        state_root = tmp_path / "state"
        workspace_root = state_root / "workspace"
    agent_home = state_root / "agent"
    workspace_root.mkdir(parents=True)
    agent_home.mkdir(parents=True, exist_ok=True)

    with pytest.raises(LocalWorkspaceError) as failure:
        await LocalWorkspaceBackend.open(
            LocalWorkspace(workspace_root),
            agent_root=state_root,
            agent_home=agent_home,
        )

    assert failure.value.code == "workspace_state_overlap"
    assert str(workspace_root) not in str(failure.value)


async def test_search_is_deterministic_bounded_and_supports_latest_metadata(
    tmp_path: Path,
) -> None:
    root, backend = await _backend(tmp_path)
    try:
        older = root / "logs" / "service-old.log"
        newer = root / "logs" / "service-new.log"
        older.parent.mkdir()
        older.write_text("alpha\nneedle across chunks\n", encoding="utf-8")
        newer.write_text("newest needle\n", encoding="utf-8")
        os.utime(
            older,
            ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000),
        )
        os.utime(
            newer,
            ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000),
        )

        paths = await backend.search(
            run_id="run-search",
            query="service",
            path="logs",
            glob="*.log",
        )
        latest = await backend.search(
            run_id="run-search",
            query="needle",
            mode="both",
            order_by="modified_desc",
        )

        assert [item.path for item in paths.matches] == [
            "logs/service-new.log",
            "logs/service-old.log",
        ]
        assert latest.matches[0].path == "logs/service-new.log"
        assert latest.matches[0].modified_at > latest.matches[-1].modified_at
        assert all(item.size_bytes > 0 for item in latest.matches)
        assert all(
            item.physical_revision.startswith("sha256:") for item in latest.matches
        )
        assert latest.scanned_entries >= 3
    finally:
        await backend.close()


async def test_computer_access_searches_multiple_external_roots_with_one_truthful_result(
    tmp_path: Path,
) -> None:
    working = tmp_path / "project"
    user_home = tmp_path / "home"
    downloads = user_home / "Downloads"
    documents = user_home / "Documents"
    state_root = user_home / ".daita"
    agent_home = state_root / "agent"
    for directory in (working, downloads, documents, agent_home):
        directory.mkdir(parents=True)
    older = documents / "report.csv"
    newer = downloads / "report.csv"
    older.write_text("source,value\ndocuments,1\n", encoding="utf-8")
    newer.write_text("source,value\ndownloads,2\n", encoding="utf-8")
    os.utime(
        older,
        ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000),
    )
    os.utime(
        newer,
        ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000),
    )
    backend = await LocalWorkspaceBackend.open(
        LocalWorkspace(working, access=LocalFileAccess.COMPUTER),
        agent_root=state_root,
        agent_home=agent_home,
        user_home=user_home,
        known_directories={
            "Downloads": downloads,
            "Documents": documents,
            "Desktop": user_home / "Desktop",
        },
    )
    try:
        result = await backend.search(
            run_id="run-computer-search",
            query=None,
            paths=(str(downloads), str(documents)),
            glob="*.csv",
            order_by="modified_desc",
        )

        assert [item.qualified_path for item in result.matches] == [
            str(newer),
            str(older),
        ]
        assert result.scan_complete is True
        assert [item.status for item in result.coverage] == ["complete", "complete"]
        assert (
            sum(item.scanned_entries for item in result.coverage)
            == result.scanned_entries
        )
        provenance = result.provenance_mapping()
        anchors = provenance["anchors"]
        assert isinstance(anchors, tuple)
        anchor_objects = tuple(
            item for item in anchors if isinstance(item, FrozenJsonObject)
        )
        assert len(anchor_objects) == len(anchors)
        assert {item["relative_path"] for item in anchor_objects} == {
            "Documents",
            "Downloads",
        }

        read = await backend.read(
            run_id="run-computer-read", path="~/Downloads/report.csv"
        )
        assert read.qualified_path == str(newer)
        assert "downloads,2" in read.content
        binding = backend.authenticate_file_binding(
            run_id="run-computer-read", token=read.binding
        )
        assert binding.workspace_id != backend.workspace_id
    finally:
        await backend.close()


async def test_computer_search_reports_partial_roots_and_prunes_private_state(
    tmp_path: Path,
) -> None:
    working = tmp_path / "project"
    user_home = tmp_path / "home"
    readable = user_home / "Documents"
    missing = user_home / "Missing"
    state_root = user_home / "private-state"
    agent_home = state_root / "agent"
    for directory in (working, readable, agent_home):
        directory.mkdir(parents=True)
    (readable / "visible.csv").write_text("id\n1\n", encoding="utf-8")
    (agent_home / "hidden.csv").write_text("secret\n", encoding="utf-8")
    backend = await LocalWorkspaceBackend.open(
        LocalWorkspace(working, access=LocalFileAccess.COMPUTER),
        agent_root=state_root,
        agent_home=agent_home,
        user_home=user_home,
    )
    try:
        partial = await backend.search(
            run_id="run-partial",
            query=None,
            paths=(str(missing), str(readable)),
            glob="*.csv",
        )
        assert [item.qualified_path for item in partial.matches] == [
            str(readable / "visible.csv")
        ]
        assert partial.scan_complete is False
        assert [item.status for item in partial.coverage] == ["failed", "complete"]
        assert partial.coverage[0].error_code == "path_not_found"

        home = await backend.search(
            run_id="run-home", query=None, path="~", glob="*.csv"
        )
        assert str(agent_home / "hidden.csv") not in {
            item.qualified_path for item in home.matches
        }
        with pytest.raises(LocalWorkspaceError) as protected:
            await backend.read(
                run_id="run-protected", path=str(agent_home / "hidden.csv")
            )
        assert protected.value.code == "path_restricted"
    finally:
        await backend.close()


async def test_bounded_workspace_rejects_absolute_outside_paths_and_search_glob_guides(
    tmp_path: Path,
) -> None:
    root, backend = await _backend(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        with pytest.raises(LocalWorkspaceError) as denied:
            await backend.read(run_id="run-bounded", path=str(outside))
        assert denied.value.code == "path_invalid"

        with pytest.raises(LocalWorkspaceError) as invalid_glob:
            await backend.search(
                run_id="run-glob", query=None, path=".", glob="**/*.csv"
            )
        assert invalid_glob.value.code == "search_invalid"
        assert "glob='*.csv'" in invalid_glob.value.message
        assert "subdirectories automatically" in invalid_glob.value.message
        assert root.exists()
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "path",
    (
        "../outside.txt",
        "https://example.test/file.txt",
        "~otheruser/file.txt",
        "bad\nname.txt",
    ),
)
async def test_computer_access_rejects_unsafe_locator_forms_before_traversal(
    tmp_path: Path,
    path: str,
) -> None:
    working = tmp_path / "working"
    user_home = tmp_path / "home"
    agent_home = user_home / ".daita" / "agent"
    working.mkdir()
    agent_home.mkdir(parents=True)
    backend = await LocalWorkspaceBackend.open(
        LocalWorkspace(working, access=LocalFileAccess.COMPUTER),
        agent_root=user_home / ".daita",
        agent_home=agent_home,
        user_home=user_home,
    )
    try:
        with pytest.raises(LocalWorkspaceError) as failure:
            await backend.read(run_id="run-invalid-path", path=path)
        assert failure.value.code == "path_invalid"
    finally:
        await backend.close()


async def test_computer_paths_support_relative_home_absolute_spaces_and_unicode(
    tmp_path: Path,
) -> None:
    working = tmp_path / "working"
    user_home = tmp_path / "home"
    documents = user_home / "Documents"
    agent_home = user_home / ".daita" / "agent"
    working.mkdir()
    documents.mkdir(parents=True)
    agent_home.mkdir(parents=True)
    relative = working / "relative file.txt"
    home_file = documents / "résumé.txt"
    relative.write_text("relative", encoding="utf-8")
    home_file.write_text("home", encoding="utf-8")
    backend = await LocalWorkspaceBackend.open(
        LocalWorkspace(working, access=LocalFileAccess.COMPUTER),
        agent_root=user_home / ".daita",
        agent_home=agent_home,
        user_home=user_home,
    )
    try:
        relative_result = await backend.read(
            run_id="run-relative", path="relative file.txt"
        )
        home_result = await backend.read(
            run_id="run-home-path", path="~/Documents/résumé.txt"
        )
        absolute_result = await backend.read(run_id="run-absolute", path=str(home_file))
        assert relative_result.qualified_path == str(relative)
        assert home_result.qualified_path == str(home_file)
        assert absolute_result.content == home_result.content == "home"
    finally:
        await backend.close()


@pytest.mark.parametrize(
    ("path", "paths", "query", "mode", "glob"),
    (
        (".", (".",), "name", "paths", None),
        (None, (), "name", "paths", None),
        (None, tuple(str(index) for index in range(9)), "name", "paths", None),
        (None, None, None, "paths", None),
        (None, None, None, "content", "*.txt"),
    ),
)
async def test_search_rejects_ambiguous_or_unbounded_arguments(
    tmp_path: Path,
    path: str | None,
    paths: tuple[str, ...] | None,
    query: str | None,
    mode: str,
    glob: str | None,
) -> None:
    _root, backend = await _backend(tmp_path)
    try:
        with pytest.raises(LocalWorkspaceError) as failure:
            await backend.search(
                run_id="run-invalid-search",
                path=path,
                paths=paths,
                query=query,
                mode=mode,
                glob=glob,
            )
        assert failure.value.code in {"path_invalid", "search_invalid"}
    finally:
        await backend.close()


async def test_overlapping_roots_are_deduplicated_and_shared_limits_are_explicit(
    tmp_path: Path,
) -> None:
    working = tmp_path / "working"
    user_home = tmp_path / "home"
    downloads = user_home / "Downloads"
    documents = user_home / "Documents"
    agent_home = user_home / ".daita" / "agent"
    for directory in (working, downloads, documents, agent_home):
        directory.mkdir(parents=True)
    (downloads / "one.csv").write_text("id\n1\n", encoding="utf-8")
    (documents / "one.csv").write_text("id\n2\n", encoding="utf-8")
    (documents / "two.csv").write_text("id\n3\n", encoding="utf-8")
    backend = await LocalWorkspaceBackend.open(
        LocalWorkspace(working, access=LocalFileAccess.COMPUTER),
        agent_root=user_home / ".daita",
        agent_home=agent_home,
        limits=LocalWorkspaceLimits(max_search_entries=1),
        user_home=user_home,
    )
    try:
        overlap = await backend.search(
            run_id="run-overlap",
            paths=(str(user_home), str(downloads)),
            query="one",
        )
        assert len(overlap.matches) <= 1
        assert [item.status for item in overlap.coverage] == [
            "partial",
            "deduplicated",
        ]

        shared = await backend.search(
            run_id="run-shared-budget",
            paths=(str(downloads), str(documents)),
            query="csv",
        )
        assert shared.scanned_entries == 1
        assert [item.status for item in shared.coverage] == [
            "complete",
            "unsearched",
        ]
        assert shared.scan_complete is False
    finally:
        await backend.close()


async def test_search_content_crosses_worker_chunks_and_skips_binary_and_secrets(
    tmp_path: Path,
) -> None:
    root, backend = await _backend(tmp_path)
    try:
        prefix = "x" * (64 * 1_024 - 3)
        (root / "large.log").write_text(prefix + "needle\n", encoding="utf-8")
        (root / "binary.bin").write_bytes(b"needle\x00binary")
        (root / ".env").write_text("needle=secret\n", encoding="utf-8")
        (root / ".git").mkdir()
        (root / ".git" / "ignored.txt").write_text("needle\n", encoding="utf-8")

        result = await backend.search(
            run_id="run-content",
            query="needle",
            mode="content",
        )

        assert [item.path for item in result.matches] == ["large.log"]
        assert result.matches[0].line == 1
        assert len(result.matches[0].excerpt or "") <= 320
    finally:
        await backend.close()


async def test_search_invalid_utf8_preserves_the_global_content_byte_limit(
    tmp_path: Path,
) -> None:
    root, backend = await _backend(
        tmp_path,
        limits=LocalWorkspaceLimits(max_content_scan_bytes=100),
    )
    try:
        (root / "a.txt").write_bytes(b"x" * 80 + b"\xff")
        (root / "b.txt").write_bytes(b"y" * 80 + b"\xff")

        result = await backend.search(
            run_id="run-invalid-utf8",
            query="never",
            mode="content",
        )

        assert result.matches == ()
        assert result.scanned_content_bytes == 100
        assert result.truncated is True
        assert result.truncation_reasons == ("content_byte_limit",)
    finally:
        await backend.close()


async def test_read_chunks_utf8_returns_binding_and_walks_backward_from_end(
    tmp_path: Path,
) -> None:
    limits = LocalWorkspaceLimits(
        max_visible_read_bytes=17,
        max_raw_read_bytes=64,
    )
    root, backend = await _backend(tmp_path, limits=limits)
    try:
        content = "head🙂\nmiddle🙂\ntail🙂\n"
        (root / "events.log").write_text(content, encoding="utf-8")

        first = await backend.read(run_id="run-read", path="events.log")
        assert first.start_offset == 0
        assert first.complete is False
        assert first.cursor is not None
        assert first.content.encode("utf-8").decode("utf-8") == first.content
        binding = backend.authenticate_file_binding(
            run_id="run-read",
            token=first.binding,
        )
        assert binding.relative_path == "events.log"
        assert binding.physical_revision == first.physical_revision

        second = await backend.read(run_id="run-read", cursor=first.cursor)
        assert second.start_offset == first.end_offset
        assert first.content + second.content == content
        assert second.complete is True

        tail = await backend.read(
            run_id="run-tail",
            path="events.log",
            position="end",
        )
        assert tail.end_offset == len(content.encode("utf-8"))
        assert tail.cursor is not None
        prior = await backend.read(run_id="run-tail", cursor=tail.cursor)
        assert prior.end_offset == tail.start_offset
        assert prior.content + tail.content == content
        assert prior.complete is True
    finally:
        await backend.close()


async def test_read_rejects_traversal_symlink_special_secret_and_drift(
    tmp_path: Path,
) -> None:
    root, backend = await _backend(
        tmp_path,
        limits=LocalWorkspaceLimits(max_visible_read_bytes=8, max_raw_read_bytes=64),
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (root / "normal.txt").write_text("0123456789abcdef", encoding="utf-8")
    (root / ".env").write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    fifo = root / "pipe"
    os.mkfifo(fifo)
    try:
        for path, code in (
            ("../outside.txt", "path_invalid"),
            ("link.txt", "symlink_not_allowed"),
            ("pipe", "not_regular_file"),
            (".env", "path_restricted"),
        ):
            with pytest.raises(LocalWorkspaceError) as failure:
                await backend.read(run_id="run-denied", path=path)
            assert failure.value.code == code

        first = await backend.read(run_id="run-drift", path="normal.txt")
        assert first.cursor is not None
        (root / "normal.txt").write_text("changed completely", encoding="utf-8")
        with pytest.raises(LocalWorkspaceError) as drift:
            await backend.read(run_id="run-drift", cursor=first.cursor)
        assert drift.value.code == "file_changed"
    finally:
        await backend.close()


async def test_cursor_and_binding_tokens_are_run_session_and_purpose_bound(
    tmp_path: Path,
) -> None:
    root, backend = await _backend(
        tmp_path,
        limits=LocalWorkspaceLimits(max_visible_read_bytes=4, max_raw_read_bytes=64),
    )
    (root / "notes.txt").write_text("abcdefgh", encoding="utf-8")
    try:
        result = await backend.read(run_id="run-token", path="notes.txt")
        assert result.cursor is not None

        with pytest.raises(LocalWorkspaceError) as wrong_run:
            await backend.read(run_id="other-run", cursor=result.cursor)
        assert wrong_run.value.code == "cursor_expired"
        with pytest.raises(LocalWorkspaceError) as wrong_purpose:
            await backend.read(run_id="run-token", cursor=result.binding)
        assert wrong_purpose.value.code == "cursor_invalid"
        with pytest.raises(LocalWorkspaceError) as explicit_position:
            await backend.read(
                run_id="run-token",
                cursor=result.cursor,
                position="start",
            )
        assert explicit_position.value.code == "cursor_invalid"
        tampered = result.cursor[:-1] + ("A" if result.cursor[-1] != "A" else "B")
        with pytest.raises(LocalWorkspaceError) as invalid:
            await backend.read(run_id="run-token", cursor=tampered)
        assert invalid.value.code == "cursor_invalid"
        with pytest.raises(LocalWorkspaceError) as binding_run:
            backend.authenticate_file_binding(
                run_id="other-run",
                token=result.binding,
            )
        assert binding_run.value.code == "file_binding_expired"

        state_root = tmp_path / "other-state"
        state_home = state_root / "agent"
        state_home.mkdir(parents=True)
        reopened = await LocalWorkspaceBackend.open(
            LocalWorkspace(root),
            agent_root=state_root,
            agent_home=state_home,
        )
        try:
            with pytest.raises(LocalWorkspaceError) as expired:
                await reopened.read(run_id="run-token", cursor=result.cursor)
            assert expired.value.code == "cursor_expired"
        finally:
            await reopened.close()
    finally:
        await backend.close()


async def test_backend_close_revokes_descriptor_and_tokens(tmp_path: Path) -> None:
    root, backend = await _backend(tmp_path)
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    result = await backend.read(run_id="run-close", path="notes.txt")

    await backend.close()
    await backend.close()

    assert backend.closed is True
    with pytest.raises(LocalWorkspaceError) as closed:
        await backend.read(run_id="run-close", path="notes.txt")
    assert closed.value.code == "workspace_unavailable"
    with pytest.raises(LocalWorkspaceError):
        backend.authenticate_file_binding(run_id="run-close", token=result.binding)
