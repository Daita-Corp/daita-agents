"""A folder source says when its catalog is out of date, and re-reads only what changed."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from daita import FolderSource
from daita.adapters import folder as folder_module
from daita.adapters.folder import FolderBackend
from daita.adapters.models import DiscoveryRequest
from daita.adapters.registry import SourceBackendContext
from tests.support.source_contract import (
    ProbeMutation,
    StaticSources,
    assert_change_detection_contract,
    reconcile_pass,
    scoped_revisions,
)

_NOW = datetime(2026, 11, 7, tzinfo=UTC)
_AGENT = "agent"
_ROOT = frozenset({"."})


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    # A rewrite inside one timestamp tick must still look different.
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


async def _open(tmp_path: Path, folder: Path):
    home = tmp_path / "state" / "agents" / "a"
    home.mkdir(parents=True)
    adapter = await FolderSource(folder).open(
        agent_id=_AGENT, attached_at=_NOW, clock=lambda: _NOW
    )
    context = SimpleNamespace(
        store=StaticSources(adapter.registration),
        home=home,
        session_workspace=None,
        session_source_id=None,
        clock=lambda: _NOW,
    )
    backend = FolderBackend(cast(SourceBackendContext, context))

    def request(**options: Any) -> DiscoveryRequest:
        return DiscoveryRequest(
            agent_id=_AGENT,
            source_id=adapter.registration.id,
            sync_id="sync",
            requested_at=_NOW,
            agent_root=home.parent.parent,
            agent_home=home,
            **options,
        )

    return adapter, backend, request


@pytest.mark.contract
async def test_folder_probe_moves_with_the_catalog(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    _write(folder / "orders.csv", "id,total\n1,5\n")
    _write(folder / "notes.md", "not data\n")
    adapter, backend, request = await _open(tmp_path, folder)

    async def rewrite() -> None:
        _write(folder / "orders.csv", "id,total,tax\n1,5,1\n")

    async def add() -> None:
        _write(folder / "sub" / "users.csv", "id\n1\n")

    async def remove() -> None:
        (folder / "sub" / "users.csv").unlink()

    async def rename() -> None:
        (folder / "orders.csv").rename(folder / "sales.csv")

    async def unrelated() -> None:
        _write(folder / "notes.md", "different text\n")
        (folder / "empty").mkdir()

    async def read() -> None:
        (folder / "orders.csv").read_text()

    try:
        await assert_change_detection_contract(
            lambda: backend.probe(
                agent_id=_AGENT, source_id=adapter.registration.id, cursor=None
            ),
            [
                ProbeMutation("file read", read),
                ProbeMutation("file rewritten", rewrite, _ROOT),
                ProbeMutation("file added in a new folder", add, _ROOT),
                ProbeMutation("file removed", remove, _ROOT),
                ProbeMutation("file renamed", rename, _ROOT),
                ProbeMutation("files the catalog never lists", unrelated),
            ],
            revisions=lambda: scoped_revisions(adapter, request(), lambda _: "."),
        )
    finally:
        await adapter.close()


async def test_folder_read_describes_only_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "data"
    for name in ("a", "b", "c"):
        _write(folder / f"{name}.csv", "id,v\n1,2\n")
    adapter, _, request = await _open(tmp_path, folder)
    described: list[tuple[str, ...]] = []
    real = folder_module._describe

    async def spy(backend: Any, paths: tuple[str, ...]):
        described.append(paths)
        return await real(backend, paths)

    monkeypatch.setattr(folder_module, "_describe", spy)
    try:
        first = await reconcile_pass(adapter, request())
        _write(folder / "b.csv", "id,v,w\n1,2,3\n")
        again = await reconcile_pass(adapter, request(previous=first.detail.entries))
        assert described[-1] == ("b.csv",)
        by_name = {e.resource.name: e for e in again.detail.entries}
        before = {e.resource.name: e for e in first.detail.entries}
        assert by_name["a.csv"].revision == before["a.csv"].revision
        assert by_name["b.csv"].revision != before["b.csv"].revision
    finally:
        await adapter.close()


async def test_folder_fill_describes_what_lacks_detail_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "data"
    for index in range(6):
        _write(folder / f"f{index}.csv", "id\n1\n")
    adapter, _, request = await _open(tmp_path, folder)
    try:
        first = await reconcile_pass(adapter, request(max_resources=3))
        files = {
            e.resource.name for e in first.detail.entries if "csv" in e.resource.name
        }
        assert files and first.detail.truncated
        seen = first.detail.entries
        for _ in range(6):
            step = await reconcile_pass(
                adapter, request(max_resources=3, previous=seen, fill=True)
            )
            assert {e.resource.id for e in seen} <= {
                e.resource.id for e in step.detail.entries
            }
            seen = step.detail.entries
            if not step.detail.truncated:
                break
        assert not step.detail.truncated
        assert sum(e.resource.name.endswith(".csv") for e in seen) == 6
    finally:
        await adapter.close()
