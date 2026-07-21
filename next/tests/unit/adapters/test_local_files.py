from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import threading

import pytest

import daita.adapters.local_files as local_files
from daita.adapters import (
    DiscoveryLimitError,
    DiscoveryRequest,
    LocalDirectoryResourceAdapter,
    LocalDirectorySource,
    LocalDirectorySourceError,
    ResourceAdapter,
    ResourceRef,
    SourceClosedError,
    StaleResourceError,
)
from daita.catalog import FacetKind, RelationshipKind, ResourceKind

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


async def _open(
    root: Path,
    *,
    max_files: int = 1_000,
    max_file_bytes: int = 2 * 1024 * 1024,
) -> LocalDirectoryResourceAdapter:
    return await LocalDirectorySource(
        root,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
    ).open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW + timedelta(seconds=5),
    )


def _request(
    adapter: LocalDirectoryResourceAdapter,
    sync_id: str = "sync-1",
    **limits: int,
) -> DiscoveryRequest:
    return DiscoveryRequest(
        agent_id="agent-1",
        source_id=adapter.registration.id,
        sync_id=sync_id,
        requested_at=NOW,
        **limits,
    )


async def test_local_directory_discovers_bounded_file_facets_and_contains_edges(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    nested = root / "daily"
    nested.mkdir(parents=True)
    (root / "customers.csv").write_text("id,name\n1,Ada\n", encoding="utf-8")
    (nested / "orders.json").write_text(
        '[{"id": 1, "customer_id": 1}]',
        encoding="utf-8",
    )
    (root / "ignored.txt").write_text("ignore", encoding="utf-8")
    outside = tmp_path / "outside.csv"
    outside.write_text("secret\nnever-read\n", encoding="utf-8")
    (root / "escape.csv").symlink_to(outside)
    adapter = await _open(root)

    try:
        result = await adapter.discover(_request(adapter))
    finally:
        await adapter.close()

    snapshot = result.snapshot
    assert snapshot.sync.adapter_id == "local-directory"
    assert snapshot.sync.resource_count == 3
    assert snapshot.sync.relationship_count == 2
    assert [item.native_identity for item in snapshot.resources] == [
        ".",
        "customers.csv",
        "daily/orders.json",
    ]
    assert {item.kind for item in snapshot.resources} == {
        ResourceKind.FOLDER,
        ResourceKind.FILE,
    }
    assert all(
        item.kind is RelationshipKind.CONTAINS for item in snapshot.relationships
    )
    file_resources = [
        item for item in snapshot.resources if item.kind is ResourceKind.FILE
    ]
    facets_by_resource = {
        resource.id: {
            facet.kind for facet in snapshot.facets if facet.resource_id == resource.id
        }
        for resource in file_resources
    }
    assert all(
        kinds == {FacetKind.FILE, FacetKind.TABULAR}
        for kinds in facets_by_resource.values()
    )
    assert all(
        item.external_uri.startswith("daita-local://")
        and str(root) not in item.external_uri
        for item in snapshot.resources
    )
    assert adapter.registration.configuration["root"] == str(root.resolve())
    assert adapter.registration.configuration["formats"] == ("csv", "json")


async def test_local_directory_file_freshness_uses_stat_mtime_not_filename_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "adversarial-freshness"
    root.mkdir()
    lexically_first = root / "a-export.csv"
    lexically_last = root / "z-export.csv"
    lexically_first.write_text("id\n1\n", encoding="utf-8")
    lexically_last.write_text("id\n2\n", encoding="utf-8")
    older_timestamp = NOW.timestamp() - 300
    newer_timestamp = NOW.timestamp() - 60
    os.utime(lexically_first, (newer_timestamp, newer_timestamp))
    os.utime(lexically_last, (older_timestamp, older_timestamp))
    adapter = await _open(root)

    try:
        result = await adapter.discover(_request(adapter))
    finally:
        await adapter.close()

    freshness_by_name: dict[str, str] = {}
    for resource in result.snapshot.resources:
        if resource.kind is not ResourceKind.FILE:
            continue
        modified_at = next(
            facet.payload["modified_at"]
            for facet in result.snapshot.facets
            if facet.resource_id == resource.id and facet.kind is FacetKind.FILE
        )
        assert isinstance(modified_at, str)
        freshness_by_name[resource.name] = modified_at
    assert tuple(sorted(freshness_by_name)) == ("a-export.csv", "z-export.csv")
    assert freshness_by_name["a-export.csv"] > freshness_by_name["z-export.csv"]


@pytest.mark.parametrize("kind", ["missing", "file", "symlink", "parent_symlink"])
async def test_local_directory_rejects_unsafe_roots(
    tmp_path: Path,
    kind: str,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    if kind == "missing":
        candidate = tmp_path / "missing"
    elif kind == "file":
        candidate = tmp_path / "file.csv"
        candidate.write_text("id\n1\n", encoding="utf-8")
    elif kind == "symlink":
        candidate = tmp_path / "linked"
        candidate.symlink_to(real, target_is_directory=True)
    else:
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real, target_is_directory=True)
        child = real / "child"
        child.mkdir()
        candidate = linked_parent / "child"

    with pytest.raises(LocalDirectorySourceError) as caught:
        await _open(candidate)

    assert caught.value.code == "local_root_invalid"


@pytest.mark.parametrize("substitution", ["symlink", "directory"])
async def test_root_admission_rejects_ancestor_component_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    admitted_parent = tmp_path / "admitted-parent"
    admitted_parent.mkdir()
    (admitted_parent / "selected-root").mkdir()
    replacement = tmp_path / "replacement-parent"
    replacement.mkdir()
    (replacement / "selected-root").mkdir()
    parked = tmp_path / "parked-parent"
    candidate = admitted_parent / "selected-root"
    real_open = local_files.os.open
    substituted = False

    def substituting_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal substituted
        path_text = local_files.os.fspath(path)
        opening_old_full_path = dir_fd is None and Path(path_text) == candidate
        opening_walked_ancestor = (
            dir_fd is not None and path_text == admitted_parent.name
        )
        if not substituted and (opening_old_full_path or opening_walked_ancestor):
            admitted_parent.rename(parked)
            if substitution == "symlink":
                admitted_parent.symlink_to(replacement, target_is_directory=True)
            else:
                replacement.rename(admitted_parent)
            substituted = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(local_files.os, "open", substituting_open)

    with pytest.raises(LocalDirectorySourceError) as caught:
        await _open(candidate)

    assert substituted is True
    assert caught.value.code == "local_root_invalid"


async def test_discovery_limits_fail_without_a_partial_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "bounded"
    root.mkdir()
    (root / "one.csv").write_text("id\n1\n", encoding="utf-8")
    (root / "two.csv").write_text("id\n2\n", encoding="utf-8")
    adapter = await _open(root, max_files=1)

    try:
        with pytest.raises(DiscoveryLimitError, match="file limit"):
            await adapter.discover(_request(adapter))
        assert adapter._latest is None
    finally:
        await adapter.close()

    bytes_adapter = await _open(root, max_file_bytes=3)
    try:
        with pytest.raises(DiscoveryLimitError, match="byte limit"):
            await bytes_adapter.discover(_request(bytes_adapter))
        assert bytes_adapter._latest is None
    finally:
        await bytes_adapter.close()


async def test_inspect_health_declarations_and_close_follow_adapter_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inspect"
    root.mkdir()
    (root / "customers.csv").write_text("id,name\n1,Ada\n", encoding="utf-8")
    adapter = await _open(root)

    assert isinstance(adapter, ResourceAdapter)
    assert not hasattr(adapter, "read")
    assert not hasattr(adapter, "execute")
    declarations = adapter.declarations()
    assert [item.id for item in declarations.capabilities] == ["data.file.read"]
    discovered = await adapter.discover(_request(adapter))
    resource = next(
        item for item in discovered.snapshot.resources if item.kind is ResourceKind.FILE
    )
    inspected = await adapter.inspect(ResourceRef.from_resource(resource))
    assert inspected.resource.id == resource.id
    assert {item.kind for item in inspected.facets} == {
        FacetKind.FILE,
        FacetKind.TABULAR,
    }
    stale = replace(
        ResourceRef.from_resource(resource),
        revision="sha256:" + "0" * 64,
    )
    with pytest.raises(StaleResourceError):
        await adapter.inspect(stale)
    assert (await adapter.health()).healthy is True

    await adapter.close()
    await adapter.close()
    assert (await adapter.health()).error_code == "source_closed"
    with pytest.raises(SourceClosedError):
        await adapter.discover(_request(adapter, "sync-closed"))


async def test_discovery_cancellation_finishes_worker_cleanup_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cancel"
    root.mkdir()
    (root / "customers.csv").write_text("id\n1\n", encoding="utf-8")
    adapter = await _open(root)
    original = local_files._scan_directory
    started = threading.Event()
    release = threading.Event()

    def blocked_scan(*args, **kwargs):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release discovery")
        return original(*args, **kwargs)

    monkeypatch.setattr(local_files, "_scan_directory", blocked_scan)
    discovery = asyncio.create_task(adapter.discover(_request(adapter)))
    close = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        discovery.cancel()
        await asyncio.sleep(0)
        assert not discovery.done()
        close = asyncio.create_task(adapter.close())
        await asyncio.sleep(0)
        assert not close.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await discovery
        await close
        assert adapter._latest is None
    finally:
        release.set()
        await asyncio.gather(discovery, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await adapter.close()
