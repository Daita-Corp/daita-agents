from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading

import pytest

import daita.adapters.local_files as local_files
from daita.adapters import (
    DiscoveryRequest,
    LocalDirectoryReadBackend,
    LocalDirectorySource,
    LocalDirectorySourceError,
)
from daita.catalog import ResourceKind

NOW = datetime(2026, 7, 19, 16, 0, tzinfo=timezone.utc)


class Sources:
    def __init__(self, registration) -> None:
        self.registration = registration

    async def register_source(self, registration):
        self.registration = registration
        return registration

    async def load_source(self, agent_id, source_id):
        if agent_id == "agent-1" and source_id == self.registration.id:
            return self.registration
        return None

    async def list_sources(self, agent_id):
        return (self.registration,) if agent_id == "agent-1" else ()

    async def detach_source(self, agent_id, source_id, detached_at):
        self.registration = self.registration.detach(detached_at)
        return self.registration


class Catalog:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot

    async def record_sync(self, sync):
        return sync

    async def commit_snapshot(self, snapshot):
        self.snapshot = snapshot
        return snapshot

    async def load_sync(self, agent_id, sync_id):
        return self.snapshot.sync if self.snapshot.sync.id == sync_id else None

    async def load_resource(self, agent_id, resource_id):
        return next(
            (item for item in self.snapshot.resources if item.id == resource_id),
            None,
        )

    async def load_revision(self, agent_id, resource_id, revision):
        return next(
            (
                item
                for item in self.snapshot.revisions
                if item.resource_id == resource_id and item.revision == revision
            ),
            None,
        )

    async def list_resources(self, agent_id, source_id=None):
        return self.snapshot.resources

    async def load_facets(self, agent_id, resource_id, revision=None):
        return tuple(
            item for item in self.snapshot.facets if item.resource_id == resource_id
        )

    async def load_incident_relationships(
        self, agent_id, resource_id, *, relationship_kinds=(), limit=50
    ):
        return tuple(
            item
            for item in self.snapshot.relationships
            if resource_id in {item.from_resource_id, item.to_resource_id}
            and (not relationship_kinds or item.kind in relationship_kinds)
        )[:limit]

    async def load_relationships(self, agent_id, relationship_ids):
        by_id = {item.id: item for item in self.snapshot.relationships}
        return tuple(by_id[item] for item in relationship_ids if item in by_id)

    async def search(self, request):
        raise NotImplementedError

    async def traverse(self, request):
        raise NotImplementedError


async def _backend(root: Path):
    adapter = await LocalDirectorySource(root).open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW + timedelta(seconds=1),
    )
    result = await adapter.discover(
        DiscoveryRequest(
            agent_id="agent-1",
            source_id=adapter.registration.id,
            sync_id="sync-1",
            requested_at=NOW,
        )
    )
    registration = adapter.registration
    await adapter.close()
    resource = next(
        item for item in result.snapshot.resources if item.kind is ResourceKind.FILE
    )
    return (
        LocalDirectoryReadBackend(Sources(registration), Catalog(result.snapshot)),
        registration,
        resource,
    )


async def test_file_backend_returns_bounded_prefix_and_complete_row_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    (root / "customers.csv").write_text(
        "id,name\n1,Ada\n2,Grace\n",
        encoding="utf-8",
    )
    backend, registration, resource = await _backend(root)

    result = await backend.execute_read(
        agent_id="agent-1",
        source_id=registration.id,
        resource_id=resource.id,
        max_rows=2,
        max_bytes=25,
    )

    assert result.columns == ("id", "name")
    assert result.projection.rows[0]["name"] == "Ada"
    assert result.projection.truncated is True
    assert result.artifact is not None
    artifact_rows = json.loads(result.artifact.content)
    assert artifact_rows == [
        {"id": "1", "name": "Ada"},
        {"id": "2", "name": "Grace"},
    ]
    assert artifact_rows[: result.projection.returned_rows] == [
        row.to_dict() for row in result.projection.rows
    ]


async def test_file_backend_rejects_content_drift_and_symlink_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stale"
    root.mkdir()
    target = root / "customers.json"
    target.write_text('[{"id": 1}]', encoding="utf-8")
    backend, registration, resource = await _backend(root)
    target.write_text('[{"id": 2}]', encoding="utf-8")

    with pytest.raises(LocalDirectorySourceError) as stale:
        await backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            resource_id=resource.id,
            max_rows=10,
            max_bytes=1_024,
        )
    assert stale.value.code == "catalog_file_stale"

    outside = tmp_path / "outside.json"
    outside.write_text('[{"secret": true}]', encoding="utf-8")
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(LocalDirectorySourceError) as unsafe:
        await backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            resource_id=resource.id,
            max_rows=10,
            max_bytes=1_024,
        )
    assert unsafe.value.code == "local_file_read_failed"


@pytest.mark.parametrize(
    "content",
    [
        "ID,id\n1,2\n",
        '[{"id": 1, "id": 2}]',
        '[{"id": NaN}]',
    ],
)
async def test_discovery_and_read_share_strict_schema_parser(
    tmp_path: Path,
    content: str,
) -> None:
    root = tmp_path / "invalid"
    root.mkdir()
    suffix = "csv" if content.startswith("ID") else "json"
    (root / f"invalid.{suffix}").write_text(content, encoding="utf-8")
    adapter = await LocalDirectorySource(root).open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW,
    )
    try:
        with pytest.raises(LocalDirectorySourceError) as caught:
            await adapter.discover(
                DiscoveryRequest(
                    agent_id="agent-1",
                    source_id=adapter.registration.id,
                    sync_id="sync-invalid",
                    requested_at=NOW,
                )
            )
        assert caught.value.code == "local_file_invalid"
    finally:
        await adapter.close()


async def test_file_backend_cancellation_waits_for_descriptor_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cancel-read"
    root.mkdir()
    (root / "customers.csv").write_text("id\n1\n", encoding="utf-8")
    backend, registration, resource = await _backend(root)
    original = local_files._read_registered_file
    started = threading.Event()
    release = threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError
        return original(*args, **kwargs)

    monkeypatch.setattr(local_files, "_read_registered_file", blocked)
    task = asyncio.create_task(
        backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            resource_id=resource.id,
            max_rows=10,
            max_bytes=1_024,
        )
    )
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
