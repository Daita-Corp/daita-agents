"""Folder sources catalog, contain and read local files through one backend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import duckdb
import pytest

from daita import Agent, FolderSource, LocalFileAccess, LocalWorkspace
from daita.adapters import folder as folder_module
from daita.adapters.folder import FolderSourceError
from daita.adapters.local_workspace import LocalWorkspaceError
from daita.adapters.models import DiscoveryRequest
from daita.adapters.protocols import ResourceAdapterError
from daita.adapters.registry import SourceOperation, SourceTypeRegistry
from daita.catalog import CatalogSearchRequest
from daita.catalog.models import FacetKind, RelationshipKind, ResourceKind
from daita.domains.data.source_files import (
    FileSearchBackend,
    FileSearchRequest,
    TextReadBackend,
    TextReadRequest,
)
from daita.security import EmptySecretProvider
from tests.support.source_contract import (
    assert_live_source_contract,
    assert_source_type_contract,
)
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _sales_folder(root: Path) -> Path:
    folder = root / "data"
    (folder / "sales" / "2026").mkdir(parents=True)
    (folder / "sales" / "2026" / "orders.csv").write_text(
        "order_id,region_code,amount\n1,N,10\n2,S,20\n3,N,5\n", encoding="utf-8"
    )
    (folder / "notes.md").write_text("# Sales notes\nQ3 was strong.\n")
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            "COPY (SELECT * FROM VALUES ('N', 'north'), ('S', 'south') "
            "regions(region_code, region_name)) TO ? (FORMAT PARQUET)",
            [str(folder / "sales" / "regions.parquet")],
        )
    finally:
        connection.close()
    return folder


async def _agent(tmp_path: Path, name: str) -> Agent:
    state = tmp_path / "state"
    return await Agent.create(name, root=state, workspace=workspace_for(state))


async def _search(
    agent: Agent, source_id: str, query: str, *, mode: str = "paths"
) -> tuple[Mapping[str, object], ...]:
    backend = cast(
        FileSearchBackend,
        agent._embedded._source_backends[SourceOperation.FILE_SEARCH]["folder"],
    )
    resource_ids = await agent._embedded._data_view.readable_resource_ids(
        agent.id, (source_id,)
    )
    result = await backend.search_files(
        FileSearchRequest(
            agent_id=agent.id,
            run_id="folder-test",
            source_id=source_id,
            query=query,
            path=None,
            paths=None,
            mode=mode,
            glob=None,
            order_by="path",
            resource_ids=resource_ids,
        )
    )
    return cast(tuple[Mapping[str, object], ...], result.data["matches"])


async def _read(agent: Agent, source_id: str, path: str) -> Mapping[str, object]:
    backend = cast(
        TextReadBackend,
        agent._embedded._source_backends[SourceOperation.TEXT_READ]["folder"],
    )
    resource_ids = await agent._embedded._data_view.readable_resource_ids(
        agent.id, (source_id,)
    )
    result = await backend.read_text(
        TextReadRequest(
            agent_id=agent.id,
            run_id="folder-test",
            source_id=source_id,
            path=path,
            cursor=None,
            position=None,
            resource_ids=resource_ids,
        )
    )
    return result.data


async def test_attached_folder_catalogs_folders_files_and_tabular_schemas(
    tmp_path: Path,
) -> None:
    folder = _sales_folder(tmp_path)
    agent = await _agent(tmp_path, "folder-catalog")
    try:
        source = await agent.attach(FolderSource(folder, name="Sales files"))
        resources = {
            item.native_identity: item
            for item in await agent.list_catalog_resources(source_id=source.id)
        }
        assert {path: item.kind for path, item in resources.items()} == {
            ".": ResourceKind.FOLDER,
            "sales": ResourceKind.FOLDER,
            "sales/2026": ResourceKind.FOLDER,
            "sales/2026/orders.csv": ResourceKind.FILE,
            "sales/regions.parquet": ResourceKind.FILE,
        }
        store = agent._embedded._store
        contains = {
            (relationship.from_resource_id, relationship.to_resource_id)
            for path in ("sales", "sales/2026")
            for relationship in await store.load_incident_relationships(
                agent.id,
                resources[path].id,
                relationship_kinds=(RelationshipKind.CONTAINS,),
            )
        }
        assert (
            resources["sales/2026"].id,
            resources["sales/2026/orders.csv"].id,
        ) in contains
        assert (
            resources["sales"].id,
            resources["sales/regions.parquet"].id,
        ) in contains

        def facets(path: str) -> dict[FacetKind, Mapping[str, object]]:
            return {
                facet.kind: facet.payload for facet in facet_records[resources[path].id]
            }

        facet_records = {
            item.id: await store.load_facets(agent.id, item.id, item.current_revision)
            for item in resources.values()
        }
        orders = facets("sales/2026/orders.csv")
        assert orders[FacetKind.FILE]["format"] == "csv"
        columns = cast(
            tuple[Mapping[str, object], ...], orders[FacetKind.TABULAR]["columns"]
        )
        assert [column["name"] for column in columns] == [
            "order_id",
            "region_code",
            "amount",
        ]
        regions = facets("sales/regions.parquet")
        assert regions[FacetKind.FILE]["format"] == "parquet"
        assert FacetKind.TABULAR in regions
        # notes.md is not a data file: an attached folder neither catalogs nor
        # reads it; the live session folder still can.
        assert "notes.md" not in resources
        with pytest.raises(FolderSourceError) as refused:
            await _read(agent, source.id, "notes.md")
        assert refused.value.error_code == "resource_read_not_allowed"

        search = await agent._embedded._data_view.search(
            CatalogSearchRequest(agent_id=agent.id, query="region_code")
        )
        assert {
            resources["sales/2026/orders.csv"].id,
            resources["sales/regions.parquet"].id,
        } <= {hit.resource_id for hit in search.hits}
        sync = await store.load_sync(agent.id, resources["."].current_sync_id)
        assert sync is not None and sync.truncated is False
    finally:
        await agent.close()


@pytest.mark.contract
async def test_attached_folder_type_satisfies_the_source_contract(
    tmp_path: Path,
) -> None:
    folder = _sales_folder(tmp_path)
    agent = await _agent(tmp_path, "folder-contract")
    try:
        await assert_source_type_contract(
            SourceTypeRegistry().require("folder"),
            FolderSource(folder, name="Contract"),
            secret_provider=EmptySecretProvider(),
            agent_home=agent.home,
        )
    finally:
        await agent.close()


@pytest.mark.contract
async def test_session_folder_source_satisfies_the_live_source_contract(
    tmp_path: Path,
) -> None:
    workspace = workspace_for(tmp_path / "state")
    (workspace.root / "notes.txt").write_text("session sentinel", encoding="utf-8")
    (workspace.root / "parts").mkdir()
    for index in range(3):
        (workspace.root / "parts" / f"p{index}.csv").write_text(
            f"id,total\n{index},{index * 10}\n", encoding="utf-8"
        )
    agent = await Agent.create(
        "session-contract", root=tmp_path / "state", workspace=workspace
    )
    try:
        assert agent.session_source_id is not None
        await assert_live_source_contract(
            agent,
            source_id=agent.session_source_id,
            text_path="notes.txt",
            text="session sentinel",
            dataset_pattern="parts/*.csv",
        )
    finally:
        await agent.close()


def _contained_case(folder: Path, case: str) -> str:
    """Create one unsafe entry; return the root-relative path it would have."""

    if case == "symlink":
        outside = folder.parent / "outside.csv"
        outside.write_text("id,note\n1,outside sentinel\n", encoding="utf-8")
        (folder / "link.csv").symlink_to(outside)
        return "link.csv"
    if case == "parent_traversal":
        (folder.parent / "outside.csv").write_text("id,note\n1,outside sentinel\n")
        return "../outside.csv"
    if case == "secret_pattern":
        (folder / "server.pem").write_text("outside sentinel")
        return "server.pem"
    if case == "special_file":
        os.mkfifo(folder / "pipe.csv")
        return "pipe.csv"
    assert case == "private_daita_state"
    (folder / ".daita").mkdir()
    (folder / ".daita" / "memory.csv").write_text("id,note\n1,outside sentinel\n")
    return ".daita/memory.csv"


@pytest.mark.parametrize(
    "case",
    (
        "symlink",
        "parent_traversal",
        "secret_pattern",
        "special_file",
        "private_daita_state",
    ),
)
async def test_unsafe_entries_are_never_cataloged_searched_or_read(
    tmp_path: Path, case: str
) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "safe.csv").write_text("id,note\n1,safe sentinel\n", encoding="utf-8")
    unsafe = _contained_case(folder, case)
    agent = await _agent(tmp_path, f"folder-{case.replace('_', '-')}")
    try:
        source = await agent.attach(FolderSource(folder))
        cataloged = {
            item.native_identity
            for item in await agent.list_catalog_resources(source_id=source.id)
        }
        assert cataloged == {".", "safe.csv"}
        for mode in ("paths", "content"):
            matches = await _search(agent, source.id, "sentinel", mode=mode)
            assert {item["path"] for item in matches} <= {"safe.csv"}
        # Only readable cataloged files are reachable, before any file I/O.
        with pytest.raises(FolderSourceError) as refused:
            await _read(agent, source.id, unsafe)
        assert refused.value.error_code == "resource_read_not_allowed"
        assert (await _read(agent, source.id, "safe.csv"))["content"] == (
            "id,note\n1,safe sentinel\n"
        )
    finally:
        await agent.close()


async def test_attached_file_changed_since_refresh_is_never_searched_or_read(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    orders = folder / "orders.csv"
    # Larger than one read chunk, so the first read returns a cursor.
    orders.write_text(
        "id,note\n" + "".join(f"{index},needle {index}\n" for index in range(6_000)),
        encoding="utf-8",
    )
    agent = await _agent(tmp_path, "folder-changed")
    try:
        source = await agent.attach(FolderSource(folder))
        first = await _read(agent, source.id, "orders.csv")
        cursor = first["cursor"]
        assert isinstance(cursor, str)
        assert await _search(agent, source.id, "needle", mode="content")

        orders.write_text("id,note\n1,needle changed\n", encoding="utf-8")
        for mode in ("paths", "content"):
            assert await _search(agent, source.id, "needle", mode=mode) == ()
        with pytest.raises(FolderSourceError) as by_path:
            await _read(agent, source.id, "orders.csv")
        assert by_path.value.error_code == "query_binding_stale"
        backend = cast(
            TextReadBackend,
            agent._embedded._source_backends[SourceOperation.TEXT_READ]["folder"],
        )
        with pytest.raises(FolderSourceError) as by_cursor:
            await backend.read_text(
                TextReadRequest(
                    agent_id=agent.id,
                    run_id="folder-test",
                    source_id=source.id,
                    path=None,
                    cursor=cursor,
                    position=None,
                    resource_ids=await agent._embedded._data_view.readable_resource_ids(
                        agent.id, (source.id,)
                    ),
                )
            )
        assert by_cursor.value.error_code == "query_binding_stale"
    finally:
        await agent.close()


@pytest.mark.parametrize("root", ["state_parent", "agent_home"])
async def test_folder_overlapping_private_agent_state_cannot_be_attached(
    tmp_path: Path, root: str
) -> None:
    agent = await _agent(tmp_path, "folder-state-overlap")
    try:
        folder = tmp_path if root == "state_parent" else agent.home
        with pytest.raises(ResourceAdapterError) as refused:
            await agent.attach(FolderSource(folder))
        assert refused.value.code == "workspace_state_overlap"
        # The filesystem root is refused before any folder is opened.
        with pytest.raises(ValueError, match="filesystem root"):
            await agent.attach(FolderSource(Path("/")))
        assert await agent.list_sources() == ()
    finally:
        await agent.close()


async def test_discovery_beyond_max_resources_is_bounded_and_marked_truncated(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    for index in range(12):
        (folder / f"part-{index:02}.csv").write_text(f"id\n{index}\n")
    agent = await _agent(tmp_path, "folder-truncation")
    try:
        adapter = await FolderSource(folder).open(
            agent_id=agent.id, attached_at=_NOW, clock=lambda: _NOW
        )
        try:
            result = await adapter.discover(
                DiscoveryRequest(
                    agent_id=agent.id,
                    source_id=adapter.registration.id,
                    sync_id="sync-bounded",
                    requested_at=_NOW,
                    max_resources=5,
                    agent_root=agent.home.parent.parent,
                    agent_home=agent.home,
                )
            )
        finally:
            await adapter.close()
    finally:
        await agent.close()
    snapshot = result.snapshot
    assert snapshot.sync.truncated is True
    assert len(snapshot.resources) == 5
    assert [item.native_identity for item in snapshot.resources] == [
        ".",
        "part-00.csv",
        "part-01.csv",
        "part-02.csv",
        "part-03.csv",
    ]


async def test_discovery_out_of_schema_budget_is_marked_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "orders.csv").write_text("id\n1\n")
    clock = iter((0.0, 1_000.0))
    # The schema budget is spent before the first batch is described.
    monkeypatch.setattr(
        folder_module, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    agent = await _agent(tmp_path, "folder-schema-budget")
    try:
        adapter = await FolderSource(folder).open(
            agent_id=agent.id, attached_at=_NOW, clock=lambda: _NOW
        )
        try:
            result = await adapter.discover(
                DiscoveryRequest(
                    agent_id=agent.id,
                    source_id=adapter.registration.id,
                    sync_id="sync-schema-budget",
                    requested_at=_NOW,
                    agent_root=agent.home.parent.parent,
                    agent_home=agent.home,
                )
            )
        finally:
            await adapter.close()
    finally:
        await agent.close()
    snapshot = result.snapshot
    assert snapshot.sync.truncated is True
    assert {item.native_identity for item in snapshot.resources} == {".", "orders.csv"}
    assert {facet.kind for facet in snapshot.facets} == {FacetKind.FILE}


async def _session_search(
    agent: Agent,
    *,
    query: str | None = None,
    paths: tuple[str, ...] | None = None,
    mode: str = "paths",
    glob: str | None = None,
    order_by: str = "path",
) -> Mapping[str, object]:
    backend = cast(
        FileSearchBackend,
        agent._embedded._source_backends[SourceOperation.FILE_SEARCH]["folder"],
    )
    result = await backend.search_files(
        FileSearchRequest(
            agent_id=agent.id,
            run_id="session-test",
            source_id=cast(str, agent.session_source_id),
            query=query,
            path=None,
            paths=paths,
            mode=mode,
            glob=glob,
            order_by=order_by,
            resource_ids=None,
        )
    )
    return result.data


async def _session_read(
    agent: Agent,
    *,
    path: str | None = None,
    cursor: str | None = None,
    position: str | None = None,
) -> Mapping[str, object]:
    backend = cast(
        TextReadBackend,
        agent._embedded._source_backends[SourceOperation.TEXT_READ]["folder"],
    )
    result = await backend.read_text(
        TextReadRequest(
            agent_id=agent.id,
            run_id="session-test",
            source_id=cast(str, agent.session_source_id),
            path=path,
            cursor=cursor,
            position=position,
            resource_ids=None,
        )
    )
    return result.data


async def test_session_folder_search_modes_glob_order_and_multi_root_coverage(
    tmp_path: Path,
) -> None:
    workspace = workspace_for(tmp_path / "state")
    (workspace.root / "a").mkdir()
    (workspace.root / "b").mkdir()
    (workspace.root / "a" / "notes.txt").write_text("first line\nneedle here\n")
    (workspace.root / "b" / "needle-data.csv").write_text("id\n1\n")
    os.utime(workspace.root / "a" / "notes.txt", (1_700_000_000, 1_700_000_000))
    os.utime(workspace.root / "b" / "needle-data.csv", (1_800_000_000, 1_800_000_000))
    agent = await Agent.create(
        "session-search", root=tmp_path / "state", workspace=workspace
    )
    try:

        def found(data: Mapping[str, object]) -> set[tuple[object, object]]:
            matches = cast(tuple[Mapping[str, object], ...], data["matches"])
            return {(item["path"], item["match_kind"]) for item in matches}

        assert found(await _session_search(agent, query="needle")) == {
            ("b/needle-data.csv", "path")
        }
        assert found(await _session_search(agent, query="needle", mode="content")) == {
            ("a/notes.txt", "content")
        }
        assert found(await _session_search(agent, query="needle", mode="both")) == {
            ("b/needle-data.csv", "path"),
            ("a/notes.txt", "content"),
        }
        assert found(await _session_search(agent, glob="*.csv")) == {
            ("b/needle-data.csv", "path")
        }
        latest = await _session_search(agent, glob="*", order_by="modified_desc")
        assert [
            item["path"]
            for item in cast(tuple[Mapping[str, object], ...], latest["matches"])
        ] == ["b/needle-data.csv", "a/notes.txt"]
        multi = await _session_search(agent, glob="*", paths=("a", "b", "missing"))
        coverage = {
            item["path"]: item["status"]
            for item in cast(tuple[Mapping[str, object], ...], multi["coverage"])
        }
        assert coverage == {"a": "complete", "b": "complete", "missing": "failed"}
        assert multi["scan_complete"] is False
    finally:
        await agent.close()


async def test_session_folder_reads_chunks_tails_and_cursors(tmp_path: Path) -> None:
    workspace = workspace_for(tmp_path / "state")
    lines = "".join(f"line {index:05d}\n" for index in range(6_000))
    (workspace.root / "big.log").write_text(lines, encoding="utf-8")
    agent = await Agent.create(
        "session-read", root=tmp_path / "state", workspace=workspace
    )
    try:
        first = await _session_read(agent, path="big.log")
        assert first["complete"] is False and first["start_offset"] == 0
        assert isinstance(first["cursor"], str)
        second = await _session_read(agent, cursor=cast(str, first["cursor"]))
        assert second["start_offset"] == first["end_offset"]
        tail = await _session_read(agent, path="big.log", position="end")
        assert cast(str, tail["content"]).endswith("line 05999\n")
        assert tail["end_offset"] == len(lines.encode())
    finally:
        await agent.close()


async def test_session_folder_access_follows_the_workspace_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = (tmp_path / "home").resolve()
    (home / "project").mkdir(parents=True)
    (home / "notes.txt").write_text("home sentinel", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside sentinel", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    computer = await Agent.create(
        "session-computer",
        root=tmp_path / "state",
        workspace=LocalWorkspace(home / "project", access=LocalFileAccess.COMPUTER),
    )
    try:
        assert (await _session_read(computer, path="~/notes.txt"))["content"] == (
            "home sentinel"
        )
        assert (await _session_read(computer, path=str(outside)))["content"] == (
            "outside sentinel"
        )
    finally:
        await computer.close()
    contained = await Agent.create(
        "session-contained",
        root=tmp_path / "state",
        workspace=LocalWorkspace(home / "project"),
    )
    try:
        # A contained workspace takes only workspace-relative paths; "~" is an
        # ordinary relative name there, never the user's home.
        for path, code in (
            (str(outside), "path_invalid"),
            ("../notes.txt", "path_invalid"),
            ("~/notes.txt", "path_not_found"),
        ):
            with pytest.raises(LocalWorkspaceError) as refused:
                await _session_read(contained, path=path)
            assert refused.value.code == code
    finally:
        await contained.close()
