"""Reusable contract checks that every registered source type must satisfy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.adapters.local_workspace import LocalWorkspaceError
from daita.adapters.models import (
    DiscoveryRequest,
    ResourceRef,
    SourceProbe,
    SourceRegistration,
)
from daita.adapters.protocols import (
    ResourceAdapter,
    ResourceSource,
    SourceClosedError,
    StaleResourceError,
)
from daita.adapters.registry import SourceOperation, SourceType
from daita.catalog.models import (
    CatalogEntry,
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogTier,
    FacetKind,
    ResourceBatch,
    TabularFacet,
)
from daita.domains.data import ReadTabularRequest, TabularFilter, TabularReadBackend
from daita.domains.data.results import BoundedResultProjection
from daita.domains.data.source_files import (
    FileSearchBackend,
    FileSearchRequest,
    PathPatternQueryRequest,
    PathPatternReadBackend,
    TextReadBackend,
    TextReadRequest,
)
from daita.domains.data.sql import validate_duckdb_read
from daita.security import SecretProvider

CONTRACT_AGENT_ID = "agent-source-contract"
CONTRACT_NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _request(
    adapter: ResourceAdapter, sync_id: str, agent_home: Path | None = None
) -> DiscoveryRequest:
    return DiscoveryRequest(
        agent_id=CONTRACT_AGENT_ID,
        source_id=adapter.registration.id,
        sync_id=sync_id,
        requested_at=CONTRACT_NOW,
        agent_root=None if agent_home is None else agent_home.parent.parent,
        agent_home=agent_home,
    )


@dataclass(frozen=True)
class ReconcilePass:
    """One reconcile pass: the inventory batch, then the single detail batch."""

    inventory: ResourceBatch
    detail: ResourceBatch

    @property
    def resources(self) -> list[CatalogResource]:
        return [entry.resource for entry in self.detail.entries]

    @property
    def facets(self) -> list[CatalogFacet]:
        return [facet for entry in self.detail.entries for facet in entry.facets]

    @property
    def relationships(self) -> tuple[CatalogRelationship, ...]:
        return self.detail.relationships

    @property
    def truncated(self) -> bool:
        return self.inventory.truncated or self.detail.truncated


async def reconcile_pass(
    adapter: ResourceAdapter, request: DiscoveryRequest
) -> ReconcilePass:
    inventory, detail = [batch async for batch in adapter.reconcile(request)]
    assert inventory.tier is CatalogTier.INVENTORY
    assert detail.tier is CatalogTier.DETAIL
    return ReconcilePass(inventory, detail)


async def _reconcile(
    adapter: ResourceAdapter, request: DiscoveryRequest
) -> tuple[CatalogEntry, ...]:
    """Run one pass, check each batch against the request, return the detail."""

    batches = [batch async for batch in adapter.reconcile(request)]
    assert batches and batches[0].tier is CatalogTier.INVENTORY
    assert [b.tier for b in batches].count(CatalogTier.INVENTORY) == 1
    assert batches[-1].tier is CatalogTier.DETAIL
    inventory, *details = batches
    detailed: list[CatalogEntry] = []
    for batch in batches:
        # Reconstruction reruns the batch's own structural validation.
        assert replace(batch) == batch
        for entry in batch.entries:
            assert entry.resource.agent_id == request.agent_id
            assert entry.resource.source_id == request.source_id
        assert len(batch.relationships) <= request.max_relationships
        for entry in batch.entries:
            for facet in entry.facets:
                if facet.kind is FacetKind.TABULAR:
                    tabular = TabularFacet.from_payload(facet.payload)
                    assert len(tabular.columns) <= request.max_columns_per_resource
                    assert len(tabular.indexes) <= request.max_indexes_per_resource
    assert len(inventory.entries) <= request.max_inventory
    for batch in details:
        detailed.extend(batch.entries)
    assert len(detailed) <= request.max_resources
    assert {e.resource.id for e in detailed} <= {
        e.resource.id for e in inventory.entries
    }
    return tuple(detailed)


async def _resource_ids(
    adapter: ResourceAdapter, request: DiscoveryRequest
) -> set[str]:
    return {entry.resource.id for entry in await _reconcile(adapter, request)}


async def assert_source_type_contract(
    source_type: SourceType,
    source: ResourceSource,
    *,
    secret_provider: SecretProvider,
    agent_home: Path | None = None,
) -> None:
    """Open, reconcile, inspect, health-check, close and rebuild one source.

    A filesystem source receives ``agent_home`` as the private state it must
    never catalog.
    """

    adapter = await source.open(
        agent_id=CONTRACT_AGENT_ID,
        attached_at=CONTRACT_NOW,
        clock=lambda: CONTRACT_NOW,
    )
    registration = adapter.registration
    try:
        assert registration.adapter_id == source_type.adapter_id

        first_request = _request(adapter, "sync-contract-first", agent_home)
        first = await _reconcile(adapter, first_request)
        assert first, "the contract needs at least one resource"
        second_request = _request(adapter, "sync-contract-second", agent_home)
        assert await _resource_ids(adapter, second_request) == {
            item.resource.id for item in first
        }

        resource = first[0].resource
        with pytest.raises(StaleResourceError):
            await adapter.inspect(
                ResourceRef(
                    agent_id=CONTRACT_AGENT_ID,
                    source_id=registration.id,
                    resource_id=resource.id,
                    native_identity=resource.native_identity,
                    kind=resource.kind,
                    revision="sha256:" + "0" * 64,
                )
            )

        health = await adapter.health()
        assert (health.source_id, health.adapter_id) == (
            registration.id,
            registration.adapter_id,
        )
        # What a person sees at attach: the role, read-only status and warnings.
        assert health.healthy
        assert isinstance(health.details.get("role", ""), str)
        assert isinstance(health.details.get("read_only", True), bool)
        assert all(isinstance(item, str) for item in health.details.get("warnings", ()))  # type: ignore[attr-defined]

        rebuilt = source_type.from_registration(
            registration, secret_provider=secret_provider
        )
        rebuilt_adapter = await rebuilt.open(
            agent_id=CONTRACT_AGENT_ID,
            attached_at=CONTRACT_NOW,
            clock=lambda: CONTRACT_NOW,
        )
        try:
            assert rebuilt_adapter.registration == registration
        finally:
            await rebuilt_adapter.close()
    finally:
        await adapter.close()

    await adapter.close()
    with pytest.raises(SourceClosedError):
        await _reconcile(adapter, _request(adapter, "sync-contract-closed", agent_home))
    with pytest.raises(SourceClosedError):
        await adapter.inspect(
            ResourceRef(
                agent_id=CONTRACT_AGENT_ID,
                source_id=registration.id,
                resource_id=resource.id,
                native_identity=resource.native_identity,
                kind=resource.kind,
            )
        )


def _rows(projection: BoundedResultProjection) -> list[str]:
    """Rows as an order-free multiset of canonical JSON text."""

    return sorted(canonical_json(row.to_dict()) for row in projection.rows)


async def assert_tabular_read_contract(
    agent: Agent,
    *,
    source_id: str,
    resource_id: str,
    columns: tuple[str, ...],
    key_column: str,
    range_column: str,
) -> None:
    """Read one attached resource through its type's TABULAR_READ backend.

    The resource needs at least three rows, a ``key_column`` and a numeric
    ``range_column``. Expected filtered rows are computed here from the
    unfiltered read, independently of the backend's implementation.
    """

    (registration,) = [
        item for item in await agent.list_sources() if item.id == source_id
    ]
    backends = agent._embedded._source_backends[SourceOperation.TABULAR_READ]
    backend = cast(TabularReadBackend, backends[registration.adapter_id])
    (resource,) = [
        item
        for item in await agent.list_catalog_resources(source_id=source_id)
        if item.id == resource_id
    ]

    async def read(
        *filters: TabularFilter,
        limit: int = 10_000,
        max_bytes: int = 4 * 1_024 * 1_024,
    ) -> BoundedResultProjection:
        projection = await backend.read_tabular(
            ReadTabularRequest(
                agent_id=agent.id,
                source_id=source_id,
                resource_id=resource_id,
                columns=columns,
                filters=filters,
                limit=limit,
                max_bytes=max_bytes,
                timeout_seconds=5.0,
            )
        )
        assert isinstance(projection, BoundedResultProjection)
        assert all(set(row) == set(columns) for row in projection.rows)
        return projection

    full = await read()
    assert not full.truncated
    rows = [cast(Mapping[str, object], row.to_dict()) for row in full.rows]
    assert len(rows) >= 3, "the tabular contract needs at least three rows"

    def expect(predicate: Callable[[Mapping[str, object]], bool]) -> list[str]:
        return sorted(canonical_json(row) for row in rows if predicate(row))

    key = rows[0][key_column]
    assert _rows(await read(TabularFilter(key_column, "eq", key))) == expect(
        lambda row: row[key_column] == key
    )
    keys = (key, next(row[key_column] for row in rows if row[key_column] != key))
    assert _rows(await read(TabularFilter(key_column, "in", keys))) == expect(
        lambda row: row[key_column] in keys
    )
    values = sorted(cast(float, row[range_column]) for row in rows)
    low, high = values[len(values) // 2], values[-1]
    assert _rows(
        await read(
            TabularFilter(range_column, "gte", low),
            TabularFilter(range_column, "lt", high),
        )
    ) == expect(lambda row: low <= cast(float, row[range_column]) < high)
    assert _rows(await read(TabularFilter(range_column, "gt", low))) == expect(
        lambda row: cast(float, row[range_column]) > low
    )
    assert _rows(await read(TabularFilter(range_column, "lte", low))) == expect(
        lambda row: cast(float, row[range_column]) <= low
    )

    by_rows = await read(limit=len(rows) - 1)
    assert by_rows.truncated and "row_limit" in by_rows.truncation_reasons
    by_bytes = await read(max_bytes=2)
    assert by_bytes.truncated and "byte_limit" in by_bytes.truncation_reasons


async def assert_live_source_contract(
    agent: Agent,
    *,
    source_id: str,
    text_path: str,
    text: str,
    dataset_pattern: str,
) -> None:
    """Search, read and query one live uncataloged source through its type.

    ``text_path`` holds ``text`` (longer than one chunk is not required), and
    ``dataset_pattern`` matches a homogeneous structured dataset.
    """

    adapter_id = await agent._embedded._data_view.source_adapter_id(agent.id, source_id)
    assert adapter_id is not None
    source_type = agent._embedded._source_types.require(adapter_id)
    assert {
        SourceOperation.FILE_SEARCH,
        SourceOperation.TEXT_READ,
        SourceOperation.SQL_READ,
    } <= source_type.operations
    backends = agent._embedded._source_backends
    search = cast(FileSearchBackend, backends[SourceOperation.FILE_SEARCH][adapter_id])
    read = cast(TextReadBackend, backends[SourceOperation.TEXT_READ][adapter_id])
    query = cast(
        PathPatternReadBackend, backends[SourceOperation.PATH_QUERY][adapter_id]
    )
    assert source_id not in {item.id for item in await agent.list_sources()}

    found = await search.search_files(
        FileSearchRequest(
            agent_id=agent.id,
            run_id="contract-run",
            source_id=source_id,
            query=PurePosixPath(text_path).name,
            path=None,
            paths=None,
            mode="paths",
            glob=None,
            order_by="path",
            resource_ids=None,
        )
    )
    assert found.data["source_id"] == source_id
    assert any(
        item["path"].endswith(text_path)
        for item in cast(tuple[Mapping[str, str], ...], found.data["matches"])
    )
    assert found.sensitivity is not None and found.provenance

    chunk = await read.read_text(
        TextReadRequest(
            agent_id=agent.id,
            run_id="contract-run",
            source_id=source_id,
            path=text_path,
            cursor=None,
            position=None,
            resource_ids=None,
        )
    )
    assert chunk.data["content"] == text and chunk.data["complete"] is True
    assert isinstance(chunk.data["binding"], str)
    with pytest.raises(LocalWorkspaceError) as misused:
        await read.read_text(
            TextReadRequest(
                agent_id=agent.id,
                run_id="contract-run",
                source_id=source_id,
                path=None,
                cursor=cast(str, chunk.data["binding"]),
                position=None,
                resource_ids=None,
            )
        )
    assert misused.value.code == "cursor_invalid"

    validated = validate_duckdb_read("SELECT COUNT(*) AS n FROM data")
    rows = await query.query_path_pattern(
        PathPatternQueryRequest(
            agent_id=agent.id,
            run_id="contract-run",
            source_id=source_id,
            path_pattern=dataset_pattern,
            canonical_sql=validated.canonical_sql,
            sql_fingerprint=validated.sql_fingerprint,
            resource_ids=None,
        )
    )
    assert rows.data["columns"] == ("n",)
    assert rows.data["source_id"] == source_id


class StaticSources:
    """A source store holding one registration, enough to build a probe backend."""

    def __init__(self, registration: SourceRegistration) -> None:
        self._registration = registration

    async def load_source(
        self, agent_id: str, source_id: str
    ) -> SourceRegistration | None:
        registration = self._registration
        return (
            registration
            if (agent_id, source_id)
            == (
                registration.agent_id,
                registration.id,
            )
            else None
        )

    async def load_source_read_scope(self, agent_id: str, source_id: str) -> None:
        return None

    async def register_source(
        self, registration: SourceRegistration
    ) -> SourceRegistration:
        raise NotImplementedError

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        return (self._registration,)

    async def detach_source(
        self, agent_id: str, source_id: str, detached_at: datetime
    ) -> SourceRegistration:
        raise NotImplementedError


@dataclass(frozen=True)
class ProbeMutation:
    """One scripted change to a source and the scopes whose token must move.

    ``exact`` also requires the catalog to change in those scopes and nowhere
    else; a mutation a probe may treat as a change the catalog never shows
    (a file replaced by an identical one) sets it False.
    """

    name: str
    apply: Callable[[], Awaitable[None]]
    moves: frozenset[str] = frozenset()
    exact: bool = True


async def assert_change_detection_contract(
    probe: Callable[[], Awaitable[SourceProbe]],
    mutations: Sequence[ProbeMutation],
    *,
    revisions: Callable[[], Awaitable[Mapping[str, Mapping[str, str]]]],
) -> None:
    """A probe moves exactly with what a catalog read would record.

    ``revisions`` reads the catalog now: scope -> resource -> revision. Before
    each mutation an unchanged source must give the same tokens twice. After
    it, the scopes whose token moved must be the ones expected, and the catalog
    must change in no scope the probe left still (a miss); with ``exact`` it
    must also change in every scope the probe moved.
    """

    for mutation in mutations:
        before, again = await probe(), await probe()
        assert before.tokens == again.tokens, mutation.name
        catalog_before = await revisions()
        await mutation.apply()
        after = await probe()
        moved = {
            scope
            for scope in {*before.tokens, *after.tokens}
            if before.tokens.get(scope) != after.tokens.get(scope)
        }
        assert moved == mutation.moves, mutation.name
        catalog_after = await revisions()
        changed = {
            scope
            for scope in {*catalog_before, *catalog_after}
            if catalog_before.get(scope) != catalog_after.get(scope)
        }
        assert changed <= moved, f"{mutation.name}: the probe missed {changed - moved}"
        if mutation.exact:
            assert changed == moved, f"{mutation.name}: moved without a change"


async def scoped_revisions(
    adapter: ResourceAdapter,
    request: DiscoveryRequest,
    scope_of: Callable[[CatalogEntry], str],
) -> dict[str, dict[str, str]]:
    """What a full read records now, per scope: resource -> revision."""

    inventory, detail = [batch async for batch in adapter.reconcile(request)]
    result: dict[str, dict[str, str]] = {}
    for entry in (*inventory.entries, *detail.entries):
        if entry.detailed:
            result.setdefault(scope_of(entry), {})[
                entry.resource.native_identity
            ] = entry.revision.revision
        else:
            result.setdefault(scope_of(entry), {}).setdefault(
                entry.resource.native_identity, "inventory"
            )
    return result


__all__ = [
    "CONTRACT_AGENT_ID",
    "ProbeMutation",
    "StaticSources",
    "assert_change_detection_contract",
    "scoped_revisions",
    "ReconcilePass",
    "reconcile_pass",
    "CONTRACT_NOW",
    "assert_live_source_contract",
    "assert_source_type_contract",
    "assert_tabular_read_contract",
]
