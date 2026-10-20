"""An in-memory source type with one table and no SQL: only discovery and
tabular reads. It stands in for third-party source types in tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from daita.adapters.models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from daita.adapters.protocols import SourceClosedError, StaleResourceError
from daita.adapters.registry import SourceBackendContext, SourceOperation, SourceType
from daita.catalog.models import (
    CatalogFacet,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from daita.domains.data.capabilities import (
    DataQueryError,
    ReadTabularRequest,
    TabularFilter,
)
from daita.domains.data.results import BoundedResultProjection, project_result_rows

MEMORY_NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)

MEMORY_ADAPTER_ID = "memory-table"


def _snapshot(registration: SourceRegistration, sync_id: str) -> SourceCatalogSnapshot:
    resource_id = catalog_resource_id(registration.id, ResourceKind.TABLE, "regions")
    facet = CatalogFacet.from_tabular(
        resource_id=resource_id,
        sync_id=sync_id,
        observed_at=MEMORY_NOW,
        facet=TabularFacet(
            columns=(
                TabularColumn(
                    name="region", native_type="text", ordinal=0, nullable=False
                ),
                TabularColumn(
                    name="revenue", native_type="integer", ordinal=1, nullable=False
                ),
            )
        ),
    )
    revision = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id=sync_id,
        observed_at=MEMORY_NOW,
        facet_revisions=(facet.revision,),
        source_revision="memory:v1",
    )
    resource = CatalogResource.build(
        agent_id=registration.agent_id,
        source_id=registration.id,
        native_identity="regions",
        external_uri="memory://regions",
        kind=ResourceKind.TABLE,
        name="regions",
        sensitivity=Sensitivity.INTERNAL,
        revision=revision,
        first_observed_at=MEMORY_NOW,
        last_observed_at=MEMORY_NOW,
    )
    return SourceCatalogSnapshot(
        sync=CatalogSync(
            id=sync_id,
            agent_id=registration.agent_id,
            source_id=registration.id,
            adapter_id=MEMORY_ADAPTER_ID,
            status=CatalogSyncStatus.SUCCEEDED,
            started_at=MEMORY_NOW,
            completed_at=MEMORY_NOW,
            source_revision="memory:v1",
            resource_count=1,
        ),
        resources=(resource,),
        revisions=(revision,),
        facets=(facet,),
    )


class _MemoryAdapter:
    """One in-memory table; no SQL, no external I/O."""

    def __init__(self, registration: SourceRegistration) -> None:
        self._registration = registration
        self._latest: SourceCatalogSnapshot | None = None
        self._closed = False

    @property
    def registration(self) -> SourceRegistration:
        return self._registration

    async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
        if self._closed:
            raise SourceClosedError(self._registration.id)
        self._latest = _snapshot(self._registration, request.sync_id)
        return DiscoveryResult(
            request=request, snapshot=self._latest, completed_at=MEMORY_NOW
        )

    async def inspect(self, resource: ResourceRef) -> ResourceSnapshot:
        if self._closed:
            raise SourceClosedError(self._registration.id)
        assert self._latest is not None
        current = self._latest.resources[0]
        if (
            resource.revision is not None
            and resource.revision != current.current_revision
        ):
            raise StaleResourceError(self._registration.id, resource.resource_id)
        return ResourceSnapshot(
            reference=resource,
            resource=current,
            revision=self._latest.revisions[0],
            facets=self._latest.facets,
            relationships=(),
            inspected_at=MEMORY_NOW,
            source_revision="memory:v1",
        )

    async def health(self) -> SourceHealth:
        return SourceHealth(
            agent_id=self._registration.agent_id,
            source_id=self._registration.id,
            adapter_id=MEMORY_ADAPTER_ID,
            healthy=not self._closed,
            checked_at=MEMORY_NOW,
        )

    async def close(self) -> None:
        self._closed = True


@dataclass(frozen=True, slots=True)
class MemorySource:
    name: str

    async def open(
        self,
        *,
        agent_id: str,
        attached_at: datetime,
        clock: Callable[[], datetime],
    ) -> _MemoryAdapter:
        del clock
        return _MemoryAdapter(
            SourceRegistration.build(
                agent_id=agent_id,
                adapter_id=MEMORY_ADAPTER_ID,
                native_identity=f"memory:{self.name}",
                display_name=self.name,
                configuration={"name": self.name},
                attached_at=attached_at,
            )
        )


def memory_from_registration(
    registration: SourceRegistration, *, secret_provider: object
) -> MemorySource:
    del secret_provider
    return MemorySource(str(registration.configuration["name"]))


REGION_ROWS = (
    {"region": "north", "revenue": 15},
    {"region": "south", "revenue": 7},
    {"region": "east", "revenue": 15},
    {"region": "west", "revenue": 3},
)
_COMPARE: dict[str, Callable[[object, object], bool]] = {
    "eq": lambda value, bound: value == bound,
    "in": lambda value, bound: value in cast(tuple[object, ...], bound),
    "lt": lambda value, bound: cast(int, value) < cast(int, bound),
    "lte": lambda value, bound: cast(int, value) <= cast(int, bound),
    "gt": lambda value, bound: cast(int, value) > cast(int, bound),
    "gte": lambda value, bound: cast(int, value) >= cast(int, bound),
}


class _MemoryTabularBackend:
    """Read the in-memory table's rows; the type has no SQL at all."""

    def __init__(self, context: SourceBackendContext) -> None:
        self._catalog = context.catalog
        self.requests: list[ReadTabularRequest] = []

    async def read_tabular(
        self, request: ReadTabularRequest
    ) -> BoundedResultProjection:
        self.requests.append(request)
        current = {
            item.resource_id: item.revision
            for item in await self._catalog.resource_schemas(
                request.agent_id, request.source_id
            )
        }
        if current.get(request.resource_id) != request.expected_revision:
            raise DataQueryError(
                "query_binding_stale", "The table changed since validation."
            )

        def matches(row: Mapping[str, object], item: TabularFilter) -> bool:
            return _COMPARE[item.operator](row[item.column], item.value)

        return project_result_rows(
            tuple(
                {column: row[column] for column in request.columns}
                for row in REGION_ROWS
                if all(matches(row, item) for item in request.filters)
            ),
            max_rows=request.limit,
            max_bytes=request.max_bytes,
        )


MEMORY_TYPE = SourceType(
    adapter_id=MEMORY_ADAPTER_ID,
    label="In-memory table",
    from_registration=memory_from_registration,
    operations=frozenset({SourceOperation.TABULAR_READ}),
    create_backends=lambda context: {
        SourceOperation.TABULAR_READ: _MemoryTabularBackend(context)
    },
)


__all__ = [
    "MEMORY_ADAPTER_ID",
    "MEMORY_NOW",
    "MEMORY_TYPE",
    "REGION_ROWS",
    "MemorySource",
    "memory_from_registration",
]
