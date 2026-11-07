"""An in-memory source type with one table and no SQL: only discovery and
tabular reads. It stands in for third-party source types in tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from daita.adapters.models import (
    DiscoveryRequest,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from daita.adapters.protocols import SourceClosedError, latest_snapshot
from daita.adapters.registry import SourceBackendContext, SourceOperation, SourceType
from daita.catalog.models import (
    CatalogEntry,
    CatalogFacet,
    CatalogResource,
    CatalogResourceRevision,
    CatalogTier,
    ResourceBatch,
    ResourceKind,
    Sensitivity,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from daita.domains.data.capabilities import (
    ReadTabularRequest,
    TabularFilter,
)
from daita.domains.data.results import BoundedResultProjection, project_result_rows

MEMORY_NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)

MEMORY_ADAPTER_ID = "memory-table"


def _detail(registration: SourceRegistration, sync_id: str) -> ResourceBatch:
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
    return ResourceBatch(
        tier=CatalogTier.DETAIL,
        entries=(
            CatalogEntry(
                resource=resource, revision=revision, facets=(facet,), detailed=True
            ),
        ),
        source_revision="memory:v1",
    )


class _MemoryAdapter:
    """One in-memory table; no SQL, no external I/O."""

    def __init__(self, registration: SourceRegistration) -> None:
        self._registration = registration
        self._latest: ResourceBatch | None = None
        self._closed = False

    @property
    def registration(self) -> SourceRegistration:
        return self._registration

    async def reconcile(
        self, request: DiscoveryRequest
    ) -> AsyncIterator[ResourceBatch]:
        if self._closed:
            raise SourceClosedError(self._registration.id)
        detail = _detail(self._registration, request.sync_id)
        self._latest = detail
        yield ResourceBatch(
            tier=CatalogTier.INVENTORY,
            entries=(
                request.inventory_entry(
                    kind=ResourceKind.TABLE,
                    native_identity="regions",
                    name="regions",
                    external_uri="memory://regions",
                ),
            ),
        )
        yield detail

    async def inspect(self, resource: ResourceRef) -> ResourceSnapshot:
        if self._closed:
            raise SourceClosedError(self._registration.id)
        return latest_snapshot(self._latest, resource, MEMORY_NOW)

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
    # A stored secret reference, for types that declare credential fields.
    credential_ref: str | None = None

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
                configuration={
                    "name": self.name,
                    **(
                        {}
                        if self.credential_ref is None
                        else {"credential_ref": self.credential_ref}
                    ),
                },
                attached_at=attached_at,
            )
        )


def memory_from_registration(
    registration: SourceRegistration, *, secret_provider: object
) -> MemorySource:
    del secret_provider
    credential = registration.configuration.get("credential_ref")
    return MemorySource(
        str(registration.configuration["name"]),
        None if credential is None else str(credential),
    )


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
