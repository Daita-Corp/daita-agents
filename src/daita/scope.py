"""Resolve explicit source/resource candidates once and narrow them at call time."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .loop.models import RunInput


@dataclass(frozen=True, slots=True)
class EffectiveSourceScope:
    """Resolved admission, where empty sets always mean no access."""

    source_ids: frozenset[str]
    resource_ids: frozenset[str]
    # source ID -> (display name, adapter id): the model addresses a source by
    # its display name, and stored transcripts resolve those names later from
    # this frozen record. Resources need no index: their ID is the hash of
    # (source, kind, native name).
    source_labels: Mapping[str, tuple[str, str]] = field(
        default_factory=dict, hash=False
    )
    # Sources this run may read in full: the engine decides what exists. For
    # these, resource_ids holds only what the catalog knows, for name resolution
    # and catalog tools; it never limits a read.
    unrestricted_source_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name, maximum in (("source_ids", 256), ("resource_ids", 10_000)):
            values = frozenset(getattr(self, name))
            if len(values) > maximum or any(
                not isinstance(item, str) or not item or len(item) > 2_048
                for item in values
            ):
                raise ValueError(f"effective {name} must be bounded exact identities")
            object.__setattr__(self, name, values)
        if self.resource_ids and not self.source_ids:
            raise ValueError("effective resources require an admitted source")
        labels = {
            source_id: (str(label[0]), str(label[1]))
            for source_id, label in self.source_labels.items()
        }
        if any(
            source_id not in self.source_ids or not label[0] or len(label[0]) > 2_048
            for source_id, label in labels.items()
        ):
            raise ValueError("source labels must name admitted sources")
        object.__setattr__(self, "source_labels", MappingProxyType(labels))
        unrestricted = frozenset(self.unrestricted_source_ids)
        if not unrestricted <= self.source_ids:
            raise ValueError("unrestricted sources must be admitted sources")
        object.__setattr__(self, "unrestricted_source_ids", unrestricted)

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "source_ids": tuple(sorted(self.source_ids)),
            "resource_ids": tuple(sorted(self.resource_ids)),
        }
        if self.source_labels:
            mapping["source_labels"] = {
                source_id: list(self.source_labels[source_id])
                for source_id in sorted(self.source_labels)
            }
        if self.unrestricted_source_ids:
            mapping["unrestricted_source_ids"] = tuple(
                sorted(self.unrestricted_source_ids)
            )
        return mapping

    def sources_named(self, name: str) -> tuple[str, ...]:
        """Return the IDs of every admitted source whose display name is ``name``."""

        folded = name.strip().casefold()
        return tuple(
            sorted(
                source_id
                for source_id, (display_name, _) in self.source_labels.items()
                if display_name.casefold() == folded
            )
        )


EFFECTIVE_RESOURCE_LIMIT = 10_000
SOURCE_REF_PREFIX = "s"
RESOURCE_REF_PREFIX = "t"


@dataclass(frozen=True, slots=True)
class ScopeRefs:
    """Run-scoped short handles for the source and resource IDs of one scope.

    Sources are ``s1``, ``s2``... and resources ``t1``, ``t2``... in sorted
    full-ID order of the run's frozen effective scope, so every step of a run
    numbers them identically. Handles are presentation only: a handle names
    nothing outside the run whose scope issued it.
    """

    handles: Mapping[str, str]
    ids: Mapping[str, str]

    def resolve(self, prefix: str, value: object) -> str | None:
        """Return the full ID one handle of ``prefix``'s family names, else None."""

        if not isinstance(value, str) or not value.startswith(prefix):
            return None
        return self.ids.get(value)


@lru_cache(maxsize=8)
def scope_refs(scope: EffectiveSourceScope | None) -> ScopeRefs:
    """Number one frozen scope deterministically; one shared definition.

    A scope that froze display names belongs to a run addressed by name. It was
    never shown a handle, so a value like ``s2`` there is only a name.
    """

    handles = {
        identity: f"{prefix}{index}"
        for prefix, identities in (
            ()
            if scope is None or scope.source_labels
            else (
                (SOURCE_REF_PREFIX, scope.source_ids),
                (RESOURCE_REF_PREFIX, scope.resource_ids),
            )
        )
        for index, identity in enumerate(sorted(identities), 1)
    }
    return ScopeRefs(
        handles=MappingProxyType(handles),
        ids=MappingProxyType(
            {handle: identity for identity, handle in handles.items()}
        ),
    )


async def resolve_machine_source_scope(
    agent_id: str,
    source_ids: tuple[str, ...],
    catalog: SourceScopeCatalog,
) -> EffectiveSourceScope:
    """The sources a machine run may read, fixed when it starts.

    The named sources when the routine or job names any, else every current
    source; never the live session folder, which belongs to a foreground run.
    """

    facts = await catalog.source_routing_facts(agent_id, tuple(sorted(source_ids)))
    sources = frozenset(
        value for fact in facts if isinstance(value := fact.get("source_id"), str)
    )
    if source_ids:
        sources &= frozenset(source_ids)
    sources -= {catalog.session_source_id}
    if not sources:
        return EffectiveSourceScope(frozenset(), frozenset())
    unrestricted, resources = await _read_access(agent_id, sources, (), catalog)
    return EffectiveSourceScope(
        sources, resources, unrestricted_source_ids=unrestricted
    )


# (sources it leaves unrestricted, the tables it lists for every other source)
ReadNarrowing = tuple[frozenset[str], frozenset[str]]


def run_read_narrowings(run: RunInput) -> tuple[ReadNarrowing, ...]:
    """What a run's frozen scope and machine ceiling each allow."""

    narrowings: list[ReadNarrowing] = []
    prepared = run.resolved_source_scope
    if prepared is not None:
        narrowings.append((prepared.unrestricted_source_ids, prepared.resource_ids))
    ceiling = run.execution_scope
    if ceiling is not None:
        narrowings.append(
            (
                frozenset(ceiling.unrestricted_source_ids),
                frozenset(ceiling.allowed_resource_ids),
            )
        )
    return tuple(narrowings)


def narrowed_read_access(
    source_id: str,
    access: frozenset[str] | None,
    narrowings: tuple[ReadNarrowing, ...],
) -> frozenset[str] | None:
    """One source's current access under a run's limits; None means unrestricted.

    Listed IDs from other sources are harmless here: an ID hashes its source, so
    it never matches a table of this one.
    """

    for unrestricted, listed in narrowings:
        if source_id not in unrestricted:
            access = listed if access is None else access & listed
    return access


async def _read_access(
    agent_id: str,
    sources: frozenset[str],
    narrowings: tuple[ReadNarrowing, ...],
    catalog: SourceScopeCatalog,
) -> tuple[frozenset[str], frozenset[str]]:
    """Split sources into unrestricted ones and the tables listed for the rest."""

    current = await catalog.source_read_access(agent_id, tuple(sorted(sources)))
    unrestricted: set[str] = set()
    listed: set[str] = set()
    for source_id, access in current.items():
        access = narrowed_read_access(source_id, access, narrowings)
        if access is None:
            unrestricted.add(source_id)
        else:
            listed |= access
    if unrestricted:
        listed |= await _cataloged_tables(agent_id, unrestricted, catalog)
    return frozenset(unrestricted), frozenset(listed)


async def _cataloged_tables(
    agent_id: str, sources: set[str], catalog: SourceScopeCatalog
) -> frozenset[str]:
    """What the catalog knows of unrestricted sources; it advises, so may fail."""

    from .catalog.protocols import CatalogStoreError

    try:
        known = await catalog.readable_resource_ids(agent_id, tuple(sorted(sources)))
    except CatalogStoreError:
        return frozenset()
    # shortcut: past this bound a table is still readable, but tools that name a
    # table by ID cannot reach it; lift it when catalog tools take names (2c).
    return frozenset(sorted(known)[:EFFECTIVE_RESOURCE_LIMIT])


class SourceScopeCatalog(Protocol):
    async def source_routing_facts(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> tuple[Mapping[str, object], ...]: ...

    async def readable_resource_ids(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> frozenset[str]: ...

    async def source_read_access(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> Mapping[str, frozenset[str] | None]: ...

    @property
    def session_source_id(self) -> str | None: ...


async def resolve_effective_source_scope(
    run: RunInput,
    catalog: SourceScopeCatalog,
) -> EffectiveSourceScope:
    """Intersect current reads, trusted filters, preparation and machine ceilings.

    Only an unprepared foreground request may interpret its empty caller filter
    as all admitted candidates. Never forward an empty-as-all sentinel after
    resolving authority. The live session source belongs only to a foreground
    user run; machine-originated runs never receive it.
    """

    from .loop.models import RunOrigin

    empty = EffectiveSourceScope(frozenset(), frozenset())
    ceiling = run.execution_scope
    prepared = run.resolved_source_scope
    requested: frozenset[str] | None = (
        frozenset(run.source_scope_ids) if run.source_scope_ids else None
    )
    if ceiling is not None:
        allowed = frozenset(ceiling.allowed_source_ids)
        requested = allowed if requested is None else requested & allowed
    elif (
        requested is not None
        and run.origin is RunOrigin.USER
        and catalog.session_source_id is not None
    ):
        # The live session folder is not a catalog source; a source filter
        # narrows catalog sources and never drops local files.
        requested |= {catalog.session_source_id}
    if prepared is not None:
        requested = (
            prepared.source_ids
            if requested is None
            else requested & prepared.source_ids
        )
    if requested == frozenset():
        return empty
    facts = await catalog.source_routing_facts(
        run.agent_id, () if requested is None else tuple(sorted(requested))
    )
    sources = frozenset(
        value for fact in facts if isinstance(value := fact.get("source_id"), str)
    )
    if requested is not None:
        sources &= requested
    if ceiling is not None or run.origin is not RunOrigin.USER:
        sources -= {catalog.session_source_id}
    if not sources:
        return empty
    unrestricted, resources = await _read_access(
        run.agent_id, sources, run_read_narrowings(run), catalog
    )
    return EffectiveSourceScope(
        sources,
        resources,
        {
            source_id: (display_name, str(fact.get("adapter_id", "")))
            for fact in facts
            if isinstance(source_id := fact.get("source_id"), str)
            and source_id in sources
            and isinstance(display_name := fact.get("display_name"), str)
            and display_name
        },
        unrestricted,
    )
