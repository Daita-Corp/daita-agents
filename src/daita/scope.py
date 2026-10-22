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
    """Number one frozen scope deterministically; one shared definition."""

    handles = {
        identity: f"{prefix}{index}"
        for prefix, identities in (
            ()
            if scope is None
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


class SourceScopeCatalog(Protocol):
    async def source_routing_facts(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> tuple[Mapping[str, object], ...]: ...

    async def readable_resource_ids(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> frozenset[str]: ...

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
    resources = await catalog.readable_resource_ids(
        run.agent_id, tuple(sorted(sources))
    )
    if ceiling is not None:
        resources &= frozenset(ceiling.allowed_resource_ids)
    if prepared is not None:
        resources &= prepared.resource_ids
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
    )
