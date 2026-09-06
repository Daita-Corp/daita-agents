"""Resolve explicit source/resource candidates once and narrow them at call time."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .loop.models import RunInput


@dataclass(frozen=True, slots=True)
class EffectiveSourceScope:
    """Resolved admission, where empty sets always mean no access."""

    source_ids: frozenset[str]
    resource_ids: frozenset[str]

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

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_ids": tuple(sorted(self.source_ids)),
            "resource_ids": tuple(sorted(self.resource_ids)),
        }


class SourceScopeCatalog(Protocol):
    async def source_routing_facts(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> tuple[Mapping[str, object], ...]: ...

    async def readable_resource_ids(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> frozenset[str]: ...


async def resolve_effective_source_scope(
    run: RunInput,
    catalog: SourceScopeCatalog,
    *,
    files_only: bool = False,
) -> EffectiveSourceScope:
    """Intersect current reads, trusted filters, preparation and machine ceilings.

    Only an unprepared foreground request may interpret its empty caller filter
    as all admitted candidates. Never forward an empty-as-all sentinel after
    resolving authority.
    """

    empty = EffectiveSourceScope(frozenset(), frozenset())
    if files_only:
        return empty
    ceiling = run.execution_scope
    prepared = run.resolved_source_scope
    requested: frozenset[str] | None = (
        frozenset(run.source_scope_ids) if run.source_scope_ids else None
    )
    if ceiling is not None:
        allowed = frozenset(ceiling.allowed_source_ids)
        requested = allowed if requested is None else requested & allowed
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
    if not sources:
        return empty
    resources = await catalog.readable_resource_ids(
        run.agent_id, tuple(sorted(sources))
    )
    if ceiling is not None:
        resources &= frozenset(ceiling.allowed_resource_ids)
    if prepared is not None:
        resources &= prepared.resource_ids
    return EffectiveSourceScope(sources, resources)
