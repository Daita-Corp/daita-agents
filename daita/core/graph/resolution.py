"""
Resolution layer for bare table references.

Every Table node in the catalog graph is keyed by ``table:<store>.<name>`` so
that tables with the same name in different stores can coexist. Some callers
— ``LineagePlugin.track(...)``, ``capture_sql_lineage(...)``,
``DataQualityPlugin.report(...)`` — legitimately don't know which store a
table lives in. This module is the one place that maps a bare name (plus
optional store) to one or more qualified ``ResolvedTable`` values.

An ``AmbiguousReferencePolicy`` controls what happens when a bare name
matches more than one store:

* ``STRICT`` (default) raises ``AmbiguousReferenceError`` — loud by design so
  callers don't silently clobber data across stores.
* ``LENIENT`` returns the most-recently-updated candidate and logs a warning.
* ``UNRESOLVED_SENTINEL`` returns a placeholder under the synthetic
  ``__unresolved__`` store; the catalog persister promotes it into a
  canonical node the next time discovery emits a matching table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from .models import AgentGraphNode, NodeType

if TYPE_CHECKING:
    from .backend import GraphBackend

logger = logging.getLogger(__name__)

UNRESOLVED_STORE = "__unresolved__"


class AmbiguousReferencePolicy(str, Enum):
    """How to handle a bare name that matches more than one store."""

    STRICT = "strict"
    LENIENT = "lenient"
    UNRESOLVED_SENTINEL = "unresolved_sentinel"


@dataclass(slots=True, frozen=True)
class ResolvedTable:
    """A fully-qualified Table node identity."""

    node_id: str
    store: str
    name: str

    @classmethod
    def from_parts(cls, store: str, name: str) -> "ResolvedTable":
        node_id = AgentGraphNode.make_id(NodeType.TABLE, f"{store}.{name}")
        return cls(node_id=node_id, store=store, name=name)


class AmbiguousReferenceError(ValueError):
    """Raised when a bare table name matches multiple stores and the caller
    did not disambiguate."""

    def __init__(self, name: str, candidates: List[ResolvedTable]):
        self.name = name
        self.candidates = candidates
        stores = ", ".join(sorted(c.store for c in candidates))
        super().__init__(
            f"Table {name!r} exists in multiple stores ({stores}). "
            f"Pass store= to disambiguate."
        )


def _strip_prefix(value: str) -> str:
    """Accept either ``table:<name>`` or a bare ``<name>``."""
    return value.split(":", 1)[1] if value.startswith("table:") else value


def unresolved_id(name: str) -> str:
    """Return the sentinel ID used for lineage references made before the
    catalog has discovered the target table."""
    return AgentGraphNode.make_id(NodeType.TABLE, f"{UNRESOLVED_STORE}.{name}")


async def resolve_table(
    graph_backend: "GraphBackend",
    name: str,
    store: Optional[str] = None,
) -> List[ResolvedTable]:
    """Find every qualified Table node matching ``name``.

    * ``store`` set  — returns zero or one match (exact store).
    * ``store`` None — returns every Table node whose name matches, across
      every store.

    The returned list never contains placeholder (``__unresolved__``) nodes:
    those exist only as stubs awaiting reconciliation.
    """
    bare = _strip_prefix(name)
    props: dict[str, object] = {"name": bare}
    if store is not None:
        props["store"] = store

    nodes = await graph_backend.find_nodes(NodeType.TABLE, props)

    out: List[ResolvedTable] = []
    for node in nodes:
        node_store = (node.properties or {}).get("store")
        if not node_store or node_store == UNRESOLVED_STORE:
            continue
        out.append(
            ResolvedTable(node_id=node.node_id, store=node_store, name=node.name)
        )
    return out


async def resolve_table_unique(
    graph_backend: "GraphBackend",
    name: str,
    store: Optional[str] = None,
    policy: AmbiguousReferencePolicy = AmbiguousReferencePolicy.STRICT,
) -> ResolvedTable:
    """Resolve ``name`` to exactly one qualified ``ResolvedTable``.

    Raises ``LookupError`` if nothing matches.

    When multiple stores match and ``store`` is None:
      * ``STRICT`` — raises ``AmbiguousReferenceError``.
      * ``LENIENT`` — returns the most recently touched candidate (by node
        ``updated_at``) and logs a warning; ``candidates`` is available on
        the caller via ``resolve_table`` if needed.
      * ``UNRESOLVED_SENTINEL`` — returns the sentinel ResolvedTable under
        ``__unresolved__``; the placeholder node is *not* created here.
    """
    candidates = await resolve_table(graph_backend, name, store=store)
    bare = _strip_prefix(name)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        if policy is AmbiguousReferencePolicy.UNRESOLVED_SENTINEL:
            return ResolvedTable.from_parts(UNRESOLVED_STORE, bare)
        raise LookupError(f"No Table node found for name={bare!r} store={store!r}")

    # Ambiguous — multiple matches, no store filter.
    if policy is AmbiguousReferencePolicy.STRICT:
        raise AmbiguousReferenceError(bare, candidates)
    if policy is AmbiguousReferencePolicy.UNRESOLVED_SENTINEL:
        return ResolvedTable.from_parts(UNRESOLVED_STORE, bare)

    # LENIENT — newest wins, fall back to lexicographic order.
    stores = ", ".join(sorted(c.store for c in candidates))
    logger.warning(
        "Ambiguous table reference %r matches stores %s — picking most recent.",
        bare,
        stores,
    )
    newest = await _pick_newest(graph_backend, candidates)
    return newest


async def _pick_newest(
    graph_backend: "GraphBackend",
    candidates: List[ResolvedTable],
) -> ResolvedTable:
    """Return the candidate whose node has the latest ``updated_at``."""
    best = candidates[0]
    best_ts = None
    for cand in candidates:
        node = await graph_backend.get_node(cand.node_id)
        if node is None:
            continue
        ts = node.updated_at
        if best_ts is None or (ts is not None and ts > best_ts):
            best = cand
            best_ts = ts
    return best


async def ensure_unresolved_placeholder(
    graph_backend: "GraphBackend",
    name: str,
    agent_id: Optional[str] = None,
) -> ResolvedTable:
    """Create (or upsert) a ``table:__unresolved__.<name>`` placeholder and
    return its ``ResolvedTable``. Used when lineage is declared before the
    catalog has discovered the referenced table."""
    bare = _strip_prefix(name)
    resolved = ResolvedTable.from_parts(UNRESOLVED_STORE, bare)
    node = AgentGraphNode(
        node_id=resolved.node_id,
        node_type=NodeType.TABLE,
        name=bare,
        created_by_agent=agent_id,
        properties={"store": UNRESOLVED_STORE, "unresolved": True},
    )
    await graph_backend.add_node(node)
    return resolved


async def resolve_or_placeholder(
    graph_backend: Optional["GraphBackend"],
    name: str,
    store: Optional[str] = None,
    agent_id: Optional[str] = None,
    policy: AmbiguousReferencePolicy = AmbiguousReferencePolicy.STRICT,
) -> str:
    """Return the qualified Table node ID for ``name``, creating an
    ``__unresolved__`` placeholder when the catalog has not seen it yet.

    This is the single entry point used by every caller that accepts bare
    table names (``LineagePlugin.track``, ``capture_sql_lineage``,
    ``DataQualityPlugin.report``, ``TransformerPlugin.transform_create``).

    Behaviour:
      * ``graph_backend`` is None — returns a best-effort qualified ID
        (``store`` or ``__unresolved__``) without persisting anything.
      * Exactly one match — returns it.
      * Multiple matches — honours ``policy``: STRICT raises, LENIENT picks
        newest, UNRESOLVED_SENTINEL routes through the placeholder.
      * Zero matches — creates (or reuses) a placeholder node when ``store``
        is None; returns a canonical qualified ID when ``store`` is set.
    """
    if not name:
        return name
    if ":" in name and not name.startswith("table:"):
        return name

    bare = _strip_prefix(name)

    if graph_backend is None:
        qualifier = store or UNRESOLVED_STORE
        return ResolvedTable.from_parts(qualifier, bare).node_id

    try:
        resolved = await resolve_table_unique(
            graph_backend, bare, store=store, policy=policy
        )
        if resolved.store == UNRESOLVED_STORE:
            await ensure_unresolved_placeholder(graph_backend, bare, agent_id=agent_id)
        return resolved.node_id
    except LookupError:
        if store:
            return ResolvedTable.from_parts(store, bare).node_id
        placeholder = await ensure_unresolved_placeholder(
            graph_backend, bare, agent_id=agent_id
        )
        return placeholder.node_id
