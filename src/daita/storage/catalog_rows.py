"""Per-resource catalog SQL shared by every ``SQLStateStore`` backend.

Statements use qmark parameters and standard SQL only, chunked under 900
parameters, so they run unchanged on SQLite and PostgreSQL.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from ..catalog.models import (
    CATALOG_CHANGE_MAX_COLUMNS,
    CatalogChange,
    CatalogChangeKind,
    CatalogChangeSide,
    CatalogCommitSummary,
    CatalogEntry,
    CatalogRelationship,
    CatalogSearchRow,
    CatalogSourceState,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTier,
    ResourceBatch,
    ResourceKind,
)
from ..catalog.search_text import (
    search_key,
    search_postings,
    search_words,
    tabular_parts,
)
from .sql_connection import SQLConnection
from .sqlite_codecs import (
    decode_catalog_change_data,
    decode_catalog_entry,
    decode_catalog_relationship,
    decode_catalog_source_state,
    decode_catalog_sync,
    decode_search_postings,
    encode_catalog_change_data,
    encode_catalog_entry,
    encode_catalog_relationship,
    encode_catalog_source_state,
    encode_catalog_sync,
    encode_search_postings,
)
from .sqlite_codecs.common import datetime_encode

_T = TypeVar("_T")
_PARAMETERS = 900
_IN = 400  # ids per IN list; some statements bind two lists
KEPT_SYNCS = 20
KEPT_CHANGES = 10_000
CHANGE_DAYS = 90
# (source_id, row id or "") of a row that failed to decode.
Note = Callable[[str, str], None]
_DECODE_ERRORS = (TypeError, ValueError, KeyError)


def stamp(value: datetime) -> str:
    text = datetime_encode(value)["__datetime__"]
    assert isinstance(text, str)
    return text


def _chunks(items: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _marks(count: int) -> str:
    return ", ".join("?" * count)


def _write_rows(
    connection: SQLConnection,
    table: str,
    columns: tuple[str, ...],
    rows: Sequence[tuple[Any, ...]],
    *,
    key: str | None = None,
    update: tuple[str, ...] = (),
) -> None:
    if not rows:
        return
    group = "(" + _marks(len(columns)) + ")"
    tail = ""
    if key is not None:
        sets = ", ".join(f"{name} = excluded.{name}" for name in update)
        tail = f" ON CONFLICT({connection.conflict(key)}) DO UPDATE SET {sets}"
    for chunk in _chunks(rows, _PARAMETERS // len(columns)):
        connection.execute(
            f"INSERT INTO {table}({', '.join(columns)}) VALUES "
            + ", ".join([group] * len(chunk))
            + tail,
            [value for row in chunk for value in row],
        )


def _fetch_in(
    connection: SQLConnection,
    query: str,
    leading: tuple[Any, ...],
    ids: Sequence[str],
) -> list[tuple[Any, ...]]:
    """Run ``query`` (with one ``{marks}``) once per chunk of ``ids``."""

    rows: list[tuple[Any, ...]] = []
    for chunk in _chunks(list(ids), _IN):
        rows.extend(
            connection.execute(
                query.format(marks=_marks(len(chunk))), (*leading, *chunk)
            ).fetchall()
        )
    return rows


def _delete_in(
    connection: SQLConnection, table: str, column: str, agent_id: str, ids: list[str]
) -> None:
    for chunk in _chunks(ids, _IN):
        connection.execute(
            f"DELETE FROM {table} WHERE agent_id = ? AND {column} IN "
            f"({_marks(len(chunk))})",
            (agent_id, *chunk),
        )


# ---------------------------------------------------------------- decoding


_RESOURCE_COLUMNS = "source_id, id, kind, native_identity, name, revision, detail, data"


def _entry(agent_id: str, row: tuple[Any, ...]) -> CatalogEntry:
    source_id, resource_id, kind, identity, name, revision, detail, data = row
    entry = decode_catalog_entry(data)
    resource = entry.resource
    if (
        resource.agent_id,
        resource.source_id,
        resource.id,
        resource.kind.value,
        resource.native_identity,
        resource.name,
        resource.current_revision,
        entry.detailed,
    ) != (
        agent_id,
        source_id,
        resource_id,
        kind,
        identity,
        name,
        revision,
        detail == "complete",
    ):
        raise ValueError("catalog row disagrees with its record")
    return entry


def _relationship(row: tuple[Any, ...]) -> CatalogRelationship:
    source_id, relationship_id, from_id, to_id, revision, data = row
    relationship = decode_catalog_relationship(data)
    if (
        relationship.source_id,
        relationship.id,
        relationship.from_resource_id,
        relationship.to_resource_id,
        relationship.revision,
    ) != (source_id, relationship_id, from_id, to_id, revision):
        raise ValueError("catalog row disagrees with its record")
    return relationship


def _decoded(
    rows: Sequence[tuple[Any, ...]],
    decode: Callable[[tuple[Any, ...]], _T],
    note: Note,
    source_of: Callable[[tuple[Any, ...]], str],
    ident: Callable[[tuple[Any, ...]], str] = lambda row: "",
) -> list[_T]:
    """Decode returned rows only; a bad row is absent and its source noted."""

    found: list[_T] = []
    for row in rows:
        try:
            found.append(decode(row))
        except _DECODE_ERRORS:
            note(source_of(row), ident(row))
    return found


def _try_relationship(row: tuple[Any, ...], note: Note) -> CatalogRelationship | None:
    try:
        return _relationship(row)
    except _DECODE_ERRORS:
        note(row[0], row[1])
        return None


_RELATIONSHIP_COLUMNS = (
    "source_id, id, from_resource_id, to_resource_id, revision, data"
)


def load_entries(
    connection: SQLConnection, agent_id: str, ids: Sequence[str], note: Note
) -> list[CatalogEntry]:
    rows = _fetch_in(
        connection,
        f"SELECT {_RESOURCE_COLUMNS} FROM catalog_resources "
        "WHERE agent_id = ? AND id IN ({marks})",
        (agent_id,),
        ids,
    )
    return _decoded(
        rows,
        lambda row: _entry(agent_id, row),
        note,
        lambda row: row[0],
        lambda row: row[1],
    )


def load_entries_by_identity(
    connection: SQLConnection,
    agent_id: str,
    source_id: str,
    native_identities: Sequence[str],
    note: Note,
) -> list[CatalogEntry]:
    rows = _fetch_in(
        connection,
        f"SELECT {_RESOURCE_COLUMNS} FROM catalog_resources "
        "WHERE agent_id = ? AND source_id = ? AND native_identity IN ({marks})",
        (agent_id, source_id),
        native_identities,
    )
    return _decoded(
        rows,
        lambda row: _entry(agent_id, row),
        note,
        lambda row: row[0],
        lambda row: row[1],
    )


def resource_names(
    connection: SQLConnection,
    agent_id: str,
    source_ids: Sequence[str],
    other_than: str | None,
    limit: int | None,
) -> list[tuple[str, str, str]]:
    """(ID, source ID, native name) of resources, from columns alone.

    All of them, or the ``limit`` smallest IDs.
    """

    query = (
        "SELECT id, source_id, native_identity FROM catalog_resources "
        "WHERE agent_id = ?"
    )
    leading: tuple[Any, ...] = (agent_id,)
    if other_than is not None:
        query += " AND kind <> ?"
        leading += (other_than,)
    tail = "" if limit is None else " ORDER BY id LIMIT ?"
    trailing: tuple[Any, ...] = () if limit is None else (limit,)
    rows: list[tuple[Any, ...]] = []
    for chunk in _chunks(list(source_ids), _IN) if source_ids else [[]]:
        scope = f" AND source_id IN ({_marks(len(chunk))})" if chunk else ""
        rows.extend(
            connection.execute(
                query + scope + tail, (*leading, *chunk, *trailing)
            ).fetchall()
        )
    if limit is not None:
        rows = sorted(rows)[:limit]
    return [(row[0], row[1], row[2]) for row in rows]


def list_entries(
    connection: SQLConnection, agent_id: str, source_id: str | None, note: Note
) -> list[CatalogEntry]:
    query = f"SELECT {_RESOURCE_COLUMNS} FROM catalog_resources WHERE agent_id = ?"
    parameters: tuple[Any, ...] = (agent_id,)
    if source_id is not None:
        query += " AND source_id = ?"
        parameters += (source_id,)
    rows = connection.execute(query, parameters).fetchall()
    entries = _decoded(
        rows,
        lambda row: _entry(agent_id, row),
        note,
        lambda row: row[0],
        lambda row: row[1],
    )
    return sorted(entries, key=lambda item: (item.resource.name, item.resource.id))


def _incident_rows(
    connection: SQLConnection, agent_id: str, ids: Sequence[str]
) -> dict[str, tuple[Any, ...]]:
    """Rows of every relationship touching ``ids``, keyed by relationship id.

    Two indexed lookups (the from and to indexes) rather than one OR, which
    SQLite and PostgreSQL both answer by scanning the agent's relationships.
    """

    rows: dict[str, tuple[Any, ...]] = {}
    for chunk in _chunks(list(ids), _IN):
        marks = _marks(len(chunk))
        found = connection.execute(
            f"SELECT {_RELATIONSHIP_COLUMNS} FROM catalog_relationships "
            f"WHERE agent_id = ? AND from_resource_id IN ({marks}) "
            f"UNION ALL SELECT {_RELATIONSHIP_COLUMNS} FROM catalog_relationships "
            f"WHERE agent_id = ? AND to_resource_id IN ({marks})",
            (agent_id, *chunk, agent_id, *chunk),
        ).fetchall()
        rows.update((row[1], row) for row in found)
    return rows


def incident_relationships(
    connection: SQLConnection,
    agent_id: str,
    resource_ids: Sequence[str],
    kinds: tuple[Any, ...],
    limit: int,
    note: Note,
) -> dict[str, tuple[CatalogRelationship, ...]]:
    """Each resource's incident relationships by id, at most ``limit`` apiece."""

    by_end: dict[str, list[str]] = {i: [] for i in resource_ids}
    rows = _incident_rows(connection, agent_id, list(by_end))
    for relationship_id, row in rows.items():
        for end in {row[2], row[3]}:
            if end in by_end:
                by_end[end].append(relationship_id)
    decoded: dict[str, CatalogRelationship | None] = {}

    def decode(relationship_id: str) -> CatalogRelationship | None:
        if relationship_id not in decoded:
            decoded[relationship_id] = _try_relationship(rows[relationship_id], note)
        return decoded[relationship_id]

    result: dict[str, tuple[CatalogRelationship, ...]] = {}
    for resource_id, relationship_ids in by_end.items():
        ordered = sorted(relationship_ids)
        if not kinds:
            ordered = ordered[:limit]  # decode only what is returned
        items = [item for item in map(decode, ordered) if item is not None]
        if kinds:
            items = [item for item in items if item.kind in kinds]
        result[resource_id] = tuple(items[:limit])
    return result


def current_syncs(
    connection: SQLConnection, agent_id: str, source_ids: tuple[str, ...], note: Note
) -> tuple[CatalogSync, ...]:
    if source_ids:
        rows = _fetch_in(
            connection,
            "SELECT source_id, data FROM catalog_source_state "
            "WHERE agent_id = ? AND source_id IN ({marks})",
            (agent_id,),
            source_ids,
        )
    else:
        rows = connection.execute(
            "SELECT source_id, data FROM catalog_source_state WHERE agent_id = ?",
            (agent_id,),
        ).fetchall()

    def state(row: tuple[Any, ...]) -> CatalogSourceState:
        decoded = decode_catalog_source_state(row[1])
        if (decoded.agent_id, decoded.source_id) != (agent_id, row[0]):
            raise ValueError("catalog row disagrees with its record")
        return decoded

    states = _decoded(rows, state, note, lambda row: row[0])
    sync_rows = _fetch_in(
        connection,
        "SELECT source_id, id, data FROM syncs "
        "WHERE agent_id = ? AND id IN ({marks})",
        (agent_id,),
        [item.current_sync_id for item in states],
    )
    by_id = {row[1]: row for row in sync_rows}
    syncs: list[CatalogSync] = []
    for item in states:
        row = by_id.get(item.current_sync_id)
        try:
            if row is None:
                raise ValueError("current sync is missing")
            sync = decode_catalog_sync(row[2])
            if (sync.agent_id, sync.source_id, sync.id) != (
                agent_id,
                item.source_id,
                item.current_sync_id,
            ):
                raise ValueError("catalog row disagrees with its record")
        except _DECODE_ERRORS:
            note(item.source_id, "")
            continue
        syncs.append(sync)
    return tuple(sorted(syncs, key=lambda item: item.source_id))


def source_state(
    connection: SQLConnection, agent_id: str, source_id: str, note: Note
) -> CatalogSourceState | None:
    row = connection.execute(
        "SELECT data FROM catalog_source_state WHERE agent_id = ? AND source_id = ?",
        (agent_id, source_id),
    ).fetchone()
    if row is None:
        return None
    try:
        state = decode_catalog_source_state(row[0])
        if (state.agent_id, state.source_id) != (agent_id, source_id):
            raise ValueError("catalog row disagrees with its record")
        return state
    except _DECODE_ERRORS:
        note(source_id, "")
        return None


def detail_counts(
    connection: SQLConnection, agent_id: str, source_id: str
) -> tuple[int, int]:
    row = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN detail = 'complete' THEN 1 ELSE 0 END), 0) "
        "FROM catalog_resources WHERE agent_id = ? AND source_id = ?",
        (agent_id, source_id),
    ).fetchone()
    assert row is not None  # an aggregate always returns a row
    return int(row[0]), int(row[1])


def save_state(connection: SQLConnection, state: CatalogSourceState) -> None:
    known = connection.execute(
        "SELECT 1 FROM sources WHERE agent_id = ? AND id = ?",
        (state.agent_id, state.source_id),
    ).fetchone()
    if known is None:
        raise ValueError("catalog source is unknown")
    _write_rows(
        connection,
        "catalog_source_state",
        ("agent_id", "source_id", "data"),
        [(state.agent_id, state.source_id, encode_catalog_source_state(state))],
        key="agent_id, source_id",
        update=("data",),
    )


def list_changes(
    connection: SQLConnection, agent_id: str, source_id: str, limit: int, note: Note
) -> tuple[CatalogChange, ...]:
    rows = connection.execute(
        "SELECT sequence, sync_id, changed_at, kind, resource_id, native_identity, "
        "data FROM catalog_changes WHERE agent_id = ? AND source_id = ? "
        "ORDER BY sequence DESC LIMIT ?",
        (agent_id, source_id, limit),
    ).fetchall()

    def change(row: tuple[Any, ...]) -> CatalogChange:
        before, after = decode_catalog_change_data(row[6])
        return CatalogChange(
            agent_id=agent_id,
            source_id=source_id,
            sequence=int(row[0]),
            sync_id=row[1],
            changed_at=datetime.fromisoformat(row[2]),
            kind=CatalogChangeKind(row[3]),
            resource_id=row[4],
            native_identity=row[5],
            before=before,
            after=after,
        )

    return tuple(_decoded(rows, change, note, lambda row: source_id))


def counts(
    connection: SQLConnection, agent_id: str, source_ids: Sequence[str]
) -> tuple[int, int]:
    totals = []
    for table in ("catalog_resources", "catalog_relationships"):
        rows = _fetch_in(
            connection,
            f"SELECT COUNT(*) FROM {table} "
            "WHERE agent_id = ? AND source_id IN ({marks})",
            (agent_id,),
            source_ids,
        )
        totals.append(sum(int(row[0]) for row in rows))
    return totals[0], totals[1]


def search(
    connection: SQLConnection,
    agent_id: str,
    source_ids: tuple[str, ...],
    terms: tuple[str, ...] | None,
    anchor_keys: tuple[str, ...],
    note: Note,
) -> tuple[CatalogSearchRow, ...]:
    conditions: list[tuple[str, tuple[str, ...]]] = []
    if terms is not None:
        conditions = [("s.words LIKE ?", (f"% {term}%",)) for term in terms]
        conditions += [
            ("(s.name_key = ? OR s.identity_key = ?)", (key, key))
            for key in anchor_keys
        ]
        if not conditions:
            return ()
    condition_groups: list[list[tuple[str, tuple[str, ...]]]] = [[]]
    used = 0
    if terms is not None and not terms:
        # Names alone: one lookup per key and index, which a combined OR would
        # trade for a scan of the whole source.
        condition_groups = [
            [(f"s.{column} = ?", (key,))]
            for column in ("name_key", "identity_key")
            for key in anchor_keys
        ]
        conditions = []
    for condition in conditions:
        if used + len(condition[1]) > _IN:
            condition_groups.append([])
            used = 0
        condition_groups[-1].append(condition)
        used += len(condition[1])
    source_groups = list(_chunks(source_ids, _IN)) or [()]
    rows: dict[str, tuple[Any, ...]] = {}
    for sources in source_groups:
        for group in condition_groups:
            if terms is None:
                # Listing every row needs no postings, so the search table is skipped.
                query = (
                    "SELECT r.source_id, r.id, '', r.name, r.native_identity, "
                    "r.kind, r.revision, r.detail FROM catalog_resources AS r "
                    "WHERE r.agent_id = ?"
                )
                column = "r.source_id"
            else:
                query = (
                    "SELECT s.source_id, s.resource_id, s.data, r.name, "
                    "r.native_identity, r.kind, r.revision, r.detail "
                    "FROM catalog_search AS s JOIN catalog_resources AS r "
                    "ON r.agent_id = s.agent_id AND r.id = s.resource_id "
                    "WHERE s.agent_id = ?"
                )
                column = "s.source_id"
            parameters: list[Any] = [agent_id]
            if sources:
                query += f" AND {column} IN ({_marks(len(sources))})"
                parameters.extend(sources)
            if group:
                query += " AND (" + " OR ".join(item[0] for item in group) + ")"
                for item in group:
                    parameters.extend(item[1])
            for row in connection.execute(query, parameters).fetchall():
                rows[row[1]] = row

    def build(row: tuple[Any, ...]) -> CatalogSearchRow:
        return CatalogSearchRow(
            resource_id=row[1],
            source_id=row[0],
            name=row[3],
            native_identity=row[4],
            kind=ResourceKind(row[5]),
            revision=row[6],
            detailed=row[7] == "complete",
            postings=() if terms is None else decode_search_postings(row[2]),
        )

    found = _decoded(
        list(rows.values()), build, note, lambda row: row[0], lambda row: row[1]
    )
    return tuple(
        sorted(
            found,
            key=lambda item: (item.source_id, item.native_identity, item.resource_id),
        )
    )


# ------------------------------------------------------------------ commit


@dataclass(frozen=True, slots=True)
class _Stored:
    revision: str
    detailed: bool
    first_observed_at: datetime
    name: str
    kind: str
    native_identity: str
    source_object_id: str | None


def _side(entry: CatalogEntry) -> CatalogChangeSide:
    columns, _ = tabular_parts(entry)
    return CatalogChangeSide(
        name=entry.resource.name,
        kind=entry.resource.kind,
        revision=entry.resource.current_revision,
        columns=tuple(columns[:CATALOG_CHANGE_MAX_COLUMNS]),
    )


def _stored_side(stored: _Stored, entry: CatalogEntry | None) -> CatalogChangeSide:
    if entry is not None:
        return _side(entry)
    return CatalogChangeSide(
        name=stored.name, kind=ResourceKind(stored.kind), revision=stored.revision
    )


def commit_pass(
    connection: SQLConnection,
    sync: CatalogSync,
    batches: tuple[ResourceBatch, ...],
    *,
    full: bool,
    note: Note,
    repair: frozenset[str] = frozenset(),
    probe_tokens: Mapping[str, str] | None = None,
    probed_at: datetime | None = None,
) -> CatalogCommitSummary:
    """Write one succeeded pass inside the caller's transaction.

    ``repair`` names rows an earlier read could not decode; the pass rewrites
    them even though their revision is unchanged.
    """

    if sync.status is not CatalogSyncStatus.SUCCEEDED:
        raise ValueError("only a succeeded sync can commit catalog rows")
    if (probe_tokens is None) != (probed_at is None):
        raise ValueError("probe tokens and probe time come together")
    agent_id, source_id = sync.agent_id, sync.source_id
    entries: dict[str, CatalogEntry] = {}
    relationships: dict[str, CatalogRelationship] = {}
    for batch in batches:
        for item in batch.entries:
            resource = item.resource
            if (resource.agent_id, resource.source_id) != (agent_id, source_id):
                raise ValueError("catalog batch and sync disagree")
            if batch.tier is CatalogTier.DETAIL or item.resource.id not in entries:
                entries[resource.id] = item
        for relationship in batch.relationships:
            if relationship.source_id != source_id:
                raise ValueError("catalog batch and sync disagree")
            relationships[relationship.id] = relationship
    has_inventory = any(b.tier is CatalogTier.INVENTORY for b in batches)
    inventory_cut = any(b.truncated for b in batches if b.tier is CatalogTier.INVENTORY)
    detail_cut = any(b.truncated for b in batches if b.tier is CatalogTier.DETAIL)
    # The stored sync always states the pass it committed, whatever the caller
    # counted: summaries read these counts when nothing was cut.
    sync = replace(
        sync,
        resource_count=len(entries),
        relationship_count=len(relationships),
        truncated=sync.truncated or inventory_cut or detail_cut,
    )
    # Without an inventory batch the detail batches are the only listing.
    listing_complete = not inventory_cut and (has_inventory or not detail_cut)

    stored: dict[str, _Stored] = {
        row[0]: _Stored(
            revision=row[1],
            detailed=row[2] == "complete",
            first_observed_at=datetime.fromisoformat(row[3]),
            name=row[4],
            kind=row[5],
            native_identity=row[6],
            source_object_id=row[7],
        )
        for row in connection.execute(
            "SELECT id, revision, detail, first_observed_at, name, kind, "
            "native_identity, source_object_id FROM catalog_resources "
            "WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchall()
    }
    stored_relationships = {
        row[0]: (row[1], row[2], row[3])
        for row in connection.execute(
            "SELECT id, revision, from_resource_id, to_resource_id "
            "FROM catalog_relationships WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchall()
    }

    added = [entries[i] for i in entries if i not in stored]
    changed: list[CatalogEntry] = []
    rewritten: list[CatalogEntry] = []  # damaged rows rewritten without a change
    for resource_id, item in entries.items():
        old = stored.get(resource_id)
        if old is None:
            continue
        # Only a new revision of a described row is a change. The first detail
        # of a listed row, or a new name or object id, is a rewrite.
        differs = (
            item.detailed and old.detailed and (old.revision != item.revision.revision)
        )
        refresh = item.detailed and (
            not old.detailed
            or old.name != item.resource.name
            or old.source_object_id != item.source_object_id
        )
        if not (
            differs
            or refresh
            or (resource_id in repair and (item.detailed or not old.detailed))
        ):
            continue
        first = old.first_observed_at
        if first > item.resource.last_observed_at:
            first = item.resource.first_observed_at
        keeps = replace(item, resource=replace(item.resource, first_observed_at=first))
        (changed if differs else rewritten).append(keeps)
    removed = sorted(i for i in stored if i not in entries) if listing_complete else []
    removed_set = set(removed)
    added.sort(key=lambda item: item.resource.native_identity)
    changed.sort(key=lambda item: item.resource.native_identity)
    removed.sort(key=lambda i: stored[i].native_identity)

    after_ids = (set(stored) - removed_set) | {item.resource.id for item in added}
    for relationship in relationships.values():
        if {relationship.from_resource_id, relationship.to_resource_id} - after_ids:
            raise ValueError("catalog relationship endpoint is not in the source")
    upserts = [
        item
        for item in relationships.values()
        if item.id in repair
        or stored_relationships.get(item.id, ("",))[0] != item.revision
    ]
    gone_ids = {
        i
        for i, (_, from_id, to_id) in stored_relationships.items()
        if from_id in removed_set or to_id in removed_set
    }
    if not detail_cut:
        gone_ids |= set(stored_relationships) - set(relationships)
    gone = sorted(gone_ids)

    # Change log rows, newest kept; the old side of a change needs its stored data.
    plan = [("added", item.resource.id) for item in added]
    plan += [("changed", item.resource.id) for item in changed]
    plan += [("removed", i) for i in removed]
    plan = plan[-KEPT_CHANGES:]
    need_old = [i for kind, i in plan if kind != "added"]
    old_entries = {
        item.resource.id: item
        for item in load_entries(connection, agent_id, need_old, note)
    }
    by_id = {item.resource.id: item for item in (*added, *changed, *rewritten)}
    completed = (sync.completed_at or sync.started_at).astimezone(UTC)
    row = connection.execute(
        "SELECT MAX(sequence) FROM catalog_changes WHERE agent_id = ? AND source_id = ?",
        (agent_id, source_id),
    ).fetchone()
    sequence = 0 if row is None or row[0] is None else int(row[0])
    change_rows: list[tuple[Any, ...]] = []
    for kind, resource_id in plan:
        sequence += 1
        old = stored.get(resource_id)
        before = (
            None if old is None else _stored_side(old, old_entries.get(resource_id))
        )
        after = None if kind == "removed" else _side(by_id[resource_id])
        identity = (
            stored[resource_id].native_identity
            if kind == "removed"
            else by_id[resource_id].resource.native_identity
        )
        change_rows.append(
            (
                agent_id,
                source_id,
                sequence,
                sync.id,
                stamp(completed),
                kind,
                resource_id,
                identity,
                encode_catalog_change_data(before, after),
            )
        )

    # Resources and relationships.
    resource_rows = [
        (
            agent_id,
            source_id,
            item.resource.id,
            item.resource.kind.value,
            item.resource.native_identity,
            item.resource.name,
            item.resource.current_revision,
            item.source_object_id,
            "complete" if item.detailed else "inventory",
            sync.id,
            stamp(item.resource.first_observed_at),
            stamp(item.resource.last_observed_at),
            encode_catalog_entry(item),
        )
        for item in (*added, *changed, *rewritten)
    ]
    _write_rows(
        connection,
        "catalog_resources",
        (
            "agent_id",
            "source_id",
            "id",
            "kind",
            "native_identity",
            "name",
            "revision",
            "source_object_id",
            "detail",
            "last_seen_sync",
            "first_observed_at",
            "last_observed_at",
            "data",
        ),
        resource_rows,
        key="agent_id, id",
        update=(
            "kind",
            "native_identity",
            "name",
            "revision",
            "source_object_id",
            "detail",
            "last_seen_sync",
            "last_observed_at",
            "data",
        ),
    )
    _delete_in(connection, "catalog_resources", "id", agent_id, removed)
    _delete_in(connection, "catalog_search", "resource_id", agent_id, removed)
    _delete_in(connection, "catalog_relationships", "id", agent_id, gone)
    _write_rows(
        connection,
        "catalog_relationships",
        (
            "agent_id",
            "source_id",
            "id",
            "from_resource_id",
            "to_resource_id",
            "revision",
            "data",
        ),
        [
            (
                agent_id,
                source_id,
                item.id,
                item.from_resource_id,
                item.to_resource_id,
                item.revision,
                encode_catalog_relationship(item),
            )
            for item in upserts
        ],
        key="agent_id, id",
        update=("from_resource_id", "to_resource_id", "revision", "data"),
    )

    # Search rows follow every resource whose own entry or relationships changed.
    touched = {item.resource.id for item in (*added, *changed, *rewritten)}
    for rel in upserts:
        touched |= {rel.from_resource_id, rel.to_resource_id}
    for relationship_id in gone:
        touched |= set(stored_relationships[relationship_id][1:])
    touched -= removed_set
    known = {i: by_id[i] for i in touched if i in by_id}
    known.update(
        {
            item.resource.id: item
            for item in load_entries(
                connection, agent_id, sorted(touched - set(known)), note
            )
        }
    )
    incident: dict[str, dict[str, CatalogRelationship]] = {i: {} for i in known}
    ids = sorted(known)
    for relationship_id, row in _incident_rows(connection, agent_id, ids).items():
        known_link = relationships.get(relationship_id)
        if known_link is None or known_link.revision != row[4]:
            known_link = _try_relationship(row, note)
        if known_link is None:
            continue
        for end in (known_link.from_resource_id, known_link.to_resource_id):
            if end in incident:
                incident[end][known_link.id] = known_link
    search_rows = []
    for resource_id in ids:
        item = known[resource_id]
        postings = search_postings(item, incident[resource_id].values())
        search_rows.append(
            (
                agent_id,
                source_id,
                resource_id,
                search_key(item.resource.name),
                search_key(item.resource.native_identity),
                search_words(postings),
                encode_search_postings(postings),
            )
        )
    _write_rows(
        connection,
        "catalog_search",
        (
            "agent_id",
            "source_id",
            "resource_id",
            "name_key",
            "identity_key",
            "words",
            "data",
        ),
        search_rows,
        key="agent_id, resource_id",
        update=("source_id", "name_key", "identity_key", "words", "data"),
    )
    _write_rows(
        connection,
        "catalog_changes",
        (
            "agent_id",
            "source_id",
            "sequence",
            "sync_id",
            "changed_at",
            "kind",
            "resource_id",
            "native_identity",
            "data",
        ),
        change_rows,
    )

    # Sync, state and bounds. The catalog lacks detail while any row is a bare listing.
    if (
        not sync.truncated
        and connection.execute(
            "SELECT 1 FROM catalog_resources WHERE agent_id = ? AND source_id = ? "
            "AND detail = 'inventory' LIMIT 1",
            (agent_id, source_id),
        ).fetchone()
    ):
        sync = replace(sync, truncated=True)
    _write_rows(
        connection,
        "syncs",
        ("agent_id", "id", "source_id", "data"),
        [(agent_id, sync.id, source_id, encode_catalog_sync(sync))],
        key="agent_id, id",
        update=("data",),
    )
    anything = bool(added or changed or removed or upserts or gone)
    previous = source_state(connection, agent_id, source_id, lambda *_: None)
    state = previous or CatalogSourceState(agent_id, source_id, sync.id)
    state = replace(
        state,
        current_sync_id=sync.id,
        last_change_at=completed if anything else state.last_change_at,
        last_full_check_at=completed if full else state.last_full_check_at,
    )
    if probe_tokens is not None:
        state = replace(
            state,
            probe_tokens=probe_tokens,
            last_probe_at=probed_at,
            last_error_code=None,
        )
    _write_rows(
        connection,
        "catalog_source_state",
        ("agent_id", "source_id", "data"),
        [(agent_id, source_id, encode_catalog_source_state(state))],
        key="agent_id, source_id",
        update=("data",),
    )
    _prune(connection, agent_id, source_id, sync, completed)
    return CatalogCommitSummary(len(added), len(changed), len(removed))


def _prune(
    connection: SQLConnection,
    agent_id: str,
    source_id: str,
    current: CatalogSync,
    completed: datetime,
) -> None:
    syncs: list[tuple[datetime, str]] = []
    for sync_id, data in connection.execute(
        "SELECT id, data FROM syncs WHERE agent_id = ? AND source_id = ?",
        (agent_id, source_id),
    ).fetchall():
        try:
            started = decode_catalog_sync(data).started_at
        except _DECODE_ERRORS:
            started = datetime.min.replace(tzinfo=UTC)
        syncs.append((started, sync_id))
    syncs.sort(reverse=True)
    keep = {sync_id for _, sync_id in syncs[:KEPT_SYNCS]} | {current.id}
    _delete_in(
        connection,
        "syncs",
        "id",
        agent_id,
        [sync_id for _, sync_id in syncs if sync_id not in keep],
    )
    newest = connection.execute(
        "SELECT MAX(sequence) FROM catalog_changes WHERE agent_id = ? AND source_id = ?",
        (agent_id, source_id),
    ).fetchone()
    if newest is not None and newest[0] is not None:
        connection.execute(
            "DELETE FROM catalog_changes WHERE agent_id = ? AND source_id = ? "
            "AND (sequence <= ? OR changed_at < ?)",
            (
                agent_id,
                source_id,
                int(newest[0]) - KEPT_CHANGES,
                stamp(completed - timedelta(days=CHANGE_DAYS)),
            ),
        )
