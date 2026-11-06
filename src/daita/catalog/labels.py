"""Advisory labels for look-alike tables: which one looks current, which stale.

Pure functions over facts a lookup already holds. A label never hides or blocks
a table; it is evidence for the model to weigh, and every fact may be unknown.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_NOISE = frozenset(
    "old new legacy backup bak archive archived tmp temp copy snapshot history "
    "current final staging deprecated".split()
)
_DEPRECATED = re.compile(
    r"\b(deprecated|obsolete|superseded|replaced by|do not use|don't use|"
    r"no longer (?:used|updated|maintained))\b",
    re.IGNORECASE,
)
_VERSION = re.compile(r"v\d+")
_COLUMN_NAME = re.compile(r"(?:^|, )([^\s,()]+) ")
_MORE = re.compile(r" \(\+\d+ more\)$")
_MAX_EVIDENCE = 3


@dataclass(frozen=True, slots=True)
class TableFacts:
    name: str
    columns: frozenset[str] | None = None
    row_estimate: int | None = None
    inbound_foreign_keys: int | None = None
    comment: str | None = None
    notes: tuple[str, ...] = ()
    last_analyzed: str | None = None


@dataclass(frozen=True, slots=True)
class TableLabel:
    status: str  # canonical, likely_stale, deprecated or unknown
    evidence: tuple[str, ...]
    look_alikes: tuple[str, ...]


def core_tokens(name: str) -> frozenset[str]:
    """What a table is about: its name without schema, plurals, years or noise."""

    bare = name.rsplit(".", 1)[-1]
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", bare).lower()
    tokens = set()
    for raw in re.split(r"[^a-z0-9]+", words):
        token = raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw
        if (
            token
            and not token.isdigit()
            and not _VERSION.fullmatch(token)
            and raw not in _NOISE
            and token not in _NOISE
        ):
            tokens.add(token)
    return frozenset(tokens)


def column_names(summary: object) -> frozenset[str] | None:
    """Names from a hit's ``name type, name type (+N more)`` summary, if it has one."""

    if not isinstance(summary, str) or not summary:
        return None
    names = _COLUMN_NAME.findall(_MORE.sub("", summary))
    return frozenset(name.lower() for name in names) or None


def _look_alike(left: TableFacts, right: TableFacts) -> bool:
    a, b = core_tokens(left.name), core_tokens(right.name)
    if not a or not b:
        return False
    if left.columns is None or right.columns is None:
        return a == b
    if not (a == b or (a < b and len(b - a) == 1) or (b < a and len(a - b) == 1)):
        return False
    shared = len(left.columns & right.columns)
    return shared / min(len(left.columns), len(right.columns)) > 0.5


def _ahead(member: TableFacts, other: TableFacts) -> tuple[bool, bool]:
    """(clearly ahead, behind) of another table; unknown facts count as neither."""

    ahead = behind = False
    pairs = (
        (member.inbound_foreign_keys, other.inbound_foreign_keys, 1),
        (member.row_estimate, other.row_estimate, 2),
    )
    for mine, theirs, factor in pairs:
        if mine is None or theirs is None:
            continue
        behind = behind or mine < theirs
        ahead = ahead or (mine > 0 and mine >= max(theirs * factor, theirs + 1))
    return ahead, behind


def _comparison(member: TableFacts, other: TableFacts) -> list[str]:
    """How the first table stands against the second, as names-free numbers."""

    notes = []
    if (
        member.row_estimate is not None
        and other.row_estimate is not None
        and member.row_estimate != other.row_estimate
    ):
        notes.append(f"{member.row_estimate:,} rows vs {other.row_estimate:,}")
    mine, theirs = member.inbound_foreign_keys, other.inbound_foreign_keys
    if mine is not None and theirs is not None and mine != theirs:
        count = "no" if mine == 0 else str(mine)
        notes.append(f"{count} inbound foreign keys vs {theirs}")
    return notes


def _deprecation(table: TableFacts) -> str | None:
    if table.comment and _DEPRECATED.search(table.comment):
        return "comment says deprecated"
    if any(_DEPRECATED.search(note) for note in table.notes):
        return "note says deprecated"
    return None


def _groups(tables: Sequence[TableFacts]) -> list[list[TableFacts]]:
    parent = list(range(len(tables)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, left in enumerate(tables):
        for j in range(i + 1, len(tables)):
            if _look_alike(left, tables[j]):
                parent[root(j)] = root(i)
    grouped: dict[int, list[TableFacts]] = {}
    for index, table in enumerate(tables):
        grouped.setdefault(root(index), []).append(table)
    return list(grouped.values())


def label_tables(tables: Sequence[TableFacts]) -> dict[str, TableLabel]:
    """Labels for the tables of ONE source that are deprecated or have look-alikes."""

    labels: dict[str, TableLabel] = {}
    for group in _groups(tables):
        deprecated = {table.name: _deprecation(table) for table in group}
        live = [table for table in group if deprecated[table.name] is None]
        names = {table.name for table in group}

        def label(table: TableFacts, status: str, evidence: list[str]) -> None:
            shown = evidence[:_MAX_EVIDENCE]
            if table.last_analyzed:
                shown.append(f"last analyzed {table.last_analyzed[:10]}")
            labels[table.name] = TableLabel(
                status,
                tuple(shown),
                tuple(sorted(names - {table.name})),
            )

        for table in group:
            if (reason := deprecated[table.name]) is not None:
                label(table, "deprecated", [reason])
        if len(group) < 2:
            continue
        if len(live) == 1:
            count = len(group) - 1
            label(live[0], "canonical", [f"{count} look-alike(s) deprecated"])
            continue
        leader = next(
            (
                table
                for table in live
                if all(
                    _ahead(table, other)[0] and not _ahead(table, other)[1]
                    for other in live
                    if other is not table
                )
            ),
            None,
        )
        if leader is None:
            for table in live:
                label(table, "unknown", [])
            continue
        evidence: list[str] = []
        for other in live:
            if other is not leader:
                evidence += _comparison(leader, other)
                label(other, "likely_stale", _comparison(other, leader))
        label(leader, "canonical", list(dict.fromkeys(evidence)))
    return labels
