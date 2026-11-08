"""Normalize catalog search text and build the per-resource search postings."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from .models import (
    CatalogEntry,
    CatalogRelationship,
    FacetKind,
    SearchPosting,
)

_SEARCH_TOKEN_LIMIT = 64
_SEARCH_TOKEN_LENGTH_LIMIT = 128
SEARCH_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "by",
        "can",
        "for",
        "from",
        "get",
        "give",
        "how",
        "in",
        "is",
        "me",
        "of",
        "on",
        "show",
        "the",
        "to",
        "what",
        "which",
    }
)
_CAMEL_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_CAMEL_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True, slots=True)
class NormalizedText:
    complete: str
    tokens: tuple[str, ...]


@lru_cache(maxsize=65_536)  # column names repeat across every table
def normalize_search_text(value: str) -> NormalizedText:
    if not isinstance(value, str):
        raise TypeError("catalog search text must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    split = _CAMEL_ACRONYM_BOUNDARY.sub(" ", normalized)
    split = _CAMEL_WORD_BOUNDARY.sub(" ", split)
    complete = normalized.casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in split.casefold():
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    bounded_tokens = tuple(
        dict.fromkeys(token[:_SEARCH_TOKEN_LENGTH_LIMIT] for token in tokens if token)
    )[:_SEARCH_TOKEN_LIMIT]
    return NormalizedText(complete=complete, tokens=bounded_tokens)


def contains_bounded_identifier(query: str, identifier: str) -> bool:
    """Return whether one normalized identity occurs at identifier boundaries."""

    if not identifier:
        return False
    start = query.find(identifier)
    while start >= 0:
        end = start + len(identifier)
        before_is_identifier = start > 0 and _is_identifier_character(query[start - 1])
        after_is_identifier = end < len(query) and _is_identifier_character(query[end])
        if not before_is_identifier and not after_is_identifier:
            return True
        start = query.find(identifier, start + 1)
    return False


def _is_identifier_character(value: str) -> bool:
    return value.isalnum() or value == "_"


_FIXED_FIELD_NAMES = {
    "resource_name": "name",
    "native_identity": "native_identity",
    "kind": "kind",
}
_PREFIXED_FIELD_KINDS = frozenset({"column", "index_field", "relationship_field"})


def posting_field_name(field_kind: str, value: str) -> str:
    """Derive a posting's field name; stored postings keep only kind and value."""

    if field_kind in _PREFIXED_FIELD_KINDS:
        return f"{field_kind}:{value}"
    try:
        return _FIXED_FIELD_NAMES[field_kind]
    except KeyError:
        raise ValueError("stored search posting kind is invalid") from None


def tabular_parts(
    entry: CatalogEntry,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (name, native_type) columns and index fields of the tabular facets.

    Reads the facet payload directly: strict ``TabularFacet.from_payload``
    validation of every column dominated first-commit time.
    """

    columns: list[tuple[str, str]] = []
    index_fields: list[str] = []
    for facet in entry.facets:
        if facet.kind is not FacetKind.TABULAR:
            continue
        payload = cast(Any, facet.payload)
        columns.extend((c["name"], c["native_type"]) for c in payload["columns"])
        for index in payload["indexes"]:
            index_fields.extend(index["columns"])
    return columns, index_fields


def search_postings(
    entry: CatalogEntry,
    relationships: Iterable[CatalogRelationship],
) -> tuple[SearchPosting, ...]:
    """Return the deduplicated, sorted postings that make one resource findable."""

    resource = entry.resource
    values = {
        ("resource_name", resource.name),
        ("native_identity", resource.native_identity),
        ("kind", f"{resource.kind.value} {resource.kind.value}s"),
    }
    columns, index_fields = tabular_parts(entry)
    values.update(("column", name) for name, _ in columns)
    values.update(("index_field", field) for field in index_fields)
    for relationship in relationships:
        for pair in relationship.field_pairs:
            if relationship.from_resource_id == resource.id:
                values.add(("relationship_field", pair.source_field))
            if relationship.to_resource_id == resource.id:
                values.add(("relationship_field", pair.target_field))
    return tuple(
        SearchPosting(kind, posting_field_name(kind, value), value)
        for kind, value in sorted(values)
    )


def search_words(postings: Iterable[SearchPosting]) -> str:
    """Return the space-bounded, sorted distinct tokens the LIKE lookup scans."""

    tokens = {
        token
        for posting in postings
        for token in normalize_search_text(posting.value).tokens
    }
    return " " + " ".join(sorted(tokens)) + " "


def search_key(value: str) -> str:
    return normalize_search_text(value).complete
