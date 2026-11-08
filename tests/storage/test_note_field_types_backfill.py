"""Upgraded notes keep the field types they were saved against, where still known."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pytest

from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticResourceFact,
    SemanticSubject,
    inspect_semantic_annotations,
)
from daita.storage.home_migrations import revision_0004
from daita.storage.sqlite_codecs import (
    decode_semantic_annotation,
    encode_catalog_snapshot,
    encode_semantic_annotation,
)
from daita.storage.sqlite_codecs.semantics import decode_revision_binding
from tests.support.catalog_records import Table, entries, snapshot

_NOW = datetime(2026, 11, 2, tzinfo=UTC)
_SOURCE = "source:shop"


def _note(
    note_id: str,
    resource_id: str,
    revision: str,
    *,
    field: str | None,
    bound_resource: str | None = None,
) -> str:
    return encode_semantic_annotation(
        SemanticAnnotation(
            id=note_id,
            agent_id="agent",
            subject=SemanticSubject(
                source_ids=(_SOURCE,),
                resource_ids=(resource_id,),
                fields=(SemanticFieldReference(resource_id, field),) if field else (),
            ),
            kind=SemanticKind.METRIC_DEFINITION,
            statement="Amount is in cents.",
            evidence=(
                SemanticEvidence(
                    SemanticEvidenceKind.USER_ASSERTION, "run-seed", message_position=0
                ),
            ),
            catalog_revisions=(
                ResourceRevisionBinding(bound_resource or resource_id, revision),
            ),
            created_at=_NOW,
            confirmed_at=_NOW,
            confirmed_by="local-user",
        )
    )


def _upgraded() -> tuple[sqlite3.Connection, str, str]:
    stored = snapshot(
        "agent",
        _SOURCE,
        "sync-1",
        (Table("main.orders", ("id", "amount")), Table("main.items", ("id",))),
        folder=None,
    )
    orders, _items = sorted(entries(stored), key=lambda item: item.resource.name)[::-1]
    absent = "catalog-resource:sha256:gone"
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE snapshots (agent_id TEXT, source_id TEXT, sync_id TEXT, data TEXT);"
        "CREATE TABLE semantic_annotations (agent_id TEXT, id TEXT, data TEXT);"
    )
    connection.execute(
        "INSERT INTO snapshots VALUES ('agent', ?, 'sync-1', ?)",
        (_SOURCE, encode_catalog_snapshot(stored)),
    )
    rid, revision = orders.resource.id, orders.resource.current_revision
    notes = {
        "named": _note("named", rid, revision, field="amount"),
        "none": _note("none", rid, revision, field=None),
        "stale": _note("stale", rid, "older-revision", field="amount"),
        "absent": _note("absent", absent, revision, field=None),
    }
    connection.executemany(
        "INSERT INTO semantic_annotations VALUES ('agent', ?, ?)", notes.items()
    )
    revision_0004._convert_catalog_snapshots(connection)
    revision_0004._backfill_note_field_types(connection)
    return connection, rid, revision


def _stored(connection: sqlite3.Connection) -> dict[str, SemanticAnnotation]:
    return {
        note_id: decode_semantic_annotation(data)
        for note_id, data in connection.execute(
            "SELECT id, data FROM semantic_annotations"
        )
    }


def test_notes_gain_types_only_where_their_revision_still_matches() -> None:
    connection, _rid, _revision = _upgraded()

    notes = _stored(connection)

    assert notes["named"].catalog_revisions[0].field_types == (("amount", "TEXT"),)
    assert notes["none"].catalog_revisions[0].field_types == ()
    assert notes["stale"].catalog_revisions[0].field_types is None
    assert notes["absent"].catalog_revisions[0].field_types is None
    assert (
        "field_types"
        not in json.loads(
            connection.execute(
                "SELECT data FROM semantic_annotations WHERE id = 'stale'"
            ).fetchone()[0]
        )["fields"]["catalog_revisions"][0]["fields"]
    )


def test_a_backfilled_note_survives_an_added_column_but_a_stale_one_stays_stale() -> (
    None
):
    connection, rid, revision = _upgraded()
    notes = _stored(connection)
    newer = SemanticResourceFact(
        resource_id=rid,
        source_id=_SOURCE,
        revision="newer-revision",
        field_names=("id", "amount", "note"),
        field_types=(("id", "TEXT"), ("amount", "TEXT"), ("note", "TEXT")),
    )
    current = SemanticResourceFact(
        resource_id=rid,
        source_id=_SOURCE,
        revision=revision,
        field_names=("id", "amount"),
        field_types=(("id", "TEXT"), ("amount", "TEXT")),
    )

    after_new_column = {
        view.annotation.id: view.state
        for view in inspect_semantic_annotations(
            (notes["named"], notes["none"], notes["stale"]), (newer,)
        )
    }
    unchanged = {
        view.annotation.id: view.state
        for view in inspect_semantic_annotations(
            (notes["named"], notes["stale"]), (current,)
        )
    }

    assert after_new_column["named"] is not SemanticAnnotationState.STALE
    assert after_new_column["none"] is not SemanticAnnotationState.STALE
    assert after_new_column["stale"] is SemanticAnnotationState.STALE
    assert unchanged["named"] is not SemanticAnnotationState.STALE
    assert unchanged["stale"] is SemanticAnnotationState.STALE


def test_old_bindings_still_decode_and_damaged_pairs_raise_the_codec_error() -> None:
    old: Any = {
        "__record__": "ResourceRevisionBinding",
        "fields": {"resource_id": "r", "revision": "v"},
    }
    assert decode_revision_binding(old).field_types is None

    for pair in (["only-name"], ["a", "b", "c"]):
        damaged = json.loads(json.dumps(old))
        damaged["fields"]["field_types"] = [pair]
        with pytest.raises(ValueError):
            decode_revision_binding(damaged)
