"""The shared ``SQLStateStore`` runs on a real PostgreSQL inside a rolled-back transaction.

Needs the disposable postgres-large fixture server
(tests/fixtures/postgres-large/compose.yaml):

    DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 DAITA_LARGE_POSTGRES_ADMIN_PASSWORD=... \
    pytest tests/live/storage/test_postgresql_state_store.py
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from daita.adapters.models import SourceRegistration, source_registration_id
from daita.catalog.models import (
    CatalogChangeKind,
    CatalogLink,
    CatalogSync,
    CatalogSyncStatus,
    RelationshipFieldPair,
    ResourceKind,
    catalog_link_id,
    catalog_resource_id,
)
from daita.catalog.search_text import search_key, tabular_parts
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
)
from daita.storage import catalog_rows
from tests.support.catalog_records import (
    OBSERVED_AT,
    SHOP,
    SHOP_REFERENCES,
    Table,
    batches,
    exercise_keeper_state,
    snapshot,
)
from tests.support.postgresql_state import (
    _PostgresDatabase,
    postgresql_state,
    postgresql_state_store,
)

psycopg2 = pytest.importorskip("psycopg2", exc_type=ImportError)

pytestmark = [
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_LARGE_FIXTURE") != "1"
        or not os.environ.get("DAITA_LARGE_POSTGRES_ADMIN_PASSWORD"),
        reason=(
            "Start tests/fixtures/postgres-large/compose.yaml, then set "
            "DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 and "
            "DAITA_LARGE_POSTGRES_ADMIN_PASSWORD."
        ),
    ),
]

_NOW = datetime(2026, 11, 7, tzinfo=UTC)
_AGENT = "agent-1"
_SOURCE = source_registration_id(_AGENT, "postgresql", "db")
_OTHER = "source:sha256:" + "b" * 64
_FROM = "catalog-resource:sha256:" + "1" * 64
_TO = "catalog-resource:sha256:" + "2" * 64


@pytest.fixture
def database() -> Iterator[_PostgresDatabase]:
    with postgresql_state(os.environ["DAITA_LARGE_POSTGRES_ADMIN_PASSWORD"]) as db:
        yield db


def test_projected_schema_installs(database: _PostgresDatabase) -> None:
    with database.connect() as connection:
        assert connection.table_exists("sources")
        assert connection.table_exists("catalog_links")
        assert connection.table_exists("jobs")
        assert not connection.table_exists("learning_candidates")


async def test_source_round_trips(database: _PostgresDatabase) -> None:
    store = postgresql_state_store(database)
    source = SourceRegistration(
        id=_SOURCE,
        agent_id=_AGENT,
        adapter_id="postgresql",
        native_identity="db",
        display_name="Orders",
        configuration={},
        attached_at=_NOW,
    )
    assert await store.register_source(source) == source
    assert await store.load_source(_AGENT, _SOURCE) == source
    assert await store.list_sources(_AGENT) == (source,)


async def test_catalog_links_and_sync_round_trip(database: _PostgresDatabase) -> None:
    store = postgresql_state_store(database)
    pairs = (RelationshipFieldPair("a", "b", 0),)
    link = CatalogLink(
        id=catalog_link_id(
            from_resource_id=_FROM, to_resource_id=_TO, field_pairs=pairs
        ),
        agent_id=_AGENT,
        from_source_id=_SOURCE,
        from_resource_id=_FROM,
        to_source_id=_OTHER,
        to_resource_id=_TO,
        field_pairs=pairs,
        declared_at=_NOW,
        declared_revisions=("sha256:" + "c" * 64, "sha256:" + "d" * 64),
    )
    await store.save_catalog_link(link)
    await store.save_catalog_link(link)
    assert await store.list_catalog_links(_AGENT) == (link,)
    assert await store.delete_catalog_link(_AGENT, link.id)
    assert not await store.delete_catalog_link(_AGENT, link.id)
    assert await store.list_catalog_links(_AGENT) == ()

    sync = CatalogSync(
        id="sync-1",
        agent_id=_AGENT,
        source_id=_SOURCE,
        adapter_id="postgresql",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=_NOW,
        completed_at=_NOW,
        resource_count=3,
    )
    assert await store.record_sync(sync) == sync
    assert await store.load_sync(_AGENT, "sync-1") == sync


def test_everything_is_rolled_back() -> None:
    password = os.environ["DAITA_LARGE_POSTGRES_ADMIN_PASSWORD"]
    with postgresql_state(password) as database:
        schema = database.schema
    fresh = psycopg2.connect(
        host="127.0.0.1",
        port=55433,
        user="postgres",
        password=password,
        dbname="postgres",
    )
    try:
        with fresh.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,))
            assert cursor.fetchone() is None
    finally:
        fresh.close()


def _shop(sync_id: str, tables=SHOP, references=SHOP_REFERENCES):
    return snapshot(_AGENT, _SOURCE, sync_id, tables, references)


async def _commit(store, value):
    return await store.commit_catalog(value.sync, batches(value))


async def test_catalog_rows_commit_change_and_search(
    database: _PostgresDatabase,
) -> None:
    store = postgresql_state_store(database)
    first = await _commit(store, _shop("sync-1"))
    assert (first.added, first.changed, first.removed) == (4, 0, 0)
    assert len(await store.list_entries(_AGENT, _SOURCE)) == 4
    counts = await store.summarize_catalog(_AGENT, (_SOURCE,))
    assert (counts.resource_count, counts.relationship_count) == (4, 2)

    same = await _commit(store, _shop("sync-2"))
    assert (same.added, same.changed, same.removed) == (0, 0, 0)
    assert [s.id for s in await store.current_syncs(_AGENT, ())] == ["sync-2"]

    wider = tuple(
        (
            replace(t, columns=(*t.columns, "note"))
            if t.native_identity == "main.orders"
            else t
        )
        for t in SHOP
    )
    changed = await _commit(store, _shop("sync-3", wider))
    assert (changed.added, changed.changed, changed.removed) == (0, 1, 0)
    (change,) = [
        c
        for c in await store.list_catalog_changes(_AGENT, _SOURCE)
        if c.kind is CatalogChangeKind.CHANGED
    ]
    assert change.before is not None and change.after is not None
    assert len(change.after.columns) == len(change.before.columns) + 1

    def names(found):
        return sorted(row.name for row in found)

    assert names(await store.search_catalog(_AGENT, (), terms=("cust",))) == [
        "customers",
        "orders",
    ]
    assert (
        names(await store.search_catalog(_AGENT, (_SOURCE,), terms=("ustomer",))) == []
    )
    anchored = await store.search_catalog(
        _AGENT, (), terms=(), anchor_keys=(search_key("main.invoices"),)
    )
    assert names(anchored) == ["invoices"]
    assert len(await store.search_catalog(_AGENT, (), terms=None)) == 4
    listed = await store.list_entries(_AGENT, _SOURCE)
    by_name = await store.load_entries_by_identity(
        _AGENT, _SOURCE, ("main.orders", "main.missing")
    )
    assert [item.resource.native_identity for item in by_name] == ["main.orders"]
    related = await store.load_incident_relationships(
        _AGENT, tuple(item.resource.id for item in listed)
    )
    assert sum(len(found) for found in related.values()) == 4  # 2 links, 2 ends each

    removed = await _commit(store, _shop("sync-4", wider[:2], SHOP_REFERENCES[:1]))
    assert removed.removed == 1
    counts = await store.summarize_catalog(_AGENT, (_SOURCE,))
    assert (counts.resource_count, counts.relationship_count) == (3, 1)


async def test_a_renamed_table_keeps_its_note_on_postgresql(
    database: _PostgresDatabase,
) -> None:
    store = postgresql_state_store(database)
    await store.register_source(
        SourceRegistration(
            id=_SOURCE,
            agent_id=_AGENT,
            adapter_id="postgresql",
            native_identity="db",
            display_name="Shop",
            configuration={},
            attached_at=_NOW,
        )
    )
    plain = SHOP[:2]
    objects = {"main.customers": "100", "main.orders": "200", "main.purchases": "200"}

    async def commit_pass(sync_id, tables):
        value = snapshot(_AGENT, _SOURCE, sync_id, tables, ())
        (batch,) = batches(value)
        described = tuple(
            replace(e, source_object_id=objects.get(e.resource.native_identity))
            for e in batch.entries
        )
        return await store.commit_catalog(
            value.sync, (replace(batch, entries=described),)
        )

    await commit_pass("sync-1", plain)
    (orders,) = [
        e
        for e in await store.list_entries(_AGENT, _SOURCE)
        if e.resource.native_identity == "main.orders"
    ]
    old = orders.resource.id
    note = SemanticAnnotation(
        id="gross",
        agent_id=_AGENT,
        subject=SemanticSubject(
            (_SOURCE,), (old,), (SemanticFieldReference(old, "total"),)
        ),
        kind=SemanticKind.METRIC_DEFINITION,
        statement="Gross.",
        evidence=(
            SemanticEvidence(
                SemanticEvidenceKind.USER_ASSERTION, "run-1", message_position=0
            ),
        ),
        catalog_revisions=(
            ResourceRevisionBinding(
                old,
                orders.resource.current_revision,
                tuple(t for t in tabular_parts(orders)[0] if t[0] == "total"),
            ),
        ),
        created_at=OBSERVED_AT,
        confirmed_at=OBSERVED_AT,
    )
    await store.save_semantic_annotation(_AGENT, note)
    before = len(await store.list_catalog_changes(_AGENT, _SOURCE))

    renamed = tuple(
        (
            replace(t, native_identity="main.purchases")
            if t.native_identity == "main.orders"
            else t
        )
        for t in plain
    )
    summary = await commit_pass("sync-2", renamed)

    assert (summary.added, summary.changed, summary.removed) == (0, 1, 0)
    new = catalog_resource_id(_SOURCE, ResourceKind.TABLE, "main.purchases")
    (kept,) = await store.list_semantic_annotations(_AGENT)
    assert kept.statement == note.statement
    assert kept.subject.resource_ids == (new,)
    assert kept.subject.fields == (SemanticFieldReference(new, "total"),)
    newest = (await store.list_catalog_changes(_AGENT, _SOURCE))[
        : len(await store.list_catalog_changes(_AGENT, _SOURCE)) - before
    ]
    (change,) = newest
    assert change.before is not None and change.after is not None
    assert (change.kind, change.before.name, change.after.name) == (
        CatalogChangeKind.CHANGED,
        "orders",
        "purchases",
    )

    (purchases,) = [
        e
        for e in await store.list_entries(_AGENT, _SOURCE)
        if e.resource.native_identity == "main.purchases"
    ]
    widened = tuple(
        (
            replace(t, columns=(*t.columns, "memo"))
            if t.native_identity == "main.purchases"
            else t
        )
        for t in renamed
    )
    await commit_pass("sync-3", widened)
    changed = store.catalog_structure_changed
    assert not await changed(_AGENT, _SOURCE, new, purchases.resource.current_revision)
    assert await changed(_AGENT, _SOURCE, new, "sha256:" + "0" * 64)


async def test_catalog_rows_inventory_truncation_and_damage(
    database: _PostgresDatabase,
) -> None:
    store = postgresql_state_store(database)
    value = _shop("sync-1")
    await _commit(store, value)
    cut = replace(value.sync, id="sync-2")
    entries_ = batches(value)[0].entries
    await store.commit_catalog(
        cut,
        (
            replace(
                batches(value)[0],
                tier=batches(value)[0].tier,
                entries=entries_[:1],
                truncated=True,
            ),
        ),
    )
    assert len(await store.list_entries(_AGENT, _SOURCE)) == 4

    with database.connect() as connection:
        connection.execute(
            "UPDATE catalog_resources SET data = ? WHERE native_identity = ?",
            ("{not json", "main.orders"),
        )
        connection.commit()
    assert len(await store.list_entries(_AGENT, _SOURCE)) == 3
    assert await store.take_unreadable_catalog_sources(_AGENT) == (_SOURCE,)
    assert await store.take_unreadable_catalog_sources(_AGENT) == ()
    assert (
        await store.load_entries_by_identity(
            _AGENT, _SOURCE, ("main.orders", "main.customers")
        )
    )[0].resource.native_identity == "main.customers"
    assert await store.take_unreadable_catalog_sources(_AGENT) == (_SOURCE,)
    await _commit(store, _shop("sync-3"))
    assert len(await store.list_entries(_AGENT, _SOURCE)) == 4


async def test_catalog_rows_are_pruned(
    database: _PostgresDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(catalog_rows, "KEPT_CHANGES", 3)
    store = postgresql_state_store(database)
    for number in range(23):
        value = _shop(
            f"sync-{number:02d}",
            (*SHOP, *(Table(f"main.t{n}", ("id",)) for n in range(number % 3))),
        )
        sync = replace(
            value.sync,
            started_at=OBSERVED_AT + timedelta(minutes=number),
            completed_at=OBSERVED_AT + timedelta(minutes=number),
        )
        await store.commit_catalog(sync, batches(value))
    with database.connect() as connection:
        row = connection.execute("SELECT COUNT(*) FROM syncs").fetchone()
    assert row is not None
    assert row[0] <= 20
    assert await store.load_sync(_AGENT, "sync-22") is not None
    assert 0 < len(await store.list_catalog_changes(_AGENT, _SOURCE)) <= 3


async def test_keeper_state_contract_holds_on_postgresql(
    database: _PostgresDatabase,
) -> None:
    store = postgresql_state_store(database)
    registration = SourceRegistration(
        id=_SOURCE,
        agent_id=_AGENT,
        adapter_id="postgresql",
        native_identity="db",
        display_name="Orders",
        configuration={},
        attached_at=_NOW,
    )
    await exercise_keeper_state(
        store, _AGENT, _SOURCE, lambda: store.register_source(registration)
    )
