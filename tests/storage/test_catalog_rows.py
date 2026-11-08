"""The catalog is stored as one row per resource and read back from those rows."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from daita.adapters.models import SourceRegistration, source_registration_id
from daita.catalog.models import (
    CatalogChangeKind,
    CatalogEntry,
    CatalogResource,
    CatalogResourceRevision,
    CatalogTier,
    ResourceBatch,
    ResourceKind,
    Sensitivity,
    catalog_resource_id,
)
from daita.catalog.search_text import search_key
from daita.storage import catalog_rows, sql
from daita.storage.sqlite import SQLiteStateStore
from tests.support.catalog_records import (
    OBSERVED_AT,
    SHOP,
    SHOP_REFERENCES,
    Table,
    batches,
    entries,
    exercise_keeper_state,
    snapshot,
)

AGENT = "agent"
SOURCE = "source:one"


@pytest.fixture
async def store(tmp_path: Path):
    opened = await SQLiteStateStore.open(tmp_path / "state.db")
    yield opened
    await opened.close()


def shop(sync_id: str, tables=SHOP, references=SHOP_REFERENCES):
    return snapshot(AGENT, SOURCE, sync_id, tables, references)


async def commit(store, value, **options):
    return await store.commit_catalog(value.sync, batches(value), **options)


def rows(tmp_path: Path, query: str) -> list[tuple]:
    with sqlite3.connect(tmp_path / "state.db") as connection:
        return connection.execute(query).fetchall()


def inventory_entry(identity: str, sync_id: str) -> CatalogEntry:
    resource_id = catalog_resource_id(SOURCE, ResourceKind.TABLE, identity)
    revision = CatalogResourceRevision.build(
        resource_id=resource_id, sync_id=sync_id, observed_at=OBSERVED_AT
    )
    resource = CatalogResource.build(
        agent_id=AGENT,
        source_id=SOURCE,
        native_identity=identity,
        external_uri=f"sqlite:///{identity}",
        kind=ResourceKind.TABLE,
        name=identity.split(".")[-1],
        sensitivity=Sensitivity.UNKNOWN,
        revision=revision,
        first_observed_at=OBSERVED_AT,
        last_observed_at=OBSERVED_AT,
    )
    return CatalogEntry(resource, revision, (), detailed=False)


def inventory(sync_id: str, *identities: str, truncated: bool = False):
    return ResourceBatch(
        tier=CatalogTier.INVENTORY,
        entries=tuple(inventory_entry(item, sync_id) for item in identities),
        truncated=truncated,
    )


def identities(entries_) -> list[str]:
    return sorted(item.resource.native_identity for item in entries_)


async def test_a_pass_reads_back_with_its_ids_and_revisions(store) -> None:
    first = shop("sync-1")
    summary = await commit(store, first)

    assert (summary.added, summary.changed, summary.removed) == (4, 0, 0)
    stored = await store.list_entries(AGENT, SOURCE)
    assert {item.resource.id: item for item in stored} == {
        item.resource.id: item for item in entries(first)
    }
    customers = next(
        i for i in stored if i.resource.native_identity == "main.customers"
    )
    assert await store.load_entry(AGENT, customers.resource.id) == customers
    assert await store.load_revision(
        AGENT, customers.resource.id, customers.revision.revision
    )
    assert (
        await store.load_revision(AGENT, customers.resource.id, "sha256:" + "0" * 64)
        is None
    )
    assert await store.current_syncs(AGENT, ()) == (first.sync,)
    related = (
        await store.load_incident_relationships(AGENT, (customers.resource.id,))
    )[customers.resource.id]
    assert [item.id for item in related] == [first.relationships[0].id]
    summary_counts = await store.summarize_catalog(AGENT, (SOURCE,))
    assert (summary_counts.resource_count, summary_counts.relationship_count) == (4, 2)


async def test_an_unchanged_pass_writes_no_resource_or_change_rows(
    store, tmp_path: Path
) -> None:
    await commit(store, shop("sync-1"))
    tables = (
        "catalog_resources",
        "catalog_relationships",
        "catalog_search",
        "catalog_changes",
    )
    before = {
        name: rows(tmp_path, f"SELECT * FROM {name} ORDER BY 1, 2, 3")
        for name in tables
    }

    summary = await commit(store, shop("sync-2"))

    assert (summary.added, summary.changed, summary.removed) == (0, 0, 0)
    assert before == {
        name: rows(tmp_path, f"SELECT * FROM {name} ORDER BY 1, 2, 3")
        for name in tables
    }
    assert [s.id for s in await store.current_syncs(AGENT, ())] == ["sync-2"]


async def test_a_changed_column_records_one_change_with_both_sides(store) -> None:
    await commit(store, shop("sync-1"))
    wider = tuple(
        (
            replace(t, columns=(*t.columns, "note"))
            if t.native_identity == "main.orders"
            else t
        )
        for t in SHOP
    )

    summary = await commit(store, shop("sync-2", wider))

    assert (summary.added, summary.changed, summary.removed) == (0, 1, 0)
    (change,) = [
        c
        for c in await store.list_catalog_changes(AGENT, SOURCE)
        if c.kind is CatalogChangeKind.CHANGED
    ]
    assert change.native_identity == "main.orders"
    assert change.before is not None and change.after is not None
    assert [name for name, _ in change.after.columns] == [
        "id",
        "customer_id",
        "total",
        "note",
    ]
    assert [name for name, _ in change.before.columns] == ["id", "customer_id", "total"]
    assert change.before.revision != change.after.revision
    stored = await store.load_entry(AGENT, change.resource_id)
    assert stored is not None
    assert stored.resource.first_observed_at == OBSERVED_AT


async def test_an_inventory_entry_never_downgrades_a_described_resource(store) -> None:
    value = shop("sync-1")
    await commit(store, value)
    summary = await store.commit_catalog(
        replace(value.sync, id="sync-2"),
        (
            inventory(
                "sync-2",
                *(
                    i.resource.native_identity
                    for i in entries(value)
                    if i.resource.kind is ResourceKind.TABLE
                ),
            ),
        ),
    )

    assert (summary.added, summary.changed) == (0, 0)
    stored = await store.list_entries(AGENT, SOURCE)
    assert all(item.detailed for item in stored if item.facets)
    assert len([item for item in stored if item.facets]) == 3


async def test_a_listing_then_a_description_fills_in_the_row(store) -> None:
    value = shop("sync-1")
    first = replace(value.sync, resource_count=0)
    await store.commit_catalog(
        first, (inventory("sync-1", "main.orders", "main.customers"),)
    )
    assert not any(i.detailed for i in await store.list_entries(AGENT, SOURCE))

    await commit(store, shop("sync-2"))

    stored = await store.list_entries(AGENT, SOURCE)
    orders = next(i for i in stored if i.resource.native_identity == "main.orders")
    assert orders.detailed and orders.facets


async def test_describing_a_listed_row_is_a_fill_and_not_a_change(store) -> None:
    value = shop("sync-1")
    first = replace(value.sync, resource_count=0)
    await store.commit_catalog(
        first, (inventory("sync-1", "main.orders", "main.customers"),)
    )

    summary = await commit(store, shop("sync-2"))

    assert summary.changed == 0
    changes = await store.list_catalog_changes(AGENT, SOURCE)
    assert {item.kind for item in changes} == {CatalogChangeKind.ADDED}
    assert len([i for i in await store.list_entries(AGENT, SOURCE) if i.facets]) == 3


async def test_a_new_name_or_object_id_on_a_same_revision_row_is_rewritten(
    store,
) -> None:
    value = shop("sync-1")
    await commit(store, value)
    before = len(await store.list_catalog_changes(AGENT, SOURCE))

    def renamed(item: CatalogEntry) -> CatalogEntry:
        if item.resource.native_identity != "main.orders":
            return item
        return replace(
            item,
            resource=replace(item.resource, name="Orders"),
            source_object_id="16384",
        )

    (batch,) = batches(value)
    again = replace(batch, entries=tuple(renamed(item) for item in batch.entries))
    summary = await store.commit_catalog(replace(value.sync, id="sync-2"), (again,))

    assert (summary.added, summary.changed, summary.removed) == (0, 0, 0)
    stored = await store.list_entries(AGENT, SOURCE)
    orders = next(i for i in stored if i.resource.native_identity == "main.orders")
    assert (orders.resource.name, orders.source_object_id) == ("Orders", "16384")
    assert len(await store.list_catalog_changes(AGENT, SOURCE)) == before


async def test_a_truncated_listing_removes_nothing(store) -> None:
    await commit(store, shop("sync-1"))
    value = shop("sync-2")
    await store.commit_catalog(
        value.sync, (inventory("sync-2", "main.orders", truncated=True),)
    )

    assert len(await store.list_entries(AGENT, SOURCE)) == 4


async def test_truncated_detail_keeps_relationships_it_did_not_restate(store) -> None:
    value = shop("sync-1")
    await commit(store, value)
    partial = shop("sync-2", SHOP[:2], SHOP_REFERENCES[:1])
    cut = ResourceBatch(
        tier=CatalogTier.DETAIL,
        entries=entries(partial)[:2],
        relationships=partial.relationships,
        truncated=True,
    )
    await store.commit_catalog(
        partial.sync,
        (inventory("sync-2", *(t.native_identity for t in SHOP), "main"), cut),
    )

    kept = await store.summarize_catalog(AGENT, (SOURCE,))
    assert kept.relationship_count == 2
    assert len(await store.list_entries(AGENT, SOURCE)) == 4


async def test_removed_tables_leave_with_their_relationships_and_search_rows(
    store, tmp_path: Path
) -> None:
    await commit(store, shop("sync-1"))

    summary = await commit(store, shop("sync-2", SHOP[:2], SHOP_REFERENCES[:1]))

    assert summary.removed == 1
    assert identities(await store.list_entries(AGENT, SOURCE)) == [
        "main",
        "main.customers",
        "main.orders",
    ]
    assert rows(tmp_path, "SELECT COUNT(*) FROM catalog_relationships") == [(1,)]
    assert rows(tmp_path, "SELECT COUNT(*) FROM catalog_search") == [(3,)]
    removed = [
        c
        for c in await store.list_catalog_changes(AGENT, SOURCE)
        if c.kind is CatalogChangeKind.REMOVED
    ]
    assert [c.native_identity for c in removed] == ["main.invoices"]
    assert removed[0].before is not None and removed[0].after is None


async def test_search_rows_follow_relationship_field_changes(store) -> None:
    await commit(store, shop("sync-1"))

    def fields(found, identity):
        row = next(r for r in found if r.native_identity == identity)
        return {p.value for p in row.postings if p.field_kind == "relationship_field"}

    found = await store.search_catalog(AGENT, (), terms=("invoices", "orders"))
    assert fields(found, "main.invoices") == {"order_id"}

    await commit(store, shop("sync-2", SHOP, SHOP_REFERENCES[:1]))

    found = await store.search_catalog(AGENT, (), terms=("invoices", "orders"))
    assert fields(found, "main.invoices") == set()
    assert fields(found, "main.orders") == {"customer_id"}


async def test_search_matches_word_prefixes_and_exact_names(store) -> None:
    await commit(store, shop("sync-1"))

    def names(found):
        return sorted(r.name for r in found)

    assert names(await store.search_catalog(AGENT, (), terms=("cust",))) == [
        "customers",
        "orders",
    ]
    assert names(await store.search_catalog(AGENT, (), terms=("ustomer",))) == []
    assert names(await store.search_catalog(AGENT, (SOURCE,), terms=("name",))) == [
        "customers"
    ]
    assert (
        names(await store.search_catalog(AGENT, ("source:other",), terms=("cust",)))
        == []
    )
    anchored = await store.search_catalog(
        AGENT, (), terms=(), anchor_keys=(search_key("main.invoices"),)
    )
    assert names(anchored) == ["invoices"]
    by_name = await store.search_catalog(
        AGENT, (), terms=(), anchor_keys=(search_key("orders"),)
    )
    assert names(by_name) == ["orders"]
    assert await store.search_catalog(AGENT, (), terms=()) == ()
    assert len(await store.search_catalog(AGENT, (), terms=None)) == 4


async def test_a_reader_during_a_commit_sees_the_whole_old_state(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    await commit(store, shop("sync-1"))
    reached, release = threading.Event(), threading.Event()
    real = sql._commit_catalog_transaction

    def paused(connection):
        reached.set()
        release.wait(10)
        real(connection)

    monkeypatch.setattr(sql, "_commit_catalog_transaction", paused)
    writing = asyncio.create_task(commit(store, shop("sync-2", SHOP[:1], ())))
    await asyncio.to_thread(reached.wait, 10)
    try:
        during = await store.list_entries(AGENT, SOURCE)
        sync = await store.current_syncs(AGENT, ())
    finally:
        release.set()
    await writing

    assert len(during) == 4
    assert [s.id for s in sync] == ["sync-1"]
    assert identities(await store.list_entries(AGENT, SOURCE)) == [
        "main",
        "main.customers",
    ]


async def test_a_damaged_row_is_skipped_and_its_source_reported_once(
    store, tmp_path: Path
) -> None:
    await commit(store, shop("sync-1"))
    with sqlite3.connect(tmp_path / "state.db") as connection:
        connection.execute(
            "UPDATE catalog_resources SET data = '{not json' WHERE native_identity = ?",
            ("main.orders",),
        )

    assert await store.take_unreadable_catalog_sources(AGENT) == ()
    listed = await store.list_entries(AGENT, SOURCE)
    assert "main.orders" not in identities(listed) and len(listed) == 3
    assert await store.take_unreadable_catalog_sources(AGENT) == (SOURCE,)
    assert await store.take_unreadable_catalog_sources(AGENT) == ()

    await commit(store, shop("sync-2"))
    assert len(await store.list_entries(AGENT, SOURCE)) == 4


async def test_tables_load_by_native_name_within_their_source(
    store, tmp_path: Path
) -> None:
    await commit(store, shop("sync-1"))
    with sqlite3.connect(tmp_path / "state.db") as connection:
        connection.execute(
            "UPDATE catalog_resources SET data = '{not json' WHERE native_identity = ?",
            ("main.orders",),
        )

    found = await store.load_entries_by_identity(
        AGENT, SOURCE, ("main.customers", "main.orders", "main.nothing")
    )
    assert identities(found) == ["main.customers"]  # damaged and unknown: absent
    assert await store.take_unreadable_catalog_sources(AGENT) == (SOURCE,)
    assert (
        await store.load_entries_by_identity(
            AGENT, "another-source", ("main.customers",)
        )
        == ()
    )
    assert await store.load_entries_by_identity(AGENT, SOURCE, ()) == ()


async def test_names_list_from_columns_alone_and_can_be_cut_to_the_smallest_ids(
    store, tmp_path: Path
) -> None:
    await commit(store, shop("sync-1"))
    expected = identities(await store.list_entries(AGENT, SOURCE))
    with sqlite3.connect(tmp_path / "state.db") as connection:
        connection.execute("UPDATE catalog_resources SET data = '{not json'")

    listed = await store.list_resource_names(AGENT, (SOURCE,))
    assert sorted(name for _, _, name in listed) == expected
    assert await store.list_resource_names(AGENT, (SOURCE,), limit=2) == tuple(
        sorted(listed)[:2]
    )
    others = await store.list_resource_names(AGENT, (), other_than=ResourceKind.TABLE)
    assert [name for _, _, name in others] == ["main"]
    assert await store.take_unreadable_catalog_sources(AGENT) == ()  # nothing decoded


async def test_old_syncs_and_changes_are_pruned_to_their_bounds(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(catalog_rows, "KEPT_CHANGES", 4)
    for number in range(25):
        value = shop(f"sync-{number:02d}")
        value = replace(
            value,
            sync=replace(
                value.sync,
                started_at=OBSERVED_AT + timedelta(minutes=number),
                completed_at=OBSERVED_AT + timedelta(minutes=number),
            ),
        )
        extra = tuple(Table(f"main.t{n}", ("id",)) for n in range(number % 3))
        await store.commit_catalog(
            value.sync,
            batches(shop(f"sync-{number:02d}", (*SHOP, *extra), SHOP_REFERENCES)),
        )
        await store.record_sync(value.sync)

    kept = rows(tmp_path, "SELECT id FROM syncs ORDER BY id")
    assert len(kept) <= 20 and ("sync-24",) in kept
    sequences = [
        c.sequence for c in await store.list_catalog_changes(AGENT, SOURCE, limit=500)
    ]
    assert 0 < len(sequences) <= 4
    state = await store.load_source_state(AGENT, SOURCE)
    assert state is not None and state.current_sync_id == "sync-24"

    with sqlite3.connect(tmp_path / "state.db") as connection:
        connection.execute(
            "UPDATE catalog_changes SET changed_at = '2000-01-01T00:00:00Z'"
        )
    await commit(store, shop("sync-25"))
    assert await store.list_catalog_changes(AGENT, SOURCE) == ()

    with pytest.raises(ValueError):
        await store.list_catalog_changes(AGENT, SOURCE, limit=501)


async def test_a_relationship_to_a_missing_table_is_refused(store) -> None:
    value = shop("sync-1", SHOP[:1], ())
    orphan = shop("sync-1").relationships[0]
    bad = ResourceBatch(
        tier=CatalogTier.DETAIL, entries=entries(value), relationships=(orphan,)
    )
    with pytest.raises(ValueError):
        await store.commit_catalog(value.sync, (bad,))
    assert await store.list_entries(AGENT, SOURCE) == ()


async def test_the_keeper_state_contract_holds_on_sqlite(store) -> None:
    source = source_registration_id(AGENT, "sqlite", "db")
    registration = SourceRegistration(
        id=source,
        agent_id=AGENT,
        adapter_id="sqlite",
        native_identity="db",
        display_name="Shop",
        configuration={},
        attached_at=OBSERVED_AT,
    )
    await exercise_keeper_state(
        store, AGENT, source, lambda: store.register_source(registration)
    )
