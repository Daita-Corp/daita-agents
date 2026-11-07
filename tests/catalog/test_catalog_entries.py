"""Catalog entries, batches and the per-resource search postings."""

from __future__ import annotations

from dataclasses import replace

import pytest

from daita.catalog.models import (
    CatalogChange,
    CatalogChangeKind,
    CatalogChangeSide,
    CatalogTier,
    ResourceBatch,
    ResourceKind,
)
from daita.catalog.search_text import (
    search_key,
    search_postings,
    search_words,
)
from tests.support.catalog_records import (
    OBSERVED_AT,
    SHOP,
    SHOP_REFERENCES,
    entries,
    snapshot,
)

_SOURCE = "source:one"


def _shop():
    return snapshot("agent", _SOURCE, "sync-1", SHOP, SHOP_REFERENCES)


def test_a_resource_is_findable_by_its_name_kind_columns_and_relationship_fields() -> (
    None
):
    shop = _shop()
    customers = next(
        item for item in entries(shop) if item.resource.name == "customers"
    )
    incident = [
        item
        for item in shop.relationships
        if customers.resource.id in (item.from_resource_id, item.to_resource_id)
    ]

    postings = search_postings(customers, incident)

    assert {(item.field_kind, item.field_name) for item in postings} == {
        ("resource_name", "name"),
        ("native_identity", "native_identity"),
        ("kind", "kind"),
        ("column", "column:id"),
        ("column", "column:CustomerName"),
        ("index_field", "index_field:CustomerName"),
        ("relationship_field", "relationship_field:id"),
    }
    words = search_words(postings).split()
    assert {"customers", "customer", "name", "id", "main", "table"} <= set(words)
    assert search_words(postings).startswith(" ")
    assert search_words(postings).endswith(" ")


def test_relationship_fields_are_searchable_only_from_the_end_that_owns_them() -> None:
    shop = _shop()
    by_name = {item.resource.name: item for item in entries(shop)}
    link = next(
        item
        for item in shop.relationships
        if item.from_resource_id == by_name["orders"].resource.id
        and item.to_resource_id == by_name["customers"].resource.id
    )

    def fields(name: str) -> set[str]:
        return {
            item.field_name
            for item in search_postings(by_name[name], [link])
            if item.field_kind == "relationship_field"
        }

    assert fields("orders") == {"relationship_field:customer_id"}
    assert fields("customers") == {"relationship_field:id"}
    assert fields("invoices") == set()
    assert search_key("Main.Customers") == "main.customers"


def test_an_entry_must_agree_with_its_own_revision_and_facets() -> None:
    entry = entries(_shop())[0]
    other = entries(_shop())[1]

    with pytest.raises(ValueError):
        replace(entry, revision=other.revision)
    with pytest.raises(ValueError):
        replace(entry, facets=())
    with pytest.raises(ValueError):
        replace(entry, detailed=False)
    bare = next(item for item in entries(_shop()) if not item.facets)
    assert replace(bare, detailed=False).facets == ()


def test_a_batch_keeps_its_tier_one_agent_and_one_source() -> None:
    shop = _shop()
    listed = entries(shop)
    bare = next(item for item in listed if not item.facets)
    inventory = replace(bare, detailed=False)

    assert ResourceBatch(CatalogTier.DETAIL, listed, shop.relationships)
    assert ResourceBatch(CatalogTier.INVENTORY, (inventory,))
    with pytest.raises(ValueError):
        ResourceBatch(CatalogTier.INVENTORY, listed)
    with pytest.raises(ValueError):
        ResourceBatch(CatalogTier.INVENTORY, (inventory,), shop.relationships)
    with pytest.raises(ValueError):
        ResourceBatch(CatalogTier.DETAIL, (inventory,))
    with pytest.raises(ValueError):
        ResourceBatch(CatalogTier.DETAIL, listed + listed[:1])
    other = snapshot("agent", "source:two", "sync-2", SHOP[:1], folder=None)
    with pytest.raises(ValueError):
        ResourceBatch(CatalogTier.DETAIL, listed + entries(other))


def test_a_change_names_only_the_sides_its_kind_has() -> None:
    entry = entries(_shop())[0]
    side = CatalogChangeSide(
        entry.resource.name, ResourceKind.TABLE, entry.resource.current_revision
    )

    def change(
        kind: CatalogChangeKind,
        before: CatalogChangeSide | None,
        after: CatalogChangeSide | None,
    ) -> CatalogChange:
        return CatalogChange(
            kind=kind,
            before=before,
            after=after,
            agent_id="agent",
            source_id=_SOURCE,
            sequence=1,
            sync_id="sync-1",
            changed_at=OBSERVED_AT,
            resource_id=entry.resource.id,
            native_identity=entry.resource.native_identity,
        )

    assert change(CatalogChangeKind.ADDED, None, side)
    assert change(CatalogChangeKind.REMOVED, side, None)
    with pytest.raises(ValueError):
        change(CatalogChangeKind.ADDED, side, side)
    with pytest.raises(ValueError):
        change(CatalogChangeKind.CHANGED, None, side)
