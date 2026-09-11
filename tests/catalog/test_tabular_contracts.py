"""Component-owned tests split from ``test_schema_slices.py``."""

from __future__ import annotations

from tests.catalog._schema_support import (
    Agent,
    CatalogStoreError,
    Path,
    SQLiteSource,
    TabularColumn,
    TabularFacet,
    TabularIndex,
    _tabular_decoder_fixture,
    pytest,
    replace,
    sqlite3,
    workspace_for,
)


def test_tabular_payload_decoders_round_trip_normalized_typed_records():
    facet = _tabular_decoder_fixture()

    assert TabularColumn.from_payload(facet.columns[0].to_payload()) == facet.columns[0]
    assert TabularIndex.from_payload(facet.indexes[0].to_payload()) == facet.indexes[0]
    assert TabularFacet.from_payload(facet.to_payload()) == facet
    assert tuple(column.name for column in facet.columns) == (
        "account_id",
        "tenant_id",
    )


def test_tabular_payload_decoders_require_exact_fields_and_scalar_types():
    facet = _tabular_decoder_fixture()
    column_payload = facet.columns[0].to_payload()
    index_payload = facet.indexes[0].to_payload()
    facet_payload = facet.to_payload()

    with pytest.raises(ValueError, match="exact current fields"):
        TabularColumn.from_payload(
            {key: value for key, value in column_payload.items() if key != "name"}
        )
    with pytest.raises(ValueError, match="exact current fields"):
        TabularColumn.from_payload({**column_payload, "unexpected": None})
    with pytest.raises(TypeError, match="nullable"):
        TabularColumn.from_payload({**column_payload, "nullable": 1})
    with pytest.raises(ValueError, match="exact current fields"):
        TabularIndex.from_payload(
            {key: value for key, value in index_payload.items() if key != "kind"}
        )
    with pytest.raises(ValueError, match="exact current fields"):
        TabularIndex.from_payload({**index_payload, "unexpected": None})
    with pytest.raises(TypeError, match="unique"):
        TabularIndex.from_payload({**index_payload, "unique": "yes"})
    with pytest.raises(ValueError, match="exact current fields"):
        TabularFacet.from_payload(
            {
                key: value
                for key, value in facet_payload.items()
                if key != "row_count_estimate"
            }
        )
    with pytest.raises(ValueError, match="exact current fields"):
        TabularFacet.from_payload({**facet_payload, "unexpected": None})
    with pytest.raises(TypeError, match="columns must be an array"):
        TabularFacet.from_payload({**facet_payload, "columns": "not-an-array"})


def test_tabular_payload_decoders_reject_bool_for_integer_fields():
    facet = _tabular_decoder_fixture()
    column_payload = facet.columns[0].to_payload()

    with pytest.raises(ValueError, match="ordinal"):
        TabularColumn.from_payload({**column_payload, "ordinal": True})
    with pytest.raises(ValueError, match="primary_key_ordinal"):
        TabularColumn.from_payload({**column_payload, "primary_key_ordinal": True})
    with pytest.raises(ValueError, match="row_count_estimate"):
        TabularFacet.from_payload({**facet.to_payload(), "row_count_estimate": True})


def test_tabular_payload_decoder_rejects_duplicate_columns_and_ordinals():
    facet = _tabular_decoder_fixture()
    first, second = (column.to_payload() for column in facet.columns)

    with pytest.raises(ValueError, match="duplicate names"):
        TabularFacet.from_payload(
            {
                **facet.to_payload(),
                "columns": (first, {**second, "name": first["name"]}),
            }
        )
    with pytest.raises(ValueError, match="duplicate ordinals"):
        TabularFacet.from_payload(
            {
                **facet.to_payload(),
                "columns": (first, {**second, "ordinal": first["ordinal"]}),
            }
        )


def test_tabular_payload_decoder_rejects_invalid_primary_key_order():
    facet = _tabular_decoder_fixture()
    first, second = (column.to_payload() for column in facet.columns)

    with pytest.raises(ValueError, match="contiguous from one"):
        TabularFacet.from_payload(
            {
                **facet.to_payload(),
                "columns": (
                    first,
                    {**second, "primary_key_ordinal": 3},
                ),
            }
        )


def test_tabular_payload_decoder_rejects_duplicate_and_unknown_column_indexes():
    facet = _tabular_decoder_fixture()
    index = facet.indexes[0].to_payload()

    with pytest.raises(ValueError, match="duplicate names"):
        TabularFacet.from_payload({**facet.to_payload(), "indexes": (index, index)})
    with pytest.raises(ValueError, match="unknown columns"):
        TabularFacet.from_payload(
            {
                **facet.to_payload(),
                "indexes": ({**index, "columns": ("missing_column",), "unique": True},),
            }
        )


async def test_catalog_boundary_rejects_malformed_tabular_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "validation-malformed.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY)")
    agent = await Agent.create(
        "catalog-validation-malformed",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        store = agent._embedded._store
        service = agent._embedded._catalog_service
        refs = await store.list_current_snapshot_refs(agent.id, (source.id,))
        assert len(refs) == 1
        snapshot = await store.load_current_snapshot(refs[0])
        assert snapshot is not None
        facet = next(item for item in snapshot.facets if item.kind.value == "tabular")
        malformed_facet = replace(
            facet,
            payload={**facet.payload, "unexpected": None},
        )
        malformed_snapshot = replace(
            snapshot,
            facets=tuple(
                malformed_facet if item is facet else item for item in snapshot.facets
            ),
        )

        async def load_malformed(_ref):
            return malformed_snapshot

        service._source_indexes.clear()
        monkeypatch.setattr(store, "load_current_snapshot", load_malformed)
        with pytest.raises(CatalogStoreError, match="invalid structure"):
            await service.tabular_resources(agent.id, source.id)
    finally:
        await agent.close()
