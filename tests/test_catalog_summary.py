from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

import daita.catalog.service as catalog_service
import daita.storage.sqlite as sqlite_store
from daita import Agent, CatalogSummary, SQLiteSource
from daita.adapters.models import SourceRegistration
from daita.catalog import (
    CatalogSearchRequest,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
)
from daita.catalog.models import CatalogSnapshotRef
from daita.catalog.protocols import CatalogStoreError
from daita.llm.models import FinishReason, ModelProfile, ModelResponse
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sqlite import SQLiteStateStore


def _database(path: Path, *, with_tables: bool = True) -> None:
    with sqlite3.connect(path) as connection:
        if not with_tables:
            return
        connection.executescript("""
            CREATE TABLE parent (id INTEGER PRIMARY KEY);
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER REFERENCES parent(id)
            );
            """)


def _snapshot_decode_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], int]:
    original_decode = sqlite_store.decode_catalog_snapshot
    counter_lock = threading.Lock()
    count = 0

    def counting_decode(value: str):
        nonlocal count
        with counter_lock:
            count += 1
        return original_decode(value)

    monkeypatch.setattr(sqlite_store, "decode_catalog_snapshot", counting_decode)
    return lambda: count


async def test_catalog_summary_aggregates_current_active_snapshots_and_latest_sync(
    tmp_path: Path,
):
    first_time = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    current_time = first_time

    def clock() -> datetime:
        return current_time

    first_database = tmp_path / "first.sqlite"
    second_database = tmp_path / "second.sqlite"
    _database(first_database)
    _database(second_database)
    agent = await Agent.create("summary", root=tmp_path, clock=clock)
    try:
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=0,
            resource_count=0,
            relationship_count=0,
            latest_successful_sync_completed_at=None,
            is_empty=True,
        )

        first = await agent.attach(SQLiteSource(first_database, name="First"))
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=1,
            resource_count=2,
            relationship_count=1,
            latest_successful_sync_completed_at=first_time,
            is_empty=False,
        )

        current_time = first_time + timedelta(minutes=5)
        second = await agent.attach(SQLiteSource(second_database, name="Second"))
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=2,
            resource_count=4,
            relationship_count=2,
            latest_successful_sync_completed_at=current_time,
            is_empty=False,
        )

        await agent.detach(first.id)
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=1,
            resource_count=2,
            relationship_count=1,
            latest_successful_sync_completed_at=current_time,
            is_empty=False,
        )
        await agent.detach(second.id)
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=0,
            resource_count=0,
            relationship_count=0,
            latest_successful_sync_completed_at=None,
            is_empty=True,
        )
    finally:
        await agent.close()


async def test_empty_successful_snapshot_is_not_ready_but_retains_sync_time(
    tmp_path: Path,
):
    completed_at = datetime(2026, 7, 22, 13, 0, tzinfo=UTC)
    database = tmp_path / "empty.sqlite"
    _database(database, with_tables=False)
    agent = await Agent.create("empty", root=tmp_path, clock=lambda: completed_at)
    try:
        await agent.attach(SQLiteSource(database))
        summary = await agent.catalog_summary()
        assert summary.active_source_count == 1
        assert summary.resource_count == 0
        assert summary.relationship_count == 0
        assert summary.latest_successful_sync_completed_at == completed_at
        assert summary.is_empty is True
    finally:
        await agent.close()


async def test_catalog_preview_contains_only_active_current_snapshot_truth(
    tmp_path: Path,
):
    first_database = tmp_path / "preview-first.sqlite"
    second_database = tmp_path / "preview-second.sqlite"
    with sqlite3.connect(first_database) as connection:
        connection.execute("CREATE TABLE first_only (id INTEGER PRIMARY KEY)")
    with sqlite3.connect(second_database) as connection:
        connection.execute("CREATE TABLE second_only (id INTEGER PRIMARY KEY)")
    agent = await Agent.create("preview-active", root=tmp_path)
    try:
        first = await agent.attach(SQLiteSource(first_database, name="First"))
        second = await agent.attach(SQLiteSource(second_database, name="Second"))
        preview = await agent.catalog_preview(limit=50)
        assert tuple(resource.name for resource in preview) == (
            "first_only",
            "second_only",
        )

        await agent.detach(first.id)
        assert all(
            key[:2] != (agent.id, first.id)
            for key in agent._embedded._store._decoded_catalog_snapshots
        )
        preview = await agent.catalog_preview(limit=50)
        assert tuple(resource.name for resource in preview) == ("second_only",)
        assert {resource.source_id for resource in preview} == {second.id}

        with sqlite3.connect(second_database) as connection:
            connection.execute("DROP TABLE second_only")
            connection.execute("CREATE TABLE refreshed_only (id INTEGER PRIMARY KEY)")
        await agent.refresh_source(second.id)
        preview = await agent.catalog_preview(limit=50)
        assert tuple(resource.name for resource in preview) == ("refreshed_only",)

        await agent.detach(second.id)
        assert all(
            key[:2] != (agent.id, second.id)
            for key in agent._embedded._store._decoded_catalog_snapshots
        )
        assert await agent.catalog_preview(limit=50) == ()
        assert (await agent.catalog_summary()).is_empty is True
    finally:
        await agent.close()


async def test_broad_catalog_discovery_matches_any_term_and_resource_kind(
    tmp_path: Path,
):
    database = tmp_path / "search.sqlite"
    _database(database)
    agent = await Agent.create("search", root=tmp_path)
    try:
        await agent.attach(SQLiteSource(database))
        result = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="tables views datasets schemas relationships",
                resource_kinds=(ResourceKind.TABLE, ResourceKind.VIEW),
                limit=50,
            )
        )

        assert tuple(hit.name for hit in result.hits) == ("child", "parent")
        assert result.total_matches == 2
        assert result.returned_count == 2
        assert all("kind" in hit.matched_fields for hit in result.hits)
    finally:
        await agent.close()


async def test_catalog_search_compiles_one_generation_once_for_concurrent_cold_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "compile-once.sqlite"
    _database(database)
    agent = await Agent.create("catalog-index-compile-once", root=tmp_path)
    try:
        await agent.attach(SQLiteSource(database))
        original_compile = catalog_service._compile_source_index
        compile_count = 0

        def counting_compile(snapshot):
            nonlocal compile_count
            compile_count += 1
            return original_compile(snapshot)

        monkeypatch.setattr(
            catalog_service,
            "_compile_source_index",
            counting_compile,
        )
        requests = tuple(
            CatalogSearchRequest(agent_id=agent.id, query="parent", limit=10)
            for _ in range(16)
        )
        results = await asyncio.gather(
            *(agent.search_catalog(request) for request in requests)
        )

        assert compile_count == 1
        assert all(result == results[0] for result in results)
        assert tuple(hit.name for hit in results[0].hits) == ("parent", "child")
    finally:
        await agent.close()


async def test_non_empty_catalog_search_ranks_only_posting_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "posting-candidates.sqlite"
    with sqlite3.connect(database) as connection:
        for index in range(128):
            signal = ", needle_signal TEXT" if index == 73 else ""
            connection.execute(
                f"CREATE TABLE resource_{index:03d} "
                f"(id INTEGER PRIMARY KEY{signal})"
            )
    agent = await Agent.create("catalog-index-posting-work", root=tmp_path)
    try:
        await agent.attach(SQLiteSource(database))
        original_rank = catalog_service._rank_index_candidate
        rank_count = 0

        def counting_rank(*args, **kwargs):
            nonlocal rank_count
            rank_count += 1
            return original_rank(*args, **kwargs)

        monkeypatch.setattr(
            catalog_service,
            "_rank_index_candidate",
            counting_rank,
        )
        result = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="needle signal",
                limit=10,
            )
        )

        assert tuple(hit.name for hit in result.hits) == ("resource_073",)
        assert rank_count == 1
    finally:
        await agent.close()


async def test_catalog_index_refresh_evicts_stale_postings_and_inactive_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "index-refresh.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE old_table " "(id INTEGER PRIMARY KEY, legacyonly TEXT)"
        )
    agent = await Agent.create("catalog-index-refresh", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        original_compile = catalog_service._compile_source_index
        compiled_sync_ids: list[str] = []

        def counting_compile(snapshot):
            compiled_sync_ids.append(snapshot.sync.id)
            return original_compile(snapshot)

        monkeypatch.setattr(
            catalog_service,
            "_compile_source_index",
            counting_compile,
        )
        first = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="legacyonly",
                limit=10,
            )
        )
        repeated = await agent.search_catalog(first.request)
        assert tuple(hit.name for hit in first.hits) == ("old_table",)
        assert repeated == first
        assert len(compiled_sync_ids) == 1

        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE old_table")
            connection.execute(
                "CREATE TABLE new_table " "(id INTEGER PRIMARY KEY, currentonly TEXT)"
            )
        await agent.refresh_source(source.id)

        stale = await agent.search_catalog(first.request)
        current = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="currentonly",
                limit=10,
            )
        )
        assert stale.hits == ()
        assert tuple(hit.name for hit in current.hits) == ("new_table",)
        assert len(compiled_sync_ids) == 2
        assert set(agent._embedded._catalog_service._source_indexes) == {
            (agent.id, source.id, compiled_sync_ids[-1])
        }

        await agent.detach(source.id)
        inactive = await agent.search_catalog(current.request)
        assert inactive.hits == ()
        assert all(
            key[:2] != (agent.id, source.id)
            for key in agent._embedded._catalog_service._source_indexes
        )
    finally:
        await agent.close()


async def test_failed_catalog_index_compilation_is_not_published_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "index-compile-failure.sqlite"
    _database(database)
    agent = await Agent.create("catalog-index-compile-failure", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        original_compile = catalog_service._compile_source_index
        compile_attempts = 0

        def fail_once(snapshot):
            nonlocal compile_attempts
            compile_attempts += 1
            if compile_attempts == 1:
                raise ValueError("forced index compilation failure")
            return original_compile(snapshot)

        monkeypatch.setattr(catalog_service, "_compile_source_index", fail_once)
        request = CatalogSearchRequest(
            agent_id=agent.id,
            query="parent",
            limit=10,
        )
        with pytest.raises(CatalogStoreError, match="index compilation failed"):
            await agent.search_catalog(request)
        assert agent._embedded._catalog_service._source_indexes == {}

        recovered = await agent.search_catalog(request)
        assert tuple(hit.name for hit in recovered.hits) == ("parent", "child")
        assert compile_attempts == 2
        assert len(agent._embedded._catalog_service._source_indexes) == 1
        assert next(iter(agent._embedded._catalog_service._source_indexes))[:2] == (
            agent.id,
            source.id,
        )
    finally:
        await agent.close()


async def test_indexed_search_normalizes_ranks_diversifies_and_exposes_evidence(
    tmp_path: Path,
):
    database = tmp_path / "indexed-ranking.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE "OrderItems" (
                id INTEGER PRIMARY KEY,
                "StraßeCode" TEXT,
                customer_id INTEGER
            );
            CREATE INDEX order_items_customer_idx ON "OrderItems"(customer_id);
            CREATE TABLE "SnakeCaseRecord" (
                id INTEGER PRIMARY KEY,
                snake_case_value TEXT
            );
            CREATE TABLE showcase (id INTEGER PRIMARY KEY);
            CREATE TABLE needle (id INTEGER PRIMARY KEY);
            CREATE TABLE column_match (
                id INTEGER PRIMARY KEY,
                needle TEXT
            );
            CREATE TABLE customer_revenue (id INTEGER PRIMARY KEY);
            CREATE TABLE customer_profiles (id INTEGER PRIMARY KEY);
            CREATE TABLE region_lookup (id INTEGER PRIMARY KEY);
            CREATE TABLE parent (id INTEGER PRIMARY KEY);
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER NOT NULL REFERENCES parent(id)
            );
            CREATE VIEW customer_view AS SELECT id FROM customer_profiles;
            """)
    agent = await Agent.create("catalog-index-ranking", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        outside_database = tmp_path / "indexed-ranking-outside.sqlite"
        with sqlite3.connect(outside_database) as connection:
            connection.executescript("""
                CREATE TABLE outside_base (id INTEGER PRIMARY KEY);
                CREATE TABLE "main.needle" (id INTEGER PRIMARY KEY);
                CREATE VIEW customer_outside AS SELECT id FROM outside_base;
                """)
        await agent.attach(SQLiteSource(outside_database))

        camel = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="ＯＲＤＥＲ items",
                limit=10,
            )
        )
        unicode = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="STRASSE",
                limit=10,
            )
        )
        snake_and_camel = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="snake case",
                limit=10,
            )
        )
        exact = await agent.search_catalog(
            CatalogSearchRequest(agent_id=agent.id, query="needle", limit=10)
        )
        qualified = await agent.search_catalog(
            CatalogSearchRequest(agent_id=agent.id, query="main.needle", limit=10)
        )
        diversified = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="show customer revenue by region",
                limit=2,
            )
        )
        relationship = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="parent id",
                limit=50,
            )
        )
        views = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="customer",
                source_ids=(source.id,),
                resource_kinds=(ResourceKind.VIEW,),
                limit=10,
            )
        )
        all_views = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="customer",
                resource_kinds=(ResourceKind.VIEW,),
                limit=10,
            )
        )

        assert camel.hits[0].name == "OrderItems"
        assert unicode.hits[0].name == "OrderItems"
        assert "column:StraßeCode" in unicode.hits[0].matched_fields
        assert snake_and_camel.hits[0].name == "SnakeCaseRecord"
        assert "column:snake_case_value" in snake_and_camel.hits[0].matched_fields
        assert exact.hits[0].name == "needle"
        assert exact.hits[0].match_reasons == ("resource_name_exact",)
        exact_column = next(hit for hit in exact.hits if hit.name == "column_match")
        assert exact_column.match_reasons == ("structural_field_exact",)
        assert exact.hits[0].score > exact_column.score
        assert qualified.hits[0].name == "needle"
        assert qualified.hits[1].name == "main.needle"
        assert qualified.hits[0].score > exact.hits[0].score
        assert tuple(hit.name for hit in diversified.hits) == (
            "customer_revenue",
            "region_lookup",
        )
        assert diversified.total_matches == 6
        assert diversified.returned_count == 2
        child = next(hit for hit in relationship.hits if hit.name == "child")
        order_items = next(hit for hit in relationship.hits if hit.name == "OrderItems")
        assert "column:parent_id" in child.matched_fields
        assert "index_field:parent_id" not in child.matched_fields
        assert "relationship_field:parent_id" in child.matched_fields
        assert "index_field:customer_id" in order_items.matched_fields
        assert tuple(hit.name for hit in views.hits) == ("customer_view",)
        assert tuple(hit.name for hit in all_views.hits) == (
            "customer_outside",
            "customer_view",
        )
    finally:
        await agent.close()


async def test_embedded_exact_name_anchor_outranks_128_lookalikes_and_has_boundaries(
    tmp_path: Path,
):
    database = tmp_path / "embedded-exact-anchor.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE stage_b_profile_probe (id INTEGER PRIMARY KEY)"
        )
        for index in range(128):
            suffix = "archive" if index % 2 == 0 else "backup"
            connection.execute(
                f"CREATE TABLE stage_b_profile_probe_{suffix}_{index:03d} "
                "(id INTEGER PRIMARY KEY)"
            )
    agent = await Agent.create("catalog-embedded-exact-anchor", root=tmp_path)
    try:
        await agent.attach(SQLiteSource(database))
        query = (
            "Profile stage_b_profile_probe and ignore archive or backup look-alikes."
        )
        result = await agent.search_catalog(
            CatalogSearchRequest(agent_id=agent.id, query=query, limit=12)
        )

        assert result.hits[0].name == "stage_b_profile_probe"
        assert result.hits[0].match_reasons == ("resource_name_exact_mention",)
        assert result.total_matches == 129
        assert result.returned_count == 12
        assert result.truncated is True

        boundary = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="Profile stage_b_profile_probe_archive_000.",
                limit=12,
            )
        )
        exact_mentions = tuple(
            hit.name
            for hit in boundary.hits
            if hit.match_reasons == ("resource_name_exact_mention",)
        )
        assert exact_mentions == ("stage_b_profile_probe_archive_000",)

        native_identity = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="Please inspect main.stage_b_profile_probe.",
                limit=12,
            )
        )
        assert native_identity.hits[0].name == "stage_b_profile_probe"
        assert native_identity.hits[0].match_reasons == ("resource_name_exact_mention",)

        context = await agent._embedded._data_view.catalog_context(
            agent.id,
            query,
            limit=12,
        )
        context_resources = context["resources"]
        assert isinstance(context_resources, tuple)
        first = context_resources[0]
        assert isinstance(first, Mapping)
        assert first["resource_id"] == result.hits[0].resource_id
        assert first["match_reasons"] == ("resource_name_exact_mention",)
        assert context["total_matches"] == 129
        assert context["returned_count"] == 12
        assert context["truncated"] is True
    finally:
        await agent.close()


async def test_embedded_exact_duplicate_names_remain_source_scoped(
    tmp_path: Path,
):
    first_database = tmp_path / "duplicate-exact-first.sqlite"
    second_database = tmp_path / "duplicate-exact-second.sqlite"
    for database in (first_database, second_database):
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    agent = await Agent.create("catalog-embedded-exact-duplicates", root=tmp_path)
    try:
        first_source = await agent.attach(SQLiteSource(first_database))
        second_source = await agent.attach(SQLiteSource(second_database))

        result = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="Please profile orders.",
                limit=12,
            )
        )

        assert tuple(hit.name for hit in result.hits) == ("orders", "orders")
        assert {hit.source_id for hit in result.hits} == {
            first_source.id,
            second_source.id,
        }
        assert all(
            hit.match_reasons == ("resource_name_exact_mention",) for hit in result.hits
        )
    finally:
        await agent.close()


async def test_catalog_context_merges_current_and_prior_queries_by_contract_priority(
    tmp_path: Path,
):
    database = tmp_path / "catalog-context-priority.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE prior_target (id INTEGER PRIMARY KEY);
            CREATE TABLE current_target (id INTEGER PRIMARY KEY);
            CREATE TABLE profile_metrics (id INTEGER PRIMARY KEY);
            """)
    agent = await Agent.create("catalog-context-priority", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        view = agent._embedded._data_view

        continuity = await view.catalog_context(
            agent.id,
            "Profile it.",
            prior_query="Use prior_target for this analysis.",
            limit=12,
        )
        continuity_resources = continuity["resources"]
        assert isinstance(continuity_resources, tuple)
        continuity_records = cast(
            tuple[Mapping[str, object], ...], continuity_resources
        )
        assert tuple(item["name"] for item in continuity_records[:2]) == (
            "prior_target",
            "profile_metrics",
        )

        switched = await view.catalog_context(
            agent.id,
            "Switch to current_target.",
            prior_query="Use prior_target for this analysis.",
            limit=12,
        )
        switched_resources = switched["resources"]
        assert isinstance(switched_resources, tuple)
        switched_records = cast(tuple[Mapping[str, object], ...], switched_resources)
        assert tuple(item["name"] for item in switched_records[:2]) == (
            "current_target",
            "prior_target",
        )

        resources = await agent.list_catalog_resources(source_id=source.id)
        prior = next(item for item in resources if item.name == "prior_target")
        exact_id = await view.catalog_context(
            agent.id,
            "text that does not match catalog metadata",
            resource_ids=(prior.id,),
            limit=12,
        )
        exact_resources = exact_id["resources"]
        exact_sources = exact_id["sources"]
        assert isinstance(exact_resources, tuple)
        assert isinstance(exact_sources, tuple)
        exact_records = cast(tuple[Mapping[str, object], ...], exact_resources)
        source_records = cast(tuple[Mapping[str, object], ...], exact_sources)
        assert tuple(item["resource_id"] for item in exact_records) == (prior.id,)
        assert "source_revision" not in exact_records[0]
        assert "sync_id" not in exact_records[0]
        assert source_records[0]["source_id"] == source.id
        assert exact_id["total_matches"] == 1
        assert exact_id["returned_count"] == 1
        assert exact_id["truncated"] is False
    finally:
        await agent.close()


async def test_catalog_search_merges_true_global_top_k_across_more_than_64_sources(
    tmp_path: Path,
):
    agent = await Agent.create("catalog-index-global-top-k", root=tmp_path)
    try:
        for index in range(65):
            database = tmp_path / f"global-{index:03d}.sqlite"
            table = "target" if index == 64 else f"target_candidate_{index:03d}"
            with sqlite3.connect(database) as connection:
                connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")
            await agent.attach(SQLiteSource(database))

        result = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="target",
                limit=3,
            )
        )

        assert tuple(hit.name for hit in result.hits) == (
            "target",
            "target_candidate_000",
            "target_candidate_001",
        )
        assert result.total_matches == 65
        assert result.returned_count == 3
        assert result.truncated is True
    finally:
        await agent.close()


async def test_current_snapshot_refs_are_agent_and_source_isolated(tmp_path: Path):
    first_database = tmp_path / "refs-first.sqlite"
    second_database = tmp_path / "refs-second.sqlite"
    _database(first_database)
    _database(second_database)
    agent = await Agent.create("snapshot-refs", root=tmp_path)
    try:
        first = await agent.attach(SQLiteSource(first_database))
        second = await agent.attach(SQLiteSource(second_database))
        store = agent._embedded._store

        refs = await store.list_current_snapshot_refs(agent.id, ())
        assert refs == tuple(
            sorted(refs, key=lambda item: (item.source_id, item.sync_id))
        )
        assert {item.source_id for item in refs} == {first.id, second.id}
        assert all(item.agent_id == agent.id for item in refs)
        assert (
            await store.list_current_snapshot_refs(
                agent.id,
                (second.id, first.id),
            )
            == refs
        )
        assert await store.list_current_snapshot_refs(agent.id, (first.id,)) == tuple(
            item for item in refs if item.source_id == first.id
        )
        assert await store.list_current_snapshot_refs("another-agent", ()) == ()
        assert (
            await store.list_current_snapshot_refs(agent.id, ("missing-source",)) == ()
        )

        with pytest.raises(ValueError, match="duplicates"):
            await store.list_current_snapshot_refs(agent.id, (first.id, first.id))
        with pytest.raises(ValueError, match="non-empty"):
            await store.list_current_snapshot_refs(agent.id, ("",))
    finally:
        await agent.close()


async def test_current_snapshot_load_requires_the_exact_current_reference(
    tmp_path: Path,
):
    database = tmp_path / "exact-ref.sqlite"
    _database(database)
    agent = await Agent.create("snapshot-exact-ref", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        store = agent._embedded._store
        ref = (await store.list_current_snapshot_refs(agent.id, (source.id,)))[0]

        snapshot = await store.load_current_snapshot(ref)
        assert snapshot is not None
        assert snapshot.sync.agent_id == ref.agent_id
        assert snapshot.sync.source_id == ref.source_id
        assert snapshot.sync.id == ref.sync_id
        assert (
            await store.load_current_snapshot(
                CatalogSnapshotRef(
                    agent_id=ref.agent_id,
                    source_id=ref.source_id,
                    sync_id="not-current",
                )
            )
            is None
        )
        assert (
            await store.load_current_snapshot(
                CatalogSnapshotRef(
                    agent_id="another-agent",
                    source_id=ref.source_id,
                    sync_id=ref.sync_id,
                )
            )
            is None
        )
    finally:
        await agent.close()


def test_snapshot_ref_rejects_unbounded_or_blank_identity_fields():
    with pytest.raises(ValueError, match="non-empty"):
        CatalogSnapshotRef(agent_id="", source_id="source", sync_id="sync")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        CatalogSnapshotRef(agent_id="agent", source_id=" source", sync_id="sync")
    with pytest.raises(ValueError, match="512"):
        CatalogSnapshotRef(agent_id="agent", source_id="source", sync_id="x" * 513)


async def test_catalog_reads_decode_one_generation_once_and_share_concurrent_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "decode-once.sqlite"
    _database(database)
    agent = await Agent.create("snapshot-decode-once", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        reader = SQLiteStateStore(agent.home / "state.db")
        ref = (await reader.list_current_snapshot_refs(agent.id, (source.id,)))[0]
        decode_count = _snapshot_decode_counter(monkeypatch)

        loaded = await asyncio.gather(
            *(reader.load_current_snapshot(ref) for _ in range(16))
        )
        assert loaded[0] is not None
        assert all(snapshot is loaded[0] for snapshot in loaded)
        snapshot = loaded[0]
        assert snapshot is not None

        resources = await reader.list_resources(agent.id, source.id)
        await reader.load_resource(agent.id, resources[0].id)
        await reader.load_revision(
            agent.id,
            resources[0].id,
            resources[0].current_revision,
        )
        await reader.load_facets(
            agent.id,
            resources[0].id,
            resources[0].current_revision,
        )
        await reader.load_incident_relationships(agent.id, resources[0].id)
        assert decode_count() == 1
    finally:
        await agent.close()


async def test_failed_snapshot_decode_does_not_publish_a_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "failed-decode.sqlite"
    _database(database)
    agent = await Agent.create("failed-snapshot-decode", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        reader = SQLiteStateStore(agent.home / "state.db")
        ref = (await reader.list_current_snapshot_refs(agent.id, (source.id,)))[0]
        original_decode = sqlite_store.decode_catalog_snapshot
        fail_next_snapshot = True

        def failing_decode(value: str):
            nonlocal fail_next_snapshot
            if fail_next_snapshot:
                fail_next_snapshot = False
                raise ValueError("forced snapshot decode failure")
            return original_decode(value)

        monkeypatch.setattr(sqlite_store, "decode_catalog_snapshot", failing_decode)
        with pytest.raises(ValueError, match="forced snapshot decode failure"):
            await reader.load_current_snapshot(ref)
        assert reader._decoded_catalog_snapshots == {}

        snapshot = await reader.load_current_snapshot(ref)
        assert snapshot is not None
        assert set(reader._decoded_catalog_snapshots) == {
            (ref.agent_id, ref.source_id, ref.sync_id)
        }
    finally:
        await agent.close()


async def test_refresh_invalidates_stale_ref_and_decodes_the_new_generation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "generation-refresh.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE old_table (id INTEGER PRIMARY KEY)")
    agent = await Agent.create("snapshot-generation-refresh", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        reader = SQLiteStateStore(agent.home / "state.db")
        decode_count = _snapshot_decode_counter(monkeypatch)
        old_ref = (await reader.list_current_snapshot_refs(agent.id, (source.id,)))[0]
        old_resources = await reader.list_resources(agent.id, source.id)
        assert tuple(item.name for item in old_resources) == ("old_table",)
        assert decode_count() == 1

        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE old_table")
            connection.execute("CREATE TABLE new_table (id INTEGER PRIMARY KEY)")
        await agent.refresh_source(source.id)

        new_ref = (await reader.list_current_snapshot_refs(agent.id, (source.id,)))[0]
        assert new_ref != old_ref
        assert await reader.load_current_snapshot(old_ref) is None
        new_resources = await reader.list_resources(agent.id, source.id)
        assert tuple(item.name for item in new_resources) == ("new_table",)
        assert await reader.load_resource(agent.id, old_resources[0].id) is None
        assert decode_count() == 2
        assert set(reader._decoded_catalog_snapshots) == {
            (new_ref.agent_id, new_ref.source_id, new_ref.sync_id)
        }
    finally:
        await agent.close()


async def test_failed_snapshot_commit_does_not_publish_candidate_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "failed-cache-publication.sqlite"
    _database(database)
    agent = await Agent.create("failed-cache-publication", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        store = agent._embedded._store
        cache_before = dict(store._decoded_catalog_snapshots)
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE uncommitted (id INTEGER PRIMARY KEY)")

        def fail_commit(connection):
            raise RuntimeError("forced catalog commit failure")

        monkeypatch.setattr(sqlite_store, "_commit_catalog_transaction", fail_commit)
        with pytest.raises(RuntimeError, match="forced catalog commit failure"):
            await agent.refresh_source(source.id)

        assert store._decoded_catalog_snapshots == cache_before
        assert tuple(
            item.name
            for item in await agent.list_catalog_resources(source_id=source.id)
        ) == ("child", "parent")
    finally:
        await agent.close()


async def test_running_partial_and_failed_syncs_never_contribute_or_replace_truth(
    tmp_path: Path,
):
    started_at = datetime(2026, 7, 22, 14, 0, tzinfo=UTC)
    agent = await Agent.create("sync-state", root=tmp_path, clock=lambda: started_at)
    store = SQLiteStateStore(agent.home / "state.db")
    try:
        registration = SourceRegistration.build(
            agent_id=agent.id,
            adapter_id="test.adapter",
            native_identity="partial-only",
            display_name="Partial only",
            configuration={},
            attached_at=started_at,
        )
        await store.register_source(registration)
        await store.record_sync(
            CatalogSync(
                id="running-sync",
                agent_id=agent.id,
                source_id=registration.id,
                adapter_id=registration.adapter_id,
                status=CatalogSyncStatus.RUNNING,
                started_at=started_at,
            )
        )
        await store.record_sync(
            CatalogSync(
                id="partial-sync",
                agent_id=agent.id,
                source_id=registration.id,
                adapter_id=registration.adapter_id,
                status=CatalogSyncStatus.PARTIAL,
                started_at=started_at,
                completed_at=started_at + timedelta(minutes=1),
                resource_count=99,
                relationship_count=88,
                error_code="bounded_partial",
            )
        )
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=1,
            resource_count=0,
            relationship_count=0,
            latest_successful_sync_completed_at=None,
            is_empty=True,
        )

        database = tmp_path / "successful.sqlite"
        _database(database)
        successful = await agent.attach(SQLiteSource(database))
        committed = await agent.catalog_summary()
        await store.record_sync(
            CatalogSync(
                id="failed-after-success",
                agent_id=agent.id,
                source_id=successful.id,
                adapter_id=successful.adapter_id,
                status=CatalogSyncStatus.FAILED,
                started_at=started_at + timedelta(minutes=2),
                completed_at=started_at + timedelta(minutes=3),
                error_code="source_attach_failed",
            )
        )
        assert await agent.catalog_summary() == committed
        assert committed.active_source_count == 2
        assert committed.resource_count == 2
        assert committed.relationship_count == 1
    finally:
        await store.close()
        await agent.close()


async def test_refresh_cancellation_before_transaction_keeps_old_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "pre-commit.sqlite"
    _database(database)
    agent = await Agent.create("pre-commit", root=tmp_path)
    source = await agent.attach(SQLiteSource(database))
    old_resources = await agent.list_catalog_resources(source_id=source.id)
    cache_before = dict(agent._embedded._store._decoded_catalog_snapshots)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE later (id INTEGER PRIMARY KEY)")

    before_start = threading.Event()
    release_start = threading.Event()
    original_start = sqlite_store._CatalogCommitGate.start

    def paused_start(self, connection):
        before_start.set()
        if not release_start.wait(5):
            raise TimeoutError("catalog transaction start was not released")
        return original_start(self, connection)

    monkeypatch.setattr(sqlite_store._CatalogCommitGate, "start", paused_start)
    refresh = asyncio.create_task(agent.refresh_source(source.id))
    try:
        assert await asyncio.to_thread(before_start.wait, 5)
        refresh.cancel()
        await asyncio.sleep(0)
        release_start.set()
        with pytest.raises(asyncio.CancelledError):
            await refresh

        current_resources = await agent.list_catalog_resources(source_id=source.id)
        assert current_resources == old_resources
        assert agent._embedded._store._decoded_catalog_snapshots == cache_before
        with sqlite3.connect(agent.home / "state.db") as connection:
            sync_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT id FROM syncs WHERE source_id = ? ORDER BY id",
                    (source.id,),
                )
            )
        store = SQLiteStateStore(agent.home / "state.db")
        syncs = tuple(
            [await store.load_sync(agent.id, sync_id) for sync_id in sync_ids]
        )
        assert any(
            sync is not None and sync.status is CatalogSyncStatus.FAILED
            for sync in syncs
        )
        current_sync_ids = {resource.current_sync_id for resource in current_resources}
        assert all(
            sync is None
            or sync.status is not CatalogSyncStatus.FAILED
            or sync.id not in current_sync_ids
            for sync in syncs
        )
    finally:
        release_start.set()
        await agent.close()


async def test_refresh_cancellation_after_transaction_start_commits_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "commit-wins.sqlite"
    _database(database)
    agent = await Agent.create("commit-wins", root=tmp_path)
    source = await agent.attach(SQLiteSource(database))
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE later (id INTEGER PRIMARY KEY)")

    before_commit = threading.Event()
    release_commit = threading.Event()
    original_commit = sqlite_store._commit_catalog_transaction

    def paused_commit(connection):
        before_commit.set()
        if not release_commit.wait(5):
            raise TimeoutError("catalog transaction commit was not released")
        original_commit(connection)

    monkeypatch.setattr(sqlite_store, "_commit_catalog_transaction", paused_commit)
    refresh = asyncio.create_task(agent.refresh_source(source.id))
    try:
        assert await asyncio.to_thread(before_commit.wait, 5)
        refresh.cancel()
        release_commit.set()
        assert await refresh == source

        resources = await agent.list_catalog_resources(source_id=source.id)
        assert tuple(resource.name for resource in resources) == (
            "child",
            "later",
            "parent",
        )
        sync_ids = {resource.current_sync_id for resource in resources}
        assert len(sync_ids) == 1
        store = SQLiteStateStore(agent.home / "state.db")
        sync = await store.load_sync(agent.id, sync_ids.pop())
        assert sync is not None
        assert sync.status is CatalogSyncStatus.SUCCEEDED
    finally:
        release_commit.set()
        await agent.close()


async def test_first_attach_cancellation_before_commit_publishes_no_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "cancel-first.sqlite"
    _database(database)
    agent = await Agent.create("cancel-first", root=tmp_path)
    before_start = threading.Event()
    release_start = threading.Event()
    original_start = sqlite_store._CatalogCommitGate.start

    def paused_start(self, connection):
        before_start.set()
        if not release_start.wait(5):
            raise TimeoutError("catalog transaction start was not released")
        return original_start(self, connection)

    monkeypatch.setattr(sqlite_store._CatalogCommitGate, "start", paused_start)
    attach = asyncio.create_task(agent.attach(SQLiteSource(database)))
    try:
        assert await asyncio.to_thread(before_start.wait, 5)
        attach.cancel()
        await asyncio.sleep(0)
        release_start.set()
        with pytest.raises(asyncio.CancelledError):
            await attach

        assert await agent.list_sources() == ()
        assert await agent.list_catalog_resources() == ()
    finally:
        release_start.set()
        await agent.close()


async def test_catalog_summary_is_not_persisted_or_added_to_model_state(tmp_path: Path):
    database = tmp_path / "modeled.sqlite"
    _database(database)
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="done"),)
    )
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=256,
        supports_tools=True,
    )
    agent = await Agent.create(
        "projection-only",
        root=tmp_path,
        model=provider,
        model_profile=profile,
    )
    state_path = agent.home / "state.db"
    try:
        await agent.attach(SQLiteSource(database))
        summary = await agent.catalog_summary()
        assert summary.is_empty is False
        result = await agent.run("Answer normally")
        transcript = await agent.transcript(result.run_id)
        request_text = repr(provider.requests[0])
        transcript_text = repr(transcript)
        for field_name in (
            "active_source_count",
            "latest_successful_sync_completed_at",
            "relationship_count",
            "readiness",
        ):
            assert field_name not in request_text
            assert field_name not in transcript_text
    finally:
        await agent.close()

    with sqlite3.connect(state_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "catalog_summaries" not in tables
    assert "readiness" not in tables
