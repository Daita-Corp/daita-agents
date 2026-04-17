"""
Live integration tests for the GCP discoverer in the Catalog plugin.

Exercises the full path against a real Google Cloud project and a real
OpenAI key:

  1. GCPDiscoverer authenticates via ADC / service-account JSON
  2. Enumerates Cloud SQL, GCS, BigQuery, Firestore, Bigtable, Pub/Sub,
     Memorystore, and GCP API Gateway resources
  3. CatalogPlugin orchestrates discover_all() across the discoverer
  4. An Agent with the CatalogPlugin asks the LLM questions that
     must be answered from the discovered catalog

Requirements (env vars):
  - OPENAI_API_KEY                         — live LLM
  - GOOGLE_APPLICATION_CREDENTIALS         — path to a service-account JSON
                                             key OR active `gcloud auth` state
  - GCP_PROJECTS or GOOGLE_CLOUD_PROJECT   — at least one project to scan
  - GCP_LOCATIONS (optional)               — CSV of locations for regional
                                             services (e.g. "us-central1")

Run:
    OPENAI_API_KEY=sk-... \\
    GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json \\
    GCP_PROJECTS=my-project \\
    pytest tests/integration/test_catalog_gcp_live.py -v -s \\
        -m "requires_llm and integration"
"""

import os

import pytest

# Skip the whole module unless the GCP core auth library is installed.
pytest.importorskip(
    "google.auth",
    reason="google-auth required: pip install 'daita-agents[gcp]'",
)

from daita.agents.agent import Agent
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.gcp import GCPDiscoverer

# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------


def _require_env(*names: str) -> None:
    """Skip the test unless every listed env var is set."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"Live GCP test needs env: {', '.join(missing)}")


def _gcp_projects() -> list[str]:
    """Resolve the target project list from env, mirroring GCPDiscoverer."""
    raw = os.environ.get("GCP_PROJECTS", "")
    projects = [p.strip() for p in raw.split(",") if p.strip()]
    if not projects and (single := os.environ.get("GOOGLE_CLOUD_PROJECT")):
        projects = [single]
    return projects


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gcp_discoverer() -> GCPDiscoverer:
    """
    Real GCPDiscoverer bound to the env-configured project(s) and locations.

    Credentials resolve via ``google.auth.default()`` — honours
    ``GOOGLE_APPLICATION_CREDENTIALS``, ``gcloud auth application-default
    login`` state, and GCE metadata.
    """
    if not _gcp_projects():
        pytest.skip("Set GCP_PROJECTS or GOOGLE_CLOUD_PROJECT")
    try:
        from google.auth import default as _adc_default

        _adc_default()
    except Exception as exc:
        pytest.skip(f"No GCP credentials available via ADC: {exc}")
    return GCPDiscoverer()


@pytest.fixture
async def catalog_with_gcp(gcp_discoverer: GCPDiscoverer) -> CatalogPlugin:
    """CatalogPlugin with the live GCPDiscoverer registered."""
    plugin = CatalogPlugin()
    plugin.add_discoverer(gcp_discoverer)
    return plugin


@pytest.fixture
def bq_constrained_dataset() -> dict[str, str]:
    """
    Seed a BigQuery dataset with two tables that declare PK + FK constraints.

    BigQuery supports ``PRIMARY KEY (col) NOT ENFORCED`` and
    ``FOREIGN KEY (col) REFERENCES other(col) NOT ENFORCED`` — both purely
    informational, zero data scanned, and visible through INFORMATION_SCHEMA.

    Idempotent: re-running recreates the tables via ``CREATE OR REPLACE``.
    Returns ``{"project": ..., "dataset": ...}`` for test consumption.
    """
    projects = _gcp_projects()
    if not projects:
        pytest.skip("Set GCP_PROJECTS or GOOGLE_CLOUD_PROJECT")

    bigquery = pytest.importorskip("google.cloud.bigquery")
    project = projects[0]
    dataset = "daita_catalog_test"

    client = bigquery.Client(project=project)
    # Dataset must exist — create it if missing.
    ds_ref = bigquery.Dataset(f"{project}.{dataset}")
    ds_ref.location = "US"
    client.create_dataset(ds_ref, exists_ok=True)

    # CREATE OR REPLACE so the tables always have the declared constraints.
    ddl = f"""
    CREATE OR REPLACE TABLE `{project}.{dataset}.customers` (
        customer_id  INT64   NOT NULL OPTIONS(description="Primary key"),
        name         STRING  NOT NULL,
        email        STRING,
        signup_date  DATE    NOT NULL,
        PRIMARY KEY (customer_id) NOT ENFORCED
    );
    CREATE OR REPLACE TABLE `{project}.{dataset}.orders` (
        order_id     INT64     NOT NULL OPTIONS(description="Primary key"),
        customer_id  INT64     NOT NULL OPTIONS(description="FK -> customers.customer_id"),
        amount       NUMERIC   NOT NULL,
        status       STRING,
        created_at   TIMESTAMP NOT NULL,
        PRIMARY KEY (order_id) NOT ENFORCED,
        FOREIGN KEY (customer_id)
            REFERENCES `{project}.{dataset}.customers`(customer_id)
            NOT ENFORCED
    );
    """
    # Multi-statement script — BigQuery handles them in one query job.
    client.query(ddl).result()
    return {"project": project, "dataset": dataset}


@pytest.fixture
def pubsub_topic_with_avro_schema() -> dict[str, str]:
    """
    Ensure a Pub/Sub topic exists bound to an Avro schema in Schema Registry.

    Idempotent: reuses existing schema/topic when present. Returns the topic
    + schema IDs so tests can assert that discovery resolves both.
    """
    projects = _gcp_projects()
    if not projects:
        pytest.skip("Set GCP_PROJECTS or GOOGLE_CLOUD_PROJECT")

    pytest.importorskip("google.cloud.pubsub_v1")
    from google.cloud import pubsub_v1
    from google.pubsub_v1 import types as pstypes
    from google.api_core import exceptions as gax
    import json

    project = projects[0]
    schema_id = "daita-event-schema"
    topic_id = "daita-events"

    schema_client = pubsub_v1.SchemaServiceClient()
    publisher = pubsub_v1.PublisherClient()
    schema_path = schema_client.schema_path(project, schema_id)
    topic_path = publisher.topic_path(project, topic_id)

    avro_def = json.dumps(
        {
            "type": "record",
            "name": "DaitaEvent",
            "namespace": "com.daita.catalog",
            "fields": [
                {"name": "event_id", "type": "string"},
                {
                    "name": "occurred_at",
                    "type": {"type": "long", "logicalType": "timestamp-millis"},
                },
                {"name": "user_id", "type": ["null", "string"], "default": None},
                {"name": "action", "type": "string"},
                {"name": "amount", "type": ["null", "double"], "default": None},
            ],
        }
    )
    try:
        schema_client.create_schema(
            request={
                "parent": f"projects/{project}",
                "schema_id": schema_id,
                "schema": pstypes.Schema(
                    type_=pstypes.Schema.Type.AVRO, definition=avro_def
                ),
            }
        )
    except gax.AlreadyExists:
        pass

    try:
        publisher.create_topic(
            request={
                "name": topic_path,
                "schema_settings": pstypes.SchemaSettings(
                    schema=schema_path, encoding=pstypes.Encoding.JSON
                ),
            }
        )
    except gax.AlreadyExists:
        pass

    return {
        "project": project,
        "schema_id": schema_id,
        "topic_id": topic_id,
        "schema_path": schema_path,
        "topic_path": topic_path,
    }


@pytest.fixture
def firestore_collection_with_index() -> dict[str, str]:
    """
    Ensure a Firestore collection exists with at least one composite index.

    Seeds three sample docs into ``events`` and creates a composite index on
    ``(user_id ASC, timestamp DESC)``. Idempotent — skips creation if the
    index already exists.
    """
    projects = _gcp_projects()
    if not projects:
        pytest.skip("Set GCP_PROJECTS or GOOGLE_CLOUD_PROJECT")

    pytest.importorskip("google.cloud.firestore")
    pytest.importorskip("google.cloud.firestore_admin_v1")
    from google.cloud import firestore, firestore_admin_v1
    from google.cloud.firestore_admin_v1 import types as ftypes
    from google.api_core import exceptions as gax

    project = projects[0]
    collection = "events"

    # Seed docs — collection must have data before indexes become meaningful.
    db = firestore.Client(project=project)
    for i, (user, action) in enumerate(
        [("alice", "login"), ("bob", "signup"), ("alice", "logout")]
    ):
        db.collection(collection).document(f"evt-{i}").set(
            {
                "user_id": user,
                "action": action,
                "timestamp": f"2026-04-16T1{i}:00:00Z",
                "amount": 10.0 + i,
            }
        )

    admin = firestore_admin_v1.FirestoreAdminClient()
    parent = f"projects/{project}/databases/(default)/collectionGroups/{collection}"
    index = ftypes.Index(
        query_scope=ftypes.Index.QueryScope.COLLECTION,
        fields=[
            ftypes.Index.IndexField(
                field_path="user_id",
                order=ftypes.Index.IndexField.Order.ASCENDING,
            ),
            ftypes.Index.IndexField(
                field_path="timestamp",
                order=ftypes.Index.IndexField.Order.DESCENDING,
            ),
        ],
    )
    try:
        admin.create_index(parent=parent, index=index)
    except gax.AlreadyExists:
        pass
    # Do NOT block on result() — index build is async (can take minutes).
    # Tests tolerate any state; they only need the index to be listable.

    return {"project": project, "collection": collection}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGCPDiscovererLive:
    """Direct discoverer calls against a real GCP account."""

    async def test_authenticate_resolves_credentials(
        self, gcp_discoverer: GCPDiscoverer
    ):
        """authenticate() populates credentials and at least one project."""
        await gcp_discoverer.authenticate()
        assert gcp_discoverer._credentials is not None
        assert gcp_discoverer._projects, "Expected at least one GCP project"

    async def test_test_access_succeeds(self, gcp_discoverer: GCPDiscoverer):
        """test_access() returns True when credentials are valid."""
        assert await gcp_discoverer.test_access() is True

    async def test_enumerate_yields_stores(self, gcp_discoverer: GCPDiscoverer, capsys):
        """
        enumerate() iterates across all configured services and yields
        DiscoveredStore instances. Empty accounts are allowed, but no
        exception should escape.
        """
        stores = [store async for store in gcp_discoverer.enumerate()]

        types = sorted({s.store_type for s in stores})
        projects = sorted({s.metadata.get("project", "") for s in stores})
        print(
            f"\n[GCP LIVE] discovered {len(stores)} store(s); "
            f"types={types or '-'}; projects={projects or '-'}"
        )

        for store in stores:
            assert store.id, "Every store must have a fingerprint id"
            assert store.store_type
            assert store.source.startswith("gcp_")
            assert store.metadata.get("project") in _gcp_projects()


@pytest.mark.integration
class TestCatalogPluginWithGCP:
    """CatalogPlugin orchestration against live GCP."""

    async def test_discover_all_returns_result(
        self, catalog_with_gcp: CatalogPlugin, capsys
    ):
        """discover_all() returns a DiscoveryResult with no discoverer errors."""
        result = await catalog_with_gcp.discover_all()

        print(
            f"\n[CATALOG] stores={result.store_count} " f"errors={result.error_count}"
        )
        for err in result.errors:
            print(f"  ! {err.discoverer_name}: {err.error}")

        assert not result.has_errors, f"GCP discoverer emitted errors: {result.errors}"
        # Populates the plugin's in-memory catalog
        assert len(catalog_with_gcp.get_stores()) == result.store_count

    async def test_find_store_by_type(self, catalog_with_gcp: CatalogPlugin):
        """get_stores(store_type=...) filters correctly after a live scan."""
        await catalog_with_gcp.discover_all()
        all_stores = catalog_with_gcp.get_stores()
        if not all_stores:
            pytest.skip("No stores in target GCP project(s) to filter on")

        first_type = all_stores[0].store_type
        filtered = catalog_with_gcp.get_stores(store_type=first_type)
        assert filtered, f"Expected at least one {first_type} store"
        assert all(s.store_type == first_type for s in filtered)


@pytest.mark.integration
class TestBigQueryConstraints:
    """Declared PK/FK must flow through discover → normalize → profile."""

    async def test_primary_and_foreign_keys_surface(
        self,
        gcp_discoverer: GCPDiscoverer,
        bq_constrained_dataset: dict[str, str],
        capsys,
    ):
        """
        Seeded dataset declares:
          customers: PK(customer_id)
          orders:    PK(order_id), FK(customer_id) -> customers(customer_id)

        After profile_store() the NormalizedSchema must expose both.
        """
        from daita.plugins.catalog.profiler import BigQueryProfiler

        plugin = CatalogPlugin()
        plugin.add_discoverer(gcp_discoverer)
        plugin.add_profiler(BigQueryProfiler())

        await plugin.discover_and_profile()

        project = bq_constrained_dataset["project"]
        dataset = bq_constrained_dataset["dataset"]

        bq_store = next(
            (
                s
                for s in plugin.get_stores(store_type="bigquery")
                if s.connection_hint.get("dataset") == dataset
                and s.metadata.get("project") == project
            ),
            None,
        )
        assert bq_store, f"Seeded dataset {project}.{dataset} not discovered"

        schema = plugin.get_schema(bq_store.id)
        assert schema, "Profiler did not produce a schema"

        tables = {t.name: t for t in schema.tables}
        assert {"customers", "orders"} <= tables.keys()

        # PRIMARY KEY surfaces as is_primary_key=True on the right column only.
        pk_columns = {
            (t.name, c.name)
            for t in schema.tables
            for c in t.columns
            if c.is_primary_key
        }
        assert pk_columns == {
            ("customers", "customer_id"),
            ("orders", "order_id"),
        }, f"Unexpected PK columns: {pk_columns}"

        # FOREIGN KEY surfaces as a NormalizedForeignKey entry.
        fk_shapes = {
            (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
            for fk in schema.foreign_keys
        }
        print(f"\n[FK] {fk_shapes}")
        assert (
            "orders",
            "customer_id",
            "customers",
            "customer_id",
        ) in fk_shapes, (
            f"FK orders.customer_id → customers.customer_id missing: {fk_shapes}"
        )


@pytest.mark.integration
class TestPubSubSchemaRegistry:
    """Declared Pub/Sub Avro schemas flow through discover → normalize → profile."""

    async def test_avro_schema_fields_become_columns(
        self,
        gcp_discoverer: GCPDiscoverer,
        pubsub_topic_with_avro_schema: dict[str, str],
        capsys,
    ):
        """
        A topic bound to an Avro schema must surface the record's fields as
        columns (not the generic message envelope), plus carry the schema
        ref + definition on the table's metadata.
        """
        from daita.plugins.catalog.discovery import discover_pubsub_topic
        from daita.plugins.catalog.normalizer import normalize_pubsub_topic

        fx = pubsub_topic_with_avro_schema

        # Discovery returns raw topic metadata including the resolved schema.
        raw = await discover_pubsub_topic(project=fx["project"], topic=fx["topic_id"])
        assert raw.get("schema"), f"Topic should be bound to a schema: {raw}"
        assert raw["schema"]["type"] == "AVRO"
        assert "DaitaEvent" in raw["schema"]["definition"]

        # Normalizer converts Avro fields into columns on the topic's table.
        norm = normalize_pubsub_topic(raw)
        table = norm["tables"][0]
        column_names = {c["name"] for c in table["columns"]}

        # Every declared Avro field must appear as a column (plus possibly
        # sub:<name> entries for any subscriptions on the topic).
        expected_fields = {"event_id", "occurred_at", "user_id", "action", "amount"}
        assert expected_fields <= column_names, (
            f"Missing Avro fields in columns: "
            f"{expected_fields - column_names}; saw {column_names}"
        )

        # Schema metadata preserved on the table for non-lossy agent access.
        tmeta = table.get("metadata", {})
        assert tmeta.get("schema_type") == "AVRO"
        assert tmeta.get("schema_name", "").endswith(fx["schema_id"])
        assert "DaitaEvent" in tmeta.get("schema_definition", "")

        print(
            f"\n[PUBSUB] topic={fx['topic_id']} "
            f"avro_cols={sorted(expected_fields & column_names)}"
        )


@pytest.mark.integration
class TestFirestoreIndexes:
    """Declared Firestore composite indexes flow through to NormalizedIndex."""

    async def test_composite_index_surfaces(
        self,
        gcp_discoverer: GCPDiscoverer,
        firestore_collection_with_index: dict[str, str],
        capsys,
    ):
        """
        Seeded collection declares a composite index on
        ``(user_id ASC, timestamp DESC)``.

        After discover_and_profile, the NormalizedTable for ``events`` must
        carry a NormalizedIndex(type="composite") covering both columns.
        """
        from daita.plugins.catalog.profiler import FirestoreProfiler

        fx = firestore_collection_with_index

        plugin = CatalogPlugin()
        plugin.add_discoverer(gcp_discoverer)
        plugin.add_profiler(FirestoreProfiler())
        await plugin.discover_and_profile()

        fs_store = next((s for s in plugin.get_stores(store_type="firestore")), None)
        assert fs_store, "Firestore database was not discovered"

        schema = plugin.get_schema(fs_store.id)
        assert schema, "Profiler did not produce a schema"

        events_table = next(
            (t for t in schema.tables if t.name == fx["collection"]), None
        )
        assert events_table, (
            f"Collection {fx['collection']!r} missing from schema; "
            f"saw {[t.name for t in schema.tables]}"
        )

        composite = [i for i in events_table.indexes if i.type == "composite"]
        assert composite, (
            f"No composite index on {fx['collection']}; "
            f"indexes={[(i.name, i.type, i.columns) for i in events_table.indexes]}"
        )
        # At least one index must cover the seeded (user_id, timestamp) pair.
        pair_found = any(
            "user_id" in idx.columns and "timestamp" in idx.columns for idx in composite
        )
        assert pair_found, (
            f"Expected composite index covering user_id + timestamp; "
            f"saw {[idx.columns for idx in composite]}"
        )

        print(
            f"\n[FIRESTORE] collection={fx['collection']} "
            f"indexes={[(i.name, i.columns) for i in composite]}"
        )


@pytest.mark.integration
class TestGraphEmissionLive:
    """Tier-2 graph emission across the live discover → profile → persist path.

    After ``discover_and_profile`` the graph backend must carry fully-qualified
    Table / Column / Index nodes with their :HAS_COLUMN / :INDEXED_BY /
    :COVERS / :REFERENCES edges — see docs/catalog_graph_tier2.md.
    """

    async def test_bigquery_fks_produce_references_edges(
        self,
        gcp_discoverer: GCPDiscoverer,
        bq_constrained_dataset: dict[str, str],
        tmp_path,
        monkeypatch,
    ):
        """Seeded orders.customer_id -> customers.customer_id FK must surface as
        a :REFERENCES edge between qualified Column nodes."""
        from daita.core.graph.local_backend import LocalGraphBackend
        from daita.core.graph.models import EdgeType
        from daita.core.graph.resolution import resolve_table
        from daita.plugins.catalog.profiler import BigQueryProfiler

        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="it_bq_fk")

        plugin = CatalogPlugin(backend=backend, auto_persist=True)
        plugin.add_discoverer(gcp_discoverer)
        plugin.add_profiler(BigQueryProfiler())
        # CatalogPlugin.initialize() auto-selects a backend only when None —
        # since we passed one in the constructor, no override is applied.
        plugin.initialize(agent_id="it-agent")

        await plugin.discover_and_profile()

        project = bq_constrained_dataset["project"]
        dataset = bq_constrained_dataset["dataset"]
        store = f"bigquery:{project}.{dataset}"

        orders = await backend.get_node(f"table:{store}.orders")
        customers = await backend.get_node(f"table:{store}.customers")
        assert orders is not None, "orders Table node missing"
        assert customers is not None, "customers Table node missing"
        assert orders.properties["store"] == store

        src_col = f"column:{store}.orders.customer_id"
        tgt_col = f"column:{store}.customers.customer_id"
        assert await backend.get_node(src_col) is not None
        assert await backend.get_node(tgt_col) is not None

        refs = [
            e
            for e in await backend.get_edges(from_node_id=src_col)
            if e.edge_type == EdgeType.REFERENCES
        ]
        assert any(e.to_node_id == tgt_col for e in refs), (
            f"Expected :REFERENCES from {src_col} -> {tgt_col}; "
            f"saw {[(e.from_node_id, e.to_node_id) for e in refs]}"
        )

        # Qualified resolution: bare 'orders' matches this BigQuery store.
        resolved = await resolve_table(backend, "orders", store=store)
        assert len(resolved) == 1
        assert resolved[0].node_id == f"table:{store}.orders"

    async def test_firestore_composite_index_produces_indexed_by_and_covers(
        self,
        gcp_discoverer: GCPDiscoverer,
        firestore_collection_with_index: dict[str, str],
        tmp_path,
        monkeypatch,
    ):
        """Seeded composite index on (user_id ASC, timestamp DESC) must surface
        as an :INDEXED_BY edge plus :COVERS edges with 0-indexed positions."""
        from daita.core.graph.local_backend import LocalGraphBackend
        from daita.core.graph.models import EdgeType, NodeType
        from daita.plugins.catalog.profiler import FirestoreProfiler

        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="it_fs_idx")

        plugin = CatalogPlugin(backend=backend, auto_persist=True)
        plugin.add_discoverer(gcp_discoverer)
        plugin.add_profiler(FirestoreProfiler())
        plugin.initialize(agent_id="it-agent")

        await plugin.discover_and_profile()

        fx = firestore_collection_with_index
        project = fx["project"]
        collection = fx["collection"]

        # Firestore store qualifier: firestore:<project>/<database>
        # The seeded fixture uses the default database.
        store = f"firestore:{project}/(default)"

        events_id = f"table:{store}.{collection}"
        events_node = await backend.get_node(events_id)
        assert (
            events_node is not None
        ), f"events Table node missing; searched {events_id}"

        indexed_by = [
            e
            for e in await backend.get_edges(from_node_id=events_id)
            if e.edge_type == EdgeType.INDEXED_BY
        ]
        assert indexed_by, (
            "No :INDEXED_BY edge emitted for events table; "
            "composite index was not reflected in the graph."
        )

        # Check that at least one index covers (user_id, timestamp) with the
        # right positions. Firestore may report multiple composite indexes;
        # filter to the one seeded by the fixture.
        matching_index = None
        for edge in indexed_by:
            idx_node = await backend.get_node(edge.to_node_id)
            if idx_node is None or idx_node.node_type != NodeType.INDEX:
                continue
            covers = sorted(
                (
                    e
                    for e in await backend.get_edges(from_node_id=edge.to_node_id)
                    if e.edge_type == EdgeType.COVERS
                ),
                key=lambda e: e.properties.get("position", -1),
            )
            covered_cols = [e.to_node_id.rsplit(".", 1)[-1] for e in covers]
            if ["user_id", "timestamp"] == covered_cols:
                matching_index = (idx_node, covers)
                break

        assert (
            matching_index is not None
        ), "No composite index covers (user_id, timestamp) with positions 0, 1"
        _, covers = matching_index
        assert [e.properties["position"] for e in covers] == [0, 1]

    async def test_cross_store_collision_safety(
        self,
        gcp_discoverer: GCPDiscoverer,
        bq_constrained_dataset: dict[str, str],
        tmp_path,
        monkeypatch,
    ):
        """Seed a synthetic second store whose table names overlap with the
        live BigQuery dataset. Both Table nodes must coexist under distinct
        qualified IDs; ``resolve_table`` must return both candidates."""
        from daita.core.graph.local_backend import LocalGraphBackend
        from daita.core.graph.resolution import (
            AmbiguousReferenceError,
            resolve_table,
            resolve_table_unique,
        )
        from daita.plugins.catalog.persistence import persist_schema_to_graph
        from daita.plugins.catalog.profiler import BigQueryProfiler

        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="it_collision")

        plugin = CatalogPlugin(backend=backend, auto_persist=True)
        plugin.add_discoverer(gcp_discoverer)
        plugin.add_profiler(BigQueryProfiler())
        plugin.initialize(agent_id="it-agent")
        await plugin.discover_and_profile()

        # Seed a synthetic postgres store with the same 'orders' table name.
        synthetic = {
            "database_type": "postgresql",
            "database_name": "public",
            "metadata": {"host": "synthetic-pg"},
            "tables": [
                {
                    "name": "orders",
                    "row_count": 0,
                    "columns": [
                        {
                            "name": "id",
                            "type": "int",
                            "nullable": False,
                            "is_primary_key": True,
                        }
                    ],
                }
            ],
            "foreign_keys": [],
        }
        await persist_schema_to_graph(synthetic, backend, agent_id="it-agent")

        project = bq_constrained_dataset["project"]
        dataset = bq_constrained_dataset["dataset"]
        bq_store = f"bigquery:{project}.{dataset}"
        pg_store = "postgresql:synthetic-pg/public"

        bq_orders = await backend.get_node(f"table:{bq_store}.orders")
        pg_orders = await backend.get_node(f"table:{pg_store}.orders")
        assert bq_orders is not None
        assert pg_orders is not None
        assert bq_orders.node_id != pg_orders.node_id

        # Bare-name resolution returns both; unique() raises.
        matches = await resolve_table(backend, "orders")
        stores = {m.store for m in matches}
        assert {bq_store, pg_store} <= stores

        with pytest.raises(AmbiguousReferenceError):
            await resolve_table_unique(backend, "orders")


@pytest.mark.requires_llm
@pytest.mark.integration
class TestAgentWithGCPCatalog:
    """End-to-end: Agent + OpenAI + CatalogPlugin with live GCP."""

    async def test_agent_lists_gcp_stores(
        self, catalog_with_gcp: CatalogPlugin, capsys
    ):
        """
        The agent must use the catalog's discover_infrastructure tool to
        enumerate GCP resources and summarise them back to the user.
        """
        _require_env("OPENAI_API_KEY")

        agent = Agent(
            name="GCPCatalogAgent",
            llm_provider="openai",
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            tools=[catalog_with_gcp],
        )

        result = await agent.run(
            "Use the discover_infrastructure tool to scan the configured GCP "
            "account, then tell me how many data stores were found and list "
            "the unique store types (e.g. bigquery, gcs). Be concise.",
            detailed=True,
        )

        answer = result["result"]
        tool_names = [c.get("tool") for c in result.get("tool_calls", [])]
        print(f"\n[AGENT] tools={tool_names}")
        print(f"[AGENT] answer={answer[:300]}")

        assert (
            "discover_infrastructure" in tool_names
        ), f"Agent did not call discover_infrastructure; saw: {tool_names}"
        assert answer, "Agent returned an empty answer"
        # Agent must reflect the real store count it observed — accept digits
        # or English word form (LLMs switch between "2" and "two" freely).
        expected_count = len(catalog_with_gcp.get_stores())
        word_forms = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        forms = {str(expected_count), word_forms.get(expected_count, "")}
        assert any(
            f and f.lower() in answer.lower() for f in forms
        ), f"Expected count {expected_count} (digit or word) in answer: {answer}"

    async def test_agent_finds_store_by_type(
        self, catalog_with_gcp: CatalogPlugin, capsys
    ):
        """
        After a discovery sweep, the agent should use find_store to query
        the catalog for a specific GCP store type.
        """
        _require_env("OPENAI_API_KEY")

        # Pre-populate the in-memory catalog so find_store has data to filter
        await catalog_with_gcp.discover_all()
        stores = catalog_with_gcp.get_stores()
        if not stores:
            pytest.skip("No GCP stores discovered — nothing for the agent to find")

        target_type = stores[0].store_type

        agent = Agent(
            name="GCPCatalogAgent",
            llm_provider="openai",
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            tools=[catalog_with_gcp],
        )

        result = await agent.run(
            f"Use the find_store tool with store_type='{target_type}' and "
            "report how many matches the catalog contains. "
            "Answer with just the number and the store type.",
            detailed=True,
        )

        answer = result["result"]
        tool_names = [c.get("tool") for c in result.get("tool_calls", [])]
        print(f"\n[AGENT] tools={tool_names}  answer={answer[:200]}")

        assert "find_store" in tool_names
        expected = len([s for s in stores if s.store_type == target_type])
        assert str(expected) in answer
        assert target_type in answer
