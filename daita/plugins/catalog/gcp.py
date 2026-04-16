"""
GCP infrastructure discoverer.

Enumerates Cloud SQL instances, GCS buckets, BigQuery datasets, Firestore
databases, Bigtable instances, Pub/Sub topics/subscriptions, Memorystore
(Redis) instances, and GCP API Gateway APIs across configured GCP projects
using the google-cloud-* client libraries.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from .base_discoverer import BaseDiscoverer, DiscoveredStore
from .discovery._gcp_common import _GCP_INSTALL_HINT, gcp_credentials

logger = logging.getLogger(__name__)


# Maps Cloud SQL databaseVersion prefix → store_type handled by existing profilers.
# e.g. "POSTGRES_15" → "postgresql", reusing PostgresProfiler.
_CLOUDSQL_ENGINES_TO_STORE_TYPE = {
    "POSTGRES": "postgresql",
    "MYSQL": "mysql",
    "SQLSERVER": "sqlserver",
}

_LOCATION_WILDCARD = "-"  # GCP-wide convention for "all locations"


def _csv_env(name: str) -> list[str]:
    """Parse a comma-separated env var into a list, filtering empty entries."""
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GCPDiscoverer(BaseDiscoverer):
    """
    Discover data stores across GCP projects and locations.

    Enumerates Cloud SQL, GCS, BigQuery, Firestore, Bigtable, Pub/Sub,
    Memorystore, and GCP API Gateway. Lazy-imports google-cloud-* client
    libraries in ``authenticate()`` and each enumerator.

    Authentication:
      * ``credentials_path``: explicit service-account JSON key file
      * ``impersonate_service_account``: chain through service-account impersonation
      * else: Application Default Credentials (ADC) — honours
        ``GOOGLE_APPLICATION_CREDENTIALS`` env var and ``gcloud auth`` state.

    Env var fallbacks mirror ``BigQueryPlugin`` conventions:
      * ``GCP_PROJECTS`` (CSV) or ``GOOGLE_CLOUD_PROJECT``
      * ``GCP_LOCATIONS`` (CSV)
      * ``GOOGLE_APPLICATION_CREDENTIALS``
      * ``GCP_IMPERSONATE_SERVICE_ACCOUNT``
    """

    name = "gcp"

    # service key → bound method name
    _SERVICE_METHODS = {
        "cloudsql": "_enumerate_cloudsql",
        "gcs": "_enumerate_gcs",
        "bigquery": "_enumerate_bigquery",
        "firestore": "_enumerate_firestore",
        "bigtable": "_enumerate_bigtable",
        "pubsub": "_enumerate_pubsub",
        "memorystore": "_enumerate_memorystore",
        "apigateway": "_enumerate_apigateway",
    }

    def __init__(
        self,
        projects: Optional[list[str]] = None,
        locations: Optional[list[str]] = None,
        credentials_path: Optional[str] = None,
        impersonate_service_account: Optional[str] = None,
        services: Optional[list[str]] = None,
    ):
        self._projects = projects or _csv_env("GCP_PROJECTS")
        if not self._projects and (single := os.environ.get("GOOGLE_CLOUD_PROJECT")):
            self._projects = [single]

        self._locations = locations or _csv_env("GCP_LOCATIONS")
        self._credentials_path = credentials_path or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self._impersonate = impersonate_service_account or os.environ.get(
            "GCP_IMPERSONATE_SERVICE_ACCOUNT"
        )
        self._services = services or list(self._SERVICE_METHODS)
        self._credentials: Any = None

    # ------------------------------------------------------------------
    # BaseDiscoverer hooks
    # ------------------------------------------------------------------

    async def authenticate(self) -> None:
        """Resolve credentials and default project."""
        self._credentials, default_project = gcp_credentials(
            credentials_path=self._credentials_path,
            impersonate_service_account=self._impersonate,
        )
        if not self._projects and default_project:
            self._projects = [default_project]
        if not self._projects:
            raise ValueError(
                "No GCP projects configured. Pass projects= or set GCP_PROJECTS / "
                "GOOGLE_CLOUD_PROJECT."
            )

    async def enumerate(self) -> AsyncIterator[DiscoveredStore]:
        """Yield discovered stores across all configured projects and services."""
        if not self._credentials:
            await self.authenticate()

        for project in self._projects:
            for service in self._services:
                method_name = self._SERVICE_METHODS.get(service)
                if not method_name:
                    continue
                async for store in getattr(self, method_name)(project):
                    yield store

    def fingerprint(self, store: DiscoveredStore) -> str:
        """GCP-specific fingerprint using project + location + resource name."""
        resource_name = store.metadata.get("resource_name", "")
        project = store.metadata.get("project", "")
        raw = f"gcp|{project}|{store.region or 'global'}|{resource_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def test_access(self) -> bool:
        """Verify GCP credentials resolve to at least one project."""
        try:
            await self.authenticate()
            return bool(self._projects)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Store builder helper
    # ------------------------------------------------------------------

    def _build_store(
        self,
        store_type: str,
        display_name: str,
        connection_hint: dict[str, Any],
        source: str,
        region: Optional[str],
        resource_name: str,
        project: str,
        metadata: dict[str, Any],
        tags: Optional[list[str]] = None,
    ) -> DiscoveredStore:
        """Build a DiscoveredStore with GCP-standard metadata and fingerprint."""
        now = _iso_now()
        merged_metadata = {
            "resource_name": resource_name,
            "project": project,
            **metadata,
        }
        store = DiscoveredStore(
            id="",
            store_type=store_type,
            display_name=display_name,
            connection_hint={"project": project, **connection_hint},
            source=source,
            region=region,
            confidence=0.95,
            tags=tags or [],
            metadata=merged_metadata,
            discovered_at=now,
            last_seen=now,
        )
        store.id = self.fingerprint(store)
        return store

    # ------------------------------------------------------------------
    # Cloud SQL
    # ------------------------------------------------------------------

    async def _enumerate_cloudsql(self, project: str) -> AsyncIterator[DiscoveredStore]:
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "google-api-python-client is required. "
                "Install with: pip install 'daita-agents[gcp]'"
            )

        try:
            service = build(
                "sqladmin", "v1", credentials=self._credentials, cache_discovery=False
            )
            resp = service.instances().list(project=project).execute()
        except Exception as exc:
            logger.warning("Cloud SQL list failed for %s: %s", project, exc)
            return

        for inst in resp.get("items", []):
            engine_prefix = (inst.get("databaseVersion") or "").split("_", 1)[0]
            store_type = _CLOUDSQL_ENGINES_TO_STORE_TYPE.get(engine_prefix, "unknown")
            name = inst.get("name", "")
            region = inst.get("region")
            ip_addresses = inst.get("ipAddresses", [])
            primary_ip = next(
                (ip["ipAddress"] for ip in ip_addresses if ip.get("type") == "PRIMARY"),
                "",
            )

            yield self._build_store(
                store_type=store_type,
                display_name=f"{name} ({region})",
                connection_hint={
                    "instance": name,
                    "connection_name": inst.get("connectionName", ""),
                    "host": primary_ip,
                    "engine": engine_prefix.lower(),
                },
                source="gcp_cloudsql",
                region=region,
                resource_name=inst.get("selfLink", ""),
                project=project,
                metadata={
                    "database_version": inst.get("databaseVersion", ""),
                    "tier": inst.get("settings", {}).get("tier", ""),
                    "state": inst.get("state", ""),
                    "availability_type": inst.get("settings", {}).get(
                        "availabilityType", ""
                    ),
                },
            )

    # ------------------------------------------------------------------
    # GCS
    # ------------------------------------------------------------------

    async def _enumerate_gcs(self, project: str) -> AsyncIterator[DiscoveredStore]:
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required. "
                "Install with: pip install 'daita-agents[gcp]'"
            )

        try:
            client = storage.Client(project=project, credentials=self._credentials)
            buckets = list(client.list_buckets())
        except Exception as exc:
            logger.warning("GCS list_buckets failed for %s: %s", project, exc)
            return

        for bucket in buckets:
            yield self._build_store(
                store_type="gcs",
                display_name=f"gs://{bucket.name} ({bucket.location or 'unknown'})",
                connection_hint={"bucket": bucket.name},
                source="gcp_gcs",
                region=bucket.location,
                resource_name=f"projects/_/buckets/{bucket.name}",
                project=project,
                metadata={
                    "storage_class": bucket.storage_class or "",
                    "created": str(bucket.time_created or ""),
                    "versioning_enabled": bool(bucket.versioning_enabled),
                },
            )

    # ------------------------------------------------------------------
    # BigQuery
    # ------------------------------------------------------------------

    async def _enumerate_bigquery(self, project: str) -> AsyncIterator[DiscoveredStore]:
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery is required. "
                "Install with: pip install 'daita-agents[gcp]'"
            )

        try:
            client = bigquery.Client(project=project, credentials=self._credentials)
            datasets = list(client.list_datasets())
        except Exception as exc:
            logger.warning("BigQuery list_datasets failed for %s: %s", project, exc)
            return

        for ds in datasets:
            dataset_id = ds.dataset_id
            try:
                ds_meta = client.get_dataset(ds.reference)
                location = ds_meta.location or ""
            except Exception:
                location = ""

            yield self._build_store(
                store_type="bigquery",
                display_name=f"{project}.{dataset_id}",
                connection_hint={"dataset": dataset_id},
                source="gcp_bigquery",
                region=location or None,
                resource_name=f"projects/{project}/datasets/{dataset_id}",
                project=project,
                metadata={"location": location},
            )

    # ------------------------------------------------------------------
    # Firestore
    # ------------------------------------------------------------------

    async def _enumerate_firestore(
        self, project: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from google.cloud import firestore_admin_v1
        except ImportError:
            raise ImportError(
                "google-cloud-firestore is required. "
                "Install with: pip install 'daita-agents[gcp]'"
            )

        try:
            client = firestore_admin_v1.FirestoreAdminClient(
                credentials=self._credentials
            )
            parent = f"projects/{project}"
            resp = client.list_databases(parent=parent)
            databases = list(resp.databases)
        except Exception as exc:
            logger.warning("Firestore list_databases failed for %s: %s", project, exc)
            return

        for db in databases:
            # db.name = "projects/{project}/databases/{db}"
            db_id = db.name.rsplit("/", 1)[-1]
            yield self._build_store(
                store_type="firestore",
                display_name=f"{project}/{db_id}",
                connection_hint={"database": db_id},
                source="gcp_firestore",
                region=db.location_id or None,
                resource_name=db.name,
                project=project,
                metadata={
                    "type": db.type_.name if db.type_ else "",
                    "concurrency_mode": (
                        db.concurrency_mode.name if db.concurrency_mode else ""
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Bigtable
    # ------------------------------------------------------------------

    async def _enumerate_bigtable(self, project: str) -> AsyncIterator[DiscoveredStore]:
        try:
            from google.cloud import bigtable
        except ImportError:
            raise ImportError(
                "google-cloud-bigtable is required. "
                "Install with: pip install 'daita-agents[gcp]'"
            )

        try:
            client = bigtable.Client(
                project=project, credentials=self._credentials, admin=True
            )
            instances, _failed = client.list_instances()
        except Exception as exc:
            logger.warning("Bigtable list_instances failed for %s: %s", project, exc)
            return

        for inst in instances:
            yield self._build_store(
                store_type="bigtable",
                display_name=f"{project}/{inst.instance_id}",
                connection_hint={"instance": inst.instance_id},
                source="gcp_bigtable",
                region=None,
                resource_name=inst.name,
                project=project,
                metadata={
                    "display_name": inst.display_name or "",
                    "state": str(inst.state) if inst.state else "",
                    "type": str(inst.type_) if inst.type_ else "",
                },
            )

    # ------------------------------------------------------------------
    # Pub/Sub (topics + subscriptions)
    # ------------------------------------------------------------------

    async def _enumerate_pubsub(self, project: str) -> AsyncIterator[DiscoveredStore]:
        try:
            from google.cloud import pubsub_v1
        except ImportError:
            raise ImportError(
                "google-cloud-pubsub is required. "
                "Install with: pip install 'daita-agents[gcp]'"
            )

        project_path = f"projects/{project}"

        # --- Topics ---
        try:
            publisher = pubsub_v1.PublisherClient(credentials=self._credentials)
            topics = list(publisher.list_topics(request={"project": project_path}))
        except Exception as exc:
            logger.warning("Pub/Sub list_topics failed for %s: %s", project, exc)
            topics = []

        for topic in topics:
            topic_id = topic.name.rsplit("/", 1)[-1]
            yield self._build_store(
                store_type="pubsub_topic",
                display_name=f"{project}/{topic_id}",
                connection_hint={"topic": topic_id},
                source="gcp_pubsub",
                region=None,
                resource_name=topic.name,
                project=project,
                metadata={"kms_key": topic.kms_key_name or ""},
            )

        # --- Subscriptions ---
        try:
            subscriber = pubsub_v1.SubscriberClient(credentials=self._credentials)
            subs = list(
                subscriber.list_subscriptions(request={"project": project_path})
            )
        except Exception as exc:
            logger.warning("Pub/Sub list_subscriptions failed for %s: %s", project, exc)
            subs = []

        for sub in subs:
            sub_id = sub.name.rsplit("/", 1)[-1]
            yield self._build_store(
                store_type="pubsub_subscription",
                display_name=f"{project}/{sub_id}",
                connection_hint={"subscription": sub_id},
                source="gcp_pubsub",
                region=None,
                resource_name=sub.name,
                project=project,
                metadata={
                    "topic": sub.topic.rsplit("/", 1)[-1] if sub.topic else "",
                    "ack_deadline_seconds": sub.ack_deadline_seconds or 0,
                },
            )

    # ------------------------------------------------------------------
    # Memorystore (Redis)
    # ------------------------------------------------------------------

    async def _enumerate_memorystore(
        self, project: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from google.cloud import redis_v1
        except ImportError:
            raise ImportError(
                "google-cloud-redis is required. "
                "Install with: pip install 'daita-agents[gcp]'"
            )

        client = redis_v1.CloudRedisClient(credentials=self._credentials)
        locations = self._locations or [_LOCATION_WILDCARD]

        for location in locations:
            parent = f"projects/{project}/locations/{location}"
            try:
                instances = list(client.list_instances(parent=parent))
            except Exception as exc:
                logger.warning(
                    "Memorystore list_instances failed for %s: %s", parent, exc
                )
                continue

            for inst in instances:
                # inst.name = "projects/{p}/locations/{loc}/instances/{id}"
                parts = inst.name.split("/")
                inst_location = parts[3] if len(parts) >= 4 else location
                inst_id = parts[-1]

                yield self._build_store(
                    store_type="memorystore",
                    display_name=f"{inst_id} ({inst_location})",
                    connection_hint={
                        "instance": inst_id,
                        "location": inst_location,
                        "host": inst.host or "",
                        "port": inst.port or 0,
                    },
                    source="gcp_memorystore",
                    region=inst_location,
                    resource_name=inst.name,
                    project=project,
                    metadata={
                        "tier": (
                            redis_v1.Instance.Tier(inst.tier).name if inst.tier else ""
                        ),
                        "redis_version": inst.redis_version or "",
                        "memory_size_gb": inst.memory_size_gb or 0,
                        "state": (
                            redis_v1.Instance.State(inst.state).name
                            if inst.state
                            else ""
                        ),
                    },
                )

    # ------------------------------------------------------------------
    # GCP API Gateway
    # ------------------------------------------------------------------

    async def _enumerate_apigateway(
        self, project: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from google.cloud import apigateway_v1
        except ImportError:
            raise ImportError(
                "google-cloud-api-gateway is required. "
                "Install with: pip install 'daita-agents[gcp]'"
            )

        client = apigateway_v1.ApiGatewayServiceClient(credentials=self._credentials)
        # API Gateway APIs are managed under a wildcard parent for "all locations".
        parent = f"projects/{project}/locations/{_LOCATION_WILDCARD}"

        try:
            apis = list(client.list_apis(parent=parent))
        except Exception as exc:
            logger.warning("API Gateway list_apis failed for %s: %s", project, exc)
            return

        for api in apis:
            # api.name = "projects/{p}/locations/{loc}/apis/{api_id}"
            parts = api.name.split("/")
            location = parts[3] if len(parts) >= 4 else "global"
            api_id = parts[-1]

            yield self._build_store(
                store_type="gcp_apigateway",
                display_name=f"{api_id} ({location})",
                connection_hint={"api_id": api_id, "location": location},
                source="gcp_apigateway",
                region=location,
                resource_name=api.name,
                project=project,
                metadata={
                    "display_name": api.display_name or "",
                    "state": (
                        apigateway_v1.Api.State(api.state).name if api.state else ""
                    ),
                    "managed_service": api.managed_service or "",
                },
            )
