"""
Azure infrastructure discoverer.

Enumerates Azure SQL, PostgreSQL Flexible Server, MySQL Flexible Server,
Cosmos DB, Blob Storage containers, Azure Cache for Redis, Event Hubs,
Service Bus, and API Management APIs across configured subscriptions.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from .base_discoverer import BaseDiscoverer, DiscoveredStore
from .discovery._azure_common import (
    azure_credential,
    azure_location,
    csv_env,
    resource_group_from_id,
)

logger = logging.getLogger(__name__)


class AzureDiscoverer(BaseDiscoverer):
    """
    Discover data stores across Azure subscriptions.

    Authentication uses ``DefaultAzureCredential`` and honors the standard
    Azure SDK environment / managed identity / CLI credential chain.

    Env var fallbacks:
      * ``AZURE_SUBSCRIPTIONS`` (CSV)
      * ``AZURE_LOCATIONS`` (CSV)
      * ``AZURE_TENANT_ID``
    """

    name = "azure"

    _SERVICE_METHODS = {
        "sql": "_enumerate_sql",
        "postgresql": "_enumerate_postgresql",
        "mysql": "_enumerate_mysql",
        "cosmosdb": "_enumerate_cosmosdb",
        "blob": "_enumerate_blob",
        "redis": "_enumerate_redis",
        "eventhub": "_enumerate_eventhub",
        "servicebus": "_enumerate_servicebus",
        "apim": "_enumerate_apim",
    }

    def __init__(
        self,
        subscriptions: Optional[list[str]] = None,
        locations: Optional[list[str]] = None,
        tenant_id: Optional[str] = None,
        services: Optional[list[str]] = None,
    ):
        self._subscriptions = subscriptions or csv_env("AZURE_SUBSCRIPTIONS")
        self._locations = locations or csv_env("AZURE_LOCATIONS")
        self._tenant_id = tenant_id or os.environ.get("AZURE_TENANT_ID")
        self._services = services or list(self._SERVICE_METHODS)
        self._credential: Any = None

    async def authenticate(self) -> None:
        """Resolve Azure credentials and subscription IDs."""
        self._credential = azure_credential(tenant_id=self._tenant_id)
        if self._subscriptions:
            return

        try:
            from azure.mgmt.subscription import SubscriptionClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-subscription is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        client = SubscriptionClient(self._credential)
        self._subscriptions = [
            s.subscription_id for s in client.subscriptions.list() if s.subscription_id
        ]
        if not self._subscriptions:
            raise ValueError(
                "No Azure subscriptions configured. Pass subscriptions= or set "
                "AZURE_SUBSCRIPTIONS."
            )

    async def enumerate(self) -> AsyncIterator[DiscoveredStore]:
        """Yield discovered stores across all configured subscriptions."""
        if not self._credential:
            await self.authenticate()

        for subscription_id in self._subscriptions:
            for service in self._services:
                method_name = self._SERVICE_METHODS.get(service)
                if not method_name:
                    continue
                async for store in getattr(self, method_name)(subscription_id):
                    yield store

    def fingerprint(self, store: DiscoveredStore) -> str:
        """Azure-specific fingerprint using subscription + resource ID."""
        resource_id = store.metadata.get("resource_id", "")
        subscription = store.metadata.get("subscription_id", "")
        raw = f"azure|{subscription}|{store.region or 'global'}|{resource_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def test_access(self) -> bool:
        """Verify Azure credentials resolve to at least one subscription."""
        try:
            await self.authenticate()
            return bool(self._subscriptions)
        except Exception:
            return False

    def _location_allowed(self, location: Optional[str]) -> bool:
        if not self._locations:
            return True
        normalized_location = (location or "").replace(" ", "").lower()
        return bool(normalized_location) and normalized_location in {
            loc.replace(" ", "").lower() for loc in self._locations
        }

    def _build_store(
        self,
        store_type: str,
        display_name: str,
        connection_hint: dict[str, Any],
        source: str,
        region: Optional[str],
        resource_id: str,
        subscription_id: str,
        metadata: dict[str, Any],
        tags: Optional[list[str]] = None,
    ) -> DiscoveredStore:
        """Build a DiscoveredStore with Azure-standard metadata."""
        now = datetime.now(timezone.utc).isoformat()
        merged_metadata = {
            "resource_id": resource_id,
            "subscription_id": subscription_id,
            "resource_group": resource_group_from_id(resource_id),
            **metadata,
        }
        store = DiscoveredStore(
            id="",
            store_type=store_type,
            display_name=display_name,
            connection_hint={
                "subscription_id": subscription_id,
                "tenant_id": self._tenant_id,
                **connection_hint,
            },
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

    async def _enumerate_sql(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.sql import SqlManagementClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-sql is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = SqlManagementClient(self._credential, subscription_id)
            servers = list(client.servers.list())
        except Exception as exc:
            logger.warning(
                "Azure SQL server list failed for %s: %s", subscription_id, exc
            )
            return

        for server in servers:
            location = azure_location(server)
            if not self._location_allowed(location):
                continue
            name = getattr(server, "name", "")
            resource_id = getattr(server, "id", "")
            fqdn = getattr(server, "fully_qualified_domain_name", "") or ""
            yield self._build_store(
                store_type="sqlserver",
                display_name=f"{name} ({location or 'unknown'})",
                connection_hint={
                    "server": name,
                    "host": fqdn,
                    "port": 1433,
                    "resource_group": resource_group_from_id(resource_id),
                },
                source="azure_sql",
                region=location,
                resource_id=resource_id,
                subscription_id=subscription_id,
                tags=[
                    f"{k}={v}" for k, v in (getattr(server, "tags", None) or {}).items()
                ],
                metadata={
                    "administrator_login": getattr(server, "administrator_login", ""),
                    "version": getattr(server, "version", ""),
                    "state": getattr(server, "state", ""),
                },
            )

    async def _enumerate_postgresql(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.rdbms.postgresql_flexibleservers import (
                PostgreSQLManagementClient,
            )
        except ImportError:
            raise ImportError(
                "azure-mgmt-rdbms is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = PostgreSQLManagementClient(self._credential, subscription_id)
            servers = list(client.servers.list())
        except Exception as exc:
            logger.warning(
                "Azure PostgreSQL list failed for %s: %s", subscription_id, exc
            )
            return

        for server in servers:
            location = azure_location(server)
            if not self._location_allowed(location):
                continue
            name = getattr(server, "name", "")
            resource_id = getattr(server, "id", "")
            fqdn = getattr(server, "fully_qualified_domain_name", "") or ""
            yield self._build_store(
                store_type="postgresql",
                display_name=f"{name} ({location or 'unknown'})",
                connection_hint={
                    "server": name,
                    "host": fqdn,
                    "port": 5432,
                    "resource_group": resource_group_from_id(resource_id),
                    "ssl_mode": "verify-full",
                },
                source="azure_postgresql",
                region=location,
                resource_id=resource_id,
                subscription_id=subscription_id,
                tags=[
                    f"{k}={v}" for k, v in (getattr(server, "tags", None) or {}).items()
                ],
                metadata={
                    "version": str(getattr(server, "version", "") or ""),
                    "state": str(getattr(server, "state", "") or ""),
                    "sku": getattr(getattr(server, "sku", None), "name", ""),
                    "storage_mb": getattr(
                        getattr(server, "storage", None), "storage_size_gb", None
                    ),
                },
            )

    async def _enumerate_mysql(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.rdbms.mysql_flexibleservers import MySQLManagementClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-rdbms is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = MySQLManagementClient(self._credential, subscription_id)
            servers = list(client.servers.list())
        except Exception as exc:
            logger.warning("Azure MySQL list failed for %s: %s", subscription_id, exc)
            return

        for server in servers:
            location = azure_location(server)
            if not self._location_allowed(location):
                continue
            name = getattr(server, "name", "")
            resource_id = getattr(server, "id", "")
            fqdn = getattr(server, "fully_qualified_domain_name", "") or ""
            yield self._build_store(
                store_type="mysql",
                display_name=f"{name} ({location or 'unknown'})",
                connection_hint={
                    "server": name,
                    "host": fqdn,
                    "port": 3306,
                    "resource_group": resource_group_from_id(resource_id),
                },
                source="azure_mysql",
                region=location,
                resource_id=resource_id,
                subscription_id=subscription_id,
                tags=[
                    f"{k}={v}" for k, v in (getattr(server, "tags", None) or {}).items()
                ],
                metadata={
                    "version": str(getattr(server, "version", "") or ""),
                    "state": str(getattr(server, "state", "") or ""),
                    "sku": getattr(getattr(server, "sku", None), "name", ""),
                },
            )

    async def _enumerate_cosmosdb(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.cosmosdb import CosmosDBManagementClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-cosmosdb is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = CosmosDBManagementClient(self._credential, subscription_id)
            accounts = list(client.database_accounts.list())
        except Exception as exc:
            logger.warning("Cosmos DB list failed for %s: %s", subscription_id, exc)
            return

        for account in accounts:
            location = azure_location(account)
            if not self._location_allowed(location):
                continue
            name = getattr(account, "name", "")
            resource_id = getattr(account, "id", "")
            endpoint = getattr(account, "document_endpoint", "") or ""
            yield self._build_store(
                store_type="cosmosdb",
                display_name=f"{name} ({location or 'unknown'})",
                connection_hint={
                    "account": name,
                    "endpoint": endpoint,
                    "resource_group": resource_group_from_id(resource_id),
                },
                source="azure_cosmosdb",
                region=location,
                resource_id=resource_id,
                subscription_id=subscription_id,
                tags=[
                    f"{k}={v}"
                    for k, v in (getattr(account, "tags", None) or {}).items()
                ],
                metadata={
                    "kind": getattr(account, "kind", ""),
                    "consistency_policy": str(
                        getattr(account, "consistency_policy", "") or ""
                    ),
                    "locations": [
                        getattr(loc, "location_name", "")
                        for loc in (getattr(account, "locations", None) or [])
                    ],
                },
            )

    async def _enumerate_blob(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.storage import StorageManagementClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-storage is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = StorageManagementClient(self._credential, subscription_id)
            accounts = list(client.storage_accounts.list())
        except Exception as exc:
            logger.warning(
                "Azure Storage account list failed for %s: %s", subscription_id, exc
            )
            return

        for account in accounts:
            account_location = azure_location(account)
            if not self._location_allowed(account_location):
                continue
            account_name = getattr(account, "name", "")
            account_id = getattr(account, "id", "")
            resource_group = resource_group_from_id(account_id)
            try:
                containers = list(
                    client.blob_containers.list(resource_group, account_name)
                )
            except Exception as exc:
                logger.warning(
                    "Azure Blob container list failed for %s/%s: %s",
                    subscription_id,
                    account_name,
                    exc,
                )
                containers = []

            for container in containers:
                container_name = getattr(container, "name", "")
                resource_id = getattr(container, "id", "") or (
                    f"{account_id}/blobServices/default/containers/{container_name}"
                )
                yield self._build_store(
                    store_type="azure_blob",
                    display_name=f"{account_name}/{container_name}",
                    connection_hint={
                        "account": account_name,
                        "container": container_name,
                        "account_url": f"https://{account_name}.blob.core.windows.net",
                        "resource_group": resource_group,
                    },
                    source="azure_blob",
                    region=account_location,
                    resource_id=resource_id,
                    subscription_id=subscription_id,
                    tags=[
                        f"{k}={v}"
                        for k, v in (getattr(account, "tags", None) or {}).items()
                    ],
                    metadata={
                        "account_resource_id": account_id,
                        "public_access": str(
                            getattr(container, "public_access", "") or ""
                        ),
                        "lease_state": str(getattr(container, "lease_state", "") or ""),
                    },
                )

    async def _enumerate_redis(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.redis import RedisManagementClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-redis is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = RedisManagementClient(self._credential, subscription_id)
            caches = list(client.redis.list_by_subscription())
        except Exception as exc:
            logger.warning("Azure Redis list failed for %s: %s", subscription_id, exc)
            return

        for cache in caches:
            location = azure_location(cache)
            if not self._location_allowed(location):
                continue
            name = getattr(cache, "name", "")
            resource_id = getattr(cache, "id", "")
            yield self._build_store(
                store_type="redis",
                display_name=f"{name} ({location or 'unknown'})",
                connection_hint={
                    "host": getattr(cache, "host_name", "") or "",
                    "port": getattr(cache, "ssl_port", None)
                    or getattr(cache, "port", 6379),
                    "resource_group": resource_group_from_id(resource_id),
                },
                source="azure_redis",
                region=location,
                resource_id=resource_id,
                subscription_id=subscription_id,
                tags=[
                    f"{k}={v}" for k, v in (getattr(cache, "tags", None) or {}).items()
                ],
                metadata={
                    "sku": getattr(getattr(cache, "sku", None), "name", ""),
                    "family": getattr(getattr(cache, "sku", None), "family", ""),
                    "capacity": getattr(getattr(cache, "sku", None), "capacity", None),
                    "provisioning_state": getattr(cache, "provisioning_state", ""),
                },
            )

    async def _enumerate_eventhub(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.eventhub import EventHubManagementClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-eventhub is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = EventHubManagementClient(self._credential, subscription_id)
            namespaces = list(client.namespaces.list())
        except Exception as exc:
            logger.warning(
                "Event Hubs namespace list failed for %s: %s", subscription_id, exc
            )
            return

        for namespace in namespaces:
            location = azure_location(namespace)
            if not self._location_allowed(location):
                continue
            ns_name = getattr(namespace, "name", "")
            ns_id = getattr(namespace, "id", "")
            resource_group = resource_group_from_id(ns_id)
            try:
                hubs = list(
                    client.event_hubs.list_by_namespace(resource_group, ns_name)
                )
            except Exception as exc:
                logger.warning("Event Hubs list failed for %s: %s", ns_name, exc)
                hubs = []

            for hub in hubs:
                hub_name = getattr(hub, "name", "")
                resource_id = getattr(hub, "id", "") or f"{ns_id}/eventhubs/{hub_name}"
                yield self._build_store(
                    store_type="eventhub",
                    display_name=f"{ns_name}/{hub_name}",
                    connection_hint={
                        "namespace": ns_name,
                        "eventhub": hub_name,
                        "resource_group": resource_group,
                    },
                    source="azure_eventhub",
                    region=location,
                    resource_id=resource_id,
                    subscription_id=subscription_id,
                    metadata={
                        "namespace_resource_id": ns_id,
                        "partition_count": getattr(hub, "partition_count", None),
                        "message_retention_days": getattr(
                            hub, "message_retention_in_days", None
                        ),
                        "status": str(getattr(hub, "status", "") or ""),
                    },
                )

    async def _enumerate_servicebus(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.servicebus import ServiceBusManagementClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-servicebus is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = ServiceBusManagementClient(self._credential, subscription_id)
            namespaces = list(client.namespaces.list())
        except Exception as exc:
            logger.warning(
                "Service Bus namespace list failed for %s: %s", subscription_id, exc
            )
            return

        for namespace in namespaces:
            location = azure_location(namespace)
            if not self._location_allowed(location):
                continue
            ns_name = getattr(namespace, "name", "")
            ns_id = getattr(namespace, "id", "")
            resource_group = resource_group_from_id(ns_id)

            try:
                queues = list(client.queues.list_by_namespace(resource_group, ns_name))
            except Exception as exc:
                logger.warning("Service Bus queue list failed for %s: %s", ns_name, exc)
                queues = []
            for queue in queues:
                queue_name = getattr(queue, "name", "")
                resource_id = getattr(queue, "id", "") or f"{ns_id}/queues/{queue_name}"
                yield self._build_store(
                    store_type="servicebus_queue",
                    display_name=f"{ns_name}/{queue_name}",
                    connection_hint={
                        "namespace": ns_name,
                        "queue": queue_name,
                        "resource_group": resource_group,
                    },
                    source="azure_servicebus",
                    region=location,
                    resource_id=resource_id,
                    subscription_id=subscription_id,
                    metadata={
                        "namespace_resource_id": ns_id,
                        "message_count": getattr(queue, "message_count", None),
                        "max_size_mb": getattr(queue, "max_size_in_megabytes", None),
                    },
                )

            try:
                topics = list(client.topics.list_by_namespace(resource_group, ns_name))
            except Exception as exc:
                logger.warning("Service Bus topic list failed for %s: %s", ns_name, exc)
                topics = []
            for topic in topics:
                topic_name = getattr(topic, "name", "")
                resource_id = getattr(topic, "id", "") or f"{ns_id}/topics/{topic_name}"
                yield self._build_store(
                    store_type="servicebus_topic",
                    display_name=f"{ns_name}/{topic_name}",
                    connection_hint={
                        "namespace": ns_name,
                        "topic": topic_name,
                        "resource_group": resource_group,
                    },
                    source="azure_servicebus",
                    region=location,
                    resource_id=resource_id,
                    subscription_id=subscription_id,
                    metadata={
                        "namespace_resource_id": ns_id,
                        "subscription_count": getattr(
                            topic, "subscription_count", None
                        ),
                        "max_size_mb": getattr(topic, "max_size_in_megabytes", None),
                    },
                )

    async def _enumerate_apim(
        self, subscription_id: str
    ) -> AsyncIterator[DiscoveredStore]:
        try:
            from azure.mgmt.apimanagement import ApiManagementClient
        except ImportError:
            raise ImportError(
                "azure-mgmt-apimanagement is required. "
                "Install with: pip install 'daita-agents[azure]'"
            )

        try:
            client = ApiManagementClient(self._credential, subscription_id)
            services = list(client.api_management_service.list())
        except Exception as exc:
            logger.warning(
                "API Management service list failed for %s: %s", subscription_id, exc
            )
            return

        for service in services:
            location = azure_location(service)
            if not self._location_allowed(location):
                continue
            service_name = getattr(service, "name", "")
            service_id = getattr(service, "id", "")
            resource_group = resource_group_from_id(service_id)
            try:
                apis = list(client.api.list_by_service(resource_group, service_name))
            except Exception as exc:
                logger.warning(
                    "API Management API list failed for %s: %s", service_name, exc
                )
                apis = []

            for api in apis:
                api_id = getattr(api, "name", "")
                resource_id = getattr(api, "id", "") or f"{service_id}/apis/{api_id}"
                yield self._build_store(
                    store_type="azure_apim",
                    display_name=f"{service_name}/{api_id}",
                    connection_hint={
                        "service": service_name,
                        "api_id": api_id,
                        "resource_group": resource_group,
                    },
                    source="azure_apim",
                    region=location,
                    resource_id=resource_id,
                    subscription_id=subscription_id,
                    metadata={
                        "service_resource_id": service_id,
                        "display_name": getattr(api, "display_name", "") or "",
                        "path": getattr(api, "path", "") or "",
                        "protocols": list(getattr(api, "protocols", None) or []),
                    },
                )
