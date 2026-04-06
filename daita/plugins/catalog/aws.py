"""
AWS infrastructure discoverer.

Enumerates RDS instances, DynamoDB tables, S3 buckets, ElastiCache clusters,
Redshift clusters, API Gateway APIs, SQS queues, SNS topics, OpenSearch domains,
DocumentDB clusters, and Kinesis streams across configured AWS regions using boto3.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from .base_discoverer import BaseDiscoverer, DiscoveredStore
from .discovery._apigateway import _build_invoke_url

logger = logging.getLogger(__name__)

# AWS services to discover and their enumeration methods
_RDS_ENGINES_TO_STORE_TYPE = {
    "postgres": "postgresql",
    "mysql": "mysql",
    "mariadb": "mysql",
    "aurora-postgresql": "postgresql",
    "aurora-mysql": "mysql",
    "oracle-ee": "oracle",
    "oracle-se2": "oracle",
    "sqlserver-ee": "sqlserver",
    "sqlserver-se": "sqlserver",
}


class AWSDiscoverer(BaseDiscoverer):
    """
    Discover data stores across AWS accounts and regions.

    Enumerates:
    - RDS instances and Aurora clusters
    - DynamoDB tables
    - S3 buckets
    - ElastiCache clusters
    - Redshift clusters
    - API Gateway REST APIs and HTTP APIs
    - SQS queues
    - SNS topics
    - OpenSearch domains
    - DocumentDB clusters
    - Kinesis Data Streams

    Lazy imports boto3 inside authenticate().
    """

    name = "aws"

    def __init__(
        self,
        regions: Optional[List[str]] = None,
        role_arn: Optional[str] = None,
        profile_name: Optional[str] = None,
        services: Optional[List[str]] = None,
    ):
        """
        Args:
            regions: AWS regions to scan. Defaults to ["us-east-1"].
            role_arn: Optional IAM role ARN to assume for cross-account discovery.
            profile_name: Optional AWS CLI profile name.
            services: Services to enumerate. Defaults to all supported.
                      Options: "rds", "dynamodb", "s3", "elasticache", "redshift",
                      "apigateway", "sqs", "sns", "opensearch", "documentdb", "kinesis"
        """
        self._regions = regions or ["us-east-1"]
        self._role_arn = role_arn
        self._profile_name = profile_name
        self._services = services or [
            "rds",
            "dynamodb",
            "s3",
            "elasticache",
            "redshift",
            "apigateway",
            "sqs",
            "sns",
            "opensearch",
            "documentdb",
            "kinesis",
        ]
        self._session = None
        self._account_id: Optional[str] = None

    async def authenticate(self) -> None:
        """Set up boto3 session and resolve account ID."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required. Install with: pip install 'daita-agents[aws]'"
            )

        kwargs: Dict[str, Any] = {}
        if self._profile_name:
            kwargs["profile_name"] = self._profile_name

        self._session = boto3.Session(**kwargs)

        if self._role_arn:
            sts = self._session.client("sts")
            creds = sts.assume_role(
                RoleArn=self._role_arn,
                RoleSessionName="daita-catalog-discovery",
            )["Credentials"]
            self._session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )

        try:
            sts = self._session.client("sts")
            self._account_id = sts.get_caller_identity()["Account"]
        except Exception:
            self._account_id = "unknown"

    async def enumerate(self) -> AsyncIterator[DiscoveredStore]:
        """Yield discovered stores across all configured regions and services.

        TODO: boto3 paginators are synchronous and block the event loop.
        Wrap calls in asyncio.to_thread() or migrate to aiobotocore for
        true async I/O when scanning many regions/services.
        """
        if not self._session:
            await self.authenticate()

        for region in self._regions:
            if "rds" in self._services:
                async for store in self._enumerate_rds(region):
                    yield store
            if "dynamodb" in self._services:
                async for store in self._enumerate_dynamodb(region):
                    yield store
            if "s3" in self._services:
                async for store in self._enumerate_s3(region):
                    yield store
            if "elasticache" in self._services:
                async for store in self._enumerate_elasticache(region):
                    yield store
            if "redshift" in self._services:
                async for store in self._enumerate_redshift(region):
                    yield store
            if "apigateway" in self._services:
                async for store in self._enumerate_apigateway(region):
                    yield store
            if "sqs" in self._services:
                async for store in self._enumerate_sqs(region):
                    yield store
            if "sns" in self._services:
                async for store in self._enumerate_sns(region):
                    yield store
            if "opensearch" in self._services:
                async for store in self._enumerate_opensearch(region):
                    yield store
            if "documentdb" in self._services:
                async for store in self._enumerate_documentdb(region):
                    yield store
            if "kinesis" in self._services:
                async for store in self._enumerate_kinesis(region):
                    yield store

    def fingerprint(self, store: DiscoveredStore) -> str:
        """AWS-specific fingerprint using account + region + ARN."""
        arn = store.metadata.get("arn", "")
        raw = f"aws|{self._account_id}|{store.region}|{arn}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def test_access(self) -> bool:
        """Verify AWS credentials are valid."""
        try:
            await self.authenticate()
            return self._account_id != "unknown"
        except Exception:
            return False

    async def _enumerate_rds(self, region: str) -> AsyncIterator[DiscoveredStore]:
        """Enumerate RDS instances and Aurora clusters in a region."""
        client = self._session.client("rds", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            paginator = client.get_paginator("describe_db_instances")
            for page in paginator.paginate():
                for db in page["DBInstances"]:
                    engine = db.get("Engine", "unknown")
                    store_type = _RDS_ENGINES_TO_STORE_TYPE.get(engine, engine)
                    identifier = db["DBInstanceIdentifier"]
                    arn = db.get("DBInstanceArn", "")

                    store = DiscoveredStore(
                        id="",  # set below
                        store_type=store_type,
                        display_name=f"{identifier} ({region})",
                        connection_hint={
                            "host": db.get("Endpoint", {}).get("Address", ""),
                            "port": db.get("Endpoint", {}).get("Port"),
                            "dbname": db.get("DBName", ""),
                            "engine": engine,
                        },
                        source="aws_rds",
                        region=region,
                        confidence=0.95,
                        tags=[
                            f"{t['Key']}={t['Value']}" for t in db.get("TagList", [])
                        ],
                        metadata={
                            "arn": arn,
                            "engine_version": db.get("EngineVersion", ""),
                            "instance_class": db.get("DBInstanceClass", ""),
                            "storage_gb": db.get("AllocatedStorage"),
                            "multi_az": db.get("MultiAZ", False),
                            "status": db.get("DBInstanceStatus", ""),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("RDS enumeration failed in %s: %s", region, exc)

    async def _enumerate_dynamodb(self, region: str) -> AsyncIterator[DiscoveredStore]:
        """Enumerate DynamoDB tables in a region."""
        client = self._session.client("dynamodb", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            paginator = client.get_paginator("list_tables")
            for page in paginator.paginate():
                for table_name in page.get("TableNames", []):
                    try:
                        desc = client.describe_table(TableName=table_name)["Table"]
                    except Exception:
                        desc = {}

                    arn = desc.get(
                        "TableArn",
                        f"arn:aws:dynamodb:{region}:{self._account_id}:table/{table_name}",
                    )
                    store = DiscoveredStore(
                        id="",
                        store_type="dynamodb",
                        display_name=f"{table_name} ({region})",
                        connection_hint={
                            "table_name": table_name,
                            "region": region,
                        },
                        source="aws_dynamodb",
                        region=region,
                        confidence=0.95,
                        metadata={
                            "arn": arn,
                            "item_count": desc.get("ItemCount"),
                            "size_bytes": desc.get("TableSizeBytes"),
                            "status": desc.get("TableStatus", ""),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("DynamoDB enumeration failed in %s: %s", region, exc)

    async def _enumerate_s3(self, region: str) -> AsyncIterator[DiscoveredStore]:
        """Enumerate S3 buckets. Only runs once (S3 is global)."""
        if region != self._regions[0]:
            return  # S3 is global, only enumerate once

        client = self._session.client("s3", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            response = client.list_buckets()
            for bucket in response.get("Buckets", []):
                bucket_name = bucket["Name"]

                # Get bucket region
                try:
                    loc = client.get_bucket_location(Bucket=bucket_name)
                    bucket_region = loc.get("LocationConstraint") or "us-east-1"
                except Exception:
                    bucket_region = "unknown"

                store = DiscoveredStore(
                    id="",
                    store_type="s3",
                    display_name=f"s3://{bucket_name} ({bucket_region})",
                    connection_hint={
                        "bucket": bucket_name,
                        "region": bucket_region,
                    },
                    source="aws_s3",
                    region=bucket_region,
                    confidence=0.95,
                    metadata={
                        "arn": f"arn:aws:s3:::{bucket_name}",
                        "created": str(bucket.get("CreationDate", "")),
                    },
                    discovered_at=now,
                    last_seen=now,
                )
                store.id = self.fingerprint(store)
                yield store
        except Exception as exc:
            logger.warning("S3 enumeration failed: %s", exc)

    async def _enumerate_elasticache(
        self, region: str
    ) -> AsyncIterator[DiscoveredStore]:
        """Enumerate ElastiCache clusters in a region."""
        client = self._session.client("elasticache", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            paginator = client.get_paginator("describe_cache_clusters")
            for page in paginator.paginate(ShowCacheNodeInfo=True):
                for cluster in page.get("CacheClusters", []):
                    cluster_id = cluster["CacheClusterId"]
                    engine = cluster.get("Engine", "redis")

                    # Get endpoint
                    endpoint = {}
                    nodes = cluster.get("CacheNodes", [])
                    if nodes:
                        ep = nodes[0].get("Endpoint", {})
                        endpoint = {
                            "host": ep.get("Address", ""),
                            "port": ep.get("Port"),
                        }

                    store = DiscoveredStore(
                        id="",
                        store_type=engine,  # "redis" or "memcached"
                        display_name=f"{cluster_id} ({region})",
                        connection_hint=endpoint,
                        source="aws_elasticache",
                        region=region,
                        confidence=0.95,
                        metadata={
                            "arn": cluster.get("ARN", ""),
                            "engine_version": cluster.get("EngineVersion", ""),
                            "node_type": cluster.get("CacheNodeType", ""),
                            "num_nodes": cluster.get("NumCacheNodes"),
                            "status": cluster.get("CacheClusterStatus", ""),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("ElastiCache enumeration failed in %s: %s", region, exc)

    async def _enumerate_redshift(self, region: str) -> AsyncIterator[DiscoveredStore]:
        """Enumerate Redshift clusters in a region."""
        client = self._session.client("redshift", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            paginator = client.get_paginator("describe_clusters")
            for page in paginator.paginate():
                for cluster in page.get("Clusters", []):
                    cluster_id = cluster["ClusterIdentifier"]
                    endpoint = cluster.get("Endpoint", {})

                    store = DiscoveredStore(
                        id="",
                        store_type="redshift",
                        display_name=f"{cluster_id} ({region})",
                        connection_hint={
                            "host": endpoint.get("Address", ""),
                            "port": endpoint.get("Port"),
                            "dbname": cluster.get("DBName", "dev"),
                        },
                        source="aws_redshift",
                        region=region,
                        confidence=0.95,
                        metadata={
                            "arn": f"arn:aws:redshift:{region}:{self._account_id}:cluster:{cluster_id}",
                            "node_type": cluster.get("NodeType", ""),
                            "num_nodes": cluster.get("NumberOfNodes"),
                            "status": cluster.get("ClusterStatus", ""),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("Redshift enumeration failed in %s: %s", region, exc)

    async def _enumerate_apigateway(
        self, region: str
    ) -> AsyncIterator[DiscoveredStore]:
        """Enumerate API Gateway REST APIs (v1) and HTTP APIs (v2) in a region."""
        now = datetime.now(timezone.utc).isoformat()

        # --- REST APIs (apigateway v1) ---
        try:
            client = self._session.client("apigateway", region_name=region)
            paginator = client.get_paginator("get_rest_apis")
            for page in paginator.paginate():
                for api in page.get("items", []):
                    api_id = api["id"]
                    api_name = api.get("name", api_id)
                    description = api.get("description", "")

                    # Get stages for this API
                    stages = []
                    try:
                        stages_resp = client.get_stages(restApiId=api_id)
                        stages = [s["stageName"] for s in stages_resp.get("item", [])]
                    except Exception:
                        stages = ["(no stage)"]

                    # Count resources/endpoints
                    resource_count = 0
                    try:
                        resources_resp = client.get_resources(restApiId=api_id)
                        for res in resources_resp.get("items", []):
                            resource_count += len(res.get("resourceMethods", {}))
                    except Exception:
                        pass

                    for stage in stages:
                        store = DiscoveredStore(
                            id="",
                            store_type="apigateway",
                            display_name=f"{api_name}/{stage} ({region})",
                            connection_hint={
                                "api_id": api_id,
                                "api_type": "rest",
                                "stage": stage,
                                "region": region,
                                "endpoint": _build_invoke_url(api_id, region, stage),
                            },
                            source="aws_apigateway",
                            region=region,
                            confidence=0.95,
                            metadata={
                                "arn": f"arn:aws:apigateway:{region}::/restapis/{api_id}",
                                "api_name": api_name,
                                "description": description,
                                "resource_count": resource_count,
                                "created_date": str(api.get("createdDate", "")),
                                "protocol_type": "REST",
                            },
                            discovered_at=now,
                            last_seen=now,
                        )
                        store.id = self.fingerprint(store)
                        yield store
        except Exception as exc:
            logger.warning("API Gateway REST enumeration failed in %s: %s", region, exc)

        # --- HTTP APIs (apigatewayv2) ---
        try:
            v2_client = self._session.client("apigatewayv2", region_name=region)
            paginator = v2_client.get_paginator("get_apis")
            for page in paginator.paginate():
                for api in page.get("Items", []):
                    api_id = api["ApiId"]
                    api_name = api.get("Name", api_id)
                    description = api.get("Description", "")
                    protocol = api.get("ProtocolType", "HTTP")

                    # Get stages
                    stages = []
                    try:
                        stages_resp = v2_client.get_stages(ApiId=api_id)
                        stages = [s["StageName"] for s in stages_resp.get("Items", [])]
                    except Exception:
                        stages = ["$default"]

                    # Count routes
                    route_count = 0
                    try:
                        routes_resp = v2_client.get_routes(ApiId=api_id)
                        route_count = len(routes_resp.get("Items", []))
                    except Exception:
                        pass

                    for stage in stages:
                        store = DiscoveredStore(
                            id="",
                            store_type="apigateway",
                            display_name=f"{api_name}/{stage} ({region})",
                            connection_hint={
                                "api_id": api_id,
                                "api_type": "http",
                                "stage": stage,
                                "region": region,
                                "endpoint": _build_invoke_url(api_id, region, stage),
                            },
                            source="aws_apigateway",
                            region=region,
                            confidence=0.95,
                            metadata={
                                "arn": f"arn:aws:apigateway:{region}::/apis/{api_id}",
                                "api_name": api_name,
                                "description": description,
                                "resource_count": route_count,
                                "created_date": str(api.get("CreatedDate", "")),
                                "protocol_type": protocol,
                            },
                            discovered_at=now,
                            last_seen=now,
                        )
                        store.id = self.fingerprint(store)
                        yield store
        except Exception as exc:
            logger.warning("API Gateway HTTP enumeration failed in %s: %s", region, exc)

    async def _enumerate_sqs(self, region: str) -> AsyncIterator[DiscoveredStore]:
        """Enumerate SQS queues in a region."""
        client = self._session.client("sqs", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            paginator = client.get_paginator("list_queues")
            for page in paginator.paginate():
                for queue_url in page.get("QueueUrls", []):
                    try:
                        attrs = client.get_queue_attributes(
                            QueueUrl=queue_url,
                            AttributeNames=["All"],
                        )["Attributes"]
                    except Exception:
                        attrs = {}

                    arn = attrs.get("QueueArn", "")
                    queue_name = (
                        arn.rsplit(":", 1)[-1] if arn else queue_url.rsplit("/", 1)[-1]
                    )

                    store = DiscoveredStore(
                        id="",
                        store_type="sqs",
                        display_name=f"{queue_name} ({region})",
                        connection_hint={
                            "queue_url": queue_url,
                            "region": region,
                        },
                        source="aws_sqs",
                        region=region,
                        confidence=0.95,
                        metadata={
                            "arn": arn,
                            "approximate_message_count": int(
                                attrs.get("ApproximateNumberOfMessages", 0)
                            ),
                            "approximate_not_visible": int(
                                attrs.get("ApproximateNumberOfMessagesNotVisible", 0)
                            ),
                            "visibility_timeout": int(
                                attrs.get("VisibilityTimeout", 30)
                            ),
                            "created_timestamp": attrs.get("CreatedTimestamp", ""),
                            "is_fifo": queue_name.endswith(".fifo"),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("SQS enumeration failed in %s: %s", region, exc)

    async def _enumerate_sns(self, region: str) -> AsyncIterator[DiscoveredStore]:
        """Enumerate SNS topics in a region."""
        client = self._session.client("sns", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            paginator = client.get_paginator("list_topics")
            for page in paginator.paginate():
                for topic in page.get("Topics", []):
                    topic_arn = topic["TopicArn"]
                    topic_name = topic_arn.rsplit(":", 1)[-1]

                    try:
                        attrs = client.get_topic_attributes(TopicArn=topic_arn)[
                            "Attributes"
                        ]
                    except Exception:
                        attrs = {}

                    store = DiscoveredStore(
                        id="",
                        store_type="sns",
                        display_name=f"{topic_name} ({region})",
                        connection_hint={
                            "topic_arn": topic_arn,
                            "region": region,
                        },
                        source="aws_sns",
                        region=region,
                        confidence=0.95,
                        metadata={
                            "arn": topic_arn,
                            "subscription_count": int(
                                attrs.get("SubscriptionsConfirmed", 0)
                            ),
                            "subscriptions_pending": int(
                                attrs.get("SubscriptionsPending", 0)
                            ),
                            "display_name": attrs.get("DisplayName", ""),
                            "is_fifo": topic_name.endswith(".fifo"),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("SNS enumeration failed in %s: %s", region, exc)

    async def _enumerate_opensearch(
        self, region: str
    ) -> AsyncIterator[DiscoveredStore]:
        """Enumerate OpenSearch (Elasticsearch) domains in a region."""
        client = self._session.client("opensearch", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            domain_names = [
                d["DomainName"]
                for d in client.list_domain_names().get("DomainNames", [])
            ]

            if not domain_names:
                return

            # describe_domains accepts up to 5 at a time
            for i in range(0, len(domain_names), 5):
                batch = domain_names[i : i + 5]
                resp = client.describe_domains(DomainNames=batch)

                for domain in resp.get("DomainStatusList", []):
                    domain_name = domain["DomainName"]
                    endpoint = domain.get("Endpoint") or domain.get(
                        "Endpoints", {}
                    ).get("vpc", "")
                    arn = domain.get("ARN", "")
                    engine_version = domain.get("EngineVersion", "")

                    cluster_config = domain.get("ClusterConfig", {})

                    store = DiscoveredStore(
                        id="",
                        store_type="opensearch",
                        display_name=f"{domain_name} ({region})",
                        connection_hint={
                            "host": endpoint,
                            "port": 443,
                            "region": region,
                        },
                        source="aws_opensearch",
                        region=region,
                        confidence=0.95,
                        metadata={
                            "arn": arn,
                            "engine_version": engine_version,
                            "instance_type": cluster_config.get("InstanceType", ""),
                            "instance_count": cluster_config.get("InstanceCount", 1),
                            "dedicated_master": cluster_config.get(
                                "DedicatedMasterEnabled", False
                            ),
                            "zone_awareness": cluster_config.get(
                                "ZoneAwarenessEnabled", False
                            ),
                            "status": (
                                "processing" if domain.get("Processing") else "active"
                            ),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("OpenSearch enumeration failed in %s: %s", region, exc)

    async def _enumerate_documentdb(
        self, region: str
    ) -> AsyncIterator[DiscoveredStore]:
        """Enumerate DocumentDB clusters in a region."""
        client = self._session.client("docdb", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            paginator = client.get_paginator("describe_db_clusters")
            for page in paginator.paginate(
                Filters=[{"Name": "engine", "Values": ["docdb"]}]
            ):
                for cluster in page.get("DBClusters", []):
                    cluster_id = cluster["DBClusterIdentifier"]
                    endpoint = cluster.get("Endpoint", "")
                    port = cluster.get("Port", 27017)

                    store = DiscoveredStore(
                        id="",
                        store_type="documentdb",
                        display_name=f"{cluster_id} ({region})",
                        connection_hint={
                            "host": endpoint,
                            "port": port,
                            "region": region,
                        },
                        source="aws_documentdb",
                        region=region,
                        confidence=0.95,
                        metadata={
                            "arn": cluster.get("DBClusterArn", ""),
                            "engine_version": cluster.get("EngineVersion", ""),
                            "instance_count": len(cluster.get("DBClusterMembers", [])),
                            "status": cluster.get("Status", ""),
                            "reader_endpoint": cluster.get("ReaderEndpoint", ""),
                            "storage_encrypted": cluster.get("StorageEncrypted", False),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("DocumentDB enumeration failed in %s: %s", region, exc)

    async def _enumerate_kinesis(self, region: str) -> AsyncIterator[DiscoveredStore]:
        """Enumerate Kinesis Data Streams in a region."""
        client = self._session.client("kinesis", region_name=region)
        now = datetime.now(timezone.utc).isoformat()

        try:
            paginator = client.get_paginator("list_streams")
            for page in paginator.paginate():
                for stream_name in page.get("StreamNames", []):
                    try:
                        summary = client.describe_stream_summary(
                            StreamName=stream_name
                        )["StreamDescriptionSummary"]
                    except Exception:
                        summary = {}

                    arn = summary.get(
                        "StreamARN",
                        f"arn:aws:kinesis:{region}:{self._account_id}:stream/{stream_name}",
                    )

                    store = DiscoveredStore(
                        id="",
                        store_type="kinesis",
                        display_name=f"{stream_name} ({region})",
                        connection_hint={
                            "stream_name": stream_name,
                            "region": region,
                        },
                        source="aws_kinesis",
                        region=region,
                        confidence=0.95,
                        metadata={
                            "arn": arn,
                            "shard_count": summary.get("OpenShardCount", 0),
                            "retention_hours": summary.get("RetentionPeriodHours", 24),
                            "status": summary.get("StreamStatus", ""),
                            "stream_mode": summary.get("StreamModeDetails", {}).get(
                                "StreamMode", "PROVISIONED"
                            ),
                            "consumer_count": summary.get("ConsumerCount", 0),
                        },
                        discovered_at=now,
                        last_seen=now,
                    )
                    store.id = self.fingerprint(store)
                    yield store
        except Exception as exc:
            logger.warning("Kinesis enumeration failed in %s: %s", region, exc)
