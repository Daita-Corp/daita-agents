"""DynamoDB profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_dynamodb
from ._common import _dict_to_normalized_schema


class DynamoDBProfiler(BaseProfiler):
    """Profiles DynamoDB tables using boto3."""

    def supports(self, store_type: str) -> bool:
        return store_type == "dynamodb"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Describe a DynamoDB table and sample its items."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required. Install with: pip install 'daita-agents[aws]'"
            )

        from ..discovery import discover_dynamodb

        conn_hint = store.connection_hint
        result = await discover_dynamodb(
            table_name=conn_hint.get("table_name", ""),
            region=conn_hint.get("region", "us-east-1"),
        )

        normalized = normalize_dynamodb(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
