"""Kinesis profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_kinesis
from ._common import _dict_to_normalized_schema


class KinesisProfiler(BaseProfiler):
    """Profiles Kinesis streams by describing shards and sampling records."""

    def supports(self, store_type: str) -> bool:
        return store_type == "kinesis"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Describe a Kinesis stream and sample records to infer schema."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required. Install with: pip install 'daita-agents[aws]'"
            )

        from ..discovery import discover_kinesis

        conn_hint = store.connection_hint
        result = await discover_kinesis(
            stream_name=conn_hint.get("stream_name", ""),
            region=conn_hint.get("region", "us-east-1"),
        )

        normalized = normalize_kinesis(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
