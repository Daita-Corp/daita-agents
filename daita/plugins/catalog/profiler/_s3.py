"""S3 profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_s3
from ._common import _dict_to_normalized_schema


class S3Profiler(BaseProfiler):
    """Profiles S3 buckets by sampling objects."""

    def supports(self, store_type: str) -> bool:
        return store_type == "s3"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Sample an S3 bucket to infer its structure."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required. Install with: pip install 'daita-agents[aws]'"
            )

        from ..discovery import discover_s3

        conn_hint = store.connection_hint
        result = await discover_s3(
            bucket=conn_hint.get("bucket", ""),
            region=conn_hint.get("region", "us-east-1"),
        )

        normalized = normalize_s3(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
