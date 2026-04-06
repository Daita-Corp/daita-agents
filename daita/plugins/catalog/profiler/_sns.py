"""SNS profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_sns
from ._common import _dict_to_normalized_schema


class SNSProfiler(BaseProfiler):
    """Profiles SNS topics by inspecting attributes and listing subscriptions."""

    def supports(self, store_type: str) -> bool:
        return store_type == "sns"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Inspect an SNS topic and its subscriptions."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required. Install with: pip install 'daita-agents[aws]'"
            )

        from ..discovery import discover_sns

        conn_hint = store.connection_hint
        result = await discover_sns(
            topic_arn=conn_hint.get("topic_arn", ""),
            region=conn_hint.get("region", "us-east-1"),
        )

        normalized = normalize_sns(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
