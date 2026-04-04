"""SQS profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_sqs
from ._common import _dict_to_normalized_schema


class SQSProfiler(BaseProfiler):
    """Profiles SQS queues by inspecting attributes and sampling messages."""

    def supports(self, store_type: str) -> bool:
        return store_type == "sqs"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Inspect an SQS queue and sample messages to infer schema."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required. Install with: pip install 'daita-agents[aws]'"
            )

        from ..discovery import discover_sqs

        conn_hint = store.connection_hint
        result = await discover_sqs(
            queue_url=conn_hint.get("queue_url", ""),
            region=conn_hint.get("region", "us-east-1"),
        )

        normalized = normalize_sqs(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
