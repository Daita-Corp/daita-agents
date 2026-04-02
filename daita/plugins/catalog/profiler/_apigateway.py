"""API Gateway profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_apigateway
from ._common import _dict_to_normalized_schema


class APIGatewayProfiler(BaseProfiler):
    """Profiles API Gateway APIs (REST and HTTP) using boto3."""

    def supports(self, store_type: str) -> bool:
        return store_type == "apigateway"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Deep-profile an API Gateway API to extract endpoints and integrations."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required. Install with: pip install 'daita-agents[aws]'"
            )

        from ..discovery import discover_apigateway

        conn_hint = store.connection_hint
        result = await discover_apigateway(
            api_id=conn_hint.get("api_id", ""),
            api_type=conn_hint.get("api_type", "rest"),
            stage=conn_hint.get("stage", ""),
            region=conn_hint.get("region", "us-east-1"),
        )

        normalized = normalize_apigateway(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
