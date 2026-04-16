"""GCP API Gateway profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_gcp_apigateway
from ._common import _dict_to_normalized_schema


class GCPAPIGatewayProfiler(BaseProfiler):
    """Profiles GCP API Gateway APIs by listing configs and gateways."""

    def supports(self, store_type: str) -> bool:
        return store_type == "gcp_apigateway"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_gcp_apigateway

        hint = store.connection_hint
        result = await discover_gcp_apigateway(
            project=hint.get("project", ""),
            api_id=hint.get("api_id", ""),
            location=hint.get("location", "global"),
            credentials_path=hint.get("credentials_path"),
            impersonate_service_account=hint.get("impersonate_service_account"),
        )
        return _dict_to_normalized_schema(
            normalize_gcp_apigateway(result), store_id=store.id
        )
