"""GCP API Gateway discovery."""

import logging
from typing import Any, Optional

from ._gcp_common import gcp_credentials

logger = logging.getLogger(__name__)


async def discover_gcp_apigateway(
    project: str,
    api_id: str,
    location: str = "global",
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
) -> dict[str, Any]:
    """Describe a GCP API Gateway API and its configs/gateways."""
    try:
        from google.cloud import apigateway_v1
    except ImportError:
        raise ImportError(
            "google-cloud-api-gateway is required. "
            "Install with: pip install 'daita-agents[gcp]'"
        )

    logger.debug("discover_gcp_apigateway: %s/%s/%s", project, location, api_id)

    creds, _ = gcp_credentials(
        credentials_path=credentials_path,
        impersonate_service_account=impersonate_service_account,
    )
    client = apigateway_v1.ApiGatewayServiceClient(credentials=creds)
    api_name = f"projects/{project}/locations/{location}/apis/{api_id}"
    api_parent = f"projects/{project}/locations/*"

    display_name = ""
    state = ""
    try:
        api = client.get_api(name=api_name)
        display_name = api.display_name or ""
        state = apigateway_v1.Api.State(api.state).name if api.state else ""
    except Exception as exc:
        logger.debug("API Gateway get_api failed for %s: %s", api_name, exc)

    configs: list[dict[str, str]] = []
    try:
        for cfg in client.list_api_configs(parent=api_name):
            configs.append(
                {
                    "name": cfg.name.split("/")[-1],
                    "display_name": cfg.display_name or "",
                    "state": (
                        apigateway_v1.ApiConfig.State(cfg.state).name
                        if cfg.state
                        else ""
                    ),
                }
            )
    except Exception as exc:
        logger.debug("API Gateway list_api_configs failed for %s: %s", api_name, exc)

    gateways: list[dict[str, str]] = []
    try:
        for gw in client.list_gateways(parent=api_parent):
            if gw.api_config and api_id in gw.api_config:
                gateways.append(
                    {
                        "name": gw.name.split("/")[-1],
                        "default_hostname": gw.default_hostname or "",
                        "state": (
                            apigateway_v1.Gateway.State(gw.state).name
                            if gw.state
                            else ""
                        ),
                    }
                )
    except Exception as exc:
        logger.debug("API Gateway list_gateways failed: %s", exc)

    return {
        "database_type": "gcp_apigateway",
        "project": project,
        "location": location,
        "api_id": api_id,
        "resource_name": api_name,
        "display_name": display_name,
        "state": state,
        "configs": configs,
        "gateways": gateways,
    }
