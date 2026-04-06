"""AWS API Gateway discovery (REST v1 and HTTP v2)."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _build_invoke_url(api_id: str, region: str, stage: str) -> str:
    """Build the API Gateway invoke URL for a given API + stage."""
    if not stage:
        return ""
    if stage == "$default":
        return f"https://{api_id}.execute-api.{region}.amazonaws.com"
    return f"https://{api_id}.execute-api.{region}.amazonaws.com/{stage}"


async def discover_apigateway(
    api_id: str,
    api_type: str = "rest",
    stage: str = "",
    region: str = "us-east-1",
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deep-profile a single API Gateway API (REST v1 or HTTP v2).

    Returns a raw result dict with endpoints, integrations, authorizers,
    and stage configuration. The integration URIs are the key data for
    lineage inference — they contain Lambda ARNs and HTTP endpoints.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required. Install with: pip install 'daita-agents[aws]'"
        )

    logger.debug("discover_apigateway: profiling %s (%s) in %s", api_id, api_type, region)

    kwargs: Dict[str, Any] = {}
    if profile_name:
        kwargs["profile_name"] = profile_name
    session = boto3.Session(**kwargs)

    if api_type == "rest":
        return await _discover_rest_api(session, api_id, stage, region)
    else:
        return await _discover_http_api(session, api_id, stage, region)


async def _discover_rest_api(
    session: Any, api_id: str, stage: str, region: str
) -> Dict[str, Any]:
    """Profile a REST API (apigateway v1)."""
    client = session.client("apigateway", region_name=region)

    # API metadata
    api_info = client.get_rest_api(restApiId=api_id)
    api_name = api_info.get("name", api_id)
    description = api_info.get("description", "")

    # Resources and methods
    endpoints = []
    resources_resp = client.get_resources(restApiId=api_id)
    for resource in resources_resp.get("items", []):
        path = resource.get("path", "/")
        for method in resource.get("resourceMethods", {}).keys():
            endpoint: Dict[str, Any] = {
                "path": path,
                "method": method,
                "authorization": "NONE",
                "api_key_required": False,
                "integration_type": "",
                "integration_uri": "",
            }
            try:
                method_info = client.get_method(
                    restApiId=api_id,
                    resourceId=resource["id"],
                    httpMethod=method,
                )
                endpoint["authorization"] = method_info.get("authorizationType", "NONE")
                endpoint["api_key_required"] = method_info.get("apiKeyRequired", False)
                integration = method_info.get("methodIntegration", {})
                endpoint["integration_type"] = integration.get("type", "")
                endpoint["integration_uri"] = integration.get("uri", "")
            except Exception:
                pass
            endpoints.append(endpoint)

    # Authorizers
    authorizers = []
    try:
        auth_resp = client.get_authorizers(restApiId=api_id)
        for auth in auth_resp.get("items", []):
            authorizers.append({
                "id": auth.get("id", ""),
                "name": auth.get("name", ""),
                "type": auth.get("type", ""),
            })
    except Exception:
        pass

    # Stage details
    stage_variables = {}
    if stage:
        try:
            stage_info = client.get_stage(restApiId=api_id, stageName=stage)
            stage_variables = stage_info.get("variables", {})
        except Exception:
            pass

    return {
        "api_type": "apigateway",
        "api_id": api_id,
        "api_name": api_name,
        "protocol_type": "REST",
        "region": region,
        "stage": stage,
        "description": description,
        "endpoint": _build_invoke_url(api_id, region, stage),
        "endpoints": endpoints,
        "endpoint_count": len(endpoints),
        "authorizers": authorizers,
        "stage_variables": stage_variables,
    }


async def _discover_http_api(
    session: Any, api_id: str, stage: str, region: str
) -> Dict[str, Any]:
    """Profile an HTTP API (apigatewayv2)."""
    client = session.client("apigatewayv2", region_name=region)

    # API metadata
    api_info = client.get_api(ApiId=api_id)
    api_name = api_info.get("Name", api_id)
    description = api_info.get("Description", "")
    protocol = api_info.get("ProtocolType", "HTTP")

    # Routes
    routes_resp = client.get_routes(ApiId=api_id)
    routes = routes_resp.get("Items", [])

    # Integrations (indexed by ID for route lookup)
    integrations_resp = client.get_integrations(ApiId=api_id)
    integrations_by_id: Dict[str, Dict[str, Any]] = {}
    for integ in integrations_resp.get("Items", []):
        integrations_by_id[integ["IntegrationId"]] = integ

    endpoints = []
    for route in routes:
        route_key = route.get("RouteKey", "")
        parts = route_key.split(" ", 1)
        method = parts[0] if parts else route_key
        path = parts[1] if len(parts) > 1 else "/"

        # Look up integration for this route
        target = route.get("Target", "")
        integration_id = target.replace("integrations/", "") if target.startswith("integrations/") else ""
        integ = integrations_by_id.get(integration_id, {})

        endpoints.append({
            "path": path,
            "method": method,
            "authorization": route.get("AuthorizationType", "NONE"),
            "api_key_required": route.get("ApiKeyRequired", False),
            "integration_type": integ.get("IntegrationType", ""),
            "integration_uri": integ.get("IntegrationUri", ""),
        })

    # Authorizers
    authorizers = []
    try:
        auth_resp = client.get_authorizers(ApiId=api_id)
        for auth in auth_resp.get("Items", []):
            authorizers.append({
                "id": auth.get("AuthorizerId", ""),
                "name": auth.get("Name", ""),
                "type": auth.get("AuthorizerType", ""),
            })
    except Exception:
        pass

    # Stage details
    stage_variables = {}
    if stage:
        try:
            stage_info = client.get_stage(ApiId=api_id, StageName=stage)
            stage_variables = stage_info.get("StageVariables", {}) or {}
        except Exception:
            pass

    return {
        "api_type": "apigateway",
        "api_id": api_id,
        "api_name": api_name,
        "protocol_type": protocol,
        "region": region,
        "stage": stage,
        "description": description,
        "endpoint": _build_invoke_url(api_id, region, stage),
        "endpoints": endpoints,
        "endpoint_count": len(endpoints),
        "authorizers": authorizers,
        "stage_variables": stage_variables,
    }
