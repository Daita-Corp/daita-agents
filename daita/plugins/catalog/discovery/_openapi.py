"""OpenAPI/Swagger spec discovery."""

from typing import Any, Dict, Optional

from ._utils import validate_openapi_url


async def discover_openapi(
    spec_url: str,
    service_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch and parse an OpenAPI/Swagger spec.

    Returns a raw result dict with endpoints and API metadata.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required. Install with: pip install 'daita-agents[github]'"
        )
    import yaml

    url_error = validate_openapi_url(spec_url)
    if url_error:
        from ...core.exceptions import ValidationError

        raise ValidationError(url_error)

    # follow_redirects=False prevents redirect-based SSRF bypasses.
    # timeout guards against slow-loris / hung internal endpoints.
    async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
        resp = await client.get(spec_url)
        resp.raise_for_status()

        if spec_url.endswith(".yaml") or spec_url.endswith(".yml"):
            spec = yaml.safe_load(resp.text)
        else:
            spec = resp.json()

    base_url = ""
    if spec.get("servers"):
        base_url = spec["servers"][0].get("url", "")

    svc_name = service_name or spec.get("info", {}).get("title", "Unknown API")

    # Extract endpoints
    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method.startswith("x-"):
                continue

            endpoints.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "summary": details.get("summary", ""),
                    "description": details.get("description", ""),
                    "parameters": details.get("parameters", []),
                    "request_body": details.get("requestBody", {}),
                    "responses": details.get("responses", {}),
                }
            )

    return {
        "api_type": "openapi",
        "service_name": svc_name,
        "base_url": base_url,
        "version": spec.get("info", {}).get("version", "unknown"),
        "endpoints": endpoints,
        "endpoint_count": len(endpoints),
    }
