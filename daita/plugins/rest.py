"""
REST API plugin for Daita Agents.

Simple REST API client - no over-engineering.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    RiskLevel,
    Task,
    ToolView,
)

from .base import ConnectorPlugin
from .manifest import PluginKind, PluginManifest

if TYPE_CHECKING:
    from ..core.tools import LocalTool

logger = logging.getLogger(__name__)

_HTTP_GET_PARAMETERS = {
    "type": "object",
    "properties": {
        "endpoint": {
            "type": "string",
            "description": "API endpoint path (without base URL, e.g., /users or /data/123)",
        },
        "params": {
            "type": "object",
            "description": "Optional query parameters as key-value pairs",
        },
    },
    "required": ["endpoint"],
}

_HTTP_BODY_PARAMETERS = {
    "type": "object",
    "properties": {
        "endpoint": {
            "type": "string",
            "description": "API endpoint path (without base URL)",
        },
        "data": {
            "type": "object",
            "description": "JSON data to send in the request body",
        },
    },
    "required": ["endpoint", "data"],
}

_HTTP_DELETE_PARAMETERS = {
    "type": "object",
    "properties": {
        "endpoint": {
            "type": "string",
            "description": "API endpoint path (without base URL)",
        },
        "params": {
            "type": "object",
            "description": "Optional query parameters",
        },
    },
    "required": ["endpoint"],
}

_REST_TOOL_DEFINITIONS = (
    {
        "name": "http_get",
        "method": "GET",
        "capability_id": "rest.http.get",
        "description": "Make an HTTP GET request to the REST API endpoint. Use for retrieving data.",
        "parameters": _HTTP_GET_PARAMETERS,
        "operation_types": frozenset({"api.http.read"}),
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": False,
    },
    {
        "name": "http_post",
        "method": "POST",
        "capability_id": "rest.http.post",
        "description": "Make an HTTP POST request to the REST API endpoint. Use for creating new resources.",
        "parameters": _HTTP_BODY_PARAMETERS,
        "operation_types": frozenset({"api.http.write"}),
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": False,
        "idempotent": False,
        "side_effecting": True,
    },
    {
        "name": "http_put",
        "method": "PUT",
        "capability_id": "rest.http.put",
        "description": "Make an HTTP PUT request to the REST API endpoint. Use for updating existing resources.",
        "parameters": _HTTP_BODY_PARAMETERS,
        "operation_types": frozenset({"api.http.write"}),
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": True,
        "idempotent": True,
        "side_effecting": True,
    },
    {
        "name": "http_patch",
        "method": "PATCH",
        "capability_id": "rest.http.patch",
        "description": "Make an HTTP PATCH request to the REST API endpoint. Use for partial resource updates.",
        "parameters": _HTTP_BODY_PARAMETERS,
        "operation_types": frozenset({"api.http.write"}),
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": False,
        "idempotent": False,
        "side_effecting": True,
    },
    {
        "name": "http_delete",
        "method": "DELETE",
        "capability_id": "rest.http.delete",
        "description": "Make an HTTP DELETE request to the REST API endpoint. Use for deleting resources.",
        "parameters": _HTTP_DELETE_PARAMETERS,
        "operation_types": frozenset({"api.http.write"}),
        "access": AccessMode.WRITE,
        "risk": RiskLevel.HIGH,
        "retry_safe": False,
        "idempotent": False,
        "side_effecting": True,
    },
)

_REST_TOOL_BY_CAPABILITY = {
    definition["capability_id"]: definition for definition in _REST_TOOL_DEFINITIONS
}


class _RestHttpExecutor:
    """Executor backing REST plugin capability declarations."""

    id = "rest.http"
    capability_ids = frozenset(_REST_TOOL_BY_CAPABILITY)

    def __init__(self, plugin: "RESTPlugin") -> None:
        self._plugin = plugin

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        definition = _REST_TOOL_BY_CAPABILITY.get(task.capability_id)
        if definition is None:
            raise ValueError(f"Unsupported REST capability: {task.capability_id}")

        tool_name = definition["name"]
        handler = self._plugin._tool_handler(tool_name)
        result = await handler(dict(task.input))
        tool_view = context.get("tool_view")
        tool_view_name = (
            tool_view.get("name") if isinstance(tool_view, Mapping) else None
        )
        return [
            Evidence(
                kind="api.http.response",
                owner="rest",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "method": definition["method"],
                    "endpoint": task.input.get("endpoint"),
                    "response": result,
                },
                metadata={
                    "capability_id": task.capability_id,
                    "tool_view": tool_view_name,
                },
            )
        ]


class RESTPlugin(ConnectorPlugin):
    """
    Simple REST API plugin for agents.

    Just makes HTTP requests and handles responses. Nothing fancy.
    """

    manifest = PluginManifest(
        id="rest",
        display_name="REST API",
        version="2.0.0",
        kind=PluginKind.CONNECTOR,
        domains=frozenset({"api", "http"}),
        provides=frozenset({"http_client", "api_requests"}),
    )

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer",
        timeout: int = 30,
        **kwargs,
    ):
        """
        Initialize REST API client.

        Args:
            base_url: Base URL for all requests
            api_key: Optional API key for authentication
            auth_header: Header name for authentication (default: "Authorization")
            auth_prefix: Prefix for auth value (default: "Bearer")
            timeout: Request timeout in seconds
            **kwargs: Additional headers or configuration
        """
        # Validate base_url
        if not base_url or not base_url.strip():
            raise ValueError("base_url cannot be empty or whitespace-only")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        self.timeout = timeout

        # Default headers
        self.default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **kwargs.get("headers", {}),
        }

        # Add auth header if API key provided - fix spacing issue
        if self.api_key:
            if self.auth_prefix:
                self.default_headers[self.auth_header] = (
                    f"{self.auth_prefix} {self.api_key}"
                )
            else:
                self.default_headers[self.auth_header] = self.api_key

        self._session = None
        self._executor = _RestHttpExecutor(self)
        logger.debug(f"REST plugin configured for {self.base_url}")

    @property
    def is_connected(self) -> bool:
        """Whether an HTTP session is currently open."""
        return self._session is not None

    async def teardown(self) -> None:
        """Release runtime-owned HTTP resources."""
        await self.disconnect()

    def declare_capabilities(self) -> tuple[Capability, ...]:
        """Declare REST operations as runtime-plannable capabilities."""
        return tuple(
            Capability(
                id=definition["capability_id"],
                owner=self.manifest.id,
                description=definition["description"],
                domains=frozenset({"api", "http"}),
                operation_types=definition["operation_types"],
                access=definition["access"],
                risk=definition["risk"],
                input_schema=definition["parameters"],
                output_evidence=frozenset({"api.http.response"}),
                executor=self._executor.id,
                model_visible=True,
                retry_safe=definition["retry_safe"],
                idempotent=definition["idempotent"],
                side_effecting=definition["side_effecting"],
                timeout_seconds=60,
                metadata={
                    "method": definition["method"],
                    "tool_name": definition["name"],
                },
            )
            for definition in _REST_TOOL_DEFINITIONS
        )

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        """Declare the typed evidence returned by REST capability execution."""
        return (
            EvidenceSchema(
                kind="api.http.response",
                owner=self.manifest.id,
                description="HTTP response evidence from a REST API operation.",
                json_schema={
                    "type": "object",
                    "properties": {
                        "method": {"type": "string"},
                        "endpoint": {"type": "string"},
                        "response": {"type": "object"},
                    },
                    "required": ["method", "endpoint", "response"],
                },
            ),
        )

    def get_executors(self) -> tuple[_RestHttpExecutor, ...]:
        """Return the executor for REST runtime capabilities."""
        return (self._executor,)

    def get_tool_views(self) -> tuple[ToolView, ...]:
        """Expose REST capabilities as model-visible tool views."""
        return tuple(
            ToolView(
                name=definition["name"],
                capability_id=definition["capability_id"],
                description=definition["description"],
                parameters=definition["parameters"],
                metadata={"method": definition["method"]},
            )
            for definition in _REST_TOOL_DEFINITIONS
        )

    async def connect(self):
        """Initialize HTTP session."""
        if self._session is not None:
            return  # Already connected

        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=self.default_headers, timeout=timeout
            )

            logger.info(f"Connected to REST API: {self.base_url}")
        except ImportError:
            raise ImportError(
                "aiohttp not found. It is a core dependency — reinstall with: pip install daita-agents"
            )

    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("Disconnected from REST API")

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            headers: Additional headers for this request

        Returns:
            Response data as dictionary

        Example:
            data = await api.get("/users", params={"page": 1})
        """
        return await self._request("GET", endpoint, params=params, headers=headers)

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint
            data: Form data to send
            json_data: JSON data to send (takes precedence over data)
            headers: Additional headers

        Returns:
            Response data as dictionary

        Example:
            result = await api.post("/users", json_data={"name": "John", "email": "john@example.com"})
        """
        return await self._request(
            "POST", endpoint, data=data, json_data=json_data, headers=headers
        )

    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a PUT request.

        Args:
            endpoint: API endpoint
            data: Form data to send
            json_data: JSON data to send
            headers: Additional headers

        Returns:
            Response data as dictionary
        """
        return await self._request(
            "PUT", endpoint, data=data, json_data=json_data, headers=headers
        )

    async def patch(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a PATCH request.

        Args:
            endpoint: API endpoint
            data: Form data to send
            json_data: JSON data to send
            headers: Additional headers

        Returns:
            Response data as dictionary
        """
        return await self._request(
            "PATCH", endpoint, data=data, json_data=json_data, headers=headers
        )

    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a DELETE request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers

        Returns:
            Response data as dictionary
        """
        return await self._request("DELETE", endpoint, params=params, headers=headers)

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with error handling.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json_data: JSON data
            headers: Additional headers

        Returns:
            Response data
        """
        await self.connect()

        # Build full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Prepare request kwargs
        request_kwargs = {
            "params": params,
            "headers": headers,
        }

        # Handle request body
        if json_data is not None:
            request_kwargs["json"] = json_data
        elif data is not None:
            request_kwargs["data"] = data

        try:
            async with self._session.request(method, url, **request_kwargs) as response:

                # Log request
                logger.debug(f"{method} {url} -> {response.status}")

                # Handle different response types
                content_type = response.headers.get("content-type", "")

                # Check for errors
                if response.status >= 400:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")

                # Parse response based on content type
                _MAX_CHARS = 50_000
                if "application/json" in content_type:
                    import json as _json

                    raw = await response.text()
                    parsed = _json.loads(raw)  # parse full response first
                    if len(raw) > _MAX_CHARS:
                        return {
                            "data": raw[:_MAX_CHARS],
                            "content_type": content_type,
                            "truncated": True,
                            "total_chars": len(raw),
                        }
                    return {"data": parsed, "content_type": content_type}
                elif "text/" in content_type:
                    text_content = await response.text()
                    truncated = len(text_content) > _MAX_CHARS
                    result = {
                        "content": text_content[:_MAX_CHARS],
                        "content_type": content_type,
                    }
                    if truncated:
                        result["truncated"] = True
                        result["total_chars"] = len(text_content)
                    return result
                else:
                    # Binary content — return metadata only
                    binary_content = await response.read()
                    return {
                        "binary": True,
                        "content_type": content_type,
                        "size": len(binary_content),
                    }

        except Exception as e:
            logger.error(f"REST request failed: {method} {url} - {str(e)}")
            raise RuntimeError(f"REST request failed: {str(e)}")

    async def upload_file(
        self,
        endpoint: str,
        file_path: str,
        field_name: str = "file",
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file.

        Args:
            endpoint: API endpoint
            file_path: Path to file to upload
            field_name: Form field name for the file
            additional_data: Additional form data

        Returns:
            Response data
        """
        import os
        import aiohttp

        await self.connect()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Create form data
        data = aiohttp.FormData()

        # Add file
        with open(file_path, "rb") as f:
            data.add_field(field_name, f, filename=os.path.basename(file_path))

            # Add additional form fields
            if additional_data:
                for key, value in additional_data.items():
                    data.add_field(key, str(value))

            # Make request
            async with self._session.post(url, data=data) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"File upload failed ({response.status}): {error_text}"
                    )

                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    return await response.json()
                else:
                    return {"content": await response.text()}

    async def download_file(
        self, endpoint: str, save_path: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Download a file.

        Args:
            endpoint: API endpoint
            save_path: Where to save the file
            params: Query parameters

        Returns:
            Path to downloaded file
        """
        import os

        await self.connect()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with self._session.get(url, params=params) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise RuntimeError(
                    f"File download failed ({response.status}): {error_text}"
                )

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Write file
            with open(save_path, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)

            logger.info(f"Downloaded file to {save_path}")
            return save_path

    def _tool_handler(self, name: str):
        """Return the legacy tool handler for a REST tool name."""
        handlers = {
            "http_get": self._tool_get,
            "http_post": self._tool_post,
            "http_put": self._tool_put,
            "http_patch": self._tool_patch,
            "http_delete": self._tool_delete,
        }
        return handlers[name]

    async def _tool_get(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_get"""
        endpoint = args.get("endpoint")
        params = args.get("params")
        result = await self.get(endpoint, params=params)
        return {**result, "endpoint": endpoint}

    async def _tool_post(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_post"""
        endpoint = args.get("endpoint")
        data = args.get("data")
        result = await self.post(endpoint, json_data=data)
        return {**result, "endpoint": endpoint}

    async def _tool_put(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_put"""
        endpoint = args.get("endpoint")
        data = args.get("data")
        result = await self.put(endpoint, json_data=data)
        return {**result, "endpoint": endpoint}

    async def _tool_patch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_patch"""
        endpoint = args.get("endpoint")
        data = args.get("data")
        result = await self.patch(endpoint, json_data=data)
        return {**result, "endpoint": endpoint}

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_delete"""
        endpoint = args.get("endpoint")
        params = args.get("params")
        result = await self.delete(endpoint, params=params)
        return {**result, "endpoint": endpoint}

    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


def rest(**kwargs) -> RESTPlugin:
    """Create REST plugin with simplified interface."""
    return RESTPlugin(**kwargs)
