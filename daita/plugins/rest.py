"""
REST API plugin for Daita Agents.

Simple REST API client - no over-engineering.
"""
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)

class RESTPlugin:
    """
    Simple REST API plugin for agents.
    
    Just makes HTTP requests and handles responses. Nothing fancy.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer",
        timeout: int = 30,
        **kwargs
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
        
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        self.timeout = timeout
        
        # Default headers
        self.default_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            **kwargs.get('headers', {})
        }
        
        # Add auth header if API key provided - fix spacing issue
        if self.api_key:
            if self.auth_prefix:
                self.default_headers[self.auth_header] = f"{self.auth_prefix} {self.api_key}"
            else:
                self.default_headers[self.auth_header] = self.api_key
        
        self._session = None
        logger.debug(f"REST plugin configured for {self.base_url}")
    
    async def connect(self):
        """Initialize HTTP session."""
        if self._session is not None:
            return  # Already connected
        
        try:
            import aiohttp
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=self.default_headers,
                timeout=timeout
            )
            
            logger.info(f"Connected to REST API: {self.base_url}")
        except ImportError:
            raise RuntimeError("aiohttp not installed. Run: pip install aiohttp")
    
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
        headers: Optional[Dict[str, str]] = None
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
        headers: Optional[Dict[str, str]] = None
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
        return await self._request("POST", endpoint, data=data, json_data=json_data, headers=headers)
    
    async def put(
        self, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
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
        return await self._request("PUT", endpoint, data=data, json_data=json_data, headers=headers)
    
    async def patch(
        self, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
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
        return await self._request("PATCH", endpoint, data=data, json_data=json_data, headers=headers)
    
    async def delete(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
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
        headers: Optional[Dict[str, str]] = None
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
            'params': params,
            'headers': headers,
        }
        
        # Handle request body
        if json_data is not None:
            request_kwargs['json'] = json_data
        elif data is not None:
            request_kwargs['data'] = data
        
        try:
            async with self._session.request(method, url, **request_kwargs) as response:
                
                # Log request
                logger.debug(f"{method} {url} -> {response.status}")
                
                # Handle different response types
                content_type = response.headers.get('content-type', '')
                
                # Check for errors
                if response.status >= 400:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")
                
                # Parse response based on content type
                if 'application/json' in content_type:
                    return await response.json()
                elif 'text/' in content_type:
                    text_content = await response.text()
                    return {'content': text_content, 'content_type': content_type}
                else:
                    # Binary content
                    binary_content = await response.read()
                    return {
                        'content': binary_content, 
                        'content_type': content_type,
                        'size': len(binary_content)
                    }
                    
        except Exception as e:
            logger.error(f"REST request failed: {method} {url} - {str(e)}")
            raise RuntimeError(f"REST request failed: {str(e)}")
    
    async def upload_file(
        self,
        endpoint: str,
        file_path: str,
        field_name: str = "file",
        additional_data: Optional[Dict[str, Any]] = None
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
        with open(file_path, 'rb') as f:
            data.add_field(
                field_name, 
                f, 
                filename=os.path.basename(file_path)
            )
            
            # Add additional form fields
            if additional_data:
                for key, value in additional_data.items():
                    data.add_field(key, str(value))
            
            # Make request
            async with self._session.post(url, data=data) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise RuntimeError(f"File upload failed ({response.status}): {error_text}")
                
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    return await response.json()
                else:
                    return {'content': await response.text()}
    
    async def download_file(
        self,
        endpoint: str,
        save_path: str,
        params: Optional[Dict[str, Any]] = None
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
                raise RuntimeError(f"File download failed ({response.status}): {error_text}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Write file
            with open(save_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded file to {save_path}")
            return save_path

    def get_tools(self) -> List['AgentTool']:
        """
        Expose REST API operations as agent tools.

        Returns:
            List of AgentTool instances for REST API operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="http_get",
                description="Make an HTTP GET request to the REST API endpoint. Use for retrieving data.",
                parameters={
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string",
                            "description": "API endpoint path (without base URL, e.g., /users or /data/123)"
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional query parameters as key-value pairs"
                        }
                    },
                    "required": ["endpoint"]
                },
                handler=self._tool_get,
                category="api",
                source="plugin",
                plugin_name="REST",
                timeout_seconds=60
            ),
            AgentTool(
                name="http_post",
                description="Make an HTTP POST request to the REST API endpoint. Use for creating new resources.",
                parameters={
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string",
                            "description": "API endpoint path (without base URL)"
                        },
                        "data": {
                            "type": "object",
                            "description": "JSON data to send in the request body"
                        }
                    },
                    "required": ["endpoint", "data"]
                },
                handler=self._tool_post,
                category="api",
                source="plugin",
                plugin_name="REST",
                timeout_seconds=60
            ),
            AgentTool(
                name="http_put",
                description="Make an HTTP PUT request to the REST API endpoint. Use for updating existing resources.",
                parameters={
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string",
                            "description": "API endpoint path (without base URL)"
                        },
                        "data": {
                            "type": "object",
                            "description": "JSON data to send in the request body"
                        }
                    },
                    "required": ["endpoint", "data"]
                },
                handler=self._tool_put,
                category="api",
                source="plugin",
                plugin_name="REST",
                timeout_seconds=60
            ),
            AgentTool(
                name="http_delete",
                description="Make an HTTP DELETE request to the REST API endpoint. Use for deleting resources.",
                parameters={
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string",
                            "description": "API endpoint path (without base URL)"
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional query parameters"
                        }
                    },
                    "required": ["endpoint"]
                },
                handler=self._tool_delete,
                category="api",
                source="plugin",
                plugin_name="REST",
                timeout_seconds=60
            )
        ]

    async def _tool_get(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_get"""
        endpoint = args.get("endpoint")
        params = args.get("params")

        result = await self.get(endpoint, params=params)

        return {
            "success": True,
            "data": result,
            "endpoint": endpoint
        }

    async def _tool_post(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_post"""
        endpoint = args.get("endpoint")
        data = args.get("data")

        result = await self.post(endpoint, json_data=data)

        return {
            "success": True,
            "data": result,
            "endpoint": endpoint
        }

    async def _tool_put(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_put"""
        endpoint = args.get("endpoint")
        data = args.get("data")

        result = await self.put(endpoint, json_data=data)

        return {
            "success": True,
            "data": result,
            "endpoint": endpoint
        }

    async def _tool_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for http_delete"""
        endpoint = args.get("endpoint")
        params = args.get("params")

        result = await self.delete(endpoint, params=params)

        return {
            "success": True,
            "data": result,
            "endpoint": endpoint
        }

    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


def rest(**kwargs) -> RESTPlugin:
    """Create REST plugin with simplified interface."""
    return RESTPlugin(**kwargs)