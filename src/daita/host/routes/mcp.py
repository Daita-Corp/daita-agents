"""Remote MCP bindings and bearer credentials."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (
    Route("GET", "/mcp", "list_mcp_servers"),
    Route("POST", "/mcp", "attach_mcp_server"),
    Route("POST", "/mcp/inspect", "inspect_mcp_server"),
    Route("POST", "/mcp/bearers", "store_mcp_bearer"),
    Route("DELETE", "/mcp/bearers", "delete_mcp_bearer"),
    Route("POST", "/mcp/{binding_id}/refresh", "refresh_mcp_server"),
    Route("POST", "/mcp/{binding_id}/revoke", "revoke_mcp_server"),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
