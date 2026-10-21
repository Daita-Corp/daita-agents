"""Catalog summary, resources and inspection."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (
    Route("GET", "/catalog", "catalog_summary"),
    Route("GET", "/catalog/resources", "list_catalog_resources"),
    Route("GET", "/catalog/resources/{resource_id}", "inspect_catalog_resource"),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
