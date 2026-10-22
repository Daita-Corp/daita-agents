"""External-effect receipts and their human recovery."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (
    Route("GET", "/effects", "list_effects"),
    Route("GET", "/effects/{receipt_id}", "inspect_effect"),
    Route("POST", "/effects/{receipt_id}/resolve", "resolve_effect", work=True),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
