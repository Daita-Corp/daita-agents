"""Delivery inspection, acknowledgement and destinations."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (
    Route("GET", "/deliveries/{delivery_id}", "inspect_delivery"),
    Route(
        "POST",
        "/deliveries/{delivery_id}/acknowledge",
        "acknowledge_delivery",
        work=True,
    ),
    Route(
        "GET",
        "/conversations/{conversation_id}/destinations",
        "distribution_destinations",
    ),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
