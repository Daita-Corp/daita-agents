"""The work board."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (Route("GET", "/work", "work_board"),)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
