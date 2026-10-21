"""Artifacts and the export destination."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (
    Route("GET", "/artifacts", "list_artifacts"),
    Route("GET", "/artifacts/{artifact_id}", "read_artifact"),
    Route("POST", "/artifacts/{artifact_id}/save", "save_artifact"),
    Route("DELETE", "/artifacts/{artifact_id}", "delete_artifact"),
    Route("GET", "/export-destination", "export_destination"),
    Route("PUT", "/export-destination", "set_export_destination"),
    Route("DELETE", "/export-destination", "reset_export_destination"),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
