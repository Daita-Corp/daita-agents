"""Source attachment, edits, permissions and PostgreSQL credentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..interactions import source_edit_handler
from . import Route, http_routes

if TYPE_CHECKING:
    from ..server import Host


def _confirm_edit(host: Host, agent: str) -> object:
    return source_edit_handler(host.hub, agent)


TABLE = (
    Route("GET", "/sources", "list_sources"),
    Route("POST", "/sources", "attach"),
    Route("GET", "/sources/resolve", "resolve_source"),
    Route("POST", "/sources/sqlite", "attach_sqlite"),
    Route("POST", "/sources/postgresql", "attach_postgresql"),
    Route("POST", "/sources/postgresql/probe", "probe_postgresql"),
    Route("POST", "/sources/postgresql/passwords", "store_postgresql_password"),
    Route("DELETE", "/sources/postgresql/passwords", "delete_postgresql_password"),
    Route(
        "POST",
        "/sources/{source_id}/sqlite",
        "edit_sqlite_source",
        callback=_confirm_edit,
    ),
    Route(
        "POST",
        "/sources/{source_id}/postgresql",
        "edit_postgresql_source",
        callback=_confirm_edit,
    ),
    Route("DELETE", "/sources/{source_id}", "detach"),
    Route("POST", "/sources/{source_id}/refresh", "refresh_source"),
    Route("GET", "/sources/{source_id}/permissions", "inspect_source_permissions"),
    Route(
        "POST", "/sources/{source_id}/permissions/preview", "preview_source_permissions"
    ),
    Route("POST", "/sources/{source_id}/permissions/apply", "apply_source_permissions"),
    Route(
        "GET",
        "/sources/{source_id}/resources/{resource_id}/update-readiness",
        "relational_update_readiness",
    ),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
