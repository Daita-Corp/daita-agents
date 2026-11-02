"""Scheduled routines; create and update confirm through an interaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..interactions import InteractionKind, approval_handler
from . import Route, http_routes

if TYPE_CHECKING:
    from ..server import Host


def _confirm(host: Host, agent: str) -> object:
    return approval_handler(host.hub, agent, InteractionKind.ROUTINE_CONFIRMATION)


TABLE = (
    Route("GET", "/routines", "list_routines"),
    Route("POST", "/routines", "create_routine", work=True, callback=_confirm),
    Route("GET", "/routines/{routine_id}", "inspect_routine"),
    Route(
        "PUT", "/routines/{routine_id}", "update_routine", work=True, callback=_confirm
    ),
    Route("POST", "/routines/{routine_id}/pause", "pause_routine", work=True),
    Route("POST", "/routines/{routine_id}/resume", "resume_routine", work=True),
    Route("POST", "/routines/{routine_id}/disable", "disable_routine", work=True),
    Route("POST", "/routines/{routine_id}/run-now", "run_routine_now", work=True),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
