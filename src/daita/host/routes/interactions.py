"""Answer or dismiss pending interactions (§7.6)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route as HttpRoute

from . import AGENT_PREFIX, guarded, host_of, read_body, respond

MEMBERS: frozenset[str] = frozenset()


async def _pending(request: Request) -> Response:
    if request.query_params.get("state", "pending") != "pending":
        raise ValueError("only pending interactions are kept")
    return respond(host_of(request).hub.pending(request.path_params["agent"]))


async def _answer(request: Request) -> Response:
    hub = host_of(request).hub
    params = request.path_params
    return respond(hub.answer(params["agent"], params["id"], await read_body(request)))


async def _dismiss(request: Request) -> Response:
    params = request.path_params
    return respond(host_of(request).hub.dismiss(params["agent"], params["id"]))


ROUTES = [
    HttpRoute(AGENT_PREFIX + "/interactions", guarded(_pending), methods=["GET"]),
    HttpRoute(AGENT_PREFIX + "/interactions/{id}", guarded(_answer), methods=["POST"]),
    HttpRoute(
        AGENT_PREFIX + "/interactions/{id}/dismiss", guarded(_dismiss), methods=["POST"]
    ),
]
