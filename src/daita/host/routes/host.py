"""Host status and graceful shutdown (§7.2)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route as HttpRoute

from . import fields, guarded, host_of, read_body, respond

MEMBERS: frozenset[str] = frozenset()


async def _status(request: Request) -> Response:
    return respond(host_of(request).status())


async def _shutdown(request: Request) -> Response:
    reason = fields(await read_body(request), {"reason": str}, {"reason": "user"})
    if reason["reason"] not in ("upgrade", "user"):
        raise ValueError('shutdown reason must be "upgrade" or "user"')
    host_of(request).request_shutdown(reason["reason"])
    return respond({"state": "shutting_down"}, 202)


ROUTES = [
    HttpRoute("/v1/host", guarded(_status), methods=["GET"]),
    HttpRoute("/v1/host/shutdown", guarded(_shutdown), methods=["POST"]),
]
