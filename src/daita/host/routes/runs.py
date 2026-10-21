"""Host-owned foreground runs (§7.4)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route as HttpRoute

from . import AGENT_PREFIX, fields, guarded, host_of, read_body, request_cwd, respond

MEMBERS = frozenset({"run", "learn"})
_SUBMIT = {
    "message": str,
    "conversation_id": str | None,
    "source_scope_ids": tuple[str, ...],
    "files_only": bool,
    "mode": str,
    "cwd": str | None,
}
_DEFAULTS: dict[str, object] = {
    "conversation_id": None,
    "source_scope_ids": (),
    "files_only": False,
    "mode": "run",
    "cwd": None,
}


async def _submit(request: Request) -> Response:
    host = host_of(request)
    name = request.path_params["agent"]
    values = fields(await read_body(request), _SUBMIT, _DEFAULTS)  # type: ignore[arg-type]
    agent = await host.registry.get(name, request_cwd(request, values.pop("cwd")))
    run = host.runs.submit(name, agent, values.pop("message"), **values)
    return respond(run.view(), 202)


async def _active(request: Request) -> Response:
    if request.query_params.get("active") != "true":
        raise ValueError("list runs with ?active=true")
    runs = host_of(request).runs.active(request.path_params["agent"])
    return respond(tuple(run.view() for run in runs))


async def _get(request: Request) -> Response:
    host = host_of(request)
    run = host.runs.get(request.path_params["agent"], request.path_params["run_ref"])
    return respond(run.view())


async def _cancel(request: Request) -> Response:
    host = host_of(request)
    run = host.runs.cancel(request.path_params["agent"], request.path_params["run_ref"])
    return respond(run.view())


ROUTES = [
    HttpRoute(AGENT_PREFIX + "/runs", guarded(_submit), methods=["POST"]),
    HttpRoute(AGENT_PREFIX + "/runs", guarded(_active), methods=["GET"]),
    HttpRoute(AGENT_PREFIX + "/runs/{run_ref}", guarded(_get), methods=["GET"]),
    HttpRoute(
        AGENT_PREFIX + "/runs/{run_ref}/cancel", guarded(_cancel), methods=["POST"]
    ),
]
