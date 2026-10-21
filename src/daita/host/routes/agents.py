"""Agent lifecycle and status (§7.3); the agent properties come from status."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route as HttpRoute

from ...agent import Agent
from . import AGENT_PREFIX, fields, guarded, host_of, read_body, request_cwd, respond

MEMBERS = frozenset(
    {
        "list",
        "create",
        "delete",
        "open",
        "close",
        "inspect_home",
        "name",
        "id",
        "home",
        "model_profile",
        "model_route",
    }
)


def status(agent: Agent) -> dict[str, object]:
    return {
        "id": agent.id,
        "name": agent.name,
        "home": agent.home,
        "model_profile": agent.model_profile,
        "model_route": agent.model_route,
        "workspace": agent.workspace,
        "session_source_id": agent.session_source_id,
        "background": agent.background_status(),
    }


async def _list(request: Request) -> Response:
    host = host_of(request)
    open_names = set(host.registry.open_names())
    names = await Agent.list(root=host.paths.root)
    return respond(tuple({"name": name, "open": name in open_names} for name in names))


async def _create(request: Request) -> Response:
    host = host_of(request)
    values = fields(
        await read_body(request), {"name": str, "cwd": str | None}, {"cwd": None}
    )
    cwd = request_cwd(request, values["cwd"])
    return respond(status(await host.registry.create(values["name"], cwd)), 201)


async def _open(request: Request) -> Response:
    host = host_of(request)
    values = fields(await read_body(request), {"cwd": str | None}, {"cwd": None})
    cwd = request_cwd(request, values["cwd"])
    return respond(status(await host.registry.get(request.path_params["agent"], cwd)))


async def _status(request: Request) -> Response:
    host = host_of(request)
    return respond(status(await host.registry.get(request.path_params["agent"])))


async def _close(request: Request) -> Response:
    await host_of(request).registry.close(request.path_params["agent"])
    return respond(None)


async def _delete(request: Request) -> Response:
    await host_of(request).registry.delete(request.path_params["agent"])
    return respond(None)


async def _home(request: Request) -> Response:
    root = host_of(request).paths.root
    return respond(await Agent.inspect_home(request.path_params["agent"], root=root))


ROUTES = [
    HttpRoute("/v1/agents", guarded(_list), methods=["GET"]),
    HttpRoute("/v1/agents", guarded(_create), methods=["POST"]),
    HttpRoute(AGENT_PREFIX, guarded(_status), methods=["GET"]),
    HttpRoute(AGENT_PREFIX, guarded(_delete), methods=["DELETE"]),
    HttpRoute(AGENT_PREFIX + "/open", guarded(_open), methods=["POST"]),
    HttpRoute(AGENT_PREFIX + "/close", guarded(_close), methods=["POST"]),
    HttpRoute(AGENT_PREFIX + "/home", guarded(_home), methods=["GET"]),
]
