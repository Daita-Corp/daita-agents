"""Model configuration and subscription login (a model_login interaction)."""

from __future__ import annotations

import asyncio

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route as HttpRoute

from ...llm.subscription_auth import CodexDevicePrompt
from ..interactions import Interaction, InteractionKind
from ..protocol import InteractionResolvedError
from . import (
    AGENT_PREFIX,
    Route,
    fields,
    guarded,
    host_of,
    http_routes,
    read_body,
    respond,
)

TABLE = (
    Route("POST", "/model", "configure_model"),
    Route("GET", "/model/explicit-limits", "model_requires_explicit_limits"),
)
MEMBERS = frozenset(
    {*(route.member for route in TABLE), "authenticate_model_subscription"}
)


async def _subscription_login(request: Request) -> Response:
    host = host_of(request)
    name = request.path_params["agent"]
    provider = fields(await read_body(request), {"provider": str}, {})["provider"]
    agent = await host.registry.get(name)
    shown: list[Interaction] = []

    def on_verification(prompt: CodexDevicePrompt) -> None:
        shown.append(
            host.hub.show(
                name,
                InteractionKind.MODEL_LOGIN,
                {"prompt": prompt, "progress": None},
                owner=login,
            )
        )

    def on_progress(message: str) -> None:
        for interaction in shown:
            host.hub.update(interaction, {"progress": message})

    login = asyncio.ensure_future(
        agent.authenticate_model_subscription(
            provider=provider, on_verification=on_verification, on_progress=on_progress
        )
    )
    try:
        credential = await login
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if not login.cancelled() or (current is not None and current.cancelling()):
            login.cancel()
            raise
        raise InteractionResolvedError("the model login was dismissed") from None
    finally:
        for interaction in shown:
            host.hub.complete(interaction)
    request.state.secrets = (credential,)
    return respond(credential)


ROUTES = [
    *http_routes(TABLE),
    HttpRoute(
        AGENT_PREFIX + "/model/subscription-login",
        guarded(_subscription_login),
        methods=["POST"],
    ),
]
