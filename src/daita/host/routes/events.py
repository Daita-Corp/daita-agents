"""The per-agent Server-Sent Events stream (§7.5)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route as HttpRoute

from ..protocol import EventType, event_envelope
from . import AGENT_PREFIX, guarded, host_of

MEMBERS: frozenset[str] = frozenset()


async def _events(request: Request) -> Response:
    host = host_of(request)
    name = request.path_params["agent"]
    raw = request.query_params.get("after", request.headers.get("last-event-id"))
    after = None if raw is None else int(raw)
    if after is not None and after < 0:
        raise ValueError("after must be a non-negative sequence")
    await host.registry.get(name)

    async def content() -> AsyncIterator[ServerSentEvent]:
        async for seq, event, text in host.bus.subscribe(name, after):
            yield ServerSentEvent(data=text, event=event, id=str(seq))

    def heartbeat() -> ServerSentEvent:
        record = event_envelope(host.bus.head(name), EventType.HEARTBEAT, None)
        return ServerSentEvent(data=json.dumps(record), event=EventType.HEARTBEAT.value)

    return EventSourceResponse(
        content(), ping=host.heartbeat_seconds, ping_message_factory=heartbeat
    )


ROUTES = [HttpRoute(AGENT_PREFIX + "/events", guarded(_events), methods=["GET"])]
