"""Map host HTTP routes onto Agent members through one thin decode-call-encode handler.

Domain reads are GET with one JSON wire value per query parameter; mutations
send one JSON object of wire values. Path parameters are plain text. Handlers
decode, call the Agent member and encode; behaviour stays in the Agent.
"""

from __future__ import annotations

import inspect
import json
import logging
import types
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route as HttpRoute

from ...agent import Agent
from ..codec import JsonValue, decode, encode
from ..protocol import (
    CLIENT_SIDE_ANNOTATIONS,
    CWD_HEADER,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    EventType,
    error_envelope,
)

if TYPE_CHECKING:
    from ..server import Host

_LOGGER = logging.getLogger("daita.host")
AGENT_PREFIX = "/v1/agents/{agent}"
# Raw secrets: redacted from error details and never logged.
SECRET_PARAMETERS = frozenset(
    {"password", "credential", "api_key", "subscription_credential"}
)
_STATUS = {
    "version_mismatch": 409,
    "conversation_busy": 409,
    "workspace_busy": 409,
    "already_resolved": 409,
    "agent_already_exists": 409,
    "agent_locked": 423,
    "shutting_down": 503,
    "agent_not_found": 404,
    "key_not_found": 404,
    "file_not_found": 404,
    "permission_denied": 403,
    "runtime_error": 500,
}

Handler = Callable[[Request], Awaitable[Response]]


@dataclass(frozen=True, slots=True)
class Route:
    """One domain route below /v1/agents/{agent} for one Agent member."""

    method: str
    path: str
    member: str
    # Publish work_changed after success: the work board may have changed.
    work: bool = False
    # Reopen the agent after success when idle, so saved changes take effect.
    reopen: bool = False
    # Supplies confirmation_handler when the client sends it as true.
    callback: Callable[[Host, str], object] | None = None


def http_routes(table: tuple[Route, ...]) -> list[HttpRoute]:
    return [
        HttpRoute(AGENT_PREFIX + route.path, _endpoint(route), methods=[route.method])
        for route in table
    ]


def guarded(handler: Handler) -> Handler:
    """Turn every failure into the one error envelope."""

    async def handle(request: Request) -> Response:
        try:
            return await handler(request)
        except Exception as error:
            return error_response(request, error)

    return handle


def respond(value: object, status: int = 200) -> Response:
    body = json.dumps(encode(value), separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("the host response exceeds its size bound")
    return Response(body, status, media_type="application/json")


def error_response(request: Request, error: BaseException) -> Response:
    secrets: tuple[str, ...] = getattr(request.state, "secrets", ())
    try:
        envelope = error_envelope(error)
    except TypeError:
        if not secrets:
            _LOGGER.exception("unexpected host error")
        envelope = error_envelope(RuntimeError("internal host error"))
    code = typing.cast(dict[str, Any], envelope["error"])["code"]
    if code == "runtime_error" and not secrets:
        _LOGGER.error("host request failed: %s", error)
    body = json.dumps(_redact(envelope, secrets)).encode("utf-8")
    return Response(body, _STATUS.get(code, 400), media_type="application/json")


async def read_body(request: Request) -> dict[str, JsonValue]:
    """Read one bounded JSON object body; an empty body is an empty object."""

    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise ValueError("the request body exceeds its size bound")
        chunks.append(chunk)
    if not chunks:
        return {}
    body = json.loads(b"".join(chunks))
    if type(body) is not dict:
        raise ValueError("the request body must be a JSON object")
    return body


def fields(
    body: dict[str, JsonValue], hints: dict[str, object], defaults: dict[str, object]
) -> dict[str, Any]:
    """Decode the named body fields of a host route; unknown fields are refused."""

    unknown = set(body) - set(hints)
    if unknown:
        raise ValueError(f"unknown request fields: {sorted(unknown)}")
    missing = set(hints) - set(body) - set(defaults)
    if missing:
        raise ValueError(f"missing request fields: {sorted(missing)}")
    return {
        name: decode(body[name], hint) if name in body else defaults[name]
        for name, hint in hints.items()
    }


def request_cwd(request: Request, value: str | None = None) -> Path | None:
    """Return the client's absolute working directory, from a field or the header."""

    raw = value if value is not None else request.headers.get(CWD_HEADER)
    if raw is None:
        return None
    cwd = Path(unquote(raw) if value is None else raw)
    if not cwd.is_absolute():
        raise ValueError("cwd must be an absolute path")
    return cwd


def host_of(request: Request) -> Host:
    return typing.cast("Host", request.app.state.host)


def _endpoint(route: Route) -> Handler:
    member = getattr(Agent, route.member)
    hints = typing.get_type_hints(member)
    client_side = CLIENT_SIDE_ANNOTATIONS.get(route.member, frozenset())
    wire = set(inspect.signature(member).parameters) - {"self"} - client_side
    callbacks = client_side if route.callback is not None else frozenset()

    async def handle(request: Request) -> Response:
        host = host_of(request)
        name = request.path_params["agent"]
        raw: dict[str, JsonValue] = (
            {key: json.loads(value) for key, value in request.query_params.items()}
            if request.method == "GET"
            else await read_body(request)
        )
        for key, value in request.path_params.items():
            if key != "agent":
                if key in raw:
                    raise ValueError(f"{key} is given twice")
                raw[key] = value
        request.state.secrets = tuple(
            value
            for key, value in raw.items()
            if key in SECRET_PARAMETERS and type(value) is str and value
        )
        unknown = set(raw) - wire - callbacks
        if unknown:
            raise ValueError(f"unknown arguments for {route.member}: {sorted(unknown)}")
        arguments = {
            key: decode(value, hints[key])
            for key, value in raw.items()
            if key not in callbacks
        }
        _resolve_paths(request, arguments, hints)
        for key in callbacks:
            if raw.get(key) is True:
                assert route.callback is not None
                arguments[key] = route.callback(host, name)
        agent = await host.registry.get(name)
        result = getattr(agent, route.member)(**arguments)
        if inspect.isawaitable(result):
            result = await result
        if route.work:
            host.bus.publish(name, EventType.WORK_CHANGED, {"member": route.member})
        if route.reopen:
            await host.registry.reopen(name)
        return respond(result)

    return guarded(handle)


def _resolve_paths(
    request: Request, arguments: dict[str, Any], hints: dict[str, object]
) -> None:
    # Relative client paths mean the client's folder, never the host's.
    for key, value in arguments.items():
        if not isinstance(value, (str, Path)) or not _admits_path(hints[key]):
            continue
        path = Path(value)
        if path.is_absolute() or str(value).startswith("~"):
            continue
        cwd = request_cwd(request)
        if cwd is None:
            raise ValueError(f"{key} is relative but the request has no {CWD_HEADER}")
        arguments[key] = cwd / path if isinstance(value, Path) else str(cwd / path)


def _admits_path(annotation: object) -> bool:
    if annotation is Path:
        return True
    return typing.get_origin(annotation) in (typing.Union, types.UnionType) and any(
        member is Path for member in typing.get_args(annotation)
    )


def _redact(value: object, secrets: tuple[str, ...]) -> object:
    if not secrets:
        return value
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    return value


__all__ = [
    "AGENT_PREFIX",
    "SECRET_PARAMETERS",
    "Route",
    "error_response",
    "fields",
    "guarded",
    "host_of",
    "http_routes",
    "read_body",
    "request_cwd",
    "respond",
]
