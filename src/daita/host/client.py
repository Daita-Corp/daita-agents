"""Reach a host-owned agent over the Unix socket with the Agent's own methods.

``RemoteAgent`` mirrors the Agent members clients use (``REMOTE_AGENT_SURFACE``).
Each method encodes its arguments, calls the member's host route, and decodes
the result or re-raises the host's exception type. Callbacks never cross the
socket: they run here and answer the host interaction that stands in for them.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import json
import logging
import os
import re
import typing
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import quote

from .. import __version__
from .._installation import repair_guidance
from .._json import FrozenJsonObject
from ..adapters.mcp import (
    MCPAuthentication,
    MCPBindingStatus,
    MCPServerInspection,
    MCPToolSelection,
)
from ..adapters.models import SourceRegistration
from ..adapters.postgresql import PostgreSQLProbeResult
from ..adapters.postgresql_write import RelationalUpdateReadiness
from ..adapters.protocols import ResourceSource
from ..artifacts.models import (
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactPayload,
    ArtifactRef,
)
from ..capabilities import ApprovalDecision, ApprovalHandler
from ..catalog.models import CatalogResource, CatalogSummary
from ..distribution import DeliveryInspection, DeliveryView, DistributionDestination
from ..hosting.embedded import (
    EmbeddedAgent,
    SourceEditConfirmationHandler,
    SourceEditResult,
)
from ..hosting.home_upgrade import AgentHomeStatus
from ..jobs.graph.models import (
    GraphInspection,
    GraphJob,
    GraphMutation,
    GraphState,
    GraphTask,
    TaskAttempt,
    TaskCheckpoint,
    TaskControl,
    TaskDependency,
    TaskResult,
    TaskState,
)
from ..jobs.owner import GraphBlockerProjection
from ..jobs.projections import GraphBoardProjection, GraphTimelinePage
from ..llm.models import ModelProfile, ModelSensitivity, ToolCall, ToolResultBlock
from ..llm.routing import ModelRoute
from ..llm.subscription_auth import CodexDevicePrompt
from ..loop.models import ConversationRun, LoopExit, Transcript
from ..observation import AgentObserver
from ..routines import (
    RoutineState,
    ScheduledRoutine,
    ScheduledRoutineDraft,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
)
from ..security import SecretReference
from ..semantics import SemanticAnnotationState, SemanticAnnotationView, SemanticKind
from ..skills import Skill, SkillSummary
from ..storage.sqlite_records import (
    EffectReceipt,
    EffectResolutionDecision,
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    SourceReadMode,
)
from ..work_board import DEFAULT_DONE_WINDOW, WorkBoard
from ..workspace import LocalFileAccess, LocalWorkspace
from .codec import JsonValue, decode, encode
from .interactions import InteractionKind
from .paths import HostPaths, host_paths
from .protocol import (
    CLIENT_SIDE_ANNOTATIONS,
    CWD_HEADER,
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    VERSION_HEADER,
    EventType,
    HostVersionMismatch,
    InteractionResolvedError,
    error_from_envelope,
    parse_event,
)
from .runs import ACTIVE_STATES
from .supervise import HostStartError, host_status, start_host, stop_host

if TYPE_CHECKING:
    import httpx

    from .routes import Route

_LOGGER = logging.getLogger("daita.host.client")
BACKOFF_START_SECONDS = 0.2
BACKOFF_MAX_SECONDS = 5.0
ATTACH_TIMEOUT_SECONDS = 30.0
# Members whose callback the host replaces with one interaction of this kind.
_CONFIRMATIONS: typing.Final = {
    "edit_sqlite_source": InteractionKind.SOURCE_EDIT_CONFIRMATION,
    "edit_postgresql_source": InteractionKind.SOURCE_EDIT_CONFIRMATION,
    "create_routine": InteractionKind.ROUTINE_CONFIRMATION,
    "update_routine": InteractionKind.ROUTINE_CONFIRMATION,
}


class HostUnavailableError(RuntimeError):
    """The Daita host could not be reached or started."""


class HostClient:
    """One HTTP client over the host socket that starts or upgrades the host once."""

    def __init__(self, paths: HostPaths) -> None:
        try:
            import httpx
        except ImportError as error:
            raise ImportError(
                f"Daita's host client dependency is unavailable. {repair_guidance()}"
            ) from error
        self.paths = paths
        self._http = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=os.fspath(paths.socket)),
            # A fresh local connection per request, so a dropped or restarted
            # host never leaves a dead pooled connection for the next call.
            limits=httpx.Limits(max_keepalive_connections=0),
            base_url="http://daita-host",
            headers={
                PROTOCOL_HEADER: str(PROTOCOL_VERSION),
                VERSION_HEADER: __version__,
            },
            # Calls can wait on a person (confirmations, logins), so only connect
            # is bounded.
            timeout=httpx.Timeout(None, connect=5.0),
        )
        self._recovery = asyncio.Lock()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, JsonValue] | None = None,
        start: bool = True,
    ) -> JsonValue:
        """Send one request; start the host or replace a mismatched one, retry once."""

        import httpx

        for attempt in (1, 2) if start else (2,):
            try:
                response = await self._http.request(
                    method, path, params=params, json=body, headers=_cwd_header()
                )
            except httpx.ConnectError as error:
                if attempt == 2:
                    raise HostUnavailableError(
                        f"the Daita host does not answer on {self.paths.socket}"
                    ) from error
                await self.ensure_host()
                continue
            payload: JsonValue = response.json()
            if response.status_code < 400:
                return payload
            failure = error_from_envelope(payload)
            if isinstance(failure, HostVersionMismatch) and attempt == 1:
                await self.ensure_host()
                continue
            raise failure
        raise AssertionError("unreachable")

    @contextlib.asynccontextmanager
    async def lines(
        self, path: str, *, after: int
    ) -> AsyncIterator[AsyncIterator[str]]:
        """Open one event stream and yield its lines; errors raise the host's type."""

        async with self._http.stream(
            "GET", path, params={"after": str(after)}, headers=_cwd_header()
        ) as response:
            if response.status_code >= 400:
                raise error_from_envelope(json.loads(await response.aread()))
            yield response.aiter_lines()

    async def ensure_host(self) -> None:
        """Start the host, first draining one that runs another version (§10.5)."""

        async with self._recovery:
            await asyncio.to_thread(_ensure_host, self.paths)

    async def aclose(self) -> None:
        await self._http.aclose()


def _ensure_host(paths: HostPaths) -> None:
    # Idempotent, so a second caller after a recovery finds the host ready.
    status = host_status(paths)
    if status is not None and status.get("version_mismatch"):
        stop_host(paths, reason="upgrade")
    try:
        start_host(paths)
    except HostStartError as error:
        raise HostUnavailableError(str(error)) from error


def _folder(cwd: str | Path | None) -> str:
    # Pinned once, so a later chdir never moves the agent's folder.
    return os.fspath(Path(os.getcwd() if cwd is None else cwd).absolute())


def _cwd_header() -> dict[str, str]:
    return {CWD_HEADER: quote(os.getcwd())}


def _segment(value: object) -> str:
    # Dots too: a bare "." or ".." segment would be resolved away in transit.
    return quote(str(value), safe="").replace(".", "%2E")


@functools.cache
def _routes() -> dict[str, Route]:
    """The host's own route tables, so a member never reaches another route."""

    from .routes import (
        artifacts,
        catalog,
        conversations,
        deliveries,
        effects,
        jobs,
        mcp,
        memory,
        model,
        routines,
        sources,
        work,
    )

    areas = (
        artifacts,
        catalog,
        conversations,
        deliveries,
        effects,
        jobs,
        mcp,
        memory,
        model,
        routines,
        sources,
        work,
    )
    return {route.member: route for area in areas for route in area.TABLE}


@functools.cache
def _returns(member: str) -> object:
    return typing.get_type_hints(getattr(RemoteAgent, member))["return"]


async def _once(root: str | Path | None, method: str, path: str, **kwargs: Any) -> Any:
    client = HostClient(host_paths(root))
    try:
        return decode(await client.request(method, path, **kwargs))
    finally:
        await client.aclose()


@dataclass(eq=False, slots=True)
class _Claim:
    """One in-flight call waiting for the host interaction that replaces its callback."""

    kind: InteractionKind
    respond: Callable[[Mapping[str, Any]], Awaitable[dict[str, object] | None]]
    progress: Callable[[str], None] | None = None
    interaction: str | None = None
    # The calling task's context: its callback runs there, as it would in process.
    context: contextvars.Context = field(default_factory=contextvars.copy_context)


async def connect(
    name: str,
    *,
    cwd: str | Path | None = None,
    root: str | Path | None = None,
    observer: AgentObserver | None = None,
    approval_handler: ApprovalHandler | None = None,
) -> RemoteAgent:
    """Attach to one agent in the Daita host, starting the host if none runs.

    The host opens the agent with computer access to ``cwd``, as the TUI does in
    the folder it starts from; ``cwd`` defaults to this process's folder.
    """

    folder = _folder(cwd)
    client = HostClient(host_paths(root))
    try:
        status = await client.request(
            "POST", f"/v1/agents/{_segment(name)}/open", body={"cwd": folder}
        )
        return await RemoteAgent._attach(
            client, decode(status), folder, observer, approval_handler
        )
    except BaseException:
        await client.aclose()
        raise


class RemoteAgent:
    """One agent the host owns, reached with the same calls as an Agent.

    Set ``on_reset`` to learn when the host could not replay missed events;
    pending interactions and run waits recover by themselves, while views of
    transcripts and the work board must be reloaded. The members in
    ``CLIENT_ONLY_MEMBERS`` report what only a host client knows: the host's
    active runs and this client's connection to it.
    """

    def __init__(
        self,
        client: HostClient,
        status: Mapping[str, Any],
        cwd: str,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> None:
        self._client = client
        self._status = dict(status)
        self._cwd = cwd
        self._observer = observer
        self._approval_handler = approval_handler
        self._prefix = f"/v1/agents/{_segment(status['name'])}"
        # Zero predates every host sequence, so the first record is a reset.
        self._seq = 0
        self._attached: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._stream: asyncio.Task[None] | None = None
        self._handlers: dict[str, asyncio.Task[None]] = {}
        self._claims: list[_Claim] = []
        self._runs: dict[str, asyncio.Event] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._connection = "connected"
        self._closed = False
        self.on_reset: Callable[[], None] | None = None

    @classmethod
    async def _attach(
        cls,
        client: HostClient,
        status: Mapping[str, Any],
        cwd: str,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> Self:
        agent = cls(client, status, cwd, observer, approval_handler)
        agent._stream = asyncio.create_task(agent._follow())
        try:
            await asyncio.wait_for(
                asyncio.shield(agent._attached), ATTACH_TIMEOUT_SECONDS
            )
        except BaseException as error:
            await agent.close()
            if isinstance(error, TimeoutError):
                raise HostUnavailableError(
                    "the Daita host event stream did not attach"
                ) from None
            raise
        return agent

    # Lifecycle and properties

    @classmethod
    async def open(
        cls,
        name: str,
        *,
        workspace: LocalWorkspace | None = None,
        root: str | Path | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> Self:
        return typing.cast(
            Self,
            await connect(
                name,
                cwd=_host_folder(workspace),
                root=root,
                observer=observer,
                approval_handler=approval_handler,
            ),
        )

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        workspace: LocalWorkspace | None = None,
        root: str | Path | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> Self:
        folder = _folder(_host_folder(workspace))
        client = HostClient(host_paths(root))
        try:
            status = await client.request(
                "POST", "/v1/agents", body={"name": name, "cwd": folder}
            )
            return await cls._attach(
                client, decode(status), folder, observer, approval_handler
            )
        except BaseException:
            await client.aclose()
            raise

    @classmethod
    async def delete(cls, name: str, *, root: str | Path | None = None) -> None:
        await _once(root, "DELETE", f"/v1/agents/{_segment(name)}")

    @classmethod
    async def list(cls, *, root: str | Path | None = None) -> tuple[str, ...]:
        agents = await _once(root, "GET", "/v1/agents")
        return tuple(agent["name"] for agent in agents)

    @classmethod
    async def inspect_home(
        cls, name: str, *, root: str | Path | None = None
    ) -> AgentHomeStatus:
        path = f"/v1/agents/{_segment(name)}/home"
        return typing.cast(AgentHomeStatus, await _once(root, "GET", path))

    async def close(self) -> None:
        """Detach this client; the agent, its runs and schedulers stay in the host.

        A run wait cancelled after close detaches too: it never cancels the run.
        """

        self._closed = True
        tasks = [*self._handlers.values(), *self._tasks]
        if self._stream is not None:
            tasks.append(self._stream)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._client.aclose()

    @property
    def id(self) -> str:
        return typing.cast(str, self._status["id"])

    @property
    def name(self) -> str:
        return typing.cast(str, self._status["name"])

    @property
    def home(self) -> Path:
        return typing.cast(Path, self._status["home"])

    @property
    def workspace(self) -> LocalWorkspace:
        return typing.cast(LocalWorkspace, self._status["workspace"])

    @property
    def session_source_id(self) -> str | None:
        return typing.cast("str | None", self._status["session_source_id"])

    @property
    def model_profile(self) -> ModelProfile | None:
        return typing.cast("ModelProfile | None", self._status["model_profile"])

    @property
    def model_route(self) -> ModelRoute | None:
        return typing.cast("ModelRoute | None", self._status["model_route"])

    @property
    def model_configuration_requires_replacement(self) -> bool:
        """The saved model settings could not be admitted, so the agent has no route."""

        return typing.cast(
            bool, self._status["model_configuration_requires_replacement"]
        )

    @property
    def connection(self) -> str:
        """This client's event stream: connected, reconnecting or stopped."""

        return self._connection

    def model_requires_explicit_limits(self, *, provider: str, model: str) -> bool:
        # Static package data, identical in the host because versions match.
        return EmbeddedAgent.model_requires_explicit_limits(
            provider=provider, model=model
        )

    # Model

    async def authenticate_model_subscription(
        self,
        *,
        provider: str,
        on_verification: Callable[[CodexDevicePrompt], None],
        on_progress: Callable[[str], None] | None = None,
    ) -> str:
        async def show(view: Mapping[str, Any]) -> None:
            on_verification(view["request"]["prompt"])
            progress = view["request"]["progress"]
            if progress is not None and on_progress is not None:
                on_progress(progress)

        login = self._request(
            "POST", "/model/subscription-login", body={"provider": provider}
        )
        claim = _Claim(InteractionKind.MODEL_LOGIN, show, on_progress)
        return typing.cast(str, decode(await self._claimed(claim, login), str))

    async def configure_model(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        subscription_credential: str | None = None,
        base_url: str | None = None,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelRoute:
        route = await self._call("configure_model", locals())
        await self._refresh()
        return typing.cast(ModelRoute, route)

    # Runs

    async def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_scope_ids: tuple[str, ...] = (),
        files_only: bool = False,
        caller_principal_id: str | None = None,
    ) -> LoopExit:
        return await self._submit("run", locals())

    async def learn(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_scope_ids: tuple[str, ...] = (),
        caller_principal_id: str | None = None,
    ) -> LoopExit:
        return await self._submit("learn", locals())

    async def _submit(self, mode: str, arguments: dict[str, object]) -> LoopExit:
        body = {key: encode(value) for key, value in arguments.items() if key != "self"}
        if body["caller_principal_id"] is None:
            # Local runs have no caller principal; the host refuses one.
            del body["caller_principal_id"]
        body.update(mode=mode, cwd=self._cwd)
        submit = asyncio.ensure_future(self._request("POST", "/runs", body=body))
        try:
            ref = decode(await asyncio.shield(submit))["run_ref"]
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self._cancel(decode(await submit)["run_ref"])
            raise
        return await self.follow_run(ref)

    async def active_runs(self) -> tuple[dict[str, Any], ...]:
        """The agent's queued, running and waiting foreground runs, oldest first.

        Each is the host's run view: run_ref, run_id, conversation_id, mode,
        state, exit and error.
        """

        return typing.cast(
            "tuple[dict[str, Any], ...]",
            decode(await self._request("GET", "/runs", params={"active": "true"})),
        )

    async def follow_run(self, run_ref: str) -> LoopExit:
        """Wait for one host run, as run() does; cancelling the wait cancels it."""

        try:
            view = await self._finished(run_ref)
        except asyncio.CancelledError:
            # Ctrl-C semantics: cancelling the wait cancels the host run.
            with contextlib.suppress(Exception):
                await self._cancel(run_ref)
            raise
        if view["exit"] is not None:
            return typing.cast(LoopExit, view["exit"])
        if view["error"] is not None:
            raise error_from_envelope(view["error"].to_dict())
        # Cancelled, or interrupted by an expired or dismissed question.
        raise asyncio.CancelledError

    async def _cancel(self, ref: str) -> None:
        if not self._closed:
            await self._request("POST", f"/runs/{_segment(ref)}/cancel", start=False)

    async def _finished(self, ref: str) -> dict[str, Any]:
        import httpx

        wake = self._runs[ref] = asyncio.Event()
        try:
            while True:
                wake.clear()
                # A new host never knows this run, so a vanished host is final.
                path = f"/runs/{_segment(ref)}"
                try:
                    view = decode(await self._request("GET", path, start=False))
                except httpx.TransportError:
                    # Dropped mid-request; the stream's replay or failure wakes us.
                    await wake.wait()
                    continue
                if view["state"] not in ACTIVE_STATES:
                    return typing.cast(dict[str, Any], view)
                await wake.wait()
        finally:
            del self._runs[ref]

    def _wake(self, ref: str | None = None) -> None:
        for key, wake in self._runs.items():
            if ref is None or key == ref:
                wake.set()

    # Transport

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, JsonValue] | None = None,
        start: bool = True,
    ) -> JsonValue:
        return await self._client.request(
            method, self._prefix + path, params=params, body=body, start=start
        )

    async def _call(self, member: str, arguments: dict[str, object]) -> Any:
        route = _routes()[member]
        values = {key: value for key, value in arguments.items() if key != "self"}
        path = route.path.format_map(
            {
                key: _segment(values.pop(key))
                for key in re.findall(r"{(\w+)}", route.path)
            }
        )
        claim = None
        for key in CLIENT_SIDE_ANNOTATIONS.get(member, frozenset()) & set(values):
            handler = values.pop(key)
            if handler is not None:
                values[key] = True
                claim = self._confirmation(_CONFIRMATIONS[member], handler)
        wire = {key: encode(value) for key, value in values.items()}
        if route.method == "GET":
            call = self._request(
                "GET",
                path,
                params={key: json.dumps(item) for key, item in wire.items()},
            )
        else:
            call = self._request(route.method, path, body=wire)
        payload = await (call if claim is None else self._claimed(claim, call))
        return decode(payload, _returns(member))

    async def _refresh(self) -> None:
        self._status = decode(await self._client.request("GET", self._prefix))

    # Callbacks answered through host interactions (§12.1)

    def _confirmation(self, kind: InteractionKind, handler: Any) -> _Claim:
        async def respond(view: Mapping[str, Any]) -> dict[str, object]:
            return await _answer(kind, handler, view["request"])

        return _Claim(kind, respond)

    async def _claimed(self, claim: _Claim, call: Awaitable[JsonValue]) -> JsonValue:
        # shortcut: a claim takes the first unclaimed interaction of its kind
        # opened while its call is in flight, so two clients confirming the same
        # kind at once could swap prompts; carry a request id if that matters.
        self._claims.append(claim)
        try:
            return await call
        except asyncio.CancelledError:
            if claim.interaction is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(self._settle(claim.interaction, None))
            raise
        finally:
            self._claims.remove(claim)

    def _interaction(self, op: str, view: Mapping[str, Any]) -> None:
        interaction_id = view["id"]
        if op in ("resolved", "expired"):
            task = self._handlers.pop(interaction_id, None)
            if task is not None:
                task.cancel()
            return
        if op == "updated":
            progress = view["request"].get("progress")
            for waiting in self._claims:
                if waiting.interaction == interaction_id and waiting.progress:
                    if progress is not None:
                        _guard(waiting.progress, progress)
            return
        if interaction_id in self._handlers:
            return
        kind = InteractionKind(view["kind"])
        respond: Callable[[Mapping[str, Any]], Awaitable[dict[str, object] | None]]
        context = None
        if kind is InteractionKind.TOOL_APPROVAL:
            if self._approval_handler is None:
                return
            respond = self._confirmation(kind, self._approval_handler).respond
        else:
            claim = next(
                (
                    item
                    for item in self._claims
                    if item.kind is kind and item.interaction is None
                ),
                None,
            )
            if claim is None:
                return
            claim.interaction = interaction_id
            respond, context = claim.respond, claim.context
        self._handlers[interaction_id] = asyncio.create_task(
            self._respond(view, respond), context=context
        )

    async def _respond(
        self,
        view: Mapping[str, Any],
        respond: Callable[[Mapping[str, Any]], Awaitable[dict[str, object] | None]],
    ) -> None:
        try:
            answer = await respond(view)
        except (Exception, asyncio.CancelledError) as error:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise  # answered elsewhere, or this client is closing
            # A callback that cannot review interrupts the question, never denies.
            _LOGGER.info("interaction callback did not answer: %r", error)
            await self._settle(view["id"], None)
            return
        if answer is not None:
            await self._settle(view["id"], answer)

    async def _settle(
        self, interaction_id: str, answer: dict[str, object] | None
    ) -> None:
        """Post an answer, or dismiss when there is none; a later answer loses."""

        path = f"/interactions/{_segment(interaction_id)}"
        try:
            if answer is None:
                await self._request("POST", path + "/dismiss")
            else:
                await self._request("POST", path, body=answer)  # type: ignore[arg-type]
        except InteractionResolvedError:
            pass  # another client answered first
        except Exception as error:
            _LOGGER.warning("could not settle interaction: %s", type(error).__name__)

    # The event stream (§7.5)

    async def _follow(self) -> None:
        delay = BACKOFF_START_SECONDS
        while True:
            try:
                async with self._client.lines(
                    self._prefix + "/events", after=self._seq
                ) as lines:
                    delay = BACKOFF_START_SECONDS
                    self._connection = "connected"
                    async for line in lines:
                        if line.startswith("data:"):
                            await self._dispatch(*parse_event(json.loads(line[5:])))
                self._connection = "reconnecting"
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _LOGGER.debug("host event stream dropped: %s", type(error).__name__)
                # Nothing listens on the socket: the host is gone, not restarting.
                self._connection = "stopped" if _refused(error) else "reconnecting"
            # Waiting runs re-read their state, or learn that the host is gone.
            self._wake()
            await asyncio.sleep(delay)
            delay = min(delay * 2, BACKOFF_MAX_SECONDS)

    async def _dispatch(self, seq: int, event: EventType, data: Any) -> None:
        if event is EventType.HEARTBEAT:
            return
        if event is EventType.RESET:
            await self._resync()
            self._seq = seq  # only now, so a failed resync resets again
            if not self._attached.done():
                self._attached.set_result(None)
            elif self.on_reset is not None:
                _guard(self.on_reset)
            return
        if seq <= self._seq:
            return  # replayed after a reconnect
        self._seq = seq
        if event is EventType.AGENT_EVENT:
            if self._observer is not None:
                _guard(self._observer, data)
        elif event is EventType.RUN_STATE:
            self._wake(data["run_ref"])
        elif event is EventType.INTERACTION:
            self._interaction(data["op"], data["interaction"])
        elif event is EventType.AGENT_STATE:
            task = asyncio.create_task(self._refresh())
            self._tasks.add(task)
            task.add_done_callback(self._refreshed)

    def _refreshed(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            _LOGGER.warning("could not refresh the agent status from the host")

    async def _resync(self) -> None:
        """Re-read what missed events would have said: questions, runs, status."""

        pending = decode(await self._request("GET", "/interactions"))
        current = {view["id"] for view in pending}
        for interaction_id in [key for key in self._handlers if key not in current]:
            self._handlers.pop(interaction_id).cancel()
        for view in pending:
            self._interaction("opened", view)
        await self._refresh()
        self._wake()

    # Agent members reached through their host route

    async def transcript(
        self, run_id: str, *, caller_principal_id: str | None = None
    ) -> Transcript:
        return await self._call("transcript", locals())

    async def tool_evidence(
        self, run_id: str, *, caller_principal_id: str | None = None
    ) -> tuple[tuple[ToolCall, ToolResultBlock | None], ...]:
        return await self._call("tool_evidence", locals())

    async def distribution_destinations(
        self,
        conversation_id: str,
        *,
        sensitivity_ceiling: ModelSensitivity = ModelSensitivity.RESTRICTED,
        caller_principal_id: str | None = None,
    ) -> tuple[DistributionDestination, ...]:
        return await self._call("distribution_destinations", locals())

    async def inspect_delivery(
        self,
        delivery_id: str,
        *,
        caller_principal_id: str | None = None,
    ) -> DeliveryInspection | None:
        return await self._call("inspect_delivery", locals())

    async def acknowledge_delivery(
        self, delivery_id: str, *, caller_principal_id: str | None = None
    ) -> DeliveryView | None:
        return await self._call("acknowledge_delivery", locals())

    async def work_board(
        self,
        *,
        done_window: timedelta = DEFAULT_DONE_WINDOW,
        caller_principal_id: str | None = None,
    ) -> WorkBoard:
        return await self._call("work_board", locals())

    async def conversation_runs(
        self,
        conversation_id: str,
        *,
        caller_principal_id: str | None = None,
    ) -> tuple[ConversationRun, ...]:
        return await self._call("conversation_runs", locals())

    async def conversation_exists(
        self, conversation_id: str, *, caller_principal_id: str | None = None
    ) -> bool:
        return await self._call("conversation_exists", locals())

    async def list_jobs(
        self,
        *,
        states: frozenset[GraphState] = frozenset(),
        limit: int = 50,
    ) -> tuple[GraphJob, ...]:
        return await self._call("list_jobs", locals())

    async def inspect_effect(
        self, receipt_id: str, *, caller_principal_id: str | None = None
    ) -> EffectReceipt | None:
        return await self._call("inspect_effect", locals())

    async def list_effects(
        self,
        *,
        unresolved_only: bool = False,
        limit: int = 20,
        offset: int = 0,
        caller_principal_id: str | None = None,
    ) -> tuple[EffectReceipt, ...]:
        return await self._call("list_effects", locals())

    async def resolve_effect(
        self,
        receipt_id: str,
        *,
        expected_digest: str,
        decision: EffectResolutionDecision,
        note: str,
        evidence_references: tuple[str, ...] = (),
        caller_principal_id: str | None = None,
    ) -> EffectReceipt:
        return await self._call("resolve_effect", locals())

    async def inspect_job(self, job_id: str) -> GraphInspection | None:
        return await self._call("inspect_job", locals())

    async def list_job_tasks(
        self,
        job_id: str,
        *,
        states: frozenset[TaskState] = frozenset(),
        limit: int = 64,
    ) -> tuple[GraphTask, ...]:
        return await self._call("list_job_tasks", locals())

    async def inspect_job_task(self, job_id: str, task_id: str) -> GraphTask | None:
        return await self._call("inspect_job_task", locals())

    async def list_job_dependencies(
        self, job_id: str, *, task_id: str | None = None, limit: int = 100
    ) -> tuple[TaskDependency, ...]:
        return await self._call("list_job_dependencies", locals())

    async def list_task_attempts(
        self, job_id: str, task_id: str, *, limit: int = 3
    ) -> tuple[TaskAttempt, ...]:
        return await self._call("list_task_attempts", locals())

    async def analysis_usage(
        self, run_id: str, *, caller_principal_id: str | None = None
    ) -> dict[str, object] | None:
        return await self._call("analysis_usage", locals())

    async def read_task_result(self, job_id: str, task_id: str) -> TaskResult | None:
        return await self._call("read_task_result", locals())

    async def list_task_checkpoints(
        self, job_id: str, task_id: str, *, limit: int = 8
    ) -> tuple[TaskCheckpoint, ...]:
        return await self._call("list_task_checkpoints", locals())

    async def list_task_artifacts(
        self, job_id: str, *, task_id: str | None = None, limit: int = 64
    ) -> tuple[str, ...]:
        return await self._call("list_task_artifacts", locals())

    async def job_timeline(
        self,
        job_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 100,
        task_id: str | None = None,
    ) -> GraphTimelinePage | None:
        return await self._call("job_timeline", locals())

    async def job_board(self, job_id: str) -> GraphBoardProjection | None:
        return await self._call("job_board", locals())

    async def read_job_result(self, job_id: str) -> TaskResult | None:
        return await self._call("read_job_result", locals())

    async def cancel_job(self, job_id: str) -> GraphJob | None:
        return await self._call("cancel_job", locals())

    async def graph_blockers(self, job_id: str) -> GraphBlockerProjection | None:
        return await self._call("graph_blockers", locals())

    async def answer_task_input(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        answer: Mapping[str, object],
        idempotency_key: str | None = None,
    ) -> TaskControl | None:
        return await self._call("answer_task_input", locals())

    async def accept_task_review(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        rationale: str,
        idempotency_key: str,
    ) -> TaskResult:
        return await self._call("accept_task_review", locals())

    async def request_task_review_changes(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        rationale: str,
        replacement_guidance: str,
        idempotency_key: str,
    ) -> TaskControl:
        return await self._call("request_task_review_changes", locals())

    async def retry_task_control(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        advisory_note: str,
        idempotency_key: str,
    ) -> TaskControl | None:
        return await self._call("retry_task_control", locals())

    async def reject_task_control(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> TaskControl | None:
        return await self._call("reject_task_control", locals())

    async def cancel_graph_job(
        self, job_id: str, *, principal_id: str
    ) -> GraphJob | None:
        return await self._call("cancel_graph_job", locals())

    async def replace_graph_task_by_policy(
        self,
        job_id: str,
        task_id: str,
        *,
        principal_id: str,
        advisory_note: str,
        idempotency_key: str,
        expected_revision: int,
    ) -> GraphMutation:
        return await self._call("replace_graph_task_by_policy", locals())

    async def propose_routine(self, draft: ScheduledRoutineDraft) -> ScheduledRoutine:
        return await self._call("propose_routine", locals())

    async def promote_routine(
        self,
        draft: ScheduledRoutineDraft,
        *,
        basis_run_id: str,
    ) -> ScheduledRoutine:
        return await self._call("promote_routine", locals())

    async def create_routine(
        self,
        proposal: ScheduledRoutine,
        *,
        confirmation_handler: ApprovalHandler | None = None,
    ) -> ScheduledRoutine:
        return await self._call("create_routine", locals())

    async def list_routines(
        self,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = 50,
    ) -> tuple[ScheduledRoutineSummary, ...]:
        return await self._call("list_routines", locals())

    async def inspect_routine(
        self, routine_id: str
    ) -> ScheduledRoutineInspection | None:
        return await self._call("inspect_routine", locals())

    async def update_routine(
        self,
        routine_id: str,
        *,
        expected_revision: int,
        draft: ScheduledRoutineDraft,
        basis_run_id: str | None = None,
        confirmation_handler: ApprovalHandler | None = None,
    ) -> ScheduledRoutine:
        return await self._call("update_routine", locals())

    async def pause_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._call("pause_routine", locals())

    async def resume_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._call("resume_routine", locals())

    async def run_routine_now(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._call("run_routine_now", locals())

    async def disable_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._call("disable_routine", locals())

    async def clear_conversations(self) -> int:
        return await self._call("clear_conversations", locals())

    async def list_artifacts(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        caller_principal_id: str | None = None,
        newest_first: bool = False,
    ) -> tuple[ArtifactRef, ...]:
        return await self._call("list_artifacts", locals())

    async def read_artifact(
        self, artifact_id: str, *, caller_principal_id: str | None = None
    ) -> ArtifactPayload:
        return await self._call("read_artifact", locals())

    async def delete_artifact(
        self, artifact_id: str, *, caller_principal_id: str | None = None
    ) -> bool:
        return await self._call("delete_artifact", locals())

    async def save_artifact(
        self,
        artifact_id: str,
        destination: Path | None = None,
        *,
        filename: str | None = None,
        caller_principal_id: str | None = None,
    ) -> ArtifactDeliveryReceipt:
        return await self._call("save_artifact", locals())

    async def export_destination(self) -> ArtifactDestination:
        return await self._call("export_destination", locals())

    async def set_export_destination(self, directory: Path) -> ArtifactDestination:
        return await self._call("set_export_destination", locals())

    async def reset_export_destination(self) -> ArtifactDestination:
        return await self._call("reset_export_destination", locals())

    async def resolve_source(self, selector: str) -> SourceRegistration:
        return await self._call("resolve_source", locals())

    async def read_memory(self) -> str:
        return await self._call("read_memory", locals())

    async def set_memory(
        self, text: str, *, sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED
    ) -> None:
        return await self._call("set_memory", locals())

    async def read_user_profile(self) -> str:
        return await self._call("read_user_profile", locals())

    async def set_user_profile(
        self, text: str, *, sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED
    ) -> None:
        return await self._call("set_user_profile", locals())

    async def list_semantic_annotations(
        self,
        *,
        source_id: str | None = None,
        resource_id: str | None = None,
        kind: SemanticKind | None = None,
        state: SemanticAnnotationState | None = None,
    ) -> tuple[SemanticAnnotationView, ...]:
        return await self._call("list_semantic_annotations", locals())

    async def read_semantic_annotation(
        self,
        annotation_id: str,
    ) -> SemanticAnnotationView | None:
        return await self._call("read_semantic_annotation", locals())

    async def delete_semantic_annotation(
        self,
        annotation_id: str,
        *,
        expected_sha256: str,
    ) -> bool:
        return await self._call("delete_semantic_annotation", locals())

    async def list_skills(self) -> tuple[SkillSummary, ...]:
        return await self._call("list_skills", locals())

    async def read_skill(self, name: str) -> Skill | None:
        return await self._call("read_skill", locals())

    async def save_skill(
        self,
        name: str,
        description: str,
        instructions: str,
        *,
        sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED,
    ) -> bool:
        return await self._call("save_skill", locals())

    async def delete_skill(self, name: str) -> bool:
        return await self._call("delete_skill", locals())

    async def store_mcp_bearer(self, credential: str) -> SecretReference:
        return await self._call("store_mcp_bearer", locals())

    async def delete_mcp_bearer(self, reference: SecretReference) -> None:
        return await self._call("delete_mcp_bearer", locals())

    async def inspect_mcp_server(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication | None = None,
        caller_principal_id: str | None = None,
    ) -> MCPServerInspection:
        return await self._call("inspect_mcp_server", locals())

    async def attach_mcp_server(
        self,
        *,
        endpoint: str,
        selections: tuple[MCPToolSelection, ...],
        authentication: MCPAuthentication | None = None,
        maximum_outbound_sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
        local_label: str | None = None,
        binding_id: str | None = None,
        caller_principal_id: str | None = None,
    ) -> MCPBindingStatus:
        return await self._call("attach_mcp_server", locals())

    async def list_mcp_servers(
        self, *, caller_principal_id: str | None = None
    ) -> tuple[MCPBindingStatus, ...]:
        return await self._call("list_mcp_servers", locals())

    async def refresh_mcp_server(
        self, binding_id: str, *, caller_principal_id: str | None = None
    ) -> MCPBindingStatus:
        return await self._call("refresh_mcp_server", locals())

    async def revoke_mcp_server(
        self, binding_id: str, *, caller_principal_id: str | None = None
    ) -> MCPBindingStatus:
        return await self._call("revoke_mcp_server", locals())

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._call("attach", locals())

    async def attach_sqlite(
        self,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> SourceRegistration:
        return await self._call("attach_sqlite", locals())

    async def edit_sqlite_source(
        self,
        source_id: str,
        path: str | Path,
        *,
        confirmation_handler: SourceEditConfirmationHandler,
        name: str | None = None,
    ) -> SourceEditResult | None:
        return await self._call("edit_sqlite_source", locals())

    async def store_postgresql_password(self, password: str) -> SecretReference:
        return await self._call("store_postgresql_password", locals())

    async def delete_postgresql_password(
        self,
        reference: SecretReference,
    ) -> None:
        return await self._call("delete_postgresql_password", locals())

    async def probe_postgresql(
        self,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        port: int = 5432,
        ssl_mode: str = "require",
        public_network_only: bool = False,
    ) -> PostgreSQLProbeResult:
        return await self._call("probe_postgresql", locals())

    async def attach_postgresql(
        self,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        schemas: tuple[str, ...],
        port: int = 5432,
        ssl_mode: str = "require",
        public_network_only: bool = False,
        name: str | None = None,
    ) -> SourceRegistration:
        return await self._call("attach_postgresql", locals())

    async def edit_postgresql_source(
        self,
        source_id: str,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        schemas: tuple[str, ...],
        confirmation_handler: SourceEditConfirmationHandler,
        port: int = 5432,
        ssl_mode: str = "require",
        public_network_only: bool = False,
        name: str | None = None,
    ) -> SourceEditResult | None:
        return await self._call("edit_postgresql_source", locals())

    async def inspect_source_permissions(
        self,
        source_id: str,
    ) -> SourcePermissionsInspection:
        return await self._call("inspect_source_permissions", locals())

    async def preview_source_permissions(
        self,
        *,
        source_id: str,
        read_mode: SourceReadMode | str,
        read_resource_ids: tuple[str, ...],
        relational_write_scopes: Mapping[str, Mapping[str, object]],
    ) -> SourcePermissionsPreview:
        return await self._call("preview_source_permissions", locals())

    async def apply_source_permissions(
        self,
        *,
        source_id: str,
        confirmation_fingerprint: str,
    ) -> SourcePermissionsInspection:
        return await self._call("apply_source_permissions", locals())

    async def relational_update_readiness(
        self,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
    ) -> RelationalUpdateReadiness:
        return await self._call("relational_update_readiness", locals())

    async def detach(self, source_id: str) -> SourceRegistration:
        return await self._call("detach", locals())

    async def refresh_source(self, source_id: str) -> SourceRegistration:
        return await self._call("refresh_source", locals())

    async def list_sources(self) -> tuple[SourceRegistration, ...]:
        return await self._call("list_sources", locals())

    async def list_catalog_resources(
        self, *, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        return await self._call("list_catalog_resources", locals())

    async def catalog_summary(self) -> CatalogSummary:
        return await self._call("catalog_summary", locals())

    async def inspect_catalog_resource(self, resource_id: str) -> FrozenJsonObject:
        return await self._call("inspect_catalog_resource", locals())


def _host_folder(workspace: LocalWorkspace | None) -> Path | None:
    """Return the folder the host opens, refusing a workspace it would widen."""

    if workspace is None:
        return None
    if workspace != LocalWorkspace(workspace.root, access=LocalFileAccess.COMPUTER):
        raise ValueError(
            "the Daita host opens a folder with computer access at the default "
            "sensitivity; use Agent.open for another workspace"
        )
    return workspace.root


async def _answer(
    kind: InteractionKind, handler: Any, request: Mapping[str, Any]
) -> dict[str, object]:
    if kind is InteractionKind.SOURCE_EDIT_CONFIRMATION:
        return {"accept": await handler(request["preview"])}
    return {"decision": ApprovalDecision(await handler(request["approval"])).value}


def _refused(error: Exception) -> bool:
    import httpx

    return isinstance(error, httpx.ConnectError)


def _guard(callback: Callable[..., object], *arguments: object) -> None:
    # Observation is best effort: a failing callback never stops the stream.
    try:
        callback(*arguments)
    except Exception:
        _LOGGER.exception("host client callback failed")


__all__ = ["HostClient", "HostUnavailableError", "RemoteAgent", "connect"]
