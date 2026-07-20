"""Thin development CLI over the local-agent request contract.

Command parsing owns no runtime behavior.  Every command is projected to one
named request so the local host remains the sole mutable-state owner.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterable, Mapping, Sequence
import json
import math
from pathlib import Path
import re
import sys
from typing import NoReturn, Protocol, TextIO
from uuid import uuid4

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

_AGENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


class CliRequest(Protocol):
    """One injectable request boundary used by the parser and focused tests."""

    async def __call__(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        idempotency_key: str | None,
    ) -> object: ...


class CliUsageError(ValueError):
    """Raised for invalid arguments without terminating the embedding process."""


class CliRequestError(RuntimeError):
    """A bounded error returned by the local-agent request boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliUsageError(message)


def _agent_name(value: str) -> str:
    if _AGENT_NAME.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "agent must start with a letter or digit and contain at most 64 "
            "letters, digits, underscores, or hyphens"
        )
    return value


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _idempotency_key(value: str) -> str:
    if not value or value != value.strip():
        raise argparse.ArgumentTypeError("must be a non-empty trimmed string")
    if len(value.encode("utf-8")) > 256:
        raise argparse.ArgumentTypeError("must be at most 256 UTF-8 bytes")
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite number {value} is not valid JSON")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        message = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        raise argparse.ArgumentTypeError(
            f"must be a strict JSON object: {message}"
        ) from error
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def _leaf(
    parents: argparse._SubParsersAction[_Parser],
    name: str,
    *,
    method: str,
    help_text: str,
    mutation: bool = False,
) -> _Parser:
    parser = parents.add_parser(name, help=help_text)
    parser.add_argument("agent", type=_agent_name)
    if mutation:
        parser.add_argument(
            "--idempotency-key",
            type=_idempotency_key,
            required=True,
        )
    parser.set_defaults(_method=method, _param_names=())
    return parser


def _params(parser: argparse.ArgumentParser, *names: str) -> None:
    parser.set_defaults(_param_names=names)


def build_parser() -> argparse.ArgumentParser:
    """Build the replaceable Phase 6 development parser."""

    parser = _Parser(prog="daita", description="Daita local-agent development CLI")
    parser.add_argument(
        "--root",
        type=Path,
        help="v2 state root (default: ~/.daita-next)",
    )
    commands = parser.add_subparsers(dest="_group", required=True)

    serve = commands.add_parser("serve", help="run one foreground local host")
    serve.add_argument("agent", type=_agent_name)
    serve.add_argument(
        "--openai-model",
        help="legacy one-shot provider override; omit after `model set`",
    )
    serve.add_argument(
        "--context-window-tokens",
        type=_positive_int,
        default=128_000,
    )
    serve.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=4_096,
    )
    serve.add_argument(
        "--cadence-seconds",
        type=_positive_float,
        default=1.0,
    )
    serve.set_defaults(
        _method="host.serve",
        _param_names=(
            "openai_model",
            "context_window_tokens",
            "max_output_tokens",
            "cadence_seconds",
        ),
        _local_serve=True,
    )

    agent = commands.add_parser("agent", help="agent bootstrap commands")
    agent_commands = agent.add_subparsers(dest="_action", required=True)
    _leaf(
        agent_commands,
        "create",
        method="agent.create",
        help_text="create an agent home through AgentHost",
        mutation=True,
    )
    _leaf(
        agent_commands,
        "init",
        method="agent.init",
        help_text="compatibility alias for agent create",
        mutation=True,
    )

    host = commands.add_parser("host", help="local host status commands")
    host_commands = host.add_subparsers(dest="_action", required=True)
    _leaf(
        host_commands,
        "status",
        method="host.status",
        help_text="inspect host status",
    )
    _leaf(
        host_commands,
        "health",
        method="host.health",
        help_text="inspect host health",
    )

    chat = commands.add_parser("chat", help="submit an interactive trigger")
    chat_commands = chat.add_subparsers(dest="_action", required=True)
    submit = _leaf(
        chat_commands,
        "submit",
        method="chat.submit",
        help_text="submit one user message",
        mutation=True,
    )
    submit.add_argument("message")
    submit.add_argument("--session-id")
    _params(submit, "message", "session_id")
    interactive = _leaf(
        chat_commands,
        "interactive",
        method="chat.interactive",
        help_text="read messages from stdin and stream committed events",
    )
    interactive.add_argument("--session-id")
    interactive.add_argument("--poll-seconds", type=_positive_float, default=0.1)
    _params(interactive, "session_id", "poll_seconds")

    operation = commands.add_parser("operation", help="operation controls")
    operation_commands = operation.add_subparsers(dest="_action", required=True)
    inspect_operation = _leaf(
        operation_commands,
        "inspect",
        method="operation.inspect",
        help_text="inspect one durable operation",
    )
    inspect_operation.add_argument("operation_id")
    _params(inspect_operation, "operation_id")
    list_operations = _leaf(
        operation_commands,
        "list",
        method="operation.list",
        help_text="list bounded durable operations",
    )
    list_operations.add_argument("--status", action="append", dest="statuses")
    list_operations.add_argument("--limit", type=_positive_int, default=100)
    _params(list_operations, "statuses", "limit")
    cancel_operation = _leaf(
        operation_commands,
        "cancel",
        method="operation.cancel",
        help_text="request durable cancellation",
        mutation=True,
    )
    cancel_operation.add_argument("operation_id")
    cancel_operation.add_argument("--reason", default="user_cancelled")
    _params(cancel_operation, "operation_id", "reason")

    approval = commands.add_parser("approval", help="approval decisions")
    approval_commands = approval.add_subparsers(dest="_action", required=True)
    for action in ("approve", "reject"):
        decision = _leaf(
            approval_commands,
            action,
            method=f"approval.{action}",
            help_text=f"{action} one waiting approval",
            mutation=True,
        )
        decision.add_argument("approval_id")
        decision.add_argument("--actor", required=True, dest="actor_id")
        decision.add_argument("--reason", required=True)
        _params(decision, "approval_id", "actor_id", "reason")
    list_approvals = _leaf(
        approval_commands,
        "list",
        method="approval.list",
        help_text="list bounded durable approvals",
    )
    list_approvals.add_argument("--status", action="append", dest="statuses")
    list_approvals.add_argument("--limit", type=_positive_int, default=100)
    _params(list_approvals, "statuses", "limit")
    inspect_approval = _leaf(
        approval_commands,
        "inspect",
        method="approval.inspect",
        help_text="inspect one durable approval",
    )
    inspect_approval.add_argument("approval_id")
    _params(inspect_approval, "approval_id")

    source = commands.add_parser("source", help="source attachment")
    source_commands = source.add_subparsers(dest="_action", required=True)
    for action in ("add", "attach"):
        attach_source = _leaf(
            source_commands,
            action,
            method="source.attach",
            help_text="attach and discover one local source",
            mutation=True,
        )
        attach_source.add_argument("kind", choices=("sqlite", "local_files"))
        attach_source.add_argument("path")
        attach_source.add_argument(
            "--write-access",
            action="store_true",
            default=None,
            help="explicitly enable controlled SQLite writes",
        )
        _params(attach_source, "kind", "path", "write_access")
    list_sources = _leaf(
        source_commands,
        "list",
        method="source.list",
        help_text="list attached sources",
    )
    list_sources.add_argument("--include-detached", action="store_true")
    _params(list_sources, "include_detached")
    detach_source = _leaf(
        source_commands,
        "detach",
        method="source.detach",
        help_text="detach one source",
        mutation=True,
    )
    detach_source.add_argument("source_id")
    _params(detach_source, "source_id")
    source_health = _leaf(
        source_commands,
        "health",
        method="source.health",
        help_text="inspect source and catalog readiness",
    )
    source_health.add_argument("source_id", nargs="?")
    _params(source_health, "source_id")

    model = commands.add_parser("model", help="model configuration status")
    model_commands = model.add_subparsers(dest="_action", required=True)
    _leaf(
        model_commands,
        "status",
        method="model.status",
        help_text="inspect the active model profile",
    )
    _leaf(
        model_commands,
        "show",
        method="model.show",
        help_text="show the non-secret retained model route",
    )
    set_model = _leaf(
        model_commands,
        "set",
        method="model.set",
        help_text="persist a retained future-operation model route",
        mutation=True,
    )
    set_model.add_argument("model_id", nargs="?")
    set_model.add_argument("--provider")
    set_model.add_argument("--model")
    set_model.add_argument("--secret")
    set_model.add_argument(
        "--context-window-tokens", type=_positive_int, default=128_000
    )
    set_model.add_argument("--max-output-tokens", type=_positive_int, default=4_096)
    set_model.add_argument("--base-url")
    set_model.add_argument("--secret-env")
    set_model.add_argument(
        "--allow-sensitivity",
        action="append",
        dest="allowed_sensitivities",
        choices=("public", "internal", "confidential", "restricted"),
    )
    _params(
        set_model,
        "model_id",
        "provider",
        "model",
        "secret",
        "context_window_tokens",
        "max_output_tokens",
        "base_url",
        "secret_env",
        "allowed_sensitivities",
    )

    catalog = commands.add_parser("catalog", help="catalog inspection")
    catalog_commands = catalog.add_subparsers(dest="_action", required=True)
    search_catalog = _leaf(
        catalog_commands,
        "search",
        method="catalog.search",
        help_text="search current catalog resources",
    )
    search_catalog.add_argument("query")
    search_catalog.add_argument("--source-id", action="append", dest="source_ids")
    search_catalog.add_argument("--kind", action="append", dest="resource_kinds")
    search_catalog.add_argument("--limit", type=_positive_int, default=20)
    _params(search_catalog, "query", "source_ids", "resource_kinds", "limit")
    show_catalog = _leaf(
        catalog_commands,
        "show",
        method="catalog.show",
        help_text="show one current catalog resource",
    )
    show_catalog.add_argument("resource_id")
    _params(show_catalog, "resource_id")

    memory = commands.add_parser("memory", help="memory inspection")
    memory_commands = memory.add_subparsers(dest="_action", required=True)
    list_memory = _leaf(
        memory_commands,
        "list",
        method="memory.list",
        help_text="list governed memory",
    )
    for name in ("user_id", "session_id", "source_id", "resource_id"):
        list_memory.add_argument(f"--{name.replace('_', '-')}")
    list_memory.add_argument("--include-superseded", action="store_true")
    list_memory.add_argument("--include-rejected", action="store_true")
    list_memory.add_argument("--limit", type=_positive_int, default=100)
    _params(
        list_memory,
        "user_id",
        "session_id",
        "source_id",
        "resource_id",
        "include_superseded",
        "include_rejected",
        "limit",
    )
    inspect_memory = _leaf(
        memory_commands,
        "inspect",
        method="memory.inspect",
        help_text="inspect one memory history",
    )
    inspect_memory.add_argument("memory_id")
    _params(inspect_memory, "memory_id")

    skill = commands.add_parser("skill", help="skill inspection")
    skill_commands = skill.add_subparsers(dest="_action", required=True)
    _leaf(
        skill_commands,
        "list",
        method="skill.list",
        help_text="list indexed skills",
    )
    inspect_skill = _leaf(
        skill_commands,
        "inspect",
        method="skill.inspect",
        help_text="inspect one skill history",
    )
    inspect_skill.add_argument("skill_id")
    _params(inspect_skill, "skill_id")

    events = commands.add_parser("events", help="committed event log")
    event_commands = events.add_subparsers(dest="_action", required=True)
    read_events = _leaf(
        event_commands,
        "read",
        method="events.read",
        help_text="read a bounded committed-event page",
    )
    read_events.add_argument("--after", type=_positive_int)
    read_events.add_argument("--limit", type=_positive_int, default=100)
    _params(read_events, "after", "limit")
    follow_events = _leaf(
        event_commands,
        "follow",
        method="events.follow",
        help_text="follow committed events from a durable cursor",
    )
    follow_events.add_argument("--after", type=_positive_int)
    follow_events.add_argument("--limit", type=_positive_int, default=100)
    follow_events.add_argument("--poll-seconds", type=_positive_float, default=0.1)
    follow_events.add_argument("--max-events", type=_positive_int)
    _params(follow_events, "after", "limit", "poll_seconds", "max_events")

    monitor = commands.add_parser("monitor", help="monitor lifecycle")
    monitor_commands = monitor.add_subparsers(dest="_action", required=True)
    propose_monitor = _leaf(
        monitor_commands,
        "propose",
        method="monitor.propose",
        help_text="persist an inert monitor proposal",
        mutation=True,
    )
    propose_monitor.add_argument("monitor_id")
    propose_monitor.add_argument("--definition", type=_json_object, required=True)
    propose_monitor.add_argument("--source-operation-id")
    _params(
        propose_monitor,
        "monitor_id",
        "definition",
        "source_operation_id",
    )
    propose_natural = _leaf(
        monitor_commands,
        "propose-natural",
        method="monitor.propose_natural",
        help_text="parse a safe natural interval monitor proposal",
        mutation=True,
    )
    propose_natural.add_argument("monitor_id")
    propose_natural.add_argument("request")
    propose_natural.add_argument("--source-operation-id")
    _params(
        propose_natural,
        "monitor_id",
        "request",
        "source_operation_id",
    )

    list_monitors = _leaf(
        monitor_commands,
        "list",
        method="monitor.list",
        help_text="list durable monitors",
    )
    list_monitors.add_argument("--status", action="append", dest="statuses")
    list_monitors.add_argument("--include-deleted", action="store_true")
    list_monitors.add_argument("--limit", type=_positive_int, default=100)
    _params(list_monitors, "statuses", "include_deleted", "limit")

    inspect_monitor = _leaf(
        monitor_commands,
        "inspect",
        method="monitor.inspect",
        help_text="inspect one durable monitor",
    )
    inspect_monitor.add_argument("monitor_id")
    _params(inspect_monitor, "monitor_id")

    confirm_monitor = _leaf(
        monitor_commands,
        "confirm",
        method="monitor.confirm",
        help_text="confirm the exact proposed monitor definition",
        mutation=True,
    )
    confirm_monitor.add_argument("proposal_id")
    confirm_monitor.add_argument("--candidate-hash", required=True)
    confirm_monitor.add_argument("--actor", required=True, dest="actor_id")
    confirm_monitor.add_argument("--reason", required=True)
    _params(
        confirm_monitor,
        "proposal_id",
        "candidate_hash",
        "actor_id",
        "reason",
    )

    for action in ("pause", "resume", "delete"):
        lifecycle = _leaf(
            monitor_commands,
            action,
            method=f"monitor.{action}",
            help_text=f"{action} one monitor",
            mutation=True,
        )
        lifecycle.add_argument("monitor_id")
        lifecycle.add_argument("--actor", required=True, dest="actor_id")
        lifecycle.add_argument("--reason", required=True)
        lifecycle.add_argument("--operation-id")
        _params(lifecycle, "monitor_id", "actor_id", "reason", "operation_id")

    run_now = _leaf(
        monitor_commands,
        "run-now",
        method="monitor.run_now",
        help_text="claim and run one manual monitor occurrence",
        mutation=True,
    )
    run_now.add_argument("monitor_id")
    run_now.add_argument("--lease-seconds", type=_positive_float, default=60.0)
    _params(run_now, "monitor_id", "lease_seconds")

    return parser


def _request_params(namespace: argparse.Namespace) -> dict[str, object]:
    return {
        name: getattr(namespace, name)
        for name in namespace._param_names
        if getattr(namespace, name) is not None
    }


def _json_projection(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_projection(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_projection(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_projection(to_dict())
    raise TypeError(f"request result contains unsupported {type(value).__name__}")


def _write_json(value: object, *, stream: TextIO | None = None) -> None:
    destination = sys.stdout if stream is None else stream
    print(
        json.dumps(
            _json_projection(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=destination,
        flush=True,
    )


async def _dispatch(namespace: argparse.Namespace, request: CliRequest) -> int:
    result = await request(
        namespace._method,
        _request_params(namespace),
        idempotency_key=getattr(namespace, "idempotency_key", None),
    )
    if isinstance(result, AsyncIterable):
        async for item in result:
            _write_json(item)
        return EXIT_OK
    _write_json(result)
    return EXIT_OK


def _agent_home(namespace: argparse.Namespace) -> Path:
    return _state_root(namespace) / "agents" / namespace.agent


def _state_root(namespace: argparse.Namespace) -> Path:
    return (
        Path.home() / ".daita-next"
        if namespace.root is None
        else namespace.root.expanduser().absolute()
    )


def _default_request(namespace: argparse.Namespace) -> CliRequest:
    """Load the local transport only when an actual CLI request needs it."""

    # The import is intentionally local: command parsing remains usable in
    # isolation and does not make transport construction a package side effect.
    from .hosting.local_protocol import (
        LocalAgentClient,
        LocalErrorResponse,
        LocalProtocolError,
        LocalRequest,
        LocalSocketSecurityError,
    )

    async def request_once(
        method: str,
        params: Mapping[str, object],
        *,
        idempotency_key: str | None = None,
    ) -> object:
        client = LocalAgentClient(_agent_home(namespace))
        response = await client.request(
            LocalRequest.create(
                request_id=f"cli-{uuid4().hex}",
                method=method,
                params=params,
                idempotency_key=idempotency_key,
            )
        )
        if isinstance(response, LocalErrorResponse):
            raise CliRequestError(response.error.code, response.error.message)
        return response.result

    async def transient_request(
        method: str,
        params: Mapping[str, object],
        *,
        idempotency_key: str | None,
    ) -> object:
        from .hosting.embedded import HostActiveError
        from .hosting.host import AgentHost
        from .hosting.local_server import LocalAgentServer

        try:
            host = await AgentHost.open(
                namespace.agent,
                root=_state_root(namespace),
            )
        except HostActiveError as error:
            raise CliRequestError("host_unavailable", str(error)) from error
        try:
            await host.start()
            response = await LocalAgentServer(host).dispatch(
                LocalRequest.create(
                    request_id=f"cli-{uuid4().hex}",
                    method=method,
                    params=params,
                    idempotency_key=idempotency_key,
                )
            )
            if isinstance(response, LocalErrorResponse):
                raise CliRequestError(response.error.code, response.error.message)
            return response.result
        finally:
            await host.stop(drain=True)

    async def set_model(params: Mapping[str, object]) -> object:
        from .hosting.embedded import HostActiveError
        from .hosting.host import AgentHost
        from .llm.models import ModelProfile, ModelSensitivity
        from .llm.routing import ModelRoute, ModelRouteCandidate
        from .security import SecretReference

        model_id = params.get("model_id")
        provider_option = params.get("provider")
        model_option = params.get("model")
        if model_id is None:
            if not isinstance(provider_option, str) or not isinstance(
                model_option, str
            ):
                raise CliUsageError(
                    "model set requires provider:model or both --provider and --model"
                )
            model_id = f"{provider_option}:{model_option}"
        elif provider_option is not None or model_option is not None:
            raise CliUsageError(
                "model set cannot combine provider:model with --provider/--model"
            )
        context_window_tokens = params["context_window_tokens"]
        max_output_tokens = params["max_output_tokens"]
        if not isinstance(model_id, str):
            raise CliUsageError("model_id must be canonical provider:model text")
        if not isinstance(context_window_tokens, int) or not isinstance(
            max_output_tokens, int
        ):
            raise CliUsageError("model token limits must be integers")
        profile = ModelProfile(
            id=model_id,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_streaming=True,
        )
        provider_name = model_id.partition(":")[0]
        base_url = params.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise CliUsageError("base_url must be text")
        if provider_name == "ollama" and base_url is None:
            base_url = "http://127.0.0.1:11434/v1"
        if provider_name in {"openai", "anthropic", "gemini", "grok"}:
            if base_url is not None:
                raise CliUsageError(f"{provider_name} uses its fixed endpoint")
        elif provider_name != "ollama" and base_url is None:
            raise CliUsageError(
                "an explicit --base-url is required for compatible providers"
            )
        secret_env = params.get("secret_env")
        secret_reference = params.get("secret")
        if secret_reference is not None:
            if secret_env is not None:
                raise CliUsageError("use only one of --secret and --secret-env")
            if not isinstance(secret_reference, str) or not secret_reference.startswith(
                "env:"
            ):
                raise CliUsageError("--secret currently supports only env:NAME")
            secret_env = secret_reference.removeprefix("env:")
        if secret_env is None:
            secret_env = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "grok": "XAI_API_KEY",
            }.get(provider_name)
        if secret_env is not None and not isinstance(secret_env, str):
            raise CliUsageError("secret_env must be text")
        raw_sensitivities = params.get("allowed_sensitivities", ["public", "internal"])
        if not isinstance(raw_sensitivities, list):
            raise CliUsageError("allowed_sensitivities must be a list")
        allowed = frozenset(ModelSensitivity(value) for value in raw_sensitivities)
        candidate = ModelRouteCandidate(
            profile=profile,
            allowed_sensitivities=allowed,
            base_url=base_url,
            secret_reference=(
                None if secret_env is None else SecretReference.environment(secret_env)
            ),
        )
        try:
            host = await AgentHost.open(
                namespace.agent,
                root=_state_root(namespace),
            )
        except HostActiveError as error:
            raise CliRequestError(
                "host_active",
                "stop the active host before changing its model route",
            ) from error
        try:
            current = host.model_route
            if current is not None and current.candidates == (candidate,):
                stored = current
                changed = False
            else:
                expected_revision = 0 if current is None else current.revision
                stored = await host.configure_model_route(
                    ModelRoute(
                        candidates=(candidate,),
                        revision=expected_revision + 1,
                    ),
                    expected_revision=expected_revision,
                )
                changed = True
            return {
                "agent_id": host.id,
                "changed": changed,
                "configured": True,
                "fingerprint": stored.fingerprint,
                "model_profile_id": stored.model_profile.id,
                "revision": stored.revision,
            }
        finally:
            await host.stop(drain=False)

    async def follow_events(params: Mapping[str, object]):
        after = params.get("after")
        limit = params.get("limit", 100)
        poll_seconds = params.get("poll_seconds", 0.1)
        maximum = params.get("max_events")
        if after is not None and not isinstance(after, int):
            raise CliUsageError("after must be an integer")
        if not isinstance(limit, int) or not isinstance(poll_seconds, (int, float)):
            raise CliUsageError("event follow bounds are invalid")
        if maximum is not None and not isinstance(maximum, int):
            raise CliUsageError("max_events must be an integer")
        emitted = 0
        while maximum is None or emitted < maximum:
            try:
                result = await request_once(
                    "events.read",
                    (
                        {"after": after, "limit": limit}
                        if after is not None
                        else {"limit": limit}
                    ),
                )
            except (ConnectionError, LocalSocketSecurityError, OSError):
                await asyncio.sleep(float(poll_seconds))
                continue
            projected = _json_projection(result)
            if not isinstance(projected, dict):
                raise CliRequestError("invalid_response", "event page is invalid")
            events = projected.get("events")
            if not isinstance(events, list):
                raise CliRequestError("invalid_response", "event page is invalid")
            for event in events:
                if not isinstance(event, dict) or not isinstance(
                    event.get("sequence"), int
                ):
                    raise CliRequestError("invalid_response", "event is invalid")
                after = event["sequence"]
                yield event
                emitted += 1
                if maximum is not None and emitted >= maximum:
                    return
            if not events:
                await asyncio.sleep(float(poll_seconds))

    async def event_tail() -> int | None:
        after = None
        while True:
            page = await request_once(
                "events.read",
                {"limit": 1_000, **({} if after is None else {"after": after})},
            )
            projected = _json_projection(page)
            if not isinstance(projected, dict) or not isinstance(
                projected.get("events"), list
            ):
                raise CliRequestError("invalid_response", "event page is invalid")
            events = projected["events"]
            if not events:
                return after
            last = events[-1]
            if not isinstance(last, dict) or not isinstance(last.get("sequence"), int):
                raise CliRequestError("invalid_response", "event is invalid")
            after = last["sequence"]
            if len(events) < 1_000:
                return after

    async def interactive_chat(params: Mapping[str, object]):
        session_id = params.get("session_id")
        poll_seconds = params.get("poll_seconds", 0.1)
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if line == "":
                return
            message = line.strip()
            if not message:
                continue
            after = await event_tail()
            result = await request_once(
                "chat.submit",
                {
                    "message": message,
                    **({} if session_id is None else {"session_id": session_id}),
                },
                idempotency_key=f"chat-{uuid4().hex}",
            )
            final_after = await event_tail()
            if final_after is not None and final_after != after:
                async for event in follow_events(
                    {
                        "after": after,
                        "limit": 100,
                        "poll_seconds": poll_seconds,
                        "max_events": (
                            final_after if after is None else final_after - after
                        ),
                    }
                ):
                    yield {"kind": "event", "event": event}
            yield {"kind": "result", "result": result}

    async def send(
        method: str,
        params: Mapping[str, object],
        *,
        idempotency_key: str | None,
    ) -> object:
        if method in {"agent.create", "agent.init"}:
            from .hosting.embedded import AgentAlreadyExistsError, HostActiveError
            from .hosting.host import AgentHost

            created = True
            try:
                host = await AgentHost.create(
                    namespace.agent,
                    root=_state_root(namespace),
                )
            except HostActiveError as error:
                raise CliRequestError("host_active", str(error)) from error
            except AgentAlreadyExistsError:
                created = False
                try:
                    host = await AgentHost.open(
                        namespace.agent,
                        root=_state_root(namespace),
                    )
                except HostActiveError as error:
                    raise CliRequestError("host_active", str(error)) from error
            try:
                return {
                    "agent_id": host.id,
                    "created": created,
                    "home": str(host.home),
                    "name": host.name,
                }
            finally:
                await host.stop(drain=False)
        if method == "model.set":
            return await set_model(params)
        if method == "events.follow":
            return follow_events(params)
        if method == "chat.interactive":
            return interactive_chat(params)
        try:
            return await request_once(
                method,
                params,
                idempotency_key=idempotency_key,
            )
        except LocalProtocolError as error:
            raise CliRequestError(error.code, error.message) from error
        except (ConnectionError, LocalSocketSecurityError, OSError) as error:
            if method in {
                "model.show",
                "model.status",
                "source.attach",
                "source.detach",
                "source.health",
                "source.list",
            }:
                return await transient_request(
                    method,
                    params,
                    idempotency_key=idempotency_key,
                )
            raise CliRequestError("host_unavailable", str(error)) from error

    return send


async def _serve_local(namespace: argparse.Namespace) -> int:
    if namespace.max_output_tokens >= namespace.context_window_tokens:
        raise CliUsageError(
            "--max-output-tokens must be smaller than --context-window-tokens"
        )
    from .hosting.embedded import AgentHomeError
    from .hosting.host import AgentHost
    from .hosting.local_server import LocalAgentServer
    from .llm.models import ModelProfile
    from .llm.providers.openai import OpenAIResponsesProvider

    provider = None
    profile = None
    if namespace.openai_model is not None:
        provider = OpenAIResponsesProvider(namespace.openai_model)
        profile = ModelProfile(
            id=provider.provider_id,
            context_window_tokens=namespace.context_window_tokens,
            max_output_tokens=namespace.max_output_tokens,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_reasoning=True,
        )
    try:
        host = await AgentHost.open(
            namespace.agent,
            root=_state_root(namespace),
            model=provider,
            model_profile=profile,
            cadence_seconds=namespace.cadence_seconds,
        )
    except AgentHomeError as error:
        raise CliRequestError("host_open_failed", str(error)) from error
    server = LocalAgentServer(host)
    try:
        await server.start()
        _write_json(
            {
                "agent_id": host.id,
                "model_profile_id": (
                    None if host.model_profile is None else host.model_profile.id
                ),
                "socket": str(server.socket_path),
                "state": "running",
            }
        )
        await server.serve_forever()
    finally:
        await server.stop(drain=True)
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    request: CliRequest | None = None,
) -> int:
    """Parse and dispatch one CLI command, returning a stable process code."""

    parser = build_parser()
    try:
        arguments = list(sys.argv[1:] if argv is None else argv)
        if "chat" in arguments:
            chat_index = arguments.index("chat")
            if chat_index + 1 < len(arguments) and arguments[chat_index + 1] not in {
                "submit",
                "interactive",
            }:
                arguments.insert(chat_index + 1, "interactive")
        namespace = parser.parse_args(arguments)
        if getattr(namespace, "_local_serve", False) and request is None:
            return asyncio.run(_serve_local(namespace))
        selected_request = request or _default_request(namespace)
        return asyncio.run(_dispatch(namespace, selected_request))
    except CliUsageError as error:
        _write_json(
            {"error": {"code": "usage_error", "message": str(error)}},
            stream=sys.stderr,
        )
        return EXIT_USAGE
    except CliRequestError as error:
        _write_json(
            {"error": {"code": error.code, "message": error.message}},
            stream=sys.stderr,
        )
        return EXIT_ERROR
    except (ConnectionError, OSError) as error:
        _write_json(
            {
                "error": {
                    "code": "host_unavailable",
                    "message": str(error),
                }
            },
            stream=sys.stderr,
        )
        return EXIT_ERROR
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
