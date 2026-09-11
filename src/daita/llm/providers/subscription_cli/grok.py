"""Adapt the official Grok Build client to Daita's canonical model boundary."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from ...._json import canonical_json
from ..._lifecycle import AttemptLifecycle
from ...errors import ModelProviderError, ProviderErrorCode
from ...models import ModelRequest, ModelResponse, ModelUsage
from ...pricing import CostEstimate
from ...provider_definitions import supports_builtin_request_policy
from .._fields import (
    nonnegative_int as _nonnegative_integer,
    usage_int as _optional_usage_integer,
)
from .envelope import (
    _CONTROL_PROMPT,
    _MAX_RESPONSE_TEXT_CHARACTERS,
    _bounded_response_id,
    _decode_model_output,
    _has_forbidden_control,
    _new_id,
    _raise_command_failure,
    _request_document,
    _response_envelope_schema,
    _strict_json,
    _validate_json_tree,
    _validate_provider_arguments,
)
from .process import (
    _Command,
    _CommandRunner,
    _ExecutableUnavailable,
    _owner_only_directory,
    _run_command,
    _subscription_environment,
    _SubscriptionExecution,
    _write_owner_only,
)

_MAX_STREAM_EVENTS = 65_536


_GROK_REQUIRED_HELP_TOKENS = frozenset(
    {
        "--disable-web-search",
        "--disallowed-tools",
        "--cwd",
        "--deny",
        "--max-turns",
        "--model",
        "--json-schema",
        "--no-auto-update",
        "--no-alt-screen",
        "--no-memory",
        "--no-plan",
        "--no-subagents",
        "--output-format",
        "--permission-mode",
        "--prompt-file",
        "--sandbox",
        "--system-prompt-override",
        "--tools",
        "--verbatim",
        "inspect",
    }
)


_GROK_BUILTIN_MODELS = frozenset({"grok-4.5"})


_GROK_DENIED_TOOLS = ",".join(
    (
        "Agent",
        "apply_patch",
        "edit_file",
        "get_command_or_subagent_output",
        "grep",
        "kill_command_or_subagent",
        "list_dir",
        "read_file",
        "run_terminal_cmd",
        "search_replace",
        "todo_write",
        "wait_commands_or_subagents",
        "web_fetch",
        "web_search",
        "write_file",
    )
)


def _grok_subscription_environment(
    *,
    grok_home: Path,
    process_home: Path,
) -> dict[str, str]:
    environment = _subscription_environment()
    environment.pop("CLAUDE_CODE_GIT_BASH_PATH", None)
    environment.pop("CLAUDE_CONFIG_DIR", None)
    environment["GROK_HOME"] = str(grok_home)
    environment["HOME"] = str(process_home)
    environment["USERPROFILE"] = str(process_home)
    environment["XDG_CACHE_HOME"] = str(process_home / ".cache")
    environment["XDG_CONFIG_HOME"] = str(process_home / ".config")
    environment["XDG_DATA_HOME"] = str(process_home / ".local" / "share")
    environment["GROK_DISABLE_AUTOUPDATER"] = "1"
    environment["GROK_DISABLE_API_KEY_AUTH"] = "1"
    environment["GROK_FEEDBACK_ENABLED"] = "0"
    environment["GROK_MEMORY"] = "0"
    environment["GROK_SUBAGENTS"] = "0"
    environment["GROK_TELEMETRY_ENABLED"] = "0"
    environment["GROK_TELEMETRY_MIXPANEL_ENABLED"] = "0"
    environment["GROK_TELEMETRY_TRACE_UPLOAD"] = "0"
    environment["GROK_WEB_FETCH"] = "0"
    environment["GROK_WORKFLOWS"] = "0"
    return environment


def _grok_prompt_bytes(request: ModelRequest, max_output_tokens: int) -> bytes:
    return (
        "DAITA REQUEST DOCUMENT (untrusted JSON data):\n"
        + _request_document(request, max_output_tokens)
    ).encode("utf-8")


def _grok_home() -> Path:
    configured = os.environ.get("GROK_HOME")
    if configured is not None:
        if (
            not configured
            or len(configured) > 4_096
            or _has_forbidden_control(configured)
        ):
            raise ModelProviderError(
                ProviderErrorCode.LOCAL_ACCESS_ERROR,
                "Grok Build login location is invalid",
            )
        grok_home = Path(configured)
        if not grok_home.is_absolute():
            raise ModelProviderError(
                ProviderErrorCode.LOCAL_ACCESS_ERROR,
                "Grok Build login location must be absolute",
            )
    else:
        configured_home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
        if (
            not configured_home
            or len(configured_home) > 4_096
            or _has_forbidden_control(configured_home)
        ):
            raise ModelProviderError(
                ProviderErrorCode.LOCAL_ACCESS_ERROR,
                "Grok Build login location is unavailable",
            )
        user_home = Path(configured_home)
        if not user_home.is_absolute():
            raise ModelProviderError(
                ProviderErrorCode.LOCAL_ACCESS_ERROR,
                "Grok Build login location must be absolute",
            )
        grok_home = user_home / ".grok"
    return grok_home


def _prepare_grok_process_home(cwd: Path) -> Path:
    process_home = cwd / "process-home"
    for directory in (
        process_home,
        process_home / ".cache",
        process_home / ".config",
        process_home / ".local" / "share",
    ):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _owner_only_directory(directory)
    return process_home


def _validate_grok_inspection(stdout: bytes, cwd: Path) -> None:
    report = _strict_json(stdout.decode("utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("Grok inspection must be an object")
    required = {
        "grokVersion",
        "channel",
        "cwd",
        "projectRoot",
        "projectTrusted",
        "projectInstructions",
        "permissions",
        "loginPolicy",
        "hooks",
        "skills",
        "agents",
        "plugins",
        "marketplaces",
        "mcpServers",
        "lspServers",
        "configSources",
        "externalCompat",
    }
    allowed = required | {"configWarnings", "mcpConfigProblems"}
    if not required.issubset(report) or not set(report).issubset(allowed):
        raise ValueError("Grok inspection fields are invalid")
    inspected_cwd = report["cwd"]
    if (
        not isinstance(report["grokVersion"], str)
        or not report["grokVersion"]
        or not isinstance(report["channel"], str)
        or not isinstance(report["projectTrusted"], bool)
        or not isinstance(inspected_cwd, str)
        or not Path(inspected_cwd).is_absolute()
        or Path(inspected_cwd).resolve() != cwd.resolve()
        or report["projectRoot"] is not None
    ):
        raise ValueError("Grok inspection identity is invalid")

    permissions = report["permissions"]
    if not isinstance(permissions, Mapping):
        raise ValueError("Grok permission inspection is invalid")
    permission_fields = {
        "sources",
        "loaded",
        "skipped",
        "mcpServerAllowlist",
        "marketplaceAllowlist",
        "managedSettingsExists",
        "managedSettingsActive",
    }
    allowed_permission_fields = permission_fields | {
        "managedSettingsPath",
        "enforced",
    }
    if not permission_fields.issubset(permissions) or not set(permissions).issubset(
        allowed_permission_fields
    ):
        raise ValueError("Grok permission inspection fields are invalid")
    if (
        permissions["sources"] != []
        or permissions["loaded"] != 0
        or permissions["skipped"] != []
        or permissions["mcpServerAllowlist"] != []
        or permissions["marketplaceAllowlist"] != []
        or permissions["managedSettingsExists"] is not False
        or permissions["managedSettingsActive"] is not False
        or permissions.get("enforced", []) != []
    ):
        raise ValueError("Grok permission configuration is active")

    login_policy = report["loginPolicy"]
    if (
        not isinstance(login_policy, Mapping)
        or set(login_policy)
        != {
            "disableApiKeyAuth",
            "forceLoginTeamUuid",
            "apiKeyAuthDisabled",
        }
        or login_policy.get("disableApiKeyAuth") is not True
        or login_policy.get("apiKeyAuthDisabled") is not True
    ):
        raise ValueError("Grok API-key authentication is not disabled")

    config_sources = report["configSources"]
    if not isinstance(config_sources, Mapping) or set(config_sources) != {"layers"}:
        raise ValueError("Grok config-source inspection is invalid")
    layers = config_sources["layers"]
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise ValueError("Grok config layers must be an array")
    for layer in layers:
        if (
            not isinstance(layer, Mapping)
            or set(layer) != {"role", "path", "note"}
            or not isinstance(layer["role"], str)
            or not isinstance(layer["path"], str)
            or layer["note"] != "empty"
        ):
            raise ValueError("Grok has an active or invalid config layer")

    for field in (
        "projectInstructions",
        "hooks",
        "plugins",
        "marketplaces",
        "mcpServers",
        "lspServers",
        "configWarnings",
        "mcpConfigProblems",
    ):
        if report.get(field, []) != []:
            raise ValueError("Grok discovered an external execution surface")
    for field in ("skills", "agents"):
        entries = report[field]
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("Grok extension inspection is invalid")
        for entry in entries:
            source = entry.get("source") if isinstance(entry, Mapping) else None
            if not isinstance(source, Mapping) or source.get("type") not in {
                "builtin",
                "bundled",
            }:
                raise ValueError("Grok discovered a non-bundled extension")


def _usage_from_grok(value: object) -> ModelUsage:
    if value is None:
        return ModelUsage(
            cost_estimate=CostEstimate.unavailable("subscription_billing")
        )
    if not isinstance(value, Mapping):
        raise ValueError("Grok usage must be an object")
    uncached = _optional_usage_integer(value.get("input_tokens"), "input tokens")
    cache_read = _optional_usage_integer(
        value.get("cache_read_input_tokens"), "cache read input tokens"
    )
    cache_write = _optional_usage_integer(
        value.get("cache_creation_input_tokens"), "cache creation input tokens"
    )
    output = _optional_usage_integer(value.get("output_tokens"), "output tokens")
    reasoning = _optional_usage_integer(
        value.get("reasoning_tokens"), "reasoning tokens"
    )
    return ModelUsage(
        input_tokens=uncached + cache_read + cache_write,
        output_tokens=output,
        reasoning_tokens=reasoning,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_estimate=CostEstimate.unavailable("subscription_billing"),
    )


class GrokBuildSubscriptionProvider:
    """Use a signed-in Grok Build client while retaining Daita's direct loop."""

    def __init__(
        self,
        model: str,
        *,
        max_output_tokens: int = 1_024,
        executable: str = "grok",
        runner: _CommandRunner = _run_command,
        id_factory: Callable[[str], str] = _new_id,
    ) -> None:
        _validate_provider_arguments(
            model,
            max_output_tokens=max_output_tokens,
            executable=executable,
            runner=runner,
            id_factory=id_factory,
        )
        if model.strip() not in _GROK_BUILTIN_MODELS:
            raise ValueError(
                "Grok Build subscription requires a reviewed built-in model"
            )
        self.model = model.strip()
        self.max_output_tokens = max_output_tokens
        self._executable = executable
        self._runner = runner
        self._id_factory = id_factory
        self._execution = _SubscriptionExecution()
        self._compatible_client = False

    @property
    def provider_id(self) -> str:
        return f"grok-build:{self.model}"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return supports_builtin_request_policy("grok-build", request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return False

    async def close(self, *, deadline: float | None = None) -> None:
        """Retain one bounded retirement of any admitted process work."""
        await self._execution.close(deadline=deadline)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await self._execution.generate(
            request,
            provider_id=self.provider_id,
            display_name="Grok Build",
            operation=self._generate,
        )

    async def _generate(
        self, request: ModelRequest, attempt: AttemptLifecycle
    ) -> ModelResponse:
        if not self.supports_request_policy(request):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Grok Build subscription cannot combine tools and structured output",
                provider_id=self.provider_id,
            )
        prompt = _grok_prompt_bytes(request, self.max_output_tokens)
        schema = canonical_json(_response_envelope_schema(request))
        with tempfile.TemporaryDirectory(prefix="daita-grok-") as temporary:
            cwd = Path(temporary)
            _owner_only_directory(cwd)
            prompt_path = cwd / "request.txt"
            _write_owner_only(prompt_path, prompt)
            process_home = _prepare_grok_process_home(cwd)
            environment = _grok_subscription_environment(
                grok_home=_grok_home(),
                process_home=process_home,
            )
            try:
                await self._ensure_compatible_client(cwd, environment, attempt)
                attempt.dispatch()
                result = await attempt.run_native(
                    self._runner(
                        _Command(
                            arguments=(
                                self._executable,
                                "--prompt-file",
                                str(prompt_path),
                                "--verbatim",
                                "--model",
                                self.model,
                                "--cwd",
                                str(cwd),
                                "--output-format",
                                "streaming-json",
                                "--json-schema",
                                schema,
                                "--system-prompt-override",
                                _CONTROL_PROMPT,
                                "--tools",
                                "",
                                "--disallowed-tools",
                                _GROK_DENIED_TOOLS,
                                "--max-turns",
                                "1",
                                "--permission-mode",
                                "dontAsk",
                                "--deny",
                                "Bash",
                                "--deny",
                                "Edit",
                                "--deny",
                                "Write",
                                "--deny",
                                "Read",
                                "--deny",
                                "Grep",
                                "--deny",
                                "Glob",
                                "--deny",
                                "NotebookRead",
                                "--deny",
                                "NotebookEdit",
                                "--deny",
                                "WebFetch",
                                "--deny",
                                "WebSearch",
                                "--deny",
                                "MCPTool",
                                "--sandbox",
                                "strict",
                                "--no-plan",
                                "--no-subagents",
                                "--no-memory",
                                "--disable-web-search",
                                "--no-auto-update",
                                "--no-alt-screen",
                            ),
                            stdin=b"",
                            cwd=cwd,
                            environment=environment,
                            deadline=cast(float, request.attempt_deadline),
                            cleanup_timeout_seconds=request.call_policy.cleanup_timeout_seconds,
                            cleanup_deadline=attempt.begin_cleanup,
                            native_owner=attempt.owner,
                        )
                    )
                )
            except _ExecutableUnavailable:
                raise ModelProviderError(
                    ProviderErrorCode.CONFIGURATION_ERROR,
                    "Grok Build is not installed; install or update it and run grok login",
                    provider_id=self.provider_id,
                ) from None
        if result.returncode != 0:
            _raise_command_failure("Grok Build", result)
        payload, response_id, usage = _decode_grok_result(result.stdout, self.model)
        return _decode_model_output(
            payload,
            request=request,
            provider_id=self.provider_id,
            provider_response_id=response_id,
            usage=usage,
            id_factory=self._id_factory,
            transport="grok_build_cli",
        )

    async def _ensure_compatible_client(
        self,
        cwd: Path,
        environment: Mapping[str, str],
        attempt: AttemptLifecycle,
    ) -> None:
        if not self._compatible_client:
            help_result = await attempt.run_native(
                self._runner(
                    _Command(
                        arguments=(self._executable, "--help"),
                        stdin=b"",
                        cwd=cwd,
                        environment=environment,
                        deadline=min(
                            cast(float, attempt.request.attempt_deadline),
                            asyncio.get_running_loop().time() + 30,
                        ),
                        cleanup_timeout_seconds=attempt.policy.cleanup_timeout_seconds,
                        cleanup_deadline=attempt.begin_cleanup,
                        native_owner=attempt.owner,
                    )
                )
            )
            if help_result.returncode != 0:
                raise ModelProviderError(
                    ProviderErrorCode.CONFIGURATION_ERROR,
                    "Grok Build could not report its features; update Grok Build",
                    provider_id=self.provider_id,
                )
            help_text = help_result.stdout.decode("utf-8")
            if (
                _has_forbidden_control(help_text)
                or "streaming-json" not in help_text.split()
                or not _GROK_REQUIRED_HELP_TOKENS.issubset(help_text.split())
            ):
                raise ModelProviderError(
                    ProviderErrorCode.CONFIGURATION_ERROR,
                    "Grok Build is incompatible; update Grok Build and run grok login",
                    provider_id=self.provider_id,
                )
            self._compatible_client = True
        await self._inspect_configuration(cwd, environment, attempt)

    async def _inspect_configuration(
        self,
        cwd: Path,
        environment: Mapping[str, str],
        attempt: AttemptLifecycle,
    ) -> None:
        result = await attempt.run_native(
            self._runner(
                _Command(
                    arguments=(self._executable, "inspect", "--json"),
                    stdin=b"",
                    cwd=cwd,
                    environment=environment,
                    deadline=min(
                        cast(float, attempt.request.attempt_deadline),
                        asyncio.get_running_loop().time() + 30,
                    ),
                    cleanup_timeout_seconds=attempt.policy.cleanup_timeout_seconds,
                    cleanup_deadline=attempt.begin_cleanup,
                    native_owner=attempt.owner,
                )
            )
        )
        if result.returncode != 0:
            _raise_command_failure("Grok Build", result)
        try:
            _validate_grok_inspection(result.stdout, cwd)
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            OSError,
        ):
            raise ModelProviderError(
                ProviderErrorCode.CONFIGURATION_ERROR,
                "Grok Build configuration could not prove subscription-only isolation; remove custom configuration and extensions",
                provider_id=self.provider_id,
            ) from None


def _decode_grok_result(
    stdout: bytes,
    requested_model: str,
) -> tuple[object, str | None, ModelUsage]:
    text = stdout.decode("utf-8")
    if _has_forbidden_control(text):
        raise ValueError("Grok result contains terminal controls")
    lines = tuple(line for line in text.splitlines() if line.strip())
    if not lines or len(lines) > _MAX_STREAM_EVENTS:
        raise ValueError("Grok result has an invalid event count")
    events = tuple(_strict_json(line) for line in lines)
    if any(not isinstance(event, Mapping) for event in events):
        raise ValueError("Grok result events must be objects")
    mappings = tuple(event for event in events if isinstance(event, Mapping))
    response_length = 0
    usage_events: list[Mapping[str, object]] = []
    end_events: list[Mapping[str, object]] = []
    available_command_events = 0
    for event in mappings:
        event_type = event.get("type")
        if event_type == "text":
            chunk = event.get("data")
            if not isinstance(chunk, str):
                raise ValueError("Grok text event is invalid")
            response_length += len(chunk)
            if response_length > _MAX_RESPONSE_TEXT_CHARACTERS:
                raise ValueError("Grok response exceeds its safety bound")
        elif event_type == "thought":
            if not isinstance(event.get("data"), str):
                raise ValueError("Grok thought event is invalid")
        elif event_type == "usage":
            usage_events.append(event)
        elif event_type == "available_commands":
            available_command_events += 1
            commands = event.get("commands")
            tools = event.get("tools")
            if (
                available_command_events > 1
                or not isinstance(commands, Sequence)
                or isinstance(commands, (str, bytes))
                or tools != []
            ):
                raise ModelProviderError(
                    ProviderErrorCode.CONFIGURATION_ERROR,
                    "Grok Build exposed a native capability despite Daita's isolation boundary",
                )
        elif event_type == "end":
            end_events.append(event)
        elif event_type == "error":
            raise ModelProviderError(
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
                "Grok Build subscription turn failed",
            )
        elif event_type == "max_turns_reached" or (
            isinstance(event_type, str) and event_type.startswith("auto_compact_")
        ):
            raise ModelProviderError(
                ProviderErrorCode.OUTPUT_LIMIT,
                "Grok Build did not finish within the single-turn boundary",
            )
        else:
            raise ModelProviderError(
                ProviderErrorCode.CONFIGURATION_ERROR,
                "Grok Build emitted an unsupported event; update Grok Build and verify native tools are disabled",
            )
    if len(end_events) != 1 or end_events[0] is not mappings[-1]:
        raise ValueError("Grok result must end with one end event")
    if len(usage_events) != 1:
        raise ValueError("Grok result must contain one model usage event")
    usage_event = usage_events[0]
    if usage_event.get("stopReason") != "end_turn":
        raise ModelProviderError(
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "Grok Build subscription turn failed",
        )
    _usage_from_grok(usage_event.get("usage"))
    _bounded_response_id(usage_event.get("messageId"))
    result = end_events[0]
    if result.get("stopReason") != "end_turn":
        raise ModelProviderError(
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "Grok Build subscription turn failed",
        )
    if _nonnegative_integer(result.get("num_turns"), "turn count") != 1:
        raise ModelProviderError(
            ProviderErrorCode.CONFIGURATION_ERROR,
            "Grok Build exceeded Daita's single-turn client boundary",
        )
    model_usage = result.get("modelUsage")
    if not isinstance(model_usage, Mapping) or requested_model not in model_usage:
        raise ModelProviderError(
            ProviderErrorCode.CONFIGURATION_ERROR,
            "Grok Build did not confirm the requested built-in model; remove custom-provider configuration",
        )
    if not isinstance(model_usage[requested_model], Mapping):
        raise ValueError("Grok model usage must be an object")
    if (
        "structuredOutput" not in result
        or result.get("structuredOutputError") is not None
    ):
        raise ValueError("Grok result did not contain validated structured output")
    payload = result["structuredOutput"]
    _validate_json_tree(payload)
    response_id = _bounded_response_id(result.get("requestId", result.get("sessionId")))
    usage = _usage_from_grok(result.get("usage"))
    return payload, response_id, usage
