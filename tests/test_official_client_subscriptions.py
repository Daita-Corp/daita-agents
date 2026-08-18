from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from pathlib import Path

import pytest

import daita.llm.providers.subscription_cli as subscription_cli
from daita import Agent
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.factory import create_llm_provider
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.pricing import CostEstimateStatus
from daita.llm.providers import (
    GeminiProvider,
    GrokBuildSubscriptionProvider,
    GrokProvider,
)
from daita.llm.providers.mock import MockModelProvider


def _request() -> ModelRequest:
    return ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.SYSTEM,
                content=(TextBlock("Keep answers grounded."),),
            ),
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("Inspect the admitted source."),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="catalog_schema",
                description="Inspect the admitted catalog schema",
                input_schema={
                    "type": "object",
                    "properties": {"source_id": {"type": "string"}},
                    "required": ["source_id"],
                    "additionalProperties": False,
                },
            ),
        ),
        allow_parallel_tool_calls=False,
    )


def _tool_envelope() -> dict[str, object]:
    return {
        "kind": "tool_calls",
        "text": "",
        "tool_calls": [
            {
                "name": "catalog_schema",
                "arguments_json": '{"source_id":"source-1"}',
            }
        ],
    }


def _grok_help() -> bytes:
    return " ".join(
        (*sorted(subscription_cli._GROK_REQUIRED_HELP_TOKENS), "streaming-json")
    ).encode()


def _grok_inspection(cwd: Path, **overrides: object) -> bytes:
    report: dict[str, object] = {
        "grokVersion": "1.2.3",
        "channel": "stable",
        "cwd": str(cwd),
        "projectRoot": None,
        "projectTrusted": True,
        "projectInstructions": [],
        "permissions": {
            "sources": [],
            "loaded": 0,
            "skipped": [],
            "mcpServerAllowlist": [],
            "marketplaceAllowlist": [],
            "managedSettingsPath": "/Library/Application Support/Grok/managed.json",
            "managedSettingsExists": False,
            "managedSettingsActive": False,
            "enforced": [],
        },
        "loginPolicy": {
            "disableApiKeyAuth": True,
            "forceLoginTeamUuid": None,
            "apiKeyAuthDisabled": True,
        },
        "hooks": [],
        "skills": [],
        "agents": [],
        "plugins": [],
        "marketplaces": [],
        "mcpServers": [],
        "lspServers": [],
        "configSources": {"layers": []},
        "externalCompat": {},
    }
    report.update(overrides)
    return json.dumps(report).encode()


def _grok_setup_result(
    command: subscription_cli._Command,
) -> subscription_cli._CompletedCommand | None:
    if command.arguments[-1] == "--help":
        return subscription_cli._CompletedCommand(0, _grok_help(), b"")
    if command.arguments[1:] == ("inspect", "--json"):
        return subscription_cli._CompletedCommand(0, _grok_inspection(command.cwd), b"")
    return None


def _grok_output(
    payload: object | None = None,
    *,
    available_tools: list[object] | None = None,
    structured_output_error: object = None,
) -> bytes:
    structured_output = payload if payload is not None else _tool_envelope()
    response = json.dumps(structured_output)
    midpoint = len(response) // 2
    events = (
        {
            "type": "available_commands",
            "commands": ["help"],
            "tools": available_tools if available_tools is not None else [],
        },
        {"type": "thought", "data": "Producing the bounded response."},
        {"type": "text", "data": response[:midpoint]},
        {"type": "text", "data": response[midpoint:]},
        {
            "type": "usage",
            "messageId": "grok-message-1",
            "stopReason": "end_turn",
            "usage": {
                "input_tokens": 20,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "output_tokens": 7,
                "reasoning_tokens": 4,
            },
            "signature": "bounded-signature",
        },
        {
            "type": "end",
            "stopReason": "end_turn",
            "sessionId": "grok-session-1",
            "requestId": "grok-request-1",
            "num_turns": 1,
            "usage": {
                "input_tokens": 20,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "output_tokens": 7,
                "reasoning_tokens": 4,
            },
            "modelUsage": {"grok-4.5": {"inputTokens": 20, "outputTokens": 7}},
            "total_cost_usd": 999,
            "structuredOutput": structured_output,
            "structuredOutputError": structured_output_error,
        },
    )
    return b"\n".join(json.dumps(event).encode() for event in events)


def test_factory_keeps_subscription_and_api_key_routes_separate():
    grok_build = create_llm_provider("grok-build:grok-4.5")
    gemini_api = create_llm_provider("gemini:gemini-3.6-flash", api_key="key")
    grok_api = create_llm_provider("grok:grok-4.5", api_key="key")

    assert isinstance(grok_build, GrokBuildSubscriptionProvider)
    assert isinstance(gemini_api, GeminiProvider)
    assert isinstance(grok_api, GrokProvider)
    with pytest.raises(ValueError, match="official client's subscription login"):
        create_llm_provider("grok-build:grok-4.5", api_key="wrong")
    with pytest.raises(ValueError, match="fixed endpoint"):
        create_llm_provider("grok-build:grok-4.5", base_url="https://api.x.ai/v1")


async def test_grok_build_uses_prompt_file_oauth_and_no_native_capabilities(
    monkeypatch,
):
    monkeypatch.delenv("GROK_HOME", raising=False)
    monkeypatch.setenv("HOME", "/safe/home")
    monkeypatch.setenv("XAI_API_KEY", "api-billing-must-not-win")
    monkeypatch.setenv("GROK_CLI_CHAT_PROXY_BASE_URL", "https://custom.invalid")
    monkeypatch.setenv("GROK_AUTH_PROVIDER_COMMAND", "print-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    commands: list[subscription_cli._Command] = []
    prompt_mode: int | None = None
    prompt_content = b""

    async def run(command):
        nonlocal prompt_content, prompt_mode
        commands.append(command)
        if (setup := _grok_setup_result(command)) is not None:
            return setup
        prompt_path = Path(
            command.arguments[command.arguments.index("--prompt-file") + 1]
        )
        prompt_content = prompt_path.read_bytes()
        prompt_mode = stat.S_IMODE(prompt_path.stat().st_mode)
        return subscription_cli._CompletedCommand(0, _grok_output(), b"")

    provider = GrokBuildSubscriptionProvider("grok-4.5", runner=run)
    response = await provider.generate(_request())

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.provider_id == "grok-build:grok-4.5"
    assert response.provider_response_id == "grok-request-1"
    assert dict(response.tool_calls[0].arguments) == {"source_id": "source-1"}
    assert response.usage.input_tokens == 25
    assert response.usage.output_tokens == 7
    assert response.usage.reasoning_tokens == 4
    assert response.usage.cache_read_tokens == 3
    assert response.usage.cache_write_tokens == 2
    assert response.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE
    assert response.usage.cost_estimate.code == "subscription_billing"
    assert [item.arguments[1:] for item in commands[:2]] == [
        ("--help",),
        ("inspect", "--json"),
    ]
    command = commands[2]
    assert command.stdin == b""
    assert prompt_mode == 0o600
    assert b"DAITA REQUEST DOCUMENT" in prompt_content
    assert subscription_cli._CONTROL_PROMPT.encode() not in prompt_content
    assert all("Inspect the admitted source" not in item for item in command.arguments)
    for flag in (
        "--disable-web-search",
        "--disallowed-tools",
        "--max-turns",
        "--json-schema",
        "--no-auto-update",
        "--no-memory",
        "--no-plan",
        "--no-subagents",
        "--sandbox",
        "--system-prompt-override",
        "--tools",
        "--verbatim",
    ):
        assert flag in command.arguments
    assert command.arguments[command.arguments.index("--tools") + 1] == ""
    assert command.arguments[command.arguments.index("--max-turns") + 1] == "1"
    assert json.loads(
        command.arguments[command.arguments.index("--json-schema") + 1]
    ) == subscription_cli._response_envelope_schema(_request())
    assert (
        command.arguments[command.arguments.index("--output-format") + 1]
        == "streaming-json"
    )
    assert command.environment["GROK_DISABLE_AUTOUPDATER"] == "1"
    assert command.environment["GROK_DISABLE_API_KEY_AUTH"] == "1"
    assert command.environment["GROK_MEMORY"] == "0"
    assert command.environment["GROK_SUBAGENTS"] == "0"
    assert command.environment["GROK_TELEMETRY_ENABLED"] == "0"
    assert command.environment["GROK_WEB_FETCH"] == "0"
    assert command.environment["GROK_WORKFLOWS"] == "0"
    assert command.environment["HOME"] != "/safe/home"
    assert command.environment["GROK_HOME"] == "/safe/home/.grok"
    for key in (
        "XAI_API_KEY",
        "GROK_CLI_CHAT_PROXY_BASE_URL",
        "GROK_AUTH_PROVIDER_COMMAND",
        "DATABASE_URL",
    ):
        assert key not in command.environment


@pytest.mark.parametrize(
    "unsafe_inspection",
    (
        {
            "loginPolicy": {
                "disableApiKeyAuth": False,
                "forceLoginTeamUuid": None,
                "apiKeyAuthDisabled": False,
            }
        },
        {
            "configSources": {
                "layers": [
                    {
                        "role": "user",
                        "path": "/safe/home/.grok/config.toml",
                    }
                ]
            }
        },
        {
            "plugins": [
                {
                    "name": "custom",
                    "scope": "user",
                    "path": "/safe/plugin",
                    "enabled": True,
                    "provides": {},
                }
            ]
        },
    ),
)
async def test_grok_rejects_unsafe_configuration_before_inference(
    unsafe_inspection,
):
    commands: list[subscription_cli._Command] = []

    async def run(command):
        commands.append(command)
        if command.arguments[-1] == "--help":
            return subscription_cli._CompletedCommand(0, _grok_help(), b"")
        if command.arguments[1:] == ("inspect", "--json"):
            return subscription_cli._CompletedCommand(
                0, _grok_inspection(command.cwd, **unsafe_inspection), b""
            )
        raise AssertionError("inference must not start after unsafe inspection")

    provider = GrokBuildSubscriptionProvider("grok-4.5", runner=run)
    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is ProviderErrorCode.CONFIGURATION_ERROR
    assert [item.arguments[1:] for item in commands] == [
        ("--help",),
        ("inspect", "--json"),
    ]


async def test_grok_rechecks_configuration_before_every_inference():
    inspections = 0
    inferences = 0

    async def run(command):
        nonlocal inferences, inspections
        if command.arguments[-1] == "--help":
            return subscription_cli._CompletedCommand(0, _grok_help(), b"")
        if command.arguments[1:] == ("inspect", "--json"):
            inspections += 1
            overrides = (
                {}
                if inspections == 1
                else {
                    "configSources": {
                        "layers": [
                            {
                                "role": "user",
                                "path": "/safe/home/.grok/config.toml",
                            }
                        ]
                    }
                }
            )
            return subscription_cli._CompletedCommand(
                0, _grok_inspection(command.cwd, **overrides), b""
            )
        inferences += 1
        return subscription_cli._CompletedCommand(0, _grok_output(), b"")

    provider = GrokBuildSubscriptionProvider("grok-4.5", runner=run)
    await provider.generate(_request())
    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is ProviderErrorCode.CONFIGURATION_ERROR
    assert inspections == 2
    assert inferences == 1


def test_grok_subscription_rejects_unreviewed_model_identity():
    with pytest.raises(ValueError, match="reviewed built-in model"):
        GrokBuildSubscriptionProvider("custom-provider/grok-4.5")


async def test_grok_uses_native_schema_and_terminal_structured_output():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    request = ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("Return an answer."),),
            ),
        ),
        response_schema=schema,
    )
    inference: subscription_cli._Command | None = None

    async def run(command):
        nonlocal inference
        if (setup := _grok_setup_result(command)) is not None:
            return setup
        inference = command
        return subscription_cli._CompletedCommand(
            0,
            _grok_output({"answer": "grounded"}),
            b"",
        )

    response = await GrokBuildSubscriptionProvider("grok-4.5", runner=run).generate(
        request
    )

    assert response.text == '{"answer":"grounded"}'
    assert inference is not None
    assert (
        json.loads(inference.arguments[inference.arguments.index("--json-schema") + 1])
        == schema
    )


@pytest.mark.parametrize(
    ("output", "code"),
    (
        (
            _grok_output(available_tools=[{"name": "read_file"}]),
            ProviderErrorCode.CONFIGURATION_ERROR,
        ),
        (
            _grok_output(structured_output_error="schema validation failed"),
            ProviderErrorCode.MALFORMED_RESPONSE,
        ),
    ),
)
async def test_grok_rejects_native_tools_and_failed_structured_output(output, code):
    async def run(command):
        if (setup := _grok_setup_result(command)) is not None:
            return setup
        return subscription_cli._CompletedCommand(0, output, b"")

    provider = GrokBuildSubscriptionProvider("grok-4.5", runner=run)
    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is code


async def test_subscription_client_rejects_terminal_control_output():
    async def run(command):
        if (setup := _grok_setup_result(command)) is not None:
            return setup
        return subscription_cli._CompletedCommand(
            0,
            _grok_output(
                {"kind": "message", "text": "unsafe\u001b[2J", "tool_calls": []}
            ),
            b"",
        )

    provider = GrokBuildSubscriptionProvider("grok-4.5", runner=run)
    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE


def test_subscription_json_and_response_bounds_fail_closed():
    nested: object = "leaf"
    for _ in range(subscription_cli._MAX_JSON_DEPTH + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="depth bound"):
        subscription_cli._strict_json(json.dumps(nested))
    with pytest.raises(ValueError, match="terminal controls"):
        subscription_cli._strict_json('{"value":"\\u001b[2J"}')

    request = ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("hello"),),
            ),
        )
    )
    with pytest.raises(ValueError, match="safety bound"):
        subscription_cli._decode_model_output(
            {
                "kind": "message",
                "text": "x" * (subscription_cli._MAX_RESPONSE_TEXT_CHARACTERS + 1),
                "tool_calls": [],
            },
            request=request,
            provider_id="grok-build:test",
            provider_response_id=None,
            usage=subscription_cli.ModelUsage(),
            id_factory=lambda prefix: prefix,
            transport="test",
        )


@pytest.mark.parametrize(
    "output",
    (
        b"\n".join(
            (
                b'{"type":"tool_use","name":"read_file"}',
                _grok_output(),
            )
        ),
        b"\n".join(
            json.dumps(event).encode()
            for event in (
                {"type": "text", "data": json.dumps(_tool_envelope())},
                {
                    "type": "usage",
                    "messageId": "message",
                    "stopReason": "end_turn",
                    "usage": {},
                },
                {
                    "type": "end",
                    "stopReason": "end_turn",
                    "sessionId": "session",
                    "requestId": "request",
                    "num_turns": 1,
                    "usage": {},
                    "modelUsage": {"custom-provider": {}},
                },
            )
        ),
    ),
)
async def test_grok_rejects_native_events_and_unconfirmed_custom_models(output):
    async def run(command):
        if (setup := _grok_setup_result(command)) is not None:
            return setup
        return subscription_cli._CompletedCommand(0, output, b"")

    provider = GrokBuildSubscriptionProvider("grok-4.5", runner=run)
    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is ProviderErrorCode.CONFIGURATION_ERROR


@pytest.mark.parametrize(
    ("provider", "stderr", "code"),
    (
        ("grok", b"run grok login", ProviderErrorCode.AUTHENTICATION_ERROR),
        ("grok", b"usage limit reached", ProviderErrorCode.RATE_LIMIT_ERROR),
        ("grok", b"configured model not available", ProviderErrorCode.MODEL_NOT_FOUND),
    ),
)
async def test_subscription_client_failures_are_normalized(provider, stderr, code):
    async def run(command):
        if (setup := _grok_setup_result(command)) is not None:
            return setup
        return subscription_cli._CompletedCommand(1, b"", stderr)

    client = GrokBuildSubscriptionProvider("grok-4.5", runner=run)
    with pytest.raises(ModelProviderError) as caught:
        await client.generate(_request())

    assert caught.value.code is code
    assert stderr.decode() not in str(caught.value)


async def test_missing_and_incompatible_clients_are_actionable():
    async def missing(command):
        raise subscription_cli._ExecutableUnavailable(command.arguments[0])

    async def old_grok(command):
        return subscription_cli._CompletedCommand(0, b"--output-format", b"")

    for client in (GrokBuildSubscriptionProvider("grok-4.5", runner=missing),):
        with pytest.raises(ModelProviderError) as caught:
            await client.generate(_request())
        assert caught.value.code is ProviderErrorCode.CONFIGURATION_ERROR
        assert "install" in str(caught.value).casefold()
    for client in (GrokBuildSubscriptionProvider("grok-4.5", runner=old_grok),):
        with pytest.raises(ModelProviderError) as caught:
            await client.generate(_request())
        assert caught.value.code is ProviderErrorCode.CONFIGURATION_ERROR
        assert "update" in str(caught.value).casefold()


async def test_grok_subscription_total_attempt_timeout_is_normalized():
    async def hang(command):
        if (setup := _grok_setup_result(command)) is not None:
            return setup
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    provider = GrokBuildSubscriptionProvider(
        "grok-4.5",
        runner=hang,
        timeout_seconds=0.01,
    )

    with pytest.raises(ModelProviderError) as caught:
        await asyncio.wait_for(provider.generate(_request()), timeout=0.25)

    assert caught.value.code is ProviderErrorCode.TIMEOUT
    assert caught.value.provider_id == "grok-build:grok-4.5"


def _process_exists(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_for_file(path: Path) -> int:
    async with asyncio.timeout(2):
        while not path.exists():
            await asyncio.sleep(0.01)
    return int(path.read_text(encoding="ascii"))


async def _wait_for_process_exit(process_id: int) -> None:
    async with asyncio.timeout(2):
        while _process_exists(process_id):
            await asyncio.sleep(0.01)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group assertion")
async def test_command_timeout_and_cancellation_terminate_subprocess_trees(tmp_path):
    script = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    timeout_pid_path = tmp_path / "timeout.pid"
    timeout_command = subscription_cli._Command(
        arguments=(sys.executable, "-c", script, str(timeout_pid_path)),
        stdin=b"",
        cwd=tmp_path,
        environment={"PATH": os.environ.get("PATH", "")},
        timeout_seconds=0.2,
    )

    with pytest.raises(ModelProviderError) as caught:
        await subscription_cli._run_command(timeout_command)
    timeout_child = await _wait_for_file(timeout_pid_path)
    await _wait_for_process_exit(timeout_child)
    assert caught.value.code is ProviderErrorCode.TIMEOUT

    cancel_pid_path = tmp_path / "cancel.pid"
    cancel_command = subscription_cli._Command(
        arguments=(sys.executable, "-c", script, str(cancel_pid_path)),
        stdin=b"",
        cwd=tmp_path,
        environment={"PATH": os.environ.get("PATH", "")},
        timeout_seconds=60,
    )
    task = asyncio.create_task(subscription_cli._run_command(cancel_command))
    cancel_child = await _wait_for_file(cancel_pid_path)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_for_process_exit(cancel_child)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group assertion")
async def test_command_timeout_terminates_descendants_after_leader_exit(tmp_path):
    child_pid_path = tmp_path / "orphaned-child.pid"
    script = (
        "import pathlib,subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
    )
    command = subscription_cli._Command(
        arguments=(sys.executable, "-c", script, str(child_pid_path)),
        stdin=b"",
        cwd=tmp_path,
        environment={"PATH": os.environ.get("PATH", "")},
        timeout_seconds=0.2,
    )

    with pytest.raises(ModelProviderError) as caught:
        await subscription_cli._run_command(command)
    child = await _wait_for_file(child_pid_path)
    await _wait_for_process_exit(child)
    assert caught.value.code is ProviderErrorCode.TIMEOUT


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group assertion")
async def test_command_output_limit_terminates_subprocess_tree(tmp_path):
    child_pid_path = tmp_path / "output-limit.pid"
    script = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid));"
        f"sys.stdout.buffer.write(b'x'*{subscription_cli._MAX_STDOUT_BYTES + 1});"
        "sys.stdout.buffer.flush();time.sleep(60)"
    )
    command = subscription_cli._Command(
        arguments=(sys.executable, "-c", script, str(child_pid_path)),
        stdin=b"",
        cwd=tmp_path,
        environment={"PATH": os.environ.get("PATH", "")},
        timeout_seconds=5,
    )

    with pytest.raises(ModelProviderError) as caught:
        await subscription_cli._run_command(command)
    child = await _wait_for_file(child_pid_path)
    await _wait_for_process_exit(child)
    assert caught.value.code is ProviderErrorCode.OUTPUT_LIMIT


async def test_subscription_route_persists_without_credentials(
    tmp_path,
):
    provider = "grok-build"
    model = "grok-4.5"
    created = await Agent.create("atlas", root=tmp_path)
    await created.close()
    provider_id = f"{provider}:{model}"
    validator = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="validation-call",
                        name="daita_validate_tool_support",
                        arguments={},
                    ),
                ),
                provider_id=provider_id,
            ),
        ),
        provider_id=provider_id,
    )
    agent = await Agent.open("atlas", root=tmp_path, model_validator=validator)
    try:
        route = await agent.configure_model(
            provider=provider,
            model=model,
            context_window_tokens=500_000,
            max_output_tokens=8_192,
        )
    finally:
        await agent.close()

    assert route.candidates[0].secret_reference is None
    persisted = (tmp_path / "agents" / "atlas" / "config.json").read_text(
        encoding="utf-8"
    )
    stored = json.loads(persisted)
    assert stored["model_route"]["candidates"][0]["secret_reference"] is None
    reopened = await Agent.open("atlas", root=tmp_path)
    try:
        assert reopened.model_route is not None
        assert reopened.model_route.candidates[0].provider_id == provider_id
    finally:
        await reopened.close()
