from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import cast

import openai
import pytest

import daita.llm.providers.codex as codex_provider
import daita.llm.providers.subscription_cli as claude_cli
import daita.llm.subscription_auth as subscription_auth
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
    ClaudeCodeSubscriptionProvider,
    CodexSubscriptionProvider,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.subscription_auth import (
    CodexDevicePrompt,
    CodexOAuthCredential,
    login_codex_subscription,
)
from daita.security import SecretReference


def _jwt(account_id: str = "account-1", *, expires_at: int | None = None) -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return ".".join(
        (
            encode({"alg": "none"}),
            encode(
                {
                    "exp": expires_at or int(time.time()) + 3_600,
                    "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
                }
            ),
            encode("signature"),
        )
    )


def _credential(*, expired: bool = False) -> CodexOAuthCredential:
    expiry = time.time() - 10 if expired else time.time() + 3_600
    return CodexOAuthCredential(
        access_token=_jwt(expires_at=int(expiry)),
        refresh_token="refresh-token",
        expires_at=expiry,
        account_id="account-1",
    )


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


class _Stream:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict[str, object]:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class _Responses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> _Stream:
        self.arguments = kwargs
        response: dict[str, object] = {
            "id": "response-1",
            "model": "gpt-test",
            "output": [],
            "service_tier": None,
            "status": "completed",
            "usage": {
                "input_tokens": 30,
                "input_tokens_details": {
                    "cached_tokens": 10,
                    "cache_write_tokens": 0,
                },
                "output_tokens": 8,
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        }
        return _Stream(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "provider-call-1",
                        "name": "catalog_schema",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": '{"source_id":"source-1"}',
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "provider-call-1",
                        "name": "catalog_schema",
                        "arguments": '{"source_id":"source-1"}',
                    },
                },
                {"type": "response.completed", "response": response},
            ]
        )


class _Client:
    def __init__(self) -> None:
        self.responses = _Responses()


class _HangingResponses:
    async def create(self, **kwargs: object) -> object:
        del kwargs
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _HangingClient:
    def __init__(self) -> None:
        self.responses = _HangingResponses()


async def test_codex_subscription_uses_direct_responses_and_daita_tool_loop():
    client = _Client()
    provider = CodexSubscriptionProvider(
        "gpt-test",
        credential=_credential().to_secret(),
        max_output_tokens=2_048,
        client=client,
    )

    response = await provider.generate(_request())

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.provider_id == "codex:gpt-test"
    assert response.provider_response_id == "response-1"
    assert response.tool_calls[0].name == "catalog_schema"
    assert dict(response.tool_calls[0].arguments) == {"source_id": "source-1"}
    assert dict(response.provider_metadata) == {
        "auth_mode": "subscription",
        "transport": "chatgpt_responses",
    }
    assert response.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE

    arguments = client.responses.arguments
    assert arguments is not None
    assert arguments["stream"] is True
    assert arguments["instructions"] == "Keep answers grounded."
    assert arguments["input"] == [
        {"role": "user", "content": "Inspect the admitted source."}
    ]
    assert "service_tier" not in arguments
    assert "max_output_tokens" not in arguments
    assert arguments["tool_choice"] == "auto"
    assert arguments["parallel_tool_calls"] is False


def test_codex_default_client_uses_bounded_transport_without_sdk_retries(monkeypatch):
    captured: dict[str, object] = {}
    client = _Client()

    def construct(**kwargs: object) -> _Client:
        captured.update(kwargs)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", construct)
    provider = CodexSubscriptionProvider(
        "gpt-test",
        credential=_credential().to_secret(),
    )

    assert provider.client is client
    timeout = captured["timeout"]
    assert isinstance(timeout, openai.Timeout)
    assert timeout.connect == 5.0
    assert timeout.read == 45.0
    assert timeout.write == 30.0
    assert timeout.pool == 5.0
    assert captured["max_retries"] == 0


async def test_codex_total_attempt_timeout_is_normalized(monkeypatch):
    monkeypatch.setattr(codex_provider, "_CODEX_ATTEMPT_TIMEOUT_SECONDS", 0.01)
    provider = CodexSubscriptionProvider(
        "gpt-test",
        credential=_credential().to_secret(),
        client=_HangingClient(),
    )

    with pytest.raises(ModelProviderError) as caught:
        await asyncio.wait_for(provider.generate(_request()), timeout=0.25)

    assert caught.value.code is ProviderErrorCode.TIMEOUT
    assert caught.value.provider_id == "codex:gpt-test"


async def test_codex_refresh_is_persisted_before_using_rotated_token(monkeypatch):
    original = _credential(expired=True)
    refreshed = _credential()
    persisted: list[str] = []

    async def refresh(value: CodexOAuthCredential) -> CodexOAuthCredential:
        assert value == original
        return refreshed

    async def persist(value: str) -> None:
        persisted.append(value)

    monkeypatch.setattr(codex_provider, "refresh_codex_subscription", refresh)
    provider = CodexSubscriptionProvider(
        "gpt-test",
        credential=original.to_secret(),
        credential_updater=persist,
        client=_Client(),
    )

    await provider.generate(_request())

    assert persisted == [refreshed.to_secret()]


async def test_claude_subscription_remains_an_official_client_transport(monkeypatch):
    monkeypatch.setenv("HOME", "/safe/home")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/safe/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-billing-must-not-win")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")
    monkeypatch.setenv("POSTGRES_PASSWORD", "unrelated-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "unmanaged-oauth-secret")
    commands = []

    async def run(command):
        commands.append(command)
        return claude_cli._CompletedCommand(
            0,
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "session_id": "claude-session-1",
                    "structured_output": {
                        "kind": "tool_calls",
                        "text": "",
                        "tool_calls": [
                            {
                                "name": "catalog_schema",
                                "arguments_json": '{"source_id":"source-1"}',
                            }
                        ],
                    },
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                }
            ).encode(),
            b"",
        )

    provider = ClaudeCodeSubscriptionProvider("sonnet", runner=run)

    response = await provider.generate(_request())

    assert response.provider_id == "claude-code:sonnet"
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].name == "catalog_schema"
    assert dict(response.provider_metadata) == {
        "auth_mode": "subscription",
        "transport": "claude_code_cli",
    }
    assert commands[0].arguments[0] == "claude"
    assert commands[0].environment["HOME"] == "/safe/home"
    assert commands[0].environment["HTTPS_PROXY"] == "https://proxy.invalid"
    assert commands[0].environment["CLAUDE_CONFIG_DIR"] == "/safe/claude"
    assert "ANTHROPIC_API_KEY" not in commands[0].environment
    assert "OPENAI_API_KEY" not in commands[0].environment
    assert "POSTGRES_PASSWORD" not in commands[0].environment
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in commands[0].environment
    assert commands[0].environment["DISABLE_TELEMETRY"] == "1"
    assert commands[0].environment["DISABLE_ERROR_REPORTING"] == "1"
    assert commands[0].environment["DISABLE_BUG_COMMAND"] == "1"


async def test_claude_subscription_total_attempt_timeout_is_normalized():
    async def hang(command):
        del command
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    provider = ClaudeCodeSubscriptionProvider(
        "claude-test",
        runner=hang,
        timeout_seconds=0.01,
    )

    with pytest.raises(ModelProviderError) as caught:
        await asyncio.wait_for(provider.generate(_request()), timeout=0.25)

    assert caught.value.code is ProviderErrorCode.TIMEOUT
    assert caught.value.provider_id == "claude-code:claude-test"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("access_token", "unsafe\naccess"),
        ("refresh_token", "unsafe\rrefresh"),
        ("account_id", "unsafe\naccount"),
        ("expires_at", float("inf")),
    ),
)
def test_codex_credential_rejects_header_injection_and_nonfinite_expiry(field, value):
    arguments = {
        "access_token": _jwt(),
        "refresh_token": "refresh-token",
        "expires_at": time.time() + 3_600,
        "account_id": "account-1",
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        CodexOAuthCredential(
            access_token=cast(str, arguments["access_token"]),
            refresh_token=cast(str, arguments["refresh_token"]),
            expires_at=cast(float, arguments["expires_at"]),
            account_id=cast(str, arguments["account_id"]),
        )


async def test_codex_device_login_rejects_terminal_controls_before_display(
    monkeypatch,
):
    async def post_json(url: str, body: object):
        return subscription_auth._HttpResult(
            200,
            json.dumps(
                {
                    "device_auth_id": "device-1",
                    "user_code": "ABCD-EFGH\x1b[2J",
                }
            ).encode(),
        )

    monkeypatch.setattr(subscription_auth, "_post_json", post_json)
    prompts: list[CodexDevicePrompt] = []

    with pytest.raises(ModelProviderError) as caught:
        await login_codex_subscription(on_verification=prompts.append)

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert prompts == []


async def test_codex_device_login_returns_a_daita_owned_credential(monkeypatch):
    access = _jwt("account-device")
    results = iter(
        (
            subscription_auth._HttpResult(
                200,
                json.dumps(
                    {
                        "device_auth_id": "device-1",
                        "user_code": "ABCD-EFGH",
                        "interval": 1,
                    }
                ).encode(),
            ),
            subscription_auth._HttpResult(
                200,
                json.dumps(
                    {
                        "authorization_code": "authorization-1",
                        "code_verifier": "verifier-1",
                    }
                ).encode(),
            ),
        )
    )
    forms: list[dict[str, str]] = []

    async def post_json(url: str, body: object):
        return next(results)

    async def post_form(url: str, body: object):
        forms.append(dict(cast(dict[str, str], body)))
        return subscription_auth._HttpResult(
            200,
            json.dumps(
                {
                    "access_token": access,
                    "refresh_token": "refresh-device",
                    "expires_in": 3_600,
                }
            ).encode(),
        )

    monkeypatch.setattr(subscription_auth, "_post_json", post_json)
    monkeypatch.setattr(subscription_auth, "_post_form", post_form)
    prompts: list[CodexDevicePrompt] = []

    encoded = await login_codex_subscription(on_verification=prompts.append)
    decoded = CodexOAuthCredential.from_secret(encoded)

    assert prompts[0].verification_url == "https://auth.openai.com/codex/device"
    assert prompts[0].user_code == "ABCD-EFGH"
    assert decoded.account_id == "account-device"
    assert decoded.refresh_token == "refresh-device"
    assert forms == [
        {
            "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
            "code": "authorization-1",
            "code_verifier": "verifier-1",
            "grant_type": "authorization_code",
            "redirect_uri": "https://auth.openai.com/deviceauth/callback",
        }
    ]


def test_factory_requires_subscription_credential_without_api_key_fallback():
    secret = _credential().to_secret()
    codex = create_llm_provider("codex:gpt-5.6-sol", subscription_credential=secret)
    claude = create_llm_provider("claude-code:claude-sonnet-5")

    assert isinstance(codex, CodexSubscriptionProvider)
    assert isinstance(claude, ClaudeCodeSubscriptionProvider)
    with pytest.raises(ValueError, match="Daita subscription login"):
        create_llm_provider("codex:gpt-5.6-sol")
    with pytest.raises(ValueError, match="does not accept an API key"):
        create_llm_provider("codex:gpt-5.6-sol", api_key="wrong")


class _FakeKeychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        self.values.pop(reference.name, None)


async def test_codex_route_persists_only_a_keychain_reference(tmp_path):
    created = await Agent.create("atlas", root=tmp_path)
    await created.close()
    keychain = _FakeKeychain()
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
                provider_id="codex:gpt-5.6-sol",
            ),
        ),
        provider_id="codex:gpt-5.6-sol",
    )
    secret = _credential().to_secret()
    agent = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        model_validator=validator,
    )
    try:
        route = await agent.configure_model(
            provider="codex",
            model="gpt-5.6-sol",
            subscription_credential=secret,
        )
    finally:
        await agent.close()

    reference = route.candidates[0].secret_reference
    assert reference is not None
    assert keychain.values[reference.name] == secret
    persisted = (tmp_path / "agents" / "atlas" / "config.json").read_text()
    assert secret not in persisted
    assert reference.to_uri() in persisted


async def test_terminal_codex_onboarding_runs_device_login_without_api_key():
    class _Agent:
        def __init__(self) -> None:
            self.arguments: dict[str, object] | None = None

        def model_requires_explicit_limits(self, *, provider: str, model: str) -> bool:
            return False

        async def configure_model(self, **kwargs: object) -> None:
            self.arguments = kwargs

        async def authenticate_model_subscription(
            self, *, provider, on_verification, on_progress
        ):
            assert provider == "codex"
            on_progress("Requesting a ChatGPT device code")
            on_verification(
                subscription_auth.CodexDevicePrompt(
                    "https://auth.openai.com/codex/device", "ABCD-EFGH", 900
                )
            )
            return _credential().to_secret()

    agent = _Agent()
    prompts: list[str] = []

    def on_verification(prompt) -> None:
        prompts.append(prompt.user_code)

    credential = await agent.authenticate_model_subscription(
        provider="codex",
        on_verification=on_verification,
        on_progress=lambda _message: None,
    )
    await agent.configure_model(
        provider="codex",
        model="gpt-5.6-sol",
        subscription_credential=credential,
    )

    assert agent.arguments is not None
    assert agent.arguments.get("api_key") is None
    assert agent.arguments["subscription_credential"] is not None
    assert "ABCD-EFGH" in prompts
