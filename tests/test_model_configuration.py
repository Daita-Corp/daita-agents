import asyncio
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from _workspace_support import workspace_for

import daita.hosting.embedded as embedded
from daita import Agent, LoopLimits
from daita.agent import AgentModelConfigurationError
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.factory import create_model_route_provider
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import provider_has_complete_pricing
from daita.llm.providers.mock import MockModelProvider
from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy
from daita.security import (
    CredentialSession,
    EmptySecretProvider,
    EnvironmentSecretProvider,
    KeychainSecretProvider,
    SecretReference,
    SecretResolutionError,
    default_secret_provider,
)


class _FakeKeychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.events: list[tuple[str, str]] = []
        self.before_delete: Callable[[SecretReference], None] | None = None

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.events.append(("set", reference.name))
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        if self.before_delete is not None:
            self.before_delete(reference)
        self.events.append(("delete", reference.name))
        self.values.pop(reference.name, None)


class _FakeKeyringClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str | None]] = []
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        self.calls.append(("get", service_name, username, None))
        return self.values.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.calls.append(("set", service_name, username, password))
        self.values[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        self.calls.append(("delete", service_name, username, None))
        self.values.pop((service_name, username), None)


def _provider(
    provider_id: str,
    result: ModelResponse | ModelProviderError | None = None,
) -> MockModelProvider:
    item = result or _tool_validation_response(provider_id)
    script = (
        (item, item)
        if isinstance(item, ModelProviderError)
        and item.code
        in {
            ProviderErrorCode.RATE_LIMIT_ERROR,
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            ProviderErrorCode.TIMEOUT,
        }
        else (item,)
    )
    return MockModelProvider(
        script,
        provider_id=provider_id,
    )


def _tool_validation_response(provider_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="validation-call",
                name="daita_validate_tool_support",
                arguments={},
            ),
        ),
        provider_id=provider_id,
    )


async def _create_unconfigured(root: Path, name: str = "atlas") -> None:
    agent = await Agent.create(name, root=root, workspace=workspace_for(root))
    await agent.close()


async def _configure(
    root: Path,
    keychain: _FakeKeychain,
    provider: MockModelProvider,
    *,
    provider_name: str = "openai",
    model: str = "test-model",
    api_key: str | None = "secret-value",
    base_url: str | None = None,
    context_window_tokens: int = 8_192,
    max_output_tokens: int = 1_024,
) -> ModelRoute:
    agent = await Agent.open(
        "atlas",
        root=root,
        keychain=keychain,
        model_validator=provider,
        workspace=workspace_for(root),
    )
    try:
        return await agent.configure_model(
            provider=provider_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
        )
    finally:
        await agent.close()


async def test_model_configuration_round_trips_without_persisting_the_key(tmp_path):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    route = await _configure(tmp_path, keychain, _provider("openai:test-model"))

    config_path = tmp_path / "agents" / "atlas" / "config.json"
    persisted = config_path.read_text(encoding="utf-8")
    assert "secret-value" not in persisted
    document = json.loads(persisted)
    stored_profile = document["model_route"]["candidates"][0]["profile"]
    assert stored_profile["input_cost_per_million_usd"] is None
    assert stored_profile["output_cost_per_million_usd"] is None
    assert route.candidates[0].secret_reference is not None
    assert route.candidates[0].secret_reference.to_uri() in persisted

    reopened = await Agent.open(
        "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
    )
    try:
        assert reopened.model_route == route
        assert reopened.model_profile == route.model_profile
    finally:
        await reopened.close()


async def test_explicit_model_injection_precedes_persisted_config_without_rewrite(
    tmp_path,
):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    await _configure(tmp_path, keychain, _provider("openai:test-model"))
    config_path = tmp_path / "agents" / "atlas" / "config.json"
    before = config_path.read_bytes()
    injected = _provider("mock:injected")
    profile = ModelProfile(
        id="mock:injected",
        context_window_tokens=1_000,
        max_output_tokens=20,
        supports_tools=True,
    )

    agent = await Agent.open(
        "atlas",
        root=tmp_path,
        model=injected,
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        assert agent.model_profile == profile
        assert agent.model_route is None
    finally:
        await agent.close()
    assert config_path.read_bytes() == before


async def test_limits_override_retains_the_persisted_model_route(tmp_path):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    route = await _configure(
        tmp_path,
        keychain,
        _provider("openai:test-model"),
    )
    limits = LoopLimits(
        max_steps=3,
        max_total_tokens=1_234,
        max_wall_time_seconds=12.5,
    )

    reopened = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        limits=limits,
        workspace=workspace_for(tmp_path),
    )
    try:
        assert reopened.model_route == route
        assert reopened.model_profile == route.model_profile
        assert reopened._embedded._limits == limits
    finally:
        await reopened.close()


@pytest.mark.parametrize("content", (b"{", b"{}", b"x" * (64 * 1024 + 1)))
async def test_malformed_incomplete_or_oversized_config_fails_closed(
    tmp_path,
    content,
):
    await _create_unconfigured(tmp_path)
    path = tmp_path / "agents" / "atlas" / "config.json"
    path.write_bytes(content)

    with pytest.raises(Exception, match="model configuration"):
        await Agent.open("atlas", root=tmp_path, workspace=workspace_for(tmp_path))


async def test_symlinked_config_fails_closed(tmp_path):
    await _create_unconfigured(tmp_path)
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    path = tmp_path / "agents" / "atlas" / "config.json"
    path.symlink_to(target)

    with pytest.raises(Exception, match="model configuration"):
        await Agent.open("atlas", root=tmp_path, workspace=workspace_for(tmp_path))


@pytest.mark.parametrize("mutation", ("missing_reference", "terminal_control"))
async def test_incomplete_or_terminal_unsafe_route_fails_closed(tmp_path, mutation):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    await _configure(tmp_path, keychain, _provider("openai:test-model"))
    path = tmp_path / "agents" / "atlas" / "config.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    candidate = document["model_route"]["candidates"][0]
    if mutation == "missing_reference":
        candidate["secret_reference"] = None
    else:
        candidate["provider_id"] = "openai:test\x1b[2J"
        candidate["profile"]["id"] = candidate["provider_id"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Exception, match="model configuration"):
        await Agent.open("atlas", root=tmp_path, workspace=workspace_for(tmp_path))


async def test_validation_requires_one_exact_tool_call_and_uses_route_retries(tmp_path):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    provider = MockModelProvider(
        (
            ModelProviderError(ProviderErrorCode.TIMEOUT),
            _tool_validation_response("anthropic:claude-test"),
        ),
        provider_id="anthropic:claude-test",
    )

    route = await _configure(
        tmp_path,
        keychain,
        provider,
        provider_name="anthropic",
        model="claude-test",
    )

    assert route.retry_policy == RetryPolicy()
    assert provider.provider_id == "anthropic:claude-test"
    assert len(provider.requests) == 2
    request = provider.requests[0]
    assert tuple(tool.name for tool in request.tools) == (
        "daita_validate_tool_support",
    )
    assert request.response_schema is None
    assert request.allow_parallel_tool_calls is False
    assert len(request.messages) == 1
    assert len(request.messages[0].content) == 1
    validation_prompt = request.messages[0].content[0]
    assert isinstance(validation_prompt, TextBlock)
    assert len(validation_prompt.text) <= 40


async def test_text_only_validation_cannot_persist_tool_support(tmp_path):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    text_only = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="OK",
                provider_id="openai:text-only",
            ),
        ),
        provider_id="openai:text-only",
    )

    with pytest.raises(ValueError, match="tool-call support"):
        await _configure(
            tmp_path,
            keychain,
            text_only,
            model="text-only",
            context_window_tokens=9_001,
            max_output_tokens=777,
        )

    assert keychain.values == {}
    assert not (tmp_path / "agents" / "atlas" / "config.json").exists()


async def test_manual_model_retains_explicit_limits_and_disables_parallel_tools(
    tmp_path,
):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    route = await _configure(
        tmp_path,
        keychain,
        _provider("openai:manual-profile"),
        model="manual-profile",
        context_window_tokens=32_123,
        max_output_tokens=4_321,
    )

    assert route.model_profile.context_window_tokens == 32_123
    assert route.model_profile.max_output_tokens == 4_321
    assert route.model_profile.supports_tools is True
    assert route.model_profile.supports_parallel_tools is False
    reopened = await Agent.open(
        "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
    )
    try:
        assert reopened.model_profile == route.model_profile
    finally:
        await reopened.close()


async def test_reviewed_openai_suggestions_use_authoritative_profile_facts(tmp_path):
    for suggestion in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        profile = reviewed_model_profile(f"openai:{suggestion}")
        assert profile is not None
        assert profile.context_window_tokens == 1_050_000
        assert profile.max_output_tokens == 128_000
        assert profile.supports_tools is True
        assert profile.supports_parallel_tools is False
        assert profile.supports_reasoning is True


async def test_stale_profile_is_rejected_with_reconfiguration_error(tmp_path):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    await _configure(
        tmp_path,
        keychain,
        _provider("openai:legacy-model"),
        model="legacy-model",
    )
    path = tmp_path / "agents" / "atlas" / "config.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["model_route"]["candidates"][0]["profile"][
        "supports_parallel_tools"
    ] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        AgentModelConfigurationError,
        match="must be replaced",
    ):
        await Agent.open(
            "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
        )


async def test_failed_validation_preserves_previous_route_and_credential(tmp_path):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    previous = await _configure(
        tmp_path,
        keychain,
        _provider("openai:old-model"),
        model="old-model",
        api_key="old-secret",
    )
    config_path = tmp_path / "agents" / "atlas" / "config.json"
    before = config_path.read_bytes()
    old_reference = previous.candidates[0].secret_reference
    assert old_reference is not None

    with pytest.raises(ModelProviderError) as caught:
        await _configure(
            tmp_path,
            keychain,
            _provider(
                "anthropic:new-model",
                ModelProviderError(ProviderErrorCode.AUTHENTICATION_ERROR),
            ),
            provider_name="anthropic",
            model="new-model",
            api_key="new-secret",
        )

    assert caught.value.code is ProviderErrorCode.AUTHENTICATION_ERROR
    assert config_path.read_bytes() == before
    assert keychain.values == {old_reference.name: "old-secret"}


async def test_successful_replacement_deletes_old_key_only_after_config_commit(
    tmp_path,
):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    previous = await _configure(
        tmp_path,
        keychain,
        _provider("openai:old-model"),
        model="old-model",
        api_key="old-secret",
    )
    old_reference = previous.candidates[0].secret_reference
    assert old_reference is not None
    config_path = tmp_path / "agents" / "atlas" / "config.json"

    def check_committed(reference):
        if reference == old_reference:
            assert "anthropic:new-model" in config_path.read_text(encoding="utf-8")

    keychain.before_delete = check_committed
    replacement = await _configure(
        tmp_path,
        keychain,
        _provider("anthropic:new-model"),
        provider_name="anthropic",
        model="new-model",
        api_key="new-secret",
    )
    new_reference = replacement.candidates[0].secret_reference
    assert new_reference is not None

    assert old_reference.name not in keychain.values
    assert keychain.values == {new_reference.name: "new-secret"}
    assert keychain.events[-1] == ("delete", old_reference.name)


async def test_config_write_failure_preserves_old_route_and_removes_new_key(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    previous = await _configure(
        tmp_path,
        keychain,
        _provider("openai:old-model"),
        model="old-model",
        api_key="old-secret",
    )
    old_reference = previous.candidates[0].secret_reference
    assert old_reference is not None
    config_path = tmp_path / "agents" / "atlas" / "config.json"
    before = config_path.read_bytes()
    agent = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        model_validator=_provider("anthropic:new-model"),
        workspace=workspace_for(tmp_path),
    )

    def fail_write(home: Path, config) -> None:
        del home, config
        raise OSError("simulated config write failure")

    monkeypatch.setattr(embedded, "_write_model_configuration", fail_write)
    try:
        with pytest.raises(OSError, match="simulated config write failure"):
            await agent.configure_model(
                provider="anthropic",
                model="new-model",
                api_key="new-secret",
                context_window_tokens=8_192,
                max_output_tokens=1_024,
            )
    finally:
        await agent.close()

    assert config_path.read_bytes() == before
    assert keychain.values == {old_reference.name: "old-secret"}


async def test_old_credential_cleanup_failure_does_not_reverse_committed_route(
    tmp_path,
):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    previous = await _configure(
        tmp_path,
        keychain,
        _provider("openai:old-model"),
        model="old-model",
        api_key="old-secret",
    )
    old_reference = previous.candidates[0].secret_reference
    assert old_reference is not None

    def fail_old_delete(reference: SecretReference) -> None:
        if reference == old_reference:
            raise SecretResolutionError(
                "secret_provider_unavailable",
                "secret-safe cleanup failure",
            )

    keychain.before_delete = fail_old_delete
    replacement = await _configure(
        tmp_path,
        keychain,
        _provider("anthropic:new-model"),
        provider_name="anthropic",
        model="new-model",
        api_key="new-secret",
    )
    new_reference = replacement.candidates[0].secret_reference
    assert new_reference is not None
    assert keychain.values == {
        old_reference.name: "old-secret",
        new_reference.name: "new-secret",
    }

    reopened = await Agent.open(
        "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
    )
    try:
        assert reopened.model_route == replacement
    finally:
        await reopened.close()


async def test_cancellation_during_post_commit_cleanup_returns_committed_route(
    tmp_path,
):
    await _create_unconfigured(tmp_path)
    entered_delete = asyncio.Event()
    never_release = asyncio.Event()

    class _CleanupBlockingKeychain(_FakeKeychain):
        def __init__(self) -> None:
            super().__init__()
            self.blocked_reference: SecretReference | None = None

        async def delete(self, reference: SecretReference) -> None:
            if reference == self.blocked_reference:
                entered_delete.set()
                await never_release.wait()
            await super().delete(reference)

    keychain = _CleanupBlockingKeychain()
    previous = await _configure(
        tmp_path,
        keychain,
        _provider("openai:old-model"),
        model="old-model",
        api_key="old-secret",
    )
    old_reference = previous.candidates[0].secret_reference
    assert old_reference is not None
    keychain.blocked_reference = old_reference
    agent = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        model_validator=_provider("anthropic:new-model"),
        workspace=workspace_for(tmp_path),
    )
    configuring = asyncio.create_task(
        agent.configure_model(
            provider="anthropic",
            model="new-model",
            api_key="new-secret",
            context_window_tokens=8_192,
            max_output_tokens=1_024,
        )
    )
    try:
        await entered_delete.wait()
        configuring.cancel()
        replacement = await configuring
    finally:
        await agent.close()

    reopened = await Agent.open(
        "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
    )
    try:
        assert reopened.model_route == replacement
    finally:
        await reopened.close()


async def test_same_instance_run_after_configuration_requires_reopen(tmp_path):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    await _configure(
        tmp_path,
        keychain,
        _provider("openai:old-model"),
        model="old-model",
    )
    agent = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        model_validator=_provider("anthropic:new-model"),
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.configure_model(
            provider="anthropic",
            model="new-model",
            api_key="new-secret",
            context_window_tokens=32_000,
            max_output_tokens=2_000,
        )
        with pytest.raises(Exception, match="close and reopen required"):
            await agent.run("Do not use the stale provider")
    finally:
        await agent.close()


async def test_persisted_model_reference_must_belong_to_current_agent_and_provider(
    tmp_path,
):
    keychain = _FakeKeychain()
    references: dict[str, SecretReference] = {}
    for name, model in (("alpha", "alpha-model"), ("beta", "beta-model")):
        await _create_unconfigured(tmp_path, name)
        agent = await Agent.open(
            name,
            root=tmp_path,
            keychain=keychain,
            model_validator=_provider(f"openai:{model}"),
            workspace=workspace_for(tmp_path),
        )
        try:
            route = await agent.configure_model(
                provider="openai",
                model=model,
                api_key=f"{name}-secret",
                context_window_tokens=8_192,
                max_output_tokens=1_024,
            )
            reference = route.candidates[0].secret_reference
            assert reference is not None
            references[name] = reference
        finally:
            await agent.close()

    beta_path = tmp_path / "agents" / "beta" / "config.json"
    document = json.loads(beta_path.read_text(encoding="utf-8"))
    document["model_route"]["candidates"][0]["secret_reference"] = references[
        "alpha"
    ].to_uri()
    beta_path.write_text(json.dumps(document), encoding="utf-8")
    before_events = tuple(keychain.events)

    with pytest.raises(Exception, match="model configuration"):
        await Agent.open(
            "beta", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
        )

    assert tuple(keychain.events) == before_events
    assert keychain.values[references["alpha"].name] == "alpha-secret"
    assert keychain.values[references["beta"].name] == "beta-secret"


async def test_keychain_set_delete_use_bounded_daita_service_and_account():
    client = _FakeKeyringClient()
    keychain = KeychainSecretProvider(client=client)
    reference = SecretReference.keychain("agent-123:openai:credential-456")

    await keychain.set(reference, "not-printed")
    assert await keychain.resolve(reference) == "not-printed"
    await keychain.delete(reference)

    assert client.calls == [
        ("set", "daita", reference.name, "not-printed"),
        ("get", "daita", reference.name, None),
        ("delete", "daita", reference.name, None),
    ]
    assert len(reference.name) <= 256


async def test_credential_session_reuses_and_invalidates_native_secrets():
    keychain = _FakeKeychain()
    session = CredentialSession(keychain)
    reference = SecretReference.keychain("agent:postgresql:credential")

    await session.set(reference, "first-secret")
    assert await session.resolve(reference) == "first-secret"
    assert await session.resolve(reference) == "first-secret"

    await session.delete(reference)
    keychain.values[reference.name] = "replacement-secret"
    assert await session.resolve(reference) == "replacement-secret"

    await session.close()
    with pytest.raises(SecretResolutionError, match="credential session is closed"):
        await session.resolve(reference)


async def test_credential_session_coalesces_concurrent_native_reads():
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockingKeychain(_FakeKeychain):
        def __init__(self) -> None:
            super().__init__()
            self.resolve_calls = 0

        async def resolve(self, reference: SecretReference) -> str:
            self.resolve_calls += 1
            entered.set()
            await release.wait()
            return await super().resolve(reference)

    keychain = _BlockingKeychain()
    reference = SecretReference.keychain("agent:postgresql:shared")
    keychain.values[reference.name] = "one-secret"
    session = CredentialSession(keychain)

    first = asyncio.create_task(session.resolve(reference))
    await entered.wait()
    second = asyncio.create_task(session.resolve(reference))
    await asyncio.sleep(0)
    assert keychain.resolve_calls == 1

    release.set()
    assert await asyncio.gather(first, second) == ["one-secret", "one-secret"]
    assert keychain.resolve_calls == 1
    await session.close()


def test_default_provider_cannot_bypass_a_credential_session():
    session = CredentialSession(_FakeKeychain())

    provider = default_secret_provider(session)

    assert provider.providers[0] is session
    assert len(provider.providers) == 2
    assert isinstance(provider.providers[1], EnvironmentSecretProvider)
    assert not any(
        isinstance(candidate, KeychainSecretProvider)
        for candidate in provider.providers[1:]
    )


async def test_missing_keyring_uses_the_application_repair_guidance():
    keychain = KeychainSecretProvider()
    reference = SecretReference.keychain("agent:openai:test")
    real_import = __import__

    def missing_keyring(name, *args, **kwargs):
        if name == "keyring":
            raise ImportError
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=missing_keyring):
        with pytest.raises(ImportError, match="pipx reinstall daita-agents"):
            await keychain.set(reference, "secret")


async def test_missing_configured_credential_is_normalized_as_authentication():
    profile = ModelProfile(
        id="openai:test-model",
        context_window_tokens=1_000,
        max_output_tokens=20,
    )
    route = ModelRoute(
        (
            ModelRouteCandidate(
                provider_id=profile.id,
                profile=profile,
                secret_reference=SecretReference.keychain("missing-key"),
            ),
        ),
        RetryPolicy(attempts=1),
    )
    provider = create_model_route_provider(
        route,
        secret_provider=EmptySecretProvider(),
    )
    request = ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("hello"),),
            ),
        )
    )

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(request)

    assert caught.value.code is ProviderErrorCode.AUTHENTICATION_ERROR
    assert "missing-key" not in str(caught.value)


def test_lazy_model_route_exposes_pricing_without_resolving_credential():
    class _CountingSecrets(EmptySecretProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def resolve(self, reference: SecretReference) -> str:
            del reference
            self.calls += 1
            return "not-read-by-pricing-preflight"

    profile = reviewed_model_profile("openai:gpt-5.6-sol")
    assert profile is not None
    route = ModelRoute(
        (
            ModelRouteCandidate(
                provider_id=profile.id,
                profile=profile,
                secret_reference=SecretReference.keychain("hidden-reference"),
            ),
        ),
        RetryPolicy(attempts=3, backoff_seconds=0),
    )
    secrets = _CountingSecrets()
    provider = create_model_route_provider(route, secret_provider=secrets)
    request = ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("hello"),),
            ),
        )
    )

    assert provider_has_complete_pricing(provider, request) is True
    assert secrets.calls == 0


@pytest.mark.parametrize(
    ("secret_code", "provider_code", "resolve_calls"),
    (
        (
            "secret_not_found",
            ProviderErrorCode.AUTHENTICATION_ERROR,
            1,
        ),
        (
            "secret_provider_unavailable",
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            2,
        ),
        (
            "secret_provider_invalid_response",
            ProviderErrorCode.CONFIGURATION_ERROR,
            1,
        ),
    ),
)
async def test_secret_failures_keep_distinct_routing_classification(
    secret_code,
    provider_code,
    resolve_calls,
):
    class _FailingSecrets(EmptySecretProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def resolve(self, reference: SecretReference) -> str:
            del reference
            self.calls += 1
            raise SecretResolutionError(secret_code, "secret-safe failure")

    secrets = _FailingSecrets()
    profile = ModelProfile(
        id="openai:test-model",
        context_window_tokens=1_000,
        max_output_tokens=20,
    )
    route = ModelRoute(
        (
            ModelRouteCandidate(
                provider_id=profile.id,
                profile=profile,
                secret_reference=SecretReference.keychain("hidden-reference"),
            ),
        ),
        RetryPolicy(attempts=2, backoff_seconds=0),
    )
    provider = create_model_route_provider(route, secret_provider=secrets)
    request = ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("hello"),),
            ),
        )
    )

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(request)

    assert caught.value.code is provider_code
    assert secrets.calls == resolve_calls
    assert "hidden-reference" not in str(caught.value)


def test_package_and_terminal_import_keep_default_integrations_lazy():
    script = """
import builtins
import sys

blocked = {
    "anthropic",
    "asyncpg",
    "google",
    "keyring",
    "openai",
    "prompt_toolkit",
    "sqlglot",
    "textual",
}
original = builtins.__import__

def guarded(name, *args, **kwargs):
    level = kwargs.get("level", args[3] if len(args) >= 4 else 0)
    if level == 0 and name.split(".")[0] in blocked:
        raise AssertionError(f"eager integration import: {name}")
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
import daita
import daita.terminal
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


async def test_custom_provider_requires_an_explicit_base_url_before_key_storage(
    tmp_path,
):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    agent = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        model_validator=_provider("custom:test-model"),
        workspace=workspace_for(tmp_path),
    )
    try:
        with pytest.raises(ValueError, match="base URL"):
            await agent.configure_model(
                provider="custom",
                model="test-model",
                api_key="secret",
            )
    finally:
        await agent.close()
    assert keychain.events == []


async def test_cancelled_close_keeps_writer_lock_until_admitted_configuration_finishes(
    tmp_path,
):
    await _create_unconfigured(tmp_path)
    entered_set = asyncio.Event()
    release_set = asyncio.Event()

    class _BlockingKeychain(_FakeKeychain):
        async def set(self, reference: SecretReference, value: str) -> None:
            entered_set.set()
            await release_set.wait()
            await super().set(reference, value)

    keychain = _BlockingKeychain()
    agent = await Agent.open(
        "atlas",
        root=tmp_path,
        keychain=keychain,
        model_validator=_provider("openai:replacement"),
        workspace=workspace_for(tmp_path),
    )
    configure = asyncio.create_task(
        agent.configure_model(
            provider="openai",
            model="replacement",
            api_key="replacement-secret",
            context_window_tokens=8_192,
            max_output_tokens=1_024,
        )
    )
    await entered_set.wait()
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)

    with pytest.raises(Exception, match="host_active"):
        await Agent.open(
            "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
        )

    release_set.set()
    route = await configure
    with pytest.raises(asyncio.CancelledError):
        await closing

    reopened = await Agent.open(
        "atlas", root=tmp_path, keychain=keychain, workspace=workspace_for(tmp_path)
    )
    try:
        assert reopened.model_route == route
    finally:
        await reopened.close()
