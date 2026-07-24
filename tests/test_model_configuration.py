import asyncio
from collections.abc import Callable
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from daita import Agent, LoopLimits
from daita.agent import AgentModelConfigurationError
from daita import terminal
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
from daita.llm.providers.mock import MockModelProvider
from daita.llm.profiles import reviewed_model_profile
from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy
from daita.security import (
    EmptySecretProvider,
    KeychainSecretProvider,
    SecretReference,
    SecretResolutionError,
)
from daita.terminal import run_terminal_application
import daita.hosting.embedded as embedded


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
    agent = await Agent.create(name, root=root)
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
    assert route.candidates[0].secret_reference is not None
    assert route.candidates[0].secret_reference.to_uri() in persisted

    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
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
        await Agent.open("atlas", root=tmp_path)


async def test_symlinked_config_fails_closed(tmp_path):
    await _create_unconfigured(tmp_path)
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    path = tmp_path / "agents" / "atlas" / "config.json"
    path.symlink_to(target)

    with pytest.raises(Exception, match="model configuration"):
        await Agent.open("atlas", root=tmp_path)


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
        await Agent.open("atlas", root=tmp_path)


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
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_profile == route.model_profile
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    ("code", "message"),
    (
        (
            ProviderErrorCode.AUTHENTICATION_ERROR,
            "The API key was rejected. Replace it and retry.",
        ),
        (
            ProviderErrorCode.MODEL_NOT_FOUND,
            "This account cannot access test-model.",
        ),
        (
            ProviderErrorCode.RATE_LIMIT_ERROR,
            "The provider rate-limited the validation request.",
        ),
        (
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "The provider could not be reached.",
        ),
        (
            ProviderErrorCode.TIMEOUT,
            "The provider did not respond before the timeout.",
        ),
        (
            ProviderErrorCode.INVALID_REQUEST,
            "The provider rejected this model configuration.",
        ),
    ),
)
async def test_terminal_maps_normalized_validation_errors(tmp_path, code, message):
    keychain = _FakeKeychain()
    provider = _provider("openai:test-model", ModelProviderError(code))
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("atlas\n1\n4\ntest-model\n8192\n1024\n"),
        output_stream=output,
        hidden_input=lambda prompt: "top-secret",
        keychain=keychain,
        model_validator=provider,
    )

    assert result == 0
    assert message in output.getvalue()
    assert "top-secret" not in output.getvalue()
    assert not (tmp_path / "agents" / "atlas" / "config.json").exists()
    assert keychain.values == {}


async def test_suggested_model_is_exactly_validated_and_only_route_is_persisted(
    tmp_path,
):
    keychain = _FakeKeychain()
    suggestion = "gemini-3.6-flash"
    validator = _provider(f"gemini:{suggestion}")
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("atlas\n3\n1\n"),
        output_stream=output,
        hidden_input=lambda prompt: "suggestion-secret",
        keychain=keychain,
        model_validator=validator,
    )

    assert result == 0
    assert validator.provider_id == f"gemini:{suggestion}"
    assert len(validator.requests) == 1
    config = (tmp_path / "agents" / "atlas" / "config.json").read_text(encoding="utf-8")
    assert f"gemini:{suggestion}" in config
    expected_profile = reviewed_model_profile(f"gemini:{suggestion}")
    assert expected_profile is not None
    assert expected_profile.context_window_tokens == 1_048_576
    assert expected_profile.max_output_tokens == 65_536
    assert expected_profile.supports_tools is True
    assert expected_profile.supports_parallel_tools is False
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_profile == expected_profile
    finally:
        await reopened.close()
    home = tmp_path / "agents" / "atlas"
    persisted = b"\n".join(
        path.read_bytes() for path in home.rglob("*") if path.is_file()
    )
    for item in terminal._MODEL_SUGGESTIONS["gemini"]:
        assert item.label.encode() not in persisted
        assert item.description.encode() not in persisted
        if item.recommendation is not None:
            assert item.recommendation.encode() not in persisted


async def test_reviewed_openai_suggestions_use_authoritative_profile_facts(tmp_path):
    for suggestion in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        profile = reviewed_model_profile(f"openai:{suggestion}")
        assert profile is not None
        assert profile.context_window_tokens == 1_050_000
        assert profile.max_output_tokens == 128_000
        assert profile.supports_tools is True
        assert profile.supports_parallel_tools is False


async def test_unreviewed_suggestion_requires_explicit_limits_before_validation(
    tmp_path,
):
    keychain = _FakeKeychain()
    suggestion = "claude-opus-4-8"
    provider_id = f"anthropic:{suggestion}"
    validator = _provider(provider_id)
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("atlas\n2\n1\n24000\n3000\n"),
        output_stream=output,
        hidden_input=lambda prompt: "suggestion-secret",
        keychain=keychain,
        model_validator=validator,
    )

    assert result == 0
    assert reviewed_model_profile(provider_id) is None
    assert "unreviewed model requires explicit hard token limits" in output.getvalue()
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_profile is not None
        assert reopened.model_profile.id == provider_id
        assert reopened.model_profile.context_window_tokens == 24_000
        assert reopened.model_profile.max_output_tokens == 3_000
        assert reopened.model_profile.supports_parallel_tools is False
    finally:
        await reopened.close()


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
        await Agent.open("atlas", root=tmp_path, keychain=keychain)


async def test_terminal_replaces_stale_profile_only_after_fresh_validation(tmp_path):
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
    candidate = document["model_route"]["candidates"][0]
    candidate["provider_id"] = "openai:gpt-5.6-sol"
    candidate["profile"]["id"] = "openai:gpt-5.6-sol"
    path.write_text(json.dumps(document), encoding="utf-8")
    before = path.read_bytes()

    cancelled_output = io.StringIO()
    cancelled = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(),
        output_stream=cancelled_output,
        keychain=keychain,
    )

    assert cancelled == 0
    assert "Setup cancelled." in cancelled_output.getvalue()
    assert path.read_bytes() == before

    output = io.StringIO()
    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("1\n1\n"),
        output_stream=output,
        hidden_input=lambda prompt: "replacement-secret",
        keychain=keychain,
        model_validator=_provider("openai:gpt-5.6-sol"),
    )

    assert result == 0
    assert "no longer meets current safety checks" in output.getvalue()
    assert path.read_bytes() != before
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_profile == reviewed_model_profile("openai:gpt-5.6-sol")
    finally:
        await reopened.close()


async def test_valid_unlisted_manual_model_uses_exact_existing_validation_path(
    tmp_path,
):
    keychain = _FakeKeychain()
    validator = _provider("gemini:private-tool-model")

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("atlas\n3\n4\nprivate-tool-model\n16384\n2048\n"),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: "manual-secret",
        keychain=keychain,
        model_validator=validator,
    )

    assert result == 0
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_route is not None
        assert (
            reopened.model_route.candidates[0].provider_id
            == "gemini:private-tool-model"
        )
        assert reopened.model_profile is not None
        assert reopened.model_profile.context_window_tokens == 16_384
        assert reopened.model_profile.max_output_tokens == 2_048
        assert reopened.model_profile.supports_parallel_tools is False
    finally:
        await reopened.close()


async def test_validation_failure_returns_to_same_provider_model_menu(tmp_path):
    keychain = _FakeKeychain()
    provider_id = "openai:retry-model"
    validator = MockModelProvider(
        (
            ModelProviderError(ProviderErrorCode.AUTHENTICATION_ERROR),
            _tool_validation_response(provider_id),
        ),
        provider_id=provider_id,
    )
    output = io.StringIO()
    secrets = iter(("first-secret", "second-secret"))

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "atlas\n"
            "1\n"
            "4\n"
            "retry-model\n"
            "8192\n"
            "1024\n"
            "4\n"
            "retry-model\n"
            "8192\n"
            "1024\n"
        ),
        output_stream=output,
        hidden_input=lambda prompt: next(secrets),
        keychain=keychain,
        model_validator=validator,
    )

    assert result == 0
    assert output.getvalue().count("Select an OpenAI model") == 2
    assert output.getvalue().count("Select a model provider") == 1
    assert "The API key was rejected. Replace it and retry." in output.getvalue()
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_route is not None
        assert reopened.model_route.candidates[0].provider_id == provider_id
    finally:
        await reopened.close()


async def test_model_menu_can_explicitly_return_to_provider_selection(tmp_path):
    keychain = _FakeKeychain()
    validator = _provider("anthropic:manual-claude")
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "atlas\n" "1\n" "5\n" "2\n" "4\n" "manual-claude\n" "8192\n" "1024\n"
        ),
        output_stream=output,
        hidden_input=lambda prompt: "anthropic-secret",
        keychain=keychain,
        model_validator=validator,
    )

    assert result == 0
    assert output.getvalue().count("Select a model provider") == 2
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_route is not None
        assert reopened.model_route.candidates[0].provider_id == (
            "anthropic:manual-claude"
        )
    finally:
        await reopened.close()


async def test_terminal_normalizes_keychain_setup_failure_without_raw_diagnostics(
    tmp_path,
):
    class FailingKeychain(_FakeKeychain):
        async def set(self, reference: SecretReference, value: str) -> None:
            del reference, value
            raise SecretResolutionError(
                "secret_provider_unavailable",
                "raw keychain diagnostic containing top-secret",
            )

    keychain = FailingKeychain()
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("atlas\n1\n4\ntest-model\n8192\n1024\n"),
        output_stream=output,
        hidden_input=lambda prompt: "top-secret",
        keychain=keychain,
        model_validator=_provider("openai:test-model"),
    )

    assert result == 0
    assert (
        "The API key could not be saved to the OS keychain. "
        "Check keychain access and retry."
    ) in output.getvalue()
    assert "raw keychain diagnostic" not in output.getvalue()
    assert "top-secret" not in output.getvalue()


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

    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
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

    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
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
        await Agent.open("beta", root=tmp_path, keychain=keychain)

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


async def test_terminal_onboarding_hides_key_before_stage_three_source_setup(tmp_path):
    keychain = _FakeKeychain()
    provider = _provider("openai:gpt-test")
    output = io.StringIO()
    secret_prompts = []

    def hidden(prompt):
        secret_prompts.append(prompt)
        return "terminal-secret"

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("atlas\n1\n4\ngpt-test\n8192\n1024\n"),
        output_stream=output,
        hidden_input=hidden,
        keychain=keychain,
        model_validator=provider,
    )

    text = output.getvalue()
    assert result == 0
    assert secret_prompts == ["API key: "]
    assert "terminal-secret" not in text
    assert "may incur a tiny API charge" in text
    assert "OpenAI · gpt-test · validated" in text
    assert "Select a data source" in text
    assert "terminal-secret" not in (
        tmp_path / "agents" / "atlas" / "config.json"
    ).read_text(encoding="utf-8")


async def test_ollama_does_not_request_an_api_key(tmp_path):
    output = io.StringIO()

    def forbidden_hidden_input(prompt):
        raise AssertionError(f"unexpected hidden prompt: {prompt}")

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("atlas\n5\n4\nllama-test\n8192\n1024\n\n"),
        output_stream=output,
        hidden_input=forbidden_hidden_input,
        keychain=_FakeKeychain(),
        model_validator=_provider("ollama:llama-test"),
    )

    assert result == 0
    assert "Ollama · llama-test · validated" in output.getvalue()


async def test_custom_terminal_provider_requires_and_persists_base_url(tmp_path):
    output = io.StringIO()
    keychain = _FakeKeychain()
    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(
            "atlas\n"
            "6\n"
            "acme\n"
            "acme-model\n"
            "8192\n"
            "1024\n"
            "https://models.acme.test/v1\n"
        ),
        output_stream=output,
        hidden_input=lambda prompt: "custom-secret",
        keychain=keychain,
        model_validator=_provider("acme:acme-model"),
    )

    assert result == 0
    assert "acme · acme-model · validated" in output.getvalue()
    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_route is not None
        assert (
            reopened.model_route.candidates[0].base_url == "https://models.acme.test/v1"
        )
    finally:
        await reopened.close()


async def test_existing_persisted_route_skips_onboarding_without_health_claim(
    tmp_path,
):
    await _create_unconfigured(tmp_path)
    keychain = _FakeKeychain()
    await _configure(tmp_path, keychain, _provider("openai:test-model"))
    output = io.StringIO()

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(""),
        output_stream=output,
        hidden_input=lambda prompt: (_ for _ in ()).throw(
            AssertionError(f"unexpected prompt: {prompt}")
        ),
        keychain=keychain,
    )

    assert result == 0
    assert "OpenAI · test-model · configured" in output.getvalue()
    assert "provider health was not checked this launch" in output.getvalue()


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


async def test_model_onboarding_interrupt_leaves_no_partial_config_or_lock(tmp_path):
    class _InterruptingInput(io.StringIO):
        def readline(self, size=-1):
            del size
            raise KeyboardInterrupt

    result = await run_terminal_application(
        root=tmp_path,
        input_stream=_InterruptingInput(""),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: "unused",
        keychain=_FakeKeychain(),
    )

    assert result == 130
    assert await Agent.list(root=tmp_path) == ()
    agent = await Agent.create("after-interrupt", root=tmp_path)
    await agent.close()


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
        await Agent.open("atlas", root=tmp_path, keychain=keychain)

    release_set.set()
    route = await configure
    with pytest.raises(asyncio.CancelledError):
        await closing

    reopened = await Agent.open("atlas", root=tmp_path, keychain=keychain)
    try:
        assert reopened.model_route == route
    finally:
        await reopened.close()


async def test_api_key_never_reaches_files_sqlite_repr_transcripts_or_output(tmp_path):
    secret = "NEVER-PERSIST-THIS-API-KEY"
    keychain = _FakeKeychain()
    provider = _provider("openai:test-model")
    output = io.StringIO()
    await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO("atlas\n1\n4\ntest-model\n8192\n1024\n"),
        output_stream=output,
        hidden_input=lambda prompt: secret,
        keychain=keychain,
        model_validator=provider,
    )

    home = tmp_path / "agents" / "atlas"
    for path in home.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    assert secret not in output.getvalue()
    reopened = await Agent.open("atlas", root=tmp_path)
    try:
        assert secret not in repr(reopened.model_route)
    finally:
        await reopened.close()
    assert all(secret not in repr(request) for request in provider.requests)
