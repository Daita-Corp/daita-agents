"""Offline qualification of subscription-billed live-test provider selection."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import pytest

from daita.llm.models import ModelUsage
from daita.llm.subscription_auth import CodexOAuthCredential
from daita.security import KeychainSecretProvider, SecretReference
from tests.support import (
    live_billing,
    mcp_routine_harness as phase_f_harness,
    postgresql_live_harness as postgres_harness,
)

pytestmark = pytest.mark.unit

_SECRET_MARKER = "refresh-token-that-must-never-be-printed"


class _Keyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.values.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.values[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        self.values.pop((service_name, username), None)


def _credential() -> str:
    return CodexOAuthCredential(
        access_token="access-token",
        refresh_token=_SECRET_MARKER,
        expires_at=time.time() + 3_600,
        account_id="account-1",
    ).to_secret()


@pytest.fixture
def keyring_client(monkeypatch: pytest.MonkeyPatch) -> _Keyring:
    client = _Keyring()
    monkeypatch.setattr(
        live_billing,
        "keychain",
        lambda: KeychainSecretProvider(live_billing.KEYCHAIN_SERVICE, client=client),
    )
    return client


def _unexpected_api_key() -> str:
    raise AssertionError("subscription billing must never read an API key")


def test_api_billing_is_the_default_and_unknown_modes_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(live_billing.BILLING_ENV, raising=False)
    assert not live_billing.subscription_billing()
    assert live_billing.live_model_id("openai:gpt-5.6-terra") == "openai:gpt-5.6-terra"
    assert live_billing.cost_cap(Decimal("0.15")) == Decimal("0.15")

    monkeypatch.setenv(live_billing.BILLING_ENV, "free")
    with pytest.raises(pytest.fail.Exception, match=live_billing.BILLING_ENV):
        live_billing.subscription_billing()


def test_subscription_maps_only_reviewed_openai_models_and_drops_dollar_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(live_billing.BILLING_ENV, "subscription")
    assert live_billing.live_model_id("openai:gpt-5.6-terra") == "codex:gpt-5.6-terra"
    assert live_billing.live_model_id("codex:gpt-5.6-sol") == "codex:gpt-5.6-sol"
    with pytest.raises(pytest.fail.Exception, match="no reviewed codex"):
        live_billing.live_model_id("anthropic:claude-opus-5-5")
    assert live_billing.cost_cap(Decimal("0.15")) is None


def test_subscription_provider_reads_only_the_test_keychain(
    monkeypatch: pytest.MonkeyPatch, keyring_client: _Keyring
) -> None:
    monkeypatch.setenv(live_billing.BILLING_ENV, "subscription")
    keyring_client.values[("daita", live_billing.CREDENTIAL.name)] = _credential()
    with pytest.raises(pytest.fail.Exception, match="tests.support.live_billing login"):
        live_billing.live_provider(
            "openai:gpt-5.6-terra",
            api_key=_unexpected_api_key,
            max_output_tokens=2_048,
        )

    keyring_client.values[
        (live_billing.KEYCHAIN_SERVICE, live_billing.CREDENTIAL.name)
    ] = _credential()
    profile, provider = live_billing.live_provider(
        "openai:gpt-5.6-terra",
        api_key=_unexpected_api_key,
        max_output_tokens=2_048,
        require_streaming=True,
    )
    assert profile.id == provider.provider_id == "codex:gpt-5.6-terra"


async def test_login_stores_the_credential_without_displaying_it(
    monkeypatch: pytest.MonkeyPatch,
    keyring_client: _Keyring,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from daita.llm import subscription_auth

    async def login(*, on_verification, on_progress=None):
        on_verification(
            subscription_auth.CodexDevicePrompt(
                "https://auth.openai.com/codex/device", "ABCD-1234", 900
            )
        )
        return _credential()

    monkeypatch.setattr(subscription_auth, "login_codex_subscription", login)
    await _run_cli(["login"])
    stored = keyring_client.values[
        (live_billing.KEYCHAIN_SERVICE, live_billing.CREDENTIAL.name)
    ]
    assert CodexOAuthCredential.from_secret(stored).refresh_token == _SECRET_MARKER
    live_billing.main(["status"])
    await _run_cli(["logout"])
    live_billing.main(["status"])
    output = capsys.readouterr().out
    assert "ABCD-1234" in output
    assert "A login is stored." in output and output.endswith("No login is stored.\n")
    assert _SECRET_MARKER not in output and "access-token" not in output


async def _run_cli(argv: list[str]) -> None:
    # main() owns its event loop; run it off the test loop like a CLI process.
    assert await asyncio.to_thread(live_billing.main, argv) == 0


def test_api_equivalent_cost_is_notional_codex_token_pricing() -> None:
    usage = ModelUsage(
        input_tokens=10_000,
        output_tokens=1_000,
        reasoning_tokens=400,
        cache_read_tokens=2_000,
    )
    notional = live_billing.api_equivalent_cost(usage, "codex:gpt-5.6-terra")
    assert notional is not None
    # 8,000 uncached at $2.50, 2,000 cached at $0.25, 1,000 output at $15 per 1M.
    assert Decimal(str(notional["api_equivalent_cost_usd"])) == Decimal("0.0355")
    assert notional["api_equivalent_cost_status"] == "complete"
    assert notional["basis"] == "codex_reported_tokens_at_api_rates"
    assert notional["rate_model_id"] == "openai:gpt-5.6-terra"
    assert live_billing.api_equivalent_cost(usage, "openai:gpt-5.6-terra") is None
    assert live_billing.api_equivalent_cost(usage, None) is None


def test_postgresql_subscription_route_uses_only_the_test_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(postgres_harness.AUTHORIZATION, "1")
    monkeypatch.delenv(postgres_harness.KEY_ENV, raising=False)
    monkeypatch.delenv("DAITA_POSTGRES_LIVE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(live_billing.BILLING_ENV, "subscription")
    config = postgres_harness.live_config("openai:gpt-5.6-terra")
    candidate = config.model_route.candidates[0]
    assert candidate.provider_id == "codex:gpt-5.6-terra"
    assert candidate.secret_reference == SecretReference.keychain(
        live_billing.CREDENTIAL.name
    )
    assert config.limits.max_estimated_cost_usd is None
    assert config.limits.max_total_tokens == postgres_harness.LIMITS.max_total_tokens
    assert config.limits.max_steps == postgres_harness.LIMITS.max_steps
    providers = postgres_harness.live_secret_provider().providers
    assert isinstance(providers[0], KeychainSecretProvider)

    monkeypatch.setenv(live_billing.BILLING_ENV, "api")
    assert not any(
        isinstance(provider, KeychainSecretProvider)
        for provider in postgres_harness.live_secret_provider().providers
    )


async def test_phase_f_subscription_route_drops_only_dollar_ceilings(
    monkeypatch: pytest.MonkeyPatch, keyring_client: _Keyring
) -> None:
    monkeypatch.setenv(phase_f_harness.AUTHORIZATION, "1")
    monkeypatch.delenv("DAITA_PHASE_F_LIVE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DAITA_PHASE_F_LIVE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(live_billing.BILLING_ENV, "subscription")
    keyring_client.values[
        (live_billing.KEYCHAIN_SERVICE, live_billing.CREDENTIAL.name)
    ] = _credential()
    limits = phase_f_harness.limits()
    assert limits.max_estimated_cost_usd is None
    assert limits.max_total_tokens == phase_f_harness.LoopLimits().max_total_tokens
    profile, provider = phase_f_harness.live_provider("openai:gpt-5.6-terra")
    try:
        assert profile.id == provider.provider_id == "codex:gpt-5.6-terra"
    finally:
        await provider.close()
