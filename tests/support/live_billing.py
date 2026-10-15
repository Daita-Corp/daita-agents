"""Opt-in ChatGPT subscription billing for live model tests.

Live suites default to pay-as-you-go API keys. Setting
``DAITA_LIVE_BILLING=subscription`` routes their reviewed ``openai:<model>``
choice through the matching ``codex:<model>`` subscription route instead.

The subscription login is stored only in the local OS keychain under a
test-only service, so live tests never read a user's ordinary Daita logins.
It is resolved when a test constructs its provider, never at import or
collection, and is never printed, logged, or accepted from the environment.

Subscription responses carry no per-request price, and Daita refuses unpriced
requests under a dollar cap. Subscription runs therefore keep their step and
token limits but drop the dollar cap, and work that requires a priced budget
(background graph tasks, routines) is skipped rather than run unbounded. Token
limits are checked against each returned response, so one run can exceed its
token cap by at most one response.

Reports record a notional API-equivalent cost: reported subscription tokens
priced at the matching ``openai:<model>`` list rates. It is evidence for
comparison only, never billed cost, and never admits or stops a run.

Log in once, then run any adopted live suite with its usual authorization:

    .venv/bin/python -m tests.support.live_billing login
    DAITA_LIVE_BILLING=subscription DAITA_RUN_LIVE_PHASE5_TOOL_SELECTION=1 \\
      .venv/bin/python -m pytest tests/live/acceptance/test_tool_selection.py \\
      -o addopts="--tb=short -q --strict-markers"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from daita import create_llm_provider
from daita.llm.models import ModelProfile, ModelUsage
from daita.llm.pricing import calculate_cost_estimate, load_bundled_pricing_schedules
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ManagedModelProvider
from daita.security import KeychainSecretProvider, SecretReference

BILLING_ENV = "DAITA_LIVE_BILLING"
API_BILLING = "api"
SUBSCRIPTION_BILLING = "subscription"
KEYCHAIN_SERVICE = "daita-live-tests"
CREDENTIAL = SecretReference.keychain("codex-subscription")
SUBSCRIPTION_PROVIDER = "codex"
API_EQUIVALENT_BASIS = "codex_reported_tokens_at_api_rates"


def subscription_billing() -> bool:
    """Return whether this process explicitly selected subscription billing."""

    value = os.environ.get(BILLING_ENV, API_BILLING)
    if value not in {API_BILLING, SUBSCRIPTION_BILLING}:
        pytest.fail(
            f"{BILLING_ENV} must be {API_BILLING!r} or {SUBSCRIPTION_BILLING!r}"
        )
    return value == SUBSCRIPTION_BILLING


def live_model_id(model_id: str) -> str:
    """Map a reviewed API model to its subscription route when selected."""

    if not subscription_billing():
        return model_id
    provider, _, model = model_id.partition(":")
    if provider == SUBSCRIPTION_PROVIDER:
        return model_id
    candidate = f"{SUBSCRIPTION_PROVIDER}:{model}"
    if provider != "openai" or reviewed_model_profile(candidate) is None:
        pytest.fail(
            f"{model_id} has no reviewed {SUBSCRIPTION_PROVIDER} subscription route; "
            f"unset {BILLING_ENV} to use API billing for it"
        )
    return candidate


def cost_cap(amount: Decimal) -> Decimal | None:
    """Return the dollar cap for API billing and none for subscription routes."""

    return None if subscription_billing() else amount


def require_api_billing(reason: str) -> None:
    """Skip work whose budget contract needs priced model usage."""

    if subscription_billing():
        pytest.skip(f"{reason} requires API billing; unset {BILLING_ENV}")


def live_provider(
    model_id: str,
    *,
    api_key: Callable[[], str],
    max_output_tokens: int,
    require_streaming: bool = False,
) -> tuple[ModelProfile, ManagedModelProvider]:
    """Construct one reviewed live provider for the selected billing mode.

    ``api_key`` is called only under API billing so subscription runs never
    require or read an API key.
    """

    selected = live_model_id(model_id)
    profile = reviewed_model_profile(selected)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{selected} is not a release-reviewed tool-capable model")
    output = min(profile.max_output_tokens, max_output_tokens)
    if not subscription_billing():
        # Codex streams internally through generate(); only API routes declare it.
        if require_streaming and not profile.supports_streaming:
            pytest.fail(f"{selected} does not have reviewed streaming support")
        return profile, create_llm_provider(
            selected, api_key=api_key(), max_output_tokens=output
        )
    return profile, create_llm_provider(
        selected,
        subscription_credential=_stored_credential(),
        credential_updater=_store_credential,
        max_output_tokens=output,
    )


def api_equivalent_cost(
    usage: ModelUsage, provider_id: str | None
) -> dict[str, str | None] | None:
    """Price subscription-reported tokens at the matching API list rates.

    Returns ``None`` for API routes, which carry their own estimate. The result
    is notional evidence only and must never gate a run.
    """

    from daita.llm.providers.openai.adapter import _billable_quantities

    if provider_id is None:
        return None
    provider, _, model = provider_id.partition(":")
    if provider != SUBSCRIPTION_PROVIDER:
        return None
    try:
        quantities = _billable_quantities(usage)
    except ValueError:
        estimate = None
    else:
        estimate = calculate_cost_estimate(
            load_bundled_pricing_schedules(),
            provider="openai",
            model=model,
            endpoint="responses",
            requested_at=datetime.now(UTC),
            qualifiers={"service_tier": "default", "region": "global"},
            usage_values={"request_input_tokens": Decimal(usage.input_tokens)},
            quantities=quantities,
        )
    return {
        "api_equivalent_cost_usd": (
            None
            if estimate is None or estimate.amount_usd is None
            else str(estimate.amount_usd)
        ),
        "api_equivalent_cost_status": (
            "usage_inconsistent" if estimate is None else estimate.status.value
        ),
        "basis": API_EQUIVALENT_BASIS,
        "rate_model_id": f"openai:{model}",
    }


def keychain() -> KeychainSecretProvider:
    """Return the test-only keychain that holds the subscription login."""

    return KeychainSecretProvider(service_name=KEYCHAIN_SERVICE)


def _stored_credential() -> str:
    value = keychain().client.get_password(KEYCHAIN_SERVICE, CREDENTIAL.name)
    if not value:
        pytest.fail(
            "No live-test subscription login is stored; run "
            "`.venv/bin/python -m tests.support.live_billing login`"
        )
    return value


async def _store_credential(value: str) -> None:
    # Codex rotates refresh tokens; persist the replacement before reuse.
    await keychain().set(CREDENTIAL, value)


async def _login() -> None:
    from daita.llm.subscription_auth import (
        CodexDevicePrompt,
        login_codex_subscription,
    )

    def show(prompt: CodexDevicePrompt) -> None:
        print(f"Open {prompt.verification_url} and enter code {prompt.user_code}")
        print(f"The code expires in {prompt.expires_in_seconds // 60} minutes.")

    await _store_credential(await login_codex_subscription(on_verification=show))
    print(f"Stored the login in keychain service {KEYCHAIN_SERVICE!r}.")


async def _logout() -> None:
    await keychain().delete(CREDENTIAL)
    print(f"Removed the login from keychain service {KEYCHAIN_SERVICE!r}.")


def _status() -> None:
    stored = bool(keychain().client.get_password(KEYCHAIN_SERVICE, CREDENTIAL.name))
    print("A login is stored." if stored else "No login is stored.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.support.live_billing",
        description="Manage the keychain login used by subscription-billed live tests.",
    )
    parser.add_argument("command", choices=("login", "status", "logout"))
    command = parser.parse_args(argv).command
    actions: dict[str, Callable[[], Coroutine[None, None, None]]] = {
        "login": _login,
        "logout": _logout,
    }
    if command == "status":
        _status()
    else:
        asyncio.run(actions[command]())
    return 0


__all__ = [
    "API_EQUIVALENT_BASIS",
    "BILLING_ENV",
    "CREDENTIAL",
    "KEYCHAIN_SERVICE",
    "SUBSCRIPTION_PROVIDER",
    "api_equivalent_cost",
    "cost_cap",
    "keychain",
    "live_model_id",
    "live_provider",
    "require_api_billing",
    "subscription_billing",
]

if __name__ == "__main__":
    sys.exit(main())
