"""Authorized live-model coverage for Stage 0 sensitivity route admission."""

from __future__ import annotations

import os
import sqlite3
from contextlib import AsyncExitStack
from collections.abc import AsyncIterator
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent, LoopLimits, SQLiteSource, create_llm_provider
from daita.catalog.models import Sensitivity
from daita.llm._lifecycle import closing_stream
from daita.llm.models import (
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamEvent,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import (
    ManagedModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
)
from daita.llm.routing import (
    ModelProviderRegistration,
    ModelRouter,
    RetryPolicy,
)
from daita.loop.models import LoopExitKind

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE0_SENSITIVITY"
_MODEL_ID = "DAITA_STAGE0_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_STAGE0_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_STAGE0_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing live model "
            "cost for the Stage 0 sensitivity integration test"
        ),
    ),
]


class _RecordingProvider:
    """Capture canonical requests around one real API-backed provider."""

    def __init__(self, delegate: ManagedModelProvider) -> None:
        self._delegate = delegate
        self.requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return self._delegate.provider_id

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._delegate.supports_request_policy(request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return provider_has_complete_pricing(self._delegate, request)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return await self._delegate.generate(request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(self._delegate, StreamingModelProvider):
            raise TypeError("the live delegate must support canonical streaming")
        self.requests.append(request)
        async with closing_stream(self._delegate.stream(request)) as events:
            async for event in events:
                yield event

    async def close(self) -> None:
        await self._delegate.close()


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        pytest.fail(f"{name} must be set for the authorized live test")
    return value


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.20")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


def _route(
    provider: ManagedModelProvider,
    profile: ModelProfile,
    *,
    allowed_sensitivities: frozenset[ModelSensitivity],
) -> ModelRouter:
    return ModelRouter(
        (
            ModelProviderRegistration(
                provider=provider,
                profile=profile,
                allowed_sensitivities=allowed_sensitivities,
            ),
        ),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE internal_rows (id INTEGER PRIMARY KEY, value TEXT)"
        )
        connection.execute("INSERT INTO internal_rows VALUES (1, 'live sentinel')")


async def test_live_model_route_admits_internal_scope_and_blocks_public_only_route(
    tmp_path: Path,
) -> None:
    """Reach one live model, then prove the ineligible route stops before I/O."""

    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    provider_name = model_id.partition(":")[0]
    if provider_name not in _API_PROVIDERS:
        pytest.fail(
            f"{_MODEL_ID} must name an API-backed provider: "
            + ", ".join(sorted(_API_PROVIDERS))
        )
    api_key = _required_environment(_MODEL_KEY)
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name one release-reviewed tool-capable model")

    delegate = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 1_024),
    )
    provider = _RecordingProvider(delegate)
    async with AsyncExitStack() as cleanup:
        cleanup.push_async_callback(provider.close)
        limits = LoopLimits(
            max_steps=3,
            max_total_tokens=8_000,
            max_wall_time_seconds=90,
            max_estimated_cost_usd=_cost_limit(),
        )
        database = tmp_path / "stage0-live.sqlite"
        _database(database)

        eligible_route = _route(
            provider,
            profile,
            allowed_sensitivities=frozenset({ModelSensitivity.INTERNAL}),
        )
        eligible = await Agent.create(
            "live-stage0-eligible",
            root=tmp_path / "eligible-agent-home",
            model=eligible_route,
            model_profile=eligible_route.model_profile,
            limits=limits,
            workspace=workspace_for(tmp_path / "eligible-agent-home"),
        )
        try:
            source = await eligible.attach(
                SQLiteSource(database, name="Internal live data")
            )
            resources = await eligible.list_catalog_resources(source_id=source.id)
            assert resources
            assert {resource.sensitivity for resource in resources} == {
                Sensitivity.INTERNAL
            }

            eligible_exit = await eligible.run(
                "This is a live routing check. Reply with exactly LIVE_STAGE0_OK. "
                "Do not call tools and do not inspect the attached data.",
                source_id=source.id,
            )
        finally:
            await eligible.close()

        assert eligible_exit.kind is LoopExitKind.COMPLETED, (
            eligible_exit.reason,
            eligible_exit.usage.cost_estimate.code,
            len(provider.requests),
        )
        assert eligible_exit.final_text is not None
        assert "LIVE_STAGE0_OK" in eligible_exit.final_text
        assert provider.requests
        assert all(
            request.sensitivity is ModelSensitivity.INTERNAL
            for request in provider.requests
        )

        live_call_count = len(provider.requests)
        ineligible_route = _route(
            provider,
            profile,
            allowed_sensitivities=frozenset({ModelSensitivity.PUBLIC}),
        )
        ineligible = await Agent.create(
            "live-stage0-ineligible",
            root=tmp_path / "ineligible-agent-home",
            model=ineligible_route,
            model_profile=ineligible_route.model_profile,
            limits=limits,
            workspace=workspace_for(tmp_path / "ineligible-agent-home"),
        )
        try:
            blocked_source = await ineligible.attach(
                SQLiteSource(database, name="Internal blocked data")
            )
            blocked_exit = await ineligible.run(
                "Reply with LIVE_STAGE0_MUST_NOT_REACH_PROVIDER.",
                source_id=blocked_source.id,
            )
        finally:
            await ineligible.close()

        assert blocked_exit.kind is LoopExitKind.FAILED
        assert blocked_exit.reason == "model_route_ineligible"
        assert blocked_exit.usage.total_tokens == 0
        assert len(provider.requests) == live_call_count
