from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3

import pytest

from daita import Agent, AgentConfig, SQLiteSource
from daita.identity import AgentIdentity
from daita.llm import (
    FinishReason,
    ModelProfile,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelRoute,
    ModelRouteCandidate,
    ModelSensitivity,
    ProviderErrorCode,
    RetryPolicy,
    RetryStrategy,
    ToolCall,
)
from daita.llm.protocols import ModelRouteConflictError
from daita.llm.routing import ModelRouter
from daita.operations.models import AgentTrigger, TriggerKind
from daita.operations.runtime import OperationRuntime
from daita.security import EmptySecretProvider, SecretReference
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import (
    SQLiteCompatibilityError,
    SQLiteCorruptionError,
    SQLiteOperationStore,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def _profile(
    provider_id: str,
    *,
    input_cost: Decimal | None = None,
    output_cost: Decimal | None = None,
) -> ModelProfile:
    return ModelProfile(
        id=provider_id,
        context_window_tokens=8_192,
        max_output_tokens=1_024,
        supports_tools=True,
        supports_structured_output=True,
        supports_streaming=True,
        input_cost_per_million_usd=input_cost,
        output_cost_per_million_usd=output_cost,
    )


def _route(*, revision: int = 1) -> ModelRoute:
    return ModelRoute(
        candidates=(
            ModelRouteCandidate(
                profile=_profile(
                    "openai:gpt-route-primary",
                    input_cost=Decimal("1.25"),
                    output_cost=Decimal("5.00"),
                ),
                allowed_sensitivities=frozenset(
                    {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
                ),
                secret_reference=SecretReference.environment("OPENAI_API_KEY"),
            ),
            ModelRouteCandidate(
                profile=_profile("ollama:route-fallback"),
                allowed_sensitivities=frozenset(
                    {
                        ModelSensitivity.PUBLIC,
                        ModelSensitivity.INTERNAL,
                        ModelSensitivity.CONFIDENTIAL,
                    }
                ),
                base_url="http://127.0.0.1:11434/v1",
            ),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_provider=2,
            strategy=RetryStrategy.LINEAR,
            base_delay_seconds=0.25,
            max_delay_seconds=0.5,
            retryable_codes=frozenset(
                {ProviderErrorCode.TIMEOUT, ProviderErrorCode.RATE_LIMIT_ERROR}
            ),
        ),
        revision=revision,
    )


def _ollama_route(*, revision: int = 1) -> ModelRoute:
    return ModelRoute(
        candidates=(
            ModelRouteCandidate(
                profile=_profile("ollama:cold-route"),
                allowed_sensitivities=frozenset(
                    {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
                ),
                base_url="http://127.0.0.1:11434/v1",
            ),
        ),
        revision=revision,
    )


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL
            );
            INSERT INTO customers (status) VALUES ('active'), ('inactive');
            """)


class _ScriptedProvider:
    def __init__(self, provider_id: str, *responses: object) -> None:
        self.provider_id = provider_id
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected model call")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, ModelResponse)
        return response


async def test_model_route_round_trips_exact_order_profiles_policy_and_references(
    tmp_path: Path,
) -> None:
    path = tmp_path / "route.db"
    identity = AgentIdentity(
        id="agent-route",
        display_name="Route",
        created_at=NOW,
    )
    route = _route()
    store = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    try:
        await store.initialize_identity(identity)
        assert await store.load_model_route(identity.id) is None
        assert (
            await store.set_model_route(
                identity.id,
                route,
                expected_revision=0,
            )
            == route
        )
        assert (
            await store.set_model_route(
                identity.id,
                route,
                expected_revision=0,
            )
            == route
        )
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    try:
        assert await reopened.load_model_route(identity.id) == route
    finally:
        await reopened.close()

    with sqlite3.connect(path) as connection:
        candidates = connection.execute(
            "SELECT position, profile_id, input_cost_per_million_usd, "
            "output_cost_per_million_usd, allowed_sensitivities_json, "
            "base_url, secret_scheme, secret_name "
            "FROM agent_model_route_candidates ORDER BY position"
        ).fetchall()
        assert candidates == [
            (
                0,
                "openai:gpt-route-primary",
                "1.25",
                "5.00",
                '["internal","public"]',
                None,
                "env",
                "OPENAI_API_KEY",
            ),
            (
                1,
                "ollama:route-fallback",
                None,
                None,
                '["confidential","internal","public"]',
                "http://127.0.0.1:11434/v1",
                None,
                None,
            ),
        ]
        durable_text = "\n".join(
            str(value)
            for row in connection.execute("SELECT * FROM agent_model_route_candidates")
            for value in row
            if value is not None
        )
        assert "resolved-secret-value" not in durable_text


async def test_v15_profile_only_home_gains_no_invented_route(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-profile.db"
    identity = AgentIdentity(
        id="agent-legacy-profile",
        display_name="Legacy profile",
        created_at=NOW,
    )
    profile = _profile("mock:legacy-profile")
    legacy = await sqlite_owner._open_with_migrations(
        path,
        migrations=sqlite_owner._MIGRATIONS[:15],
        clock=lambda: NOW,
    )
    await legacy.initialize_identity(identity)
    await legacy.bind_model_profile(identity.id, profile)
    await legacy.close()

    upgraded = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    try:
        assert await upgraded.load_model_profile(identity.id) == profile
        assert await upgraded.load_model_route(identity.id) is None
    finally:
        await upgraded.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)
        assert (
            connection.execute(
                "SELECT model_route_revision, model_route_fingerprint FROM operations"
            ).fetchall()
            == []
        )


async def test_model_route_compare_and_set_binds_operations_and_blocks_live_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "route-cas.db"
    identity = AgentIdentity(
        id="agent-route-cas",
        display_name="Route CAS",
        created_at=NOW,
    )
    first = _ollama_route()
    second = _ollama_route(revision=2)
    store = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    try:
        await store.initialize_identity(identity)
        await store.set_model_route(identity.id, first, expected_revision=0)
        with pytest.raises(ModelRouteConflictError, match="compare-and-set"):
            await store.set_model_route(identity.id, second, expected_revision=0)

        runtime = OperationRuntime(
            store=store,
            clock=lambda: NOW,
            id_factory=_ids(),
            model_route_revision=first.revision,
            model_route_fingerprint=first.fingerprint,
        )
        snapshot = await runtime.begin(
            AgentTrigger(
                id="trigger-route",
                agent_id=identity.id,
                kind=TriggerKind.USER,
                source_id="user-route",
                payload={"message": "keep the route stable"},
                created_at=NOW,
            )
        )
        assert snapshot.operation.model_route_revision == first.revision
        assert snapshot.operation.model_route_fingerprint == first.fingerprint
        assert snapshot.events[-1].payload["model_route_revision"] == first.revision
        with pytest.raises(ModelRouteConflictError, match="nonterminal"):
            await store.set_model_route(identity.id, second, expected_revision=1)
        await runtime.fail(snapshot.operation.id, "test_terminal")
        assert (
            await store.set_model_route(
                identity.id,
                second,
                expected_revision=1,
            )
            == second
        )
        assert (
            await store.set_model_route(
                identity.id,
                second,
                expected_revision=1,
            )
            == second
        )
        assert await store.load_model_route(identity.id) == second
    finally:
        await store.close()


async def test_cold_configured_agent_reconstructs_route_and_runs_without_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daita.llm import factory as factory_owner

    database = tmp_path / "customers.db"
    _database(database)
    route = _ollama_route()
    scripted = _ScriptedProvider(
        route.candidates[0].provider_id,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="query",
                    name="data_query_sqlite",
                    arguments={
                        "source_id": "placeholder",
                        "sql": "SELECT COUNT(*) AS total FROM customers",
                    },
                ),
            ),
        ),
        ModelResponse(
            text="There are 2 customers. [evidence:evidence-1]",
            finish_reason=FinishReason.STOP,
        ),
    )
    monkeypatch.setattr(factory_owner, "create_llm_provider", lambda *a, **k: scripted)

    agent = await Agent.create(
        "cold-route",
        root=tmp_path / "state",
        config=AgentConfig(model_route=route),
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    registration = await agent.attach(SQLiteSource(database))
    await agent.close()
    scripted._responses[0] = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="query",
                name="data_query_sqlite",
                arguments={
                    "source_id": registration.id,
                    "sql": "SELECT COUNT(*) AS total FROM customers",
                },
            ),
        ),
    )

    reopened = await Agent.open(
        "cold-route",
        root=tmp_path / "state",
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    try:
        assert reopened.model_route == route
        result = await reopened.run("Count the customers.")
        snapshot = await reopened.inspect(result.operation_id)
        assert snapshot.operation.model_route_revision == route.revision
        assert snapshot.operation.model_route_fingerprint == route.fingerprint
        assert len(scripted.requests) == 2
        assert snapshot.evidence[0].accepted is True
    finally:
        await reopened.close()

    inspector = await Agent.open("cold-route", root=tmp_path / "state")
    try:
        durable = await inspector.inspect(result.operation_id)
        assert durable.operation.model_route_revision == route.revision
        assert durable.operation.model_route_fingerprint == route.fingerprint
    finally:
        await inspector.close()


async def test_missing_secret_and_extra_fail_before_provider_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daita.llm import factory as factory_owner

    route = ModelRoute(
        candidates=(
            ModelRouteCandidate(
                profile=_profile("openai:gpt-no-secret"),
                allowed_sensitivities=frozenset({ModelSensitivity.PUBLIC}),
                secret_reference=SecretReference.environment("MISSING_ROUTE_SECRET"),
            ),
        )
    )
    calls = 0

    def provider_factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider construction must follow secret resolution")

    monkeypatch.setattr(factory_owner, "create_llm_provider", provider_factory)
    created = await Agent.create(
        "missing-secret",
        root=tmp_path,
        config=AgentConfig(model_route=route),
        secret_provider=EmptySecretProvider(),
    )
    await created.close()
    reopened = await Agent.open(
        "missing-secret",
        root=tmp_path,
        secret_provider=EmptySecretProvider(),
    )
    try:
        result = await reopened.run("Do not reach provider I/O.")
        assert result.reason == "model_provider_failure"
        assert calls == 0
    finally:
        await reopened.close()

    no_secret_route = _ollama_route()
    missing_extra_calls = 0

    def missing_extra(*args, **kwargs):
        nonlocal missing_extra_calls
        missing_extra_calls += 1
        raise ImportError(
            "openai is required. Install with: pip install 'daita-agents[openai]'"
        )

    monkeypatch.setattr(factory_owner, "create_llm_provider", missing_extra)
    extra = await Agent.create(
        "missing-extra",
        root=tmp_path,
        config=AgentConfig(model_route=no_secret_route),
    )
    try:
        result = await extra.run("Do not reach provider I/O.")
        assert result.reason == "model_provider_failure"
        assert missing_extra_calls == 1
    finally:
        await extra.close()

    class StaticSecretProvider:
        async def resolve(self, reference: SecretReference) -> str:
            assert reference == route.candidates[0].secret_reference
            return "resolved-secret-value"

    observed_key: str | None = None

    def capture_key(*args, api_key=None, **kwargs):
        nonlocal observed_key
        observed_key = api_key
        raise ImportError(
            "openai is required. Install with: pip install 'daita-agents[openai]'"
        )

    monkeypatch.setattr(factory_owner, "create_llm_provider", capture_key)
    resolved = await Agent.create(
        "resolved-secret",
        root=tmp_path,
        config=AgentConfig(model_route=route),
        secret_provider=StaticSecretProvider(),
    )
    resolved_home = resolved.home
    try:
        result = await resolved.run("Resolve, but never persist, the secret.")
        assert result.reason == "model_provider_failure"
        assert observed_key == "resolved-secret-value"
    finally:
        await resolved.close()
    assert b"resolved-secret-value" not in (resolved_home / "state.db").read_bytes()
    assert b"resolved-secret-value" not in (resolved_home / "agent.toml").read_bytes()


async def test_route_tampering_fails_closed_before_reconstruction(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "tampered-route",
        root=tmp_path,
        config=AgentConfig(model_route=_ollama_route()),
    )
    state_path = agent.home / "state.db"
    await agent.close()

    with sqlite3.connect(state_path) as connection:
        connection.execute("DROP TRIGGER agent_model_route_candidates_reject_update")
        connection.execute(
            "UPDATE agent_model_route_candidates SET base_url = ?",
            ("http://127.0.0.1:9999/v1",),
        )
        connection.commit()

    with pytest.raises((SQLiteCompatibilityError, SQLiteCorruptionError)):
        await Agent.open("tampered-route", root=tmp_path)


def test_route_rejects_credential_bearing_endpoints_and_uses_router_for_retry() -> None:
    with pytest.raises(ValueError, match="without credentials"):
        ModelRouteCandidate(
            profile=_profile("ollama:unsafe-endpoint"),
            allowed_sensitivities=frozenset({ModelSensitivity.PUBLIC}),
            base_url="https://user:secret@example.test/v1",
        )

    route = ModelRoute(
        candidates=(
            ModelRouteCandidate(
                profile=_profile("ollama:retry-route"),
                allowed_sensitivities=frozenset(
                    {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
                ),
                base_url="http://127.0.0.1:11434/v1",
            ),
        ),
        retry_policy=RetryPolicy(max_attempts_per_provider=2),
    )
    assert route.model_profile.id.startswith("router:")


async def test_reconstructed_single_candidate_retry_is_owned_by_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daita.llm import factory as factory_owner
    from daita.llm.factory import create_model_route_provider
    from daita.llm.models import CanonicalMessage, MessageRole, ModelRequest, TextBlock

    route = ModelRoute(
        candidates=(
            ModelRouteCandidate(
                profile=_profile("ollama:retry-owner"),
                allowed_sensitivities=frozenset({ModelSensitivity.PUBLIC}),
                base_url="http://127.0.0.1:11434/v1",
            ),
        ),
        retry_policy=RetryPolicy(max_attempts_per_provider=2),
    )
    scripted = _ScriptedProvider(
        "ollama:retry-owner",
        ModelProviderError(ProviderErrorCode.TIMEOUT),
        ModelResponse(text="ready", finish_reason=FinishReason.STOP),
    )
    monkeypatch.setattr(factory_owner, "create_llm_provider", lambda *a, **k: scripted)
    provider = create_model_route_provider(route)
    assert isinstance(provider, ModelRouter)
    response = await provider.generate(
        ModelRequest(
            operation_id="operation-route",
            turn_id="turn-route",
            sensitivity=ModelSensitivity.PUBLIC,
            messages=(
                CanonicalMessage(
                    agent_id="agent-route",
                    operation_id="operation-route",
                    turn_id="turn-route",
                    role=MessageRole.USER,
                    content=(TextBlock("retry"),),
                ),
            ),
            context_selection={
                "schema_version": 1,
                "estimated_input_tokens": 1,
                "sensitivity_class": "public",
            },
        )
    )
    assert response.text == "ready"
    assert len(scripted.requests) == 2


async def test_reconstructed_fallback_preserves_declared_candidate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daita.llm import factory as factory_owner
    from daita.llm.factory import create_model_route_provider
    from daita.llm.models import CanonicalMessage, MessageRole, ModelRequest, TextBlock

    route = ModelRoute(
        candidates=tuple(
            ModelRouteCandidate(
                profile=_profile(provider_id),
                allowed_sensitivities=frozenset({ModelSensitivity.INTERNAL}),
                base_url=endpoint,
            )
            for provider_id, endpoint in (
                ("local-primary:model", "https://primary.example.test/v1"),
                ("local-fallback:model", "https://fallback.example.test/v1"),
            )
        )
    )
    primary = _ScriptedProvider(
        "local-primary:model",
        ModelProviderError(ProviderErrorCode.TIMEOUT),
    )
    fallback = _ScriptedProvider(
        "local-fallback:model",
        ModelResponse(text="fallback-ready", finish_reason=FinishReason.STOP),
    )
    providers = {
        primary.provider_id: primary,
        fallback.provider_id: fallback,
    }
    constructed: list[str] = []

    def construct(model_id: str, **kwargs):
        del kwargs
        constructed.append(model_id)
        return providers[model_id]

    monkeypatch.setattr(factory_owner, "create_llm_provider", construct)
    provider = create_model_route_provider(route)
    response = await provider.generate(
        ModelRequest(
            operation_id="operation-fallback",
            turn_id="turn-fallback",
            messages=(
                CanonicalMessage(
                    agent_id="agent-fallback",
                    operation_id="operation-fallback",
                    turn_id="turn-fallback",
                    role=MessageRole.USER,
                    content=(TextBlock("fallback"),),
                ),
            ),
            context_selection={
                "schema_version": 1,
                "estimated_input_tokens": 1,
            },
        )
    )
    assert response.text == "fallback-ready"
    assert response.provider_id == "local-fallback:model"
    assert constructed == ["local-primary:model", "local-fallback:model"]
