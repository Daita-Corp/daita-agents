from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from daita import Agent, AgentConfig, ConfigError, RetryPolicy, RetryStrategy
from daita.agent import AgentNotConfiguredError
from daita.llm import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelProviderRegistration,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelSensitivity,
    ProviderErrorCode,
    TextBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopBudgets
from daita.operations.governance import DefaultPolicyProfile


def _profile(provider_id: str = "mock:configured") -> ModelProfile:
    return ModelProfile(
        id=provider_id,
        context_window_tokens=8_192,
        max_output_tokens=1_024,
        supports_tools=True,
        supports_structured_output=True,
        supports_streaming=True,
    )


def test_agent_config_binds_existing_typed_configuration_owners() -> None:
    profile = _profile()
    budgets = LoopBudgets(max_turns=4, max_actions=6)
    retry = RetryPolicy(
        max_attempts_per_provider=3,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay_seconds=0.25,
        max_delay_seconds=1.0,
    )
    policy = DefaultPolicyProfile(version="2")
    config = AgentConfig(
        model_profile=profile,
        budgets=budgets,
        retry_policy=retry,
        policy_profile=policy,
    )

    assert config.schema_version == 1
    assert config.model_profile is profile
    assert config.budgets is budgets
    assert config.retry_policy is retry
    assert config.policy_profile is policy
    assert not hasattr(config, "settings")
    with pytest.raises(FrozenInstanceError):
        setattr(config, "schema_version", 2)


@pytest.mark.parametrize("schema_version", (0, 2))
def test_agent_config_rejects_unknown_schema_versions(schema_version: int) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        AgentConfig(schema_version=schema_version)


async def test_public_agent_accepts_config_and_persists_its_model_binding(
    tmp_path,
) -> None:
    model = MockModelProvider((), provider_id="mock:configured")
    config = AgentConfig(model_profile=_profile())

    created = await Agent.create(
        "configured-agent",
        root=tmp_path,
        config=config,
        model=model,
    )
    try:
        assert created.model_profile == config.model_profile
    finally:
        await created.close()

    reopened = await Agent.open(
        "configured-agent",
        root=tmp_path,
        config=config,
        model=model,
    )
    try:
        assert reopened.model_profile == config.model_profile
    finally:
        await reopened.close()


async def test_public_agent_reopens_with_state_owned_runtime_defaults(
    tmp_path,
) -> None:
    config = AgentConfig(
        budgets=LoopBudgets(max_turns=3, max_actions=5),
        policy_profile=DefaultPolicyProfile(version="2"),
    )
    created = await Agent.create(
        "runtime-default-agent",
        root=tmp_path,
        config=config,
    )
    try:
        assert created._embedded.runtime_defaults == config.runtime_defaults
    finally:
        await created.close()

    reopened = await Agent.open("runtime-default-agent", root=tmp_path)
    try:
        assert reopened._embedded.runtime_defaults == config.runtime_defaults
        assert reopened._embedded._runtime._policy.profile == config.policy_profile
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    ("changed", "section"),
    (
        (
            AgentConfig(
                budgets=LoopBudgets(max_turns=4, max_actions=5),
                policy_profile=DefaultPolicyProfile(version="2"),
            ),
            "budgets",
        ),
        (
            AgentConfig(
                budgets=LoopBudgets(max_turns=3, max_actions=5),
                policy_profile=DefaultPolicyProfile(version="3"),
            ),
            "policy",
        ),
    ),
)
async def test_public_agent_rejects_persisted_runtime_default_drift(
    tmp_path,
    changed: AgentConfig,
    section: str,
) -> None:
    original = AgentConfig(
        budgets=LoopBudgets(max_turns=3, max_actions=5),
        policy_profile=DefaultPolicyProfile(version="2"),
    )
    created = await Agent.create(
        "runtime-default-drift-agent",
        root=tmp_path,
        config=original,
    )
    await created.close()

    with pytest.raises(ConfigError) as captured:
        await Agent.open(
            "runtime-default-drift-agent",
            root=tmp_path,
            config=changed,
        )

    assert captured.value.error_code == "config_conflict"
    assert captured.value.section == section
    reopened = await Agent.open("runtime-default-drift-agent", root=tmp_path)
    try:
        assert reopened._embedded.runtime_defaults == original.runtime_defaults
    finally:
        await reopened.close()


async def test_public_agent_rejects_conflicting_config_before_home_mutation(
    tmp_path,
) -> None:
    config = AgentConfig(budgets=LoopBudgets(max_turns=2))

    with pytest.raises(ConfigError) as captured:
        await Agent.create(
            "conflicting-agent",
            root=tmp_path,
            config=config,
            budgets=LoopBudgets(max_turns=3),
        )

    assert captured.value.error_code == "config_conflict"
    assert not (tmp_path / "agents" / "conflicting-agent").exists()


async def test_retry_policy_is_applied_by_the_model_router_owner() -> None:
    model = MockModelProvider(
        (
            ModelProviderError(ProviderErrorCode.TIMEOUT),
            ModelResponse(text="ready", finish_reason=FinishReason.STOP),
        ),
        provider_id="mock:configured",
    )
    profile = _profile()
    policy = RetryPolicy(
        max_attempts_per_provider=2,
        strategy=RetryStrategy.LINEAR,
        base_delay_seconds=0.5,
        max_delay_seconds=1.0,
    )
    registration = ModelProviderRegistration(
        provider=model,
        profile=profile,
        allowed_sensitivities=frozenset({ModelSensitivity.INTERNAL}),
    )
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    router = ModelRouter(
        registration,
        retry_policy=policy,
        sleep=record_delay,
    )
    request = ModelRequest(
        operation_id="operation-1",
        turn_id="turn-1",
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id="operation-1",
                turn_id="turn-1",
                role=MessageRole.USER,
                content=(TextBlock("continue"),),
            ),
        ),
        context_selection={
            "schema_version": 1,
            "estimated_input_tokens": 10,
        },
    )

    response = await router.generate(request)

    assert router.retry_policy is policy
    assert response.text == "ready"
    assert delays == [0.5]
    assert policy.delay_after(1) == 0.5
    assert policy.delay_after(3) == 1.0
    assert not hasattr(policy, "execute_with_retry")


async def test_nondefault_retry_policy_cannot_be_silently_ignored_by_raw_model(
    tmp_path,
) -> None:
    config = AgentConfig(
        retry_policy=RetryPolicy(
            max_attempts_per_provider=2,
            strategy=RetryStrategy.FIXED,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        )
    )
    model = MockModelProvider((), provider_id="mock:configured")

    with pytest.raises(ConfigError) as captured:
        await Agent.create(
            "retry-owner-agent",
            root=tmp_path,
            config=config,
            model=model,
            model_profile=_profile(),
        )

    assert captured.value.error_code == "config_owner_required"
    assert not (tmp_path / "agents" / "retry-owner-agent").exists()


async def test_persisted_router_binding_rejects_retry_policy_drift(
    tmp_path,
) -> None:
    first_retry = RetryPolicy(
        max_attempts_per_provider=2,
        strategy=RetryStrategy.FIXED,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
    )
    registration = ModelProviderRegistration(
        provider=MockModelProvider((), provider_id="mock:configured"),
        profile=_profile(),
        allowed_sensitivities=frozenset({ModelSensitivity.INTERNAL}),
    )
    first_router = ModelRouter(registration, retry_policy=first_retry)
    first_config = AgentConfig(
        model_profile=first_router.profile,
        retry_policy=first_retry,
    )
    created = await Agent.create(
        "retry-binding-agent",
        root=tmp_path,
        config=first_config,
        model=first_router,
    )
    await created.close()

    changed_retry = RetryPolicy(
        max_attempts_per_provider=3,
        strategy=RetryStrategy.FIXED,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
    )
    changed_router = ModelRouter(registration, retry_policy=changed_retry)
    with pytest.raises(
        AgentNotConfiguredError,
        match="configured model provider differs from the stored profile",
    ):
        await Agent.open(
            "retry-binding-agent",
            root=tmp_path,
            config=AgentConfig(
                model_profile=changed_router.profile,
                retry_policy=changed_retry,
            ),
            model=changed_router,
        )
