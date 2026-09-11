"""Policy validation and monotonic allowance precedence."""

from dataclasses import FrozenInstanceError, fields, replace
from decimal import Decimal

import pytest

from daita.config import AgentConfig
from daita.hosting.embedded import (
    _decode_call_policy,
    _encode_call_policy,
    _model_execution_contracts,
)
from daita.llm import ModelCallPolicy, RetryPolicy
from daita.llm._lifecycle import materialize_request
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider


def request(**kwargs):
    return ModelRequest(
        (CanonicalMessage(MessageRole.USER, content=(TextBlock("public"),)),), **kwargs
    )


@pytest.mark.parametrize(
    "bad", [True, False, 0, -1, float("inf"), float("nan"), "15", None]
)
@pytest.mark.parametrize("name", [field.name for field in fields(ModelCallPolicy)])
def test_policy_rejects_invalid_duration(name, bad):
    with pytest.raises(ValueError):
        ModelCallPolicy(**{name: bad})


def test_policy_limits_and_immutability():
    for kwargs in (
        {"max_request_seconds": 100},
        {"first_progress_timeout_seconds": 121},
        {"progress_idle_timeout_seconds": 121},
        {"input_count_timeout_seconds": 61},
        {"cleanup_timeout_seconds": 31},
        {"read_timeout_seconds": 3601},
    ):
        with pytest.raises(ValueError):
            ModelCallPolicy(**kwargs)
    policy = ModelCallPolicy(read_timeout_seconds=42)
    assert type(policy.read_timeout_seconds) is float
    with pytest.raises(FrozenInstanceError):
        setattr(policy, "read_timeout_seconds", 43)  # noqa: B010 - test frozen mutation
    assert AgentConfig(model_call_policy=policy).model_call_policy == policy


def test_deadlines_are_materialized_without_widening_caller_bounds():
    req = materialize_request(request(), now=100)
    assert (req.deadline, req.attempt_deadline) == (280, 220)
    narrowed = materialize_request(replace(req, attempt_deadline=150), now=110)
    assert (narrowed.deadline, narrowed.attempt_deadline) == (280, 150)
    zero = ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
    future = materialize_request(
        request(deadline=10**12, attempt_deadline=10**12), now=10**11
    )
    assert future.remaining_after(zero) == future
    assert materialize_request(request(deadline=90), now=100).attempt_deadline == 90


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attempt_deadline": True},
        {"attempt_deadline": -1},
        {"attempt_deadline": float("inf")},
        {"deadline": 10, "attempt_deadline": 11},
    ],
)
def test_request_rejects_invalid_attempt_deadline(kwargs):
    with pytest.raises(ValueError):
        request(**kwargs)


def test_policy_exact_roundtrip_and_contract_digest():
    policy = ModelCallPolicy(read_timeout_seconds=48)
    encoded = _encode_call_policy(policy)
    assert _decode_call_policy(encoded) == policy
    for value in ({**encoded, "unknown": 1}, {"read_timeout_seconds": 48}):
        with pytest.raises(ValueError):
            _decode_call_policy(value)
    model = MockModelProvider([])
    profile = model.model_profile
    before = _model_execution_contracts(model, profile, None, policy)
    after = _model_execution_contracts(
        model, profile, None, replace(policy, read_timeout_seconds=49)
    )
    assert before != after


def test_attempt_names_and_total_precedence():
    assert RetryPolicy().max_total_attempts == 3
    assert (
        RetryPolicy(
            max_attempts_per_candidate=5, max_total_attempts=1
        ).max_total_attempts
        == 1
    )
    with pytest.raises(TypeError):
        RetryPolicy(**{"attempts": 1})  # noqa: PIE804
    for count in (0, 26, True, 1.5):
        with pytest.raises(ValueError):
            RetryPolicy(max_total_attempts=count)  # type: ignore[arg-type]


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("changed_field", ["policy", "per_candidate", "total"])
def test_policy_and_retry_changes_bind_configured_and_injected_routes(
    configured, changed_field
):
    from daita.llm.routing import (
        ModelProviderRegistration,
        ModelRoute,
        ModelRouteCandidate,
        ModelRouter,
    )

    provider = MockModelProvider([])
    profile = provider.model_profile
    policy = ModelCallPolicy(read_timeout_seconds=48)
    retry = RetryPolicy()

    def contracts(call_policy, retry_policy):
        route = ModelRoute(
            (ModelRouteCandidate(provider_id=provider.provider_id, profile=profile),),
            retry_policy=retry_policy,
        )
        model = ModelRouter(
            (ModelProviderRegistration(provider=provider, profile=profile),),
            retry_policy=retry_policy,
        )
        return _model_execution_contracts(
            model, profile, route if configured else None, call_policy
        )

    original = contracts(policy, retry)
    assert original == contracts(replace(policy), replace(retry))
    if changed_field == "policy":
        changed = contracts(replace(policy, read_timeout_seconds=49), retry)
    elif changed_field == "per_candidate":
        changed = contracts(policy, replace(retry, max_attempts_per_candidate=3))
    else:
        changed = contracts(policy, replace(retry, max_total_attempts=4))
    assert original.keys() == changed.keys()
    assert original[provider.provider_id] != changed[provider.provider_id]
