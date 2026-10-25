from tests.capabilities._effect_contract_support import (
    _RECEIPT,
    _capability,
)

"""Capability policies are authority, never presentation."""

from dataclasses import replace

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ExecutionContractBindings,
    OperationalEffect,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
    capability_contract_digest,
)
from daita.llm.models import ModelSensitivity
from tests.support.capability_runtime import StaticTestDomain, static_registry

_DIGEST = "sha256:" + "a" * 64


class _Executor:
    executor_id = "test.action.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        raise AssertionError("contract validation must not dispatch an action")


def test_effect_policy_invariants_and_execution_digests():
    capability = _capability()
    for change in (
        lambda: replace(capability, effect_receipt_policy=None),
        lambda: replace(capability, operational_effect=OperationalEffect.NONE),
    ):
        with pytest.raises(ValueError):
            change()
    digest = capability_contract_digest(capability, domain_owner_id="test")
    for changed in (
        replace(
            capability, effect_receipt_policy=replace(_RECEIPT, receipt_kind="test.new")
        ),
        replace(
            capability,
            effect_receipt_policy=replace(
                _RECEIPT, success_evidence_basis=EffectEvidenceBasis.ADAPTER_VERIFIED
            ),
        ),
    ):
        assert capability_contract_digest(changed, domain_owner_id="test") != digest


def test_bindings_require_exact_coverage_and_are_immutable():
    bindings = ExecutionContractBindings(
        capability_contracts={"test.action": _DIGEST},
        tool_origins={"test.action": _DIGEST},
        resource_revisions={"resource": _DIGEST},
        model_routes={"route": _DIGEST},
    )
    bindings.validate_coverage(
        capability_ids=("test.action",),
        resource_ids=("resource",),
        route_ids=("route",),
        mcp_capability_ids=("test.action",),
    )
    with pytest.raises(ValueError, match="exact coverage"):
        bindings.validate_coverage(
            capability_ids=("test.action", "other"),
            resource_ids=("resource",),
            route_ids=("route",),
            mcp_capability_ids=("test.action",),
        )
    with pytest.raises(ValueError):
        replace(bindings, tool_origins={"foreign": _DIGEST})
    assert bindings.digest != replace(bindings, tool_origins={}).digest


def test_observation_validation_preserves_evidence_basis_and_schema():
    capability = _capability()
    registry = static_registry(StaticTestDomain((capability,), ()), (_Executor(),))
    observation = EffectObservation(
        EffectOutcome.SUCCEEDED,
        EffectEvidenceBasis.SERVER_REPORTED,
        FrozenJsonObject.from_mapping({"target": "admitted-target"}),
    )
    assert (
        registry.validate_effect_observation(capability.id, observation) == observation
    )
    with pytest.raises(ToolOutputValidationError):
        registry.validate_effect_observation(
            capability.id,
            replace(observation, evidence_basis=EffectEvidenceBasis.ADAPTER_VERIFIED),
        )
    with pytest.raises(ToolOutputValidationError):
        registry.validate_effect_observation(
            capability.id,
            replace(
                observation, payload=FrozenJsonObject.from_mapping({"other": True})
            ),
        )
    with pytest.raises(ValueError):
        EffectObservation(
            EffectOutcome.NOT_APPLIED,
            EffectEvidenceBasis.SERVER_REPORTED,
            observation.payload,
        )
    with pytest.raises(ValueError):
        EffectObservation(EffectOutcome.SUCCEEDED, EffectEvidenceBasis.SERVER_REPORTED)


def test_model_execution_bindings_cover_declared_routes_profiles_and_retry_configuration():
    from daita.hosting.configuration import _model_execution_contracts
    from daita.llm.models import ModelProfile
    from daita.llm.providers.mock import MockModelProvider
    from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy

    model = MockModelProvider(())
    profile = ModelProfile(
        id=model.provider_id, context_window_tokens=64000, max_output_tokens=1000
    )
    candidate = ModelRouteCandidate(
        provider_id=model.provider_id,
        profile=profile,
        base_url="https://models.example.test/v1",
    )
    route = ModelRoute((candidate,))
    original = _model_execution_contracts(model, profile, route)
    assert set(original) == {model.provider_id}
    assert original == _model_execution_contracts(
        model, replace(profile), replace(route)
    )
    for changed in (
        replace(
            route,
            candidates=(replace(candidate, base_url="https://other.example.test/v1"),),
        ),
        replace(
            route,
            candidates=(
                replace(candidate, profile=replace(profile, max_output_tokens=2000)),
            ),
        ),
        replace(
            route,
            candidates=(
                replace(
                    candidate,
                    allowed_sensitivities=frozenset({ModelSensitivity.PUBLIC}),
                ),
            ),
        ),
        replace(
            route,
            retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
        ),
    ):
        assert _model_execution_contracts(model, profile, changed) != original
