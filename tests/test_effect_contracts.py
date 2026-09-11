"""Capability policies and normalized grants are authority, never presentation."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from _capability_runtime_support import StaticTestDomain, static_registry

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    AutomationGrantPolicy,
    AutomationScopeProposal,
    Capability,
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    EffectReceiptPolicy,
    ExecutionContractBindings,
    OperationalEffect,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
    capability_contract_digest,
)
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import ModelSensitivity

_DIGEST = "sha256:" + "a" * 64
_SCHEMA = {
    "type": "object",
    "properties": {"target": {"type": "string", "minLength": 1, "maxLength": 64}},
    "required": ["target"],
    "additionalProperties": False,
}
_RECEIPT = EffectReceiptPolicy(
    "test.action", _SCHEMA, EffectEvidenceBasis.SERVER_REPORTED
)
_GRANT = AutomationGrantPolicy("test.target", _SCHEMA)


def _capability() -> Capability:
    return Capability(
        id="test.action",
        description="One unrelated external action.",
        input_schema=_SCHEMA,
        output_kind="test.action.result",
        output_schema=_SCHEMA,
        executor_id="test.action.executor",
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.EXTERNAL_ACTION,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        automation_grant_policy=_GRANT,
        effect_receipt_policy=_RECEIPT,
    )


def _proposal() -> AutomationScopeProposal:
    return AutomationScopeProposal(
        agent_id="agent",
        principal_id="human",
        allowed_source_ids=(),
        allowed_resource_ids=(),
        allowed_connector_binding_ids=(),
        allowed_capability_ids=("test.action",),
        allowed_access_modes=frozenset({AccessMode.NONE}),
        allowed_operational_effects=frozenset({OperationalEffect.EXTERNAL_ACTION}),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=("route",),
        per_run_max_cost_usd=Decimal("1"),
        per_run_max_tokens=1000,
        expires_at=datetime.now(UTC) + timedelta(days=1),
        distribution_plan_digest=_DIGEST,
    )


class _Executor:
    executor_id = "test.action.executor"
    calls = 0

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.calls += 1
        raise AssertionError("grant preparation must not dispatch an action")


class _GrantDomain(StaticTestDomain):
    preparations = 0
    corrupt = False

    async def prepare_automation_grant(
        self,
        capability: Capability,
        constraints: FrozenJsonObject,
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> FrozenJsonObject:
        self.preparations += 1
        if max_calls_per_occurrence > 2 or constraints["target"] != "admitted-target":
            raise CapabilityInputError(
                "automation_grant_outside_scope",
                "Target or call ceiling is not admitted.",
            )
        return (
            FrozenJsonObject.from_mapping({"invalid": True})
            if self.corrupt
            else constraints
        )


def test_effect_policy_invariants_and_execution_digests():
    capability = _capability()
    for change in (
        lambda: replace(capability, effect_receipt_policy=None),
        lambda: replace(capability, automation_grant_policy=None),
        lambda: replace(capability, operational_effect=OperationalEffect.NONE),
        lambda: replace(
            capability, automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY
        ),
    ):
        with pytest.raises(ValueError):
            change()
    digest = capability_contract_digest(capability, domain_owner_id="test")
    for changed in (
        replace(
            capability,
            automation_grant_policy=replace(_GRANT, constraints_kind="test.new"),
        ),
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


async def test_grant_preparation_validates_before_and_after_domain_without_execution():
    capability = _capability()
    domain = _GrantDomain((capability,), ())
    executor = _Executor()
    registry = static_registry(domain, (executor,))
    runtime = CapabilityRuntime(registry, (domain,))
    proposal = _proposal()
    grant = await runtime.prepare_automation_grant(
        capability.id, {"target": "admitted-target"}, 2, proposal
    )
    assert grant.capability_contract_digest == registry.contract_digest(capability.id)
    assert grant.constraints.to_dict() == {"target": "admitted-target"}
    assert grant.grant_digest != replace(grant, max_calls_per_occurrence=1).grant_digest
    assert executor.calls == 0 and domain.preparations == 1
    with pytest.raises(CapabilityInputError):
        await runtime.prepare_automation_grant(
            capability.id, {"target": "admitted-target", "grant": True}, 1, proposal
        )
    assert domain.preparations == 1
    with pytest.raises(CapabilityInputError, match="Target or call ceiling"):
        await runtime.prepare_automation_grant(
            capability.id, {"target": "admitted-target"}, 3, proposal
        )
    with pytest.raises(CapabilityInputError, match="proposed ceilings"):
        await runtime.prepare_automation_grant(
            capability.id,
            {"target": "admitted-target"},
            1,
            replace(proposal, allowed_capability_ids=("other",)),
        )
    domain.corrupt = True
    with pytest.raises(CapabilityInputError):
        await runtime.prepare_automation_grant(
            capability.id, {"target": "admitted-target"}, 1, proposal
        )
    assert executor.calls == 0


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
    from daita.hosting.embedded import _model_execution_contracts
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
