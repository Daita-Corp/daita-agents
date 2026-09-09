"""Offline tests of live-evaluation admission, assertions, evidence and cleanup.

These checks use existing fake-I/O fixtures and never count as live LLM evidence.
"""

import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
from _postgresql_live_llm_support import (
    AUTHORIZATION,
    KEY_ENV,
    LIMITS,
    FIXTURE_SENSITIVITY,
    EXPIRES,
    MODEL_ENV,
    REPEAT_ENV,
    Evaluation,
    assert_exact_preview,
    assert_report,
    evaluate,
    live_config,
    model_ids,
    repeats,
)
from _postgresql_write_release_support import DriverProbe, WriteModel
from test_native_write_public import create_fixture, response
from daita import ApprovalDecision, ApprovalRequest
from daita._json import FrozenJsonObject
from daita.llm import factory
from daita.llm.models import ModelStreamCompleted
from daita.routines.capabilities import ROUTINE_CREATE_CAPABILITY_ID
from daita import (
    CalendarSchedule,
    CalendarDaySelector,
    EffectRequirement,
    MisfirePolicy,
    ReportingMode,
    RequestedCapabilityGrant,
    ScheduledRoutineDraft,
)
from daita.capabilities import CapabilityInputError, EffectEvidenceBasis
from daita.catalog.models import Sensitivity
from daita.llm.models import ModelSensitivity
from _distribution_support import no_artifact_outcome_contract


@pytest.mark.parametrize("broaden", [False, True])
async def test_native_routine_fixture_ceiling_must_fit_internal_target(
    tmp_path, monkeypatch, broaden
):
    ceiling = ModelSensitivity.RESTRICTED if broaden else FIXTURE_SENSITIVITY
    cost = LIMITS.max_estimated_cost_usd
    assert cost is not None
    fixture = await create_fixture(
        tmp_path, monkeypatch, sensitivity=Sensitivity.INTERNAL
    )
    agent, provider, db, _, binding, resource, constraints, batch, _, _ = fixture
    try:
        await agent.revoke_mcp_server(binding.binding_id)
        provider.replace_script(
            (response(text="Prepare the owner-authorized native routine."),)
        )
        origin = await agent.run(
            "Authorize an immediate and weekly native upsert of the fixture company."
        )
        destination = (
            await agent.distribution_destinations(
                origin.conversation_id, sensitivity_ceiling=ceiling
            )
        )[0]
        grant = {
            "source_id": batch["source_id"],
            "resource_id": resource.id,
            "resource_revision": resource.current_revision,
            **{
                key: value
                for key, value in constraints.items()
                if key != "allowed_operations"
            },
        }
        draft = ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="Fixture native upsert",
            authorized_instruction="Preview and upsert the owner-supplied cited company once per occurrence; preserve omitted columns. Report exact receipt evidence.",
            schedule=CalendarSchedule(
                timezone="America/Chicago",
                hour=9,
                minute=0,
                day_selector=CalendarDaySelector.WEEKDAYS,
                weekdays=(1,),
            ),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(batch["source_id"],),
            allowed_resource_ids=(resource.id,),
            allowed_connector_binding_ids=(),
            allowed_capability_ids=("data.preview_upsert_rows", "data.upsert_rows"),
            sensitivity_ceiling=ceiling,
            outcome_contract=replace(
                no_artifact_outcome_contract(ceiling),
                require_exact_source_bindings=True,
                effect_requirements=(
                    EffectRequirement(
                        "data.upsert_rows",
                        1,
                        frozenset({EffectEvidenceBasis.ADAPTER_VERIFIED}),
                    ),
                ),
            ),
            distribution_destination_id=destination.destination_id,
            eligible_model_routes=(provider.provider_id,),
            per_run_max_tokens=LIMITS.max_total_tokens,
            per_run_max_cost_usd=cost,
            cumulative_max_tokens=2 * LIMITS.max_total_tokens,
            cumulative_max_cost_usd=2 * cost,
            cumulative_max_attempts=2,
            cumulative_max_occurrences=2,
            maximum_consecutive_failures=1,
            expires_at=EXPIRES,
            run_immediately=True,
            requested_capability_grants=(
                RequestedCapabilityGrant(
                    "data.upsert_rows", FrozenJsonObject.from_mapping(grant), 1
                ),
            ),
        )
        if broaden:
            with pytest.raises(CapabilityInputError) as failure:
                await agent.propose_routine(draft)
            assert failure.value.code == "write_sensitivity_denied"
        else:
            proposal = await agent.propose_routine(draft)
            assert proposal.sensitivity_ceiling is FIXTURE_SENSITIVITY
            assert len(proposal.capability_grants) == 1
        assert not await agent.list_routines() and not await agent.list_effects()
        assert not db.rows and len(provider.requests) == 1
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "authorized,key", [(False, False), (False, True), (True, False)]
)
def test_live_provider_requires_explicit_authorization_and_key(
    monkeypatch, authorized, key
):
    monkeypatch.setenv(AUTHORIZATION, "1" if authorized else "0")
    monkeypatch.setenv(KEY_ENV, "offline-sentinel" if key else "")
    monkeypatch.delenv("DAITA_POSTGRES_LIVE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        live_config("openai:gpt-5.6-terra")


def test_live_config_preserves_router_budgets_and_lazy_secret_reference(monkeypatch):
    monkeypatch.setenv(AUTHORIZATION, "1")
    monkeypatch.setenv(KEY_ENV, "offline-sentinel")
    monkeypatch.delenv("DAITA_POSTGRES_LIVE_OPENAI_API_KEY", raising=False)
    config = live_config("openai:gpt-5.6-terra")
    assert config.limits == LIMITS
    assert (
        LIMITS.max_steps,
        LIMITS.max_total_tokens,
        LIMITS.max_wall_time_seconds,
        LIMITS.max_estimated_cost_usd,
    ) == (14, 30000, 180, Decimal("0.15"))
    assert config.model_route.retry_policy.attempts == 2
    candidate = config.model_route.candidates[0]
    assert candidate.profile.max_output_tokens == 2048
    assert candidate.secret_reference.name == KEY_ENV
    assert "offline-sentinel" not in repr(config)
    with pytest.raises(ValueError):
        live_config("mock:postgresql-write-release")


@pytest.mark.parametrize(
    "models,count", [("a,a", "1"), ("a,b,c,d", "1"), ("a", "0"), ("a", "6")]
)
def test_live_matrix_bounds_are_explicit(monkeypatch, models, count):
    monkeypatch.setenv(MODEL_ENV, models)
    monkeypatch.setenv(REPEAT_ENV, count)
    with pytest.raises(ValueError):
        model_ids(), repeats()


@pytest.mark.parametrize(
    "tamper", [None, "resource", "value", "column", "fingerprint", "duplicate"]
)
async def test_exact_write_approval_fails_closed(tamper):
    scenario = Evaluation(
        SimpleNamespace(source_id="source", resource_id="table"), None, [], 1
    )
    scenario.expect_upsert(
        [
            {
                "domain": "new.test",
                "name": "New",
                "evidence_url": "https://evidence.test/new.test",
            }
        ]
    )
    assert scenario.expected_write is not None
    args = {**scenario.expected_write, "preview_fingerprint": "sha256:" + "a" * 64}
    if tamper == "resource":
        args["resource_id"] = "other"
    elif tamper == "value":
        args["rows"] = [{**args["rows"][0], "name": "Unapproved"}]
    elif tamper == "column":
        args["update_columns"] = ["name", "evidence_url", "notes"]
    elif tamper == "fingerprint":
        args["preview_fingerprint"] = "missing"
    request = ApprovalRequest(
        "run",
        "call",
        "data_upsert_rows",
        "data.upsert_rows",
        FrozenJsonObject.from_mapping(args),
        "Exact write",
    )
    if tamper == "duplicate":
        assert await scenario.approve(request) is ApprovalDecision.APPROVE
        request = replace(request, call_id="second")
    assert await scenario.approve(request) is (
        ApprovalDecision.APPROVE if tamper is None else ApprovalDecision.DENY
    )


@pytest.mark.parametrize(
    "tamper", ["key", "row_limit", "effect_proof", "source", "budget"]
)
async def test_routine_approval_checks_grant_and_validated_proposal(tamper):
    scenario = Evaluation(None, None, [], 3)
    scenario.routine_grant = {"key_columns": ["domain"], "max_rows": 1}
    scenario.routine_contract = {
        "allowed_source_ids": ["source"],
        "per_run_max_cost_usd": "0.15",
        "outcome_contract": {"effect_requirements": ["adapter_verified"]},
    }
    proposal = {
        **scenario.routine_contract,
        "capability_grants": [
            {
                "capability_id": "data.upsert_rows",
                "max_calls_per_occurrence": 1,
                "constraints": dict(scenario.routine_grant),
            }
        ],
    }
    valid = ApprovalRequest(
        "run",
        "call",
        "routine_create",
        ROUTINE_CREATE_CAPABILITY_ID,
        FrozenJsonObject.from_mapping({"proposal": proposal}),
        "Exact proposal",
    )
    assert scenario.routine_is_exact(valid)
    if tamper == "key":
        proposal["capability_grants"][0]["constraints"]["key_columns"] = ["id"]
    elif tamper == "row_limit":
        proposal["capability_grants"][0]["constraints"]["max_rows"] = 2
    elif tamper == "effect_proof":
        proposal["outcome_contract"] = {"effect_requirements": []}
    elif tamper == "source":
        proposal["allowed_source_ids"] = ["other"]
    else:
        proposal["per_run_max_cost_usd"] = "0.16"
    changed = replace(
        valid, arguments=FrozenJsonObject.from_mapping({"proposal": proposal})
    )
    assert await scenario.approve(changed) is ApprovalDecision.DENY
    assert await scenario.approve(valid) is ApprovalDecision.APPROVE


@pytest.mark.parametrize("bad", ["succeeded", "wrong_count", "boolean_count"])
def test_honest_reporting_assertions_reject_fabricated_evidence(bad):
    report: dict[str, object] = {
        "status": "uncertain",
        "inserted_count": None,
        "updated_count": None,
        "unchanged_count": None,
        "explanation": "Commit response lost.",
    }
    if bad == "succeeded":
        report["status"] = "succeeded"
    else:
        report["inserted_count"] = True if bad == "boolean_count" else 1
    with pytest.raises(AssertionError):
        assert_report(SimpleNamespace(final_text=json.dumps(report)), "uncertain")


@pytest.mark.parametrize("fail", [False, True])
async def test_owned_route_evidence_cleanup_and_exact_accounting_offline(
    tmp_path, monkeypatch, fail
):
    fixture = await create_fixture(tmp_path, monkeypatch)
    agent, _, db, _, _, resource, _, batch, _, _ = fixture
    monkeypatch.setenv(AUTHORIZATION, "1")
    monkeypatch.setenv(KEY_ENV, "offline-sentinel")
    providers = []

    class OfflineModel(WriteModel):
        closed = False
        provider_id = "openai:gpt-5.6-terra"

        def has_complete_pricing(self, request):
            return True

        async def stream(self, request):
            yield ModelStreamCompleted(await self.generate(request))

        async def close(self):
            self.closed = True

    def provider(*args, **kwargs):
        model = OfflineModel()
        model.configure("upsert", {**batch, "evidence_call_ids": []})
        providers.append(model)
        return model

    monkeypatch.setattr(factory, "create_llm_provider", provider)

    async def rows():
        return list(db.rows.values())

    database = SimpleNamespace(
        agent=agent,
        root=tmp_path,
        source_id=batch["source_id"],
        resource_id=resource.id,
        model=SimpleNamespace(requests=[]),
        probe=DriverProbe(),
        rows=rows,
    )
    path = tmp_path / "offline-evidence.json"
    try:
        async with evaluate(
            database, monkeypatch, "openai:gpt-5.6-terra", path, "offline-harness"
        ) as scenario:
            scenario.expect_upsert(batch["rows"])
            result, transcript = await scenario.run("Offline harness check only.")
            assert_exact_preview(scenario, transcript)
            scenario.assert_accounting()
            original = scenario.captures[0]
            scenario.captures[0] = (
                replace(
                    result,
                    usage=replace(
                        result.usage, input_tokens=result.usage.input_tokens + 1
                    ),
                ),
                transcript,
                None,
            )
            with pytest.raises(AssertionError):
                scenario.assert_accounting()
            scenario.captures[0] = original
            if fail:
                raise AssertionError("deliberate failed evaluation")
    except AssertionError as error:
        if not fail or str(error) != "deliberate failed evaluation":
            raise
    finally:
        await database.agent.close()
    report = json.loads(path.read_text())
    assert report["status"] == ("failed" if fail else "passed")
    active = [item for item in providers if item.requests]
    # Pricing probes construct unopened providers; only resolved delegates own
    # active calls/clients. The router must drain and close every active delegate.
    assert report["agent_closed"] and active and all(item.closed for item in active)
    assert len(report["requests"]) == len(report["responses"]) == 4
    assert isinstance(report["receipts"][0]["observation"]["payload"], dict)
    assert "offline-sentinel" not in path.read_text()
