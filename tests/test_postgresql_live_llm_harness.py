"""Offline tests of live-evaluation admission, assertions, evidence and cleanup.

These checks use existing fake-I/O fixtures and never count as live LLM evidence.
"""

import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from _distribution_support import no_artifact_outcome_contract
from _postgresql_live_llm_support import (
    AUTHORIZATION,
    EXPIRES,
    FIXTURE_SENSITIVITY,
    KEY_ENV,
    LIMITS,
    MODEL_ENV,
    PROFILE_ENV,
    REPEAT_ENV,
    USER_FLOW_LIMITS,
    Evaluation,
    assert_exact_preview,
    assert_report,
    evaluate,
    evaluation_profile,
    live_config,
    model_ids,
    repeats,
)
from _postgresql_write_release_support import DriverProbe, WriteModel
from test_native_write_public import create_fixture, response

from daita import (
    ApprovalDecision,
    ApprovalRequest,
    CalendarDaySelector,
    CalendarSchedule,
    EffectRequirement,
    MisfirePolicy,
    ReportingMode,
    RequestedCapabilityGrant,
    ScheduledRoutineDraft,
)
from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import CapabilityInputError, EffectEvidenceBasis
from daita.catalog.models import Sensitivity
from daita.llm import factory
from daita.llm.models import (
    MessageRole,
    ModelSensitivity,
    ModelStreamCompleted,
    TextBlock,
)
from daita.loop.models import LoopExitKind
from daita.routines.capabilities import ROUTINE_CREATE_CAPABILITY_ID


@pytest.mark.parametrize(
    "broaden,operation,prerequisite",
    [
        (False, "upsert", "matching"),
        (True, "upsert", "matching"),
        (False, "upsert", "omitted"),
        (False, "upsert", "wrong"),
        (False, "update", "matching"),
        (False, "update", "omitted"),
        (False, "update", "wrong"),
    ],
)
async def test_native_routine_fixture_ceiling_must_fit_internal_target(
    tmp_path, monkeypatch, broaden, operation, prerequisite
):
    ceiling = ModelSensitivity.RESTRICTED if broaden else FIXTURE_SENSITIVITY
    cost = LIMITS.max_estimated_cost_usd
    assert cost is not None
    fixture = await create_fixture(
        tmp_path, monkeypatch, sensitivity=Sensitivity.INTERNAL
    )
    agent, provider, db, _, binding, resource, constraints, batch, clock, _ = fixture
    try:
        if operation == "update":
            constraints = {
                **constraints,
                "allowed_operations": ("update",),
                "key_columns": ("id",),
                "allowed_insert_columns": (),
                "generated_identity_columns": (),
            }
            permission = await agent.preview_source_permissions(
                source_id=batch["source_id"],
                read_mode="all",
                read_resource_ids=(),
                relational_write_scopes={resource.id: constraints},
            )
            await agent.apply_source_permissions(
                source_id=batch["source_id"],
                confirmation_fingerprint=permission.confirmation_fingerprint,
            )
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
        write = f"data.{operation}_rows"
        preview_operation = "update" if operation == "upsert" else "upsert"
        preview = f"data.preview_{operation if prerequisite == 'matching' else preview_operation}_rows"
        draft = replace(
            draft,
            allowed_capability_ids=(
                (write,) if prerequisite == "omitted" else (preview, write)
            ),
            requested_capability_grants=(
                RequestedCapabilityGrant(
                    write, FrozenJsonObject.from_mapping(grant), 1
                ),
            ),
            outcome_contract=replace(
                draft.outcome_contract,
                effect_requirements=(
                    EffectRequirement(
                        write, 1, frozenset({EffectEvidenceBasis.ADAPTER_VERIFIED})
                    ),
                ),
            ),
        )
        if prerequisite != "matching":
            before = list(db.log)
            with pytest.raises(CapabilityInputError) as failure:
                await agent.propose_routine(draft)
            assert failure.value.code == "automation_grant_preview_required"
            assert f"data.preview_{operation}_rows" in str(failure.value)
            assert db.log == before
            assert not await agent.list_routines() and not await agent.list_effects()
            assert not fixture[-1]
            return
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
        if not broaden and operation == "upsert":
            from live.test_postgresql_live_llm import admit_owner_routine

            from daita.routines.owner import _routine_proposal_payload

            # Exercise the independent live-stage setup without starting its host.
            await agent._embedded._routine_supervisor.close()
            scenario = Evaluation(
                SimpleNamespace(agent=agent),
                SimpleNamespace(limits=LIMITS),
                [],
                2,
                setup_mode="owner_admitted_routine",
            )
            scenario.routine_contract = _routine_proposal_payload(proposal)
            scenario.routine_grant = grant
            clock[0] = scenario.clock
            routine = await admit_owner_routine(
                scenario, "Owner authorizes the bounded fixture."
            )
            assert routine.run_immediately is True
            assert routine.allowed_resource_ids == (resource.id,)
            assert routine.capability_grants == proposal.capability_grants
            assert (
                routine.per_run_max_tokens
                == scenario.routine_budgets["per_run_max_tokens"]
            )
            assert routine.cumulative_max_cost_usd == 2 * cost
            assert set(routine.allowed_capability_ids) == {
                "data.preview_upsert_rows",
                "data.upsert_rows",
            }
            scenario.profile = "user_flow"
            natural = await admit_owner_routine(
                scenario, "Keep this company current now and each Monday."
            )
            assert set(natural.allowed_capability_ids) == {
                "catalog.schema",
                "data.preview_upsert_rows",
                "data.upsert_rows",
            }
            assert natural.allowed_source_ids == routine.allowed_source_ids
            assert natural.allowed_resource_ids == routine.allowed_resource_ids
            assert natural.capability_grants == routine.capability_grants
            assert "catalog_schema" not in natural.authorized_instruction
            assert "preview_fingerprint" not in natural.authorized_instruction
            assert len(provider.requests) == 1 and not await agent.list_effects()
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
    route = config.model_route
    assert route is not None
    assert route.retry_policy.max_attempts_per_candidate == 2
    candidate = route.candidates[0]
    assert candidate.profile.max_output_tokens == 2048
    secret_reference = candidate.secret_reference
    assert secret_reference is not None and secret_reference.name == KEY_ENV
    assert "offline-sentinel" not in repr(config)
    with pytest.raises(ValueError):
        live_config("mock:postgresql-write-release")


def test_diagnostic_limits_are_explicit_and_do_not_replace_strict_defaults(monkeypatch):
    monkeypatch.setenv(AUTHORIZATION, "1")
    monkeypatch.setenv(KEY_ENV, "offline-sentinel")
    limits = replace(
        LIMITS, max_total_tokens=50_000, max_estimated_cost_usd=Decimal("0.25")
    )
    config = live_config("openai:gpt-5.6-terra", limits=limits)
    scenario = Evaluation(None, config, [], 3)
    assert scenario.routine_budgets == {
        "per_run_max_tokens": 50000,
        "per_run_max_cost_usd": "0.25",
        "cumulative_max_tokens": 100000,
        "cumulative_max_cost_usd": "0.50",
        "cumulative_max_attempts": 2,
        "cumulative_max_occurrences": 2,
    }
    assert live_config("openai:gpt-5.6-terra").limits == LIMITS
    with pytest.raises(ValueError, match="finite cost cap"):
        live_config(
            "openai:gpt-5.6-terra", limits=replace(limits, max_estimated_cost_usd=None)
        )


def test_user_flow_profile_is_explicit_bounded_and_keeps_strict_defaults(monkeypatch):
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert evaluation_profile() == "strict"
    monkeypatch.setenv(PROFILE_ENV, "user_flow")
    assert evaluation_profile() == "user_flow"
    assert (
        USER_FLOW_LIMITS.max_steps,
        USER_FLOW_LIMITS.max_total_tokens,
        USER_FLOW_LIMITS.max_wall_time_seconds,
        USER_FLOW_LIMITS.max_estimated_cost_usd,
    ) == (24, 100000, 300, Decimal("0.50"))
    assert LIMITS.max_total_tokens == 30000
    assert LIMITS.max_estimated_cost_usd == Decimal("0.15")
    monkeypatch.setenv(PROFILE_ENV, "unbounded")
    with pytest.raises(ValueError, match="strict or user_flow"):
        evaluation_profile()


def test_reporting_failure_identifies_budget_stop_before_json_parsing():
    with pytest.raises(AssertionError, match="token_budget_insufficient"):
        assert_report(
            SimpleNamespace(final_text=None, reason="token_budget_insufficient"),
            "succeeded",
        )


@pytest.mark.parametrize(
    "models,count", [("a,a", "1"), ("a,b,c,d", "1"), ("a", "0"), ("a", "6")]
)
def test_live_matrix_bounds_are_explicit(monkeypatch, models, count):
    monkeypatch.setenv(MODEL_ENV, models)
    monkeypatch.setenv(REPEAT_ENV, count)
    with pytest.raises(ValueError):
        model_ids()
        repeats()


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
        FrozenJsonObject.from_mapping(
            {
                "arguments": args,
                "target": {"source_id": "source", "resource_id": "table"},
                "preview": {},
            }
        ),
        "Exact write",
    )
    if tamper == "duplicate":
        assert await scenario.approve(request) is ApprovalDecision.APPROVE
        request = replace(request, call_id="second")
    assert await scenario.approve(request) is (
        ApprovalDecision.APPROVE if tamper is None else ApprovalDecision.DENY
    )


@pytest.mark.parametrize(
    "column,value,accepted",
    [
        ("id", 1, True),
        ("domain", "existing.test", True),
        ("name", "existing.test", False),
        ("id", 2, False),
        ("domain", "different.test", False),
    ],
)
async def test_update_approval_distinguishes_identity_from_a_positive_wrong_match(
    column, value, accepted
):
    # Both entities have a legitimate one-row selector. Count alone cannot
    # distinguish the intended domain from the other company's name.
    fixture_rows = [
        {"id": 1, "domain": "existing.test", "name": "Before"},
        {"id": 2, "domain": "different.test", "name": "existing.test"},
    ]
    assert len([row for row in fixture_rows if row[column] == value]) == 1
    scenario = Evaluation(
        SimpleNamespace(source_id="source", resource_id="table"), None, [], 1
    )
    scenario.expect_update()
    assert scenario.expected_write is not None
    request = ApprovalRequest(
        "run",
        "call",
        "data_update_rows",
        "data.update_rows",
        FrozenJsonObject.from_mapping(
            {
                "arguments": {
                    **scenario.expected_write,
                    "where": [{"column": column, "operator": "eq", "value": value}],
                    "preview_fingerprint": "sha256:" + "a" * 64,
                },
                "target": {"source_id": "source", "resource_id": "table"},
                "preview": {},
            }
        ),
        "Review the exact selected company",
    )
    assert await scenario.approve(request) is (
        ApprovalDecision.APPROVE if accepted else ApprovalDecision.DENY
    )


@pytest.mark.parametrize(
    "tamper", ["key", "row_limit", "effect_proof", "source", "budget", "capability"]
)
@pytest.mark.parametrize("profile", ["strict", "user_flow"])
async def test_routine_approval_checks_grant_and_validated_proposal(tamper, profile):
    scenario = Evaluation(None, None, [], 3, profile=profile)
    scenario.routine_grant = {"key_columns": ["domain"], "max_rows": 1}
    scenario.routine_contract = {
        "allowed_source_ids": ["source"],
        "allowed_capability_ids": ["data.preview_upsert_rows", "data.upsert_rows"],
        "per_run_max_cost_usd": "0.15",
        "outcome_contract": {"effect_requirements": ["adapter_verified"]},
    }
    assert scenario.routine_contract is not None
    assert scenario.routine_grant is not None
    proposal: dict[str, Any] = {
        **scenario.routine_contract,
        "capability_grants": [
            {
                "capability_id": "data.upsert_rows",
                "max_calls_per_occurrence": 1,
                "constraints": dict(scenario.routine_grant),
            }
        ],
    }
    if profile == "user_flow":
        proposal["allowed_capability_ids"] = [
            "catalog.schema",
            "data.preview_upsert_rows",
            "data.upsert_rows",
        ]
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
    elif tamper == "budget":
        proposal["per_run_max_cost_usd"] = "0.16"
    else:
        proposal["allowed_capability_ids"] = [
            "data.preview_upsert_rows",
            "data.upsert_rows",
            "data.update_rows",
        ]
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


@pytest.mark.parametrize("profile", ["strict", "user_flow"])
@pytest.mark.parametrize("failure", [None, "body", "accounting", "both", "budget"])
async def test_owned_route_evidence_cleanup_and_exact_accounting_offline(
    tmp_path, monkeypatch, failure, profile
):
    fixture = await create_fixture(tmp_path, monkeypatch)
    agent, _, db, _, _, resource, _, batch, _, _ = fixture
    limits = USER_FLOW_LIMITS if profile == "user_flow" else LIMITS
    origin = None
    if profile == "user_flow":
        fixture[1].replace_script((response(text="Owner request retained."),))
        origin = await agent.run("Remember this conversation's company request.")
    monkeypatch.setenv(AUTHORIZATION, "1")
    monkeypatch.setenv(KEY_ENV, "offline-sentinel")
    providers = []

    class OfflineModel(WriteModel):
        closed = False
        provider_id = "openai:gpt-5.6-terra"

        def has_complete_pricing(self, request):
            return True

        async def generate(self, request):
            result = await super().generate(request)
            if failure == "budget" and self.phase == 4:
                result = replace(
                    result,
                    usage=replace(result.usage, input_tokens=limits.max_total_tokens),
                )
            return result

        async def stream(self, request):
            yield ModelStreamCompleted(await self.generate(request))

        async def close(self, *, deadline: float | None = None):
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
    caught = None
    evidence_checked = False
    try:
        async with evaluate(
            database,
            monkeypatch,
            "openai:gpt-5.6-terra",
            path,
            "offline-harness",
            max_runs=2 if failure == "budget" else 1,
            profile=profile,
            limits=limits,
            origin=origin,
        ) as scenario:
            scenario.expect_upsert(batch["rows"])
            if profile == "user_flow":
                with pytest.raises(AssertionError, match="ordinary user request"):
                    await scenario.run("Never submit this structured prompt.")
                assert not any(item.requests for item in providers)
            natural = "Please save the company details and tell me what changed."
            result, transcript = await scenario.run(
                "Offline harness check only.", user_prompt=natural
            )
            first_request = next(item for item in providers if item.requests).requests[
                0
            ]
            user_text = [
                block.text
                for message in first_request.messages
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextBlock)
            ]
            if profile == "user_flow":
                assert natural in user_text
                assert "Offline harness check only." not in user_text
                assert origin is not None
                assert result.conversation_id == origin.conversation_id
                assert "Remember this conversation's company request." in user_text
                if failure != "budget":
                    scenario.assert_report(result, "succeeded", (1, 0, 0))
            assert_exact_preview(scenario, transcript)
            assert len(await scenario.agent.list_effects()) == 1
            evidence_checked = True
            if failure == "budget":
                assert result.kind is LoopExitKind.FAILED
                assert result.usage.total_tokens > limits.max_total_tokens
                request_count = sum(len(item.requests) for item in providers)
                with pytest.raises(AssertionError):
                    await scenario.run("A failed run cannot fund a follow-up.")
                assert sum(len(item.requests) for item in providers) == request_count
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
            if failure not in {"accounting", "both"}:
                scenario.captures[0] = original
            if failure in {"body", "both"}:
                raise AssertionError("deliberate failed evaluation")
    except AssertionError as error:
        caught = error
    finally:
        await database.agent.close()
    report = json.loads(path.read_text())
    assert report["evaluation_profile"] == profile
    if profile == "user_flow":
        assert report["answer_accuracy"]["status"] == "requires_separate_review"
        if failure != "budget":
            assert (
                report["answer_accuracy"]["reviews"][0]["status"]
                == "pending_evidence_review"
            )
    assert evidence_checked
    assert (caught is not None) == (failure is not None)
    if failure in {"body", "both"}:
        assert str(caught) == "deliberate failed evaluation"
        assert report["failure"] == "AssertionError: deliberate failed evaluation"
    assert report["status"] == ("failed" if failure else "passed")
    assert report["accounting"]["status"] == (
        "failed" if failure in {"accounting", "both"} else "passed"
    )
    if failure in {"accounting", "both"}:
        assert "input_tokens" in report["accounting"]["failure"]
    active = [item for item in providers if item.requests]
    # Pricing probes construct unopened providers; only resolved delegates own
    # active calls/clients. The router must drain and close every active delegate.
    assert report["agent_closed"] and active and all(item.closed for item in active)
    assert len(report["requests"]) == len(report["responses"]) == 4
    assert isinstance(report["receipts"][0]["observation"]["payload"], dict)
    assert "offline-sentinel" not in path.read_text()


@pytest.mark.parametrize(
    "case",
    [
        "id",
        "business_key",
        "guards",
        "reordered",
        "composite",
        "wrong_id",
        "name_only",
        "incomplete_key",
        "broad",
        "operator",
        "conflict",
        "null",
        "boolean",
        "assignment",
        "count",
        "source",
        "table",
        "fingerprint",
        "preview_identity",
        "preview_count",
        "extra_envelope",
        "extra_argument",
        "extra_target",
        "duplicate",
    ],
)
async def test_user_flow_update_oracle_preserves_exact_guarded_identity(case):
    row = {
        "id": 1,
        "domain": "existing.test",
        "name": "Shared Company",
        "tenant": "north",
    }
    scenario = Evaluation(
        SimpleNamespace(source_id="source", resource_id="table"),
        None,
        [],
        1,
        profile="user_flow",
    )
    scenario.expect_update(
        intended_row=row,
        unique_keys=(
            (("id",), ("tenant", "domain"))
            if case in {"composite", "incomplete_key"}
            else (("id",), ("domain",))
        ),
    )
    where = [
        {"column": key, "operator": "eq", "value": row[key]}
        for key in ("id", "domain", "name")
    ]
    if case == "id":
        where = where[:1]
    if case in {"business_key", "incomplete_key"}:
        where = where[1:2]
    if case == "reordered":
        where.reverse()
    if case == "composite":
        where = [
            {"column": key, "operator": "eq", "value": row[key]}
            for key in ("tenant", "domain", "name")
        ]
    if case == "wrong_id":
        where[0]["value"] = 2
    if case == "name_only":
        where = where[2:]
    if case == "broad":
        where = []
    if case == "operator":
        where[0]["operator"] = "gte"
    if case == "conflict":
        where.append({"column": "domain", "operator": "eq", "value": "other.test"})
    if case == "null":
        where[0]["value"] = None
    if case == "boolean":
        where[0]["value"] = True
    assert scenario.expected_write is not None
    args = {
        **scenario.expected_write,
        "where": where,
        "preview_fingerprint": "sha256:" + "a" * 64,
    }
    review: dict[str, Any] = {
        "arguments": args,
        "target": {"source_id": "source", "resource_id": "table"},
        "preview": {
            "matched_rows": 1,
            "samples": [
                {
                    "primary_key": [{"column": "id", "value": 1}],
                    "before": [{"column": "name", "value": "Shared Company"}],
                    "after": [{"column": "name", "value": "Updated"}],
                }
            ],
            "warnings": [],
        },
    }
    if case == "assignment":
        args["assignments"] = [{"column": "notes", "value": "Updated"}]
    if case == "count":
        args["expected_affected_rows"] = 2
    if case == "source":
        args["source_id"] = "other"
    if case == "table":
        review["target"]["resource_id"] = "other"
    if case == "fingerprint":
        args["preview_fingerprint"] = "altered"
    if case == "preview_identity":
        review["preview"]["samples"][0]["primary_key"][0]["value"] = 2
    if case == "preview_count":
        review["preview"]["matched_rows"] = 2
    if case == "extra_envelope":
        review["extra"] = True
    if case == "extra_argument":
        args["extra"] = True
    if case == "extra_target":
        review["target"]["extra"] = True
    request = ApprovalRequest(
        "run",
        "call",
        "data_update_rows",
        "data.update_rows",
        FrozenJsonObject.from_mapping(review),
        "Exact intended company",
    )
    original = canonical_json(request.arguments)
    if case == "duplicate":
        assert await scenario.approve(request) is ApprovalDecision.APPROVE
        request = replace(request, call_id="second")
    assert await scenario.approve(request) is (
        ApprovalDecision.APPROVE
        if case in {"id", "business_key", "guards", "reordered", "composite"}
        else ApprovalDecision.DENY
    )
    assert canonical_json(request.arguments) == original
