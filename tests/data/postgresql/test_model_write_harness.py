"""Offline tests of live-evaluation admission, assertions, evidence and cleanup.

These checks use existing fake-I/O fixtures and never count as live LLM evidence.
"""

import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from daita import ApprovalDecision, ApprovalRequest
from daita._json import FrozenJsonObject
from daita.catalog.models import Sensitivity
from daita.llm import factory
from daita.llm.models import (
    MessageRole,
    ModelStreamCompleted,
    TextBlock,
)
from daita.loop.models import LoopExitKind
from daita.routines.capabilities import ROUTINE_CREATE_CAPABILITY_ID
from tests.support.native_writes import create_fixture, response
from tests.support.postgresql_live_harness import (
    AUTHORIZATION,
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
from tests.support.postgresql_write_release import DriverProbe, WriteModel


async def test_owner_admitted_routine_fixture_saves_the_owner_intent(
    tmp_path, monkeypatch
):
    from daita.routines.models import CalendarSchedule
    from tests.support.postgresql_model_write import admit_owner_routine

    fixture = await create_fixture(
        tmp_path, monkeypatch, sensitivity=Sensitivity.INTERNAL
    )
    agent, provider, db, _, _, _, _, batch, clock, _ = fixture
    try:
        # Exercise the independent live-stage setup without starting its host.
        await agent._embedded._routine_supervisor.close()
        scenario = Evaluation(
            SimpleNamespace(agent=agent),
            SimpleNamespace(limits=LIMITS),
            [],
            2,
            setup_mode="owner_admitted_routine",
        )
        clock[0] = scenario.clock
        requests = len(provider.requests)
        routine = await admit_owner_routine(scenario, batch["source_id"])
        assert routine.source_ids == (batch["source_id"],)
        assert isinstance(routine.schedule, CalendarSchedule)
        assert routine.schedule.weekdays == (1,) and routine.schedule.hour == 9
        assert scenario.conversation_id == routine.conversation_id
        inspection = await agent.inspect_routine(routine.routine_id)
        assert inspection is not None
        # One immediate occurrence is claimed at creation.
        assert len(inspection.recent_occurrences) == 1
        scenario.profile = "user_flow"
        natural = await admit_owner_routine(scenario, batch["source_id"])
        assert natural.source_ids == routine.source_ids
        assert "catalog_describe" not in natural.authorized_instruction
        assert "preview_fingerprint" not in natural.authorized_instruction
        assert len(provider.requests) == requests and not await agent.list_effects()
        assert not db.rows
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "authorized,key", [(False, False), (False, True), (True, False)]
)
def test_model_write_harness_requires_explicit_authorization_and_key(
    monkeypatch, authorized, key
):
    monkeypatch.setenv(AUTHORIZATION, "1" if authorized else "0")
    monkeypatch.setenv(KEY_ENV, "offline-sentinel" if key else "")
    monkeypatch.delenv("DAITA_POSTGRES_LIVE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        live_config("openai:gpt-5.6-terra")


def test_model_write_config_preserves_router_budgets_and_lazy_secret_reference(
    monkeypatch,
):
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
    ) == (24, 200000, 300, Decimal("0.15"))
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
    assert config.limits == limits
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
    ) == (24, 200000, 300, Decimal("0.50"))
    assert LIMITS.max_total_tokens == 200000
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
def test_model_write_matrix_bounds_are_explicit(monkeypatch, models, count):
    monkeypatch.setenv(MODEL_ENV, models)
    monkeypatch.setenv(REPEAT_ENV, count)
    with pytest.raises(ValueError):
        model_ids()
        repeats()


_NEW_ROW = {
    "domain": "new.test",
    "name": "New",
    "evidence_url": "https://evidence.test/new.test",
}


def _preview(**changes: object) -> dict[str, object]:
    return {
        "sql": "INSERT INTO t (domain) VALUES ($1)",
        "operation": "insert",
        "columns_changed": ["domain", "name", "evidence_url"],
        "affected_rows": 1,
        "inserted_rows": 1,
        "updated_rows": 0,
        "deleted_rows": 0,
        "skipped_rows": 0,
        "samples": [{"action": "insert", "values": dict(_NEW_ROW)}],
        "triggers": [],
        "cascades": [],
        "warnings": [],
        "verification": "rows",
        **changes,
    }


def _review(preview: dict[str, object], **target: object) -> dict[str, object]:
    return {
        "arguments": {"source_id": "source", "sql": "SQL", "parameters": []},
        "target": {
            "source_id": "source",
            "resource_id": "table",
            "name": "public.companies",
            "source_name": "Shop",
            **target,
        },
        "preview": preview,
    }


def _write_request(review: dict[str, object]) -> ApprovalRequest:
    return ApprovalRequest(
        "run",
        "call",
        "data_write",
        "data.write",
        FrozenJsonObject.from_mapping(review),
        "Exact write",
    )


@pytest.mark.parametrize(
    "tamper", [None, "unchanged", "resource", "value", "column", "count", "duplicate"]
)
async def test_exact_write_approval_fails_closed(tamper):
    scenario = Evaluation(
        SimpleNamespace(source_id="source", resource_id="table"), None, [], 1
    )
    scenario.expect_upsert([dict(_NEW_ROW)])
    preview = _preview()
    target: dict[str, object] = {}
    if tamper == "unchanged":
        # An upsert that skips rows already correct is right, not a tampered card.
        preview = _preview(affected_rows=0, inserted_rows=0, skipped_rows=1, samples=[])
    elif tamper == "resource":
        target = {"resource_id": "other"}
    elif tamper == "value":
        preview = _preview(
            samples=[{"action": "insert", "values": {**_NEW_ROW, "name": "Unapproved"}}]
        )
    elif tamper == "column":
        preview = _preview(columns_changed=["domain", "name", "evidence_url", "notes"])
    elif tamper == "count":
        preview = _preview(affected_rows=2)
    request = _write_request(_review(preview, **target))
    if tamper == "duplicate":
        assert await scenario.approve(request) is ApprovalDecision.APPROVE
        request = replace(request, call_id="second")
    assert await scenario.approve(request) is (
        ApprovalDecision.APPROVE
        if tamper in {None, "unchanged"}
        else ApprovalDecision.DENY
    )


@pytest.mark.parametrize("profile", ["strict", "user_flow"])
@pytest.mark.parametrize(
    ("before", "after", "columns", "affected", "accepted"),
    [
        ("Before", "Updated", ["name"], 1, True),
        # The statement selected the other company, whose name is the domain.
        ("existing.test", "Updated", ["name"], 1, False),
        ("Before", "Other", ["name"], 1, False),
        ("Before", "Updated", ["name", "notes"], 1, False),
        ("Before", "Updated", ["name"], 2, False),
    ],
)
async def test_update_approval_distinguishes_the_intended_row_from_a_wrong_match(
    before, after, columns, affected, accepted, profile
):
    scenario = Evaluation(
        SimpleNamespace(source_id="source", resource_id="table"),
        None,
        [],
        1,
        profile=profile,
    )
    scenario.expect_update()
    request = _write_request(
        _review(
            _preview(
                operation="update",
                columns_changed=columns,
                affected_rows=affected,
                inserted_rows=0,
                updated_rows=affected,
                samples=[
                    {
                        "action": "update",
                        "before": {"name": before},
                        "after": {"name": after},
                    }
                ],
            )
        )
    )
    assert await scenario.approve(request) is (
        ApprovalDecision.APPROVE if accepted else ApprovalDecision.DENY
    )


@pytest.mark.parametrize("tamper", ["title", "schedule", "prompt", "sources"])
@pytest.mark.parametrize("profile", ["strict", "user_flow"])
async def test_routine_approval_checks_the_exact_confirmation_card(tamper, profile):
    scenario = Evaluation(None, None, [], 3, profile=profile)
    scenario.routine_contract = {
        "title": "Company finding",
        "schedule": "Now, then calendar on Mon at 09:00 America/Chicago",
        "sources": ["companies"],
    }
    card: dict[str, Any] = {
        **scenario.routine_contract,
        "prompt": "Keep company new.test current.",
    }
    valid = ApprovalRequest(
        "run",
        "call",
        "routine_save",
        ROUTINE_CREATE_CAPABILITY_ID,
        FrozenJsonObject.from_mapping(card),
        "Save this routine?",
    )
    assert scenario.routine_is_exact(valid)
    if tamper == "prompt":
        # The prompt is the model's to write; only named card values are checked.
        card["prompt"] = "Something else."
    else:
        card[tamper] = ["other"] if tamper == "sources" else f"changed {card[tamper]}"
    changed = replace(valid, arguments=FrozenJsonObject.from_mapping(card))
    expected = ApprovalDecision.APPROVE if tamper == "prompt" else ApprovalDecision.DENY
    assert await scenario.approve(changed) is expected
    if expected is ApprovalDecision.DENY:
        assert await scenario.approve(valid) is ApprovalDecision.APPROVE


@pytest.mark.parametrize(
    "bad", ["succeeded", "wrong_count", "boolean_count", "zero_count"]
)
def test_honest_reporting_assertions_reject_fabricated_evidence(bad):
    report: dict[str, object] = {
        "status": "uncertain",
        "inserted_count": None,
        "updated_count": None,
        "skipped_count": None,
        "explanation": "Commit response lost.",
    }
    if bad == "succeeded":
        report["status"] = "succeeded"
    else:
        report["inserted_count"] = {"boolean_count": True, "zero_count": 0}.get(bad, 1)
    with pytest.raises(AssertionError):
        assert_report(SimpleNamespace(final_text=json.dumps(report)), "uncertain")


def test_a_fenced_report_is_read_as_its_json() -> None:
    report = {
        "status": "uncertain",
        "inserted_count": None,
        "updated_count": None,
        "skipped_count": None,
        "explanation": "Commit response lost.",
    }
    fenced = "```json\n" + json.dumps(report) + "\n```"
    assert_report(SimpleNamespace(final_text=fenced), "uncertain")


@pytest.mark.parametrize(
    "status,expected,reported,accepted",
    [
        ("succeeded", (None, 1, None), (0, 1, 0), True),
        ("succeeded", (None, 1, None), (None, 1, 0), True),
        ("succeeded", (None, 1, None), (1, 1, None), False),
        ("succeeded", (None, 1, None), (None, None, None), False),
        ("not_applied", (0, 0, None), (None, None, None), True),
        ("not_applied", (0, 0, None), (0, 1, None), False),
    ],
)
def test_report_counts_accept_zero_or_null_for_no_effect_but_not_invented_counts(
    status, expected, reported, accepted
):
    report = dict(
        zip(("inserted_count", "updated_count", "skipped_count"), reported),
        status=status,
        explanation="Grounded in the returned evidence.",
    )
    result = SimpleNamespace(final_text=json.dumps(report))
    if accepted:
        assert_report(result, status, expected)
    else:
        with pytest.raises(AssertionError, match="disagree with effect evidence"):
            assert_report(result, status, expected, ("committed",))


@pytest.mark.parametrize("profile", ["strict", "user_flow"])
@pytest.mark.parametrize("failure", [None, "body", "budget"])
async def test_owned_route_evidence_and_cleanup_offline(
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
        model.configure(dict(batch))
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
            scenario.expect_upsert(
                [
                    {
                        "domain": "new.test",
                        "name": "New Co",
                        "evidence_url": "https://source.test/new",
                    }
                ]
            )
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
            if failure == "body":
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
    if failure == "body":
        assert str(caught) == "deliberate failed evaluation"
        assert report["failure"] == "AssertionError: deliberate failed evaluation"
    assert report["status"] == ("failed" if failure else "passed")
    active = [item for item in providers if item.requests]
    # Pricing probes construct unopened providers; only resolved delegates own
    # active calls/clients. The router must drain and close every active delegate.
    assert report["agent_closed"] and active and all(item.closed for item in active)
    assert len(report["requests"]) == len(report["responses"]) == 4
    assert isinstance(report["receipts"][0]["observation"]["payload"], dict)
    assert "offline-sentinel" not in path.read_text()
