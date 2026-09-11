"""Real-model PostgreSQL acceptance with strict and ordinary-user profiles.

Only setup and explicit database faults are deterministic. These tests never
start Docker or script evaluated model responses. Independent recovery and
scheduled cases explicitly seed owner fixtures; they do not prove model creation.
"""

import json
import os
from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

import pytest
from _distribution_support import no_artifact_outcome_contract
from _postgresql_live_llm_support import (
    AUTHORIZATION,
    EXPIRES,
    FIXTURE_SENSITIVITY,
    NEXT_SLOT,
    NOW,
    REPORT_INSTRUCTION,
    LIMITS,
    USER_FLOW_LIMITS,
    assert_exact_preview,
    assert_preview_binding,
    calls_for,
    evaluate,
    evaluation_profile,
    model_ids,
    repeats,
    report_path,
)
from test_postgresql_write_release import database, row, _TABLE

from daita import EffectRequirement, ScheduledRoutineDraft
from daita._json import FrozenJsonObject
from daita.capabilities import EffectEvidenceBasis, EffectOutcome
from daita.distribution.models import OutcomeState, outcome_contract_projection
from daita.storage.sqlite_records import EffectResolutionDecision
from daita.llm.models import CanonicalMessage, MessageRole, TextBlock
from daita.loop.models import RunInput, LoopExit, LoopExitKind
from daita.routines.capabilities import _parsed_spec

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get(AUTHORIZATION) != "1",
        reason=(
            f"Set {AUTHORIZATION}=1 only after authorizing real API calls and the disposable "
            "postgres-large canary: strict 17 runs/$2.55; user_flow 19 runs/$9.50 "
            "estimated per model/repetition"
        ),
    ),
]


@pytest.fixture(params=model_ids())
def model_id(request):
    return request.param


@pytest.fixture(params=range(repeats()))
def repetition(request):
    return request.param


@pytest.fixture
def scenario_factory(
    database, monkeypatch, model_id, repetition, request, record_property
):
    profile = evaluation_profile()
    path = report_path(request.node.nodeid).resolve()
    if profile == "user_flow":
        path = path.parent / profile / path.name
    record_property("postgresql_live_llm_evidence", str(path))

    def start(*, max_runs=1, setup_mode="model_authored", origin=None):
        return evaluate(
            database,
            monkeypatch,
            model_id,
            path,
            request.node.nodeid,
            max_runs=max_runs,
            setup_mode=setup_mode,
            limits=USER_FLOW_LIMITS if profile == "user_flow" else LIMITS,
            profile=profile,
            origin=origin,
        )

    return start


@pytest.mark.parametrize("phrasing", [0, 1, 2])
async def test_live_source_free_update(database, scenario_factory, phrasing):
    await database.permissions("update", max_rows=1)
    await database.seed()
    if phrasing == 2:
        # A positive count on the edited column selects the wrong company.
        await database.admin.execute(
            f"INSERT INTO {_TABLE} (domain, name, evidence_url) VALUES "
            "('other.test', 'existing.test', 'https://evidence.test/other')"
        )
    original = await database.rows()
    async with scenario_factory() as scenario:
        scenario.expect_update()
        result, transcript = await scenario.run(
            (
                "Find the companies table and change the name of existing.test to Updated. Change exactly that one name; preserve every other value.",
                "Our stored company existing.test needs its name corrected to Updated. Locate the record and save that correction, leaving all other fields alone.",
                "Change the name of the company whose domain is existing.test to Updated. Preserve every other field and company.",
            )[phrasing]
            + " "
            + REPORT_INSTRUCTION,
            user_prompt=(
                "The company with domain existing.test is now called Updated. Please correct its name in our companies table and tell me what changed.",
                "Please rename the company whose website is existing.test to Updated in our records. Leave its other details alone.",
                "The company whose domain is existing.test should be called Updated. Please fix its name in the companies table without changing anything else.",
            )[phrasing],
        )
        assert_exact_preview(scenario, transcript, "update")
        receipt = await database.receipt(EffectOutcome.SUCCEEDED)
        assert receipt.payload["affected_rows"] == 1
        assert (
            await database.rows()
            == [
                {
                    "id": 1,
                    "domain": "existing.test",
                    "name": "Updated",
                    "evidence_url": "https://evidence.test/existing",
                    "notes": "Retained",
                }
            ]
            + original[1:]
        )
        assert database.probe.mutations == database.probe.server_commits == 1
        scenario.stages.append("exact_update_committed")
        scenario.assert_report(result, "succeeded", (None, 1, None))


@pytest.mark.skipif(
    evaluation_profile() != "user_flow", reason="Ordinary-user clarification flow"
)
async def test_live_ambiguous_company_then_user_clarifies(database, scenario_factory):
    await database.permissions("update", max_rows=1)
    await database.seed()
    await database.admin.execute(
        f"UPDATE {_TABLE} SET name = 'Shared Company' WHERE domain = 'existing.test'"
    )
    await database.admin.execute(
        f"INSERT INTO {_TABLE} (domain, name, evidence_url) VALUES "
        "('other.test', 'Shared Company', 'https://evidence.test/other')"
    )
    original = await database.rows()
    async with scenario_factory(max_runs=2) as scenario:
        clarification, transcript = await scenario.run(
            "", user_prompt="Rename Shared Company to Updated in our companies table."
        )
        scenario.assert_completed_runs()
        assert not calls_for(transcript, "data_update_rows")
        assert not calls_for(transcript, "data_upsert_rows")
        assert not scenario.approvals and not await scenario.agent.list_effects()
        assert await database.rows() == original
        assert database.probe.mutations == database.probe.commit_attempts == 0
        scenario.assert_report(clarification, "clarification_needed")
        scenario.stages.append("ambiguous_request_completed_without_effect_attempt")
        scenario.expect_update(intended_row=original[0])
        result, transcript = await scenario.run(
            "", user_prompt="I mean the one whose domain is existing.test."
        )
        assert_exact_preview(scenario, transcript, "update")
        receipt = await database.receipt(EffectOutcome.SUCCEEDED)
        assert receipt.payload["affected_rows"] == 1
        assert await database.rows() == [
            {**original[0], "name": "Updated"},
            original[1],
        ]
        assert database.probe.mutations == database.probe.server_commits == 1
        scenario.assert_report(result, "succeeded", (None, 1, None))
        scenario.stages.append("user_clarification_selected_exact_company")


async def test_live_mixed_upsert_and_unchanged_repeat(database, scenario_factory):
    await database.seed()
    await database.admin.execute(
        f"INSERT INTO {_TABLE} (domain, name, evidence_url) VALUES ('stable.test', 'Stable', 'https://evidence.test/stable.test')"
    )
    rows = [row("existing.test", "After"), row(), row("stable.test", "Stable")]
    rows[0]["evidence_url"] = "https://evidence.test/existing"
    async with scenario_factory(max_runs=2) as scenario:
        scenario.expect_upsert(rows)
        prompt = (
            "Save these company findings in the companies table as one batch, matching domain. "
            "Insert missing companies and update name/evidence_url on existing ones; preserve omitted notes, "
            "and let PostgreSQL generate IDs for new companies. Findings: "
            + json.dumps(rows)
            + " "
            + REPORT_INSTRUCTION
        )
        result, transcript = await scenario.run(
            prompt,
            user_prompt=(
                "Please save these company details in our companies table. Add any missing "
                "companies and bring the names and evidence links for existing domains up "
                "to date. Leave other details alone. Tell me what changed. "
                + json.dumps(rows)
            ),
        )
        assert_exact_preview(scenario, transcript)
        receipt = await database.receipt(EffectOutcome.SUCCEEDED)
        assert tuple(
            receipt.payload[key]
            for key in ("inserted_count", "updated_count", "unchanged_count")
        ) == (1, 1, 1)
        actual = await database.rows()
        assert [(item["domain"], item["name"]) for item in actual] == [
            ("existing.test", "After"),
            ("new.test", "New"),
            ("stable.test", "Stable"),
        ]
        assert actual[0]["id"] == 1 and actual[0]["notes"] == "Retained"
        assert actual[1]["id"] > 2 and actual[1]["notes"] is None
        scenario.assert_report(result, "succeeded", (1, 1, 1))

        repeated, repeated_transcript = await scenario.run(
            "Recheck and apply the same exact findings as a new authorized operation. Obtain current evidence even if nothing changed. "
            + REPORT_INSTRUCTION,
            user_prompt="Please check and save that same batch again, even if nothing has changed, and tell me the result.",
        )
        assert_exact_preview(scenario, repeated_transcript)
        receipts = await scenario.agent.list_effects()
        assert len(receipts) == 2
        unchanged = next(
            item for item in receipts if item.receipt_id != receipt.receipt_id
        )
        assert unchanged.outcome is EffectOutcome.SUCCEEDED
        assert tuple(
            unchanged.payload[key]
            for key in ("inserted_count", "updated_count", "unchanged_count")
        ) == (0, 0, 3)
        assert await database.rows() == actual
        assert database.probe.mutations == 2 and database.probe.server_commits == 2
        scenario.stages.append("mixed_and_unchanged_upserts_verified")
        scenario.assert_report(repeated, "succeeded", (0, 0, 3))


@pytest.mark.parametrize("denial", ["missing_permission", "approval_denied"])
async def test_live_write_denial_preserves_database(database, scenario_factory, denial):
    if denial == "missing_permission":
        await database.permissions("none")
    async with scenario_factory() as scenario:
        scenario.expect_upsert([row()])
        scenario.deny_write = denial == "approval_denied"
        result, transcript = await scenario.run(
            "Save one company in the companies table: domain new.test, name New, evidence_url https://evidence.test/new.test. Let the database assign its ID; preserve omitted fields. "
            + REPORT_INSTRUCTION,
            user_prompt="Please add new.test to our companies table with the name New and evidence link https://evidence.test/new.test. Tell me whether it was saved.",
        )
        assert await database.rows() == [] and not await scenario.agent.list_effects()
        assert database.probe.mutations == database.probe.commit_attempts == 0
        if denial == "missing_permission":
            assert not scenario.approvals
            assert not (
                await scenario.agent.inspect_source_permissions(database.source_id)
            ).state.relational_write_scopes
        else:
            assert (
                len(scenario.approvals) == 1 and not scenario.approvals[0]["approved"]
            )
            assert len(calls_for(transcript, "data_upsert_rows")) == 1
        scenario.stages.append(denial)
        scenario.assert_report(result, "not_applied")


async def test_live_drift_during_approval_does_not_reapply(database, scenario_factory):
    await database.permissions("update", max_rows=1)
    await database.seed()
    async with scenario_factory() as scenario:
        scenario.expect_update()

        async def change_row():
            await database.admin.execute(
                f"UPDATE {_TABLE} SET name = 'Concurrent writer' WHERE id = 1"
            )

        scenario.approval_hook = change_row
        result, transcript = await scenario.run(
            "Change only the name of existing.test to Updated in the companies table. If the target changes during approval, stop and report the conflict without overwriting it. "
            + REPORT_INSTRUCTION,
            user_prompt="Please change the name of the company whose domain is existing.test to Updated. Leave everything else alone and let me know the result.",
        )
        assert_exact_preview(scenario, transcript, "update")
        assert [item["name"] for item in await database.rows()] == ["Concurrent writer"]
        assert not await scenario.agent.list_effects()
        assert database.probe.mutations == database.probe.commit_attempts == 0
        assert scenario.approval_hook is None
        scenario.stages.append("approval_drift_hook_executed_and_write_blocked")
        scenario.assert_report(result, "not_applied")


async def test_live_unique_constraint_failure_rolls_back_whole_batch(
    database, scenario_factory
):
    rows = [
        row("a.test"),
        {**row("b.test"), "evidence_url": "https://evidence.test/a.test"},
    ]
    async with scenario_factory() as scenario:
        if scenario.profile == "user_flow":
            # Keep the user's input valid. A concurrent writer takes the second
            # evidence URL after preview, so real PostgreSQL must roll back the
            # first insert as well. The model receives no fault-specific hint.
            rows = [row("a.test"), row("b.test")]

            async def occupy_evidence_url():
                await database.admin.execute(
                    f"INSERT INTO {_TABLE} (domain, name, evidence_url) VALUES "
                    "('collision.test', 'Concurrent writer', 'https://evidence.test/b.test')"
                )

            scenario.approval_hook = occupy_evidence_url
        scenario.expect_upsert(rows)
        result, transcript = await scenario.run(
            "Attempt to save these two company findings together in one atomic batch in the companies table. Match on domain. Preserve the supplied evidence URLs exactly, even though they are equal; report any constraint failure and do not repair or split the batch: "
            + json.dumps(rows)
            + " "
            + REPORT_INSTRUCTION,
            user_prompt=(
                "Please save these two companies together. Keep the details exactly as "
                "supplied; if they cannot both be saved, leave the table unchanged and "
                "tell me why. " + json.dumps(rows)
            ),
        )
        assert_exact_preview(scenario, transcript)
        receipt = await database.receipt(EffectOutcome.NOT_APPLIED)
        assert receipt.payload["normalized_error_code"] == "write_constraint_violation"
        assert (
            receipt.payload["inserted_count"] == receipt.payload["updated_count"] == 0
        )
        assert await database.rows() == (
            [
                {
                    "id": 1,
                    "domain": "collision.test",
                    "name": "Concurrent writer",
                    "evidence_url": "https://evidence.test/b.test",
                    "notes": None,
                }
            ]
            if scenario.profile == "user_flow"
            else []
        )
        assert database.probe.mutations == 1 and database.probe.commit_attempts == 0
        scenario.stages.append("constraint_rollback_verified")
        scenario.assert_report(result, "not_applied", (0, 0, None))


@pytest.mark.parametrize("setup", ["model_authored", "injected_commit_loss_fixture"])
async def test_live_uncertain_commit_restart_and_human_recovery(
    database, scenario_factory, setup
):
    database.probe.mode = "disconnect_after_commit"
    original_prompt = (
        "Please add the company with domain new.test, name New and evidence link "
        "https://evidence.test/new.test to our companies table. Tell me whether it was saved."
    )
    origin = None
    if setup == "injected_commit_loss_fixture":
        database.upsert([row()])
        origin = await database.agent.run(original_prompt)
        assert origin.kind is LoopExitKind.COMPLETED
        assert len(database.model.requests) == 4
        await database.receipt(EffectOutcome.UNCERTAIN)
    async with scenario_factory(
        max_runs=2 if setup == "model_authored" else 1, setup_mode=setup, origin=origin
    ) as scenario:
        scenario.expect_upsert([row()])
        if setup == "model_authored":
            result, transcript = await scenario.run(
                "Save one company in the companies table, matching domain: new.test, name New, evidence_url https://evidence.test/new.test. Let PostgreSQL assign the ID. "
                + REPORT_INSTRUCTION,
                user_prompt=original_prompt,
            )
            assert_exact_preview(scenario, transcript)
        receipt = await database.receipt(EffectOutcome.UNCERTAIN)
        actual = await database.rows()
        assert len(actual) == 1 and actual[0]["domain"] == "new.test"
        assert database.probe.commit_attempts == database.probe.server_commits == 1
        scenario.stages.append("injected_commit_loss_receipt_and_row_verified")
        if setup == "model_authored":
            scenario.assert_report(result, "uncertain")
        await scenario.reopen()
        assert await scenario.agent.inspect_effect(receipt.receipt_id) == receipt
        blocked, _ = await scenario.run(
            "Please finish saving company new.test, name New, evidence_url https://evidence.test/new.test. "
            "The connection dropped before I received confirmation. Do not resolve uncertainty or change connector permissions. "
            + REPORT_INSTRUCTION,
            user_prompt=(
                "The connection dropped before I got confirmation. Please finish saving "
                "new.test with name New and evidence link https://evidence.test/new.test, "
                "and tell me whether it was saved."
            ),
        )
        scenario.assert_report(blocked, "uncertain")
        assert len(await scenario.agent.list_effects()) == 1
        assert await database.rows() == actual and database.probe.commit_attempts == 1
        scenario.stages.append("restart_and_live_refusal_verified")

        await database.permissions("none")
        scenario.recovery_digest = receipt.receipt_digest
        request_count = sum(len(item.requests) for item in scenario.recordings)
        resolved = await scenario.agent.resolve_effect(
            receipt.receipt_id,
            expected_digest=receipt.receipt_digest,
            decision=EffectResolutionDecision.CLOSE_WITHOUT_RETRY,
            note="Fixture administrator verified the committed row; close without retry.",
        )
        assert (
            resolved.resolution.decision is EffectResolutionDecision.CLOSE_WITHOUT_RETRY
        )
        assert (
            resolved.outcome is EffectOutcome.UNCERTAIN
            and resolved.payload == receipt.payload
        )
        assert sum(len(item.requests) for item in scenario.recordings) == request_count
        assert database.probe.commit_attempts == 1 and await database.rows() == actual
        assert not (
            await scenario.agent.inspect_source_permissions(database.source_id)
        ).state.relational_write_scopes
        await scenario.reopen()
        assert await scenario.agent.inspect_effect(receipt.receipt_id) == resolved
        scenario.stages.append(
            "human_recovery_persisted_without_model_or_source_action"
        )


@pytest.mark.parametrize("setup", ["model_authored", "owner_admitted_routine"])
async def test_live_model_authors_and_executes_immediate_weekly_upsert(
    database, scenario_factory, model_id, setup
):
    await database.permissions("upsert", max_rows=1)
    async with scenario_factory(
        max_runs=3 if setup == "model_authored" else 2, setup_mode=setup
    ) as scenario:
        state = (
            await scenario.agent.inspect_source_permissions(database.source_id)
        ).state
        permission = state.relational_write_scopes[0]
        scenario.routine_grant = {
            "source_id": database.source_id,
            "resource_id": database.resource_id,
            "resource_revision": permission.resource_revision,
            "key_columns": ["domain"],
            "allowed_insert_columns": ["domain", "evidence_url", "name"],
            "allowed_update_columns": ["evidence_url", "name"],
            "generated_identity_columns": ["id"],
            "max_rows": 1,
        }
        outcome = replace(
            no_artifact_outcome_contract(FIXTURE_SENSITIVITY),
            require_exact_source_bindings=True,
            effect_requirements=(
                EffectRequirement(
                    "data.upsert_rows",
                    1,
                    frozenset({EffectEvidenceBasis.ADAPTER_VERIFIED}),
                ),
            ),
        )
        scenario.routine_contract = {
            "schedule": {
                "kind": "calendar",
                "timezone": "America/Chicago",
                "hour": 9,
                "minute": 0,
                "day_selector": "weekdays",
                "weekdays": [1],
                "month_days": [],
                "months": [],
                "nonexistent_time_policy": "skip",
                "ambiguous_time_policy": "first",
            },
            "run_immediately": True,
            "allowed_source_ids": [database.source_id],
            "allowed_resource_ids": [database.resource_id],
            "allowed_connector_binding_ids": [],
            "allowed_capability_ids": ["data.preview_upsert_rows", "data.upsert_rows"],
            "eligible_model_routes": [model_id],
            "sensitivity_ceiling": FIXTURE_SENSITIVITY.value,
            "skill_bindings": [],
            "precheck": None,
            "misfire_policy": "latest_only",
            "reporting_mode": "always",
            **scenario.routine_budgets,
            "maximum_consecutive_failures": 1,
            "expires_at": EXPIRES.isoformat(),
            "outcome_contract": outcome_contract_projection(outcome),
        }
        prompt = (
            "Create one routine to maintain this owner-supplied company finding in the companies table immediately and every Monday at 09:00 America/Chicago: "
            "domain new.test, name New, evidence_url https://evidence.test/new.test. Match domain, insert missing rows and update only name/evidence_url; preserve omitted notes and let PostgreSQL assign IDs. "
            "Each occurrence must obtain a fresh preview and invoke one bounded upsert, even when unchanged. Grant one row and one upsert per occurrence. "
            "Use only the preview and upsert capabilities, this exact source/table, no other connectors, skills, artifacts, or precheck. "
            "Discover the current conversation inbox and always report there. Require one adapter-verified successful upsert and a terminal conclusion with current-run provenance/exact bindings; "
            f"restrict sensitivity to {FIXTURE_SENSITIVITY.value}. "
            "Use latest_only misfires, skip nonexistent times, and the first ambiguous time. Stop after two occurrences/two attempts, or "
            + EXPIRES.isoformat()
            + ". Allow one consecutive failure. "
            f"Authorize {scenario.routine_budgets['per_run_max_tokens']} tokens and ${scenario.routine_budgets['per_run_max_cost_usd']} per run, "
            f"{scenario.routine_budgets['cumulative_max_tokens']} tokens and ${scenario.routine_budgets['cumulative_max_cost_usd']} cumulatively. Do not perform a separate foreground write. "
            "Save this reporting instruction in the routine: "
            + REPORT_INSTRUCTION
            + " "
            "Current owner-provided target references (discover all tool and grant contracts yourself): "
            + json.dumps(
                {
                    "source_id": database.source_id,
                    "resource_id": database.resource_id,
                    "model_route": model_id,
                }
            )
        )
        user_prompt = (
            "Please keep company new.test in our companies table with name New and "
            "evidence link https://evidence.test/new.test. Add it if missing, otherwise "
            "update its name and evidence link, leaving other details alone. Do this now "
            "and every Monday at 9am America/Chicago. Check and save the record each time, "
            "even when unchanged, and report the result here after every run. "
            "Only use this company table. Allow one company-saving operation per run. "
            "Stop after two runs, after any failure, or on "
            + EXPIRES.isoformat()
            + ". "
            "If a run is missed, only catch up the most recent one. Skip nonexistent clock "
            "times and use the first occurrence of an ambiguous clock time. "
            f"Use the current model with a budget of {scenario.routine_budgets['per_run_max_tokens']} tokens "
            f"and ${scenario.routine_budgets['per_run_max_cost_usd']} per run, "
            f"{scenario.routine_budgets['cumulative_max_tokens']} tokens and "
            f"${scenario.routine_budgets['cumulative_max_cost_usd']} overall. "
            "Keep the work at the table's current sensitivity. Report success only with "
            "database-confirmed evidence for the exact table. Include the immediate save "
            "in the scheduled work, rather than saving it separately first."
        )
        if setup == "model_authored":
            creation, _ = await scenario.run(prompt, user_prompt=user_prompt)
            scenario.assert_completed_runs()
            if scenario.profile == "user_flow":
                scenario.assert_report(creation, "assignment_saved")
        else:
            await admit_owner_routine(
                scenario, user_prompt if scenario.profile == "user_flow" else prompt
            )
            creation = None
            scenario.stages.append("owner_admitted_fixture_not_model_creation")
        routines = await scenario.agent.list_routines()
        assert len(routines) == 1, creation
        inspection = await scenario.agent.inspect_routine(routines[0].routine_id)
        assert len(inspection.routine.capability_grants) == 1
        assert inspection.routine.capability_grants[0].max_calls_per_occurrence == 1
        if setup == "model_authored":
            scenario.stages.append("model_authored_assignment_saved_with_exact_grant")
        immediate, immediate_transcript = await scenario.scheduled_result(1)
        scenario.assert_report(immediate, "succeeded", (1, 0, 0))
        immediate_call = assert_preview_binding(immediate_transcript)
        scenario.stages.append("immediate_report_and_preview_binding_verified")
        scenario.clock = NEXT_SLOT
        scenario.agent._embedded._routine_supervisor.wake()
        weekly, weekly_transcript = await scenario.scheduled_result(2)
        scenario.assert_report(weekly, "succeeded", (0, 0, 1))
        weekly_call = assert_preview_binding(weekly_transcript)
        scenario.stages.append("weekly_report_and_preview_binding_verified")
        receipts = await scenario.agent.list_effects()
        assert len(receipts) == 2
        assert all(
            item.outcome is EffectOutcome.SUCCEEDED
            and item.evidence_basis is EffectEvidenceBasis.ADAPTER_VERIFIED
            for item in receipts
        )
        assert database.probe.mutations == 1 and database.probe.server_commits == 2
        for call, run, counts in (
            (immediate_call, immediate, (1, 0, 0)),
            (weekly_call, weekly, (0, 0, 1)),
        ):
            receipt = next(item for item in receipts if item.run_id == run.run_id)
            assert receipt.call_id == call.id
            assert (
                receipt.payload["preview_fingerprint"]
                == call.arguments["preview_fingerprint"]
            )
            assert (
                tuple(
                    receipt.payload[key]
                    for key in ("inserted_count", "updated_count", "unchanged_count")
                )
                == counts
            )
        assert await database.rows() == [
            {
                "id": 1,
                "domain": "new.test",
                "name": "New",
                "evidence_url": "https://evidence.test/new.test",
                "notes": None,
            }
        ]
        if setup == "model_authored":
            assert len(scenario.approvals) == 1 and scenario.approvals[0]["approved"]
        else:
            assert scenario.approvals == []
        inbox = await scenario.agent.inbox(conversation_id=scenario.conversation_id)
        assert len(inbox) == 2
        for item in inbox:
            delivery = await scenario.agent.inspect_delivery(item.delivery_id)
            assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        scenario.stages.append("two_exact_receipts_and_inbox_deliveries_verified")


async def admit_owner_routine(scenario, owner_prompt):
    """Seed foreground owner provenance, then admit through the public Agent API.

    This follows the independent scheduled-live fixture pattern. The seeded
    transcript is explicitly fixture data, excluded from measured live usage.
    It establishes no model-authored creation or discovery effectiveness.
    """
    assert scenario.setup_mode == "owner_admitted_routine"
    origin = RunInput(
        id=f"run-{uuid4().hex}",
        agent_id=scenario.agent.id,
        conversation_id=f"conversation-{uuid4().hex}",
        message=owner_prompt,
        created_at=NOW,
        history_sensitivity=FIXTURE_SENSITIVITY,
    )
    assert origin.conversation_id is not None
    store = scenario.agent._embedded._store
    await store.start(origin)
    await store.append(origin.id, origin.start_message())
    await store.complete(
        LoopExit(
            run_id=origin.id,
            conversation_id=origin.conversation_id,
            created_at=NOW,
            kind=LoopExitKind.COMPLETED,
            reason="completed",
            final_text="Owner fixture authorization recorded.",
            sensitivity=FIXTURE_SENSITIVITY,
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("Owner fixture authorization recorded."),),
        ),
    )
    scenario.conversation_id = origin.conversation_id
    destination = (
        await scenario.agent.distribution_destinations(
            origin.conversation_id, sensitivity_ceiling=FIXTURE_SENSITIVITY
        )
    )[0]
    arguments = {
        **scenario.routine_contract,
        "title": "Owner-authorized company finding",
        "authorized_instruction": (
            "Maintain the owner-supplied finding in the exact admitted companies table: "
            "domain new.test, name New, evidence_url https://evidence.test/new.test. "
            "Obtain a current preview; upsert exactly this one row by domain, insert missing "
            "rows and update only name/evidence_url, preserve omitted notes and let PostgreSQL "
            "generate IDs. Invoke once even if unchanged. " + REPORT_INSTRUCTION
        ),
        "distribution_destination_id": destination.destination_id,
        "skill_names": [],
        "requested_capability_grants": [
            {
                "capability_id": "data.upsert_rows",
                "constraints": scenario.routine_grant,
                "max_calls_per_occurrence": 1,
            }
        ],
    }
    if scenario.profile == "user_flow":
        # This owner-authored natural fixture explicitly authorizes schema access
        # to the same frozen table. It is not inferred by routine admission.
        arguments["allowed_capability_ids"] = sorted(
            {*arguments["allowed_capability_ids"], "catalog.schema"}
        )
        arguments["authorized_instruction"] = (
            "Keep company new.test in our companies table with name New and evidence "
            "link https://evidence.test/new.test. Add it if missing, otherwise update "
            "its name and evidence link, leaving other details alone. Check and save "
            "it even when unchanged, and tell me what changed."
        )
    parsed = cast(
        dict[str, Any], dict(_parsed_spec(FrozenJsonObject.from_mapping(arguments)))
    )
    parsed.pop("basis_run_id")
    draft = ScheduledRoutineDraft(origin_run_id=origin.id, **parsed)
    proposal = await scenario.agent.propose_routine(draft)
    return await scenario.agent.create_routine(proposal)
