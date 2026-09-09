"""Nine real-LLM + real-PostgreSQL cases, at most 13 runs/model/repetition.

Only setup and explicit database faults are deterministic. These tests never
start Docker, script model responses, inject previews, or seed routine proposals.
"""

import json
import os
from dataclasses import replace

import pytest
from _distribution_support import no_artifact_outcome_contract
from _postgresql_live_llm_support import (
    AUTHORIZATION,
    EXPIRES,
    FIXTURE_SENSITIVITY,
    LIMITS,
    NEXT_SLOT,
    REPORT_INSTRUCTION,
    assert_exact_preview,
    assert_preview_binding,
    assert_report,
    calls_for,
    evaluate,
    model_ids,
    repeats,
    report_path,
)
from test_postgresql_write_release import database, row, _TABLE

from daita import EffectRequirement
from daita.capabilities import EffectEvidenceBasis, EffectOutcome
from daita.distribution.models import OutcomeState, outcome_contract_projection
from daita.storage.sqlite_records import EffectResolutionDecision

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get(AUTHORIZATION) != "1",
        reason=(
            f"Set {AUTHORIZATION}=1 only after authorizing real API calls and the disposable "
            "postgres-large canary: at most 13 runs, $1.95 estimated per model/repetition"
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
    path = report_path(request.node.nodeid).resolve()
    record_property("postgresql_live_llm_evidence", str(path))

    def start(*, max_runs=1):
        return evaluate(
            database,
            monkeypatch,
            model_id,
            path,
            request.node.nodeid,
            max_runs=max_runs,
        )

    return start


@pytest.mark.parametrize("phrasing", [0, 1])
async def test_live_source_free_update(database, scenario_factory, phrasing):
    await database.permissions("update", max_rows=1)
    await database.seed()
    async with scenario_factory() as scenario:
        scenario.expect_update()
        result, transcript = await scenario.run(
            (
                "Find the companies table and change the name of existing.test to Updated. Change exactly that one name; preserve every other value.",
                "Our stored company existing.test needs its name corrected to Updated. Locate the record and save that correction, leaving all other fields alone.",
            )[phrasing]
            + " "
            + REPORT_INSTRUCTION
        )
        assert_exact_preview(scenario, transcript, "update")
        receipt = await database.receipt(EffectOutcome.SUCCEEDED)
        assert receipt.payload["affected_rows"] == 1
        assert await database.rows() == [
            {
                "id": 1,
                "domain": "existing.test",
                "name": "Updated",
                "evidence_url": "https://evidence.test/existing",
                "notes": "Retained",
            }
        ]
        assert database.probe.mutations == database.probe.server_commits == 1
        assert_report(result, "succeeded", (None, 1, None))


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
        result, transcript = await scenario.run(prompt)
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
        assert_report(result, "succeeded", (1, 1, 1))

        repeated, repeated_transcript = await scenario.run(
            "Recheck and apply the same exact findings as a new authorized operation. Obtain current evidence even if nothing changed. "
            + REPORT_INSTRUCTION
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
        assert_report(repeated, "succeeded", (0, 0, 3))


@pytest.mark.parametrize("denial", ["missing_permission", "approval_denied"])
async def test_live_write_denial_preserves_database(database, scenario_factory, denial):
    if denial == "missing_permission":
        await database.permissions("none")
    async with scenario_factory() as scenario:
        scenario.expect_upsert([row()])
        scenario.deny_write = denial == "approval_denied"
        result, transcript = await scenario.run(
            "Save one company in the companies table: domain new.test, name New, evidence_url https://evidence.test/new.test. Let the database assign its ID; preserve omitted fields. "
            + REPORT_INSTRUCTION
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
        assert_report(result, "not_applied")


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
            + REPORT_INSTRUCTION
        )
        assert_exact_preview(scenario, transcript, "update")
        assert [item["name"] for item in await database.rows()] == ["Concurrent writer"]
        assert not await scenario.agent.list_effects()
        assert database.probe.mutations == database.probe.commit_attempts == 0
        assert_report(result, "not_applied")


async def test_live_unique_constraint_failure_rolls_back_whole_batch(
    database, scenario_factory
):
    rows = [
        row("a.test"),
        {**row("b.test"), "evidence_url": "https://evidence.test/a.test"},
    ]
    async with scenario_factory() as scenario:
        scenario.expect_upsert(rows)
        result, transcript = await scenario.run(
            "Attempt to save these two company findings together in one atomic batch in the companies table. Match on domain. Preserve the supplied evidence URLs exactly, even though they are equal; report any constraint failure and do not repair or split the batch: "
            + json.dumps(rows)
            + " "
            + REPORT_INSTRUCTION
        )
        assert_exact_preview(scenario, transcript)
        receipt = await database.receipt(EffectOutcome.NOT_APPLIED)
        assert receipt.payload["normalized_error_code"] == "write_constraint_violation"
        assert (
            receipt.payload["inserted_count"] == receipt.payload["updated_count"] == 0
        )
        assert await database.rows() == []
        assert database.probe.mutations == 1 and database.probe.commit_attempts == 0
        assert_report(result, "not_applied", (0, 0, None))


async def test_live_uncertain_commit_restart_and_human_recovery(
    database, scenario_factory
):
    database.probe.mode = "disconnect_after_commit"
    async with scenario_factory(max_runs=2) as scenario:
        scenario.expect_upsert([row()])
        result, transcript = await scenario.run(
            "Save one company in the companies table, matching domain: new.test, name New, evidence_url https://evidence.test/new.test. Let PostgreSQL assign the ID. "
            + REPORT_INSTRUCTION
        )
        assert_exact_preview(scenario, transcript)
        receipt = await database.receipt(EffectOutcome.UNCERTAIN)
        assert_report(result, "uncertain")
        actual = await database.rows()
        assert len(actual) == 1 and actual[0]["domain"] == "new.test"
        assert database.probe.commit_attempts == database.probe.server_commits == 1
        await scenario.reopen()
        assert await scenario.agent.inspect_effect(receipt.receipt_id) == receipt
        blocked, _ = await scenario.run(
            "Please finish my original company-saving request now. Do not resolve uncertainty or change connector permissions. "
            + REPORT_INSTRUCTION
        )
        assert_report(blocked, "uncertain")
        assert len(await scenario.agent.list_effects()) == 1
        assert await database.rows() == actual and database.probe.commit_attempts == 1

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


async def test_live_model_authors_and_executes_immediate_weekly_upsert(
    database, scenario_factory, model_id
):
    await database.permissions("upsert", max_rows=1)
    async with scenario_factory(max_runs=3) as scenario:
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
            "per_run_max_tokens": 30000,
            "per_run_max_cost_usd": "0.15",
            "cumulative_max_tokens": 60000,
            "cumulative_max_cost_usd": "0.30",
            "cumulative_max_attempts": 2,
            "cumulative_max_occurrences": 2,
            "maximum_consecutive_failures": 1,
            "expires_at": EXPIRES.isoformat(),
            "outcome_contract": outcome_contract_projection(outcome),
        }
        creation, _ = await scenario.run(
            "Create one routine to maintain this owner-supplied company finding in the companies table immediately and every Monday at 09:00 America/Chicago: "
            "domain new.test, name New, evidence_url https://evidence.test/new.test. Match domain, insert missing rows and update only name/evidence_url; preserve omitted notes and let PostgreSQL assign IDs. "
            "Each occurrence must obtain a fresh preview and invoke one bounded upsert, even when unchanged. Grant one row and one upsert per occurrence. "
            "Use only the preview and upsert capabilities, this exact source/table, no other connectors, skills, artifacts, or precheck. "
            "Discover the current conversation inbox and always report there. Require one adapter-verified successful upsert and a terminal conclusion with current-run provenance/exact bindings; "
            f"restrict sensitivity to {FIXTURE_SENSITIVITY.value}. "
            "Use latest_only misfires, skip nonexistent times, and the first ambiguous time. Stop after two occurrences/two attempts, or "
            + EXPIRES.isoformat()
            + ". Allow one consecutive failure. "
            "Authorize 30000 tokens and $0.15 per run, 60000 tokens and $0.30 cumulatively. Do not perform a separate foreground write. "
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
        routines = await scenario.agent.list_routines()
        assert len(routines) == 1, creation.final_text
        inspection = await scenario.agent.inspect_routine(routines[0].routine_id)
        assert len(inspection.routine.capability_grants) == 1
        assert inspection.routine.capability_grants[0].max_calls_per_occurrence == 1
        immediate, immediate_transcript = await scenario.scheduled_result(1)
        assert_report(immediate, "succeeded", (1, 0, 0))
        immediate_call = assert_preview_binding(immediate_transcript)
        scenario.clock = NEXT_SLOT
        scenario.agent._embedded._routine_supervisor.wake()
        weekly, weekly_transcript = await scenario.scheduled_result(2)
        assert_report(weekly, "succeeded", (0, 0, 1))
        weekly_call = assert_preview_binding(weekly_transcript)
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
        assert len(scenario.approvals) == 1 and scenario.approvals[0]["approved"]
        inbox = await scenario.agent.inbox(conversation_id=scenario.conversation_id)
        assert len(inbox) == 2
        for item in inbox:
            delivery = await scenario.agent.inspect_delivery(item.delivery_id)
            assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
