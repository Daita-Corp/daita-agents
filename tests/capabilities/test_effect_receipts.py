from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import timedelta

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import EffectEvidenceBasis, EffectObservation, EffectOutcome
from daita.llm.models import ModelSensitivity
from daita.loop.models import RunInput
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_codecs import decode_receipt, encode_receipt
from daita.storage.sqlite_records import (
    EffectReceipt,
    EffectReceiptConflictError,
    EffectResolution,
    EffectResolutionDecision,
    EffectUnresolvedError,
    effect_receipt_id,
)
from tests.capabilities._effect_receipt_support import STARTED_AT, _store

COMPLETED_AT = STARTED_AT + timedelta(seconds=1)
_DIGEST = "sha256:" + "3" * 64


def _started(
    call_id: str = "call-one", *, operation_key: str = _DIGEST
) -> EffectReceipt:
    agent_id, run_id = "agent-effect", "run-effect"
    return EffectReceipt(
        receipt_id=effect_receipt_id(
            agent_id=agent_id,
            run_id=run_id,
            call_id=call_id,
            operation_key=operation_key,
        ),
        receipt_kind="test.rows",
        agent_id=agent_id,
        run_id=run_id,
        call_id=call_id,
        capability_id="test.effect",
        domain_owner_id="test",
        capability_contract_digest=_DIGEST,
        operation_key=operation_key,
        argument_fingerprint=_DIGEST,
        sensitivity=ModelSensitivity.INTERNAL,
        started_at=STARTED_AT,
    )


@pytest.mark.parametrize(
    "outcome,basis,payload",
    (
        (
            EffectOutcome.SUCCEEDED,
            EffectEvidenceBasis.ADAPTER_VERIFIED,
            {"affected_rows": 7},
        ),
        (
            EffectOutcome.SUCCEEDED,
            EffectEvidenceBasis.SERVER_REPORTED,
            {"invocation_completed": True},
        ),
        (
            EffectOutcome.NOT_APPLIED,
            EffectEvidenceBasis.ADAPTER_VERIFIED,
            {"affected_rows": 0},
        ),
        (EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED, None),
        (EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN, None),
    ),
)
async def test_receipt_terminal_observation_is_durable_and_immutable(
    tmp_path, outcome, basis, payload
):
    store = await _store(tmp_path / "state.db")
    started = _started()
    try:
        assert await store.start_effect_receipt(started) == started
        assert (
            await store.load_effect_receipt(started.agent_id, started.receipt_id)
            == started
        )
        assert (
            await store.load_effect_receipt_for_call(
                started.agent_id, started.run_id, started.call_id
            )
            == started
        )
        assert (
            await store.load_effect_receipt_for_operation(
                started.agent_id, started.operation_key
            )
            == started
        )
        terminal = started.finish(
            EffectObservation(
                outcome,
                basis,
                None if payload is None else FrozenJsonObject.from_mapping(payload),
            ),
            finished_at=COMPLETED_AT,
        )
        assert await store.finish_effect_receipt(terminal) == terminal
        assert await store.finish_effect_receipt(terminal) == terminal
        conflicting = started.finish(
            EffectObservation(EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN),
            finished_at=COMPLETED_AT + timedelta(seconds=1),
        )
        with pytest.raises(EffectReceiptConflictError, match="immutable"):
            await store.finish_effect_receipt(conflicting)
        assert decode_receipt(encode_receipt(terminal)) == terminal
    finally:
        await store.close()


async def test_reservation_requires_owned_live_run_and_prevents_call_and_operation_duplicates(
    tmp_path,
):
    store = await _store(tmp_path / "state.db")
    started = _started()
    try:
        foreign = replace(
            started,
            agent_id="foreign",
            receipt_id=effect_receipt_id(
                agent_id="foreign",
                run_id=started.run_id,
                call_id=started.call_id,
                operation_key=started.operation_key,
            ),
        )
        with pytest.raises(EffectReceiptConflictError, match="owned nonterminal run"):
            await store.start_effect_receipt(foreign)
        with pytest.raises(ValueError, match="terminal observation"):
            await store.finish_effect_receipt(started)
        await store.start_effect_receipt(started)
        with pytest.raises(EffectUnresolvedError):
            await store.start_effect_receipt(_started("other-call"))
        terminal = started.finish(
            EffectObservation(
                EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
            ),
            finished_at=COMPLETED_AT,
        )
        await store.finish_effect_receipt(terminal)
        for duplicate in (
            started,
            _started("other-call"),
            _started(operation_key="sha256:" + "9" * 64),
        ):
            with pytest.raises(
                EffectReceiptConflictError, match="already has a receipt"
            ):
                await store.start_effect_receipt(duplicate)
        with pytest.raises(EffectReceiptConflictError, match="bound is exhausted"):
            await store.start_effect_receipt(
                _started("different", operation_key="sha256:" + "4" * 64),
                max_receipts_per_run=1,
            )
    finally:
        await store.close()


async def test_started_recovery_blocks_after_restart_and_history_clear_then_preserves_human_resolution(
    tmp_path,
):
    path = tmp_path / "state.db"
    store = await _store(path)
    started = _started()
    await store.start_effect_receipt(started)
    await store.close()
    reopened = await SQLiteStateStore.open(path, clock=lambda: COMPLETED_AT)
    try:
        recovered = await reopened.load_effect_receipt(
            started.agent_id, started.receipt_id
        )
        assert recovered is not None
        assert (
            recovered.outcome is EffectOutcome.UNCERTAIN and recovered.payload is None
        )
        assert (
            recovered.finished_at == COMPLETED_AT and recovered.started_at == STARTED_AT
        )
        await reopened.clear_conversations(started.agent_id)
        with pytest.raises(EffectUnresolvedError) as caught:
            await reopened.require_effects_unblocked(started.agent_id)
        assert caught.value.receipt_ids == (started.receipt_id,)
        assert caught.value.omitted_count == 0
        assert await reopened.load_effect_receipt("foreign", started.receipt_id) is None
        resolution = EffectResolution(
            receipt_id=recovered.receipt_id,
            receipt_digest=recovered.receipt_digest,
            decision=EffectResolutionDecision.ALLOW_FUTURE_WORK,
            approving_principal_id=started.agent_id,
            control_id="foreground-control",
            resolved_at=COMPLETED_AT,
            note="Checked the remote system; accept possible duplication in future authorized work.",
        )
        with pytest.raises(EffectReceiptConflictError, match="observation changed"):
            await reopened.resolve_effect_receipt(
                started.agent_id,
                replace(resolution, receipt_digest="sha256:" + "f" * 64),
            )
        resolved = await reopened.resolve_effect_receipt(started.agent_id, resolution)
        assert resolved.receipt_digest == recovered.receipt_digest
        assert resolved.outcome is EffectOutcome.UNCERTAIN
        assert (
            await reopened.resolve_effect_receipt(started.agent_id, resolution)
            == resolved
        )
        await reopened.require_effects_unblocked(started.agent_id)
        with pytest.raises(EffectReceiptConflictError, match="immutable"):
            await reopened.resolve_effect_receipt(
                started.agent_id,
                replace(
                    resolution, decision=EffectResolutionDecision.CLOSE_WITHOUT_RETRY
                ),
            )
        assert decode_receipt(encode_receipt(resolved)) == resolved
    finally:
        await reopened.close()


async def test_receipt_persisted_shape_is_bounded_evidence_not_raw_arguments(tmp_path):
    path = tmp_path / "state.db"
    store = await _store(path)
    started = _started()
    await store.start_effect_receipt(started)
    await store.close()
    with sqlite3.connect(path) as connection:
        payload = connection.execute(
            "SELECT data FROM effect_receipts WHERE agent_id = ? AND id = ?",
            (started.agent_id, started.receipt_id),
        ).fetchone()[0]
    fields = json.loads(payload)["fields"]
    assert set(fields) == set(started.material()) | {"receipt_digest", "resolution"}
    assert fields["payload"] is None
    assert not {"sql", "password", "assignments", "arguments"}.intersection(fields)
    corrupt = json.loads(payload)
    corrupt["fields"]["sensitivity"] = "public"
    with pytest.raises(ValueError, match="digest"):
        decode_receipt(json.dumps(corrupt))


async def test_public_recovery_requires_exact_approval_and_keeps_original_observation(
    tmp_path,
):
    from daita import Agent
    from daita.capabilities import ApprovalDecision

    approvals = []
    permitted = False

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE if permitted else ApprovalDecision.DENY

    agent = await Agent.create(
        "effect-recovery",
        root=tmp_path,
        hosted=True,
        clock=lambda: COMPLETED_AT,
        approval_handler=approve,
    )
    try:
        original = _started()
        started = replace(
            original,
            agent_id=agent.id,
            receipt_id=effect_receipt_id(
                agent_id=agent.id,
                run_id=original.run_id,
                call_id=original.call_id,
                operation_key=original.operation_key,
            ),
        )
        await agent._embedded._store.start(
            RunInput(
                id=started.run_id,
                agent_id=agent.id,
                message="exact action",
                created_at=STARTED_AT,
                conversation_id="recovery-conversation",
            )
        )
        await agent._embedded._store.start_effect_receipt(started)
        uncertain = started.finish(
            EffectObservation(EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN),
            finished_at=COMPLETED_AT,
        )
        await agent._embedded._store.finish_effect_receipt(uncertain)
        assert await agent.inspect_effect(uncertain.receipt_id) == uncertain
        assert await agent.list_effects(unresolved_only=True) == (uncertain,)
        assert "resolve_effect" not in agent._embedded._capabilities.tool_names
        with pytest.raises(ValueError, match="expected observation"):
            await agent.resolve_effect(
                uncertain.receipt_id,
                expected_digest="sha256:" + "f" * 64,
                decision=EffectResolutionDecision.ALLOW_FUTURE_WORK,
                note="Accept future work.",
            )
        assert approvals == []
        with pytest.raises(PermissionError, match="denied"):
            await agent.resolve_effect(
                uncertain.receipt_id,
                expected_digest=uncertain.receipt_digest,
                decision=EffectResolutionDecision.ALLOW_FUTURE_WORK,
                note="Accept future work.",
            )
        assert await agent.inspect_effect(uncertain.receipt_id) == uncertain
        permitted = True
        resolved = await agent.resolve_effect(
            uncertain.receipt_id,
            expected_digest=uncertain.receipt_digest,
            decision=EffectResolutionDecision.ALLOW_FUTURE_WORK,
            note="Accept future work.",
        )
        assert resolved.resolution is not None
        assert (
            resolved.outcome is EffectOutcome.UNCERTAIN
            and resolved.receipt_digest == uncertain.receipt_digest
        )
        assert approvals[-1].arguments["receipt_digest"] == uncertain.receipt_digest
        assert approvals[-1].arguments["decision"] == "allow_future_work"
        assert (
            await agent.resolve_effect(
                uncertain.receipt_id,
                expected_digest=uncertain.receipt_digest,
                decision=EffectResolutionDecision.ALLOW_FUTURE_WORK,
                note="Accept future work.",
            )
            == resolved
        )
        assert len(approvals) == 2
        assert await agent.list_effects(unresolved_only=True) == ()
    finally:
        await agent.close()


async def test_reopened_foreground_context_reports_bounded_durable_effect_block(
    tmp_path,
):
    from daita import Agent
    from daita.capabilities import ApprovalDecision
    from daita.llm.models import FinishReason, ModelProfile, ModelResponse, TextBlock
    from daita.llm.providers.mock import MockModelProvider

    provider = MockModelProvider(
        tuple(
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Report current evidence."
            )
            for _ in range(2)
        )
    )
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=64000,
        max_output_tokens=1000,
        supports_tools=True,
    )

    async def approve(request):
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "effect-context",
        root=tmp_path,
        hosted=True,
        model=provider,
        model_profile=profile,
        approval_handler=approve,
    )
    receipts = []
    try:
        store = agent._embedded._store
        for index in range(21):
            original = _started(
                f"call-{index}", operation_key="sha256:" + f"{index:064x}"
            )
            started = replace(
                original,
                agent_id=agent.id,
                sensitivity=ModelSensitivity.RESTRICTED,
                receipt_id=effect_receipt_id(
                    agent_id=agent.id,
                    run_id=original.run_id,
                    call_id=original.call_id,
                    operation_key=original.operation_key,
                ),
            )
            # Fixture insertion models existing evidence; production admission correctly
            # forbids a new effect while an earlier one is unresolved.
            if index == 0:
                await store.start(
                    RunInput(
                        id=started.run_id,
                        agent_id=agent.id,
                        message="earlier action",
                        created_at=STARTED_AT,
                        conversation_id="earlier-conversation",
                    )
                )
            uncertain = started.finish(
                EffectObservation(
                    EffectOutcome.UNCERTAIN,
                    EffectEvidenceBasis.UNKNOWN,
                    payload=FrozenJsonObject.from_mapping(
                        {"private_detail": "DO_NOT_PROJECT_RECEIPT_PAYLOAD"}
                    ),
                ),
                finished_at=COMPLETED_AT,
            )
            with sqlite3.connect(store.path) as connection:
                connection.execute(
                    "INSERT INTO effect_receipts(agent_id,id,run_id,call_id,operation_key,routine_id,occurrence_id,grant_digest,unresolved,data) VALUES (?,?,?,?,?,NULL,NULL,NULL,1,?)",
                    (
                        agent.id,
                        uncertain.receipt_id,
                        uncertain.run_id,
                        uncertain.call_id,
                        uncertain.operation_key,
                        encode_receipt(uncertain),
                    ),
                )
            receipts.append(uncertain)
    finally:
        await agent.close()
    agent = await Agent.open(
        "effect-context",
        root=tmp_path,
        hosted=True,
        model=provider,
        model_profile=profile,
        approval_handler=approve,
    )
    try:
        await agent.run("Are the requested values present? Do not repeat any action.")
        system = "\n".join(
            block.text
            for block in provider.requests[-1].messages[0].content
            if isinstance(block, TextBlock)
        )
        assert '"unresolved_effects"' in system
        assert '"omitted_count":1' in system
        assert sum(receipt.receipt_id in system for receipt in receipts) == 20
        assert "DO_NOT_PROJECT_RECEIPT_PAYLOAD" not in system
        assert "Current values do not resolve an earlier operation" in system
        assert provider.requests[-1].sensitivity is ModelSensitivity.INTERNAL
        assert len(await agent.list_effects(unresolved_only=True)) == 20
        for receipt in receipts:
            await agent.resolve_effect(
                receipt.receipt_id,
                expected_digest=receipt.receipt_digest,
                decision=EffectResolutionDecision.CLOSE_WITHOUT_RETRY,
                note="Close without action.",
            )
        assert len(provider.requests) == 1
        await agent.run("Report current status.")
        system = "\n".join(
            block.text
            for block in provider.requests[-1].messages[0].content
            if isinstance(block, TextBlock)
        )
        assert '"unresolved_effects"' not in system
        assert not await agent.list_effects(unresolved_only=True)
    finally:
        await agent.close()
