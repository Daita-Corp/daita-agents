"""One SQLite-backed lifecycle for unrelated adapter and server effects."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from _capability_runtime_support import (
    StaticTestDomain,
    execute_projected,
    presentation_metadata,
    static_registry,
)
from test_effect_contracts import _capability
from test_effect_receipts import STARTED_AT, _store

from daita._json import FrozenJsonObject
from daita.capabilities import (
    ApprovalDecision,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolView,
)
from daita.capability_runtime import CapabilityRuntime, SideEffectPlan
from daita.llm.models import ToolCall
from daita.loop.models import RunInput
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_records import EffectReceipt


class _EffectDomain(StaticTestDomain):
    async def side_effect_plan(self, run, call, capability, execution, fingerprint):
        return SideEffectPlan(
            effect_intent=FrozenJsonObject.from_mapping(
                {"arguments": execution.arguments}
            )
        )


class _EffectExecutor:
    executor_id = "test.action.executor"

    def __init__(
        self, *, store, basis=EffectEvidenceBasis.SERVER_REPORTED, mode="success"
    ):
        self.store = store
        self.basis = basis
        self.mode = mode
        self.calls = 0
        self.preflights = 0

    async def preflight(self, request):
        self.preflights += 1
        return FrozenJsonObject.from_mapping({"current": True})

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.calls += 1
        assert request.effect_receipt_id is not None
        started = await self.store.load_effect_receipt(
            "agent-effect", request.effect_receipt_id
        )
        assert started is not None and started.outcome is EffectOutcome.STARTED
        if self.mode == "disconnect":
            raise ConnectionError("acted then disconnected")
        payload = FrozenJsonObject.from_mapping({"target": request.arguments["target"]})
        observation = EffectObservation(EffectOutcome.SUCCEEDED, self.basis, payload)
        return ToolOutput(
            kind="bad-kind" if self.mode == "bad-output" else "test.action.result",
            data=payload,
            effect_observation=observation,
        )


async def _runtime_fixture(
    tmp_path: Path,
    *,
    mode="success",
    basis=EffectEvidenceBasis.SERVER_REPORTED,
    approve=True,
):
    store = await _store(tmp_path / "state.db")
    capability = _capability()
    assert capability.effect_receipt_policy is not None
    capability = replace(
        capability,
        effect_receipt_policy=replace(
            capability.effect_receipt_policy, success_evidence_basis=basis
        ),
    )
    view = ToolView(
        name="test_action",
        capability_id=capability.id,
        description=capability.description,
        presentation=presentation_metadata(load_mode=ToolLoadMode.ON_DEMAND),
    )
    domain = _EffectDomain((capability,), (view,))
    executor = _EffectExecutor(store=store, basis=basis, mode=mode)
    approvals = []

    async def approval(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE if approve else ApprovalDecision.DENY

    runtime = CapabilityRuntime(
        static_registry(domain, (executor,)),
        (domain,),
        effect_receipts=store,
        approval_handler=approval,
        clock=lambda: STARTED_AT,
    )
    run = RunInput(
        id="run-effect",
        agent_id="agent-effect",
        message="exact effect",
        created_at=STARTED_AT,
        conversation_id="conversation-effect",
    )
    return store, runtime, executor, run, approvals


def _call(call_id="first", target="destination") -> ToolCall:
    return ToolCall(id=call_id, name="test_action", arguments={"target": target})


async def test_runtime_reserves_before_dispatch_and_deduplicates_new_model_call_ids(
    tmp_path,
):
    store, runtime, executor, run, approvals = await _runtime_fixture(tmp_path)
    try:
        first = (await execute_projected(runtime, run, (_call(),))).ordered_results[0]
        assert not first.is_error
        reference = first.output["effect_receipt"]
        assert reference["evidence_basis"] == "server_reported"
        duplicate = (
            await execute_projected(runtime, run, (_call("second"),))
        ).ordered_results[0]
        assert duplicate.is_error
        assert duplicate.output["effect_receipt"] == reference
        assert executor.calls == 1
        assert len(await store.list_effect_receipts(run.agent_id)) == 1
        same_call = (await execute_projected(runtime, run, (_call(),))).ordered_results[
            0
        ]
        assert same_call.output["effect_receipt"] == reference and executor.calls == 1
        assert len(approvals) == 2
    finally:
        await store.close()


@pytest.mark.parametrize(
    "basis,expected",
    (
        (EffectEvidenceBasis.SERVER_REPORTED, EffectOutcome.UNCERTAIN),
        (EffectEvidenceBasis.ADAPTER_VERIFIED, EffectOutcome.SUCCEEDED),
    ),
)
async def test_invalid_tool_output_preserves_only_independent_adapter_commit_evidence(
    tmp_path, basis, expected
):
    store, runtime, executor, run, _ = await _runtime_fixture(
        tmp_path, mode="bad-output", basis=basis
    )
    try:
        result = (await execute_projected(runtime, run, (_call(),))).ordered_results[0]
        assert result.is_error and executor.calls == 1
        receipt = await store.load_effect_receipt_for_call(
            run.agent_id, run.id, "first"
        )
        assert receipt is not None and receipt.outcome is expected
        assert (
            result.output["effect_receipt"]["receipt_digest"] == receipt.receipt_digest
        )
    finally:
        await store.close()


async def test_disconnect_blocks_changed_arguments_and_later_calls_with_ordered_recovery_references(
    tmp_path,
):
    store, runtime, executor, run, approvals = await _runtime_fixture(
        tmp_path, mode="disconnect"
    )
    try:
        results = (
            await execute_projected(runtime, run, (_call(), _call("second", "other")))
        ).ordered_results
        assert [result.call_id for result in results] == ["first", "second"]
        assert all(result.is_error for result in results)
        assert executor.calls == 1 and len(approvals) == 1
        receipt = await store.load_effect_receipt_for_call(
            run.agent_id, run.id, "first"
        )
        assert receipt is not None and receipt.unresolved
        assert results[1].output["error"]["details"]["receipt_ids"] == (
            receipt.receipt_id,
        )
    finally:
        await store.close()


async def test_denied_approval_and_failed_reservation_never_dispatch(
    tmp_path, monkeypatch
):
    store, runtime, executor, run, _ = await _runtime_fixture(tmp_path, approve=False)
    try:
        result = (await execute_projected(runtime, run, (_call(),))).ordered_results[0]
        assert result.is_error and executor.calls == 0
        assert not await store.list_effect_receipts(run.agent_id)

        async def approve(request):
            return ApprovalDecision.APPROVE

        runtime._approval_handler = approve

        async def failed_reservation(*args, **kwargs):
            raise OSError("unavailable")

        monkeypatch.setattr(store, "start_effect_receipt", failed_reservation)
        result = (
            await execute_projected(runtime, run, (_call("second"),))
        ).ordered_results[0]
        assert result.is_error and executor.calls == 0
        assert not await store.list_effect_receipts(run.agent_id)
    finally:
        await store.close()


async def test_terminal_persistence_failure_keeps_started_receipt_and_blocks_dispatch(
    tmp_path, monkeypatch
):
    store, runtime, executor, run, _ = await _runtime_fixture(tmp_path)
    try:

        async def failed_finish(receipt: EffectReceipt):
            raise OSError("disk unavailable after the action")

        monkeypatch.setattr(store, "finish_effect_receipt", failed_finish)
        first = (await execute_projected(runtime, run, (_call(),))).ordered_results[0]
        assert first.is_error and first.output["effect_receipt"]["outcome"] == "started"
        second = (
            await execute_projected(runtime, run, (_call("second", "changed"),))
        ).ordered_results[0]
        assert second.is_error and executor.calls == 1
    finally:
        await store.close()
    reopened = await SQLiteStateStore.open(
        tmp_path / "state.db", clock=lambda: STARTED_AT
    )
    try:
        receipt = await reopened.load_effect_receipt_for_call(
            run.agent_id, run.id, "first"
        )
        assert receipt is not None and receipt.outcome is EffectOutcome.UNCERTAIN
    finally:
        await reopened.close()


async def test_cancellation_during_committed_reservation_does_not_dispatch(
    tmp_path, monkeypatch
):
    import asyncio

    store, runtime, executor, run, _ = await _runtime_fixture(tmp_path)
    original = store.start_effect_receipt

    async def reserve_then_cancel(*args, **kwargs):
        receipt = await original(*args, **kwargs)
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        return receipt

    monkeypatch.setattr(store, "start_effect_receipt", reserve_then_cancel)
    try:
        batch = await execute_projected(runtime, run, (_call(),))
        assert batch.ordered_results[0].is_error and executor.calls == 0
        receipt = await store.load_effect_receipt_for_call(
            run.agent_id, run.id, "first"
        )
        assert receipt is not None and receipt.outcome is EffectOutcome.NOT_APPLIED
        assert receipt.evidence_basis is EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
        assert not receipt.unresolved
    finally:
        await store.close()
