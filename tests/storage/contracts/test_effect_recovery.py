"""Opening another handle is independent of recovering an abandoned writer."""

from datetime import timedelta
from decimal import Decimal
from hashlib import sha256

import pytest

from daita._json import canonical_json
from daita.capabilities import (
    AccessMode,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ExecutionContractBindings,
    ExecutionScope,
    ExecutionScopeKind,
    OperationalEffect,
)
from daita.identity import AgentIdentity
from daita.llm.models import ModelSensitivity
from daita.loop.models import (
    InstructionAuthority,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)
from daita.storage.protocols import StateStore
from daita.storage.sqlite_records import EffectReceipt, effect_receipt_id
from tests.storage._support import StateStoreFactory
from tests.support.distribution import inbox_distribution_plan
from tests.support.graph import GRAPH_NOW
from tests.support.jobs import stored_job

pytestmark = [pytest.mark.integration, pytest.mark.contract]


async def test_connect_is_passive_and_explicit_recovery_is_scoped_and_idempotent(
    state_store: StateStore, state_store_factory: StateStoreFactory
) -> None:
    store = state_store
    await store.initialize_identity(AgentIdentity("agent-1", "Effects", GRAPH_NOW))
    await store.start(
        RunInput(
            id="effect-run",
            agent_id="agent-1",
            message="test",
            conversation_id="conversation-1",
            created_at=GRAPH_NOW,
        )
    )
    digest = "sha256:" + "3" * 64
    receipt = EffectReceipt(
        receipt_id=effect_receipt_id(
            agent_id="agent-1",
            run_id="effect-run",
            call_id="call-1",
            operation_key=digest,
        ),
        receipt_kind="test.effect",
        agent_id="agent-1",
        run_id="effect-run",
        call_id="call-1",
        capability_id="test.effect",
        domain_owner_id="test",
        capability_contract_digest=digest,
        operation_key=digest,
        argument_fingerprint=digest,
        sensitivity=ModelSensitivity.INTERNAL,
        started_at=GRAPH_NOW,
    )
    await store.start_effect_receipt(receipt)
    async with state_store_factory() as observer:
        assert (
            await observer.load_effect_receipt("agent-1", receipt.receipt_id) == receipt
        )
        assert await store.load_effect_receipt("agent-1", receipt.receipt_id) == receipt
    await store.recover_started_effect_receipts("another-agent", recovered_at=GRAPH_NOW)
    assert await store.load_effect_receipt("agent-1", receipt.receipt_id) == receipt
    await store.recover_started_effect_receipts(
        "agent-1", recovered_at=GRAPH_NOW + timedelta(seconds=1)
    )
    recovered = await store.load_effect_receipt("agent-1", receipt.receipt_id)
    assert recovered is not None and recovered.outcome is EffectOutcome.UNCERTAIN
    await store.recover_started_effect_receipts(
        "agent-1", recovered_at=GRAPH_NOW + timedelta(seconds=2)
    )
    assert await store.load_effect_receipt("agent-1", receipt.receipt_id) == recovered


def _job_run(run_id: str, job_id: str) -> RunInput:
    instruction = "Work on the job."
    digest = "sha256:" + "4" * 64
    scope = ExecutionScope(
        contract_bindings=ExecutionContractBindings(
            capability_contracts={"test.effect": digest},
            model_routes={"route-1": digest},
        ),
        scope_id=f"scope:{run_id}",
        revision=1,
        agent_id="agent-1",
        principal_id="agent:agent-1",
        grant_id=f"job:{job_id}",
        job_id=job_id,
        job_revision=1,
        allowed_source_ids=(),
        allowed_resource_ids=(),
        allowed_capability_ids=("test.effect",),
        allowed_access_modes=frozenset(AccessMode),
        allowed_operational_effects=frozenset(OperationalEffect),
        sensitivity_ceiling=ModelSensitivity.RESTRICTED,
        eligible_model_routes=("route-1",),
        per_run_max_cost_usd=Decimal("0.01"),
        per_run_max_tokens=1_000,
        distribution_plan_digest=inbox_distribution_plan("conversation-1").plan_digest,
        scope_kind=ExecutionScopeKind.JOB,
    )
    return RunInput(
        id=run_id,
        agent_id="agent-1",
        message=instruction,
        created_at=GRAPH_NOW,
        conversation_id=job_id,
        start=RunStartEnvelope(
            origin=RunOrigin.JOB,
            instruction_authority=InstructionAuthority.CODE_OWNED,
            trusted_instruction_id=f"job:{job_id}:run:1",
            trusted_instruction=instruction,
            instruction_digest="sha256:" + sha256(instruction.encode()).hexdigest(),
            untrusted_payload={},
            payload_digest="sha256:" + sha256(canonical_json({}).encode()).hexdigest(),
            execution_scope=scope,
        ),
    )


def _receipt(run_id: str, call_id: str) -> EffectReceipt:
    digest = "sha256:" + sha256(f"{run_id}:{call_id}".encode()).hexdigest()
    return EffectReceipt(
        receipt_id=effect_receipt_id(
            agent_id="agent-1", run_id=run_id, call_id=call_id, operation_key=digest
        ),
        receipt_kind="test.effect",
        agent_id="agent-1",
        run_id=run_id,
        call_id=call_id,
        capability_id="test.effect",
        domain_owner_id="test",
        capability_contract_digest=digest,
        operation_key=digest,
        argument_fingerprint=digest,
        sensitivity=ModelSensitivity.INTERNAL,
        started_at=GRAPH_NOW,
    )


def _finished(receipt: EffectReceipt) -> EffectReceipt:
    return receipt.finish(
        EffectObservation(
            EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
        ),
        finished_at=GRAPH_NOW + timedelta(seconds=1),
    )


async def test_job_receipts_read_back_in_reservation_order(
    state_store: StateStore, state_store_factory: StateStoreFactory
) -> None:
    store = state_store
    await store.initialize_identity(AgentIdentity("agent-1", "Effects", GRAPH_NOW))
    await store.admit_job(stored_job("job-1"))
    await store.admit_job(stored_job("job-2"))
    for run_id, job_id in (
        ("job-run-1", "job-1"),
        ("job-run-2", "job-1"),
        ("other-run", "job-2"),
    ):
        await store.start(_job_run(run_id, job_id))
    # Each receipt's ID sorts below the previous one, so neither ID nor call
    # order gives the reservation order.
    reserved: list[EffectReceipt] = []
    for run_id in ("job-run-1", "job-run-2", "job-run-1"):
        reserved.append(
            next(
                receipt
                for index in range(64)
                if (receipt := _receipt(run_id, f"call-{index}")) not in reserved
                and (not reserved or receipt.receipt_id < reserved[-1].receipt_id)
            )
        )
    other_job = _receipt("other-run", "call-0")
    expected: list[EffectReceipt] = []
    for receipt in reserved:
        await store.start_effect_receipt(receipt)
        # Finishing rewrites the row; its place in the order stays.
        expected.append(await store.finish_effect_receipt(_finished(receipt)))
        if receipt is reserved[0]:
            await store.start_effect_receipt(other_job)
            await store.finish_effect_receipt(_finished(other_job))

    assert await store.job_effect_receipts("agent-1", "job-1") == tuple(expected)
    assert await store.job_effect_receipts("foreign", "job-1") == ()
    async with state_store_factory() as reopened:
        assert await reopened.job_effect_receipts("agent-1", "job-1") == tuple(expected)
