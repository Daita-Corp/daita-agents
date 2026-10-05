"""Opening another handle is independent of recovering an abandoned writer."""

from datetime import timedelta

import pytest

from daita.capabilities import EffectOutcome
from daita.identity import AgentIdentity
from daita.llm.models import ModelSensitivity
from daita.loop.models import RunInput
from daita.storage.protocols import StateStore
from daita.storage.sqlite_records import EffectReceipt, effect_receipt_id
from tests.storage._support import StateStoreFactory
from tests.support.graph import GRAPH_NOW

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
