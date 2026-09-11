from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from daita.distribution import (
    Delivery,
    DeliveryState,
    DeliverySubjectKind,
    DistributionOwner,
    OutcomeConclusionKind,
    OutcomeState,
    conversation_inbox_destination_id,
)
from daita.distribution.owner import construct_logical_delivery
from daita.llm import ModelSensitivity

NOW = datetime(2026, 8, 28, 15, tzinfo=UTC)
DIGEST = "sha256:" + "a" * 64


class FakeDistributionStore:
    def __init__(self, values: tuple[Delivery, ...]) -> None:
        self.values = {value.delivery_id: value for value in values}

    async def conversation_exists(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> bool:
        return agent_id == "agent-1" and conversation_id == "conversation-1"

    async def load_delivery(self, agent_id: str, delivery_id: str) -> Delivery | None:
        value = self.values.get(delivery_id)
        return value if value is not None and value.agent_id == agent_id else None

    async def list_deliveries(
        self,
        agent_id: str,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = 50,
    ) -> tuple[Delivery, ...]:
        return tuple(
            value
            for value in self.values.values()
            if value.agent_id == agent_id
            and (conversation_id is None or value.conversation_id == conversation_id)
            and (
                include_acknowledged
                or value.visibility_state is not DeliveryState.ACKNOWLEDGED
            )
        )[:limit]

    async def acknowledge_delivery(
        self,
        agent_id: str,
        delivery_id: str,
        *,
        acknowledged_at: datetime,
    ) -> Delivery | None:
        current = await self.load_delivery(agent_id, delivery_id)
        if current is None:
            return None
        if current.visibility_state is DeliveryState.ACKNOWLEDGED:
            return current
        updated = replace(
            current,
            visibility_state=DeliveryState.ACKNOWLEDGED,
            acknowledged_at=acknowledged_at,
            updated_at=acknowledged_at,
        )
        self.values[delivery_id] = updated
        return updated


def _delivery(owner: DistributionOwner, *, agent_id: str = "agent-1") -> Delivery:
    target = owner.resolve_conversation_inbox(
        "conversation-1",
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
    )
    return construct_logical_delivery(
        delivery_id="delivery-1",
        agent_id=agent_id,
        conversation_id="conversation-1",
        subject_kind=DeliverySubjectKind.AUTONOMOUS_FOLLOWUP,
        subject_id="followup-1",
        target=target,
        conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
        conclusion_state=OutcomeState.SUCCEEDED,
        conclusion_id="run-1",
        conclusion_digest=DIGEST,
        conclusion_preview="Bounded report",
        conclusion_preview_truncated=False,
        resulting_run_id="run-1",
        artifact_references=(),
        effective_sensitivity=ModelSensitivity.INTERNAL,
        provenance_digest=DIGEST,
        failure_code=None,
        observed_at=NOW,
    )


async def test_distribution_owner_projects_only_the_exact_current_inbox() -> None:
    store = FakeDistributionStore(())
    owner = DistributionOwner(agent_id="agent-1", store=store)

    destinations = owner.list_destinations(
        "conversation-1",
        sensitivity_ceiling=ModelSensitivity.CONFIDENTIAL,
    )

    assert len(destinations) == 1
    assert destinations[0].destination_id == conversation_inbox_destination_id(
        "conversation-1"
    )
    assert destinations[0].kind == "conversation_inbox"
    assert destinations[0].selectable is True
    assert destinations[0].sensitivity_ceiling is ModelSensitivity.CONFIDENTIAL


async def test_distribution_owner_lists_inspects_and_acknowledges_safe_views() -> None:
    empty_store = FakeDistributionStore(())
    owner = DistributionOwner(agent_id="agent-1", store=empty_store)
    exact = _delivery(owner)
    empty_store.values[exact.delivery_id] = exact

    listed = await owner.list(conversation_id="conversation-1")
    inspected = await owner.inspect(exact.delivery_id)
    acknowledged = await owner.acknowledge(exact.delivery_id, acknowledged_at=NOW)

    assert len(listed) == 1
    assert listed[0].conclusion_preview == "Bounded report"
    assert inspected is not None and inspected.delivery == exact
    assert acknowledged is not None
    assert acknowledged.state is DeliveryState.ACKNOWLEDGED
    assert await owner.list() == ()
    assert len(await owner.list(include_acknowledged=True)) == 1


async def test_distribution_owner_does_not_disclose_cross_agent_delivery() -> None:
    empty_store = FakeDistributionStore(())
    owner = DistributionOwner(agent_id="agent-1", store=empty_store)
    foreign = _delivery(owner, agent_id="agent-2")
    empty_store.values[foreign.delivery_id] = foreign

    assert await owner.list(include_acknowledged=True) == ()
    assert await owner.inspect(foreign.delivery_id) is None
    assert await owner.acknowledge(foreign.delivery_id, acknowledged_at=NOW) is None
