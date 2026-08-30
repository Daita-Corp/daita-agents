"""Own exact inbox destinations and bounded logical-delivery projections."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..artifacts.models import ArtifactPayload, ArtifactRef
from ..llm.models import ModelSensitivity
from .models import (
    CONVERSATION_INBOX_DESTINATION_REVISION,
    MAX_DELIVERY_LIST_PAGE_SIZE,
    ConversationInboxTarget,
    Delivery,
    DeliveryInspection,
    DeliveryState,
    DeliverySubjectKind,
    DistributionDestination,
    DistributionDestinationState,
    DistributionPlan,
    InboxView,
    OutcomeArtifactReference,
    OutcomeConclusionKind,
    OutcomeContract,
    OutcomeReference,
    OutcomeState,
    conversation_inbox_destination_id,
    distribution_plan_digest,
    logical_delivery_key,
    outcome_artifact_reference,
    target_fingerprint,
    validate_outcome_artifact_references,
)


def construct_logical_delivery(
    *,
    delivery_id: str,
    agent_id: str,
    conversation_id: str,
    subject_kind: DeliverySubjectKind,
    subject_id: str,
    target: ConversationInboxTarget,
    conclusion_kind: OutcomeConclusionKind,
    conclusion_state: OutcomeState,
    conclusion_id: str,
    conclusion_digest: str,
    conclusion_preview: str,
    conclusion_preview_truncated: bool,
    resulting_run_id: str | None,
    artifact_references: tuple[OutcomeArtifactReference, ...],
    effective_sensitivity: ModelSensitivity,
    provenance_digest: str,
    failure_code: str | None,
    observed_at: datetime,
) -> Delivery:
    """Construct one immutable logical delivery for any admitted producer."""

    if target.conversation_id != conversation_id:
        raise ValueError("delivery target conversation identity differs")
    visibility_state = (
        DeliveryState.AVAILABLE
        if effective_sensitivity.routing_rank <= target.sensitivity_ceiling.routing_rank
        else DeliveryState.BLOCKED
    )
    visible_preview = (
        conclusion_preview
        if conclusion_state is not OutcomeState.FAILED
        and visibility_state is DeliveryState.AVAILABLE
        else ""
    )
    outcome = OutcomeReference(
        conclusion_kind=conclusion_kind,
        conclusion_state=conclusion_state,
        conclusion_id=conclusion_id,
        conclusion_digest=conclusion_digest,
        conclusion_preview=visible_preview,
        conclusion_preview_truncated=conclusion_preview_truncated,
        resulting_run_id=resulting_run_id,
        artifact_references=artifact_references,
        effective_sensitivity=effective_sensitivity,
        provenance_digest=provenance_digest,
        failure_code=failure_code,
        observed_at=observed_at,
    )
    return Delivery(
        delivery_id=delivery_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        subject_kind=subject_kind,
        subject_id=subject_id,
        logical_key=logical_delivery_key(
            agent_id=agent_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            target_fingerprint=target.target_fingerprint,
        ),
        target=target,
        outcome=outcome,
        visibility_state=visibility_state,
        acknowledged_at=None,
        blocked_reason_code=(
            "sensitivity_exceeds_destination"
            if visibility_state is DeliveryState.BLOCKED
            else None
        ),
        created_at=observed_at,
        updated_at=observed_at,
    )


class DistributionStore(Protocol):
    async def conversation_exists(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> bool: ...

    async def load_delivery(
        self,
        agent_id: str,
        delivery_id: str,
    ) -> Delivery | None: ...

    async def list_deliveries(
        self,
        agent_id: str,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = MAX_DELIVERY_LIST_PAGE_SIZE,
    ) -> tuple[Delivery, ...]: ...

    async def acknowledge_delivery(
        self,
        agent_id: str,
        delivery_id: str,
        *,
        acknowledged_at: datetime,
    ) -> Delivery | None: ...


class OutcomeArtifactReader(Protocol):
    async def read_ref(self, ref: ArtifactRef) -> ArtifactPayload: ...


class DistributionOwner:
    """The sole D2 destination and logical-delivery lifecycle owner."""

    def __init__(self, *, agent_id: str, store: DistributionStore) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("distribution owner agent_id must be non-empty")
        self._agent_id = agent_id
        self._store = store

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def resolve_conversation_inbox(
        self,
        conversation_id: str,
        *,
        sensitivity_ceiling: ModelSensitivity,
    ) -> ConversationInboxTarget:
        destination_id = conversation_inbox_destination_id(conversation_id)
        fingerprint = target_fingerprint(
            conversation_id=conversation_id,
            destination_id=destination_id,
            destination_revision=CONVERSATION_INBOX_DESTINATION_REVISION,
            sensitivity_ceiling=sensitivity_ceiling,
        )
        return ConversationInboxTarget(
            conversation_id=conversation_id,
            destination_id=destination_id,
            destination_revision=CONVERSATION_INBOX_DESTINATION_REVISION,
            sensitivity_ceiling=sensitivity_ceiling,
            target_fingerprint=fingerprint,
        )

    def list_destinations(
        self,
        conversation_id: str,
        *,
        sensitivity_ceiling: ModelSensitivity,
    ) -> tuple[DistributionDestination, ...]:
        target = self.resolve_conversation_inbox(
            conversation_id,
            sensitivity_ceiling=sensitivity_ceiling,
        )
        return (
            DistributionDestination(
                destination_id=target.destination_id,
                kind="conversation_inbox",
                label="Current conversation Inbox",
                state=DistributionDestinationState.AVAILABLE,
                revision=target.destination_revision,
                sensitivity_ceiling=target.sensitivity_ceiling,
                supported_outcome_media=(
                    "application/json",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "text/csv",
                    "text/markdown",
                    "text/plain",
                ),
                selectable=True,
            ),
        )

    async def discover_destinations(
        self,
        conversation_id: str,
        *,
        sensitivity_ceiling: ModelSensitivity,
    ) -> tuple[DistributionDestination, ...]:
        """Return destinations only for one current agent-owned conversation."""

        if not await self._store.conversation_exists(
            self._agent_id,
            conversation_id,
        ):
            return ()
        return self.list_destinations(
            conversation_id,
            sensitivity_ceiling=sensitivity_ceiling,
        )

    def resolve_plan(
        self,
        conversation_id: str,
        *,
        destination_id: str,
        sensitivity_ceiling: ModelSensitivity,
    ) -> DistributionPlan:
        """Resolve the exact currently selectable D2 distribution plan."""

        target = self.resolve_conversation_inbox(
            conversation_id,
            sensitivity_ceiling=sensitivity_ceiling,
        )
        if destination_id != target.destination_id:
            raise ValueError("distribution destination is not currently selectable")
        targets = (target,)
        return DistributionPlan(
            targets=targets,
            required_target_count=1,
            plan_digest=distribution_plan_digest(
                targets=targets,
                required_target_count=1,
            ),
        )

    async def list(
        self,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = MAX_DELIVERY_LIST_PAGE_SIZE,
    ) -> tuple[InboxView, ...]:
        deliveries = await self._store.list_deliveries(
            self._agent_id,
            conversation_id=conversation_id,
            include_acknowledged=include_acknowledged,
            limit=limit,
        )
        return tuple(_inbox_view(value) for value in deliveries)

    async def validate_outcome_artifacts(
        self,
        artifacts: OutcomeArtifactReader,
        refs: tuple[ArtifactRef, ...],
        *,
        contract: OutcomeContract,
        resulting_run_id: str,
    ) -> tuple[OutcomeArtifactReference, ...]:
        """Resolve bytes once and validate the exact frozen artifact contract."""

        projected: list[OutcomeArtifactReference] = []
        for ref in sorted(refs, key=lambda item: item.artifact_id):
            payload = await artifacts.read_ref(ref)
            if payload.ref != ref:
                raise ValueError("outcome artifact identity changed during validation")
            projected.append(outcome_artifact_reference(ref))
        return validate_outcome_artifact_references(
            tuple(projected),
            contract=contract,
            resulting_run_id=resulting_run_id,
        )

    async def inspect(self, delivery_id: str) -> DeliveryInspection | None:
        delivery = await self._store.load_delivery(self._agent_id, delivery_id)
        return None if delivery is None else DeliveryInspection(delivery=delivery)

    async def acknowledge(
        self,
        delivery_id: str,
        *,
        acknowledged_at: datetime,
    ) -> InboxView | None:
        delivery = await self._store.acknowledge_delivery(
            self._agent_id,
            delivery_id,
            acknowledged_at=acknowledged_at,
        )
        return None if delivery is None else _inbox_view(delivery)


def _inbox_view(value: Delivery) -> InboxView:
    outcome = value.outcome
    blocked = value.blocked_reason_code is not None
    return InboxView(
        delivery_id=value.delivery_id,
        conversation_id=value.conversation_id,
        subject_kind=value.subject_kind,
        subject_id=value.subject_id,
        conclusion_kind=outcome.conclusion_kind,
        conclusion_state=outcome.conclusion_state,
        conclusion_digest=outcome.conclusion_digest,
        conclusion_preview="" if blocked else outcome.conclusion_preview,
        conclusion_preview_truncated=outcome.conclusion_preview_truncated,
        resulting_run_id=outcome.resulting_run_id,
        artifact_references=outcome.artifact_references,
        effective_sensitivity=outcome.effective_sensitivity,
        provenance_digest=outcome.provenance_digest,
        destination_id=value.target.destination_id,
        state=value.visibility_state,
        created_at=value.created_at,
        updated_at=value.updated_at,
        acknowledged_at=value.acknowledged_at,
        blocked_reason_code=value.blocked_reason_code,
        failure_code=outcome.failure_code,
    )


__all__ = [
    "DistributionOwner",
    "DistributionStore",
    "OutcomeArtifactReader",
    "construct_logical_delivery",
]
