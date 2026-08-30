from __future__ import annotations

from daita.distribution import (
    ConversationInboxTarget,
    DistributionPlan,
    OutcomeContract,
    conversation_inbox_destination_id,
    distribution_plan_digest,
    target_fingerprint,
)
from daita.llm.models import ModelSensitivity

_CONVERSATION_INBOX_DESTINATION_REVISION = 1


def no_artifact_outcome_contract(
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
) -> OutcomeContract:
    return OutcomeContract(
        require_terminal_conclusion=True,
        artifact_requirements=(),
        maximum_total_artifact_bytes=0,
        maximum_effective_sensitivity=sensitivity,
        require_current_run_provenance=True,
        require_exact_source_bindings=False,
    )


def inbox_distribution_plan(
    conversation_id: str,
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
) -> DistributionPlan:
    destination_id = conversation_inbox_destination_id(conversation_id)
    target = ConversationInboxTarget(
        conversation_id=conversation_id,
        destination_id=destination_id,
        destination_revision=_CONVERSATION_INBOX_DESTINATION_REVISION,
        sensitivity_ceiling=sensitivity,
        target_fingerprint=target_fingerprint(
            conversation_id=conversation_id,
            destination_id=destination_id,
            destination_revision=_CONVERSATION_INBOX_DESTINATION_REVISION,
            sensitivity_ceiling=sensitivity,
        ),
    )
    targets = (target,)
    return DistributionPlan(
        targets=targets,
        required_target_count=1,
        plan_digest=distribution_plan_digest(
            targets=targets,
            required_target_count=1,
        ),
    )


__all__ = ["inbox_distribution_plan", "no_artifact_outcome_contract"]
