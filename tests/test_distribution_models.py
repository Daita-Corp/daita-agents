from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from daita.artifacts.models import ArtifactAuthorship
from daita.distribution import (
    ArtifactRequirement,
    ConversationInboxTarget,
    Delivery,
    DeliveryState,
    DeliverySubjectKind,
    DistributionPlan,
    OutcomeArtifactReference,
    OutcomeConclusionKind,
    OutcomeContract,
    OutcomeReference,
    OutcomeState,
    conversation_inbox_destination_id,
    distribution_plan_digest,
    logical_delivery_key,
    target_fingerprint,
)
from daita.distribution.models import (
    MAX_DISTRIBUTION_TARGETS,
    MAX_OUTCOME_ARTIFACT_BYTES,
    MAX_OUTCOME_ARTIFACT_REFERENCES,
    MAX_OUTCOME_ARTIFACT_REQUIREMENTS,
    MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES,
    MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
)
from daita.llm import ModelSensitivity
from daita.storage.sqlite_codecs import (
    decode_delivery,
    decode_distribution_plan,
    decode_outcome_contract,
    encode_delivery,
    encode_distribution_plan,
    encode_outcome_contract,
)

NOW = datetime(2026, 8, 28, tzinfo=UTC)
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def requirement(
    *,
    capability_id: str = "artifact.snapshot_result",
    media_type: str = "application/json",
    maximum_bytes: int = MAX_OUTCOME_ARTIFACT_BYTES,
) -> ArtifactRequirement:
    return ArtifactRequirement(
        required=True,
        minimum_count=1,
        maximum_count=1,
        allowed_media_types=(media_type,),
        allowed_authorships=(ArtifactAuthorship.EXACT_SOURCE_DATA,),
        allowed_producer_capability_ids=(capability_id,),
        maximum_artifact_bytes=maximum_bytes,
        maximum_total_bytes=maximum_bytes,
        maximum_sensitivity=ModelSensitivity.CONFIDENTIAL,
    )


def contract(
    requirements: tuple[ArtifactRequirement, ...] = (),
) -> OutcomeContract:
    return OutcomeContract(
        require_terminal_conclusion=True,
        artifact_requirements=requirements,
        maximum_total_artifact_bytes=(
            MAX_OUTCOME_TOTAL_ARTIFACT_BYTES if requirements else 0
        ),
        maximum_effective_sensitivity=ModelSensitivity.CONFIDENTIAL,
        require_current_run_provenance=True,
        require_exact_source_bindings=bool(requirements),
    )


def target(
    *,
    conversation_id: str = "conversation-1",
    ceiling: ModelSensitivity = ModelSensitivity.CONFIDENTIAL,
) -> ConversationInboxTarget:
    destination_id = conversation_inbox_destination_id(conversation_id)
    fingerprint = target_fingerprint(
        conversation_id=conversation_id,
        destination_id=destination_id,
        destination_revision=1,
        sensitivity_ceiling=ceiling,
    )
    return ConversationInboxTarget(
        conversation_id=conversation_id,
        destination_id=destination_id,
        destination_revision=1,
        sensitivity_ceiling=ceiling,
        target_fingerprint=fingerprint,
    )


def plan(binding: ConversationInboxTarget | None = None) -> DistributionPlan:
    bindings = (binding or target(),)
    return DistributionPlan(
        targets=bindings,
        required_target_count=1,
        plan_digest=distribution_plan_digest(
            targets=bindings,
            required_target_count=1,
        ),
    )


def artifact(
    *,
    index: int = 0,
    byte_size: int = 128,
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
) -> OutcomeArtifactReference:
    return OutcomeArtifactReference(
        artifact_id=f"artifact-{index:032x}",
        producing_run_id="run-1",
        producing_call_id=f"call-{index}",
        producer_capability_id="artifact.snapshot_result",
        sha256=DIGEST_A,
        media_type="application/json",
        byte_size=byte_size,
        sensitivity=sensitivity,
        provenance_digest=DIGEST_B,
        authorship=ArtifactAuthorship.EXACT_SOURCE_DATA,
    )


def outcome(
    *,
    artifacts: tuple[OutcomeArtifactReference, ...] = (),
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
) -> OutcomeReference:
    return OutcomeReference(
        conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
        conclusion_state=OutcomeState.SUCCEEDED,
        conclusion_id="run-1",
        conclusion_digest=DIGEST_A,
        conclusion_preview="Completed with exact evidence.",
        conclusion_preview_truncated=False,
        resulting_run_id="run-1",
        artifact_references=artifacts,
        effective_sensitivity=sensitivity,
        provenance_digest=DIGEST_B,
        failure_code=None,
        observed_at=NOW,
    )


def delivery(
    *,
    binding: ConversationInboxTarget | None = None,
    result: OutcomeReference | None = None,
    state: DeliveryState = DeliveryState.AVAILABLE,
    acknowledged_at: datetime | None = None,
    blocked_reason_code: str | None = None,
) -> Delivery:
    exact_target = binding or target()
    key = logical_delivery_key(
        agent_id="agent-1",
        subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
        subject_id="occurrence-1",
        target_fingerprint=exact_target.target_fingerprint,
    )
    return Delivery(
        delivery_id="delivery-1",
        agent_id="agent-1",
        conversation_id=exact_target.conversation_id,
        subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
        subject_id="occurrence-1",
        logical_key=key,
        target=exact_target,
        outcome=result or outcome(),
        visibility_state=state,
        acknowledged_at=acknowledged_at,
        blocked_reason_code=blocked_reason_code,
        created_at=NOW,
        updated_at=NOW,
    )


def test_contracts_normalize_unordered_sets_and_digest_canonically() -> None:
    first = ArtifactRequirement(
        required=True,
        minimum_count=1,
        maximum_count=2,
        allowed_media_types=("text/plain", "application/json"),
        allowed_authorships=(
            ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
            ArtifactAuthorship.EXACT_SOURCE_DATA,
        ),
        allowed_producer_capability_ids=("producer.z", "producer.a"),
        maximum_artifact_bytes=100,
        maximum_total_bytes=200,
        maximum_sensitivity=ModelSensitivity.INTERNAL,
    )
    reordered = ArtifactRequirement(
        required=True,
        minimum_count=1,
        maximum_count=2,
        allowed_media_types=("application/json", "text/plain"),
        allowed_authorships=(
            ArtifactAuthorship.EXACT_SOURCE_DATA,
            ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
        ),
        allowed_producer_capability_ids=("producer.a", "producer.z"),
        maximum_artifact_bytes=100,
        maximum_total_bytes=200,
        maximum_sensitivity=ModelSensitivity.INTERNAL,
    )
    assert first == reordered
    assert first.digest == reordered.digest
    assert contract((first,)).digest == contract((reordered,)).digest


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"required": False}, "required and count disagree"),
        ({"minimum_count": 2, "maximum_count": 1}, "count range"),
        ({"maximum_count": 9}, "maximum_count exceeds"),
        ({"maximum_artifact_bytes": 0}, "inclusive bound"),
        (
            {"maximum_total_bytes": MAX_OUTCOME_TOTAL_ARTIFACT_BYTES + 1},
            "inclusive bound",
        ),
        ({"allowed_media_types": ("Application/JSON",)}, "media type"),
    ],
)
def test_artifact_requirement_rejects_malformed_bounds(
    changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(requirement(), **changes)


def test_outcome_contract_accepts_all_inclusive_requirement_bounds() -> None:
    requirements = tuple(
        requirement(capability_id=f"producer-{index}")
        for index in range(MAX_OUTCOME_ARTIFACT_REQUIREMENTS)
    )
    value = contract(requirements)
    assert len(value.artifact_requirements) == MAX_OUTCOME_ARTIFACT_REQUIREMENTS
    assert value.maximum_total_artifact_bytes == MAX_OUTCOME_TOTAL_ARTIFACT_BYTES


def test_outcome_contract_requires_terminal_current_run_provenance() -> None:
    with pytest.raises(ValueError, match="terminal conclusion"):
        replace(contract(), require_terminal_conclusion=False)
    with pytest.raises(ValueError, match="current-run provenance"):
        replace(contract(), require_current_run_provenance=False)


def test_target_fingerprint_is_identity_and_sensitivity_bound() -> None:
    original = target()
    changed = target(ceiling=ModelSensitivity.RESTRICTED)
    assert original.target_fingerprint != changed.target_fingerprint
    with pytest.raises(ValueError, match="fingerprint"):
        replace(original, target_fingerprint=DIGEST_A)
    with pytest.raises(ValueError, match="destination identity"):
        replace(original, destination_id="conversation_inbox:another")


def test_distribution_plan_is_exactly_one_closed_d2_target() -> None:
    assert plan().required_target_count == 1
    duplicated = (target(), target())
    with pytest.raises(ValueError, match="exactly one"):
        DistributionPlan(
            targets=duplicated,
            required_target_count=2,
            plan_digest=distribution_plan_digest(
                targets=duplicated,
                required_target_count=2,
            ),
        )
    assert MAX_DISTRIBUTION_TARGETS == 4


def test_logical_delivery_key_is_stable_and_exactly_identity_bound() -> None:
    exact_target = target()
    first = logical_delivery_key(
        agent_id="agent-1",
        subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
        subject_id="occurrence-1",
        target_fingerprint=exact_target.target_fingerprint,
    )
    again = logical_delivery_key(
        agent_id="agent-1",
        subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
        subject_id="occurrence-1",
        target_fingerprint=exact_target.target_fingerprint,
    )
    changed = logical_delivery_key(
        agent_id="agent-1",
        subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
        subject_id="occurrence-2",
        target_fingerprint=exact_target.target_fingerprint,
    )
    assert first == again
    assert first != changed


def test_outcome_reference_orders_artifacts_and_accepts_inclusive_bounds() -> None:
    references = tuple(
        artifact(
            index=index,
            byte_size=MAX_OUTCOME_TOTAL_ARTIFACT_BYTES
            // MAX_OUTCOME_ARTIFACT_REFERENCES,
        )
        for index in reversed(range(MAX_OUTCOME_ARTIFACT_REFERENCES))
    )
    result = outcome(artifacts=references)
    assert tuple(item.artifact_id for item in result.artifact_references) == tuple(
        sorted(item.artifact_id for item in references)
    )
    assert sum(item.byte_size for item in result.artifact_references) == (
        MAX_OUTCOME_TOTAL_ARTIFACT_BYTES
    )


def test_outcome_reference_rejects_sensitivity_downgrade_and_cross_identity() -> None:
    restricted = artifact(sensitivity=ModelSensitivity.RESTRICTED)
    with pytest.raises(ValueError, match="cannot downgrade"):
        outcome(artifacts=(restricted,), sensitivity=ModelSensitivity.CONFIDENTIAL)
    with pytest.raises(ValueError, match="exact run identity"):
        replace(outcome(), conclusion_id="run-other")
    with pytest.raises(ValueError, match="cannot claim a resulting run"):
        replace(
            outcome(),
            conclusion_kind=OutcomeConclusionKind.NO_MODEL_OCCURRENCE,
            conclusion_id="occurrence-1",
        )


def test_outcome_preview_and_artifact_bytes_are_hard_bounded() -> None:
    with pytest.raises(ValueError, match="preview exceeds"):
        replace(
            outcome(),
            conclusion_preview="x" * (MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES + 1),
        )
    with pytest.raises(ValueError, match="byte_size exceeds"):
        artifact(byte_size=MAX_OUTCOME_ARTIFACT_BYTES + 1)


def test_delivery_rejects_identity_and_acknowledgment_mismatch() -> None:
    with pytest.raises(ValueError, match="logical key"):
        replace(delivery(), logical_key="delivery:" + DIGEST_A)
    with pytest.raises(ValueError, match="target conversation"):
        replace(delivery(), conversation_id="conversation-other")
    with pytest.raises(ValueError, match="acknowledgment state"):
        replace(delivery(), acknowledged_at=NOW)
    acknowledged = delivery(
        state=DeliveryState.ACKNOWLEDGED,
        acknowledged_at=NOW,
    )
    assert acknowledged.acknowledged_at == NOW


def test_delivery_blocks_instead_of_downgrading_sensitivity() -> None:
    public_target = target(ceiling=ModelSensitivity.PUBLIC)
    internal_outcome = outcome(sensitivity=ModelSensitivity.INTERNAL)
    with pytest.raises(ValueError, match="exceeds its destination"):
        delivery(binding=public_target, result=internal_outcome)
    blocked = delivery(
        binding=public_target,
        result=internal_outcome,
        state=DeliveryState.BLOCKED,
        blocked_reason_code="sensitivity_exceeds_destination",
    )
    assert blocked.visibility_state is DeliveryState.BLOCKED


def test_no_model_conclusion_is_only_valid_without_a_run() -> None:
    no_model = replace(
        outcome(),
        conclusion_kind=OutcomeConclusionKind.NO_MODEL_OCCURRENCE,
        conclusion_id="occurrence-1",
        resulting_run_id=None,
        artifact_references=(),
    )
    created = delivery(result=no_model)
    assert created.outcome.resulting_run_id is None


def test_malformed_timestamps_fail_closed() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        replace(outcome(), observed_at=datetime(2026, 8, 28))


def test_legacy_version_suffixed_routine_symbols_are_absent() -> None:
    import daita.routines as routines

    assert not hasattr(routines, "ScheduledRoutineV1")
    assert not hasattr(routines, "RoutineOccurrenceV1")
    assert hasattr(routines, "ScheduledRoutine")
    assert hasattr(routines, "RoutineOccurrence")


def test_delivery_and_embedded_contract_codecs_round_trip_one_current_shape() -> None:
    exact_contract = contract((requirement(),))
    exact_plan = plan()
    exact_delivery = delivery()

    assert decode_outcome_contract(encode_outcome_contract(exact_contract)) == (
        exact_contract
    )
    assert decode_distribution_plan(encode_distribution_plan(exact_plan)) == exact_plan
    encoded_delivery = encode_delivery(exact_delivery)
    assert (
        decode_delivery(
            encoded_delivery,
            agent_id=exact_delivery.agent_id,
            delivery_id=exact_delivery.delivery_id,
        )
        == exact_delivery
    )
    with pytest.raises(ValueError, match="version is unsupported"):
        decode_delivery(
            encoded_delivery.replace('"version":1', '"version":2'),
            agent_id=exact_delivery.agent_id,
            delivery_id=exact_delivery.delivery_id,
        )
