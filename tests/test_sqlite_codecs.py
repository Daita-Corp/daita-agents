from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TypeVar

import pytest

from daita.adapters.models import SourceRegistration
from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactDeliveryReceipt,
    ArtifactProvenance,
    ArtifactRef,
)
from daita.catalog.models import (
    CatalogSync,
    CatalogSyncStatus,
    Sensitivity,
    SourceCatalogSnapshot,
)
from daita.identity import AgentIdentity
from daita.learning_candidates import (
    DocumentCandidateContent,
    LearningCandidate,
    LearningCandidateReviewStamp,
    LearningCandidateRunReference,
    LearningCandidateStatus,
    LearningCandidateTarget,
)
from daita.llm.models import CanonicalMessage, MessageRole, ModelUsage, TextBlock
from daita.llm.pricing import CostBasis, CostComponent, CostEstimate
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
)
from daita.storage.sqlite import DatabaseWriteOutcome, DatabaseWriteReceipt
from daita.storage.sqlite_codecs import (
    decode_catalog_snapshot,
    decode_catalog_sync,
    decode_identifier,
    decode_identity,
    decode_learning_candidate,
    decode_loop_exit,
    decode_message,
    decode_postgresql_update_scope,
    decode_receipt,
    decode_review_stamps,
    decode_run_input,
    decode_semantic_annotation,
    decode_source,
    decode_source_read_scope,
    encode_catalog_snapshot,
    encode_catalog_sync,
    encode_identifier,
    encode_identity,
    encode_learning_candidate,
    encode_loop_exit,
    encode_message,
    encode_postgresql_update_scope,
    encode_receipt,
    encode_review_stamps,
    encode_run_input,
    encode_semantic_annotation,
    encode_source,
    encode_source_read_scope,
)
from daita.storage.sqlite_records import (
    PostgreSQLUpdateScope,
    SourceReadMode,
    SourceReadScope,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
_RecordT = TypeVar("_RecordT")


def _assert_round_trip(
    value: _RecordT,
    encode: Callable[[_RecordT], str],
    decode: Callable[[str], _RecordT],
) -> None:
    first = encode(value)
    assert decode(first) == value
    assert encode(decode(first)) == first


def _source() -> SourceRegistration:
    return SourceRegistration.build(
        agent_id="agent-codec",
        adapter_id="postgresql",
        native_identity="postgresql:codec",
        display_name="Codec warehouse",
        configuration={
            "database": "warehouse",
            "host": "db.example.test",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "reader",
            "write_access": True,
        },
        attached_at=NOW,
    )


def _receipt() -> DatabaseWriteReceipt:
    return DatabaseWriteReceipt.start(
        agent_id="agent-codec",
        run_id="run-codec",
        call_id="call-codec",
        capability_id="data.postgresql.update",
        source_id="source:sha256:" + "1" * 64,
        resource_id="catalog-resource:sha256:" + "2" * 64,
        intent_sha256="sha256:" + "3" * 64,
        preview_fingerprint="sha256:" + "4" * 64,
        started_at=NOW,
    ).finish(
        DatabaseWriteOutcome.COMMITTED,
        completed_at=NOW + timedelta(seconds=1),
        affected_rows=1,
        normalized_error_code=None,
    )


def _sync() -> CatalogSync:
    return CatalogSync(
        id="sync-codec",
        agent_id="agent-codec",
        source_id="source-codec",
        adapter_id="postgresql",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        source_revision="catalog:sha256:" + "a" * 64,
        resource_count=0,
        relationship_count=0,
    )


def _annotation() -> SemanticAnnotation:
    return SemanticAnnotation(
        id="annotation-codec",
        agent_id="agent-codec",
        subject=SemanticSubject(
            source_ids=("source-codec",),
            resource_ids=("resource-codec",),
            fields=(SemanticFieldReference("resource-codec", "booked_at"),),
        ),
        kind=SemanticKind.METRIC_DEFINITION,
        statement="Booked revenue uses booked_at.",
        evidence=(
            SemanticEvidence(
                SemanticEvidenceKind.USER_ASSERTION,
                "run-codec",
                message_position=0,
            ),
        ),
        catalog_revisions=(
            ResourceRevisionBinding("resource-codec", "revision-codec"),
        ),
        created_at=NOW,
        confirmed_at=NOW,
    )


def _candidate() -> tuple[LearningCandidate, LearningCandidateReviewStamp]:
    digest = "1" * 64
    reference = LearningCandidateRunReference("run-codec", digest)
    candidate = LearningCandidate(
        id="candidate-codec",
        agent_id="agent-codec",
        target=LearningCandidateTarget.MEMORY,
        content=DocumentCandidateContent("A durable definition."),
        source_ids=(),
        reviewed_runs=(reference,),
        supporting_run_ids=(reference.run_id,),
        review_fingerprint=digest,
        artifact_state_sha256="2" * 64,
        catalog_revisions=(),
        candidate_fingerprint="3" * 64,
        status=LearningCandidateStatus.AWAITING_REVIEW,
        created_at=NOW,
        updated_at=NOW,
    )
    stamp = LearningCandidateReviewStamp(
        run_id=reference.run_id,
        transcript_sha256=digest,
        artifact_state_sha256="2" * 64,
        catalog_state_sha256="4" * 64,
    )
    return candidate, stamp


def _loop_exit() -> LoopExit:
    artifact = ArtifactRef(
        artifact_id="artifact-00000000000000000000000000000001",
        run_id="run-codec",
        conversation_id="conversation-codec",
        call_id="call-codec",
        capability_id="artifact.create_document",
        filename="result.txt",
        media_type="text/plain",
        byte_size=8,
        sha256="sha256:" + "5" * 64,
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
        ),
        created_at=NOW,
    )
    delivery = ArtifactDeliveryReceipt(
        artifact_id=artifact.artifact_id,
        destination_id="destination-system-downloads",
        filename=artifact.filename,
        saved_path="/verified/Downloads/result.txt",
        byte_size=artifact.byte_size,
        sha256=artifact.sha256,
        renamed_for_collision=False,
        delivered_at=NOW,
    )
    amount = Decimal("0.1250")
    usage = ModelUsage(
        input_tokens=10,
        output_tokens=5,
        cost_estimate=CostEstimate.complete(
            amount,
            basis=CostBasis.PUBLIC_LIST,
            rate_schedule_id="public:test:2026-08",
            components=(CostComponent("model", amount),),
        ),
    )
    return LoopExit(
        run_id=artifact.run_id,
        conversation_id=artifact.conversation_id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=NOW,
        final_text="Done.",
        steps=2,
        usage=usage,
        artifacts=(artifact,),
        artifact_deliveries=(delivery,),
    )


def test_every_persisted_root_record_family_round_trips_deterministically() -> None:
    identity = AgentIdentity("agent-codec", "Codec", NOW)
    source = _source()
    receipt = _receipt()
    sync = _sync()
    snapshot = SourceCatalogSnapshot(sync=sync, resources=(), revisions=())
    run = RunInput(
        "run-codec",
        "agent-codec",
        "Question?",
        NOW,
        conversation_id="conversation-codec",
        source_id="source-codec",
        conversation_source_id="source-codec",
    )
    message = CanonicalMessage(MessageRole.USER, content=(TextBlock("Question?"),))
    result = _loop_exit()
    annotation = _annotation()
    candidate, stamp = _candidate()

    _assert_round_trip(identity, encode_identity, decode_identity)
    _assert_round_trip(receipt, encode_receipt, decode_receipt)
    _assert_round_trip(sync, encode_catalog_sync, decode_catalog_sync)
    _assert_round_trip(snapshot, encode_catalog_snapshot, decode_catalog_snapshot)
    _assert_round_trip(run, encode_run_input, decode_run_input)
    _assert_round_trip(message, encode_message, decode_message)
    _assert_round_trip(result, encode_loop_exit, decode_loop_exit)
    _assert_round_trip(
        annotation,
        encode_semantic_annotation,
        decode_semantic_annotation,
    )
    _assert_round_trip(
        candidate,
        encode_learning_candidate,
        decode_learning_candidate,
    )

    encoded_source = encode_source(source)
    stored_source = decode_source(encoded_source)
    assert stored_source.configuration.get("write_access") is None
    assert encode_source(stored_source) == encoded_source
    assert decode_identifier(encode_identifier("source-codec")) == "source-codec"
    assert decode_review_stamps(encode_review_stamps((stamp,))) == (stamp,)

    read_scope = SourceReadScope(
        agent_id="agent-codec",
        source_id="source:sha256:" + "1" * 64,
        mode=SourceReadMode.SELECTED,
        resource_ids=(
            "catalog-resource:sha256:" + "3" * 64,
            "catalog-resource:sha256:" + "2" * 64,
        ),
    )
    encoded_read_scope = encode_source_read_scope(read_scope)
    assert (
        decode_source_read_scope(
            encoded_read_scope,
            agent_id=read_scope.agent_id,
            source_id=read_scope.source_id,
        )
        == read_scope
    )
    assert encode_source_read_scope(read_scope) == encoded_read_scope

    update_scope = PostgreSQLUpdateScope(
        agent_id=read_scope.agent_id,
        source_id=read_scope.source_id,
        resource_id="catalog-resource:sha256:" + "2" * 64,
        allowed_assignment_columns=("status", "amount"),
        authorization_fingerprint="sha256:" + "4" * 64,
    )
    encoded_update_scope = encode_postgresql_update_scope(update_scope)
    assert (
        decode_postgresql_update_scope(
            encoded_update_scope,
            agent_id=update_scope.agent_id,
            source_id=update_scope.source_id,
            resource_id=update_scope.resource_id,
            authorization_fingerprint=update_scope.authorization_fingerprint,
        )
        == update_scope
    )
    assert encode_postgresql_update_scope(update_scope) == encoded_update_scope


def test_source_permission_codecs_reject_unknown_versions_and_noncanonical_sets() -> (
    None
):
    read_scope = SourceReadScope.allow_all(
        agent_id="agent-codec",
        source_id="source:sha256:" + "1" * 64,
    )
    payload = json.loads(encode_source_read_scope(read_scope))
    payload["fields"]["version"] = 2
    with pytest.raises(ValueError, match="version is unsupported"):
        decode_source_read_scope(
            json.dumps(payload),
            agent_id=read_scope.agent_id,
            source_id=read_scope.source_id,
        )

    with pytest.raises(ValueError, match="cannot contain duplicates"):
        SourceReadScope(
            agent_id=read_scope.agent_id,
            source_id=read_scope.source_id,
            mode=SourceReadMode.SELECTED,
            resource_ids=(
                "catalog-resource:sha256:" + "2" * 64,
                "catalog-resource:sha256:" + "2" * 64,
            ),
        )


def test_additive_defaults_decode_without_a_database_migration() -> None:
    run = RunInput("run-codec", "agent-codec", "Question?", NOW)
    payload = json.loads(encode_run_input(run))
    del payload["fields"]["conversation_source_id"]
    del payload["fields"]["source_id"]
    decoded = decode_run_input(json.dumps(payload))
    assert decoded.conversation_source_id is None
    assert decoded.source_id is None

    candidate, _stamp = _candidate()
    candidate_payload = json.loads(encode_learning_candidate(candidate))
    del candidate_payload["fields"]["candidate_identity_sha256"]
    assert decode_learning_candidate(json.dumps(candidate_payload)) == candidate


def test_unknown_fields_and_stored_class_names_are_rejected_explicitly() -> None:
    payload = json.loads(encode_identity(AgentIdentity("agent-codec", "Codec", NOW)))
    payload["fields"]["future"] = "not silently tolerated"
    with pytest.raises(ValueError, match="unknown fields"):
        decode_identity(json.dumps(payload))

    payload["__record__"] = "pathlib.Path"
    del payload["fields"]["future"]
    with pytest.raises(ValueError, match="not AgentIdentity"):
        decode_identity(json.dumps(payload))


def test_malformed_enum_datetime_decimal_and_nested_records_are_rejected() -> None:
    message = CanonicalMessage(MessageRole.USER, content=(TextBlock("Question?"),))
    message_payload = json.loads(encode_message(message))
    message_payload["fields"]["role"]["value"] = "administrator"
    with pytest.raises(ValueError):
        decode_message(json.dumps(message_payload))

    identity_payload = json.loads(
        encode_identity(AgentIdentity("agent-codec", "Codec", NOW))
    )
    identity_payload["fields"]["created_at"]["__datetime__"] = "2026-08-11T12:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        decode_identity(json.dumps(identity_payload))

    result_payload = json.loads(encode_loop_exit(_loop_exit()))
    result_payload["fields"]["usage"]["fields"]["cost_estimate"]["fields"][
        "amount_usd"
    ]["__decimal__"] = "not-a-decimal"
    with pytest.raises(ValueError, match="stored decimal is invalid"):
        decode_loop_exit(json.dumps(result_payload))

    result_payload["fields"]["usage"]["fields"]["cost_estimate"]["fields"][
        "amount_usd"
    ]["__decimal__"] = "NaN"
    with pytest.raises(ValueError, match="stored decimal must be finite"):
        decode_loop_exit(json.dumps(result_payload))

    message_payload = json.loads(encode_message(message))
    message_payload["fields"]["content"][0]["__record__"] = "UnknownBlock"
    with pytest.raises(ValueError, match="unsupported"):
        decode_message(json.dumps(message_payload))


def test_current_source_codec_rejects_embedded_postgresql_admission() -> None:
    payload = json.loads(encode_source(_source()))
    payload["fields"]["configuration"]["write_access"] = True
    with pytest.raises(ValueError, match="embedded write admission"):
        decode_source(json.dumps(payload))

    source = _source()
    invalid_projection = replace(
        source,
        configuration={**dict(source.configuration), "write_access": "yes"},
    )
    with pytest.raises(ValueError, match="admission must be boolean"):
        encode_source(invalid_projection)
