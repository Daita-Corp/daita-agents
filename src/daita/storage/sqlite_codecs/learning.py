"""Explicit SQLite codecs for learning-candidate and review-stamp records."""

from __future__ import annotations

from ...learning_candidates import (
    DocumentCandidateContent,
    LearningCandidate,
    LearningCandidateAction,
    LearningCandidateRejectionReason,
    LearningCandidateReviewStamp,
    LearningCandidateRunReference,
    LearningCandidateStatus,
    LearningCandidateTarget,
    SemanticCandidateContent,
    SkillCandidateContent,
)
from ...semantics import SemanticKind
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    enum_decode,
    enum_encode,
    load_payload,
    optional_text,
    record,
    record_fields,
    sequence,
    text,
)
from .semantics import (
    decode_revision_binding,
    decode_subject,
    encode_revision_binding,
    encode_subject,
)


def encode_learning_candidate(value: LearningCandidate) -> str:
    if not isinstance(value, LearningCandidate):
        raise TypeError("learning codec requires LearningCandidate")
    return dump_payload(_encode_candidate(value))


def decode_learning_candidate(value: str) -> LearningCandidate:
    return _decode_candidate(load_payload(value))


def encode_review_stamps(values: tuple[LearningCandidateReviewStamp, ...]) -> str:
    if any(not isinstance(item, LearningCandidateReviewStamp) for item in values):
        raise TypeError("review-stamp codec requires review stamps")
    return dump_payload([_encode_review_stamp(item) for item in values])


def decode_review_stamps(value: str) -> tuple[LearningCandidateReviewStamp, ...]:
    return tuple(
        _decode_review_stamp(item)
        for item in sequence(load_payload(value), "learning review stamps")
    )


def _encode_run_reference(
    value: LearningCandidateRunReference,
) -> dict[str, JsonValue]:
    return record(
        "LearningCandidateRunReference",
        {"run_id": value.run_id, "transcript_sha256": value.transcript_sha256},
    )


def _decode_run_reference(value: JsonValue) -> LearningCandidateRunReference:
    fields = record_fields(
        value,
        "LearningCandidateRunReference",
        ("run_id", "transcript_sha256"),
    )
    return LearningCandidateRunReference(
        run_id=text(fields["run_id"], "candidate run_id"),
        transcript_sha256=text(
            fields["transcript_sha256"], "candidate transcript sha256"
        ),
    )


def _encode_review_stamp(
    value: LearningCandidateReviewStamp,
) -> dict[str, JsonValue]:
    return record(
        "LearningCandidateReviewStamp",
        {
            "run_id": value.run_id,
            "transcript_sha256": value.transcript_sha256,
            "artifact_state_sha256": value.artifact_state_sha256,
            "catalog_state_sha256": value.catalog_state_sha256,
        },
    )


def _decode_review_stamp(value: JsonValue) -> LearningCandidateReviewStamp:
    fields = record_fields(
        value,
        "LearningCandidateReviewStamp",
        (
            "run_id",
            "transcript_sha256",
            "artifact_state_sha256",
            "catalog_state_sha256",
        ),
    )
    return LearningCandidateReviewStamp(
        run_id=text(fields["run_id"], "review stamp run_id"),
        transcript_sha256=text(
            fields["transcript_sha256"], "review stamp transcript sha256"
        ),
        artifact_state_sha256=text(
            fields["artifact_state_sha256"], "review stamp artifact sha256"
        ),
        catalog_state_sha256=text(
            fields["catalog_state_sha256"], "review stamp catalog sha256"
        ),
    )


def _encode_document_content(
    value: DocumentCandidateContent,
) -> dict[str, JsonValue]:
    return record("DocumentCandidateContent", {"text": value.text})


def _decode_document_content(value: JsonValue) -> DocumentCandidateContent:
    fields = record_fields(value, "DocumentCandidateContent", ("text",))
    return DocumentCandidateContent(text(fields["text"], "candidate document text"))


def _encode_semantic_content(
    value: SemanticCandidateContent,
) -> dict[str, JsonValue]:
    return record(
        "SemanticCandidateContent",
        {
            "action": enum_encode(value.action, "LearningCandidateAction"),
            "subject": None if value.subject is None else encode_subject(value.subject),
            "kind": (
                None if value.kind is None else enum_encode(value.kind, "SemanticKind")
            ),
            "statement": value.statement,
            "catalog_revisions": [
                encode_revision_binding(item) for item in value.catalog_revisions
            ],
            "annotation_id": value.annotation_id,
            "supersedes_id": value.supersedes_id,
        },
    )


def _decode_semantic_content(value: JsonValue) -> SemanticCandidateContent:
    fields = record_fields(
        value,
        "SemanticCandidateContent",
        (
            "action",
            "subject",
            "kind",
            "statement",
            "catalog_revisions",
            "annotation_id",
            "supersedes_id",
        ),
    )
    return SemanticCandidateContent(
        action=enum_decode(
            fields["action"], LearningCandidateAction, "LearningCandidateAction"
        ),
        subject=(
            None if fields["subject"] is None else decode_subject(fields["subject"])
        ),
        kind=(
            None
            if fields["kind"] is None
            else enum_decode(fields["kind"], SemanticKind, "SemanticKind")
        ),
        statement=optional_text(fields["statement"], "semantic candidate statement"),
        catalog_revisions=tuple(
            decode_revision_binding(item)
            for item in sequence(
                fields["catalog_revisions"], "candidate catalog revisions"
            )
        ),
        annotation_id=optional_text(fields["annotation_id"], "candidate annotation id"),
        supersedes_id=optional_text(fields["supersedes_id"], "candidate supersedes id"),
    )


def _encode_skill_content(value: SkillCandidateContent) -> dict[str, JsonValue]:
    return record(
        "SkillCandidateContent",
        {
            "action": enum_encode(value.action, "LearningCandidateAction"),
            "name": value.name,
            "description": value.description,
            "instructions": value.instructions,
        },
    )


def _decode_skill_content(value: JsonValue) -> SkillCandidateContent:
    fields = record_fields(
        value,
        "SkillCandidateContent",
        ("action", "name", "description", "instructions"),
    )
    return SkillCandidateContent(
        action=enum_decode(
            fields["action"], LearningCandidateAction, "LearningCandidateAction"
        ),
        name=text(fields["name"], "candidate skill name"),
        description=optional_text(fields["description"], "candidate skill description"),
        instructions=optional_text(
            fields["instructions"], "candidate skill instructions"
        ),
    )


def _decode_content(value: JsonValue) -> object:
    if not isinstance(value, dict):
        raise ValueError("stored learning candidate content is invalid")
    record_name = value.get("__record__")
    if record_name == "DocumentCandidateContent":
        return _decode_document_content(value)
    if record_name == "SemanticCandidateContent":
        return _decode_semantic_content(value)
    if record_name == "SkillCandidateContent":
        return _decode_skill_content(value)
    raise ValueError("stored learning candidate content type is unsupported")


def _encode_candidate(value: LearningCandidate) -> dict[str, JsonValue]:
    if isinstance(value.content, DocumentCandidateContent):
        content = _encode_document_content(value.content)
    elif isinstance(value.content, SemanticCandidateContent):
        content = _encode_semantic_content(value.content)
    elif isinstance(value.content, SkillCandidateContent):
        content = _encode_skill_content(value.content)
    else:
        raise TypeError("learning candidate content type is unsupported")
    return record(
        "LearningCandidate",
        {
            "id": value.id,
            "agent_id": value.agent_id,
            "target": enum_encode(value.target, "LearningCandidateTarget"),
            "content": content,
            "source_ids": list(value.source_ids),
            "reviewed_runs": [
                _encode_run_reference(item) for item in value.reviewed_runs
            ],
            "supporting_run_ids": list(value.supporting_run_ids),
            "review_fingerprint": value.review_fingerprint,
            "artifact_state_sha256": value.artifact_state_sha256,
            "catalog_revisions": [
                encode_revision_binding(item) for item in value.catalog_revisions
            ],
            "candidate_fingerprint": value.candidate_fingerprint,
            "status": enum_encode(value.status, "LearningCandidateStatus"),
            "created_at": datetime_encode(value.created_at),
            "updated_at": datetime_encode(value.updated_at),
            "rejection_reason": (
                None
                if value.rejection_reason is None
                else enum_encode(
                    value.rejection_reason,
                    "LearningCandidateRejectionReason",
                )
            ),
            "candidate_identity_sha256": value.candidate_identity_sha256,
        },
    )


def _decode_candidate(value: JsonValue) -> LearningCandidate:
    fields = record_fields(
        value,
        "LearningCandidate",
        (
            "id",
            "agent_id",
            "target",
            "content",
            "source_ids",
            "reviewed_runs",
            "supporting_run_ids",
            "review_fingerprint",
            "artifact_state_sha256",
            "catalog_revisions",
            "candidate_fingerprint",
            "status",
            "created_at",
            "updated_at",
            "rejection_reason",
            "candidate_identity_sha256",
        ),
    )
    return LearningCandidate(
        id=text(fields["id"], "learning candidate id"),
        agent_id=text(fields["agent_id"], "learning candidate agent_id"),
        target=enum_decode(
            fields["target"], LearningCandidateTarget, "LearningCandidateTarget"
        ),
        content=_decode_content(fields["content"]),  # type: ignore[arg-type]
        source_ids=tuple(
            text(item, "candidate source_id")
            for item in sequence(fields["source_ids"], "candidate source_ids")
        ),
        reviewed_runs=tuple(
            _decode_run_reference(item)
            for item in sequence(fields["reviewed_runs"], "candidate reviewed runs")
        ),
        supporting_run_ids=tuple(
            text(item, "candidate supporting run id")
            for item in sequence(
                fields["supporting_run_ids"], "candidate supporting runs"
            )
        ),
        review_fingerprint=text(
            fields["review_fingerprint"], "candidate review fingerprint"
        ),
        artifact_state_sha256=text(
            fields["artifact_state_sha256"], "candidate artifact state sha256"
        ),
        catalog_revisions=tuple(
            decode_revision_binding(item)
            for item in sequence(
                fields["catalog_revisions"], "candidate catalog revisions"
            )
        ),
        candidate_fingerprint=text(
            fields["candidate_fingerprint"], "candidate fingerprint"
        ),
        status=enum_decode(
            fields["status"], LearningCandidateStatus, "LearningCandidateStatus"
        ),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
        rejection_reason=(
            None
            if fields["rejection_reason"] is None
            else enum_decode(
                fields["rejection_reason"],
                LearningCandidateRejectionReason,
                "LearningCandidateRejectionReason",
            )
        ),
        candidate_identity_sha256=text(
            fields["candidate_identity_sha256"], "candidate identity sha256"
        ),
    )
