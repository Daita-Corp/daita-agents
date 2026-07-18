from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import hashlib
from typing import Any

import pytest

import daita.capabilities as capability_models
from daita.capabilities import EvidenceCandidate
from daita.operations.models import Evidence

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
ARTIFACT_CONTENT = b'{"rows":[{"key":"alpha","value":"ALPHA"}]}'
ARTIFACT_DIGEST = "sha256:" + hashlib.sha256(ARTIFACT_CONTENT).hexdigest()


def _artifact_type() -> type[Any]:
    artifact_type = getattr(capability_models, "EvidenceArtifact", None)
    assert artifact_type is not None, (
        "daita.capabilities must define the immutable untrusted "
        "EvidenceArtifact boundary record"
    )
    return artifact_type


def _artifact(**overrides: object) -> Any:
    values: dict[str, object] = {
        "content": ARTIFACT_CONTENT,
        "media_type": "application/json",
        "sensitivity_class": "internal",
        "retention_class": "operation",
        "encryption_metadata": {
            "algorithm": "AES-256-GCM",
            "key_id": "agent-home-key-1",
        },
    }
    values.update(overrides)
    return _artifact_type()(**values)


def _evidence(**overrides: object) -> Evidence:
    values: dict[str, object] = {
        "id": "evidence-1",
        "operation_id": "operation-1",
        "task_id": "task-1",
        "turn_id": "turn-1",
        "capability_id": "fake.read",
        "executor_id": "fake.read.executor",
        "kind": "fake.read.result",
        "schema_version": 1,
        "attempt": 1,
        "accepted": True,
        "payload": {"key": "alpha", "value": "ALPHA"},
        "content_hash": ARTIFACT_DIGEST,
        "created_at": NOW,
    }
    values.update(overrides)
    evidence_type: Any = Evidence
    return evidence_type(**values)


def test_evidence_artifact_is_an_immutable_untrusted_bytes_record() -> None:
    artifact = _artifact()
    candidate_factory: Any = EvidenceCandidate
    candidate = candidate_factory(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
        artifact=artifact,
    )

    assert artifact.content == ARTIFACT_CONTENT
    assert artifact.media_type == "application/json"
    assert artifact.sensitivity_class == "internal"
    assert artifact.retention_class == "operation"
    assert artifact.encryption_metadata.to_dict() == {
        "algorithm": "AES-256-GCM",
        "key_id": "agent-home-key-1",
    }
    assert candidate.artifact is artifact
    assert not any(
        hasattr(artifact, authoritative_identity)
        for authoritative_identity in (
            "blob_id",
            "evidence_id",
            "operation_id",
            "task_id",
        )
    )

    with pytest.raises(FrozenInstanceError):
        setattr(artifact, "content", b"forged")
    with pytest.raises(TypeError):
        artifact.encryption_metadata["key_id"] = "forged"
    with pytest.raises(FrozenInstanceError):
        setattr(candidate, "artifact", None)


def test_inline_evidence_candidate_keeps_artifact_optional() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )

    assert candidate.artifact is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("content", bytearray(ARTIFACT_CONTENT)),
        ("media_type", "   "),
        ("sensitivity_class", "   "),
        ("retention_class", "   "),
    ],
)
def test_evidence_artifact_rejects_mutable_bytes_and_blank_metadata(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field_name):
        _artifact(**{field_name: invalid_value})


def test_evidence_blob_id_is_an_explicit_optional_immutable_link() -> None:
    inline = _evidence()
    artifact = _evidence(blob_id="blob-1")

    assert inline.blob_id is None
    assert artifact.blob_id == "blob-1"
    assert artifact.content_hash == ARTIFACT_DIGEST
    with pytest.raises(FrozenInstanceError):
        setattr(artifact, "blob_id", "blob-forged")


def test_evidence_rejects_a_blank_blob_id() -> None:
    with pytest.raises(ValueError, match="blob_id"):
        _evidence(blob_id="   ")
