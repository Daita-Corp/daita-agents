"""Bounded S3 byte adapter behavior with an injected object-service client."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO

import pytest

from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactError,
    ArtifactProvenance,
    ArtifactRef,
)
from daita.artifacts.s3 import S3ArtifactByteStorage
from daita.catalog.models import Sensitivity
from tests.artifacts.byte_storage_support import ObjectClient, ObjectServiceError
from tests.support.graph import GRAPH_NOW


@pytest.fixture
def ref():
    return ArtifactRef(
        artifact_id="artifact-" + "1" * 32,
        run_id="run-" + "2" * 32,
        conversation_id="conversation-one",
        call_id="call-one",
        capability_id="artifact.create_document",
        filename="result.txt",
        media_type="text/plain",
        byte_size=3,
        sha256="sha256:" + sha256(b"one").hexdigest(),
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
        ),
        created_at=GRAPH_NOW,
    )


def test_s3_publication_is_encrypted_exclusive_and_scoped(ref):
    client = ObjectClient()
    store = S3ArtifactByteStorage(
        client, bucket="fixture-bucket", prefix="homes/one", kms_key_id="key-one"
    )
    store.publish("agent-one", ref, b"one")
    request = client.publications[0]
    assert request["ServerSideEncryption"] == "aws:kms"
    assert request["SSEKMSKeyId"] == "key-one"
    assert request["Key"] == (
        f"homes/one/agent-one/{ref.run_id}/{ref.artifact_id}/payload"
    )
    with pytest.raises(ArtifactError):
        store.publish("agent-one", ref, b"two")
    assert store.read("agent-one", ref) == b"one"
    assert store.read("agent-two", ref) is None
    assert client.storage("homes/two").read("agent-one", ref) is None
    assert all(body.closed for body in client.bodies)


@pytest.mark.parametrize("code", ["AccessDenied", "NoSuchBucket", "SlowDown", "404"])
def test_only_no_such_key_means_absent(ref, monkeypatch, code):
    client = ObjectClient()

    def failed(**kwargs):
        raise ObjectServiceError(code)

    monkeypatch.setattr(client, "get_object", failed)
    with pytest.raises(ArtifactError) as error:
        client.storage().read("agent-one", ref)
    assert error.value.code == "artifact_storage_failed"


def test_oversized_response_is_closed_without_reading(ref, monkeypatch):
    client = ObjectClient()

    class Unreadable(BytesIO):
        def read(self, size=-1):
            pytest.fail("oversized remote response must not be downloaded")

    body = Unreadable(b"too long")
    monkeypatch.setattr(
        client, "get_object", lambda **kwargs: {"Body": body, "ContentLength": 8}
    )
    with pytest.raises(ArtifactError) as error:
        client.storage().read("agent-one", ref)
    assert error.value.code == "artifact_corrupt"
    assert body.closed


def test_response_read_is_bounded_and_transfer_failure_closes_it(ref, monkeypatch):
    client = ObjectClient()

    class Broken(BytesIO):
        def read(self, size=-1):
            assert size == 4
            raise TimeoutError("transfer interrupted")

    body = Broken()
    monkeypatch.setattr(
        client, "get_object", lambda **kwargs: {"Body": body, "ContentLength": 3}
    )
    with pytest.raises(ArtifactError) as error:
        client.storage().read("agent-one", ref)
    assert error.value.code == "artifact_storage_failed"
    assert body.closed


@pytest.mark.parametrize(
    "prefix", ["", "/root", "root/", "root/../other", "root//other"]
)
def test_prefix_cannot_escape_its_selected_scope(prefix):
    with pytest.raises(ValueError):
        S3ArtifactByteStorage(ObjectClient(), bucket="fixture-bucket", prefix=prefix)


def test_object_identity_cannot_escape_its_selected_scope(ref):
    client = ObjectClient()
    store = client.storage()
    with pytest.raises(ArtifactError):
        store.read("../other", ref)
    with pytest.raises(ArtifactError):
        store.publish("agent-one", replace(ref, run_id="../other"), b"one")
    assert not client.reads and not client.publications
