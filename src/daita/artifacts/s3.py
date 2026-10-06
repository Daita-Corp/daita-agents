"""S3 artifact bytes; identity, manifests and lifecycle stay in the registry."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

from .models import ArtifactError, ArtifactRef

_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
S3_ARTIFACT_KEY = "{prefix}/{agent_id}/{run_id}/{artifact_id}/payload"


class S3Client(Protocol):
    """The small surface supplied by an explicitly configured S3 SDK client."""

    def put_object(self, **kwargs: Any) -> Any: ...
    def get_object(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def delete_object(self, **kwargs: Any) -> Any: ...


class S3ArtifactByteStorage:
    """Immutable payloads under a private, caller-selected S3 prefix.

    Supply a caller-owned S3 client with certificate verification, finite connect
    and read timeouts, and automatic retries disabled. Authentication, region,
    endpoint, prefix permissions and client cleanup belong to the caller. No SDK
    is imported and no credential is resolved by this module.

    Keys include the agent, run and artifact IDs. Conditional publication never
    overwrites an object. The existing registry owns the full ``ArtifactRef``;
    a second remote manifest or a bucket inventory is not an authority. Bucket
    policy must require encryption and deny public access. Optional KMS selection
    uses the same caller-owned policy. Deletion hides the current object; removal
    of historical versions follows the bucket's retention policy.
    """

    def __init__(
        self,
        client: S3Client,
        *,
        bucket: str,
        prefix: str,
        kms_key_id: str | None = None,
    ) -> None:
        if (
            not isinstance(bucket, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None
        ):
            raise ValueError("artifact bucket must be a valid S3 bucket name")
        if (
            not isinstance(prefix, str)
            or len(prefix) > 512
            or any(_SEGMENT.fullmatch(part) is None for part in prefix.split("/"))
        ):
            raise ValueError("artifact prefix must contain bounded safe segments")
        if kms_key_id is not None and (
            not isinstance(kms_key_id, str) or not kms_key_id.strip()
        ):
            raise ValueError("KMS key ID must be nonempty text")
        self._client = client
        self._bucket = bucket
        self._prefix = prefix
        self._kms_key_id = kms_key_id

    def _key(self, agent_id: str, ref: ArtifactRef) -> str:
        if not isinstance(agent_id, str) or _SEGMENT.fullmatch(agent_id) is None:
            raise ValueError("artifact agent ID must be one bounded safe segment")
        if re.fullmatch(r"run-[0-9a-f]{32}", ref.run_id) is None:
            raise ValueError("artifact run ID must use run-<32 lowercase hex>")
        return S3_ARTIFACT_KEY.format(
            prefix=self._prefix,
            agent_id=agent_id,
            run_id=ref.run_id,
            artifact_id=ref.artifact_id,
        )

    def publish(self, agent_id: str, ref: ArtifactRef, content: bytes) -> None:
        encryption = (
            {"ServerSideEncryption": "AES256"}
            if self._kms_key_id is None
            else {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self._kms_key_id}
        )
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._key(agent_id, ref),
                Body=content,
                ContentLength=len(content),
                ContentType=ref.media_type,
                IfNoneMatch="*",
                **encryption,
            )
        except Exception as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact publication could not be confirmed.",
                {"stage": "publish"},
            ) from error

    def read(self, agent_id: str, ref: ArtifactRef) -> bytes | None:
        try:
            response = self._client.get_object(
                Bucket=self._bucket, Key=self._key(agent_id, ref)
            )
        except Exception as error:
            error_response = getattr(error, "response", None)
            if (
                isinstance(error_response, Mapping)
                and isinstance(error_response.get("Error"), Mapping)
                and error_response["Error"].get("Code") == "NoSuchKey"
            ):
                return None
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact bytes could not be read.",
                {"stage": "read"},
            ) from error
        body = response["Body"]
        try:
            if response.get("ContentLength") != ref.byte_size:
                raise ArtifactError(
                    "artifact_corrupt",
                    "Artifact byte length does not match its registered identity.",
                    {"artifact_id": ref.artifact_id, "stage": "size_mismatch"},
                )
            content = body.read(ref.byte_size + 1)
            if not isinstance(content, bytes):
                raise TypeError("artifact response must contain bytes")
            return content
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact byte transfer did not complete.",
                {"stage": "read"},
            ) from error
        finally:
            body.close()

    def delete(self, agent_id: str, ref: ArtifactRef) -> None:
        try:
            self._client.delete_object(
                Bucket=self._bucket, Key=self._key(agent_id, ref)
            )
        except Exception as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact cleanup could not be confirmed.",
                {"stage": "delete_cleanup"},
            ) from error
