"""Disposable object service dependency for real artifact lifecycle tests."""

from io import BytesIO
from typing import Any

from daita.artifacts.s3 import S3ArtifactByteStorage


class ObjectServiceError(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class ObjectClient:
    """A test dependency, not evidence of deployed S3 or SDK behavior."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.publications: list[dict[str, Any]] = []
        self.reads: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []
        self.bodies: list[BytesIO] = []

    def storage(self, prefix: str = "homes/example") -> S3ArtifactByteStorage:
        return S3ArtifactByteStorage(self, bucket="fixture-bucket", prefix=prefix)

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["ContentLength"] == len(kwargs["Body"])
        self.publications.append(kwargs)
        key = (kwargs["Bucket"], kwargs["Key"])
        if key in self.objects:
            raise ObjectServiceError("PreconditionFailed")
        self.objects[key] = kwargs["Body"]
        return {}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = (kwargs["Bucket"], kwargs["Key"])
        self.reads.append(key)
        if key not in self.objects:
            raise ObjectServiceError("NoSuchKey")
        body = BytesIO(self.objects[key])
        self.bodies.append(body)
        return {"ContentLength": len(self.objects[key]), "Body": body}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        key = (kwargs["Bucket"], kwargs["Key"])
        self.deleted.append(key)
        self.objects.pop(key, None)
        return {}
