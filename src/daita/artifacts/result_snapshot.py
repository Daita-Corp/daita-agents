"""Validate and canonically serialize one bounded structured tool result."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic

from .._json import canonical_json
from .models import ArtifactError

MAX_RESULT_SNAPSHOT_BYTES = 8 * 1024 * 1024
MAX_RESULT_SNAPSHOT_DEPTH = 16
MAX_RESULT_SNAPSHOT_OBJECT_KEYS = 128
MAX_RESULT_SNAPSHOT_ARRAY_ITEMS = 10_000
MAX_RESULT_SNAPSHOT_SECONDS = 60.0
RESULT_SNAPSHOT_MEDIA_TYPE = "application/json"
RESULT_SNAPSHOT_ALLOWED_EXTENSIONS = ((RESULT_SNAPSHOT_MEDIA_TYPE, (".json",)),)


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    """One complete canonical JSON result snapshot."""

    content: bytes
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.content, bytes)
            or not self.content
            or len(self.content) > MAX_RESULT_SNAPSHOT_BYTES
        ):
            raise ValueError("result snapshot content is outside its byte bound")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 71
            or not self.sha256.startswith("sha256:")
        ):
            raise ValueError("result snapshot checksum is invalid")


def serialize_result_snapshot(
    value: Mapping[str, object],
    *,
    clock: Callable[[], float] = monotonic,
    maximum_seconds: float = MAX_RESULT_SNAPSHOT_SECONDS,
) -> ResultSnapshot:
    """Return canonical UTF-8 JSON after applying the fixed D2 bounds."""

    if not isinstance(value, Mapping):
        raise ArtifactError(
            "artifact_snapshot_invalid",
            "The selected tool result is not a structured JSON object.",
        )
    if (
        not isinstance(maximum_seconds, (int, float))
        or isinstance(maximum_seconds, bool)
        or not 0 < float(maximum_seconds) <= MAX_RESULT_SNAPSHOT_SECONDS
    ):
        raise ValueError("result snapshot time bound is invalid")
    started = clock()
    array_items = 0

    def check_time() -> None:
        if clock() - started > float(maximum_seconds):
            raise ArtifactError(
                "artifact_snapshot_limited",
                "The result snapshot exceeded its execution-time limit.",
                {"reason": "time_limit"},
            )

    def validate(item: object, *, depth: int) -> None:
        nonlocal array_items
        check_time()
        if depth > MAX_RESULT_SNAPSHOT_DEPTH:
            raise ArtifactError(
                "artifact_snapshot_limited",
                "The selected tool result exceeds the JSON depth limit.",
                {"reason": "depth_limit"},
            )
        if isinstance(item, Mapping):
            if len(item) > MAX_RESULT_SNAPSHOT_OBJECT_KEYS:
                raise ArtifactError(
                    "artifact_snapshot_limited",
                    "The selected tool result has too many keys in one object.",
                    {"reason": "object_key_limit"},
                )
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ArtifactError(
                        "artifact_snapshot_invalid",
                        "The selected tool result contains a non-text object key.",
                    )
                validate(nested, depth=depth + 1)
            return
        if isinstance(item, (tuple, list)):
            array_items += len(item)
            if array_items > MAX_RESULT_SNAPSHOT_ARRAY_ITEMS:
                raise ArtifactError(
                    "artifact_snapshot_limited",
                    "The selected tool result exceeds the aggregate array-item limit.",
                    {"reason": "array_item_limit"},
                )
            for nested in item:
                validate(nested, depth=depth + 1)
            return
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        raise ArtifactError(
            "artifact_snapshot_invalid",
            "The selected tool result contains a non-JSON value.",
            {"value_type": type(item).__name__},
        )

    try:
        validate(value, depth=1)
        encoded = canonical_json(value).encode("utf-8")
    except ArtifactError:
        raise
    except (TypeError, ValueError) as error:
        raise ArtifactError(
            "artifact_snapshot_invalid",
            "The selected tool result is not valid finite JSON.",
        ) from error
    check_time()
    if not encoded or len(encoded) > MAX_RESULT_SNAPSHOT_BYTES:
        raise ArtifactError(
            "artifact_snapshot_limited",
            "The selected tool result exceeds the snapshot byte limit.",
            {"reason": "byte_limit", "byte_size": len(encoded)},
        )
    return ResultSnapshot(
        content=encoded,
        sha256="sha256:" + sha256(encoded).hexdigest(),
    )


__all__ = [
    "MAX_RESULT_SNAPSHOT_ARRAY_ITEMS",
    "MAX_RESULT_SNAPSHOT_BYTES",
    "MAX_RESULT_SNAPSHOT_DEPTH",
    "MAX_RESULT_SNAPSHOT_OBJECT_KEYS",
    "MAX_RESULT_SNAPSHOT_SECONDS",
    "RESULT_SNAPSHOT_ALLOWED_EXTENSIONS",
    "RESULT_SNAPSHOT_MEDIA_TYPE",
    "ResultSnapshot",
    "serialize_result_snapshot",
]
