from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from daita.artifacts.models import ArtifactError
from daita.artifacts.result_snapshot import (
    MAX_RESULT_SNAPSHOT_ARRAY_ITEMS,
    MAX_RESULT_SNAPSHOT_BYTES,
    MAX_RESULT_SNAPSHOT_DEPTH,
    MAX_RESULT_SNAPSHOT_OBJECT_KEYS,
    serialize_result_snapshot,
)


def test_result_snapshot_is_canonical_utf8_json() -> None:
    snapshot = serialize_result_snapshot(
        {"z": (3, 2, 1), "a": {"unicode": "Daita ✓", "enabled": True}}
    )

    assert (
        snapshot.content
        == ('{"a":{"enabled":true,"unicode":"Daita ✓"},"z":[3,2,1]}').encode()
    )
    assert json.loads(snapshot.content) == {
        "a": {"enabled": True, "unicode": "Daita ✓"},
        "z": [3, 2, 1],
    }
    assert snapshot.sha256.startswith("sha256:")


@pytest.mark.parametrize(
    "value",
    [
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": {1, 2}},
    ],
)
def test_result_snapshot_rejects_non_json_values(value: object) -> None:
    with pytest.raises(ArtifactError, match="valid finite JSON|non-JSON"):
        serialize_result_snapshot(value)  # type: ignore[arg-type]


def test_result_snapshot_enforces_structural_bounds() -> None:
    with pytest.raises(ArtifactError, match="too many keys"):
        serialize_result_snapshot(
            {str(index): index for index in range(MAX_RESULT_SNAPSHOT_OBJECT_KEYS + 1)}
        )
    with pytest.raises(ArtifactError, match="array-item"):
        serialize_result_snapshot(
            {"items": tuple(range(MAX_RESULT_SNAPSHOT_ARRAY_ITEMS + 1))}
        )
    nested: object = "leaf"
    for _index in range(MAX_RESULT_SNAPSHOT_DEPTH):
        nested = {"nested": nested}
    with pytest.raises(ArtifactError, match="depth"):
        serialize_result_snapshot(nested)  # type: ignore[arg-type]


def test_result_snapshot_enforces_time_bound() -> None:
    moments: Iterator[float] = iter((0.0, 0.1, 61.0))

    with pytest.raises(ArtifactError, match="execution-time"):
        serialize_result_snapshot(
            {"value": "bounded"},
            clock=lambda: next(moments),
        )


def test_result_snapshot_enforces_encoded_byte_bound() -> None:
    with pytest.raises(ArtifactError, match="byte limit"):
        serialize_result_snapshot({"value": "x" * MAX_RESULT_SNAPSHOT_BYTES})
