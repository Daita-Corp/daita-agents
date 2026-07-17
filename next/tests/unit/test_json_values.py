from __future__ import annotations

import math

import pytest

from daita._json import FrozenJsonObject, canonical_json, freeze_json, thaw_json


def test_json_values_are_recursively_frozen_and_isolated_from_input_mutation() -> None:
    filters: dict[str, object] = {"active": True}
    keys = ["alpha", "beta"]
    original: dict[str, object] = {
        "filters": filters,
        "keys": keys,
        "limit": 10,
    }

    frozen = FrozenJsonObject.from_mapping(original)
    filters["active"] = False
    keys.append("gamma")

    assert thaw_json(frozen) == {
        "filters": {"active": True},
        "keys": ["alpha", "beta"],
        "limit": 10,
    }
    assert isinstance(frozen["filters"], FrozenJsonObject)
    assert frozen["keys"] == ("alpha", "beta")

    with pytest.raises(TypeError):
        frozen["limit"] = 20  # type: ignore[index]


def test_canonical_json_is_stable_across_mapping_order() -> None:
    left = {"z": [3, 2, 1], "a": {"b": 2, "a": 1}}
    right = {"a": {"a": 1, "b": 2}, "z": [3, 2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":{"a":1,"b":2},"z":[3,2,1]}'


def test_direct_frozen_object_construction_still_validates_its_boundary() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        FrozenJsonObject((("key", 1), ("key", 2)))

    with pytest.raises(TypeError, match="Unsupported JSON"):
        FrozenJsonObject((("bad", object()),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        {"bad": object()},
        {"bad": {"set"}},
        {"bad": math.nan},
        {"bad": math.inf},
        {1: "non-string key"},
    ],
)
def test_non_json_values_are_rejected_strictly(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        freeze_json(value)
