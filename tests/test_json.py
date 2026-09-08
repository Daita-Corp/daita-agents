"""Canonical JSON remains validated and scales with nested immutable records."""

import json
from dataclasses import FrozenInstanceError

import pytest

from daita import _json


def test_nested_schema_freezing_does_not_revisit_validated_subtrees(monkeypatch):
    original = _json.freeze_json
    calls = 0

    def counted(value, *, _path="$"):
        nonlocal calls
        calls += 1
        return original(value, _path=_path)

    monkeypatch.setattr(_json, "freeze_json", counted)
    value: dict[str, object] = {"type": "string"}
    depth = 12
    for _ in range(depth):
        value = {"properties": {"child": value}, "type": "object"}
    frozen = _json.FrozenJsonObject.from_mapping(value)
    assert calls < 20 * depth
    calls = 0
    assert _json.canonical_json(frozen) == json.dumps(
        value, sort_keys=True, separators=(",", ":")
    )
    assert calls == 1


def test_freezing_detaches_mutable_inputs_and_thawing_never_mutates_evidence():
    value = {"nested": [{"values": [1, 2]}]}
    frozen = _json.FrozenJsonObject.from_mapping(value)
    encoded = _json.canonical_json(frozen)
    value["nested"][0]["values"].append(3)
    copy = frozen.to_dict()
    copied_nested = copy["nested"]
    assert isinstance(copied_nested, list)
    copied_nested.append("changed")
    assert _json.canonical_json(frozen) == encoded == '{"nested":[{"values":[1,2]}]}'
    with pytest.raises(FrozenInstanceError):
        setattr(frozen, "_items", ())


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), object(), {1: "bad key"}]
)
def test_frozen_constructor_and_mappings_both_reject_invalid_children(invalid):
    with pytest.raises((TypeError, ValueError)):
        _json.freeze_json({"nested": [invalid]})
    with pytest.raises((TypeError, ValueError)):
        _json.FrozenJsonObject((("nested", (invalid,)),))


def test_frozen_constructor_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="Duplicate"):
        _json.FrozenJsonObject((("a", 1), ("a", 2)))


def test_subclass_cannot_bypass_mapping_validation():
    class InvalidSubclass(_json.FrozenJsonObject):
        def items(self):
            return (("invalid", float("nan")),)

    with pytest.raises(ValueError, match="Non-finite"):
        _json.freeze_json(InvalidSubclass(()))
