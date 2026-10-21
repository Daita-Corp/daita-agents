"""Every type crossing the host socket round-trips exactly; nothing else encodes."""

from __future__ import annotations

import dataclasses
import json
import typing
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from daita.agent import Agent
from daita.host.codec import decode, encode, supports
from daita.host.protocol import CLIENT_SIDE_ANNOTATIONS, REMOTE_AGENT_SURFACE
from daita.llm.models import ModelSensitivity


def _wire_annotations() -> dict[str, object]:
    """Name every surface parameter and result that crosses the socket."""

    found: dict[str, object] = {}
    for member in sorted(REMOTE_AGENT_SURFACE):
        attribute = vars(Agent)[member]
        function = (
            attribute.fget
            if isinstance(attribute, property)
            else getattr(attribute, "__func__", attribute)
        )
        hints = typing.get_type_hints(function)
        client_side = CLIENT_SIDE_ANNOTATIONS.get(member, frozenset())
        assert client_side <= set(hints), f"{member} has no {client_side - set(hints)}"
        found.update(
            (f"{member}.{name}", annotation)
            for name, annotation in hints.items()
            if name not in client_side
        )
    return found


WIRE_ANNOTATIONS = _wire_annotations()
WIRE_TYPES = tuple(dict.fromkeys(WIRE_ANNOTATIONS.values()))


@pytest.mark.unit
@pytest.mark.parametrize("annotation", WIRE_ANNOTATIONS.values(), ids=WIRE_ANNOTATIONS)
def test_every_surface_parameter_and_result_has_a_codec(annotation: object) -> None:
    assert supports(annotation)


@pytest.mark.integration
@pytest.mark.parametrize("annotation", WIRE_TYPES, ids=repr)
def test_surface_values_round_trip_with_identical_types(
    annotation: object, surface_values: dict[object, tuple[object, ...]]
) -> None:
    assert annotation in surface_values, f"no realistic value for {annotation!r}"
    for value in surface_values[annotation]:
        wire = json.loads(json.dumps(encode(value), allow_nan=False))
        decoded = decode(wire, annotation)
        assert decoded == value
        _assert_same_types(decoded, value)


@dataclasses.dataclass(frozen=True)
class _Unregistered:
    name: str


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    (
        object(),
        _Unregistered("x"),
        ["a list"],
        ("nested", {1, 2}),
        {1: "non-text key"},
    ),
    ids=("object", "dataclass", "list", "nested-set", "int-key"),
)
def test_unknown_types_fail_loudly_at_encode(value: object) -> None:
    with pytest.raises(TypeError):
        encode(value)


@pytest.mark.unit
def test_naive_datetimes_unknown_tags_and_wrong_types_are_refused() -> None:
    with pytest.raises(ValueError, match="naive"):
        encode(datetime(2026, 10, 20, 9))
    with pytest.raises(ValueError, match="naive"):
        decode({"$type": "datetime", "v": "2026-10-20T09:00:00"}, datetime)
    with pytest.raises(TypeError, match="unknown host wire type"):
        decode({"$type": "daita.Pickled", "v": "gASV"})
    with pytest.raises(TypeError):
        decode(encode(ModelSensitivity.INTERNAL), str)
    assert decode(encode(datetime(2026, 10, 20, 9, tzinfo=UTC)), datetime) == (
        datetime(2026, 10, 20, 9, tzinfo=UTC)
    )


def _assert_same_types(decoded: object, original: object) -> None:
    assert type(decoded) is type(original), (decoded, original)
    if isinstance(original, tuple):
        assert isinstance(decoded, tuple)
        for left, right in zip(decoded, original, strict=True):
            _assert_same_types(left, right)
    elif isinstance(original, Mapping):
        assert isinstance(decoded, Mapping)
        for key, item in original.items():
            _assert_same_types(decoded[key], item)
    elif dataclasses.is_dataclass(original):
        for field in dataclasses.fields(original):
            if field.compare:
                _assert_same_types(
                    getattr(decoded, field.name), getattr(original, field.name)
                )
