"""Strict immutable JSON values used at runtime trust boundaries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
import math
from typing import TypeAlias, Union

JsonScalar: TypeAlias = None | bool | int | float | str


@dataclass(frozen=True, slots=True)
class FrozenJsonObject(Mapping[str, "FrozenJsonValue"]):
    """A recursively immutable JSON object with deterministic key ordering."""

    _items: tuple[tuple[str, "FrozenJsonValue"], ...]

    def __post_init__(self) -> None:
        normalized: list[tuple[str, FrozenJsonValue]] = []
        seen: set[str] = set()
        for key, value in self._items:
            if not isinstance(key, str):
                raise TypeError("Frozen JSON object keys must be strings")
            if key in seen:
                raise ValueError(f"Duplicate JSON object key: {key}")
            seen.add(key)
            normalized.append((key, freeze_json(value, _path=f"$.{key}")))
        object.__setattr__(self, "_items", tuple(sorted(normalized)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FrozenJsonObject":
        frozen = freeze_json(value)
        if not isinstance(frozen, cls):
            raise TypeError("Expected a JSON object")
        return frozen

    def __getitem__(self, key: str) -> "FrozenJsonValue":
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, object]:
        """Return a new mutable JSON object without exposing internal values."""

        return {key: thaw_json(value) for key, value in self._items}


FrozenJsonValue: TypeAlias = Union[
    JsonScalar,
    tuple["FrozenJsonValue", ...],
    FrozenJsonObject,
]


def freeze_json(value: object, *, _path: str = "$") -> FrozenJsonValue:
    """Validate and recursively freeze a JSON-compatible value."""

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite number at {_path} is not valid JSON")
        return value

    if isinstance(value, Mapping):
        items: list[tuple[str, FrozenJsonValue]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {_path} must be a string")
            items.append((key, freeze_json(item, _path=f"{_path}.{key}")))
        return FrozenJsonObject(tuple(sorted(items, key=lambda pair: pair[0])))

    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, _path=f"{_path}[{index}]")
            for index, item in enumerate(value)
        )

    raise TypeError(
        f"Unsupported JSON value {type(value).__name__} at {_path}; "
        "implicit string conversion is forbidden"
    )


def thaw_json(value: FrozenJsonValue) -> JsonScalar | list[object] | dict[str, object]:
    """Return a new mutable JSON-compatible projection of a frozen value."""

    if isinstance(value, FrozenJsonObject):
        return {key: thaw_json(item) for key, item in value._items}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Encode a value using the one deterministic runtime JSON representation."""

    return json.dumps(
        thaw_json(freeze_json(value)),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
