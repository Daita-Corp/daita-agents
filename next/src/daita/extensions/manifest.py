"""Immutable identity and declaration metadata for narrow extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re

from ..capabilities import ExtensionDeclarations

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_EXTRA_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_MAX_IDENTIFIER_BYTES = 128
_MAX_HINT_BYTES = 256


def _validate_extension_id(value: str, field_name: str = "extension id") -> None:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} must be a bounded lowercase dotted identifier")


class ExtensionKind(str, Enum):
    """The three executable extension categories admitted by the MVP."""

    RESOURCE_ADAPTER = "resource_adapter"
    CAPABILITY_PROVIDER = "capability_provider"
    BACKEND_PROVIDER = "backend_provider"


@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    """Stable metadata and capability declarations for one extension."""

    id: str
    version: str
    kind: ExtensionKind
    declarations: ExtensionDeclarations = field(default_factory=ExtensionDeclarations)
    dependency_hints: tuple[str, ...] = ()
    extra: str | None = None

    def __post_init__(self) -> None:
        _validate_extension_id(self.id)
        if (
            not isinstance(self.version, str)
            or len(self.version.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
            or _VERSION_RE.fullmatch(self.version) is None
        ):
            raise ValueError("extension version must be bounded version text")
        if not isinstance(self.kind, ExtensionKind):
            raise TypeError("extension kind must be an ExtensionKind")
        if not isinstance(self.declarations, ExtensionDeclarations):
            raise TypeError("extension declarations must be ExtensionDeclarations")
        if isinstance(self.dependency_hints, (str, bytes)):
            raise TypeError("extension dependency_hints must be a string sequence")
        hints = tuple(self.dependency_hints)
        if len(hints) > 32:
            raise ValueError("extension dependency_hints exceed 32 items")
        for hint in hints:
            if (
                not isinstance(hint, str)
                or not hint.strip()
                or hint != hint.strip()
                or len(hint.encode("utf-8")) > _MAX_HINT_BYTES
                or not hint.isprintable()
            ):
                raise ValueError("extension dependency hints must be bounded text")
        if len(hints) != len(set(hints)):
            raise ValueError("extension dependency_hints must be unique")
        if self.extra is not None and (
            not isinstance(self.extra, str)
            or len(self.extra.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
            or _EXTRA_RE.fullmatch(self.extra) is None
        ):
            raise ValueError("extension extra must be a bounded package-extra name")
        object.__setattr__(self, "dependency_hints", hints)


__all__ = ["ExtensionKind", "ExtensionManifest"]
