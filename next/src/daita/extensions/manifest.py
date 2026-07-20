"""Immutable identity and declaration metadata for narrow extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import re

from .._json import canonical_json
from ..capabilities import ExtensionDeclarations

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_EXTRA_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_MAX_IDENTIFIER_BYTES = 128
_MAX_HINT_BYTES = 256
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _validate_extension_id(value: str, field_name: str = "extension id") -> None:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} must be a bounded lowercase dotted identifier")


class ExtensionKind(str, Enum):
    """Recognized categories; only capability providers execute in the MVP."""

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

    @property
    def declaration_fingerprint(self) -> str:
        declarations = self.declarations
        material = {
            "capabilities": tuple(
                {
                    "contract_fingerprint": capability.contract_fingerprint,
                    "id": capability.id,
                }
                for capability in declarations.capabilities
            ),
            "executor_ids": declarations.executor_ids,
            "tool_views": tuple(
                {
                    "capability_id": view.capability_id,
                    "description": view.description,
                    "name": view.name,
                }
                for view in declarations.tool_views
            ),
        }
        return _hash_json(material)

    @property
    def fingerprint(self) -> str:
        return _hash_json(
            {
                "declaration_fingerprint": self.declaration_fingerprint,
                "dependency_hints": self.dependency_hints,
                "extra": self.extra,
                "id": self.id,
                "kind": self.kind.value,
                "version": self.version,
            }
        )

    @property
    def binding(self) -> ExtensionBinding:
        return ExtensionBinding(
            id=self.id,
            version=self.version,
            kind=self.kind,
            declaration_fingerprint=self.declaration_fingerprint,
            manifest_fingerprint=self.fingerprint,
        )


@dataclass(frozen=True, slots=True)
class ExtensionBinding:
    """Durable non-executable identity for one explicitly configured manifest."""

    id: str
    version: str
    kind: ExtensionKind
    declaration_fingerprint: str
    manifest_fingerprint: str

    def __post_init__(self) -> None:
        _validate_extension_id(self.id)
        if (
            not isinstance(self.version, str)
            or len(self.version.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
            or _VERSION_RE.fullmatch(self.version) is None
        ):
            raise ValueError("extension binding version must be bounded version text")
        if not isinstance(self.kind, ExtensionKind):
            raise TypeError("extension binding kind must be an ExtensionKind")
        for field_name in ("declaration_fingerprint", "manifest_fingerprint"):
            if _SHA256_RE.fullmatch(getattr(self, field_name)) is None:
                raise ValueError(f"extension binding {field_name} must be sha256")


class ExtensionBindingConflictError(RuntimeError):
    """A durable Agent Home is bound to another configured manifest set."""


def extension_set_fingerprint(bindings: tuple[ExtensionBinding, ...]) -> str:
    if isinstance(bindings, (str, bytes)) or any(
        not isinstance(binding, ExtensionBinding) for binding in bindings
    ):
        raise TypeError("extension set must contain ExtensionBinding records")
    if len(bindings) > 64:
        raise ValueError("extension set exceeds 64 configured manifests")
    ids = tuple(binding.id for binding in bindings)
    if len(ids) != len(set(ids)):
        raise ValueError("extension set identities must be unique")
    return _hash_json(
        {
            "extensions": tuple(
                {
                    "declaration_fingerprint": binding.declaration_fingerprint,
                    "id": binding.id,
                    "kind": binding.kind.value,
                    "manifest_fingerprint": binding.manifest_fingerprint,
                    "version": binding.version,
                }
                for binding in bindings
            ),
            "schema_version": 1,
        }
    )


def _hash_json(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "ExtensionBinding",
    "ExtensionBindingConflictError",
    "ExtensionKind",
    "ExtensionManifest",
    "extension_set_fingerprint",
]
