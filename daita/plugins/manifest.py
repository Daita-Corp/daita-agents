"""
Stable plugin identity declarations for the extension registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping

_PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class PluginKind(str, Enum):
    """Primary category for a plugin's extension surface."""

    CONNECTOR = "connector"
    DOMAIN_SERVICE = "domain_service"
    RUNTIME_EXTENSION = "runtime_extension"
    WORKER_PROVIDER = "worker_provider"
    OBSERVABILITY = "observability"
    SKILL = "skill"


def _frozen_strings(
    values: frozenset[str] | set[str] | tuple[str, ...] | list[str],
) -> frozenset[str]:
    frozen = frozenset(values)
    for value in frozen:
        if not isinstance(value, str):
            raise TypeError("manifest collection values must be strings")
    return frozen


@dataclass(frozen=True)
class PluginManifest:
    """Runtime identity and declared package-level metadata for a plugin."""

    id: str
    display_name: str
    version: str
    kind: PluginKind
    domains: frozenset[str] = field(default_factory=frozenset)
    provides: frozenset[str] = field(default_factory=frozenset)
    requires: frozenset[str] = field(default_factory=frozenset)
    optional_dependencies: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not _PLUGIN_ID_RE.match(self.id):
            raise ValueError(
                "plugin id must be lowercase snake case "
                "(for example 'catalog' or 'data_quality')"
            )
        if not self.display_name:
            raise ValueError("display_name is required")
        if not self.version:
            raise ValueError("version is required")
        object.__setattr__(self, "kind", PluginKind(self.kind))
        object.__setattr__(self, "domains", _frozen_strings(self.domains))
        object.__setattr__(self, "provides", _frozen_strings(self.provides))
        object.__setattr__(self, "requires", _frozen_strings(self.requires))
        object.__setattr__(
            self,
            "optional_dependencies",
            _frozen_strings(self.optional_dependencies),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "version": self.version,
            "kind": self.kind.value,
            "domains": sorted(self.domains),
            "provides": sorted(self.provides),
            "requires": sorted(self.requires),
            "optional_dependencies": sorted(self.optional_dependencies),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PluginManifest":
        values = dict(data)
        values["kind"] = PluginKind(values["kind"])
        values["domains"] = frozenset(values.get("domains", ()))
        values["provides"] = frozenset(values.get("provides", ()))
        values["requires"] = frozenset(values.get("requires", ()))
        values["optional_dependencies"] = frozenset(
            values.get("optional_dependencies", ())
        )
        return cls(**values)
