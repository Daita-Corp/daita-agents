"""Narrow extension manifests, admission, and local capability declarations."""

from .local import LocalCapability, tool
from .manifest import (
    ExtensionBinding,
    ExtensionBindingConflictError,
    ExtensionKind,
    ExtensionManifest,
)
from .registry import (
    ConfiguredExtension,
    ExtensionLoadError,
    ExtensionRegistration,
    ExtensionRegistry,
    RegistryDiagnostic,
)

__all__ = [
    "ConfiguredExtension",
    "ExtensionBinding",
    "ExtensionBindingConflictError",
    "ExtensionKind",
    "ExtensionLoadError",
    "ExtensionManifest",
    "ExtensionRegistration",
    "ExtensionRegistry",
    "LocalCapability",
    "RegistryDiagnostic",
    "tool",
]
