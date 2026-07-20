"""Narrow extension manifests, admission, and local capability declarations."""

from .local import LocalCapability, tool
from .manifest import ExtensionKind, ExtensionManifest
from .registry import (
    ConfiguredExtension,
    ExtensionLoadError,
    ExtensionRegistration,
    ExtensionRegistry,
    RegistryDiagnostic,
)

__all__ = [
    "ConfiguredExtension",
    "ExtensionKind",
    "ExtensionLoadError",
    "ExtensionManifest",
    "ExtensionRegistration",
    "ExtensionRegistry",
    "LocalCapability",
    "RegistryDiagnostic",
    "tool",
]
