"""Contracts for the Phase 9 extension registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, replace

import pytest
import daita

from daita.capabilities import AccessMode, RiskLevel
from daita.extensions import (
    ConfiguredExtension,
    ExtensionKind,
    ExtensionLoadError,
    ExtensionManifest,
    ExtensionRegistration,
    ExtensionRegistry,
    LocalCapability,
    tool,
)

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def _local(
    *,
    extension_id: str = "example",
    capability_id: str = "example.read",
    executor_id: str = "example.read.executor",
    tool_name: str = "example_read",
) -> LocalCapability:
    async def handler(arguments: Mapping[str, object]) -> Mapping[str, object]:
        return {"value": arguments["value"]}

    return tool(
        handler,
        id=capability_id,
        owner=extension_id,
        executor_id=executor_id,
        name=tool_name,
        description="Read one example value.",
        input_schema=SCHEMA,
        output_schema=SCHEMA,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
    )


def _registration(
    *,
    extension_id: str = "example",
    capability_id: str = "example.read",
    executor_id: str = "example.read.executor",
    tool_name: str = "example_read",
) -> ExtensionRegistration:
    local = _local(
        extension_id=extension_id,
        capability_id=capability_id,
        executor_id=executor_id,
        tool_name=tool_name,
    )
    manifest = ExtensionManifest(
        id=extension_id,
        version="1.0.0",
        kind=ExtensionKind.CAPABILITY_PROVIDER,
        dependency_hints=("example-sdk>=1",),
        extra="example",
        declarations=local.declarations(),
    )
    return ExtensionRegistration(manifest=manifest, executors=(local.executor,))


def test_manifest_and_registry_are_immutable_validated_declarations() -> None:
    registration = _registration()
    registry = ExtensionRegistry((registration,))

    assert registry.extension_ids == ("example",)
    assert registry.manifests == (registration.manifest,)
    assert registry.manifest("example") is registration.manifest
    assert registry.capability_registry.capability_ids == frozenset({"example.read"})
    assert registry.capability_registry.tool_definition("example_read").name == (
        "example_read"
    )
    assert tuple(item.code for item in registry.diagnostics) == (
        "extension.loaded",
        "extension.declared",
        "extension.declared",
        "extension.declared",
    )
    assert {
        (item.declaration_kind, item.declaration_id)
        for item in registry.diagnostics[1:]
    } == {
        ("capability", "example.read"),
        ("executor", "example.read.executor"),
        ("tool_view", "example_read"),
    }

    with pytest.raises(FrozenInstanceError):
        registration.manifest.version = "2.0.0"  # type: ignore[misc]
    with pytest.raises(KeyError, match="unknown extension"):
        registry.manifest("missing")


@pytest.mark.parametrize("extension_id", ("", "Example", "bad/id", "bad id"))
def test_manifest_rejects_unstable_extension_ids(extension_id: str) -> None:
    with pytest.raises(ValueError, match="extension id"):
        ExtensionManifest(
            id=extension_id,
            version="1.0.0",
            kind=ExtensionKind.CAPABILITY_PROVIDER,
        )


def test_registry_rejects_manifest_ownership_and_implementation_mismatches() -> None:
    registration = _registration()
    foreign_manifest = replace(registration.manifest, id="foreign")

    with pytest.raises(ExtensionLoadError) as ownership:
        ExtensionRegistry((replace(registration, manifest=foreign_manifest),))
    assert ownership.value.diagnostic.code == "extension.owner_mismatch"
    assert "Read one example value" not in str(ownership.value)

    with pytest.raises(ValueError, match="exactly match"):
        ExtensionRegistration(manifest=registration.manifest, executors=())


@pytest.mark.parametrize(
    ("second", "declaration_kind", "declaration_id"),
    (
        (
            _registration(
                extension_id="second",
                capability_id="example.read",
                executor_id="second.read.executor",
                tool_name="second_read",
            ),
            "capability",
            "example.read",
        ),
        (
            _registration(
                extension_id="second",
                capability_id="second.read",
                executor_id="example.read.executor",
                tool_name="second_read",
            ),
            "executor",
            "example.read.executor",
        ),
        (
            _registration(
                extension_id="second",
                capability_id="second.read",
                executor_id="second.read.executor",
                tool_name="example_read",
            ),
            "tool_view",
            "example_read",
        ),
    ),
)
def test_registry_rejects_declaration_collisions_atomically(
    second: ExtensionRegistration,
    declaration_kind: str,
    declaration_id: str,
) -> None:
    with pytest.raises(ExtensionLoadError) as failure:
        ExtensionRegistry((_registration(), second))

    diagnostic = failure.value.diagnostic
    assert diagnostic.code == "extension.declaration_collision"
    assert diagnostic.extension_id == "second"
    assert diagnostic.declaration_kind == declaration_kind
    assert diagnostic.declaration_id == declaration_id


def test_explicit_loading_validates_ids_before_running_any_factory() -> None:
    calls: list[str] = []

    def factory() -> ExtensionRegistration:
        calls.append("called")
        return _registration()

    configured = ConfiguredExtension(id="example", factory=factory)
    assert calls == []

    with pytest.raises(ExtensionLoadError) as duplicate:
        ExtensionRegistry.load((configured, configured))

    assert duplicate.value.diagnostic.code == "extension.id_collision"
    assert calls == []


def test_explicit_loading_is_atomic_and_does_not_expose_loader_errors() -> None:
    secret = "super-secret-extension-token"
    calls: list[str] = []

    def good() -> ExtensionRegistration:
        calls.append("good")
        return _registration()

    def bad() -> ExtensionRegistration:
        calls.append("bad")
        raise RuntimeError(secret)

    configured: tuple[ConfiguredExtension, ...] = (
        ConfiguredExtension(id="example", factory=good),
        ConfiguredExtension(id="failing", factory=bad),
    )
    with pytest.raises(ExtensionLoadError) as failure:
        ExtensionRegistry.load(configured)

    assert calls == ["good", "bad"]
    assert failure.value.diagnostic.code == "extension.load_failed"
    assert failure.value.diagnostic.extension_id == "failing"
    assert secret not in str(failure.value)
    assert secret not in repr(failure.value.diagnostic)
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None


def test_explicit_loader_cannot_substitute_a_different_manifest_identity() -> None:
    factory: Callable[[], ExtensionRegistration] = lambda: _registration()

    with pytest.raises(ExtensionLoadError) as failure:
        ExtensionRegistry.load((ConfiguredExtension(id="configured", factory=factory),))

    assert failure.value.diagnostic.code == "extension.identity_mismatch"
    assert failure.value.diagnostic.extension_id == "configured"


def test_registry_has_no_cross_lifecycle_plugin_hooks() -> None:
    public_names = {
        name for name in ExtensionRegistry.__dict__ if not name.startswith("_")
    }

    assert "setup" not in public_names
    assert "teardown" not in public_names
    assert "before_turn" not in public_names
    assert "after_turn" not in public_names
    assert "install" not in public_names


def test_root_surface_exports_narrow_extensions_without_plugin_base_classes() -> None:
    assert daita.ExtensionRegistry is ExtensionRegistry
    assert daita.ExtensionManifest is ExtensionManifest
    assert daita.RegistryDiagnostic.__name__ == "RegistryDiagnostic"
    assert daita.tool is tool
    for removed in (
        "BasePlugin",
        "ConnectorPlugin",
        "DomainServicePlugin",
        "ObservabilityPlugin",
        "RuntimeExtensionPlugin",
        "SkillPlugin",
        "WorkerProviderPlugin",
    ):
        assert not hasattr(daita, removed)
