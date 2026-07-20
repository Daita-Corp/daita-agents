"""Contracts for the Phase 9 extension registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, replace

import pytest
import daita

from daita.capabilities import AccessMode, CapabilityRegistry, RiskLevel
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


def test_manifest_and_ordered_set_fingerprints_bind_runtime_declarations() -> None:
    registration = _registration()
    registry = ExtensionRegistry((registration,))
    binding = registry.bindings[0]

    assert binding.id == "example"
    assert binding.version == "1.0.0"
    assert binding.kind is ExtensionKind.CAPABILITY_PROVIDER
    assert binding.declaration_fingerprint == (
        registration.manifest.declaration_fingerprint
    )
    assert binding.manifest_fingerprint == registration.manifest.fingerprint
    assert registry.fingerprint.startswith("sha256:")
    assert len(registry.fingerprint) == 71

    changed_version = ExtensionRegistry(
        (
            replace(
                registration, manifest=replace(registration.manifest, version="2.0.0")
            ),
        )
    )
    changed_description = _registration()
    changed_local = _local()
    changed_manifest = replace(
        changed_description.manifest,
        declarations=replace(
            changed_description.manifest.declarations,
            tool_views=(
                replace(changed_local.tool_view, description="Changed projection."),
            ),
        ),
    )
    changed_projection = ExtensionRegistry(
        (
            ExtensionRegistration(
                manifest=changed_manifest,
                executors=changed_description.executors,
            ),
        )
    )

    assert changed_version.fingerprint != registry.fingerprint
    assert changed_projection.fingerprint != registry.fingerprint


def test_extension_registry_rejects_more_than_64_manifests() -> None:
    registrations = tuple(
        ExtensionRegistration(
            manifest=ExtensionManifest(
                id=f"example.extension-{index}",
                version="1.0.0",
                kind=ExtensionKind.CAPABILITY_PROVIDER,
            )
        )
        for index in range(65)
    )

    with pytest.raises(ValueError, match="exceeds 64 configured manifests"):
        ExtensionRegistry(registrations)


def test_extension_registry_composes_atomically_with_a_complete_base() -> None:
    base_local = _local(
        extension_id="builtin",
        capability_id="builtin.read",
        executor_id="builtin.read.executor",
        tool_name="builtin_read",
    )
    base = CapabilityRegistry(
        capabilities=(base_local.capability,),
        executors=(base_local.executor,),
        tool_views=(base_local.tool_view,),
    )
    registry = ExtensionRegistry((_registration(),))

    composed = registry.compose_with(base)

    assert composed.capability_ids == frozenset({"builtin.read", "example.read"})
    assert composed.tool_names == frozenset({"builtin_read", "example_read"})

    collision = ExtensionRegistry(
        (
            _registration(
                extension_id="colliding",
                capability_id="builtin.read",
                executor_id="colliding.read.executor",
                tool_name="colliding_read",
            ),
        )
    )
    with pytest.raises(ExtensionLoadError) as failure:
        collision.compose_with(base)
    assert failure.value.diagnostic.code == "extension.declaration_collision"
    assert failure.value.diagnostic.declaration_kind == "capability"
    assert base.capability_ids == frozenset({"builtin.read"})


@pytest.mark.parametrize(
    "kind",
    (ExtensionKind.RESOURCE_ADAPTER, ExtensionKind.BACKEND_PROVIDER),
)
def test_unimplemented_manifest_categories_are_explicitly_post_mvp(
    kind: ExtensionKind,
) -> None:
    registration = _registration()

    with pytest.raises(ExtensionLoadError) as failure:
        ExtensionRegistry(
            (replace(registration, manifest=replace(registration.manifest, kind=kind)),)
        )

    assert failure.value.diagnostic.code == "extension.kind_unsupported"


def test_durable_binding_requires_the_exact_configured_manifest_set() -> None:
    registry = ExtensionRegistry((_registration(),))
    registry.validate_binding(registry.bindings)

    with pytest.raises(ExtensionLoadError) as missing:
        ExtensionRegistry().validate_binding(registry.bindings)
    assert missing.value.diagnostic.code == "extension.configuration_missing"

    with pytest.raises(ExtensionLoadError) as drift:
        ExtensionRegistry(
            (
                replace(
                    _registration(),
                    manifest=replace(_registration().manifest, version="2.0.0"),
                ),
            )
        ).validate_binding(registry.bindings)
    assert drift.value.diagnostic.code == "extension.configuration_drift"


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
