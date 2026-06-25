from dataclasses import dataclass

import daita
import daita.plugins as plugins
from daita.plugins import (
    BasePlugin,
    ConnectorPlugin,
    DomainServicePlugin,
    EmptySecretProvider,
    ObservabilityPlugin,
    PluginContext,
    PluginKind,
    PluginManifest,
    RuntimeExtensionPlugin,
    SecretProvider,
    ServiceRegistry,
    SkillPlugin,
    WorkerProviderPlugin,
)
import daita.plugins.base as plugin_base
from daita.plugins.registry import ExtensionRegistry, RegistryDiagnostic
from daita.runtime import AccessMode, Capability, RiskLevel


class MinimalRuntimeExtension(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="minimal_runtime",
        display_name="Minimal Runtime",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )


class CapturingDomainService(DomainServicePlugin):
    manifest = PluginManifest(
        id="catalog",
        display_name="Catalog",
        version="2.0.0",
        kind=PluginKind.DOMAIN_SERVICE,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.context = None
        self.teardown_called = False

    async def setup(self, context: PluginContext) -> None:
        self.context = context

    async def teardown(self) -> None:
        self.teardown_called = True

    def declare_capabilities(self):
        return (
            Capability(
                id="catalog.schema.search",
                owner="catalog",
                description="Search catalog schema.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"schema.query"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"schema.search_result"}),
                executor="catalog.schema.search",
                side_effecting=False,
            ),
        )


class ConcreteConnector(ConnectorPlugin):
    manifest = PluginManifest(
        id="sqlite",
        display_name="SQLite",
        version="1.0.0",
        kind=PluginKind.CONNECTOR,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected


@dataclass(frozen=True)
class RecordingSecretProvider:
    value: str

    def get_secret(self, name: str) -> str | None:
        return self.value if name == "api_key" else None


def test_base_plugin_declares_extension_first_noop_surfaces():
    plugin = MinimalRuntimeExtension()

    assert plugin.declare_capabilities() == ()
    assert plugin.get_executors() == ()
    assert plugin.declare_policies() == ()
    assert plugin.declare_evidence_schemas() == ()
    assert plugin.get_context_providers() == ()
    assert plugin.get_tool_views() == ()
    assert plugin.get_workers() == ()
    assert not hasattr(plugin, "initialize")
    assert not hasattr(plugin, "get_tools")


async def test_plugin_setup_receives_structured_context_and_teardown_runs():
    services = ServiceRegistry({"catalog_backend": object()})
    secrets = RecordingSecretProvider("secret-value")
    context = PluginContext(
        runtime_id="db-runtime-1",
        runtime_kind="db",
        agent_id="agent-1",
        services=services,
        config={"profile": "analyst"},
        secrets=secrets,
    )
    plugin = CapturingDomainService()

    await plugin.setup(context)
    await plugin.teardown()

    assert plugin.context is context
    assert plugin.context.services.require("catalog_backend") is services.require(
        "catalog_backend"
    )
    assert plugin.context.config["profile"] == "analyst"
    assert plugin.context.secrets.get_secret("api_key") == "secret-value"
    assert plugin.teardown_called is True


async def test_connector_plugin_owns_connection_lifecycle():
    plugin = ConcreteConnector()

    await plugin.connect()
    assert plugin.is_connected is True

    await plugin.disconnect()
    assert plugin.is_connected is False


def test_specialized_base_classes_carry_expected_plugin_kinds():
    assert DomainServicePlugin.expected_kind is PluginKind.DOMAIN_SERVICE
    assert RuntimeExtensionPlugin.expected_kind is PluginKind.RUNTIME_EXTENSION
    assert WorkerProviderPlugin.expected_kind is PluginKind.WORKER_PROVIDER
    assert ObservabilityPlugin.expected_kind is PluginKind.OBSERVABILITY
    assert SkillPlugin.expected_kind is PluginKind.SKILL
    assert ConcreteConnector.expected_kind is PluginKind.CONNECTOR


def test_service_registry_and_empty_secret_provider_are_safe_defaults():
    services = ServiceRegistry()
    service = object()

    services.register("runtime_store", service)

    assert services.get("runtime_store") is service
    assert services.get("missing", "fallback") == "fallback"
    assert services.as_dict() == {"runtime_store": service}
    assert EmptySecretProvider().get_secret("anything") is None


def test_lifecycle_plugin_has_been_removed_from_plugin_surfaces():
    assert not hasattr(plugin_base, "LifecyclePlugin")
    assert "LifecyclePlugin" not in daita.__all__
    assert not hasattr(daita, "LifecyclePlugin")
    assert "LifecyclePlugin" not in plugins.__all__
    assert not hasattr(plugins, "LifecyclePlugin")


def test_extension_contracts_are_promoted_from_top_level_package():
    exported = {
        "ConnectorPlugin": ConnectorPlugin,
        "DomainServicePlugin": DomainServicePlugin,
        "EmptySecretProvider": EmptySecretProvider,
        "ObservabilityPlugin": ObservabilityPlugin,
        "PluginContext": PluginContext,
        "PluginKind": PluginKind,
        "PluginManifest": PluginManifest,
        "RuntimeExtensionPlugin": RuntimeExtensionPlugin,
        "SecretProvider": SecretProvider,
        "ServiceRegistry": ServiceRegistry,
        "SkillPlugin": SkillPlugin,
        "WorkerProviderPlugin": WorkerProviderPlugin,
        "ExtensionRegistry": ExtensionRegistry,
        "RegistryDiagnostic": RegistryDiagnostic,
    }

    for name, value in exported.items():
        assert name in daita.__all__
        assert getattr(daita, name) is value
