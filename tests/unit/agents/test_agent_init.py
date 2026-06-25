"""
Unit tests for Agent and BaseAgent initialisation (daita/agents/agent.py,
daita/agents/base.py).

Covers:
  - Agent ID and name assignment
  - Lazy LLM initialisation
  - configure_defaults()
  - add_plugin()
  - health property structure
  - get_token_usage() structure
"""

import re

import pytest

from daita.agents.agent import Agent
from daita.config.settings import settings
from daita.core.tools import LocalTool
from daita.llm.mock import MockLLMProvider
from daita.plugins import (
    ExtensionRegistry,
    PluginKind,
    PluginManifest,
    RuntimeExtensionPlugin,
)
from daita.runtime import (
    AccessMode,
    Capability,
    ContextAudience,
    ContextBlock,
    Evidence,
    EvidenceSchema,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    ToolView,
    Worker,
)

# ===========================================================================
# Helpers
# ===========================================================================


def make_tool(name="tool_x"):
    async def h(args):
        return "ok"

    return LocalTool(name=name, description="Desc", parameters={}, handler=h)


class MinimalPlugin:
    """Non-manifest plugin with no runtime declarations."""


class InitializablePlugin:
    """Plugin that stores the agent_id passed to initialize()."""

    def __init__(self):
        self.received_agent_id = None

    def initialize(self, agent_id: str):
        self.received_agent_id = agent_id


class ManifestPlugin(RuntimeExtensionPlugin):
    def __init__(self, plugin_id="manifest_plugin"):
        self.manifest = PluginManifest(
            id=plugin_id,
            display_name="Manifest Plugin",
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
        )
        self.setup_contexts = []
        self.teardown_count = 0

    async def setup(self, context):
        self.setup_contexts.append(context)

    async def teardown(self):
        self.teardown_count += 1


class SetupAndInitializePlugin(ManifestPlugin):
    def __init__(self):
        super().__init__("setup_and_initialize_plugin")
        self.received_agent_id = None

    def initialize(self, agent_id: str):
        self.received_agent_id = agent_id


class ManifestInitializeOnlyPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="manifest_initialize_only_plugin",
        display_name="Manifest Initialize Only Plugin",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self):
        self.received_agent_id = None

    def initialize(self, agent_id: str):
        self.received_agent_id = agent_id


class StaticContextProvider:
    id = "agent.context"
    owner = "context_plugin"
    audiences = frozenset({ContextAudience.PRIMARY_MODEL})

    def __init__(self, content="Registry context.", priority=0):
        self.content = content
        self.priority = priority
        self.render_calls = []

    async def render(self, context, audience, token_budget):
        self.render_calls.append((context, audience, token_budget))
        return ContextBlock(
            id=self.id,
            owner=self.owner,
            audience=audience,
            content=self.content,
            priority=self.priority,
        )


class OperationOnlyContextProvider(StaticContextProvider):
    id = "agent.operation_context"
    audiences = frozenset({ContextAudience.OPERATION_INSPECTOR})


class ContextProviderPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="context_plugin",
        display_name="Context Plugin",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self, *providers):
        self.providers = providers

    def get_context_providers(self):
        return self.providers


class ToolViewExecutor:
    capability_ids = frozenset({"agent.lookup"})

    def __init__(
        self,
        *,
        owner="tool_view_plugin",
        executor_id="tool_view_plugin.lookup",
    ):
        self.id = executor_id
        self.owner = owner
        self.calls = []

    async def execute(self, task, operation, context):
        self.calls.append((task, operation, context))
        return [
            Evidence(
                kind="lookup.result",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                payload={"value": task.input["query"]},
            )
        ]


class ToolViewPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="tool_view_plugin",
        display_name="Tool View Plugin",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self):
        self.executor = ToolViewExecutor()

    def declare_capabilities(self):
        return (
            Capability(
                id="agent.lookup",
                owner="tool_view_plugin",
                description="Lookup a value.",
                domains=frozenset({"agent"}),
                operation_types=frozenset({"agent.lookup"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                output_evidence=frozenset({"lookup.result"}),
                executor=self.executor.id,
                side_effecting=False,
                replay_safe=True,
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def get_tool_views(self):
        return (
            ToolView(
                name="registry_lookup",
                capability_id="agent.lookup",
                description="Lookup a value through the registry executor.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        )


class DeclarationExecutor:
    id = "declaration_plugin.audit"
    owner = "declaration_plugin"
    capability_ids = frozenset({"agent.audit"})

    async def execute(self, task, operation, context):
        return [
            Evidence(
                kind="audit.result",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                payload={"ok": True},
            )
        ]


class DeclarationPolicy:
    id = "declaration_plugin.require_audit"
    owner = "declaration_plugin"

    def applies_to(self, request, operation_type):
        return operation_type == "agent.audit"

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation):
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.ALLOW,
            reason="Audit capability allowed.",
            severity=RiskLevel.LOW,
            operation_id=operation.id,
        )


class DeclarationPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="declaration_plugin",
        display_name="Declaration Plugin",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self):
        self.executor = DeclarationExecutor()
        self.policy = DeclarationPolicy()
        self.schema = EvidenceSchema(
            kind="audit.result",
            owner="declaration_plugin",
            json_schema={"type": "object"},
        )
        self.worker = Worker(
            id="declaration_plugin.reviewer",
            owner="declaration_plugin",
            role="reviewer",
            capability_ids=frozenset({"agent.audit"}),
            input_schema={"type": "object"},
            output_evidence=frozenset({"audit.result"}),
        )

    def declare_capabilities(self):
        return (
            Capability(
                id="agent.audit",
                owner="declaration_plugin",
                description="Audit an agent operation.",
                domains=frozenset({"agent"}),
                operation_types=frozenset({"agent.audit"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"audit.result"}),
                executor=self.executor.id,
                side_effecting=False,
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def declare_policies(self):
        return (self.policy,)

    def declare_evidence_schemas(self):
        return (self.schema,)

    def get_workers(self):
        return (self.worker,)


class AlternateToolViewPlugin(ToolViewPlugin):
    manifest = PluginManifest(
        id="alternate_tool_view_plugin",
        display_name="Alternate Tool View Plugin",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self):
        self.executor = ToolViewExecutor(
            owner="alternate_tool_view_plugin",
            executor_id="alternate_tool_view_plugin.lookup",
        )

    def declare_capabilities(self):
        capability = super().declare_capabilities()[0]
        return (
            Capability(
                id=capability.id,
                owner="alternate_tool_view_plugin",
                description=capability.description,
                domains=capability.domains,
                operation_types=capability.operation_types,
                access=capability.access,
                risk=capability.risk,
                input_schema=capability.input_schema,
                output_evidence=capability.output_evidence,
                executor=self.executor.id,
                side_effecting=capability.side_effecting,
                replay_safe=capability.replay_safe,
            ),
        )

    def get_tool_views(self):
        return (
            ToolView(
                name="alternate_registry_lookup",
                capability_id="agent.lookup",
                description="Lookup a value through the alternate registry executor.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        )


# ===========================================================================
# ID and naming
# ===========================================================================


class TestAgentIdentity:
    def test_agent_id_generated_from_name(self, mock_llm):
        agent = Agent(name="My Agent", llm_provider=mock_llm)
        # slug_<8-hex> pattern
        assert re.match(r"^my_agent_[0-9a-f]{8}$", agent.agent_id)

    def test_agent_id_uses_explicit_value(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, agent_id="my-custom-id")
        assert agent.agent_id == "my-custom-id"

    def test_agent_name_stored(self, mock_llm):
        agent = Agent(name="NamedAgent", llm_provider=mock_llm)
        assert agent.name == "NamedAgent"

    def test_agent_default_name(self, mock_llm):
        agent = Agent(llm_provider=mock_llm)
        assert agent.name == "Agent"

    def test_agent_ids_are_unique(self, mock_llm):
        a1 = Agent(name="MyAgent", llm_provider=mock_llm)
        a2 = Agent(name="MyAgent", llm_provider=mock_llm)
        assert a1.agent_id != a2.agent_id


# ===========================================================================
# Lazy LLM initialisation
# ===========================================================================


class TestLazyLLM:
    def test_llm_is_none_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(settings, "openai_api_key", None)
        # No API key, string provider — LLM should stay None
        agent = Agent(name="X", llm_provider="openai")
        assert agent._llm is None

    def test_llm_provider_name_stored_for_string_provider(self):
        # When a string provider name is given the provider name is stored
        # for lazy creation (no actual LLM object yet).
        agent = Agent(name="X", llm_provider="openai")
        assert agent._llm_provider_name == "openai"

    def test_default_model_is_current_cost_sensitive_openai_model(self):
        agent = Agent(name="X", llm_provider="openai")
        assert agent._llm_model == "gpt-5.4-mini"

    def test_llm_instance_is_stored_directly(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        assert agent._llm is mock_llm

    def test_llm_instance_is_accessible(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        assert agent.llm is mock_llm

    def test_llm_setter_replaces_provider(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        new_mock = MockLLMProvider(delay=0)
        agent.llm = new_mock
        assert agent.llm is new_mock

    def test_llm_setter_calls_set_agent_id(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        new_mock = MockLLMProvider(delay=0)
        agent.llm = new_mock
        assert new_mock.agent_id == agent.agent_id


# ===========================================================================
# configure_defaults()
# ===========================================================================


class TestConfigureDefaults:
    def setup_method(self):
        # Snapshot class-level defaults so we can restore after each test
        self._original_provider = Agent._default_llm_provider
        self._original_model = Agent._default_model

    def teardown_method(self):
        Agent._default_llm_provider = self._original_provider
        Agent._default_model = self._original_model

    def test_set_default_llm_provider(self):
        Agent.configure_defaults(llm_provider="anthropic")
        assert Agent._default_llm_provider == "anthropic"

    def test_set_default_model(self):
        Agent.configure_defaults(model="gpt-3.5-turbo")
        assert Agent._default_model == "gpt-3.5-turbo"

    def test_unknown_key_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown keys"):
            Agent.configure_defaults(temperature=0.5)

    def test_multiple_keys_set_together(self):
        Agent.configure_defaults(llm_provider="grok", model="grok-2")
        assert Agent._default_llm_provider == "grok"
        assert Agent._default_model == "grok-2"


# ===========================================================================
# Plugin and tool management
# ===========================================================================


class TestPluginAndToolManagement:
    def test_add_plugin_registers_manifest_plugins_with_extension_registry(
        self, mock_llm
    ):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = ManifestPlugin()

        agent.add_plugin(plugin)

        assert agent.attached_plugin_ids == ["manifest_plugin"]
        assert agent.get_attached_plugin("manifest_plugin") is plugin

    def test_attached_plugins_uses_registry_without_tool_sources(self, mock_llm):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        assert agent.attached_plugins == [plugin]
        assert agent.attached_plugin_ids == ["manifest_plugin"]

    def test_attached_plugins_deduplicates_registry_and_tool_sources(self, mock_llm):
        manifest_plugin = ManifestPlugin()
        legacy_plugin = MinimalPlugin()

        agent = Agent(name="X", llm_provider=mock_llm)
        agent.add_plugin(manifest_plugin)
        agent.add_plugin(legacy_plugin)

        assert agent.attached_plugins == [manifest_plugin, legacy_plugin]
        assert agent.attached_plugin_ids == ["manifest_plugin", "MinimalPlugin"]

    def test_get_attached_plugin_uses_registry_without_tool_sources(self, mock_llm):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        assert agent.get_attached_plugin("manifest_plugin") is plugin
        assert agent.get_attached_plugin(ManifestPlugin) is plugin
        assert agent.get_attached_plugin("missing") is None

    def test_get_attached_plugin_falls_back_to_legacy_class_name(self, mock_llm):
        plugin = MinimalPlugin()
        agent = Agent(name="X", llm_provider=mock_llm)

        agent.add_plugin(plugin)

        assert agent.get_attached_plugin("MinimalPlugin") is plugin
        assert agent.get_attached_plugin(MinimalPlugin) is plugin

    def test_capabilities_use_registry_without_tool_sources(self, mock_llm):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        capabilities = agent.capabilities

        assert [capability.id for capability in capabilities] == ["agent.lookup"]
        assert capabilities[0].owner == "tool_view_plugin"

    def test_find_capabilities_delegates_to_registry_without_tool_sources(
        self, mock_llm
    ):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        matches = agent.find_capabilities(
            domain="agent",
            operation_type="agent.lookup",
        )

        assert [capability.id for capability in matches] == ["agent.lookup"]
        assert matches[0].owner == "tool_view_plugin"
        assert agent.find_capabilities(domain="missing") == []

    def test_registry_metadata_surfaces_without_tool_sources(self, mock_llm):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        assert agent.plugin_manifests == [plugin.manifest]
        assert [view.name for view in agent.tool_views] == ["registry_lookup"]
        assert [
            (diagnostic.declaration_type, diagnostic.declaration_id)
            for diagnostic in agent.extension_diagnostics
        ] == [
            ("capability", "agent.lookup"),
            ("executor", "tool_view_plugin.lookup"),
            ("tool_view", "registry_lookup"),
        ]

    def test_registry_lookup_surfaces_without_tool_sources(self, mock_llm):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        assert agent.get_plugin_manifest("tool_view_plugin") == plugin.manifest
        assert agent.get_plugin_manifest("missing") is None
        assert agent.get_capability("agent.lookup").owner == "tool_view_plugin"
        assert agent.get_executor("tool_view_plugin.lookup") is plugin.executor
        assert agent.get_tool_view("registry_lookup").capability_id == "agent.lookup"
        assert agent.get_tool_view("missing") is None
        assert agent.get_tool_view_owner("registry_lookup") == "tool_view_plugin"

    def test_get_capability_requires_owner_for_ambiguous_registry_ids(self, mock_llm):
        first = ToolViewPlugin()
        second = AlternateToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(first)
        registry.register(second)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        with pytest.raises(ValueError, match="multiple owners"):
            agent.get_capability("agent.lookup")

        assert (
            agent.get_capability(
                "agent.lookup",
                owner="alternate_tool_view_plugin",
            ).owner
            == "alternate_tool_view_plugin"
        )

    def test_declaration_surfaces_use_registry_without_tool_sources(self, mock_llm):
        plugin = DeclarationPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        assert agent.executors == [plugin.executor]
        assert agent.policies == [plugin.policy]
        assert agent.evidence_schemas == [plugin.schema]
        assert agent.workers == [plugin.worker]

    def test_declaration_lookup_surfaces_use_registry_without_tool_sources(
        self, mock_llm
    ):
        plugin = DeclarationPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        assert agent.get_policy("declaration_plugin.require_audit") is plugin.policy
        assert agent.get_evidence_schema("audit.result") is plugin.schema
        assert agent.get_worker("declaration_plugin.reviewer") is plugin.worker
        assert (
            agent.get_policy(
                "declaration_plugin.require_audit",
                owner="declaration_plugin",
            )
            is plugin.policy
        )

    async def test_execute_capability_uses_registry_without_tool_sources(
        self, mock_llm
    ):
        plugin = DeclarationPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        result = await agent.execute_capability("agent.audit", {"subject": "alpha"})

        assert result["capability_id"] == "agent.audit"
        assert result["evidence"][0]["kind"] == "audit.result"
        assert result["evidence"][0]["payload"] == {"ok": True}
        operations = await agent.runtime_store.list_operations()
        tasks = await agent.runtime_store.list_tasks(operations[0].id)
        assert tasks[0].capability_id == "agent.audit"
        assert tasks[0].status.value == "succeeded"

    async def test_execute_capability_disambiguates_owner_without_tool_sources(
        self, mock_llm
    ):
        first = ToolViewPlugin()
        second = AlternateToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(first)
        registry.register(second)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        result = await agent.execute_capability(
            "agent.lookup",
            {"query": "beta"},
            owner="alternate_tool_view_plugin",
        )

        assert result["capability_id"] == "agent.lookup"
        assert result["evidence"][0]["owner"] == "alternate_tool_view_plugin"
        assert result["evidence"][0]["payload"] == {"value": "beta"}
        assert len(first.executor.calls) == 0
        assert len(second.executor.calls) == 1

    def test_constructor_plugins_register_manifest_plugins_with_extension_registry(
        self, mock_llm
    ):
        plugin = ManifestPlugin()

        agent = Agent(name="X", llm_provider=mock_llm, plugins=[plugin])

        assert agent.attached_plugin_ids == ["manifest_plugin"]
        assert agent.get_attached_plugin("manifest_plugin") is plugin
        assert "plugins" not in agent._llm_kwargs

    async def test_constructor_plugins_setup_with_plugin_context(self, mock_llm):
        plugin = ManifestPlugin()
        agent = Agent(name="X", llm_provider=mock_llm, plugins=[plugin])

        await agent._setup_tools()

        assert len(plugin.setup_contexts) == 1
        assert plugin.setup_contexts[0].runtime_kind == "agent"
        assert plugin.setup_contexts[0].agent_id == agent.agent_id

    async def test_constructor_plugins_register_tool_views(self, mock_llm):
        plugin = ToolViewPlugin()
        agent = Agent(name="X", llm_provider=mock_llm, plugins=[plugin])

        result = await agent.call_tool("registry_lookup", {"query": "alpha"})

        assert "registry_lookup" in agent.tool_names
        assert result["capability_id"] == "agent.lookup"
        assert result["evidence"][0]["payload"] == {"value": "alpha"}
        assert len(plugin.executor.calls) == 1

    async def test_constructor_extension_registry_registers_tool_views_without_tool_sources(
        self, mock_llm
    ):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)
        result = await agent.call_tool("registry_lookup", {"query": "alpha"})

        assert agent.extension_registry is registry
        assert "extension_registry" not in agent._llm_kwargs
        assert "registry_lookup" in agent.tool_names
        assert result["capability_id"] == "agent.lookup"
        assert result["evidence"][0]["payload"] == {"value": "alpha"}
        assert len(plugin.executor.calls) == 1

    async def test_constructor_extension_registry_setup_uses_plugin_context(
        self, mock_llm
    ):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)
        await agent._setup_tools()
        assert len(plugin.setup_contexts) == 1
        assert plugin.setup_contexts[0].runtime_kind == "agent"
        assert plugin.setup_contexts[0].agent_id == agent.agent_id

    async def test_setup_extensions_uses_plugin_context_without_tool_sources(
        self, mock_llm
    ):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        assert agent.extension_setup_plugin_ids == []
        assert agent.pending_extension_setup_plugin_ids == ["manifest_plugin"]
        assert agent.extensions_setup_complete is False

        await agent.setup_extensions()

        assert agent.extension_setup_plugin_ids == ["manifest_plugin"]
        assert agent.pending_extension_setup_plugin_ids == []
        assert agent.extensions_setup_complete is True
        assert len(plugin.setup_contexts) == 1
        assert plugin.setup_contexts[0].runtime_kind == "agent"
        assert plugin.setup_contexts[0].agent_id == agent.agent_id

    async def test_setup_extensions_is_idempotent_without_tool_sources(self, mock_llm):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        await agent.setup_extensions()
        await agent.setup_extensions()

        assert agent.extension_setup_plugin_ids == ["manifest_plugin"]
        assert agent.pending_extension_setup_plugin_ids == []
        assert len(plugin.setup_contexts) == 1

    async def test_teardown_extensions_clears_setup_state_without_tool_sources(
        self, mock_llm
    ):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        await agent.setup_extensions()
        await agent.teardown_extensions()

        assert plugin.teardown_count == 1
        assert agent.extension_setup_plugin_ids == []
        assert agent.pending_extension_setup_plugin_ids == ["manifest_plugin"]
        assert agent.extensions_setup_complete is False

    async def test_teardown_extensions_runs_before_setup_without_tool_sources(
        self, mock_llm
    ):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        await agent.teardown_extensions()

        assert plugin.teardown_count == 1
        assert agent.extension_setup_plugin_ids == []
        assert agent.pending_extension_setup_plugin_ids == ["manifest_plugin"]

    async def test_manifest_plugin_prefers_setup_over_initialize(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = SetupAndInitializePlugin()

        agent.add_plugin(plugin)
        await agent._setup_tools()

        assert plugin.received_agent_id is None
        assert len(plugin.setup_contexts) == 1
        assert plugin.setup_contexts[0].agent_id == agent.agent_id

    def test_manifest_plugin_without_setup_does_not_use_initialize(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = ManifestInitializeOnlyPlugin()

        agent.add_plugin(plugin)

        assert plugin.received_agent_id is None

    async def test_setup_tools_sets_up_manifest_plugins_with_plugin_context(
        self, mock_llm
    ):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = ManifestPlugin()
        agent.add_plugin(plugin)

        await agent._setup_tools()
        await agent._setup_tools()

        assert len(plugin.setup_contexts) == 1
        context = plugin.setup_contexts[0]
        assert context.runtime_id == agent.agent_id
        assert context.runtime_kind == "agent"
        assert context.agent_id == agent.agent_id

    async def test_setup_tools_sets_up_manifest_plugins_added_after_first_setup(
        self, mock_llm
    ):
        agent = Agent(name="X", llm_provider=mock_llm)
        first = ManifestPlugin("first_plugin")
        second = ManifestPlugin("second_plugin")

        agent.add_plugin(first)
        await agent._setup_tools()
        agent.add_plugin(second)
        await agent._setup_tools()

        assert len(first.setup_contexts) == 1
        assert len(second.setup_contexts) == 1

    async def test_stop_tears_down_setup_manifest_plugins(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = ManifestPlugin()
        agent.add_plugin(plugin)

        await agent._setup_tools()
        await agent.stop()

        assert plugin.teardown_count == 1
        assert agent._extension_setup_plugin_ids == set()

    async def test_stop_tears_down_manifest_plugins_even_before_setup(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = ManifestPlugin()
        agent.add_plugin(plugin)

        await agent.stop()

        assert plugin.teardown_count == 1
        assert agent._extension_setup_plugin_ids == set()

    async def test_initial_conversation_includes_registry_context_provider_blocks(
        self, mock_llm
    ):
        provider = StaticContextProvider("Context A.", priority=10)
        agent = Agent(name="X", llm_provider=mock_llm, prompt="Base prompt.")
        agent.add_plugin(ContextProviderPlugin(provider))

        conversation = await agent.runtime._build_initial_conversation("Hi")
        system_msg = conversation[0]["content"]

        assert "Base prompt." in system_msg
        assert "## Runtime Context" in system_msg
        assert "### agent.context\nContext A." in system_msg
        assert provider.render_calls == [
            (
                {
                    "prompt": "Hi",
                    "runtime_id": agent.agent_id,
                    "agent_id": agent.agent_id,
                },
                ContextAudience.PRIMARY_MODEL,
                2000,
            )
        ]

    async def test_context_providers_use_registry_without_tool_sources(self, mock_llm):
        provider = StaticContextProvider("Context A.")
        plugin = ContextProviderPlugin(provider)
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        blocks = await agent.render_context_blocks("Hi")

        assert agent.context_providers == [provider]
        assert agent.get_context_provider("agent.context") is provider
        assert (
            agent.get_context_provider("agent.context", owner="context_plugin")
            is provider
        )
        assert [block.content for block in blocks] == ["Context A."]
        assert provider.render_calls == [
            (
                {
                    "prompt": "Hi",
                    "runtime_id": agent.agent_id,
                    "agent_id": agent.agent_id,
                },
                ContextAudience.PRIMARY_MODEL,
                2000,
            )
        ]

    async def test_render_context_blocks_supports_non_primary_audiences(self, mock_llm):
        provider = OperationOnlyContextProvider("Inspector context.")
        plugin = ContextProviderPlugin(provider)
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        blocks = await agent.render_context_blocks(
            "Inspect",
            audience=ContextAudience.OPERATION_INSPECTOR,
            token_budget=123,
        )

        assert [block.content for block in blocks] == ["Inspector context."]
        assert provider.render_calls == [
            (
                {
                    "prompt": "Inspect",
                    "runtime_id": agent.agent_id,
                    "agent_id": agent.agent_id,
                },
                ContextAudience.OPERATION_INSPECTOR,
                123,
            )
        ]

    async def test_initial_conversation_skips_non_primary_registry_context(
        self, mock_llm
    ):
        provider = OperationOnlyContextProvider("Inspector context.")
        agent = Agent(name="X", llm_provider=mock_llm)
        agent.add_plugin(ContextProviderPlugin(provider))

        conversation = await agent.runtime._build_initial_conversation("Hi")

        assert provider.render_calls == []
        assert conversation == [{"role": "user", "content": "Hi"}]

    async def test_tool_views_register_as_agent_tools_and_execute_via_registry(
        self, mock_llm
    ):
        plugin = ToolViewPlugin()
        agent = Agent(name="X", llm_provider=mock_llm)

        agent.add_plugin(plugin)
        result = await agent.call_tool("registry_lookup", {"query": "alpha"})

        assert "registry_lookup" in agent.tool_names
        tool = agent.available_tools[0]
        assert tool.capability_ids == ("agent.lookup",)
        assert tool.replay_safe is True
        assert tool.side_effecting is False
        assert result["capability_id"] == "agent.lookup"
        assert result["evidence"][0]["payload"] == {"value": "alpha"}
        task, operation, context = plugin.executor.calls[0]
        assert task.capability_id == "agent.lookup"
        assert operation.operation_type == "agent.lookup"
        assert context["agent_id"] == agent.agent_id

    async def test_registry_tool_views_do_not_require_tool_sources(self, mock_llm):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)
        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        assert "registry_lookup" in agent.tool_names
        assert agent.available_tools[0].name == "registry_lookup"

        result = await agent.call_tool("registry_lookup", {"query": "alpha"})

        assert result["capability_id"] == "agent.lookup"
        assert result["evidence"][0]["payload"] == {"value": "alpha"}
        assert len(plugin.executor.calls) == 1

    def test_tools_property_projects_registry_tool_views_without_tool_sources(
        self, mock_llm
    ):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)
        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        tools = agent.tools

        assert [tool.name for tool in tools] == ["registry_lookup"]
        assert tools[0].capability_ids == ("agent.lookup",)

    def test_registry_tool_views_are_resolved_without_tool_sources(self, mock_llm):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)
        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        resolved = agent._resolve_tools(["registry_lookup"])

        assert resolved[0].name == "registry_lookup"
        assert resolved[0].capability_ids == ("agent.lookup",)

    async def test_registry_tool_views_resolve_capability_by_contributing_owner(
        self, mock_llm
    ):
        first = ToolViewPlugin()
        second = AlternateToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(first)
        registry.register(second)
        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        first_result = await agent.call_tool("registry_lookup", {"query": "one"})
        second_result = await agent.call_tool(
            "alternate_registry_lookup",
            {"query": "two"},
        )

        assert first_result["evidence"][0]["owner"] == "tool_view_plugin"
        assert second_result["evidence"][0]["owner"] == "alternate_tool_view_plugin"
        assert len(first.executor.calls) == 1
        assert len(second.executor.calls) == 1
        assert first.executor.calls[0][0].executor_id == "tool_view_plugin.lookup"
        assert (
            second.executor.calls[0][0].executor_id
            == "alternate_tool_view_plugin.lookup"
        )

    async def test_manifest_tool_views_execute_through_runtime_registry(self, mock_llm):
        plugin = ToolViewPlugin()
        agent = Agent(name="X", llm_provider=mock_llm)

        agent.add_plugin(plugin)
        result = await agent.call_tool("registry_lookup", {"query": "alpha"})

        tool = agent.available_tools[0]
        assert tool.capability_ids == ("agent.lookup",)
        assert result["capability_id"] == "agent.lookup"
        assert result["evidence"][0]["payload"] == {"value": "alpha"}
        assert len(plugin.executor.calls) == 1

    def test_non_manifest_plugin_initialize_is_not_runtime_setup(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = InitializablePlugin()
        agent.add_plugin(plugin)
        assert plugin.received_agent_id is None

    def test_add_plugin_without_initialize_no_error(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = MinimalPlugin()
        agent.add_plugin(plugin)  # Should not raise

    async def test_tool_increases_count(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[make_tool("my_tool")])
        await agent._setup_tools()
        assert len(agent.tools) == 1

    async def test_multiple_tools_added(self, mock_llm):
        agent = Agent(
            name="X", llm_provider=mock_llm, tools=[make_tool("t1"), make_tool("t2")]
        )
        await agent._setup_tools()
        assert len(agent.tools) == 2

    async def test_tool_names_reflects_registry(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[make_tool("alpha")])
        await agent._setup_tools()
        assert "alpha" in agent.tool_names

    async def test_available_tools_returns_list(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[make_tool("beta")])
        await agent._setup_tools()
        tools = agent.available_tools
        assert isinstance(tools, list)
        assert tools[0].name == "beta"


# ===========================================================================
# health property
# ===========================================================================


class TestHealthProperty:
    def test_health_has_required_keys(self, basic_agent):
        h = basic_agent.health
        for key in (
            "id",
            "name",
            "type",
            "running",
            "metrics",
            "tools",
            "extensions",
            "llm",
        ):
            assert key in h, f"Missing key: {key}"

    async def test_health_tools_count_matches_registry(self, mock_llm, simple_tool):
        agent = Agent(name="TestAgent", llm_provider=mock_llm, tools=[simple_tool])
        await agent._setup_tools()
        assert agent.health["tools"]["count"] == 1

    def test_health_reports_extension_registry_diagnostics(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = ToolViewPlugin()
        agent.add_plugin(plugin)

        extensions = agent.health["extensions"]

        assert extensions["plugin_ids"] == ["tool_view_plugin"]
        assert extensions["capability_count"] == 1
        assert extensions["tool_view_count"] == 1
        assert extensions["context_provider_count"] == 0
        assert extensions["executor_count"] == 1
        assert extensions["policy_count"] == 0
        assert extensions["evidence_schema_count"] == 0
        assert extensions["worker_count"] == 0
        assert extensions["diagnostic_count"] >= 3
        assert extensions["manifest_ids"] == ["tool_view_plugin"]
        assert extensions["capability_ids"] == ["agent.lookup"]
        assert extensions["tool_view_names"] == ["registry_lookup"]
        assert extensions["context_provider_ids"] == []
        assert extensions["executor_ids"] == ["tool_view_plugin.lookup"]
        assert extensions["policy_ids"] == []
        assert extensions["evidence_schema_kinds"] == []
        assert extensions["worker_ids"] == []
        assert extensions["diagnostic_ids"] == [
            "agent.lookup",
            "tool_view_plugin.lookup",
            "registry_lookup",
        ]

    def test_health_reports_registry_tool_names_before_setup(self, mock_llm):
        plugin = ToolViewPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)
        tools = agent.health["tools"]

        assert tools["setup"] is False
        assert tools["count"] == 1
        assert tools["names"] == ["registry_lookup"]

    def test_health_reports_constructor_plugin_tool_names_before_setup(self, mock_llm):
        plugin = ToolViewPlugin()

        agent = Agent(name="X", llm_provider=mock_llm, plugins=[plugin])
        tools = agent.health["tools"]

        assert tools["setup"] is False
        assert tools["count"] == 1
        assert tools["names"] == ["registry_lookup"]

    def test_health_reports_pending_extension_setup_without_tool_sources(
        self, mock_llm
    ):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)

        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)
        extensions = agent.health["extensions"]

        assert extensions["plugin_ids"] == ["manifest_plugin"]
        assert extensions["setup_plugin_ids"] == []
        assert extensions["pending_setup_plugin_ids"] == ["manifest_plugin"]
        assert extensions["setup_complete"] is False

    async def test_health_reports_completed_extension_setup_without_tool_sources(
        self, mock_llm
    ):
        plugin = ManifestPlugin()
        registry = ExtensionRegistry()
        registry.register(plugin)
        agent = Agent(name="X", llm_provider=mock_llm, extension_registry=registry)

        await agent._setup_tools()
        extensions = agent.health["extensions"]

        assert extensions["setup_plugin_ids"] == ["manifest_plugin"]
        assert extensions["pending_setup_plugin_ids"] == []
        assert extensions["setup_complete"] is True
        assert len(plugin.setup_contexts) == 1

    def test_health_llm_available_with_mock(self, basic_agent):
        assert basic_agent.health["llm"]["available"] is True

    def test_health_id_matches_agent_id(self, basic_agent):
        assert basic_agent.health["id"] == basic_agent.agent_id


# ===========================================================================
# get_token_usage()
# ===========================================================================


class TestGetTokenUsage:
    def test_returns_dict_with_expected_keys(self, basic_agent):
        usage = basic_agent.get_token_usage()
        for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
            assert key in usage

    def test_no_llm_returns_zeros(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        # Before any LLM calls, all values should be zero
        usage = agent.get_token_usage()
        assert usage["total_tokens"] == 0
