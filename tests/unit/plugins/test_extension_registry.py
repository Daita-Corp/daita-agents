from dataclasses import dataclass

import pytest

from daita.plugins import (
    DomainServicePlugin,
    ExtensionRegistry,
    PluginContext,
    PluginKind,
    PluginManifest,
    ServiceRegistry,
)
from daita.runtime import (
    AccessMode,
    Capability,
    ContextAudience,
    ContextBlock,
    Evidence,
    EvidenceSchema,
    Operation,
    RiskLevel,
    Task,
    ToolView,
    Worker,
)


@dataclass(frozen=True)
class SimpleExecutor:
    id: str
    capability_ids: frozenset[str]

    async def execute(self, task: Task, operation: Operation, context):
        return [
            Evidence(
                kind="query.result",
                owner="sqlite",
                operation_id=operation.id,
                task_id=task.id,
                payload={"rows": []},
            )
        ]


class NoopPlugin:
    manifest = PluginManifest(
        id="noop",
        display_name="No-op",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )


class CatalogDomainService(DomainServicePlugin):
    manifest = PluginManifest(
        id="catalog",
        display_name="Catalog",
        version="2.0.0",
        kind=PluginKind.DOMAIN_SERVICE,
        domains=frozenset({"db"}),
    )

    def declare_capabilities(self):
        return [
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
            )
        ]

    def get_executors(self):
        return [
            SimpleExecutor(
                id="catalog.schema.search",
                capability_ids=frozenset({"catalog.schema.search"}),
            )
        ]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="schema.search_result",
                owner="catalog",
                json_schema={"type": "object"},
            )
        ]


class MisclassifiedDomainService(DomainServicePlugin):
    manifest = PluginManifest(
        id="misclassified",
        display_name="Misclassified",
        version="1.0.0",
        kind=PluginKind.CONNECTOR,
    )


class SqliteExtension:
    manifest = PluginManifest(
        id="sqlite",
        display_name="SQLite",
        version="1.0.0",
        kind=PluginKind.CONNECTOR,
        domains=frozenset({"db"}),
        provides=frozenset({"sql"}),
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="db.sql.execute_read",
                owner="sqlite",
                description="Execute read-only SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query"}),
                access=AccessMode.READ,
                risk=RiskLevel.MEDIUM,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="sqlite.sql.execute_read",
                model_visible=True,
                side_effecting=False,
            )
        ]

    def get_executors(self):
        return [
            SimpleExecutor(
                id="sqlite.sql.execute_read",
                capability_ids=frozenset({"db.sql.execute_read"}),
            )
        ]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="query.result",
                owner="sqlite",
                json_schema={"type": "object"},
            )
        ]

    def get_tool_views(self):
        return [
            ToolView(
                name="sqlite_query",
                capability_id="db.sql.execute_read",
                description="Run a SQLite read query.",
                parameters={"type": "object"},
            )
        ]


class PostgresExtension:
    manifest = PluginManifest(
        id="postgresql",
        display_name="PostgreSQL",
        version="1.0.0",
        kind=PluginKind.CONNECTOR,
        domains=frozenset({"db"}),
        provides=frozenset({"sql"}),
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="db.sql.execute_read",
                owner="postgresql",
                description="Execute read-only SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query"}),
                access=AccessMode.READ,
                risk=RiskLevel.MEDIUM,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="postgresql.sql.execute_read",
                runtime_only=True,
                side_effecting=False,
            )
        ]

    def get_executors(self):
        return [
            SimpleExecutor(
                id="postgresql.sql.execute_read",
                capability_ids=frozenset({"db.sql.execute_read"}),
            )
        ]


class MissingExecutorPlugin:
    manifest = PluginManifest(
        id="broken",
        display_name="Broken",
        version="1.0.0",
        kind=PluginKind.CONNECTOR,
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="db.sql.execute_read",
                owner="broken",
                description="References a missing executor.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="broken.sql.execute_read",
            )
        ]


class DuplicateCapabilityPlugin:
    manifest = PluginManifest(
        id="dupe",
        display_name="Dupe",
        version="1.0.0",
        kind=PluginKind.CONNECTOR,
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="db.sql.execute_read",
                owner="dupe",
                description="First duplicate.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="dupe.sql.execute_read",
            ),
            Capability(
                id="db.sql.execute_read",
                owner="dupe",
                description="Second duplicate.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="dupe.sql.execute_read",
            ),
        ]

    def get_executors(self):
        return [
            SimpleExecutor(
                id="dupe.sql.execute_read",
                capability_ids=frozenset({"db.sql.execute_read"}),
            )
        ]


class DuplicateToolViewPlugin(SqliteExtension):
    def get_tool_views(self):
        return [
            ToolView(
                name="sqlite_query",
                capability_id="db.sql.execute_read",
                description="Run a SQLite read query.",
                parameters={"type": "object"},
            ),
            ToolView(
                name="sqlite_query",
                capability_id="db.sql.execute_read",
                description="Run a SQLite read query again.",
                parameters={"type": "object"},
            ),
        ]


class MissingToolViewCapabilityPlugin(NoopPlugin):
    def get_tool_views(self):
        return [
            ToolView(
                name="missing_capability_tool",
                capability_id="missing.capability",
                description="References a missing capability.",
                parameters={"type": "object"},
            )
        ]


class CrossOwnerToolViewPlugin(NoopPlugin):
    manifest = PluginManifest(
        id="cross_owner",
        display_name="Cross Owner Tool View",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def get_tool_views(self):
        return [
            ToolView(
                name="cross_owner_query",
                capability_id="db.sql.execute_read",
                description="Attempts to expose another plugin's capability.",
                parameters={"type": "object"},
            )
        ]


class HiddenToolViewCapabilityPlugin(NoopPlugin):
    manifest = PluginManifest(
        id="hidden_tool_view",
        display_name="Hidden Tool View",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="hidden.lookup",
                owner="hidden_tool_view",
                description="Hidden lookup.",
                domains=frozenset({"chat"}),
                operation_types=frozenset({"chat.tool_call"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"hidden.result"}),
                executor="hidden_tool_view.lookup",
                runtime_only=True,
            )
        ]

    def get_executors(self):
        return [
            SimpleExecutor(
                id="hidden_tool_view.lookup",
                capability_ids=frozenset({"hidden.lookup"}),
            )
        ]

    def get_tool_views(self):
        return [
            ToolView(
                name="hidden_lookup",
                capability_id="hidden.lookup",
                description="Should not be model visible.",
                parameters={"type": "object"},
            )
        ]


@dataclass(frozen=True)
class CrossOwnerExecutor:
    id: str
    owner: str
    capability_ids: frozenset[str]

    async def execute(self, task: Task, operation: Operation, context):
        return []


class CrossOwnerExecutorPlugin(NoopPlugin):
    manifest = PluginManifest(
        id="cross_owner_executor",
        display_name="Cross Owner Executor",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="cross.lookup",
                owner="cross_owner_executor",
                description="Lookup.",
                domains=frozenset({"chat"}),
                operation_types=frozenset({"chat.tool_call"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"cross.result"}),
                executor="other.lookup",
            )
        ]

    def get_executors(self):
        return [
            CrossOwnerExecutor(
                id="other.lookup",
                owner="other",
                capability_ids=frozenset({"cross.lookup"}),
            )
        ]


class InvalidCapabilityContributionPlugin(NoopPlugin):
    def declare_capabilities(self):
        return [{"id": "not.a.capability"}]


@dataclass(frozen=True)
class SimplePolicy:
    id: str
    owner: str

    def applies_to(self, request, operation_type: str) -> bool:
        return True

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation: Operation):
        return None


@dataclass(frozen=True)
class SimpleContextProvider:
    id: str
    owner: str
    audiences: frozenset[ContextAudience]

    async def render(self, context, audience: ContextAudience, token_budget: int):
        return ContextBlock(
            id="governance.context",
            owner=self.owner,
            audience=audience,
            content="policy context",
        )


class FullSurfacePlugin:
    manifest = PluginManifest(
        id="governance",
        display_name="Governance",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="governance.pii_check",
                owner="governance",
                description="Check for PII policy concerns.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"governance.decision"}),
                executor="governance.pii_check",
                runtime_only=True,
                side_effecting=False,
            )
        ]

    def get_executors(self):
        return [
            SimpleExecutor(
                id="governance.pii_check",
                capability_ids=frozenset({"governance.pii_check"}),
            )
        ]

    def declare_policies(self):
        return [SimplePolicy(id="governance.require_masking", owner="governance")]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="governance.decision",
                owner="governance",
                json_schema={"type": "object"},
            )
        ]

    def get_context_providers(self):
        return [
            SimpleContextProvider(
                id="governance.context",
                owner="governance",
                audiences=frozenset({ContextAudience.PRIMARY_MODEL}),
            )
        ]

    def get_workers(self):
        return [
            Worker(
                id="governance.reviewer",
                owner="governance",
                role="policy_reviewer",
                capability_ids=frozenset({"governance.pii_check"}),
                input_schema={"type": "object"},
                output_evidence=frozenset({"governance.decision"}),
            )
        ]


class SetupTeardownPlugin(NoopPlugin):
    def __init__(self):
        self.setup_context = None
        self.teardown_called = False

    async def setup(self, context):
        self.setup_context = context

    async def teardown(self):
        self.teardown_called = True


class NamedSetupTeardownPlugin:
    def __init__(self, plugin_id):
        self.manifest = PluginManifest(
            id=plugin_id,
            display_name=plugin_id,
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
        )
        self.setup_context = None
        self.teardown_called = False

    async def setup(self, context):
        self.setup_context = context

    async def teardown(self):
        self.teardown_called = True


class FailingSetupPlugin:
    manifest = PluginManifest(
        id="failing_setup",
        display_name="Failing setup",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self):
        self.teardown_called = False

    async def setup(self, context):
        raise RuntimeError("setup exploded")

    async def teardown(self):
        self.teardown_called = True


class SetupRegistersPlugin:
    manifest = PluginManifest(
        id="setup_registers",
        display_name="Setup Registers",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self, child):
        self.child = child

    async def setup(self, context):
        context.services.require("extension_registry").register(self.child)


def test_manifest_is_stable_and_serializable():
    manifest = PluginManifest(
        id="data_quality",
        display_name="Data Quality",
        version="2.0.0",
        kind=PluginKind.DOMAIN_SERVICE,
        domains=frozenset({"db"}),
        provides=frozenset({"quality"}),
        optional_dependencies=frozenset({"scipy"}),
    )

    assert PluginManifest.from_dict(manifest.to_dict()) == manifest

    with pytest.raises(ValueError):
        PluginManifest(
            id="DataQuality",
            display_name="Data Quality",
            version="2.0.0",
            kind=PluginKind.DOMAIN_SERVICE,
        )


def test_registry_accepts_noop_plugin():
    registry = ExtensionRegistry()

    registry.register(NoopPlugin())

    assert registry.plugin_ids == ("noop",)
    assert registry.capabilities == ()
    assert registry.diagnostics == ()


def test_registry_accepts_extension_first_base_subclasses():
    registry = ExtensionRegistry()

    registry.register(CatalogDomainService())

    assert registry.plugin_ids == ("catalog",)
    assert registry.get_capability("catalog.schema.search").owner == "catalog"
    assert registry.evidence_schemas[0].kind == "schema.search_result"


def test_registry_gets_owned_declarations():
    registry = ExtensionRegistry()

    registry.register(FullSurfacePlugin())

    assert registry.get_policy("governance.require_masking").owner == "governance"
    assert registry.get_evidence_schema("governance.decision").owner == "governance"
    assert registry.get_context_provider("governance.context").owner == "governance"
    assert registry.get_worker("governance.reviewer").owner == "governance"
    assert (
        registry.get_policy(
            "governance.require_masking",
            owner="governance",
        ).id
        == "governance.require_masking"
    )

    with pytest.raises(KeyError):
        registry.get_worker("missing.worker")


def test_registry_rejects_manifest_kind_that_conflicts_with_base_contract():
    registry = ExtensionRegistry()

    with pytest.raises(ValueError, match="manifest kind"):
        registry.register(MisclassifiedDomainService())

    assert registry.plugin_ids == ()


def test_registry_rejects_missing_and_invalid_manifests():
    registry = ExtensionRegistry()

    with pytest.raises(ValueError, match="manifest"):
        registry.register(object())

    class InvalidManifestPlugin:
        manifest = "not a manifest"

    with pytest.raises(TypeError, match="PluginManifest"):
        registry.register(InvalidManifestPlugin())


def test_duplicate_plugin_ids_fail():
    registry = ExtensionRegistry()
    registry.register(NoopPlugin())

    with pytest.raises(ValueError, match="duplicate plugin id"):
        registry.register(NoopPlugin())


def test_missing_executor_references_fail():
    registry = ExtensionRegistry()

    with pytest.raises(ValueError, match="missing executor"):
        registry.register(MissingExecutorPlugin())


def test_provider_neutral_capability_ids_can_have_distinct_owners():
    registry = ExtensionRegistry()

    registry.register(SqliteExtension())
    registry.register(PostgresExtension())

    matches = registry.find_capabilities(domain="db", operation_type="data.query")
    assert [capability.owner for capability in matches] == ["sqlite", "postgresql"]
    assert (
        registry.get_capability("db.sql.execute_read", owner="postgresql").executor
        == "postgresql.sql.execute_read"
    )
    with pytest.raises(ValueError, match="multiple owners"):
        registry.get_capability("db.sql.execute_read")


def test_duplicate_capabilities_from_same_owner_fail_atomically():
    registry = ExtensionRegistry()

    with pytest.raises(ValueError, match="duplicate capability"):
        registry.register(DuplicateCapabilityPlugin())

    assert registry.plugin_ids == ()
    assert registry.capabilities == ()
    assert registry.executors == ()


def test_duplicate_tool_views_fail_atomically():
    registry = ExtensionRegistry()

    with pytest.raises(ValueError, match="duplicate tool view name"):
        registry.register(DuplicateToolViewPlugin())

    assert registry.plugin_ids == ()
    assert registry.tool_views == ()


def test_tool_views_must_reference_registered_capabilities():
    registry = ExtensionRegistry()

    with pytest.raises(ValueError, match="missing capability"):
        registry.register(MissingToolViewCapabilityPlugin())


def test_tool_views_cannot_expose_runtime_only_capabilities():
    registry = ExtensionRegistry()

    with pytest.raises(ValueError, match="cannot expose hidden capability"):
        registry.register(HiddenToolViewCapabilityPlugin())


def test_executor_owner_must_match_manifest_id():
    registry = ExtensionRegistry()

    with pytest.raises(ValueError, match="executor .* owner"):
        registry.register(CrossOwnerExecutorPlugin())


def test_tool_view_owner_tracks_contributing_plugin():
    registry = ExtensionRegistry()

    registry.register(SqliteExtension())

    assert registry.get_tool_view_owner("sqlite_query") == "sqlite"


def test_tool_views_cannot_project_another_plugin_capability_owner():
    registry = ExtensionRegistry()
    registry.register(SqliteExtension())

    with pytest.raises(ValueError, match="missing capability"):
        registry.register(CrossOwnerToolViewPlugin())


def test_capability_contributions_must_be_capability_instances():
    registry = ExtensionRegistry()

    with pytest.raises(TypeError, match="Capability"):
        registry.register(InvalidCapabilityContributionPlugin())


def test_registry_collects_declarations_and_diagnostics():
    registry = ExtensionRegistry()

    registry.register(SqliteExtension())

    assert len(registry.capabilities) == 1
    assert len(registry.executors) == 1
    assert len(registry.evidence_schemas) == 1
    assert len(registry.tool_views) == 1
    assert {diagnostic.declaration_type for diagnostic in registry.diagnostics} == {
        "capability",
        "executor",
        "evidence_schema",
        "tool_view",
    }
    assert all(diagnostic.plugin_id == "sqlite" for diagnostic in registry.diagnostics)


def test_registry_collects_policy_context_worker_diagnostics():
    registry = ExtensionRegistry()

    registry.register(FullSurfacePlugin())

    assert len(registry.policies) == 1
    assert len(registry.context_providers) == 1
    assert len(registry.workers) == 1
    diagnostic_types = {
        diagnostic.declaration_type for diagnostic in registry.diagnostics
    }
    assert {
        "capability",
        "executor",
        "policy",
        "evidence_schema",
        "context_provider",
        "worker",
    } <= diagnostic_types
    assert all(
        diagnostic.plugin_id == "governance" for diagnostic in registry.diagnostics
    )


def test_registry_finds_capabilities_by_domain_and_operation_type():
    registry = ExtensionRegistry()
    registry.register(SqliteExtension())

    matches = registry.find_capabilities(domain="db", operation_type="data.query")

    assert [capability.id for capability in matches] == ["db.sql.execute_read"]
    assert registry.get_capability("db.sql.execute_read", owner="sqlite") == matches[0]
    assert registry.find_capabilities(domain="file") == ()


async def test_registry_runs_setup_and_teardown_hooks():
    plugin = SetupTeardownPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)

    context = PluginContext(runtime_id="runtime-1", runtime_kind="test")
    await registry.setup_all(context)
    await registry.teardown_all()

    assert plugin.setup_context is context
    assert plugin.teardown_called is True


async def test_registry_setup_all_runs_setup_time_registered_plugins():
    child = NamedSetupTeardownPlugin("setup_child")
    parent = SetupRegistersPlugin(child)
    registry = ExtensionRegistry()
    registry.register(parent)

    context = PluginContext(
        runtime_id="runtime-1",
        runtime_kind="test",
        services=ServiceRegistry({"extension_registry": registry}),
    )
    await registry.setup_all(context)

    assert child.setup_context is context


async def test_registry_rolls_back_completed_setup_on_later_failure():
    first = NamedSetupTeardownPlugin("first")
    failing = FailingSetupPlugin()
    registry = ExtensionRegistry()
    registry.register(first)
    registry.register(failing)

    context = PluginContext(runtime_id="runtime-1", runtime_kind="test")
    with pytest.raises(RuntimeError, match="setup exploded"):
        await registry.setup_all(context)

    assert first.setup_context is context
    assert first.teardown_called is True
    assert failing.teardown_called is False
