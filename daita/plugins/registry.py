"""Extension registry for runtime-aware plugin declarations.

The registry is the 1.0 plugin declaration owner. Plugins declare runtime
contracts here; runtimes plan against those declarations and the runtime kernel
executes them.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Iterable, Mapping

from daita.runtime import (
    Capability,
    ContextProvider,
    EvidenceSchema,
    Executor,
    Policy,
    ToolView,
    Worker,
)

from .base import PluginContext
from .manifest import PluginKind, PluginManifest

_CONTRIBUTOR_METHODS = {
    "capability": "declare_capabilities",
    "executor": "get_executors",
    "policy": "declare_policies",
    "evidence_schema": "declare_evidence_schemas",
    "context_provider": "get_context_providers",
    "tool_view": "get_tool_views",
    "worker": "get_workers",
}


@dataclass(frozen=True)
class RegistryDiagnostic:
    """Explains a declaration contributed by one plugin."""

    plugin_id: str
    declaration_type: str
    declaration_id: str
    message: str


class ExtensionRegistry:
    """
    Registry for plugin manifests and runtime extension declarations.

    Display names are never used as keys. Stable plugin IDs from
    :class:`PluginManifest` own all lookup and diagnostics.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Any] = {}
        self._manifests: dict[str, PluginManifest] = {}
        self._capabilities: dict[tuple[str, str], Capability] = {}
        self._executors: dict[str, Executor] = {}
        self._policies: dict[tuple[str, str], Policy] = {}
        self._evidence_schemas: dict[tuple[str, str], EvidenceSchema] = {}
        self._context_providers: dict[tuple[str, str], ContextProvider] = {}
        self._tool_views: dict[str, ToolView] = {}
        self._tool_view_owners: dict[str, str] = {}
        self._workers: dict[tuple[str, str], Worker] = {}
        self._diagnostics: list[RegistryDiagnostic] = []

    @property
    def plugin_ids(self) -> tuple[str, ...]:
        """Registered plugin IDs in registration order."""
        return tuple(self._plugins.keys())

    @property
    def manifests(self) -> tuple[PluginManifest, ...]:
        """Registered plugin manifests."""
        return tuple(self._manifests.values())

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        """Registered runtime capabilities."""
        return tuple(self._capabilities.values())

    @property
    def executors(self) -> tuple[Executor, ...]:
        """Registered task executors."""
        return tuple(self._executors.values())

    @property
    def evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        """Registered evidence schemas."""
        return tuple(self._evidence_schemas.values())

    @property
    def policies(self) -> tuple[Policy, ...]:
        """Registered policies."""
        return tuple(self._policies.values())

    @property
    def context_providers(self) -> tuple[ContextProvider, ...]:
        """Registered context providers."""
        return tuple(self._context_providers.values())

    @property
    def tool_views(self) -> tuple[ToolView, ...]:
        """Registered model-facing tool views."""
        return tuple(self._tool_views.values())

    @property
    def workers(self) -> tuple[Worker, ...]:
        """Registered worker declarations."""
        return tuple(self._workers.values())

    @property
    def diagnostics(self) -> tuple[RegistryDiagnostic, ...]:
        """Diagnostics describing which plugin contributed each declaration."""
        return tuple(self._diagnostics)

    def register(self, plugin: Any) -> None:
        """Register one plugin and validate its extension declarations."""
        manifest = self._get_manifest(plugin)
        if manifest.id in self._plugins:
            raise ValueError(f"duplicate plugin id: {manifest.id}")

        declarations = _CollectedDeclarations(
            capabilities=tuple(
                self._collect(plugin, "capability", Capability, manifest.id)
            ),
            executors=tuple(self._collect(plugin, "executor", None, manifest.id)),
            policies=tuple(self._collect(plugin, "policy", None, manifest.id)),
            evidence_schemas=tuple(
                self._collect(plugin, "evidence_schema", EvidenceSchema, manifest.id)
            ),
            context_providers=tuple(
                self._collect(plugin, "context_provider", None, manifest.id)
            ),
            tool_views=tuple(self._collect(plugin, "tool_view", ToolView, manifest.id)),
            workers=tuple(self._collect(plugin, "worker", Worker, manifest.id)),
        )
        self._validate_declarations(manifest, declarations)

        self._plugins[manifest.id] = plugin
        self._manifests[manifest.id] = manifest
        self._commit_declarations(manifest.id, declarations)

    def register_many(self, plugins: Iterable[Any]) -> None:
        """Register plugins in order."""
        for plugin in plugins:
            self.register(plugin)

    async def setup_all(self, context: PluginContext) -> None:
        """Call setup(context) on registered plugins."""
        completed: list[Any] = []
        completed_ids: set[str] = set()
        while True:
            pending_ids = [
                plugin_id
                for plugin_id in self.plugin_ids
                if plugin_id not in completed_ids
            ]
            if not pending_ids:
                return
            plugin = self.get_plugin(pending_ids[0])
            try:
                await self._setup_plugin(plugin, context)
            except Exception:
                await self._teardown_plugins(reversed(completed))
                raise
            completed.append(plugin)
            completed_ids.add(pending_ids[0])

    async def setup_plugin(self, plugin_id: str, context: PluginContext) -> None:
        """Call setup(context) on one registered plugin."""
        await self._setup_plugin(self.get_plugin(plugin_id), context)

    async def _setup_plugin(self, plugin: Any, context: PluginContext) -> None:
        """Call setup(context) on one plugin if it defines a setup hook."""
        setup = getattr(plugin, "setup", None)
        if setup is None:
            return
        result = setup(context)
        if inspect.isawaitable(result):
            await result

    async def teardown_all(self) -> None:
        """Call teardown() on registered plugins in reverse registration order."""
        await self._teardown_plugins(reversed(tuple(self._plugins.values())))

    async def _teardown_plugins(self, plugins: Iterable[Any]) -> None:
        """Call teardown() on the provided plugins in order."""
        for plugin in plugins:
            teardown = getattr(plugin, "teardown", None)
            if teardown is None:
                continue
            result = teardown()
            if inspect.isawaitable(result):
                await result

    def get_plugin(self, plugin_id: str) -> Any:
        """Return a registered plugin by stable plugin ID."""
        return self._plugins[plugin_id]

    def get_capability(
        self, capability_id: str, owner: str | None = None
    ) -> Capability:
        """Return one capability, optionally disambiguated by owner."""
        matches = [
            capability
            for (
                registered_id,
                registered_owner,
            ), capability in self._capabilities.items()
            if registered_id == capability_id
            and (owner is None or registered_owner == owner)
        ]
        if not matches:
            raise KeyError(capability_id)
        if owner is None and len(matches) > 1:
            raise ValueError(
                f"capability {capability_id!r} has multiple owners; pass owner"
            )
        return matches[0]

    def get_executor(self, executor_id: str) -> Executor:
        """Return one executor by stable executor ID."""
        try:
            return self._executors[executor_id]
        except KeyError as exc:
            raise KeyError(executor_id) from exc

    def get_policy(self, policy_id: str, owner: str | None = None) -> Policy:
        """Return one policy, optionally disambiguated by owner."""
        return self._get_owned_declaration(
            "policy",
            self._policies,
            policy_id,
            owner,
        )

    def get_evidence_schema(
        self, kind: str, owner: str | None = None
    ) -> EvidenceSchema:
        """Return one evidence schema, optionally disambiguated by owner."""
        return self._get_owned_declaration(
            "evidence schema",
            self._evidence_schemas,
            kind,
            owner,
        )

    def get_context_provider(
        self, provider_id: str, owner: str | None = None
    ) -> ContextProvider:
        """Return one context provider, optionally disambiguated by owner."""
        return self._get_owned_declaration(
            "context provider",
            self._context_providers,
            provider_id,
            owner,
        )

    def get_tool_view_owner(self, tool_view_name: str) -> str:
        """Return the plugin ID that contributed a registered tool view."""
        try:
            return self._tool_view_owners[tool_view_name]
        except KeyError as exc:
            raise KeyError(tool_view_name) from exc

    def get_worker(self, worker_id: str, owner: str | None = None) -> Worker:
        """Return one worker declaration, optionally disambiguated by owner."""
        return self._get_owned_declaration(
            "worker",
            self._workers,
            worker_id,
            owner,
        )

    def find_capabilities(
        self,
        *,
        domain: str | None = None,
        operation_type: str | None = None,
    ) -> tuple[Capability, ...]:
        """Find capabilities matching a domain and/or operation type."""
        matches = []
        for capability in self._capabilities.values():
            if domain is not None and domain not in capability.domains:
                continue
            if (
                operation_type is not None
                and operation_type not in capability.operation_types
            ):
                continue
            matches.append(capability)
        return tuple(matches)

    def _get_owned_declaration(
        self,
        label: str,
        declarations: dict[tuple[str, str], Any],
        declaration_id: str,
        owner: str | None,
    ) -> Any:
        matches = [
            declaration
            for (
                registered_id,
                registered_owner,
            ), declaration in declarations.items()
            if registered_id == declaration_id
            and (owner is None or registered_owner == owner)
        ]
        if not matches:
            raise KeyError(declaration_id)
        if owner is None and len(matches) > 1:
            raise ValueError(
                f"{label} {declaration_id!r} has multiple owners; pass owner"
            )
        return matches[0]

    def _get_manifest(self, plugin: Any) -> PluginManifest:
        manifest = getattr(plugin, "manifest", None)
        if manifest is None:
            raise ValueError("registered plugins must expose a manifest")
        if isinstance(manifest, Mapping):
            manifest = PluginManifest.from_dict(manifest)
        if not isinstance(manifest, PluginManifest):
            raise TypeError("plugin manifest must be a PluginManifest")
        expected_kind = getattr(plugin, "expected_kind", None)
        if expected_kind is not None and manifest.kind is not PluginKind(expected_kind):
            raise ValueError(
                f"plugin {manifest.id!r} manifest kind {manifest.kind.value!r} "
                f"does not match base contract {PluginKind(expected_kind).value!r}"
            )
        return manifest

    def _collect(
        self,
        plugin: Any,
        declaration_type: str,
        expected_type: type[Any] | None,
        plugin_id: str,
    ) -> tuple[Any, ...]:
        method_name = _CONTRIBUTOR_METHODS[declaration_type]
        method = getattr(plugin, method_name, None)
        if method is None:
            return ()
        contributed = method()
        if contributed is None:
            return ()
        items = tuple(contributed)
        if expected_type is not None:
            for item in items:
                if not isinstance(item, expected_type):
                    raise TypeError(
                        f"{method_name} on plugin {plugin_id!r} must return "
                        f"{expected_type.__name__} items"
                    )
        return items

    def _validate_declarations(
        self,
        manifest: PluginManifest,
        declarations: "_CollectedDeclarations",
    ) -> None:
        local_executor_ids: set[str] = set()
        for executor in declarations.executors:
            executor_id = _declaration_id(executor)
            executor_owner = getattr(executor, "owner", manifest.id)
            if executor_owner != manifest.id:
                raise ValueError(
                    f"executor {executor_id!r} owner {executor_owner!r} "
                    f"does not match plugin id {manifest.id!r}"
                )
            if executor_id in local_executor_ids:
                raise ValueError(f"duplicate executor id: {executor_id}")
            local_executor_ids.add(executor_id)

        local_capability_keys: set[tuple[str, str]] = set()
        for capability in declarations.capabilities:
            if capability.owner != manifest.id:
                raise ValueError(
                    f"capability {capability.id!r} owner {capability.owner!r} "
                    f"does not match plugin id {manifest.id!r}"
                )
            key = (capability.id, capability.owner)
            if key in local_capability_keys:
                raise ValueError(
                    f"duplicate capability {capability.id!r} from owner "
                    f"{capability.owner!r}"
                )
            local_capability_keys.add(key)
            if key in self._capabilities:
                raise ValueError(
                    f"duplicate capability {capability.id!r} from owner "
                    f"{capability.owner!r}"
                )
            if capability.executor not in local_executor_ids:
                raise ValueError(
                    f"capability {capability.id!r} references missing executor "
                    f"{capability.executor!r}"
                )

        capability_keys = set(self._capabilities)
        capability_keys.update(
            (capability.id, capability.owner)
            for capability in declarations.capabilities
        )
        for executor in declarations.executors:
            executor_id = _declaration_id(executor)
            if executor_id in self._executors:
                raise ValueError(f"duplicate executor id: {executor_id}")
            for capability_id in getattr(executor, "capability_ids", frozenset()):
                if (capability_id, manifest.id) not in capability_keys:
                    raise ValueError(
                        f"executor {executor_id!r} references missing capability "
                        f"{capability_id!r}"
                    )

        self._validate_unique_by_owner(
            "policy", self._policies, declarations.policies, manifest.id
        )
        self._validate_unique_by_owner(
            "evidence schema",
            self._evidence_schemas,
            declarations.evidence_schemas,
            manifest.id,
        )
        self._validate_unique_by_owner(
            "context provider",
            self._context_providers,
            declarations.context_providers,
            manifest.id,
        )
        self._validate_unique_by_owner(
            "worker", self._workers, declarations.workers, manifest.id
        )

        local_tool_view_names: set[str] = set()
        for tool_view in declarations.tool_views:
            if tool_view.name in local_tool_view_names:
                raise ValueError(f"duplicate tool view name: {tool_view.name}")
            local_tool_view_names.add(tool_view.name)
            if tool_view.name in self._tool_views:
                raise ValueError(f"duplicate tool view name: {tool_view.name}")
            if (tool_view.capability_id, manifest.id) not in capability_keys:
                raise ValueError(
                    f"tool view {tool_view.name!r} references missing capability "
                    f"{tool_view.capability_id!r}"
                )
            capability = next(
                capability
                for capability in declarations.capabilities
                if capability.id == tool_view.capability_id
                and capability.owner == manifest.id
            )
            if capability.runtime_only or capability.specialist_only:
                hidden_flags = []
                if capability.runtime_only:
                    hidden_flags.append("runtime_only")
                if capability.specialist_only:
                    hidden_flags.append("specialist_only")
                raise ValueError(
                    f"tool view {tool_view.name!r} cannot expose hidden capability "
                    f"{tool_view.capability_id!r} ({', '.join(hidden_flags)})"
                )

    def _validate_unique_by_owner(
        self,
        label: str,
        existing: dict[tuple[str, str], Any],
        declarations: tuple[Any, ...],
        plugin_id: str,
    ) -> None:
        local_keys: set[tuple[str, str]] = set()
        for declaration in declarations:
            owner = getattr(declaration, "owner", plugin_id)
            if owner != plugin_id:
                raise ValueError(
                    f"{label} {_declaration_id(declaration)!r} owner {owner!r} "
                    f"does not match plugin id {plugin_id!r}"
                )
            key = (_declaration_id(declaration), owner)
            if key in local_keys:
                raise ValueError(
                    f"duplicate {label} {_declaration_id(declaration)!r} "
                    f"from owner {owner!r}"
                )
            local_keys.add(key)
            if key in existing:
                raise ValueError(
                    f"duplicate {label} {_declaration_id(declaration)!r} "
                    f"from owner {owner!r}"
                )

    def _commit_declarations(
        self,
        plugin_id: str,
        declarations: "_CollectedDeclarations",
    ) -> None:
        for capability in declarations.capabilities:
            self._capabilities[(capability.id, capability.owner)] = capability
            self._diagnostics.append(
                RegistryDiagnostic(
                    plugin_id=plugin_id,
                    declaration_type="capability",
                    declaration_id=capability.id,
                    message=f"plugin {plugin_id!r} contributed capability {capability.id!r}",
                )
            )
        for executor in declarations.executors:
            executor_id = _declaration_id(executor)
            self._executors[executor_id] = executor
            self._diagnostics.append(
                RegistryDiagnostic(
                    plugin_id=plugin_id,
                    declaration_type="executor",
                    declaration_id=executor_id,
                    message=f"plugin {plugin_id!r} contributed executor {executor_id!r}",
                )
            )
        for policy in declarations.policies:
            self._policies[(_declaration_id(policy), policy.owner)] = policy
            self._add_diagnostic(plugin_id, "policy", _declaration_id(policy))
        for schema in declarations.evidence_schemas:
            self._evidence_schemas[(schema.kind, schema.owner)] = schema
            self._add_diagnostic(plugin_id, "evidence_schema", schema.kind)
        for provider in declarations.context_providers:
            self._context_providers[(_declaration_id(provider), provider.owner)] = (
                provider
            )
            self._add_diagnostic(
                plugin_id, "context_provider", _declaration_id(provider)
            )
        for tool_view in declarations.tool_views:
            self._tool_views[tool_view.name] = tool_view
            self._tool_view_owners[tool_view.name] = plugin_id
            self._add_diagnostic(plugin_id, "tool_view", tool_view.name)
        for worker in declarations.workers:
            self._workers[(worker.id, worker.owner)] = worker
            self._add_diagnostic(plugin_id, "worker", worker.id)

    def _add_diagnostic(
        self,
        plugin_id: str,
        declaration_type: str,
        declaration_id: str,
    ) -> None:
        self._diagnostics.append(
            RegistryDiagnostic(
                plugin_id=plugin_id,
                declaration_type=declaration_type,
                declaration_id=declaration_id,
                message=(
                    f"plugin {plugin_id!r} contributed {declaration_type} "
                    f"{declaration_id!r}"
                ),
            )
        )


@dataclass(frozen=True)
class _CollectedDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    policies: tuple[Policy, ...]
    evidence_schemas: tuple[EvidenceSchema, ...]
    context_providers: tuple[ContextProvider, ...]
    tool_views: tuple[ToolView, ...]
    workers: tuple[Worker, ...]


def _declaration_id(declaration: Any) -> str:
    declaration_id = getattr(declaration, "id", None)
    if declaration_id is None:
        declaration_id = getattr(declaration, "kind", None)
    if declaration_id is None:
        raise ValueError(f"declaration {declaration!r} must expose an id")
    return declaration_id
