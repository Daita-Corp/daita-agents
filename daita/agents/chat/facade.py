"""Facade helpers for the secondary generic chat Agent."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from daita.core.exceptions import SkillError
from daita.core.tools import LocalTool
from daita.plugins.base import PluginContext, ServiceRegistry
from daita.plugins.manifest import PluginManifest
from daita.plugins.registry import RegistryDiagnostic
from daita.runtime import (
    Capability,
    ContextAudience,
    ContextBlock,
    ContextProvider,
    EvidenceSchema,
    Executor,
    Policy,
    ToolView,
    Worker,
)
from daita.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class ChatAgentFacadeMixin:
    """Public chat-agent projections backed by registry and ChatRuntime owners."""

    def _register_tool_source(self, source) -> None:
        """Register a local LocalTool source into the compatibility registry."""
        if isinstance(source, LocalTool):
            self.local_tool_catalog.register(source)
        else:
            logger.warning(
                "Ignoring non-manifest plugin %s; runtime plugins must declare a "
                "PluginManifest with capabilities and executors.",
                source.__class__.__name__,
            )

    async def _setup_tools(self):
        """Set up local tool declarations. Tools are registered eagerly in add_plugin()."""
        await self._setup_extension_plugins()

        if self._tools_setup:
            return  # Already setup

        self._tools_setup = True
        logger.info(
            f"Agent {self.name} ready with {self.local_tool_catalog.tool_count} tools"
        )

    async def _setup_extension_plugins(self) -> None:
        """Set up runtime-aware plugins through the extension registry."""
        context = PluginContext(
            runtime_id=self.agent_id,
            runtime_kind="agent",
            agent_id=self.agent_id,
            services=ServiceRegistry(
                {
                    "extension_registry": self.extension_registry,
                    "runtime_store": self.runtime_store,
                    "runtime_kernel": self.runtime_kernel,
                }
            ),
        )
        while pending_plugin_ids := self._pending_extension_setup_plugin_ids():
            plugin_id = pending_plugin_ids[0]
            await self.extension_registry.setup_plugin(plugin_id, context)
            self._extension_setup_plugin_ids.add(plugin_id)

    def _pending_extension_setup_plugin_ids(self) -> List[str]:
        """Return registered plugin IDs that have not run PluginContext setup."""
        return [
            plugin_id
            for plugin_id in self.extension_registry.plugin_ids
            if plugin_id not in self._extension_setup_plugin_ids
        ]

    async def setup_extensions(self) -> None:
        """Set up attached registry plugins through ``PluginContext``."""
        await self._setup_extension_plugins()

    async def teardown_extensions(self) -> None:
        """Tear down attached registry plugins and clear setup state."""
        if not self.extension_registry.plugin_ids:
            return
        try:
            await self.extension_registry.teardown_all()
        finally:
            self._extension_setup_plugin_ids.clear()

    @property
    def extension_setup_plugin_ids(self) -> List[str]:
        """Return registry plugin IDs that have completed ``PluginContext`` setup."""
        return [
            plugin_id
            for plugin_id in self.extension_registry.plugin_ids
            if plugin_id in self._extension_setup_plugin_ids
        ]

    @property
    def pending_extension_setup_plugin_ids(self) -> List[str]:
        """Return registry plugin IDs pending ``PluginContext`` setup."""
        return self._pending_extension_setup_plugin_ids()

    @property
    def extensions_setup_complete(self) -> bool:
        """Return True when all registry plugins have completed setup."""
        return not self.pending_extension_setup_plugin_ids

    def _resolve_tools(
        self, tools: Optional[List[Union[str, LocalTool]]]
    ) -> List[LocalTool]:
        """Resolve tool names to LocalTool instances. If None, returns all registered tools."""
        if tools is None:
            # Use all registered tools
            return list(self.available_tools)

        tool_list = []
        for t in tools:
            if isinstance(t, str):
                # Tool name - look up in registry
                tool = next(
                    (item for item in self.available_tools if item.name == t),
                    None,
                )
                if tool is None:
                    raise ValueError(f"Tool '{t}' not found in registry")
                tool_list.append(tool)
            else:
                # Already an LocalTool instance
                tool_list.append(t)

        return tool_list

    def _emit_event(self, on_event: Optional[Callable], event_type, **kwargs):
        """Emit event only if callback provided. Zero overhead when None."""
        if on_event:
            from daita.core.streaming import AgentEvent

            on_event(AgentEvent(type=event_type, **kwargs))

    async def _execute_autonomous_with_retry(
        self,
        prompt: str,
        tools: Optional[List[Union[str, "LocalTool"]]],
        max_iterations: int,
        on_event: Optional[Callable],
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute autonomous tool calling with retry logic via ChatRuntime."""
        result = await self.runtime.run_with_retry(
            prompt=prompt,
            tools=tools,
            max_iterations=max_iterations,
            on_event=on_event,
            **kwargs,
        )
        return result.to_agent_result()

    @property
    def context_providers(self) -> List[ContextProvider]:
        """Return context providers declared by attached registry plugins."""
        return list(self.extension_registry.context_providers)

    def get_context_provider(
        self, provider_id: str, *, owner: Optional[str] = None
    ) -> ContextProvider:
        """Return one context provider declaration, optionally disambiguated by owner."""
        return self.extension_registry.get_context_provider(provider_id, owner=owner)

    async def render_context_blocks(
        self,
        prompt: str,
        *,
        audience: ContextAudience = ContextAudience.PRIMARY_MODEL,
        token_budget: int = 2000,
    ) -> List[ContextBlock]:
        """Render context blocks for a target runtime audience."""
        return await self.runtime.render_context_blocks(
            prompt,
            audience=audience,
            token_budget=token_budget,
        )

    async def _execute_autonomous(
        self,
        prompt: str,
        tools: Optional[List[Union[str, "LocalTool"]]],
        max_iterations: int,
        on_event: Optional[Callable],
        initial_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delegate generic chat execution to ChatRuntime."""
        result = await self.runtime.run(
            prompt=prompt,
            tools=tools,
            max_iterations=max_iterations,
            on_event=on_event,
            initial_messages=initial_messages,
            **kwargs,
        )
        return result.to_agent_result()

    # User customization methods

    def _attached_plugin_sources(self) -> List[Any]:
        """Return attached plugins with registry-owned plugins first."""
        return [
            *(
                self.extension_registry.get_plugin(plugin_id)
                for plugin_id in self.extension_registry.plugin_ids
            ),
            *self._local_tool_sources,
        ]

    @property
    def attached_plugins(self) -> List[Any]:
        """Return attached plugin objects using registry-first ownership order."""
        return self._attached_plugin_sources()

    @property
    def attached_plugin_ids(self) -> List[str]:
        """Return stable IDs for manifest plugins and class names for local tool sources."""
        plugin_ids: List[str] = []
        for source in self._attached_plugin_sources():
            manifest = getattr(source, "manifest", None)
            if isinstance(manifest, dict):
                plugin_id = manifest.get("id")
            else:
                plugin_id = getattr(manifest, "id", None)
            plugin_ids.append(plugin_id or source.__class__.__name__)
        return plugin_ids

    def get_attached_plugin(self, identifier: Union[str, type]) -> Optional[Any]:
        """Return one attached plugin by stable ID, class name, or type."""
        for source in self._attached_plugin_sources():
            if isinstance(identifier, type):
                if isinstance(source, identifier):
                    return source
                continue

            manifest = getattr(source, "manifest", None)
            plugin_id = (
                manifest.get("id")
                if isinstance(manifest, dict)
                else getattr(manifest, "id", None)
            )
            if identifier in {plugin_id, source.__class__.__name__}:
                return source
        return None

    @property
    def plugin_manifests(self) -> List[PluginManifest]:
        """Return manifests declared by attached registry plugins."""
        return list(self.extension_registry.manifests)

    def get_plugin_manifest(self, plugin_id: str) -> Optional[PluginManifest]:
        """Return one attached registry plugin manifest by stable plugin ID."""
        return next(
            (
                manifest
                for manifest in self.plugin_manifests
                if manifest.id == plugin_id
            ),
            None,
        )

    @property
    def capabilities(self) -> List[Capability]:
        """Return runtime capabilities declared by attached registry plugins."""
        return list(self.extension_registry.capabilities)

    def get_capability(
        self, capability_id: str, *, owner: Optional[str] = None
    ) -> Capability:
        """Return one runtime capability, optionally disambiguated by owner."""
        return self.extension_registry.get_capability(capability_id, owner=owner)

    def find_capabilities(
        self,
        *,
        domain: Optional[str] = None,
        operation_type: Optional[str] = None,
    ) -> List[Capability]:
        """Find declared runtime capabilities by domain or operation type."""
        return list(
            self.extension_registry.find_capabilities(
                domain=domain,
                operation_type=operation_type,
            )
        )

    @property
    def tool_views(self) -> List[ToolView]:
        """Return model-facing tool view declarations from registry plugins."""
        return list(self.extension_registry.tool_views)

    def get_tool_view(self, name: str) -> Optional[ToolView]:
        """Return one model-facing tool view declaration by tool name."""
        return next((view for view in self.tool_views if view.name == name), None)

    def get_tool_view_owner(self, name: str) -> str:
        """Return the stable plugin ID that owns a registry tool view."""
        return self.extension_registry.get_tool_view_owner(name)

    @property
    def executors(self) -> List[Executor]:
        """Return executors declared by attached registry plugins."""
        return list(self.extension_registry.executors)

    def get_executor(self, executor_id: str) -> Executor:
        """Return one executor declared by attached registry plugins."""
        return self.extension_registry.get_executor(executor_id)

    async def execute_capability(
        self,
        capability_id: str,
        arguments: Dict[str, Any],
        *,
        owner: Optional[str] = None,
        operation_type: Optional[str] = None,
        operation_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        executor_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one registry capability through the shared runtime kernel."""
        await self.setup_extensions()

        capability = self.get_capability(capability_id, owner=owner)
        selected_operation_type = (
            operation_type
            or (
                sorted(capability.operation_types)[0]
                if capability.operation_types
                else None
            )
            or "capability.execute"
        )
        execution_metadata = {
            "capability_id": capability.id,
            **(metadata or {}),
        }
        context = {
            "agent_id": self.agent_id,
            "runtime_id": self.agent_id,
            "capability": capability.to_dict(),
        }
        context.update(executor_context or {})

        if operation_id is None:
            operation = await self.runtime_kernel.create_operation(
                operation_type=selected_operation_type,
                request=arguments,
                required_evidence=capability.output_evidence,
                metadata=execution_metadata,
            )
        else:
            operation = await self.runtime_store.load_operation(operation_id)
            if operation is None:
                operation = await self.runtime_kernel.create_operation(
                    operation_id=operation_id,
                    operation_type=selected_operation_type,
                    request=arguments,
                    required_evidence=capability.output_evidence,
                    metadata=execution_metadata,
                )
        task = await self.runtime_kernel.plan_task(
            task_id=task_id,
            operation_id=operation.id,
            capability_id=capability.id,
            owner=capability.owner,
            input=arguments,
            metadata=execution_metadata,
        )
        result = await self.runtime_kernel.execute_task(task.id, context=context)
        return {
            "capability_id": capability.id,
            "evidence": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in result.evidence
            ],
        }

    @property
    def policies(self) -> List[Policy]:
        """Return policies declared by attached registry plugins."""
        return list(self.extension_registry.policies)

    def get_policy(self, policy_id: str, *, owner: Optional[str] = None) -> Policy:
        """Return one policy declaration, optionally disambiguated by owner."""
        return self.extension_registry.get_policy(policy_id, owner=owner)

    @property
    def evidence_schemas(self) -> List[EvidenceSchema]:
        """Return evidence schemas declared by attached registry plugins."""
        return list(self.extension_registry.evidence_schemas)

    def get_evidence_schema(
        self, kind: str, *, owner: Optional[str] = None
    ) -> EvidenceSchema:
        """Return one evidence schema declaration, optionally disambiguated by owner."""
        return self.extension_registry.get_evidence_schema(kind, owner=owner)

    @property
    def workers(self) -> List[Worker]:
        """Return workers declared by attached registry plugins."""
        return list(self.extension_registry.workers)

    def get_worker(self, worker_id: str, *, owner: Optional[str] = None) -> Worker:
        """Return one worker declaration, optionally disambiguated by owner."""
        return self.extension_registry.get_worker(worker_id, owner=owner)

    @property
    def extension_diagnostics(self) -> List[RegistryDiagnostic]:
        """Return diagnostics for registry-owned extension declarations."""
        return list(self.extension_registry.diagnostics)

    def add_plugin(self, plugin: Any):
        """
        Add a plugin or local LocalTool to the agent.

        Manifest-bearing plugins are registered with the extension registry so
        runtime semantics are keyed by stable plugin IDs. Non-manifest sources
        are accepted only when they are local LocalTool projection objects.
        """
        has_manifest = getattr(plugin, "manifest", None) is not None
        if has_manifest:
            self.extension_registry.register(plugin)
        else:
            self._local_tool_sources.append(plugin)
            self._register_tool_source(plugin)
        logger.debug(f"Added plugin: {plugin.__class__.__name__}")

    def add_skill(self, skill: "BaseSkill"):
        """Add a skill to the agent.

        Capability requirements declared in ``skill.requires_capabilities()``
        are resolved against already-registered extension capabilities. Add
        capability-providing plugins before skills that need them.
        """
        requirements = skill.requires_capabilities()
        if requirements:
            capabilities_by_id: dict[str, list] = {}
            for capability in self.extension_registry.capabilities:
                capabilities_by_id.setdefault(capability.id, []).append(capability)
            unmet = [
                capability_id
                for capability_id in requirements
                if capability_id not in capabilities_by_id
            ]
            if unmet:
                raise SkillError(
                    f"Skill '{skill.name}' requires capabilities not yet available: "
                    f"{', '.join(unmet)}. Add capability-providing plugin(s) before "
                    f"adding this skill.",
                    plugin_name=skill.name,
                )
            skill._resolved_capabilities = {
                capability_id: tuple(capabilities_by_id[capability_id])
                for capability_id in requirements
            }

        self.add_plugin(skill)

    @property
    def skills(self) -> List["BaseSkill"]:
        """Return all attached skills."""
        return [
            source
            for source in self._attached_plugin_sources()
            if isinstance(source, BaseSkill)
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name with arguments."""
        await self._setup_tools()
        view = self.get_tool_view(name)
        if view is not None:
            owner = self.extension_registry.get_tool_view_owner(name)
            capability = self.extension_registry.get_capability(
                view.capability_id,
                owner=owner,
            )
            operation_type = (
                sorted(capability.operation_types)[0]
                if capability.operation_types
                else "tool.call"
            )
            return await self.execute_capability(
                view.capability_id,
                arguments,
                owner=owner,
                operation_type=operation_type,
                metadata={"tool_view": view.name},
                executor_context={"tool_view": view.to_dict(), "tool_owner": owner},
            )

        tool = self.local_tool_catalog.get(name)
        if tool is None:
            raise RuntimeError(f"Tool '{name}' not found in registry")
        spec = self.runtime._register_local_tool(tool)
        execution = await self.runtime_kernel.execute_capability(
            spec.capability_id,
            owner=spec.owner,
            operation_type="chat.tool_call",
            input=arguments,
            context={
                "tool_view": spec.tool_view.to_dict(),
                "tool_owner": spec.owner,
            },
        )
        return self.runtime._render_evidence_for_model(tuple(execution.evidence))

    @property
    def available_tools(self) -> List[LocalTool]:
        """Get model-visible local tools plus registry ToolView projections."""
        projected = []
        for view in self.extension_registry.tool_views:
            if not view.model_visible:
                continue
            owner = self.extension_registry.get_tool_view_owner(view.name)
            capability = self.extension_registry.get_capability(
                view.capability_id,
                owner=owner,
            )
            projected.append(
                LocalTool(
                    name=view.name,
                    description=view.description,
                    parameters=view.parameters,
                    handler=lambda arguments, tool_name=view.name: self.call_tool(
                        tool_name,
                        arguments,
                    ),
                    source="plugin",
                    plugin_name=owner,
                    capability_ids=(view.capability_id,),
                    output_evidence=tuple(capability.output_evidence),
                    timeout_seconds=capability.timeout_seconds,
                    retry_safe=capability.retry_safe,
                    replay_safe=capability.replay_safe,
                    idempotent=capability.idempotent,
                    side_effecting=capability.side_effecting,
                    metadata=dict(view.metadata),
                )
            )
        return [*projected, *self.local_tool_catalog.tools.copy()]

    @property
    def tools(self) -> List[LocalTool]:
        """Return model-visible tools using the registry-backed projection."""
        return self.available_tools

    @property
    def tool_names(self) -> List[str]:
        """Get list of all tool names"""
        return [tool.name for tool in self.available_tools]

    async def __aenter__(self):
        """Support ``async with agent:`` for automatic lifecycle management."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Call stop() on exit to flush plugins and release resources."""
        try:
            await self.stop()
        except Exception as e:
            if exc_type is None:
                raise
            logger.error("Error during agent stop: %s", e, exc_info=True)
        return False

    async def stop(self) -> None:
        """Stop agent and clean up runtime extension resources."""
        await self.teardown_extensions()

        # Call parent stop for standard cleanup
        await super().stop()

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics from automatic tracing."""
        if not self.llm or not hasattr(self.llm, "get_token_stats"):
            # Fallback for agents without LLM or tracing
            return {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0,
            }

        return self.llm.get_token_stats()

    @property
    def health(self) -> Dict[str, Any]:
        """Enhanced health information for Agent."""
        base_health = super().health

        # Add Agent-specific health info
        base_health.update(
            {
                "tools": {
                    "count": len(self.available_tools),
                    "setup": self._tools_setup,
                    "names": self.tool_names,
                },
                "extensions": {
                    "plugin_ids": list(self.extension_registry.plugin_ids),
                    "manifest_ids": [manifest.id for manifest in self.plugin_manifests],
                    "capability_ids": [
                        capability.id for capability in self.capabilities
                    ],
                    "capability_count": len(self.capabilities),
                    "tool_view_names": [view.name for view in self.tool_views],
                    "tool_view_count": len(self.tool_views),
                    "context_provider_ids": [
                        provider.id for provider in self.context_providers
                    ],
                    "context_provider_count": len(self.context_providers),
                    "executor_ids": [executor.id for executor in self.executors],
                    "executor_count": len(self.executors),
                    "policy_ids": [policy.id for policy in self.policies],
                    "policy_count": len(self.policies),
                    "evidence_schema_kinds": [
                        schema.kind for schema in self.evidence_schemas
                    ],
                    "evidence_schema_count": len(self.evidence_schemas),
                    "worker_ids": [worker.id for worker in self.workers],
                    "worker_count": len(self.workers),
                    "diagnostic_ids": [
                        diagnostic.declaration_id
                        for diagnostic in self.extension_diagnostics
                    ],
                    "diagnostic_count": len(self.extension_diagnostics),
                    "setup_plugin_ids": self.extension_setup_plugin_ids,
                    "pending_setup_plugin_ids": self.pending_extension_setup_plugin_ids,
                    "setup_complete": self.extensions_setup_complete,
                },
                "llm": {
                    "available": self.llm is not None,
                    "provider": (
                        self.llm.provider_name
                        if self.llm and hasattr(self.llm, "provider_name")
                        else None
                    ),
                },
            }
        )

        return base_health
