"""
Extension-first base contracts for Daita plugins.

Plugins declare runtime capabilities, executors, evidence schemas, policy,
context providers, tool views, and workers. Model-visible tools are a view over
capabilities, not the primary plugin contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, ClassVar, Mapping, Protocol

from daita.runtime import (
    Capability,
    ContextProvider,
    EvidenceSchema,
    Executor,
    Policy,
    ToolView,
    Worker,
)

from .manifest import PluginKind, PluginManifest


class SecretProvider(Protocol):
    """Runtime-provided secret lookup for plugins."""

    def get_secret(self, name: str) -> str | None:
        """Return a secret value by name, or None when unavailable."""


class EmptySecretProvider:
    """Secret provider used when a runtime has no secret source configured."""

    def get_secret(self, name: str) -> str | None:
        return None


class ServiceRegistry:
    """Small runtime service container passed through PluginContext."""

    def __init__(self, services: Mapping[str, Any] | None = None) -> None:
        self._services = dict(services or {})

    def register(self, name: str, service: Any) -> None:
        """Register a runtime-local service by name."""
        self._services[name] = service

    def get(self, name: str, default: Any = None) -> Any:
        """Return a registered service, or default when missing."""
        return self._services.get(name, default)

    def require(self, name: str) -> Any:
        """Return a registered service, raising KeyError when missing."""
        return self._services[name]

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy for diagnostics and tests."""
        return dict(self._services)


@dataclass(frozen=True)
class PluginContext:
    """Structured setup context for any runtime that hosts plugins."""

    runtime_id: str
    runtime_kind: str
    agent_id: str | None = None
    services: ServiceRegistry = field(default_factory=ServiceRegistry)
    config: Mapping[str, Any] = field(default_factory=dict)
    secrets: SecretProvider = field(default_factory=EmptySecretProvider)
    logger: Any = field(default_factory=lambda: logging.getLogger("daita.plugins"))


class BasePlugin(ABC):
    """
    Base class for extension-first plugins.

    Subclasses should declare a stable manifest and contribute runtime
    declarations through the methods below. Direct Python APIs remain useful on
    concrete plugins, but runtime setup and model-visible declarations flow
    through this contract.
    """

    manifest: ClassVar[PluginManifest]

    async def setup(self, context: PluginContext) -> None:
        """Prepare the plugin for a runtime."""

    async def teardown(self) -> None:
        """Release resources owned by the plugin."""

    def declare_capabilities(self) -> tuple[Capability, ...]:
        """Declare runtime-plannable behavior."""
        return ()

    def get_executors(self) -> tuple[Executor, ...]:
        """Return executors that perform declared capabilities."""
        return ()

    def declare_policies(self) -> tuple[Policy, ...]:
        """Declare policies that can shape or evaluate operations."""
        return ()

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        """Declare typed evidence schemas the plugin can produce."""
        return ()

    def get_context_providers(self) -> tuple[ContextProvider, ...]:
        """Return audience-specific context providers."""
        return ()

    def get_tool_views(self) -> tuple[ToolView, ...]:
        """Return optional model-visible views over capabilities."""
        return ()

    def get_workers(self) -> tuple[Worker, ...]:
        """Return specialist or background worker declarations."""
        return ()


class ConnectorPlugin(BasePlugin):
    """Base for source connectors that own resource lifecycle and execution."""

    expected_kind: ClassVar[PluginKind] = PluginKind.CONNECTOR

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the source."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the connector is currently connected."""


class DomainServicePlugin(BasePlugin):
    """Base for domain services such as catalog, lineage, memory, and quality."""

    expected_kind: ClassVar[PluginKind] = PluginKind.DOMAIN_SERVICE


class RuntimeExtensionPlugin(BasePlugin):
    """Base for plugins that extend runtime planning, policy, or governance."""

    expected_kind: ClassVar[PluginKind] = PluginKind.RUNTIME_EXTENSION


class WorkerProviderPlugin(BasePlugin):
    """Base for plugins that contribute specialist or background workers."""

    expected_kind: ClassVar[PluginKind] = PluginKind.WORKER_PROVIDER


class ObservabilityPlugin(BasePlugin):
    """Base for plugins that contribute audit, metrics, tracing, or logs."""

    expected_kind: ClassVar[PluginKind] = PluginKind.OBSERVABILITY


class SkillPlugin(BasePlugin):
    """Base for skills represented as runtime extension declarations."""

    expected_kind: ClassVar[PluginKind] = PluginKind.SKILL
