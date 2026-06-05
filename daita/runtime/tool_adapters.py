"""Shared adapters for projecting local tools into runtime declarations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from daita.plugins.manifest import PluginKind, PluginManifest

from .primitives import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
    ToolView,
)


@dataclass(frozen=True)
class LocalToolRuntimeRegistration:
    """Runtime declarations registered for one local tool."""

    plugin_id: str
    capability_id: str
    executor_id: str
    evidence_kind: str
    tool_view: ToolView | None
    capability: Capability


class LocalToolExecutor:
    """Runtime executor for a local tool declaration."""

    def __init__(
        self,
        executor_id: str,
        capability_id: str,
        tool: Any,
        evidence_kind: str,
    ) -> None:
        self.id = executor_id
        self.capability_ids = frozenset({capability_id})
        self._tool = tool
        self._evidence_kind = evidence_kind

    def replace_tool(self, tool: Any) -> None:
        """Replace the callable when declarations are unchanged."""
        self._tool = tool

    async def execute(self, task, operation, context):
        tool = self._tool
        if not callable(getattr(tool, "handler", None)):
            raise RuntimeError(
                f"Tool '{getattr(tool, 'name', task.capability_id)}' has non-callable handler"
            )

        async def invoke():
            return await tool.handler(task.input)

        timeout = getattr(tool, "timeout_seconds", None)
        if timeout:
            result = await asyncio.wait_for(invoke(), timeout=timeout)
        else:
            result = await invoke()
        if isinstance(result, Evidence):
            return [result]
        if isinstance(result, (list, tuple)) and all(
            isinstance(item, Evidence) for item in result
        ):
            return list(result)
        if isinstance(result, Mapping):
            payload = dict(result)
        else:
            payload = {
                "tool": getattr(tool, "name", task.capability_id),
                "arguments": dict(task.input),
                "result": result,
            }
        return [
            Evidence(
                kind=self._evidence_kind,
                owner=context.get("tool_owner"),
                payload=payload,
                metadata={"source": getattr(tool, "source", "custom")},
            )
        ]


class LocalToolPlugin:
    """One local tool represented as runtime declarations."""

    def __init__(
        self,
        tool: Any,
        *,
        plugin_id: str,
        capability_id: str,
        executor: LocalToolExecutor,
    ) -> None:
        output_evidence = tuple(
            getattr(tool, "output_evidence", ()) or (f"{capability_id}.result",)
        )
        evidence_kind = output_evidence[0]
        self.tool = tool
        self.manifest = PluginManifest(
            id=plugin_id,
            display_name=f"Local Tool {tool.name}",
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
            domains=frozenset(getattr(tool, "domains", None) or ("chat",)),
        )
        access = getattr(tool, "access", None) or AccessMode.READ
        risk = getattr(tool, "risk", None) or RiskLevel.LOW
        self._capability = Capability(
            id=capability_id,
            owner=plugin_id,
            description=tool.description,
            domains=frozenset(getattr(tool, "domains", None) or ("chat",)),
            operation_types=frozenset(
                getattr(tool, "operation_types", None) or ("chat.tool_call",)
            ),
            access=AccessMode(access),
            risk=RiskLevel(risk),
            input_schema=tool.parameters,
            output_evidence=frozenset(output_evidence),
            executor=executor.id,
            model_visible=bool(getattr(tool, "model_visible", True)),
            runtime_only=bool(getattr(tool, "runtime_only", False)),
            specialist_only=bool(getattr(tool, "specialist_only", False)),
            timeout_seconds=getattr(tool, "timeout_seconds", None),
            retry_safe=bool(getattr(tool, "retry_safe", False)),
            replay_safe=bool(getattr(tool, "replay_safe", False)),
            idempotent=bool(getattr(tool, "idempotent", False)),
            side_effecting=bool(getattr(tool, "side_effecting", True)),
            metadata=dict(getattr(tool, "metadata", {}) or {}),
        )
        self._executor = executor
        self._evidence_schema = EvidenceSchema(
            kind=evidence_kind,
            owner=plugin_id,
            json_schema={"type": "object"},
            description=f"Result evidence for local tool {tool.name}.",
        )
        self._tool_view = ToolView(
            name=tool.name,
            capability_id=capability_id,
            description=tool.description,
            parameters=tool.parameters,
            model_visible=bool(getattr(tool, "model_visible", True)),
            metadata={
                "adapter": "local_tool",
                **dict(getattr(tool, "metadata", {}) or {}),
            },
        )
        self.declaration_fingerprint = _declaration_fingerprint(
            self._capability,
            self._tool_view,
            self._evidence_schema,
        )

    def replace_tool(self, tool: Any) -> None:
        """Replace handler state only after caller verifies declarations match."""
        self.tool = tool
        self._executor.replace_tool(tool)

    def declare_capabilities(self):
        return (self._capability,)

    def get_executors(self):
        return (self._executor,)

    def declare_evidence_schemas(self):
        return (self._evidence_schema,)

    def get_tool_views(self):
        if (
            not self._capability.model_visible
            or self._capability.runtime_only
            or self._capability.specialist_only
        ):
            return ()
        return (self._tool_view,)


class LocalToolRuntimeAdapter:
    """Project local tools into registry-backed runtime declarations."""

    def __init__(self, *, owner_namespace: str = "local_tool") -> None:
        self.owner_namespace = owner_namespace

    def plugin_id_for(self, tool: Any) -> str:
        return f"{_identifier(self.owner_namespace)}_{_identifier(tool.name)}"

    def capability_id_for(self, tool: Any) -> str:
        capability_ids = tuple(getattr(tool, "capability_ids", ()) or ())
        if capability_ids:
            return capability_ids[0]
        return f"agent.local.{_identifier(tool.name)}"

    def plugin_for(
        self,
        tool: Any,
        *,
        plugin_id: str | None = None,
        capability_id: str | None = None,
    ) -> LocalToolPlugin:
        resolved_plugin_id = plugin_id or self.plugin_id_for(tool)
        resolved_capability_id = capability_id or self.capability_id_for(tool)
        output_evidence = tuple(
            getattr(tool, "output_evidence", ())
            or (f"{resolved_capability_id}.result",)
        )
        executor = LocalToolExecutor(
            f"{resolved_plugin_id}.execute",
            resolved_capability_id,
            tool,
            output_evidence[0],
        )
        return LocalToolPlugin(
            tool,
            plugin_id=resolved_plugin_id,
            capability_id=resolved_capability_id,
            executor=executor,
        )

    def register(
        self,
        registry: Any,
        tool: Any,
        *,
        plugin_id: str | None = None,
        capability_id: str | None = None,
    ) -> LocalToolRuntimeRegistration:
        plugin = self.plugin_for(
            tool,
            plugin_id=plugin_id,
            capability_id=capability_id,
        )
        if plugin.manifest.id in registry.plugin_ids:
            existing = registry.get_plugin(plugin.manifest.id)
            if not isinstance(existing, LocalToolPlugin):
                raise ValueError(
                    f"local tool plugin id {plugin.manifest.id!r} is already registered"
                )
            if existing.declaration_fingerprint != plugin.declaration_fingerprint:
                raise ValueError(
                    f"local tool {tool.name!r} is already registered with different "
                    "runtime declarations"
                )
            existing.replace_tool(tool)
            plugin = existing
        else:
            registry.register(plugin)
        capability = plugin.declare_capabilities()[0]
        tool_views = plugin.get_tool_views()
        tool_view = tool_views[0] if tool_views else None
        return LocalToolRuntimeRegistration(
            plugin_id=plugin.manifest.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            evidence_kind=next(iter(capability.output_evidence)),
            tool_view=tool_view,
            capability=capability,
        )


def local_tool_plugin_for(
    tool: Any,
    *,
    plugin_id: str | None = None,
    capability_id: str | None = None,
    owner_namespace: str = "local_tool",
) -> LocalToolPlugin:
    """Return runtime declarations for one local tool."""
    return LocalToolRuntimeAdapter(owner_namespace=owner_namespace).plugin_for(
        tool,
        plugin_id=plugin_id,
        capability_id=capability_id,
    )


def register_local_tool(
    registry: Any,
    tool: Any,
    *,
    owner_namespace: str = "local_tool",
    plugin_id: str | None = None,
    capability_id: str | None = None,
) -> LocalToolRuntimeRegistration:
    """Register one local tool in an extension registry."""
    return LocalToolRuntimeAdapter(owner_namespace=owner_namespace).register(
        registry,
        tool,
        plugin_id=plugin_id,
        capability_id=capability_id,
    )


def _declaration_fingerprint(
    capability: Capability,
    tool_view: ToolView,
    evidence_schema: EvidenceSchema,
) -> str:
    encoded = json.dumps(
        {
            "capability": capability.to_dict(),
            "tool_view": tool_view.to_dict(),
            "evidence_schema": evidence_schema.to_dict(),
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = f"tool_{uuid4().hex[:8]}"
    if not cleaned[0].isalpha():
        cleaned = f"tool_{cleaned}"
    return cleaned
