"""
Extension declarations for MemoryPlugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

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
)

from ..manifest import PluginKind, PluginManifest

MEMORY_MANIFEST = PluginManifest(
    id="memory",
    display_name="Memory",
    version="2.0.0",
    kind=PluginKind.DOMAIN_SERVICE,
    domains=frozenset({"db", "agent"}),
    provides=frozenset({"semantic_memory", "facts", "context"}),
)


def memory_capabilities() -> tuple[Capability, ...]:
    common_schema = {"type": "object"}
    return (
        Capability(
            id="memory.semantic.recall",
            owner="memory",
            description="Recall semantically relevant durable memory.",
            domains=frozenset({"db", "agent"}),
            operation_types=frozenset(
                {
                    "data.query",
                    "memory.recall",
                    "memory.list",
                    "memory.inspect",
                    "schema.query",
                    "schema.relationships",
                }
            ),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"memory.semantic.recall"}),
            executor="memory.semantic.recall",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="memory.semantic.write",
            owner="memory",
            description="Write semantic memory for future operations.",
            domains=frozenset({"db", "agent"}),
            operation_types=frozenset({"memory.update"}),
            access=AccessMode.WRITE,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"memory.semantic.write"}),
            executor="memory.semantic.write",
            runtime_only=True,
            side_effecting=True,
        ),
        Capability(
            id="memory.fact.query",
            owner="memory",
            description="Query structured facts extracted from memory.",
            domains=frozenset({"db", "agent"}),
            operation_types=frozenset({"memory.query", "schema.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"memory.fact.query"}),
            executor="memory.fact.query",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="memory.context.render",
            owner="memory",
            description="Render bounded memory context for a runtime audience.",
            domains=frozenset({"db", "agent"}),
            operation_types=frozenset({"context.render"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"memory.context"}),
            executor="memory.context.render",
            runtime_only=True,
            side_effecting=False,
        ),
    )


def memory_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    object_schema = {"type": "object"}
    return (
        EvidenceSchema(
            kind="memory.semantic.recall",
            owner="memory",
            json_schema=object_schema,
            description="Semantic memory recall result.",
        ),
        EvidenceSchema(
            kind="memory.semantic.write",
            owner="memory",
            json_schema=object_schema,
            description="Semantic memory write result.",
        ),
        EvidenceSchema(
            kind="memory.fact.query",
            owner="memory",
            json_schema=object_schema,
            description="Structured memory fact query result.",
        ),
        EvidenceSchema(
            kind="memory.context",
            owner="memory",
            json_schema=object_schema,
            description="Rendered memory context.",
        ),
    )


@dataclass(frozen=True)
class MemoryExecutor:
    """Executor that delegates one task to a MemoryPlugin method."""

    id: str
    capability_ids: frozenset[str]
    evidence_kind: str
    handler: Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        handler_input = dict(task.input)
        handler_input["_runtime_task_metadata"] = dict(task.metadata)
        payload = await self.handler(handler_input)
        return [
            Evidence(
                kind=self.evidence_kind,
                owner="memory",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
            )
        ]


@dataclass(frozen=True)
class MemoryContextProvider:
    """Render MemoryPlugin context for specialized runtimes."""

    plugin: Any
    id: str = "memory.context"
    owner: str = "memory"
    audiences: frozenset[ContextAudience] = frozenset(
        {
            ContextAudience.PRIMARY_MODEL,
            ContextAudience.FINAL_SYNTHESIZER,
            ContextAudience.OPERATION_INSPECTOR,
        }
    )

    async def render(
        self,
        context: Mapping[str, Any],
        audience: ContextAudience,
        token_budget: int,
    ) -> ContextBlock | None:
        if audience not in self.audiences:
            return None
        content = await self.plugin.render_context(
            prompt=str(context.get("prompt") or ""),
            token_budget=token_budget,
        )
        if not content:
            return None
        return ContextBlock(
            id=self.id,
            owner=self.owner,
            audience=audience,
            content=content,
            priority=20,
            metadata={"source": "memory"},
        )
