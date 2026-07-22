"""Fixed declaration for one approval-gated memory replacement."""

from __future__ import annotations

from dataclasses import dataclass

from .._json import FrozenJsonObject
from ..capabilities import (
    AccessMode,
    Capability,
    CapabilityInputError,
    Executor,
    SideEffectExecutor,
    ToolApplicability,
    ToolExecution,
    ToolOutput,
    ToolView,
)
from .store import (
    MEMORY_MAX_CHARACTERS,
    MemoryStore,
    MemoryStoreError,
    MemoryValidationError,
)

MEMORY_SET_CAPABILITY_ID = "memory.set"
MEMORY_SET_EXECUTOR_ID = "memory.set.executor"
MEMORY_SET_OUTPUT_KIND = "memory.replacement"
MEMORY_SET_TOOL_NAME = "memory_set"


@dataclass(frozen=True, slots=True)
class MemoryDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class MemorySetExecutor:
    executor_id = MEMORY_SET_EXECUTOR_ID

    def __init__(self, store: MemoryStore) -> None:
        if not isinstance(store, MemoryStore):
            raise TypeError("store must be MemoryStore")
        self._store = store

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        target, content = _replacement_arguments(request)
        try:
            exists, digest, state_digest = await self._store.preflight_replacement(
                target, content
            )
        except MemoryValidationError as error:
            raise CapabilityInputError(
                "memory_invalid_content",
                str(error),
                {"target": target},
            ) from error
        except MemoryStoreError as error:
            raise CapabilityInputError(
                "memory_unavailable",
                "The selected memory document is unavailable or invalid.",
                {"target": target},
            ) from error
        return FrozenJsonObject.from_mapping(
            {
                "target": target,
                "exists": exists,
                "current_sha256": digest,
                "state_sha256": state_digest,
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        target, content = _replacement_arguments(request)
        await self._store.replace_from_tool(target, content)
        return ToolOutput(
            kind=MEMORY_SET_OUTPUT_KIND,
            data={"target": target, "replaced": True},
        )


def memory_set_declarations(store: MemoryStore) -> MemoryDeclarations:
    executor: SideEffectExecutor = MemorySetExecutor(store)
    capability = Capability(
        id=MEMORY_SET_CAPABILITY_ID,
        description=(
            "Replace one complete bounded advisory memory document after exact "
            "caller approval. Do not store secrets, raw rows, query results, "
            "schemas, transient status, or approval claims."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["memory", "user"],
                    "minLength": 4,
                    "maxLength": 6,
                },
                "content": {
                    "type": "string",
                    "maxLength": MEMORY_MAX_CHARACTERS,
                },
            },
            "required": ["target", "content"],
            "additionalProperties": False,
        },
        output_kind=MEMORY_SET_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["memory", "user"]},
                "replaced": {"type": "boolean"},
            },
            "required": ["target", "replaced"],
            "additionalProperties": False,
        },
        executor_id=executor.executor_id,
        access_mode=AccessMode.WRITE,
        side_effecting=True,
    )
    return MemoryDeclarations(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name=MEMORY_SET_TOOL_NAME,
                capability_id=capability.id,
                description=capability.description,
                applicability=ToolApplicability(minimum_active_sources=0),
            ),
        ),
    )


def _replacement_arguments(request: ToolExecution) -> tuple[str, str]:
    if request.capability_id != MEMORY_SET_CAPABILITY_ID:
        raise ValueError("memory executor received another capability")
    target = request.arguments["target"]
    content = request.arguments["content"]
    if not isinstance(target, str) or not isinstance(content, str):
        raise TypeError("memory replacement arguments must be text")
    return target, content


__all__ = [
    "MEMORY_SET_CAPABILITY_ID",
    "MEMORY_SET_EXECUTOR_ID",
    "MEMORY_SET_OUTPUT_KIND",
    "MEMORY_SET_TOOL_NAME",
    "MemoryDeclarations",
    "MemorySetExecutor",
    "memory_set_declarations",
]
