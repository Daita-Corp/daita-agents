"""Declare and execute approval-gated replacement of agent memory documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from .._json import FrozenJsonObject
from ..capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    OperationalEffect,
    SideEffectExecutor,
    ToolDiscoveryMetadata,
    ToolExecution,
    ToolExposureClass,
    ToolOutput,
    ToolView,
)
from ..capability_runtime import CapabilityFailure, SideEffectPlan
from ..llm.models import ModelSensitivity, ToolCall
from ..loop.models import RunInput
from ..domains.learning import LearningCandidateGuard
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
MEMORY_DOMAIN_OWNER_ID = "memory"


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
            "Replace complete bounded MEMORY.md or USER.md via the sole approval card. "
            "USER.md(target=user)=durable preferences; "
            "MEMORY.md(target=memory)=schema-independent definitions/conventions; "
            "SKILL.md=procedures. Text ends run: call first. Preserve unrelated content; "
            "replace duplicates. Exclude results, schema/transient values, secrets, "
            "permissions, and assumptions."
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
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.CHANGE_ADVISORY_CONTEXT,
    )
    return MemoryDeclarations(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name=MEMORY_SET_TOOL_NAME,
                capability_id=capability.id,
                description=capability.description,
                discovery=ToolDiscoveryMetadata(
                    summary="Replace one bounded advisory memory or user-profile document.",
                    when_to_use="Use only for explicit durable definitions or preferences.",
                    keywords=("memory", "user", "preference", "remember"),
                    exposure_class=ToolExposureClass.DEFERRED,
                    eager_priority=200,
                ),
            ),
        ),
    )


class MemoryCapabilityDomain:
    """Own projection and result semantics for bounded memory replacement."""

    domain_owner_id = MEMORY_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        learning: LearningCandidateGuard,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("memory declarations have the wrong domain owner")
        if {item.id for item in declarations.capabilities} != {
            MEMORY_SET_CAPABILITY_ID
        }:
            raise ValueError("memory domain requires its exact capability")
        self._declarations = declarations
        self._learning = learning
        self._views = tuple(declarations.tool_views)
        self._capabilities = {item.id: item for item in declarations.capabilities}

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        return tuple(
            view.name
            for view in self._views
            if self._learning.allows(
                run.id,
                view.name,
                effectful=(
                    self._capabilities[view.capability_id].operational_effect
                    is not OperationalEffect.NONE
                ),
            )
        )

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        return arguments

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        del request_sensitivity
        self._learning.validate_effect(run.id, call)
        return arguments

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        return SideEffectPlan()

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> ToolOutput:
        del request_sensitivity
        self._learning.mark_effect_succeeded(run.id)
        if output.sensitivity is not None:
            return output
        return replace(
            output,
            sensitivity=ModelSensitivity.INTERNAL,
            sensitivity_provenance={
                "authority": "memory_domain",
                "capability_id": capability.id,
            },
        )

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        if isinstance(error, MemoryStoreError):
            return CapabilityFailure(
                "memory_unavailable",
                "The selected memory document is unavailable or invalid.",
            )
        return None


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
    "MEMORY_DOMAIN_OWNER_ID",
    "MemoryCapabilityDomain",
    "MemoryDeclarations",
    "MemorySetExecutor",
    "memory_set_declarations",
]
