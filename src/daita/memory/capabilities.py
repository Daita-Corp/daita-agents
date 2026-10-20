"""Declare and execute approval-gated entry edits to agent memory documents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace

from .._json import FrozenJsonObject
from ..capabilities import (
    AccessMode,
    AutomationEligibility,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    OperationalEffect,
    SideEffectExecutor,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from ..capability_runtime import CapabilityFailure, SideEffectPlan
from ..llm.models import ModelSensitivity, ToolCall
from ..loop.models import RunInput
from ..loop.session import RunSession
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
        target, edit = _edit_arguments(request)
        try:
            _, exists, digest, state_digest = await self._store.preflight_edit(
                target, edit
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
        # The runtime rechecked the preflight state under the mutation lock, so
        # this applies the approved edit to exactly the document it validated.
        target, edit = _edit_arguments(request)
        content, _, _, _ = await self._store.preflight_edit(target, edit)
        await self._store.replace_from_tool(
            target, content, sensitivity=request.request_sensitivity
        )
        return ToolOutput(
            kind=MEMORY_SET_OUTPUT_KIND,
            data={"target": target, "replaced": True},
        )


def memory_set_declarations(store: MemoryStore) -> MemoryDeclarations:
    executor: SideEffectExecutor = MemorySetExecutor(store)
    capability = Capability(
        id=MEMORY_SET_CAPABILITY_ID,
        description=(
            "Edit one entry of MEMORY.md or USER.md via the sole approval card; "
            "never resend the whole document. USER.md(target=user)=durable "
            "preferences; MEMORY.md(target=memory)=schema-independent "
            "definitions/conventions; SKILL.md=procedures. action=add appends "
            "content; replace swaps old_text for content; remove deletes old_text. "
            "old_text must match exactly one current passage. Text ends run: call "
            "first. Replace instead of duplicating. Exclude results, "
            "schema/transient values, secrets, permissions, and assumptions."
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
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove"],
                    "minLength": 3,
                    "maxLength": 7,
                },
                "content": {
                    "type": "string",
                    "maxLength": MEMORY_MAX_CHARACTERS,
                },
                "old_text": {
                    "type": "string",
                    "maxLength": MEMORY_MAX_CHARACTERS,
                },
            },
            "required": ["target", "action"],
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
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    return MemoryDeclarations(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name=MEMORY_SET_TOOL_NAME,
                capability_id=capability.id,
                description=capability.description,
                presentation=ToolPresentation(
                    toolbox_id=ToolboxId.KNOWLEDGE,
                    load_mode=ToolLoadMode.ON_DEMAND,
                    text_trust=ToolTextTrust.CODE,
                    summary="Add, replace or remove one memory or user-profile entry.",
                    when_to_use="Use only for explicit durable definitions or preferences.",
                    keywords=("memory", "user", "preference", "remember"),
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
        *,
        validate_schema_independent: (
            Callable[[RunInput, str], Awaitable[None]] | None
        ) = None,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("memory declarations have the wrong domain owner")
        if {item.id for item in declarations.capabilities} != {
            MEMORY_SET_CAPABILITY_ID
        }:
            raise ValueError("memory domain requires its exact capability")
        self._declarations = declarations
        self._validate_schema_independent = validate_schema_independent
        self._views = tuple(declarations.tool_views)
        self._capabilities = {item.id: item for item in declarations.capabilities}

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(
        self,
        run: RunInput,
        session: RunSession | None = None,
    ) -> tuple[str, ...]:
        return tuple(view.name for view in self._views)

    project_session = project

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
        session: RunSession | None = None,
    ) -> FrozenJsonObject:
        del request_sensitivity
        if (
            arguments["target"] == "memory"
            and self._validate_schema_independent is not None
        ):
            await self._validate_schema_independent(
                run, str(arguments.get("content", ""))
            )
        return arguments

    prepare_session_call = prepare_call

    async def prepare_automation_grant(
        self,
        capability: Capability,
        constraints: FrozenJsonObject,
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> FrozenJsonObject:
        raise CapabilityInputError(
            "automation_grant_unsupported",
            "This domain does not admit unattended external effects.",
        )

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
        session: RunSession | None = None,
    ) -> ToolOutput:
        if output.sensitivity is not None:
            return output
        return replace(
            output,
            sensitivity=request_sensitivity,
            sensitivity_provenance={
                "authority": "memory_domain",
                "capability_id": capability.id,
            },
        )

    finalize_session_output = finalize_output

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


def _edit_arguments(request: ToolExecution) -> tuple[str, Callable[[str], str]]:
    if request.capability_id != MEMORY_SET_CAPABILITY_ID:
        raise ValueError("memory executor received another capability")
    arguments = request.arguments
    target = arguments["target"]
    action = arguments["action"]
    content = arguments.get("content")
    old_text = arguments.get("old_text")
    if not isinstance(target, str) or not isinstance(action, str):
        raise TypeError("memory edit arguments must be text")
    if (content is None) != (action == "remove") or (old_text is None) != (
        action == "add"
    ):
        raise CapabilityInputError(
            "memory_invalid_arguments",
            "add takes content; replace takes old_text and content; "
            "remove takes old_text.",
            {"target": target, "action": action},
        )
    return target, lambda current: apply_memory_edit(
        current, action, content=content, old_text=old_text
    )


def apply_memory_edit(
    current: str,
    action: str,
    *,
    content: object = None,
    old_text: object = None,
) -> str:
    """Return ``current`` with one entry added, replaced or removed."""

    if content is not None and (not isinstance(content, str) or not content.strip()):
        raise CapabilityInputError(
            "memory_invalid_content", "content must be non-empty text."
        )
    if action == "add":
        assert isinstance(content, str)
        entry = content.strip()
        if entry in (line.strip() for line in current.splitlines()):
            raise CapabilityInputError(
                "memory_entry_exists",
                "This entry is already saved; use replace to change it.",
            )
        head = current.rstrip("\n")
        return f"{head}\n{entry}" if head else entry
    if not isinstance(old_text, str) or not old_text:
        raise CapabilityInputError(
            "memory_invalid_arguments", "old_text must be non-empty text."
        )
    matches = current.count(old_text)
    if matches != 1:
        raise CapabilityInputError(
            "memory_text_not_found" if matches == 0 else "memory_text_ambiguous",
            "old_text must match exactly one passage of the current document.",
            {"matches": matches},
        )
    start = current.index(old_text)
    end = start + len(old_text)
    if action == "replace":
        assert isinstance(content, str)
        return current[:start] + content.strip() + current[end:]
    # Removing a whole line also removes its line break.
    if (start == 0 or current[start - 1] == "\n") and current[end : end + 1] == "\n":
        end += 1
    return current[:start] + current[end:]


__all__ = [
    "apply_memory_edit",
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
