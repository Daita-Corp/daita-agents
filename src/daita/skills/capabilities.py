"""Fixed declarations for progressive skill reads and approved skill writes."""

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
    SKILL_DESCRIPTION_MAX_CHARACTERS,
    SKILL_INSTRUCTIONS_MAX_CHARACTERS,
    SkillNotFoundError,
    SkillStore,
    SkillStoreError,
    SkillValidationError,
)

SKILL_VIEW_CAPABILITY_ID = "skill.view"
SKILL_VIEW_EXECUTOR_ID = "skill.view.executor"
SKILL_VIEW_OUTPUT_KIND = "skill.document"
SKILL_VIEW_TOOL_NAME = "skill_view"

SKILL_SAVE_CAPABILITY_ID = "skill.save"
SKILL_SAVE_EXECUTOR_ID = "skill.save.executor"
SKILL_SAVE_OUTPUT_KIND = "skill.saved"
SKILL_SAVE_TOOL_NAME = "skill_save"

SKILL_DELETE_CAPABILITY_ID = "skill.delete"
SKILL_DELETE_EXECUTOR_ID = "skill.delete.executor"
SKILL_DELETE_OUTPUT_KIND = "skill.deleted"
SKILL_DELETE_TOOL_NAME = "skill_delete"

_SKILL_NAME_PATTERN = "^[a-z][a-z0-9-]{0,63}$"


@dataclass(frozen=True, slots=True)
class SkillDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class SkillViewExecutor:
    executor_id = SKILL_VIEW_EXECUTOR_ID

    def __init__(self, store: SkillStore) -> None:
        if not isinstance(store, SkillStore):
            raise TypeError("store must be SkillStore")
        self._store = store

    async def execute(self, request: ToolExecution) -> ToolOutput:
        name = request.arguments["name"]
        assert isinstance(name, str)
        skill = await self._store.read_skill(name)
        if skill is None:
            raise SkillNotFoundError(name)
        return ToolOutput(
            kind=SKILL_VIEW_OUTPUT_KIND,
            data={
                "name": skill.name,
                "description": skill.description,
                "instructions": skill.instructions,
            },
        )


class SkillSaveExecutor:
    executor_id = SKILL_SAVE_EXECUTOR_ID

    def __init__(self, store: SkillStore) -> None:
        if not isinstance(store, SkillStore):
            raise TypeError("store must be SkillStore")
        self._store = store

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        name, description, instructions = _save_arguments(request)
        try:
            exists, document_digest, state_digest, index_digest = (
                await self._store.preflight_save(name, description, instructions)
            )
        except SkillValidationError as error:
            raise CapabilityInputError(
                "skill_invalid_document",
                str(error),
                {"name": name},
            ) from error
        except SkillStoreError as error:
            raise CapabilityInputError(
                "skill_unavailable",
                "The skill collection is unavailable or invalid.",
                {"name": name},
            ) from error
        return _fingerprint(
            name,
            exists,
            document_digest,
            state_digest,
            index_digest,
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        name, description, instructions = _save_arguments(request)
        changed = await self._store.save_from_tool(
            name,
            description,
            instructions,
        )
        return ToolOutput(
            kind=SKILL_SAVE_OUTPUT_KIND,
            data={"name": name, "changed": changed},
        )


class SkillDeleteExecutor:
    executor_id = SKILL_DELETE_EXECUTOR_ID

    def __init__(self, store: SkillStore) -> None:
        if not isinstance(store, SkillStore):
            raise TypeError("store must be SkillStore")
        self._store = store

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        name = _delete_arguments(request)
        try:
            exists, document_digest, state_digest, index_digest = (
                await self._store.preflight_delete(name)
            )
        except SkillNotFoundError:
            raise
        except SkillValidationError as error:
            raise CapabilityInputError(
                "skill_invalid_document",
                str(error),
                {"name": name},
            ) from error
        except SkillStoreError as error:
            raise CapabilityInputError(
                "skill_unavailable",
                "The skill collection is unavailable or invalid.",
                {"name": name},
            ) from error
        return _fingerprint(
            name,
            exists,
            document_digest,
            state_digest,
            index_digest,
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        name = _delete_arguments(request)
        deleted = await self._store.delete_from_tool(name)
        if not deleted:
            raise SkillNotFoundError(name)
        return ToolOutput(
            kind=SKILL_DELETE_OUTPUT_KIND,
            data={"name": name, "deleted": True},
        )


def skill_declarations(store: SkillStore) -> SkillDeclarations:
    view = SkillViewExecutor(store)
    save: SideEffectExecutor = SkillSaveExecutor(store)
    delete: SideEffectExecutor = SkillDeleteExecutor(store)
    capabilities = (
        Capability(
            id=SKILL_VIEW_CAPABILITY_ID,
            description=(
                "Load one complete user-authorized procedural skill by its indexed "
                "name."
            ),
            input_schema=_name_input_schema(),
            output_kind=SKILL_VIEW_OUTPUT_KIND,
            output_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "instructions": {"type": "string"},
                },
                "required": ["name", "description", "instructions"],
                "additionalProperties": False,
            },
            executor_id=view.executor_id,
            access_mode=AccessMode.READ,
            side_effecting=False,
        ),
        Capability(
            id=SKILL_SAVE_CAPABILITY_ID,
            description=(
                "Create or replace one bounded reusable procedural skill after exact "
                "caller approval. Prefer improving an existing skill over creating a "
                "near-duplicate."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": _SKILL_NAME_PATTERN,
                        "maxLength": 64,
                    },
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": SKILL_DESCRIPTION_MAX_CHARACTERS,
                    },
                    "instructions": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": SKILL_INSTRUCTIONS_MAX_CHARACTERS,
                    },
                },
                "required": ["name", "description", "instructions"],
                "additionalProperties": False,
            },
            output_kind=SKILL_SAVE_OUTPUT_KIND,
            output_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "changed": {"type": "boolean"},
                },
                "required": ["name", "changed"],
                "additionalProperties": False,
            },
            executor_id=save.executor_id,
            access_mode=AccessMode.WRITE,
            side_effecting=True,
        ),
        Capability(
            id=SKILL_DELETE_CAPABILITY_ID,
            description=(
                "Delete one exact procedural skill after caller approval. The skill "
                "must exist when approval is requested."
            ),
            input_schema=_name_input_schema(),
            output_kind=SKILL_DELETE_OUTPUT_KIND,
            output_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "deleted": {"type": "boolean"},
                },
                "required": ["name", "deleted"],
                "additionalProperties": False,
            },
            executor_id=delete.executor_id,
            access_mode=AccessMode.WRITE,
            side_effecting=True,
        ),
    )
    return SkillDeclarations(
        capabilities=capabilities,
        executors=(view, save, delete),
        tool_views=tuple(
            ToolView(
                name=name,
                capability_id=capability.id,
                description=capability.description,
                applicability=ToolApplicability(minimum_active_sources=0),
            )
            for name, capability in zip(
                (
                    SKILL_VIEW_TOOL_NAME,
                    SKILL_SAVE_TOOL_NAME,
                    SKILL_DELETE_TOOL_NAME,
                ),
                capabilities,
                strict=True,
            )
        ),
    )


def _name_input_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": _SKILL_NAME_PATTERN,
                "maxLength": 64,
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }


def _save_arguments(request: ToolExecution) -> tuple[str, str, str]:
    if request.capability_id != SKILL_SAVE_CAPABILITY_ID:
        raise ValueError("skill save executor received another capability")
    name = request.arguments["name"]
    description = request.arguments["description"]
    instructions = request.arguments["instructions"]
    if (
        not isinstance(name, str)
        or not isinstance(description, str)
        or not isinstance(instructions, str)
    ):
        raise TypeError("skill save arguments must be text")
    return name, description, instructions


def _delete_arguments(request: ToolExecution) -> str:
    if request.capability_id != SKILL_DELETE_CAPABILITY_ID:
        raise ValueError("skill delete executor received another capability")
    name = request.arguments["name"]
    if not isinstance(name, str):
        raise TypeError("skill delete name must be text")
    return name


def _fingerprint(
    name: str,
    exists: bool,
    document_digest: str,
    state_digest: str,
    index_digest: str,
) -> FrozenJsonObject:
    return FrozenJsonObject.from_mapping(
        {
            "name": name,
            "exists": exists,
            "current_sha256": document_digest,
            "state_sha256": state_digest,
            "index_sha256": index_digest,
        }
    )


__all__ = [
    "SKILL_DELETE_CAPABILITY_ID",
    "SKILL_DELETE_EXECUTOR_ID",
    "SKILL_DELETE_OUTPUT_KIND",
    "SKILL_DELETE_TOOL_NAME",
    "SKILL_SAVE_CAPABILITY_ID",
    "SKILL_SAVE_EXECUTOR_ID",
    "SKILL_SAVE_OUTPUT_KIND",
    "SKILL_SAVE_TOOL_NAME",
    "SKILL_VIEW_CAPABILITY_ID",
    "SKILL_VIEW_EXECUTOR_ID",
    "SKILL_VIEW_OUTPUT_KIND",
    "SKILL_VIEW_TOOL_NAME",
    "SkillDeclarations",
    "SkillDeleteExecutor",
    "SkillSaveExecutor",
    "SkillViewExecutor",
    "skill_declarations",
]
