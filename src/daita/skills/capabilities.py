"""Project progressive skill reads and execute approved skill saves or deletions."""

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
from ..domains.learning import LearningCandidateGuard
from ..llm.models import ModelSensitivity, ToolCall
from ..loop.models import RunInput
from .store import (
    SKILL_DESCRIPTION_MAX_CHARACTERS,
    SKILL_INSTRUCTIONS_MAX_CHARACTERS,
    SkillNotFoundError,
    SkillStore,
    SkillStoreError,
    SkillValidationError,
    validate_skill_name,
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
SKILL_DOMAIN_OWNER_ID = "skills"


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
        skill, current_sha256 = await self._store.read_skill_with_digest(name)
        if skill is None:
            raise SkillNotFoundError(name)
        return ToolOutput(
            kind=SKILL_VIEW_OUTPUT_KIND,
            data={
                "name": skill.name,
                "description": skill.description,
                "instructions": skill.instructions,
                "current_sha256": current_sha256,
            },
        )


class SkillSaveExecutor:
    executor_id = SKILL_SAVE_EXECUTOR_ID

    def __init__(self, store: SkillStore) -> None:
        if not isinstance(store, SkillStore):
            raise TypeError("store must be SkillStore")
        self._store = store

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        name, description, instructions, expected_sha256 = _save_arguments(request)
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
        _validate_replacement_digest(
            name,
            exists=exists,
            current_sha256=document_digest,
            expected_sha256=expected_sha256,
        )
        return _fingerprint(
            name,
            exists,
            document_digest,
            state_digest,
            index_digest,
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        name, description, instructions, _expected_sha256 = _save_arguments(request)
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
                "Load one complete procedural skill and current_sha256; use that digest "
                "as skill_save.expected_sha256 when replacing it."
            ),
            input_schema=_name_input_schema(),
            output_kind=SKILL_VIEW_OUTPUT_KIND,
            output_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "instructions": {"type": "string"},
                    "current_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                },
                "required": [
                    "name",
                    "description",
                    "instructions",
                    "current_sha256",
                ],
                "additionalProperties": False,
            },
            executor_id=view.executor_id,
            access_mode=AccessMode.NONE,
        ),
        Capability(
            id=SKILL_SAVE_CAPABILITY_ID,
            description=(
                "Create/replace complete bounded procedural SKILL.md via the sole "
                "approval card. Use for reusable validated steps with use, verification, "
                "and failure guidance—not one-off results/schema. Description states "
                "when, process/sources, exclusions, and result. Text ends run: call "
                "first. Replace via skill_view current_sha256 as expected_sha256; "
                "blind/stale fails."
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
                    "expected_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                        "minLength": 64,
                        "maxLength": 64,
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
            access_mode=AccessMode.NONE,
            operational_effect=OperationalEffect.CHANGE_ADVISORY_CONTEXT,
        ),
        Capability(
            id=SKILL_DELETE_CAPABILITY_ID,
            description="Delete one existing skill after exact approval.",
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
            access_mode=AccessMode.NONE,
            operational_effect=OperationalEffect.CHANGE_ADVISORY_CONTEXT,
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
                discovery=discovery,
            )
            for name, capability, discovery in zip(
                (
                    SKILL_VIEW_TOOL_NAME,
                    SKILL_SAVE_TOOL_NAME,
                    SKILL_DELETE_TOOL_NAME,
                ),
                capabilities,
                (
                    ToolDiscoveryMetadata(
                        summary="Load one complete user-authorized procedural skill.",
                        when_to_use="Use when an indexed procedure is relevant to the request.",
                        keywords=("skill", "procedure", "guidance", "view"),
                        exposure_class=ToolExposureClass.STANDARD,
                        eager_priority=650,
                    ),
                    ToolDiscoveryMetadata(
                        summary="Create or replace one bounded procedural skill.",
                        when_to_use="Use only for an explicit validated reusable procedure.",
                        keywords=("skill", "procedure", "save", "learn"),
                        exposure_class=ToolExposureClass.DEFERRED,
                        eager_priority=180,
                    ),
                    ToolDiscoveryMetadata(
                        summary="Delete one exact existing procedural skill.",
                        when_to_use="Use only for an explicit exact skill deletion.",
                        keywords=("skill", "procedure", "delete"),
                        exposure_class=ToolExposureClass.DEFERRED,
                        eager_priority=170,
                    ),
                ),
                strict=True,
            )
        ),
    )


class SkillCapabilityDomain:
    """Own fixed skill applicability, validation, and safe failure semantics."""

    domain_owner_id = SKILL_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        learning: LearningCandidateGuard,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("skill declarations have the wrong domain owner")
        if {item.id for item in declarations.capabilities} != {
            SKILL_VIEW_CAPABILITY_ID,
            SKILL_SAVE_CAPABILITY_ID,
            SKILL_DELETE_CAPABILITY_ID,
        }:
            raise ValueError("skill domain requires its exact capabilities")
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
        if capability.operational_effect is not OperationalEffect.NONE:
            self._learning.validate_effect(run.id, call)
        if capability.id == SKILL_VIEW_CAPABILITY_ID:
            name = arguments.get("name")
            try:
                if not isinstance(name, str):
                    raise TypeError("skill name must be text")
                validate_skill_name(name)
            except (TypeError, SkillValidationError) as error:
                raise CapabilityInputError(
                    "skill_invalid_name",
                    "Skill names must match [a-z][a-z0-9-]{0,63}.",
                ) from error
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
        if capability.operational_effect is not OperationalEffect.NONE:
            self._learning.mark_effect_succeeded(run.id)
        if output.sensitivity is not None:
            return output
        return replace(
            output,
            sensitivity=ModelSensitivity.INTERNAL,
            sensitivity_provenance={
                "authority": "skill_domain",
                "capability_id": capability.id,
            },
        )

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        if (
            isinstance(error, CapabilityInputError)
            and error.details.get("name") == "name"
        ):
            return CapabilityFailure(
                "skill_invalid_name",
                "Skill names must match [a-z][a-z0-9-]{0,63}.",
            )
        if isinstance(error, SkillNotFoundError):
            return CapabilityFailure(
                "skill_not_found",
                "The requested skill is not available.",
                {"name": call.arguments.get("name")},
            )
        if isinstance(error, SkillStoreError):
            return CapabilityFailure(
                "skill_unavailable",
                "The requested skill document is unavailable or invalid.",
            )
        return None


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


def _save_arguments(request: ToolExecution) -> tuple[str, str, str, str | None]:
    if request.capability_id != SKILL_SAVE_CAPABILITY_ID:
        raise ValueError("skill save executor received another capability")
    name = request.arguments["name"]
    description = request.arguments["description"]
    instructions = request.arguments["instructions"]
    expected_sha256 = request.arguments.get("expected_sha256")
    if (
        not isinstance(name, str)
        or not isinstance(description, str)
        or not isinstance(instructions, str)
        or (expected_sha256 is not None and not isinstance(expected_sha256, str))
    ):
        raise TypeError("skill save arguments must be text")
    return name, description, instructions, expected_sha256


def _validate_replacement_digest(
    name: str,
    *,
    exists: bool,
    current_sha256: str,
    expected_sha256: str | None,
) -> None:
    if exists and expected_sha256 is None:
        raise CapabilityInputError(
            "skill_expected_sha256_required",
            (
                "Replacing an existing skill requires expected_sha256 from a current "
                "skill_view result."
            ),
            {"name": name},
        )
    if (exists and expected_sha256 != current_sha256) or (
        not exists and expected_sha256 is not None
    ):
        raise CapabilityInputError(
            "skill_stale_replacement",
            "The skill changed or disappeared; load it again with skill_view.",
            {"name": name},
        )


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
    "SKILL_DOMAIN_OWNER_ID",
    "SKILL_SAVE_CAPABILITY_ID",
    "SKILL_SAVE_EXECUTOR_ID",
    "SKILL_SAVE_OUTPUT_KIND",
    "SKILL_SAVE_TOOL_NAME",
    "SKILL_VIEW_CAPABILITY_ID",
    "SKILL_VIEW_EXECUTOR_ID",
    "SKILL_VIEW_OUTPUT_KIND",
    "SKILL_VIEW_TOOL_NAME",
    "SkillDeclarations",
    "SkillCapabilityDomain",
    "SkillDeleteExecutor",
    "SkillSaveExecutor",
    "SkillViewExecutor",
    "skill_declarations",
]
