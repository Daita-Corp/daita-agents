"""Operation contracts built from deterministic DB safety lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from daita.plugins import ExtensionRegistry
from daita.runtime import AccessMode, Capability
from daita.skills import SkillRuntimeEffects

from .models import DbOperationContract, DbRequest, DbRuntimeConfig
from .safety import DbCapabilityLane, DbSafetyFrame


@dataclass(frozen=True)
class CapabilitySelection:
    """One declared capability selected for a lane contract."""

    capability: Capability
    reason: str


class DbContractBuilder:
    """Build operation contracts from verified safety frames."""

    def __init__(self, registry: ExtensionRegistry, config: DbRuntimeConfig) -> None:
        self.registry = registry
        self.config = config

    def build(
        self,
        request: DbRequest | str,
        safety_frame: DbSafetyFrame,
        *,
        skill_effects: tuple[SkillRuntimeEffects, ...] = (),
    ) -> DbOperationContract:
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        safety_frame_dict = safety_frame.to_dict()
        forbidden = tuple(safety_frame.forbidden_capabilities)
        required = tuple(
            capability
            for capability in safety_frame.required_capabilities
            if capability not in forbidden
        )
        requested_capabilities = _ordered_unique(
            (
                *db_request.requested_capabilities,
                *safety_frame.requested_capabilities,
                *[
                    capability_id
                    for effect in skill_effects
                    for capability_id in effect.requested_capabilities
                ],
            )
        )
        selectable_required = _ordered_unique(
            (
                *required,
                *(
                    capability_id
                    for capability_id in requested_capabilities
                    if capability_id not in forbidden
                ),
            )
        )
        required_selections = self._select_capabilities(selectable_required)
        support_selections = self._select_capabilities(
            tuple(
                capability_id
                for capability_id in _ordered_unique(
                    (
                        *_support_capabilities_for_lanes(safety_frame.granted_lanes),
                        *_support_capabilities_for_selected_capabilities(
                            required_selections
                        ),
                    )
                )
                if capability_id not in forbidden
            )
        )
        selections = _dedupe_selections((*required_selections, *support_selections))
        selected_required_ids = _ordered_unique(
            selection.capability.id for selection in required_selections
        )
        selected_ids = _ordered_unique(
            selection.capability.id for selection in selections
        )
        required_evidence = _ordered_unique(
            (
                *[
                    evidence
                    for selection in required_selections
                    for evidence in selection.capability.output_evidence
                ],
                *[
                    evidence
                    for effect in skill_effects
                    for evidence in effect.required_evidence
                ],
            )
        )
        skill_contract_metadata = _merge_skill_metadata(
            effect.contract_metadata for effect in skill_effects
        )
        verifier_metadata = _merge_skill_metadata(
            effect.verifier_metadata for effect in skill_effects
        )
        synthesis_metadata = _merge_skill_metadata(
            effect.synthesis_metadata for effect in skill_effects
        )
        missing_capabilities = self._missing_capabilities(
            required=(*selectable_required,),
            selected_ids=selected_required_ids,
            forbidden=forbidden,
        )
        operation_type = _operation_type_for_request(
            db_request,
            safety_frame.granted_lanes,
            required_selections,
            safety_frame,
        )
        access = _access_for_contract(safety_frame.granted_lanes, selections)
        policy_ids = _ordered_unique(
            (
                *(
                    ("runtime:approval_required_for_safety_lane",)
                    if safety_frame.approval_required
                    else ()
                ),
                *[
                    policy_id
                    for effect in skill_effects
                    for policy_id in effect.policy_ids
                ],
            )
        )

        metadata = {
            "profile": self.config.profile,
            "granted_lanes": [lane.value for lane in safety_frame.granted_lanes],
            "required_capabilities": list(required),
            "forbidden_capabilities": list(forbidden),
            "blocked_capabilities": self._blocked_capabilities(forbidden),
            "missing_capabilities": list(missing_capabilities),
            "approval_required": safety_frame.approval_required,
            "safety_frame": safety_frame_dict,
            "diagnostics": {
                "safety_frame": safety_frame_dict,
                "requested_capabilities": list(requested_capabilities),
                "skill_requested_capabilities": [
                    capability_id
                    for effect in skill_effects
                    for capability_id in effect.requested_capabilities
                ],
            },
            "planned_operation": {
                "operation_type": operation_type,
                "granted_lanes": [lane.value for lane in safety_frame.granted_lanes],
                "access": access.value,
                "admin": DbCapabilityLane.ADMIN in safety_frame.granted_lanes,
                "destructive": safety_frame.destructive,
                "approval_required": safety_frame.approval_required,
                "source": "safety_contract_builder",
            },
            "selected_capabilities": [
                {
                    "id": selection.capability.id,
                    "owner": selection.capability.owner,
                    "executor": selection.capability.executor,
                    "reason": selection.reason,
                    "access": selection.capability.access.value,
                    "risk": selection.capability.risk.value,
                    "side_effecting": selection.capability.side_effecting,
                }
                for selection in selections
            ],
            "skills": [
                {
                    "skill_id": effect.skill_id,
                    "requested_capabilities": list(effect.requested_capabilities),
                    "required_evidence": list(effect.required_evidence),
                    "policy_ids": list(effect.policy_ids),
                    "tool_view_names": list(effect.tool_view_names),
                }
                for effect in skill_effects
            ],
            "skill_contract_metadata": skill_contract_metadata,
            "skill_verifier_metadata": verifier_metadata,
            "skill_synthesis_metadata": synthesis_metadata,
        }
        return DbOperationContract(
            operation_type=operation_type,
            required_capabilities=tuple(selected_required_ids),
            required_evidence=tuple(required_evidence),
            access=access,
            limits=self.config.limits,
            policy_ids=tuple(policy_ids),
            metadata=metadata,
        )

    def _select_capabilities(
        self,
        required: tuple[str, ...],
    ) -> tuple[CapabilitySelection, ...]:
        selections: list[CapabilitySelection] = []
        capabilities_by_id = _capabilities_by_id(self.registry.capabilities)
        for capability_id in required:
            for capability in capabilities_by_id.get(capability_id, ()):
                selections.append(
                    CapabilitySelection(
                        capability=capability,
                        reason="required_by_safety_lane",
                    )
                )
        return _dedupe_selections(selections)

    def _blocked_capabilities(self, forbidden: tuple[str, ...]) -> list[dict[str, Any]]:
        blocked: list[dict[str, Any]] = []
        capabilities_by_id = _capabilities_by_id(self.registry.capabilities)
        for capability_id in forbidden:
            matches = capabilities_by_id.get(capability_id, ())
            if matches:
                for capability in matches:
                    blocked.append(
                        {
                            "id": capability.id,
                            "owner": capability.owner,
                            "executor": capability.executor,
                            "reason": "forbidden_by_safety_lane",
                        }
                    )
                continue
            blocked.append(
                {
                    "id": capability_id,
                    "owner": None,
                    "executor": None,
                    "reason": "forbidden_by_safety_lane",
                }
            )
        return blocked

    def _missing_capabilities(
        self,
        *,
        required: tuple[str, ...],
        selected_ids: tuple[str, ...],
        forbidden: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            capability_id
            for capability_id in _ordered_unique(required)
            if capability_id not in selected_ids and capability_id not in forbidden
        )


def _operation_type_for_lanes(lanes: tuple[DbCapabilityLane, ...]) -> str:
    if not lanes or lanes == (DbCapabilityLane.NONE,):
        return "db.none"
    if len(lanes) == 1:
        lane = lanes[0]
        return {
            DbCapabilityLane.NONE: "db.none",
            DbCapabilityLane.SCHEMA: "schema.query",
            DbCapabilityLane.MEMORY_ANSWER: "memory.answer",
            DbCapabilityLane.MEMORY_WRITE: "memory.update",
            DbCapabilityLane.READ: "data.query",
            DbCapabilityLane.WRITE_PROPOSE: "write.propose",
            DbCapabilityLane.WRITE_EXECUTE: "write.execute",
            DbCapabilityLane.ADMIN: "admin",
            DbCapabilityLane.MONITOR_READ: "monitor.read",
            DbCapabilityLane.MONITOR_WRITE: "monitor.write",
            DbCapabilityLane.MONITOR_EXECUTE: "monitor.execute",
        }[lane]
    return "db.multi_lane"


def _operation_type_for_request(
    request: DbRequest,
    lanes: tuple[DbCapabilityLane, ...],
    selections: tuple[CapabilitySelection, ...],
    safety_frame: DbSafetyFrame | None = None,
) -> str:
    if request.mode and any(
        request.mode in selection.capability.operation_types for selection in selections
    ):
        return str(request.mode)
    if (
        safety_frame is not None
        and DbCapabilityLane.MEMORY_ANSWER in lanes
        and safety_frame.direct_memory_operation in {"recall", "list", "inspect"}
    ):
        return f"memory.{safety_frame.direct_memory_operation}"
    return _operation_type_for_lanes(lanes)


def _support_capabilities_for_lanes(
    lanes: tuple[DbCapabilityLane, ...],
) -> tuple[str, ...]:
    support: list[str] = []
    lane_set = set(lanes)
    if DbCapabilityLane.SCHEMA in lane_set:
        support.extend(
            (
                "catalog.source.register",
                "db.planning.context.build",
                "catalog.relationship_paths.find",
                "memory.semantic.recall",
                "db.memory.answer_context.build",
            )
        )
    if DbCapabilityLane.READ in lane_set:
        support.extend(
            (
                "db.schema.inspect",
                "catalog.source.register",
                "db.query.plan",
                "db.query.repair",
                "db.query.prepare_read",
                "db.query.plan.validate",
                "db.planning.context.build",
                "catalog.schema.search",
                "catalog.relationship_paths.find",
                "db.column_values.profile",
                "catalog.column_values.register",
                "catalog.column_values.search",
                "catalog.column_value_hints.resolve",
                "memory.semantic.recall",
                "db.memory.answer_context.build",
                "db.analysis.plan",
                "db.analysis.plan.validate",
                "db.analysis.checkpoint",
                "db.analysis.summarize",
                "db.analysis.replan",
            )
        )
    if DbCapabilityLane.MEMORY_WRITE in lane_set:
        support.extend(("memory.semantic.write", "db.schema.inspect"))
    return tuple(_ordered_unique(support))


def _support_capabilities_for_selected_capabilities(
    selections: tuple[CapabilitySelection, ...],
) -> tuple[str, ...]:
    support: list[str] = []
    for selection in selections:
        capability = selection.capability
        if (
            "db" in capability.domains
            and capability.access in {AccessMode.METADATA_READ, AccessMode.READ}
            and not capability.side_effecting
            and capability.id != "db.schema.inspect"
        ):
            support.append("db.schema.inspect")
    return tuple(_ordered_unique(support))


def _access_for_contract(
    lanes: tuple[DbCapabilityLane, ...],
    selections: tuple[CapabilitySelection, ...],
) -> AccessMode:
    lane_access = [_access_for_lane(lane) for lane in lanes]
    selected_access = [selection.capability.access for selection in selections]
    return _max_access((*lane_access, *selected_access, AccessMode.NONE))


def _access_for_lane(lane: DbCapabilityLane) -> AccessMode:
    if lane in {
        DbCapabilityLane.SCHEMA,
        DbCapabilityLane.MEMORY_ANSWER,
        DbCapabilityLane.WRITE_PROPOSE,
        DbCapabilityLane.MONITOR_READ,
    }:
        return AccessMode.METADATA_READ
    if lane is DbCapabilityLane.READ:
        return AccessMode.READ
    if lane in {
        DbCapabilityLane.MEMORY_WRITE,
        DbCapabilityLane.WRITE_EXECUTE,
        DbCapabilityLane.MONITOR_WRITE,
        DbCapabilityLane.MONITOR_EXECUTE,
    }:
        return AccessMode.WRITE
    if lane is DbCapabilityLane.ADMIN:
        return AccessMode.ADMIN
    return AccessMode.NONE


def _max_access(values: Iterable[AccessMode]) -> AccessMode:
    order = {
        AccessMode.NONE: 0,
        AccessMode.METADATA_READ: 1,
        AccessMode.READ: 2,
        AccessMode.WRITE: 3,
        AccessMode.ADMIN: 4,
    }
    return max((AccessMode(value) for value in values), key=lambda value: order[value])


def _merge_skill_metadata(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        for key, item in dict(value).items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(item, dict)
            ):
                merged[key] = {**merged[key], **item}
            else:
                merged[key] = item
    return merged


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return tuple(out)


def _capabilities_by_id(
    capabilities: Iterable[Capability],
) -> dict[str, tuple[Capability, ...]]:
    indexed: dict[str, list[Capability]] = {}
    for capability in capabilities:
        indexed.setdefault(capability.id, []).append(capability)
    return {capability_id: tuple(items) for capability_id, items in indexed.items()}


def _dedupe_selections(
    selections: Iterable[CapabilitySelection],
) -> tuple[CapabilitySelection, ...]:
    seen: set[tuple[str, str]] = set()
    out: list[CapabilitySelection] = []
    for selection in selections:
        key = (selection.capability.id, selection.capability.owner)
        if key in seen:
            continue
        seen.add(key)
        out.append(selection)
    return tuple(out)
