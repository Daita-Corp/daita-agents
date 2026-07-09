"""Memory helpers for the DB agent loop."""

from __future__ import annotations

from typing import Any, Mapping

from daita.runtime import Evidence, Operation

from ..memory import (
    db_memory_options_from_runtime_metadata,
    db_memory_planning_recall_decision,
)
from ..models import DbIntentKind
from ..planner_protocol import DbLoopState, DbPlannerAction, DbPlannerActionKind
from .actions import _summary_id
from .summaries import _state_has_accepted_evidence
from .utils import _float_option, _stable_hash, _string_list


def _memory_context_for_state(
    runtime: Any,
    operation: Operation,
    accepted: tuple[Evidence, ...],
) -> dict[str, Any]:
    memory_config = db_memory_options_from_runtime_metadata(runtime.config.metadata)
    if not memory_config:
        return {"enabled": False}
    prompt = str(operation.request.get("prompt") or "")
    schema = _latest_schema_payload(accepted)
    decision = db_memory_planning_recall_decision(
        prompt=prompt,
        intent_kind=_memory_recall_intent_kind(operation),
        schema=schema,
        memory_config=memory_config,
    )
    return {
        "enabled": bool(memory_config.get("enabled")),
        "source_identity": memory_config.get("source_identity"),
        "retrieval_mode": memory_config.get("retrieval_mode") or "structured",
        "limit": int(memory_config.get("limit") or 3),
        "char_budget": int(memory_config.get("char_budget") or 800),
        "score_threshold": _float_option(memory_config, "score_threshold", 0.45),
        "recall": memory_config.get("recall") or "auto",
        "recall_decision": decision,
        "has_recall_evidence": any(
            item.kind == "memory.semantic.recall" and item.accepted for item in accepted
        ),
    }


def _latest_schema_payload(accepted: tuple[Evidence, ...]) -> dict[str, Any]:
    for evidence in reversed(accepted):
        if evidence.kind == "schema.asset_profile" and isinstance(
            evidence.payload, dict
        ):
            return dict(evidence.payload)
    return {}


def _memory_recall_intent_kind(operation: Operation) -> str:
    mode = operation.request.get("mode")
    if isinstance(mode, str) and mode.strip() == "memory.update":
        return "memory.update"
    return "data.query"


def _state_should_recall_memory_for_planning(state: DbLoopState) -> bool:
    memory_context = state.memory_context or {}
    decision = memory_context.get("recall_decision")
    if not isinstance(decision, Mapping) or decision.get("recall") is not True:
        return False
    if memory_context.get("has_recall_evidence") is True:
        return False
    return not _state_has_accepted_evidence(state, "memory.semantic.recall")


def _memory_recall_task_input(state: DbLoopState) -> dict[str, Any]:
    memory_context = state.memory_context or {}
    decision = memory_context.get("recall_decision")
    decision = decision if isinstance(decision, Mapping) else {}
    limit = int(memory_context.get("limit") or 3)
    return {
        "query": str(
            decision.get("query") or state.normalized_user_request.get("prompt") or ""
        ),
        "category": "db_semantics",
        "limit": max(limit * 3, limit),
        "score_threshold": _float_option(memory_context, "score_threshold", 0.45),
        "retrieval_mode": str(memory_context.get("retrieval_mode") or "structured"),
        "source_identity": memory_context.get("source_identity"),
    }


def _memory_update_runtime_continuation_action(
    state: DbLoopState,
    *,
    current_action_ids: set[str],
) -> DbPlannerAction | None:
    if not _state_is_memory_update_operation(state):
        return None
    proposals = _uncommitted_memory_proposal_summaries(state)
    if not proposals:
        return None
    owner = _memory_commit_capability_owner_for_state(state, proposals)
    action_id = _runtime_memory_commit_action_id(proposals, current_action_ids)
    action_input: dict[str, Any] = {}
    if owner:
        action_input["owner"] = owner
    if len(proposals) > 1:
        diagnostic = {
            "status": "blocked",
            "source": "runtime_continuation",
            "error": "ambiguous_continuation:latest_uncommitted_memory_proposal",
            "role": "latest_uncommitted_memory_proposal",
            "evidence_kind": "db.memory.proposal",
            "candidate_count": len(proposals),
            "candidate_ids": [_summary_id(item) for item in proposals],
        }
        return DbPlannerAction(
            action_id=action_id,
            kind=DbPlannerActionKind.COMMIT_MEMORY_UPDATE,
            input=action_input,
            rationale=("Runtime found multiple accepted uncommitted memory proposals."),
            metadata={
                "runtime_continuation": True,
                "continuation_resolution": diagnostic,
            },
        )

    proposal = proposals[0]
    proposal_id = _summary_id(proposal)
    proposal_fingerprint = str(
        proposal.get("proposal_fingerprint")
        or proposal.get("payload_fingerprint")
        or ""
    ).strip()
    action_input["proposal_evidence_id"] = proposal_id
    if proposal_fingerprint:
        action_input["proposal_fingerprint"] = proposal_fingerprint
    diagnostic = {
        "status": "resolved",
        "source": "runtime_continuation",
        "role": "latest_uncommitted_memory_proposal",
        "evidence_kind": "db.memory.proposal",
        "evidence_id": proposal_id,
        "candidate_count": 1,
    }
    return DbPlannerAction(
        action_id=action_id,
        kind=DbPlannerActionKind.COMMIT_MEMORY_UPDATE,
        input=action_input,
        rationale="Runtime continuation for accepted DB memory proposal.",
        metadata={
            "runtime_continuation": True,
            "continuation_resolution": diagnostic,
        },
    )


def _state_is_memory_update_operation(state: DbLoopState) -> bool:
    candidates: list[Any] = [
        state.explicit_mode,
        state.normalized_user_request.get("mode"),
        state.normalized_user_request.get("operation_type"),
    ]
    snapshot = state.latest_compiled_contract_snapshot
    if isinstance(snapshot, Mapping):
        candidates.append(snapshot.get("operation_type"))
        metadata = snapshot.get("metadata")
        if isinstance(metadata, Mapping):
            planner_intent = metadata.get("planner_intent")
            if isinstance(planner_intent, Mapping):
                candidates.append(planner_intent.get("operation_type"))
    for candidate in candidates:
        normalized = str(candidate or "").strip().lower().replace("_", ".")
        if normalized == DbIntentKind.MEMORY_UPDATE.value:
            return True
    return False


def _uncommitted_memory_proposal_summaries(
    state: DbLoopState,
) -> tuple[dict[str, Any], ...]:
    proposals = tuple(
        dict(summary)
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "db.memory.proposal"
        and summary.get("accepted", True) is True
        and _summary_id(summary)
    )
    if not proposals:
        return ()
    committed = _accepted_memory_definition_refs(state)
    uncommitted: list[dict[str, Any]] = []
    for proposal in proposals:
        proposal_id = _summary_id(proposal)
        proposal_fingerprint = str(
            proposal.get("proposal_fingerprint")
            or proposal.get("payload_fingerprint")
            or ""
        ).strip()
        if proposal_id in committed["ids"]:
            continue
        if proposal_fingerprint and proposal_fingerprint in committed["fingerprints"]:
            continue
        uncommitted.append(proposal)
    return tuple(uncommitted)


def _accepted_memory_definition_refs(
    state: DbLoopState,
) -> dict[str, set[str]]:
    proposal_ids: set[str] = set()
    proposal_fingerprints: set[str] = set()
    for summary in state.accepted_evidence_summaries:
        if summary.get("kind") != "db.memory.definition":
            continue
        if summary.get("accepted", True) is not True:
            continue
        proposal_id = str(summary.get("proposal_evidence_id") or "").strip()
        if proposal_id:
            proposal_ids.add(proposal_id)
        proposal_fingerprint = str(summary.get("proposal_fingerprint") or "").strip()
        if proposal_fingerprint:
            proposal_fingerprints.add(proposal_fingerprint)
    return {"ids": proposal_ids, "fingerprints": proposal_fingerprints}


def _single_source_owner_for_state(state: DbLoopState) -> str | None:
    source_scope = _string_list(state.source_scope) or _string_list(
        state.normalized_user_request.get("source_scope")
    )
    unique_scope = tuple(dict.fromkeys(source_scope))
    if len(unique_scope) == 1:
        return unique_scope[0]

    source_capability_owners = tuple(
        dict.fromkeys(
            str(summary.get("owner") or "").strip()
            for summary in state.capability_summaries
            if summary.get("id") in {"db.schema.inspect", "db.column_values.profile"}
            and str(summary.get("owner") or "").strip()
        )
    )
    if len(source_capability_owners) == 1:
        return source_capability_owners[0]
    return None


def _single_capability_owner_for_state(
    state: DbLoopState,
    capability_id: str,
) -> str | None:
    owners = tuple(
        str(summary.get("owner") or "").strip()
        for summary in state.capability_summaries
        if summary.get("id") == capability_id
        and str(summary.get("owner") or "").strip()
    )
    unique = tuple(dict.fromkeys(owners))
    if len(unique) == 1:
        return unique[0]
    return None


def _memory_commit_capability_owner_for_state(
    state: DbLoopState,
    proposals: tuple[dict[str, Any], ...],
) -> str | None:
    proposal_owners = tuple(
        str(proposal.get("owner") or "").strip()
        for proposal in proposals
        if str(proposal.get("owner") or "").strip()
    )
    unique_proposal_owners = tuple(dict.fromkeys(proposal_owners))
    capability_owners = {
        str(summary.get("owner") or "").strip()
        for summary in state.capability_summaries
        if summary.get("id") == "db.memory.commit_update"
        and str(summary.get("owner") or "").strip()
    }
    if (
        len(unique_proposal_owners) == 1
        and unique_proposal_owners[0] in capability_owners
    ):
        return unique_proposal_owners[0]
    return _single_capability_owner_for_state(state, "db.memory.commit_update")


def _runtime_memory_commit_action_id(
    proposals: tuple[dict[str, Any], ...],
    current_action_ids: set[str],
) -> str:
    if len(proposals) == 1:
        seed: Mapping[str, Any] = {
            "proposal_evidence_id": _summary_id(proposals[0]),
            "proposal_fingerprint": proposals[0].get("proposal_fingerprint"),
        }
    else:
        seed = {
            "candidate_ids": [_summary_id(proposal) for proposal in proposals],
        }
    action_id = f"runtime_memory_commit_{_stable_hash(seed)[:12]}"
    if action_id not in current_action_ids:
        return action_id
    return (
        f"{action_id}_{_stable_hash({'existing_ids': sorted(current_action_ids)})[:8]}"
    )


def _resolve_memory_proposal_for_action(
    action: DbPlannerAction,
    state: DbLoopState,
) -> tuple[dict[str, Any] | None, str | None]:
    proposal_id = action.input.get("proposal_evidence_id")
    summaries = tuple(
        summary
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "db.memory.proposal"
        and summary.get("accepted") is True
    )
    if proposal_id is not None:
        proposal_id = str(proposal_id).strip()
        if not proposal_id:
            return None, "missing_proposal_evidence_id"
        matches = tuple(
            summary for summary in summaries if summary.get("id") == proposal_id
        )
        if len(matches) > 1:
            return None, f"ambiguous_memory_proposal:{proposal_id}"
        if not matches:
            return None, f"memory_proposal_not_found:{proposal_id}"
        return dict(matches[0]), None
    if not summaries:
        return None, "missing_accepted_memory_proposal"
    if len(summaries) > 1:
        return None, "ambiguous_memory_proposal"
    return dict(summaries[-1]), None
