"""Memory helpers for the DB agent loop."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from daita.runtime import Evidence, Operation

from ..analysis import structural_schema_fingerprint
from ..context_projection import policy_summary_from_source
from ..fingerprints import persisted_fingerprint
from ..memory import (
    db_memory_options_from_runtime_metadata,
    db_memory_planning_recall_decision,
)
from ..models import DbIntentKind
from ..planner_protocol import DbLoopState, DbPlannerAction, DbPlannerActionKind
from .actions import _summary_id
from .utils import _float_option, _string_list


def _memory_context_for_state(
    runtime: Any,
    operation: Operation,
    accepted: tuple[Evidence, ...],
    *,
    safety_frame: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    memory_config = db_memory_options_from_runtime_metadata(runtime.config.metadata)
    if not memory_config:
        return {"enabled": False}
    prompt = str(operation.request.get("prompt") or "")
    schema = _latest_schema_payload(accepted)
    schema_fingerprint = structural_schema_fingerprint(schema) if schema else None
    decision = db_memory_planning_recall_decision(
        prompt=prompt,
        intent_kind=_memory_recall_intent_kind(operation),
        schema=schema,
        memory_config=memory_config,
    )
    policy_summary = policy_summary_from_source(_runtime_source(runtime))
    policy_fingerprint = persisted_fingerprint(
        {
            "connector_policy": policy_summary,
            "runtime_policy_ids": sorted(
                f"{getattr(policy, 'owner', '')}:{getattr(policy, 'id', '')}"
                for policy in getattr(runtime.registry, "policies", ())
            ),
            "safety_frame": dict(safety_frame or {}),
        }
    )
    configuration_fingerprint = persisted_fingerprint(memory_config)
    freshness_fingerprint = persisted_fingerprint(
        {
            "default_ttl_days": memory_config.get("default_ttl_days"),
            "guards": ["active", "stale", "expires_at"],
        }
    )
    source_scope = tuple(
        str(item)
        for item in (
            operation.request.get("source_scope")
            or operation.metadata.get("source_scope")
            or ()
        )
    )
    recall_identity = {
        "prompt": prompt,
        "source_identity": memory_config.get("source_identity"),
        "source_scope": list(source_scope),
        "policy_fingerprint": policy_fingerprint,
        "freshness_fingerprint": freshness_fingerprint,
        "configuration_fingerprint": configuration_fingerprint,
    }
    recall_fingerprint = persisted_fingerprint(recall_identity)
    recall_binding = {
        **recall_identity,
        "schema_fingerprint": schema_fingerprint,
        "recall_fingerprint": recall_fingerprint,
    }
    matching_recall = _matching_recall_evidence(
        accepted,
        recall_fingerprint=recall_fingerprint,
    )
    matching_context = _matching_memory_planning_context(
        accepted,
        recall_evidence=matching_recall,
        prompt=prompt,
        schema_fingerprint=schema_fingerprint,
        source_identity=memory_config.get("source_identity"),
        policy_fingerprint=policy_fingerprint,
        freshness_fingerprint=freshness_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        limit=int(memory_config.get("limit") or 3),
        char_budget=int(memory_config.get("char_budget") or 800),
        score_threshold=_float_option(memory_config, "score_threshold", 0.45),
    )
    context = {
        "enabled": bool(memory_config.get("enabled")),
        "source_identity": memory_config.get("source_identity"),
        "retrieval_mode": memory_config.get("retrieval_mode") or "structured",
        "limit": int(memory_config.get("limit") or 3),
        "char_budget": int(memory_config.get("char_budget") or 800),
        "score_threshold": _float_option(memory_config, "score_threshold", 0.45),
        "recall": memory_config.get("recall") or "auto",
        "recall_decision": decision,
        "recall_binding": recall_binding,
        "recall_fingerprint": recall_fingerprint,
        "matching_recall_evidence": (
            _matching_evidence_facts(matching_recall) if matching_recall else None
        ),
        "has_matching_planning_context": matching_context is not None,
    }
    return context


def _runtime_source(runtime: Any) -> Any:
    source = getattr(runtime, "source", None)
    if source is not None:
        return source
    for plugin in getattr(getattr(runtime, "config", None), "plugins", ()) or ():
        if getattr(plugin, "sql_dialect", None) and hasattr(plugin, "query"):
            return plugin
    return None


def _matching_recall_evidence(
    accepted: tuple[Evidence, ...],
    *,
    recall_fingerprint: str,
) -> Evidence | None:
    for evidence in reversed(accepted):
        if evidence.kind != "memory.semantic.recall" or not evidence.accepted:
            continue
        payload = evidence.payload if isinstance(evidence.payload, Mapping) else {}
        binding = payload.get("recall_binding")
        if not isinstance(binding, Mapping):
            continue
        if str(binding.get("recall_fingerprint") or "") == recall_fingerprint:
            return evidence
    return None


def _matching_memory_planning_context(
    accepted: tuple[Evidence, ...],
    *,
    recall_evidence: Evidence | None,
    prompt: str,
    schema_fingerprint: str | None,
    source_identity: Any,
    policy_fingerprint: str,
    freshness_fingerprint: str,
    configuration_fingerprint: str,
    limit: int,
    char_budget: int,
    score_threshold: float,
) -> Evidence | None:
    if recall_evidence is None or not recall_evidence.id:
        return None
    selection_by_task: dict[str, Evidence] = {}
    for evidence in accepted:
        if evidence.kind != "db.memory.selection" or not evidence.accepted:
            continue
        payload = evidence.payload if isinstance(evidence.payload, Mapping) else {}
        if recall_evidence.id not in (payload.get("recall_evidence_refs") or ()):
            continue
        if payload.get("source_identity") != source_identity:
            continue
        if payload.get("schema_fingerprint") != schema_fingerprint:
            continue
        budget = payload.get("budget_usage")
        budget = budget if isinstance(budget, Mapping) else {}
        if int(budget.get("limit") or 0) != max(0, int(limit)):
            continue
        if int(budget.get("char_budget") or 0) != max(0, int(char_budget)):
            continue
        if float(budget.get("score_threshold") or 0.0) != float(score_threshold):
            continue
        freshness = payload.get("freshness")
        freshness = freshness if isinstance(freshness, Mapping) else {}
        valid_until = freshness.get("valid_until")
        if valid_until and _timestamp_is_expired(valid_until):
            continue
        if evidence.task_id:
            selection_by_task[evidence.task_id] = evidence
    if not selection_by_task:
        return None

    for evidence in reversed(accepted):
        if evidence.kind != "planning.context" or not evidence.accepted:
            continue
        if not evidence.task_id or evidence.task_id not in selection_by_task:
            continue
        payload = evidence.payload if isinstance(evidence.payload, Mapping) else {}
        if str(payload.get("prompt") or "") != prompt:
            continue
        if payload.get("schema_fingerprint") != schema_fingerprint:
            continue
        diagnostics = payload.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        binding = diagnostics.get("memory_recall_binding")
        if not isinstance(binding, Mapping):
            continue
        if binding.get("policy_fingerprint") != policy_fingerprint:
            continue
        if binding.get("freshness_fingerprint") != freshness_fingerprint:
            continue
        if binding.get("configuration_fingerprint") != configuration_fingerprint:
            continue
        return evidence
    return None


def _timestamp_is_expired(value: Any) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= datetime.now(timezone.utc)


def _matching_evidence_facts(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "owner": evidence.owner,
        "task_id": evidence.task_id,
        "payload_fingerprint": evidence.metadata.get("payload_fingerprint"),
        "task_input_hash": evidence.metadata.get("task_input_hash"),
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
    if memory_context.get("has_matching_planning_context") is True:
        return False
    return not isinstance(memory_context.get("matching_recall_evidence"), Mapping)


def _memory_recall_task_input(state: DbLoopState) -> dict[str, Any]:
    memory_context = state.memory_context or {}
    decision = memory_context.get("recall_decision")
    decision = decision if isinstance(decision, Mapping) else {}
    limit = int(memory_context.get("limit") or 3)
    task_input = {
        "query": str(
            decision.get("query") or state.normalized_user_request.get("prompt") or ""
        ),
        "category": "db_semantics",
        "limit": max(limit * 3, limit),
        "score_threshold": _float_option(memory_context, "score_threshold", 0.45),
        "retrieval_mode": str(memory_context.get("retrieval_mode") or "structured"),
        "source_identity": memory_context.get("source_identity"),
    }
    recall_binding = memory_context.get("recall_binding")
    if isinstance(recall_binding, Mapping) and recall_binding:
        task_input["recall_binding"] = dict(recall_binding)
    return task_input


def _matching_memory_recall_summary(state: DbLoopState) -> dict[str, Any] | None:
    match = (state.memory_context or {}).get("matching_recall_evidence")
    return dict(match) if isinstance(match, Mapping) else None


def _required_memory_recall_runtime_continuation_action(
    state: DbLoopState,
    *,
    current_action_ids: set[str],
) -> DbPlannerAction | None:
    memory_context = state.memory_context or {}
    decision = memory_context.get("recall_decision")
    if not isinstance(decision, Mapping) or decision.get("recall") is not True:
        return None
    if memory_context.get("has_matching_planning_context") is True:
        return None
    binding = dict(memory_context.get("recall_binding") or {})
    action_id = f"runtime_memory_recall_{persisted_fingerprint(binding)[:12]}"
    if action_id in current_action_ids:
        action_id = (
            f"{action_id}_"
            f"{persisted_fingerprint({'existing_ids': sorted(current_action_ids)})[:8]}"
        )
    action_input: dict[str, Any] = {"memory_recall_binding": binding}
    source_owner = _single_source_owner_for_state(state)
    if source_owner:
        action_input["source_owner"] = source_owner
    return DbPlannerAction(
        action_id=action_id,
        kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
        input=action_input,
        rationale="Runtime continuation for required DB memory recall.",
        metadata={
            "runtime_continuation": True,
            "continuation": "memory.recall.bootstrap",
            "memory_recall_fingerprint": memory_context.get("recall_fingerprint"),
        },
    )


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

    proposal = next(iter(proposals))
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
    action_id = f"runtime_memory_commit_{persisted_fingerprint(seed)[:12]}"
    if action_id not in current_action_ids:
        return action_id
    return f"{action_id}_{persisted_fingerprint({'existing_ids': sorted(current_action_ids)})[:8]}"


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
    return dict(next(iter(summaries))), None
