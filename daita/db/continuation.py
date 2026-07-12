"""Continuation resolution for planner actions that reference durable DB evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from .planner_protocol import DbLoopState, DbPlannerAction, DbPlannerActionKind


@dataclass(frozen=True)
class _ContinuationRole:
    role: str
    evidence_kind: str
    action_kinds: frozenset[DbPlannerActionKind]
    role_ref_keys: tuple[str, ...] = ()
    evidence_id_keys: tuple[str, ...] = ()
    implicit_blocking_keys: tuple[str, ...] = ()
    resolved_role_ref_key: str | None = None
    resolved_evidence_id_key: str | None = None
    require_sql: bool = False
    evidence_source: str = "accepted"
    require_valid: bool | None = None
    allow_single_candidate_without_stale_dependency: bool = False
    latest_only: bool = False
    bind_explicit_evidence_id: bool = False
    missing_error: str | None = None
    ambiguous_error: str | None = None


class DbContinuationResolver:
    """Resolve stale planner references to accepted durable evidence roles."""

    _ROLES: tuple[_ContinuationRole, ...] = (
        _ContinuationRole(
            role="latest_accepted_query_plan",
            evidence_kind="query.plan.proposal",
            action_kinds=frozenset({DbPlannerActionKind.EXECUTE_VALIDATED_READ}),
            role_ref_keys=("query_plan_ref",),
            evidence_id_keys=("plan_evidence_id",),
            implicit_blocking_keys=("sql", "plan_evidence_id", "query_plan_ref"),
            resolved_role_ref_key="query_plan_ref",
            require_sql=True,
        ),
        _ContinuationRole(
            role="latest_failed_query_plan_validation",
            evidence_kind="query.plan.validation",
            action_kinds=frozenset({DbPlannerActionKind.REPAIR_QUERY_PLAN}),
            evidence_id_keys=("failure_evidence_id",),
            implicit_blocking_keys=("failure_evidence_id",),
            resolved_evidence_id_key="failure_evidence_id",
            evidence_source="rejected",
            require_valid=False,
            allow_single_candidate_without_stale_dependency=True,
            latest_only=True,
            bind_explicit_evidence_id=True,
            ambiguous_error="ambiguous_repair_failure_evidence",
        ),
        _ContinuationRole(
            role="latest_failed_sql_validation",
            evidence_kind="sql.validation",
            action_kinds=frozenset({DbPlannerActionKind.REPAIR_QUERY_PLAN}),
            evidence_id_keys=("failure_evidence_id",),
            implicit_blocking_keys=("failure_evidence_id",),
            resolved_evidence_id_key="failure_evidence_id",
            evidence_source="rejected",
            require_valid=False,
            allow_single_candidate_without_stale_dependency=True,
            latest_only=True,
            bind_explicit_evidence_id=True,
            missing_error="missing_repair_failure_evidence",
            ambiguous_error="ambiguous_repair_failure_evidence",
        ),
        _ContinuationRole(
            role="latest_accepted_query_plan",
            evidence_kind="query.plan.proposal",
            action_kinds=frozenset({DbPlannerActionKind.REPAIR_QUERY_PLAN}),
            role_ref_keys=("query_plan_ref", "prior_plan_ref"),
            evidence_id_keys=("prior_plan_evidence_id", "plan_evidence_id"),
            implicit_blocking_keys=(
                "query_plan_ref",
                "prior_plan_ref",
                "prior_plan_evidence_id",
                "plan_evidence_id",
            ),
            resolved_evidence_id_key="prior_plan_evidence_id",
            require_sql=True,
            require_valid=True,
            allow_single_candidate_without_stale_dependency=True,
            latest_only=True,
            bind_explicit_evidence_id=True,
            missing_error="missing_repair_prior_plan",
        ),
        _ContinuationRole(
            role="latest_accepted_memory_proposal",
            evidence_kind="db.memory.proposal",
            action_kinds=frozenset({DbPlannerActionKind.COMMIT_MEMORY_UPDATE}),
            role_ref_keys=(
                "memory_proposal_ref",
                "proposal_ref",
                "proposal_evidence_ref",
            ),
            evidence_id_keys=("proposal_evidence_id",),
            implicit_blocking_keys=(
                "memory_proposal_ref",
                "proposal_ref",
                "proposal_evidence_ref",
                "proposal_evidence_id",
            ),
            resolved_evidence_id_key="proposal_evidence_id",
            allow_single_candidate_without_stale_dependency=True,
        ),
    )

    def resolve(
        self,
        action: DbPlannerAction,
        state: DbLoopState,
        current_action_ids: set[str],
    ) -> DbPlannerAction:
        """Return an action normalized to durable evidence when it is safe."""
        resolved_action = action
        for role in self._ROLES:
            if resolved_action.kind not in role.action_kinds:
                continue
            resolved_action = self._resolve_role(
                resolved_action,
                state,
                current_action_ids=current_action_ids,
                role=role,
            )
            if self.blocked_diagnostic(resolved_action) is not None:
                return resolved_action
        return resolved_action

    @staticmethod
    def blocked_diagnostic(action: DbPlannerAction) -> dict[str, Any] | None:
        diagnostic = action.metadata.get("continuation_resolution")
        if isinstance(diagnostic, Mapping) and diagnostic.get("status") == "blocked":
            return dict(diagnostic)
        return None

    def _resolve_role(
        self,
        action: DbPlannerAction,
        state: DbLoopState,
        *,
        current_action_ids: set[str],
        role: _ContinuationRole,
    ) -> DbPlannerAction:
        stale_dependencies = tuple(
            dependency
            for dependency in action.depends_on
            if dependency not in current_action_ids
        )
        explicit_id = self._explicit_evidence_id(action, role)
        if explicit_id is not None:
            matches = tuple(
                candidate
                for candidate in self._candidate_summaries(
                    role,
                    state,
                    latest_only=False,
                )
                if _summary_id(candidate) == explicit_id
            )
            bind_explicit = role.bind_explicit_evidence_id
            if len(matches) == 1 and (stale_dependencies or bind_explicit):
                return self._resolved_action(
                    action,
                    role=role,
                    candidate=matches[0],
                    current_action_ids=current_action_ids,
                    source="explicit_evidence_id",
                    bind_candidate=bind_explicit,
                )
            if len(matches) > 1:
                return self._blocked_action(
                    action,
                    role=role,
                    candidates=matches,
                    stale_dependencies=stale_dependencies,
                    error=(
                        role.ambiguous_error
                        or f"ambiguous_continuation_evidence:{explicit_id}"
                    ),
                )
            return action

        explicit_role = self._explicit_role(action, role)
        if explicit_role is not None:
            if explicit_role != role.role:
                return action
            candidates = self._candidate_summaries(role, state)
            if not candidates:
                if role.missing_error is not None:
                    return self._blocked_action(
                        action,
                        role=role,
                        candidates=(),
                        stale_dependencies=stale_dependencies,
                        error=role.missing_error,
                    )
                return action
            return self._resolved_action(
                action,
                role=role,
                candidate=candidates[-1],
                current_action_ids=current_action_ids,
                source="explicit_role",
                bind_candidate=True,
            )

        if not stale_dependencies and not (
            role.allow_single_candidate_without_stale_dependency
            and not self._has_implicit_blocking_input(action, role)
        ):
            return action
        if self._has_implicit_blocking_input(action, role):
            return action

        candidates = self._candidate_summaries(role, state)
        if not candidates:
            if role.missing_error is not None:
                return self._blocked_action(
                    action,
                    role=role,
                    candidates=(),
                    stale_dependencies=stale_dependencies,
                    error=role.missing_error,
                )
            return action
        if len(candidates) > 1:
            return self._blocked_action(
                action,
                role=role,
                candidates=candidates,
                stale_dependencies=stale_dependencies,
                error=role.ambiguous_error or f"ambiguous_continuation:{role.role}",
            )
        candidate = next(iter(candidates))
        return self._resolved_action(
            action,
            role=role,
            candidate=candidate,
            current_action_ids=current_action_ids,
            source=("stale_dependency" if stale_dependencies else "single_candidate"),
            bind_candidate=True,
        )

    def _candidate_summaries(
        self,
        role: _ContinuationRole,
        state: DbLoopState,
        *,
        latest_only: bool | None = None,
    ) -> tuple[dict[str, Any], ...]:
        candidates: list[dict[str, Any]] = []
        summaries = (
            state.accepted_evidence_summaries
            if role.evidence_source == "accepted"
            else state.rejected_evidence_summaries
        )
        expected_accepted = role.evidence_source == "accepted"
        for summary in summaries:
            if summary.get("kind") != role.evidence_kind:
                continue
            if summary.get("accepted", True) is not expected_accepted:
                continue
            evidence_id = _summary_id(summary)
            if not evidence_id:
                continue
            if (
                role.require_valid is not None
                and summary.get("valid") is not role.require_valid
            ):
                continue
            if role.require_sql:
                if summary.get("valid") is not True:
                    continue
                sql = summary.get("sql")
                if not isinstance(sql, str) or not sql.strip():
                    continue
            candidates.append(dict(summary))
        use_latest = role.latest_only if latest_only is None else latest_only
        if use_latest and candidates:
            return (candidates[-1],)
        return tuple(candidates)

    def _explicit_evidence_id(
        self,
        action: DbPlannerAction,
        role: _ContinuationRole,
    ) -> str | None:
        for key in role.evidence_id_keys:
            if key not in action.input:
                continue
            value = str(action.input.get(key) or "").strip()
            if value and value != role.role:
                return value
        return None

    def _explicit_role(
        self,
        action: DbPlannerAction,
        role: _ContinuationRole,
    ) -> str | None:
        for key in role.role_ref_keys:
            if key in action.input:
                return str(action.input.get(key) or "").strip()
        for key in role.evidence_id_keys:
            value = str(action.input.get(key) or "").strip()
            if value == role.role:
                return value
        return None

    def _has_implicit_blocking_input(
        self,
        action: DbPlannerAction,
        role: _ContinuationRole,
    ) -> bool:
        return any(key in action.input for key in role.implicit_blocking_keys)

    def _resolved_action(
        self,
        action: DbPlannerAction,
        *,
        role: _ContinuationRole,
        candidate: Mapping[str, Any],
        current_action_ids: set[str],
        source: str,
        bind_candidate: bool,
    ) -> DbPlannerAction:
        evidence_id = _summary_id(candidate)
        action_input = dict(action.input)
        if bind_candidate:
            for key in role.role_ref_keys:
                if key != role.resolved_role_ref_key:
                    action_input.pop(key, None)
            if role.resolved_role_ref_key is not None:
                action_input[role.resolved_role_ref_key] = role.role
            for key in role.evidence_id_keys:
                if (
                    key != role.resolved_evidence_id_key
                    or action_input.get(key) == role.role
                ):
                    action_input.pop(key, None)
            if role.resolved_evidence_id_key is not None:
                action_input[role.resolved_evidence_id_key] = evidence_id
        kept_dependencies = tuple(
            dependency
            for dependency in action.depends_on
            if dependency in current_action_ids
        )
        stale_dependencies = tuple(
            dependency
            for dependency in action.depends_on
            if dependency not in current_action_ids
        )
        diagnostic = {
            "status": "resolved",
            "source": source,
            "role": role.role,
            "evidence_kind": role.evidence_kind,
            "evidence_id": evidence_id,
            "candidate_count": 1,
        }
        if stale_dependencies:
            diagnostic["stale_dependency_ids"] = list(stale_dependencies)
        metadata = {
            **action.metadata,
            "continuation_resolution": diagnostic,
        }
        if stale_dependencies:
            metadata.update(
                {
                    "recovered_prior_turn_dependency": True,
                    "dependency_recovery": role.role,
                }
            )
        return replace(
            action,
            input=action_input,
            depends_on=kept_dependencies,
            metadata=metadata,
        )

    def _blocked_action(
        self,
        action: DbPlannerAction,
        *,
        role: _ContinuationRole,
        candidates: tuple[Mapping[str, Any], ...],
        stale_dependencies: tuple[str, ...],
        error: str,
    ) -> DbPlannerAction:
        diagnostic: dict[str, Any] = {
            "status": "blocked",
            "error": error,
            "role": role.role,
            "evidence_kind": role.evidence_kind,
            "candidate_count": len(candidates),
            "candidate_ids": [_summary_id(candidate) for candidate in candidates],
        }
        if stale_dependencies:
            diagnostic["stale_dependency_ids"] = list(stale_dependencies)
        return replace(
            action,
            metadata={
                **action.metadata,
                "continuation_resolution": diagnostic,
            },
        )


def _summary_id(summary: Mapping[str, Any]) -> str:
    return str(summary.get("id") or "").strip()
