"""Governance helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from daita.plugins import ExtensionRegistry, PluginKind
from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Capability,
    GovernanceAuditRecord,
    Operation,
    PolicyDecisionTrace,
    PolicyEvaluator,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStore,
    Task,
)
from daita.runtime import Evidence
from daita.runtime import GovernanceResult, PolicyEffect

from ..governance import default_db_policies
from ..models import DbOperationContract, DbRuntimeConfig
from .resume import _db_contract_from_context, _operation_has_run_context
from .tasks.common import _stable_hash
from .tasks.runtime import DbTaskRuntime
from .types import _GovernancePersistence, _MonitorEffectGovernanceDecision


class DbRuntimeGovernanceMixin:
    if TYPE_CHECKING:
        source: Any
        store: RuntimeStore
        tasks: DbTaskRuntime
        config: DbRuntimeConfig
        registry: ExtensionRegistry
        runtime_id: str
        runtime_kind: str

        def _runtime_event(
            self,
            *,
            type: RuntimeEventType,
            operation_id: str,
            message: str,
            task_id: str | None = None,
            task: Task | None = None,
            capability: Capability | None = None,
            payload: dict[str, Any] | None = None,
            policy_id: str | None = None,
            approval_id: str | None = None,
            evidence_id: str | None = None,
        ) -> RuntimeEvent: ...

    async def evaluate_governance_persistence(
        self,
        operation: Operation,
        *,
        task: Task | None = None,
        capability: Capability | None = None,
        stage: str,
    ) -> _GovernancePersistence:
        """Build DB-owned governance facts for kernel task execution."""
        contract = (
            _db_contract_from_context(operation)
            if _operation_has_run_context(operation)
            else None
        )
        return await self._evaluate_governance(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=stage,
        )

    async def evaluate_monitor_effect_governance(
        self,
        operation: Operation,
        *,
        capability: Capability,
        task: Task | None = None,
        intent: dict[str, Any],
        phase: str,
        mutate_approvals: bool = False,
        operation_override: dict[str, Any] | None = None,
    ) -> _MonitorEffectGovernanceDecision:
        """Evaluate monitor side-effect governance without owning approval state."""

        contract = (
            _db_contract_from_context(operation)
            if _operation_has_run_context(operation)
            else None
        )
        monitor_context = {}
        for key in ("monitor_action_context", "monitor_delivery_context"):
            value = operation.metadata.get(key)
            if isinstance(value, dict):
                monitor_context.update(value)
        extra_governance_facts: dict[str, Any] = {
            "monitor_effect": {
                "phase": phase,
                **monitor_context,
                "intent": dict(intent),
                "mutate_approvals": mutate_approvals,
            }
        }
        if operation_override:
            extra_governance_facts["operation_override"] = dict(operation_override)
        persistence = await self._evaluate_governance(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=f"monitor.{phase}",
            extra_governance_facts=extra_governance_facts,
            mutate_approvals=mutate_approvals,
        )
        await self.store.commit_governance_evaluation(
            decisions=persistence.result.decisions,
            audit_record=persistence.audit_record,
            approval_requests=(
                persistence.approvals_to_request if mutate_approvals else ()
            ),
            events=persistence.events,
        )
        if persistence.result.blocked:
            status = "blocked"
            reason = "governance_blocked"
        elif persistence.result.pending_approval:
            status = "blocked"
            reason = "governance_approval_required"
        else:
            status = "allowed"
            reason = None
        return _MonitorEffectGovernanceDecision(
            status=status,
            reason=reason,
            result=persistence.result,
            audit_record=persistence.audit_record,
        )

    async def _evaluate_governance(
        self,
        operation: Operation,
        contract: DbOperationContract | None = None,
        *,
        task: Task | None = None,
        capability: Capability | None = None,
        stage: str,
        extra_governance_facts: dict[str, Any] | None = None,
        mutate_approvals: bool = True,
    ) -> _GovernancePersistence:
        policies = self._active_governance_policies(
            contract,
            capability=capability,
        )
        current_evidence = tuple(await self.store.list_evidence(operation.id))
        current_approvals = tuple(await self.store.list_approval_requests(operation.id))
        authoritative_validation_evidence = (
            await self.tasks.authoritative_validation_evidence(operation, task)
        )
        governance_facts = _governance_fact_envelope(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=stage,
            source=self.source,
            evidence=current_evidence,
            authoritative_validation_evidence=authoritative_validation_evidence,
            approvals=current_approvals,
        )
        if extra_governance_facts:
            governance_facts = {
                **governance_facts,
                **extra_governance_facts,
                "fact_source": {
                    **dict(governance_facts.get("fact_source") or {}),
                    "sources": _ordered_unique(
                        (
                            *(
                                governance_facts.get("fact_source", {}).get("sources")
                                or ()
                            ),
                            "monitor",
                        )
                    ),
                    "monitor": True,
                },
            }
        governance_operation = _operation_for_governance(
            operation,
            task=task,
            capability=capability,
            stage=stage,
            governance_facts=governance_facts,
        )
        governance_contract = _governance_contract(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=stage,
            governance_facts=governance_facts,
        )
        result = PolicyEvaluator(policies).evaluate_operation(
            governance_operation,
            contract=governance_contract,
        )
        if mutate_approvals:
            result, approvals_to_request = await self._reconcile_approval_state(result)
        else:
            approvals_to_request = ()
        audit_record = await self._governance_audit_record(
            operation,
            result,
            policies=policies,
            contract=governance_contract,
            task=task,
            capability=capability,
            stage=stage,
            approvals_to_request=approvals_to_request,
            governance_facts=governance_facts,
        )
        events: list[RuntimeEvent] = []
        for decision in result.decisions:
            events.append(
                self._runtime_event(
                    type=RuntimeEventType.POLICY_DECISION,
                    operation_id=operation.id,
                    task_id=task.id if task is not None else None,
                    task=task,
                    capability=capability,
                    message=(
                        f"Policy {decision.owner}:{decision.policy_id} returned "
                        f"{decision.effect.value}."
                    ),
                    policy_id=decision.policy_id,
                    payload={"decision": decision.to_dict()},
                )
            )
        for approval in approvals_to_request:
            events.append(
                self._runtime_event(
                    type=RuntimeEventType.APPROVAL_REQUESTED,
                    operation_id=operation.id,
                    task_id=task.id if task is not None else None,
                    task=task,
                    capability=capability,
                    message=f"Approval {approval.approval_id} requested.",
                    policy_id=approval.requested_by_policy_id,
                    approval_id=approval.approval_id,
                    payload={"approval": approval.to_dict()},
                )
            )
        return _GovernancePersistence(
            result=result,
            audit_record=audit_record,
            approvals_to_request=approvals_to_request,
            events=tuple(events),
        )

    async def _governance_audit_record(
        self,
        operation: Operation,
        result: GovernanceResult,
        *,
        policies: tuple[Any, ...],
        contract: dict[str, Any],
        task: Task | None,
        capability: Capability | None,
        stage: str,
        approvals_to_request: tuple[ApprovalRequest, ...],
        governance_facts: dict[str, Any],
    ) -> GovernanceAuditRecord:
        audit_id = f"governance-audit-{uuid4()}"
        actor = _actor_context(operation)
        tenant = _tenant_context(operation)
        source_scope = _source_scope_context(operation)
        resource = _resource_context(operation, self.source, capability)
        current_evidence = tuple(await self.store.list_evidence(operation.id))
        current_approvals = tuple(await self.store.list_approval_requests(operation.id))
        evidence_context = {
            "evidence": [_evidence_trace_summary(item) for item in current_evidence],
        }
        approval_context = {
            "approval_statuses": dict(result.metadata.get("approval_statuses") or {}),
            "pending_request_ids": [
                request.approval_id for request in result.approval_requests
            ],
            "new_request_ids": [
                request.approval_id for request in approvals_to_request
            ],
            "existing": [
                _approval_trace_summary(request) for request in current_approvals
            ],
            "requested": [
                _approval_trace_summary(request) for request in approvals_to_request
            ],
        }
        runtime_facts = {
            "runtime_id": self.runtime_id,
            "runtime_kind": self.runtime_kind,
            "stage": stage,
            "operation_type": operation.operation_type,
            "contract": contract,
            "governance_facts": governance_facts,
            "result": {
                "allowed": result.allowed,
                "blocked": result.blocked,
                "pending_approval": result.pending_approval,
            },
            "policies": [_policy_trace_summary(policy) for policy in policies],
            "decision_count": len(result.decisions),
        }
        evaluation_trace = _governance_evaluation_trace(
            result,
            policies=policies,
            runtime_facts=runtime_facts,
        )
        traces = tuple(
            PolicyDecisionTrace(
                trace_id=f"{audit_id}:decision:{index}",
                operation_id=operation.id,
                policy_id=decision.policy_id,
                owner=decision.owner,
                policy_version=decision.policy_version,
                policy_identity=str(decision.policy_identity),
                effect=decision.effect,
                reason=decision.reason,
                stage=stage,
                task_id=task.id if task is not None else None,
                capability_id=capability.id if capability is not None else None,
                approval_ids=_approval_ids_for_decision(
                    decision, result, approvals_to_request
                ),
                evidence_ids=tuple(
                    _ordered_unique(
                        (
                            *(
                                item.id
                                for item in decision.evidence
                                if item.id is not None
                            ),
                            *(decision.metadata.get("validation_evidence_ids") or ()),
                        )
                    )
                ),
                actor=actor,
                tenant=tenant,
                source_scope=source_scope,
                resource=resource,
                runtime_facts={
                    "contract": contract,
                    "governance_facts": governance_facts,
                    "approval_context": approval_context,
                    "evidence_context": evidence_context,
                    "decision_metadata": decision.metadata,
                },
            )
            for index, decision in enumerate(result.decisions, start=1)
        )
        return GovernanceAuditRecord(
            audit_id=audit_id,
            operation_id=operation.id,
            stage=stage,
            allowed=result.allowed,
            blocked=result.blocked,
            pending_approval=result.pending_approval,
            policy_decisions=result.decisions,
            traces=traces,
            task_id=task.id if task is not None else None,
            capability_id=capability.id if capability is not None else None,
            actor=actor,
            tenant=tenant,
            source_scope=source_scope,
            resource=resource,
            operation_context=_operation_trace_context(operation),
            task_context=_task_trace_context(task),
            capability_context=(
                _capability_governance_facts(capability)
                if capability is not None
                else {}
            ),
            approval_context=approval_context,
            evidence_context=evidence_context,
            runtime_facts=runtime_facts,
            evaluation_trace=evaluation_trace,
            metadata={
                "governance_metadata": result.metadata,
                "policy_identities": [
                    decision.policy_identity for decision in result.decisions
                ],
            },
        )

    def _active_governance_policies(
        self,
        contract: DbOperationContract | None,
        *,
        capability: Capability | None = None,
    ) -> tuple[Any, ...]:
        active_policy_ids = set(contract.policy_ids if contract is not None else ())
        policies: list[Any] = [*default_db_policies(), *self.config.policies]
        if (
            capability is not None
            and capability.id == "db.answer.synthesize"
            and capability.access is AccessMode.NONE
            and not capability.side_effecting
        ):
            # Final prose consumes already-governed evidence and never touches
            # connector state, so extension policies stay scoped to DB work.
            return tuple(policies)

        for policy in self.registry.policies:
            identity = f"{policy.owner}:{policy.id}"
            owner_plugin = self.registry.get_plugin(policy.owner)
            owner_manifest = getattr(owner_plugin, "manifest", None)
            is_skill_policy = (
                owner_manifest is not None
                and getattr(owner_manifest, "kind", None) is PluginKind.SKILL
            )
            if not is_skill_policy or identity in active_policy_ids:
                policies.append(policy)

        return tuple(policies)

    async def _reconcile_approval_state(
        self, result: GovernanceResult
    ) -> tuple[GovernanceResult, tuple[ApprovalRequest, ...]]:
        if not result.approval_requests:
            return result, ()
        existing_by_id = {
            approval.approval_id: approval
            for approval in await self.store.list_approval_requests()
        }
        pending_requests: list[ApprovalRequest] = []
        approvals_to_request: list[ApprovalRequest] = []
        approval_statuses: dict[str, str] = {}
        terminal_blocking_statuses: dict[str, str] = {}

        for approval in result.approval_requests:
            existing = existing_by_id.get(approval.approval_id)
            if existing is None:
                pending_requests.append(approval)
                approvals_to_request.append(approval)
                approval_statuses[approval.approval_id] = ApprovalStatus.PENDING.value
                continue
            approval_statuses[approval.approval_id] = existing.status.value
            if existing.status is ApprovalStatus.APPROVED:
                continue
            if existing.status is ApprovalStatus.PENDING:
                pending_requests.append(existing)
                continue
            terminal_blocking_statuses[existing.approval_id] = existing.status.value

        blocked = result.blocked or bool(terminal_blocking_statuses)
        pending_approval = bool(pending_requests)
        metadata = {
            **result.metadata,
            "approval_statuses": approval_statuses,
        }
        if terminal_blocking_statuses:
            metadata["terminal_approval_statuses"] = terminal_blocking_statuses
        return (
            GovernanceResult(
                allowed=not blocked and not pending_approval,
                blocked=blocked,
                pending_approval=pending_approval,
                decisions=result.decisions,
                approval_requests=tuple(pending_requests),
                modified_contract=result.modified_contract,
                metadata=metadata,
            ),
            tuple(approvals_to_request),
        )


def _approval_governance_facts(
    approvals: tuple[ApprovalRequest, ...],
) -> dict[str, Any]:
    return {
        "ids": [approval.approval_id for approval in approvals],
        "pending_ids": [
            approval.approval_id
            for approval in approvals
            if approval.status is ApprovalStatus.PENDING
        ],
        "approved_ids": [
            approval.approval_id
            for approval in approvals
            if approval.status is ApprovalStatus.APPROVED
        ],
        "statuses": {
            approval.approval_id: approval.status.value for approval in approvals
        },
        "policy_ids": sorted(
            {
                approval.requested_by_policy_id
                for approval in approvals
                if approval.requested_by_policy_id
            }
        ),
    }


def _sql_validation_governance_facts(
    evidence: tuple[Evidence, ...],
) -> dict[str, Any]:
    statements: list[dict[str, Any]] = []
    for item in evidence:
        if item.kind != "sql.validation" or not item.accepted:
            continue
        payload = item.payload
        raw_facts = payload.get("statement_facts")
        facts = raw_facts if isinstance(raw_facts, dict) else {}
        statement_type = str(
            facts.get("statement_type") or payload.get("statement_type") or ""
        ).upper()
        mutating = _statement_classes(
            facts.get("mutating_statement_classes")
            or facts.get("mutating_statement_types")
            or payload.get("mutating_statement_classes")
            or payload.get("mutating_statement_types")
            or ()
        )
        destructive = _statement_classes(
            facts.get("destructive_statement_classes")
            or payload.get("destructive_statement_classes")
            or ()
        )
        admin = _statement_classes(
            facts.get("admin_statement_classes")
            or payload.get("admin_statement_classes")
            or ()
        )
        if statement_type:
            if statement_type in {
                "DELETE",
                "DROP",
                "ALTER",
                "TRUNCATETABLE",
                "TRUNCATE",
            }:
                destructive = _ordered_unique((*destructive, statement_type))
            if statement_type in {
                "CREATE",
                "DROP",
                "ALTER",
                "TRUNCATETABLE",
                "TRUNCATE",
            }:
                admin = _ordered_unique((*admin, statement_type))
        target_resources = _safe_string_list(
            facts.get("target_resources")
            or payload.get("target_resources")
            or payload.get("tables")
            or payload.get("referenced_tables")
            or ()
        )
        statements.append(
            {
                "evidence_id": item.id,
                "task_id": item.task_id,
                "owner": item.owner,
                "valid": bool(payload.get("valid") or payload.get("ok")),
                "statement_type": statement_type.lower() if statement_type else None,
                "statement_count": facts.get("statement_count")
                or payload.get("statement_count"),
                "is_read": facts.get("is_read", payload.get("is_read")),
                "mutating_statement_classes": list(mutating),
                "destructive_statement_classes": list(destructive),
                "admin_statement_classes": list(admin),
                "target_resources": list(target_resources),
                "guardrail_result": facts.get("guardrail_result")
                or payload.get("guardrail_result")
                or ("passed" if payload.get("valid") or payload.get("ok") else None),
                "sql_fingerprint": facts.get("sql_fingerprint")
                or payload.get("sql_fingerprint"),
            }
        )
    return {
        "evidence_ids": [
            item["evidence_id"] for item in statements if item["evidence_id"]
        ],
        "task_ids": [item["task_id"] for item in statements if item["task_id"]],
        "statement_types": _ordered_unique(
            item["statement_type"] for item in statements if item["statement_type"]
        ),
        "mutating_statement_classes": _ordered_unique(
            cls for item in statements for cls in item["mutating_statement_classes"]
        ),
        "destructive_statement_classes": _ordered_unique(
            cls for item in statements for cls in item["destructive_statement_classes"]
        ),
        "admin_statement_classes": _ordered_unique(
            cls for item in statements for cls in item["admin_statement_classes"]
        ),
        "target_resources": _ordered_unique(
            resource for item in statements for resource in item["target_resources"]
        ),
        "guardrail_results": _ordered_unique(
            item["guardrail_result"] for item in statements if item["guardrail_result"]
        ),
        "sql_fingerprints": _ordered_unique(
            item["sql_fingerprint"] for item in statements if item["sql_fingerprint"]
        ),
        "statements": statements,
    }


def _statement_classes(values: Any) -> tuple[str, ...]:
    return tuple(
        str(value).upper() for value in _safe_string_list(values) if str(value).strip()
    )


def _ordered_unique(values: Any) -> tuple[Any, ...]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _safe_string_list(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, (list, tuple, set, frozenset)):
        return tuple(str(value) for value in values if value is not None)
    return (str(values),)


def _governance_policy_block_reason(governance: GovernanceResult) -> str | None:
    for decision in governance.decisions:
        if decision.effect is PolicyEffect.DENY:
            return decision.policy_id
    if governance.blocked:
        return "governance_blocked"
    if governance.pending_approval:
        return "governance_approval_required"
    return None


def _evidence_trace_summary(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "operation_id": evidence.operation_id,
        "task_id": evidence.task_id,
        "accepted": evidence.accepted,
        "schema_version": evidence.schema_version,
        "metadata": evidence.metadata,
        "payload_keys": sorted(str(key) for key in evidence.payload),
    }


def _approval_trace_summary(request: ApprovalRequest) -> dict[str, Any]:
    return {
        "approval_id": request.approval_id,
        "operation_id": request.operation_id,
        "status": request.status.value,
        "requested_by_policy_id": request.requested_by_policy_id,
        "owner": request.owner,
        "risk": request.risk.value,
        "evidence_ids": list(request.evidence_ids),
        "metadata": _metadata_summary(request.metadata),
        "proposed_action": _approval_action_summary(request.proposed_action),
    }


def _policy_trace_summary(policy: Any) -> dict[str, Any]:
    owner = str(getattr(policy, "owner", "runtime"))
    policy_id = str(getattr(policy, "id", "unknown"))
    version = str(
        getattr(policy, "policy_version", None)
        or getattr(policy, "version", None)
        or "1"
    )
    return {
        "owner": owner,
        "policy_id": policy_id,
        "policy_version": version,
        "policy_identity": f"{owner}:{policy_id}@{version}",
        "class": type(policy).__name__,
    }


def _governance_evaluation_trace(
    result: GovernanceResult,
    *,
    policies: tuple[Any, ...],
    runtime_facts: dict[str, Any],
) -> dict[str, Any]:
    if result.blocked:
        effect = "deny"
        reason = "At least one policy denied execution."
    elif result.pending_approval:
        effect = "require_approval"
        reason = "At least one policy required approval before execution."
    elif any(decision.effect.value == "modify" for decision in result.decisions):
        effect = "modify"
        reason = "Policy modifications were applied and execution was allowed."
    elif any(decision.effect.value == "warn" for decision in result.decisions):
        effect = "warn"
        reason = "Policy warnings were recorded and execution was allowed."
    else:
        effect = "allow"
        reason = "No applicable policy denied, modified, warned, or required approval."
    return {
        "effect": effect,
        "reason": reason,
        "policy_count": result.metadata.get("policy_count", len(policies)),
        "applicable_policy_count": result.metadata.get("applicable_policy_count"),
        "evaluated_policy_identities": [
            item["policy_identity"] for item in runtime_facts.get("policies", ())
        ],
        "decision_policy_identities": [
            decision.policy_identity for decision in result.decisions
        ],
        "runtime_facts": {
            "runtime_id": runtime_facts.get("runtime_id"),
            "runtime_kind": runtime_facts.get("runtime_kind"),
            "stage": runtime_facts.get("stage"),
            "operation_type": runtime_facts.get("operation_type"),
            "decision_count": runtime_facts.get("decision_count"),
            "result": runtime_facts.get("result"),
        },
    }


def _request_summary(request: dict[str, Any]) -> dict[str, Any]:
    prompt = request.get("prompt")
    raw_input_payload = request.get("input")
    input_payload = raw_input_payload if isinstance(raw_input_payload, dict) else {}
    raw_metadata = request.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    return {
        "has_prompt": bool(prompt),
        "prompt_hash": _stable_hash(prompt) if prompt else None,
        "user_id": request.get("user_id"),
        "session_id": request.get("session_id"),
        "source_scope": list(_source_scope_from_value(request.get("source_scope"))),
        "mode": request.get("mode"),
        "requested_capabilities": list(request.get("requested_capabilities") or ()),
        "constraint_keys": sorted(
            str(key) for key in dict(request.get("constraints") or {})
        ),
        "metadata_keys": sorted(str(key) for key in metadata),
        "input_keys": sorted(str(key) for key in input_payload),
        "capability_id": request.get("capability_id"),
        "capability_owner": request.get("capability_owner"),
        "governance_stage": request.get("governance_stage"),
    }


def _task_input_summary(input: dict[str, Any]) -> dict[str, Any]:
    sql = input.get("sql")
    prompt = input.get("prompt")
    query = input.get("query")
    return {
        "keys": sorted(str(key) for key in input),
        "input_hash": input.get("input_hash") or _stable_hash(input),
        "sql_hash": _stable_hash(sql) if sql else None,
        "prompt_hash": _stable_hash(prompt) if prompt else None,
        "query_hash": _stable_hash(query) if query else None,
        "sql_ref": input.get("sql_ref"),
        "validated_evidence_id": input.get("validated_evidence_id"),
        "operation": input.get("operation"),
    }


def _approval_action_summary(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_type": action.get("operation_type"),
        "approval": action.get("approval"),
        "keys": sorted(str(key) for key in action),
        "request": (
            _request_summary(action["request"])
            if isinstance(action.get("request"), dict)
            else None
        ),
        "contract_keys": sorted(str(key) for key in dict(action.get("contract") or {})),
    }


def _metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    safe_values = {}
    for key in ("runtime_id", "intent_kind", "access", "governance_stage"):
        if key in metadata:
            safe_values[key] = metadata[key]
    return {
        **safe_values,
        "keys": sorted(str(key) for key in metadata),
    }


def _actor_context(operation: Operation) -> dict[str, Any]:
    request = operation.request
    metadata = (
        request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
    )
    input_payload = (
        request.get("input") if isinstance(request.get("input"), dict) else {}
    )
    actor = _dict_without_none(
        {
            "user_id": request.get("user_id")
            or input_payload.get("user_id")
            or metadata.get("user_id"),
            "session_id": request.get("session_id") or metadata.get("session_id"),
            "actor_id": request.get("actor_id")
            or input_payload.get("actor_id")
            or metadata.get("actor_id"),
            "actor_type": request.get("actor_type")
            or input_payload.get("actor_type")
            or metadata.get("actor_type"),
        }
    )
    nested_actor = metadata.get("actor") or input_payload.get("actor")
    if isinstance(nested_actor, dict):
        actor.update(nested_actor)
    return actor


def _tenant_context(operation: Operation) -> dict[str, Any]:
    request = operation.request
    metadata = (
        request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
    )
    input_payload = (
        request.get("input") if isinstance(request.get("input"), dict) else {}
    )
    tenant = _dict_without_none(
        {
            "tenant_id": request.get("tenant_id")
            or input_payload.get("tenant_id")
            or metadata.get("tenant_id")
            or operation.metadata.get("tenant_id"),
            "workspace_id": request.get("workspace_id")
            or input_payload.get("workspace_id")
            or metadata.get("workspace_id")
            or operation.metadata.get("workspace_id"),
        }
    )
    nested_tenant = metadata.get("tenant") or input_payload.get("tenant")
    if isinstance(nested_tenant, dict):
        tenant.update(nested_tenant)
    return tenant


def _source_scope_context(operation: Operation) -> tuple[str, ...]:
    value = operation.request.get("source_scope")
    if value is None and isinstance(operation.request.get("metadata"), dict):
        value = operation.request["metadata"].get("source_scope")
    if value is None:
        value = operation.metadata.get("source_scope")
    return _source_scope_from_value(value)


def _source_scope_from_value(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if item is not None)
    return ()


def _resource_context(
    operation: Operation,
    source: Any,
    capability: Capability | None,
) -> dict[str, Any]:
    resource = {
        "source_type": type(source).__name__ if source is not None else "none",
        "source_repr": _safe_source_repr(source),
        "operation_type": operation.operation_type,
    }
    if capability is not None:
        resource["capability"] = {
            "id": capability.id,
            "owner": capability.owner,
            "access": capability.access.value,
            "risk": capability.risk.value,
        }
    return resource


def _operation_trace_context(operation: Operation) -> dict[str, Any]:
    return {
        "id": operation.id,
        "operation_type": operation.operation_type,
        "status": operation.status.value,
        "required_evidence": sorted(operation.required_evidence),
        "request": _request_summary(operation.request),
        "metadata": _metadata_summary(operation.metadata),
    }


def _task_trace_context(task: Task | None) -> dict[str, Any]:
    if task is None:
        return {}
    return {
        "id": task.id,
        "operation_id": task.operation_id,
        "capability_id": task.capability_id,
        "executor_id": task.executor_id,
        "status": task.status.value,
        "input": _task_input_summary(task.input),
        "required_evidence": sorted(task.required_evidence),
        "dependencies": [dependency.to_dict() for dependency in task.dependencies],
        "metadata": _metadata_summary(task.metadata),
    }


def _approval_ids_for_decision(
    decision: Any,
    result: GovernanceResult,
    approvals_to_request: tuple[ApprovalRequest, ...],
) -> tuple[str, ...]:
    ids: list[str] = []
    for request in (*result.approval_requests, *approvals_to_request):
        if (
            request.requested_by_policy_id == decision.policy_id
            and request.owner == decision.owner
            and request.approval_id not in ids
        ):
            ids.append(request.approval_id)
    for approval_id in result.metadata.get("approval_statuses") or {}:
        if (
            f":{decision.policy_id}:" in str(approval_id)
            and str(approval_id) not in ids
        ):
            ids.append(str(approval_id))
    return tuple(ids)


def _dict_without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _governance_fact_envelope(
    operation: Operation,
    *,
    contract: DbOperationContract | None,
    task: Task | None,
    capability: Capability | None,
    stage: str,
    source: Any,
    evidence: tuple[Evidence, ...],
    authoritative_validation_evidence: tuple[Evidence, ...],
    approvals: tuple[ApprovalRequest, ...],
) -> dict[str, Any]:
    planned = _planned_governance_facts(operation, contract, capability)
    validation_context = _sql_validation_governance_facts(evidence)
    authoritative_validation = _sql_validation_governance_facts(
        authoritative_validation_evidence
    )
    task_facts = _task_governance_facts(task)
    approval_facts = _approval_governance_facts(approvals)
    sources = ["runtime"]
    if planned.get("has_planned_facts"):
        sources.append("planning")
    if validation_context.get("evidence_ids"):
        sources.append("sql_validation")
    if authoritative_validation.get("evidence_ids"):
        sources.append("task_dependency")
    if capability is not None:
        sources.append("capability")
    if task is not None:
        sources.append("task")
    if approvals:
        sources.append("approval_store")
    return {
        "version": "15.10",
        "fact_source": {
            "sources": sources,
            "planned": planned.get("has_planned_facts", False),
            "validated": bool(validation_context.get("evidence_ids")),
            "authoritative_validation": bool(
                authoritative_validation.get("evidence_ids")
            ),
            "task": task is not None,
            "capability": capability is not None,
            "approvals": bool(approvals),
        },
        "stage": stage,
        "authoritative": {
            "source": _authoritative_fact_source(
                stage=stage,
                task=task,
                authoritative_validation=authoritative_validation,
                planned=planned,
            ),
            "operation": planned,
            "capability": (
                _capability_governance_facts(capability)
                if capability is not None
                else {}
            ),
            "task": task_facts,
            "validation": authoritative_validation,
        },
        "context": {
            "validation": {
                "operation_wide": validation_context,
            },
        },
        "operation": planned,
        "capability": (
            _capability_governance_facts(capability) if capability is not None else {}
        ),
        "task": task_facts,
        "validation": authoritative_validation,
        "approvals": approval_facts,
        "actor": _actor_context(operation),
        "tenant": _tenant_context(operation),
        "source_scope": list(_source_scope_context(operation)),
        "resource": _resource_context(operation, source, capability),
    }


def _planned_governance_facts(
    operation: Operation,
    contract: DbOperationContract | None,
    capability: Capability | None,
) -> dict[str, Any]:
    contract_metadata = contract.metadata if contract is not None else {}
    planned = dict(contract_metadata.get("planned_operation") or {})
    access = (
        contract.access.value
        if contract is not None
        else operation.metadata.get("access")
    )
    if access is None and capability is not None:
        access = capability.access.value
    selected = contract_metadata.get("selected_capabilities") or ()
    side_effecting = (
        bool(capability.side_effecting) if capability is not None else False
    )
    side_effecting = side_effecting or any(
        bool(item.get("side_effecting")) for item in selected if isinstance(item, dict)
    )
    operation_type = (
        contract.operation_type if contract is not None else operation.operation_type
    )
    admin = bool(planned.get("admin")) or access == AccessMode.ADMIN.value
    destructive = bool(planned.get("destructive"))
    write_or_admin = (
        operation_type in {"write.propose", "write.execute", "admin"}
        or access in {AccessMode.WRITE.value, AccessMode.ADMIN.value}
        or side_effecting
    )
    return {
        **planned,
        "has_planned_facts": bool(planned or contract is not None or capability),
        "operation_id": operation.id,
        "operation_type": operation_type,
        "access": access,
        "intent_kind": operation.metadata.get("intent_kind")
        or planned.get("intent_kind"),
        "required_evidence": sorted(operation.required_evidence),
        "capability_ids": (
            list(contract.required_capabilities)
            if contract is not None
            else ([capability.id] if capability is not None else [])
        ),
        "admin": admin,
        "destructive": destructive,
        "write_or_admin_context": write_or_admin,
        "side_effecting": side_effecting,
    }


def _authoritative_fact_source(
    *,
    stage: str,
    task: Task | None,
    authoritative_validation: dict[str, Any],
    planned: dict[str, Any],
) -> str:
    if task is not None and authoritative_validation.get("evidence_ids"):
        return "task_dependency"
    if planned.get("has_planned_facts"):
        return "planning"
    return stage


def _task_governance_facts(task: Task | None) -> dict[str, Any]:
    if task is None:
        return {}
    return {
        "id": task.id,
        "operation_id": task.operation_id,
        "capability_id": task.capability_id,
        "executor_id": task.executor_id,
        "status": task.status.value,
        "required_evidence": sorted(task.required_evidence),
        "input": _task_input_summary(task.input),
        "dependencies": [
            {
                "kind": dependency.kind.value,
                "evidence_kind": dependency.evidence_kind,
                "producer_task_id": dependency.producer_task_id,
                "producer_capability_id": dependency.producer_capability_id,
                "producer_executor_id": dependency.producer_executor_id,
                "approval_policy_id": dependency.approval_policy_id,
                "approval_name": dependency.approval_name,
                "approval_status": (
                    dependency.approval_status.value
                    if dependency.approval_status is not None
                    else None
                ),
                "payload_fingerprint": dependency.payload_fingerprint,
                "evidence_payload_keys": sorted(
                    str(key) for key in dependency.evidence_payload
                ),
            }
            for dependency in task.dependencies
        ],
        "metadata": _metadata_summary(task.metadata),
    }


def _operation_for_governance(
    operation: Operation,
    *,
    task: Task | None,
    capability: Capability | None,
    stage: str,
    governance_facts: dict[str, Any],
) -> Operation:
    request = dict(operation.request)
    request["governance_stage"] = stage
    request["governance_facts"] = governance_facts
    operation_override = _governance_operation_override(governance_facts)
    override_access = operation_override.get("access")
    if override_access:
        request["access"] = str(override_access)
        request_metadata = request.get("metadata")
        if isinstance(request_metadata, dict):
            request["metadata"] = {
                **request_metadata,
                "access": str(override_access),
            }
    if task is not None:
        request["task"] = {
            "id": task.id,
            "capability_id": task.capability_id,
            "executor_id": task.executor_id,
            "input": _task_input_summary(task.input),
            "metadata": task.metadata,
        }
    if capability is not None:
        request["capability"] = _capability_governance_facts(capability)
    metadata = {
        **operation.metadata,
        "governance_stage": stage,
    }
    if override_access:
        metadata["access"] = str(override_access)
    if task is not None:
        metadata["task_id"] = task.id
    if capability is not None:
        metadata["capability_id"] = capability.id
        metadata["capability_owner"] = capability.owner
    operation_type = str(
        operation_override.get("operation_type") or operation.operation_type
    )
    return replace(
        operation,
        operation_type=operation_type,
        request=request,
        metadata=metadata,
    )


def _governance_contract(
    operation: Operation,
    *,
    contract: DbOperationContract | None,
    task: Task | None,
    capability: Capability | None,
    stage: str,
    governance_facts: dict[str, Any],
) -> dict[str, Any]:
    if contract is not None:
        shaped: dict[str, Any] = {
            "operation_type": contract.operation_type,
            "required_capabilities": list(contract.required_capabilities),
            "required_evidence": list(contract.required_evidence),
            "access": contract.access.value,
            "policy_ids": list(contract.policy_ids),
            "metadata": contract.metadata,
        }
    else:
        shaped = {
            "operation_type": operation.operation_type,
            "required_capabilities": [],
            "required_evidence": sorted(operation.required_evidence),
            "metadata": operation.metadata,
        }
    shaped["governance_stage"] = stage
    shaped["governance_facts"] = governance_facts
    operation_override = _governance_operation_override(governance_facts)
    if operation_override.get("operation_type"):
        shaped["operation_type"] = str(operation_override["operation_type"])
    if operation_override.get("access"):
        shaped["access"] = str(operation_override["access"])
    if task is not None:
        shaped["task"] = {
            "id": task.id,
            "capability_id": task.capability_id,
            "executor_id": task.executor_id,
            "required_evidence": sorted(task.required_evidence),
            "metadata": task.metadata,
        }
    if capability is not None:
        shaped["capability"] = _capability_governance_facts(capability)
    return shaped


def _governance_operation_override(
    governance_facts: dict[str, Any],
) -> dict[str, Any]:
    override = governance_facts.get("operation_override")
    return dict(override) if isinstance(override, dict) else {}


def _capability_governance_facts(capability: Capability) -> dict[str, Any]:
    return {
        "id": capability.id,
        "owner": capability.owner,
        "domains": sorted(capability.domains),
        "operation_types": sorted(capability.operation_types),
        "access": capability.access.value,
        "risk": capability.risk.value,
        "runtime_only": capability.runtime_only,
        "side_effecting": capability.side_effecting,
        "executor": capability.executor,
        "output_evidence": sorted(capability.output_evidence),
    }


def _safe_source_repr(source: Any) -> str:
    if source is None:
        return "none"
    if not isinstance(source, str):
        return type(source).__name__
    try:
        parts = urlsplit(source)
    except ValueError:
        return "<source>"
    if not parts.scheme:
        return source
    netloc = parts.netloc
    if "@" in netloc and ":" in netloc.split("@", 1)[0]:
        credentials, host = netloc.rsplit("@", 1)
        user = credentials.split(":", 1)[0]
        netloc = f"{user}:***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
