"""Runtime capability, task, evidence, governance, and approval assertions."""

from __future__ import annotations

from ..analysis import RunEvidence
from ..config import Expectations, SuiteDefaults
from ..models import AssertionResult, RuntimeTaskEvidence
from .common import fail, matches, matches_any


def capability_assertions(
    exp: Expectations, evidence: RunEvidence, defaults: SuiteDefaults
) -> list[AssertionResult]:
    capabilities = [task.capability_id for task in evidence.tasks]
    owners = [task.owner for task in evidence.tasks if task.owner]
    results: list[AssertionResult] = []
    for index, capability_id in enumerate(exp.capabilities.required):
        if not matches_any(capability_id, capabilities):
            results.append(
                fail(
                    f"capabilities.required[{index}]",
                    "required_capability_missing",
                    f"Required capability was not executed: {capability_id}.",
                    f"expectations.capabilities.required[{index}]",
                    observed=capabilities,
                    expected=capability_id,
                )
            )
    for index, capability_id in enumerate(exp.capabilities.forbidden):
        matched = [
            task.id
            for task in evidence.tasks
            if matches(capability_id, task.capability_id)
        ]
        if matched:
            results.append(
                fail(
                    f"capabilities.forbidden[{index}]",
                    "forbidden_capability_executed",
                    f"Forbidden capability was executed: {capability_id}.",
                    f"expectations.capabilities.forbidden[{index}]",
                    observed=capability_id,
                    expected=f"not {capability_id}",
                    related_task_ids=matched,
                )
            )
    for index, owner in enumerate(exp.capabilities.required_owners):
        if not matches_any(owner, owners):
            results.append(
                fail(
                    f"capabilities.required_owners[{index}]",
                    "required_capability_owner_missing",
                    f"Required capability owner was not used: {owner}.",
                    f"expectations.capabilities.required_owners[{index}]",
                    observed=owners,
                    expected=owner,
                )
            )
    for index, owner in enumerate(exp.capabilities.forbidden_owners):
        matched = [
            task.id
            for task in evidence.tasks
            if task.owner and matches(owner, task.owner)
        ]
        if matched:
            results.append(
                fail(
                    f"capabilities.forbidden_owners[{index}]",
                    "forbidden_capability_owner_used",
                    f"Forbidden capability owner was used: {owner}.",
                    f"expectations.capabilities.forbidden_owners[{index}]",
                    observed=owner,
                    expected=f"not {owner}",
                    related_task_ids=matched,
                )
            )
    max_calls = (
        exp.capabilities.max_calls
        if exp.capabilities.max_calls is not None
        else defaults.max_capability_calls
    )
    if max_calls is not None and len(evidence.tasks) > max_calls:
        results.append(
            fail(
                "capabilities.max_calls",
                "too_many_capability_calls",
                f"Expected at most {max_calls} runtime tasks, got {len(evidence.tasks)}.",
                "expectations.capabilities.max_calls",
                observed=len(evidence.tasks),
                expected=max_calls,
                related_task_ids=[task.id for task in evidence.tasks],
            )
        )
    return results


def task_assertions(exp: Expectations, evidence: RunEvidence) -> list[AssertionResult]:
    statuses = [task.status for task in evidence.tasks]
    results: list[AssertionResult] = []
    for index, status in enumerate(exp.tasks.required_statuses):
        if status not in statuses:
            results.append(
                fail(
                    f"tasks.required_statuses[{index}]",
                    "required_task_status_missing",
                    f"Required task status was not observed: {status}.",
                    f"expectations.tasks.required_statuses[{index}]",
                    observed=statuses,
                    expected=status,
                )
            )
    for index, status in enumerate(exp.tasks.forbidden_statuses):
        matched = [task.id for task in evidence.tasks if task.status == status]
        if matched:
            results.append(
                fail(
                    f"tasks.forbidden_statuses[{index}]",
                    "forbidden_task_status_observed",
                    f"Forbidden task status was observed: {status}.",
                    f"expectations.tasks.forbidden_statuses[{index}]",
                    observed=status,
                    expected=f"not {status}",
                    related_task_ids=matched,
                )
            )
    if exp.tasks.max_errors is not None:
        errored = _errored_tasks(evidence.tasks)
        if len(errored) > exp.tasks.max_errors:
            results.append(
                fail(
                    "tasks.max_errors",
                    "too_many_task_errors",
                    "Too many runtime task errors were observed.",
                    "expectations.tasks.max_errors",
                    observed=[task.status for task in errored],
                    expected=exp.tasks.max_errors,
                    related_task_ids=[task.id for task in errored],
                )
            )
    return results


def evidence_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    records = evidence.evidence
    kinds = [record.kind for record in records]
    owners = [record.owner for record in records if record.owner]
    results: list[AssertionResult] = []
    for index, kind in enumerate(exp.evidence.required_kinds):
        if not matches_any(kind, kinds):
            results.append(
                fail(
                    f"evidence.required_kinds[{index}]",
                    "required_evidence_missing",
                    f"Required evidence kind was not produced: {kind}.",
                    f"expectations.evidence.required_kinds[{index}]",
                    observed=kinds,
                    expected=kind,
                )
            )
    for index, kind in enumerate(exp.evidence.forbidden_kinds):
        matched = [record.id or "" for record in records if matches(kind, record.kind)]
        if matched:
            results.append(
                fail(
                    f"evidence.forbidden_kinds[{index}]",
                    "forbidden_evidence_produced",
                    f"Forbidden evidence kind was produced: {kind}.",
                    f"expectations.evidence.forbidden_kinds[{index}]",
                    observed=kind,
                    expected=f"not {kind}",
                    related_evidence_ids=[item for item in matched if item],
                )
            )
    for index, owner in enumerate(exp.evidence.required_owners):
        if not matches_any(owner, owners):
            results.append(
                fail(
                    f"evidence.required_owners[{index}]",
                    "required_evidence_owner_missing",
                    f"Required evidence owner was not observed: {owner}.",
                    f"expectations.evidence.required_owners[{index}]",
                    observed=owners,
                    expected=owner,
                )
            )
    for index, owner in enumerate(exp.evidence.forbidden_owners):
        matched = [
            record.id or ""
            for record in records
            if record.owner and matches(owner, record.owner)
        ]
        if matched:
            results.append(
                fail(
                    f"evidence.forbidden_owners[{index}]",
                    "forbidden_evidence_owner_observed",
                    f"Forbidden evidence owner was observed: {owner}.",
                    f"expectations.evidence.forbidden_owners[{index}]",
                    observed=owner,
                    expected=f"not {owner}",
                    related_evidence_ids=[item for item in matched if item],
                )
            )
    if exp.evidence.require_accepted:
        rejected = [record for record in records if not record.accepted]
        if rejected:
            results.append(
                fail(
                    "evidence.require_accepted",
                    "rejected_evidence_observed",
                    "Rejected runtime evidence was observed.",
                    "expectations.evidence.require_accepted",
                    observed=[record.kind for record in rejected],
                    expected="accepted evidence only",
                    related_evidence_ids=[
                        record.id for record in rejected if record.id
                    ],
                )
            )
    return results


def governance_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    governance = evidence.governance
    results: list[AssertionResult] = []
    if governance is None:
        if _governance_expected(exp):
            results.append(
                fail(
                    "governance",
                    "governance_missing",
                    "No runtime governance result was observed.",
                    "expectations.governance",
                    observed=None,
                    expected="runtime governance result",
                )
            )
        return results

    _bool_expectation(
        results,
        "allowed",
        exp.governance.allowed,
        governance.allowed,
    )
    _bool_expectation(
        results,
        "blocked",
        exp.governance.blocked,
        governance.blocked,
    )
    _bool_expectation(
        results,
        "pending_approval",
        exp.governance.pending_approval,
        governance.pending_approval,
    )
    policy_ids = _policy_ids(governance.decisions)
    for index, policy_id in enumerate(exp.governance.required_policies):
        if not matches_any(policy_id, policy_ids):
            results.append(
                fail(
                    f"governance.required_policies[{index}]",
                    "required_governance_policy_missing",
                    f"Required governance policy was not evaluated: {policy_id}.",
                    f"expectations.governance.required_policies[{index}]",
                    observed=policy_ids,
                    expected=policy_id,
                )
            )
    for index, policy_id in enumerate(exp.governance.forbidden_policies):
        if any(matches(policy_id, observed) for observed in policy_ids):
            results.append(
                fail(
                    f"governance.forbidden_policies[{index}]",
                    "forbidden_governance_policy_evaluated",
                    f"Forbidden governance policy was evaluated: {policy_id}.",
                    f"expectations.governance.forbidden_policies[{index}]",
                    observed=policy_id,
                    expected=f"not {policy_id}",
                )
            )
    return results


def approval_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    approvals = evidence.approvals
    statuses = [str(item.get("status") or "") for item in approvals]
    policy_ids = [
        str(item.get("requested_by_policy_id") or item.get("policy_id") or "")
        for item in approvals
        if item.get("requested_by_policy_id") or item.get("policy_id")
    ]
    results: list[AssertionResult] = []
    for index, status in enumerate(exp.approvals.required_statuses):
        if status not in statuses:
            results.append(
                fail(
                    f"approvals.required_statuses[{index}]",
                    "required_approval_status_missing",
                    f"Required approval status was not observed: {status}.",
                    f"expectations.approvals.required_statuses[{index}]",
                    observed=statuses,
                    expected=status,
                )
            )
    for index, status in enumerate(exp.approvals.forbidden_statuses):
        if status in statuses:
            results.append(
                fail(
                    f"approvals.forbidden_statuses[{index}]",
                    "forbidden_approval_status_observed",
                    f"Forbidden approval status was observed: {status}.",
                    f"expectations.approvals.forbidden_statuses[{index}]",
                    observed=status,
                    expected=f"not {status}",
                )
            )
    for index, policy_id in enumerate(exp.approvals.required_policies):
        if not matches_any(policy_id, policy_ids):
            results.append(
                fail(
                    f"approvals.required_policies[{index}]",
                    "required_approval_policy_missing",
                    f"Required approval policy was not observed: {policy_id}.",
                    f"expectations.approvals.required_policies[{index}]",
                    observed=policy_ids,
                    expected=policy_id,
                )
            )
    return results


def _errored_tasks(tasks: list[RuntimeTaskEvidence]) -> list[RuntimeTaskEvidence]:
    return [
        task
        for task in tasks
        if task.status.lower() in {"failed", "error", "errored"}
        or bool(task.metadata.get("error"))
    ]


def _governance_expected(exp: Expectations) -> bool:
    gov = exp.governance
    return any(
        (
            gov.allowed is not None,
            gov.blocked is not None,
            gov.pending_approval is not None,
            bool(gov.required_policies),
            bool(gov.forbidden_policies),
        )
    )


def _bool_expectation(
    results: list[AssertionResult],
    name: str,
    expected: bool | None,
    observed: bool | None,
) -> None:
    if expected is None or observed == expected:
        return
    results.append(
        fail(
            f"governance.{name}",
            f"governance_{name}_mismatch",
            f"Governance {name} did not match expectation.",
            f"expectations.governance.{name}",
            observed=observed,
            expected=expected,
        )
    )


def _policy_ids(decisions: list[dict]) -> list[str]:
    return [
        str(item.get("policy_id") or item.get("id") or "")
        for item in decisions
        if item.get("policy_id") or item.get("id")
    ]
