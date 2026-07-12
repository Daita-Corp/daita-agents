"""Optional LLM judge assertions."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from .analysis import RunEvidence, metric_delta, metric_snapshot, preview_text
from .config import EvalCaseConfig, JudgeCriterion, JudgeExpectations
from .models import AssertionResult, JudgeCriterionResult, JudgeResult


@dataclass
class JudgeEvaluation:
    assertion: AssertionResult
    result: JudgeResult
    input_payload: dict[str, Any]
    output_payload: dict[str, Any]


async def evaluate_judge(
    case: EvalCaseConfig,
    evidence: RunEvidence,
    *,
    deterministic_failed: bool,
) -> JudgeEvaluation | None:
    config = case.expectations.judge
    if config is None:
        return None
    if config.run_when == "after_deterministic_pass" and deterministic_failed:
        return None

    input_payload = build_judge_input(case.prompt, evidence, config)
    raw_response, metrics = await call_judge(
        config, json.dumps(input_payload, indent=2)
    )
    output_payload = parse_judge_output(raw_response)
    judge_result = build_judge_result(config, output_payload, raw_response, metrics)
    assertion = build_judge_assertion(judge_result)
    return JudgeEvaluation(
        assertion=assertion,
        result=judge_result,
        input_payload=input_payload,
        output_payload=output_payload,
    )


def build_judge_input(
    prompt: str,
    evidence: RunEvidence,
    config: JudgeExpectations,
) -> dict[str, Any]:
    criteria = normalized_criteria(config)
    runtime_tasks = [
        {
            "id": task.id,
            "capability_id": task.capability_id,
            "executor_id": task.executor_id,
            "owner": task.owner,
            "status": task.status,
            "input": task.input,
            "required_evidence": task.required_evidence,
        }
        for task in evidence.tasks
    ]
    runtime_evidence: list[dict[str, Any]] = []
    for item in evidence.evidence:
        record = {
            "id": item.id,
            "kind": item.kind,
            "owner": item.owner,
            "task_id": item.task_id,
            "accepted": item.accepted,
        }
        if config.include_evidence_payloads:
            record["payload"] = preview_text(
                item.payload, config.max_evidence_payload_chars
            )
        runtime_evidence.append(record)

    return {
        "task": (
            "Evaluate the agent answer using the criteria. Return JSON only with "
            "score, reasoning, and criteria results."
        ),
        "response_schema": {
            "score": "number from 0 to 1",
            "reasoning": "brief explanation",
            "criteria": [
                {
                    "id": "criterion id",
                    "passed": "boolean",
                    "score": "number from 0 to 1",
                    "evidence": "brief evidence from answer or runtime evidence",
                }
            ],
        },
        "pass_score": config.pass_score,
        "require_all_criteria_pass": config.require_all_criteria_pass,
        "user_prompt": prompt,
        "final_answer": evidence.answer,
        "runtime": {
            "operation_id": evidence.operation_id,
            "operation_status": evidence.operation_status,
            "operation_type": evidence.operation_type,
            "intent": evidence.intent,
            "tasks": runtime_tasks,
            "evidence": runtime_evidence,
            "governance": (
                evidence.governance.model_dump(mode="json")
                if evidence.governance is not None
                else None
            ),
            "approvals": evidence.approvals,
            "warnings": evidence.warnings,
        },
        "criteria": [
            {
                "id": criterion.id,
                "description": criterion.description,
                "required": criterion.required,
                "weight": criterion.weight,
            }
            for criterion in criteria
        ],
    }


async def call_judge(
    config: JudgeExpectations,
    prompt: str,
) -> tuple[str, dict[str, Any]]:
    from daita.llm.factory import create_llm_provider

    provider = create_llm_provider(
        config.provider,
        model=config.model,
        api_key=config.api_key,
        temperature=0,
        max_tokens=512,
    )
    before = metric_snapshot(provider)
    start = time.perf_counter()
    result = await provider.generate(prompt)
    latency_ms = (time.perf_counter() - start) * 1000
    after = metric_snapshot(provider)
    metrics = metric_delta(before, after)
    metrics["latency_ms"] = latency_ms
    response = result if isinstance(result, str) else json.dumps(result, default=str)
    return response, metrics


def parse_judge_output(response: str) -> dict[str, Any]:
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.S)
        data = json.loads(match.group(0)) if match else {}
    return data if isinstance(data, dict) else {}


def build_judge_result(
    config: JudgeExpectations,
    output: dict[str, Any],
    raw_response: str,
    metrics: dict[str, Any],
) -> JudgeResult:
    criteria_config = {
        criterion.id: criterion for criterion in normalized_criteria(config)
    }
    criteria_results = [
        normalize_criterion_result(item, criteria_config)
        for item in output.get("criteria", [])
        if isinstance(item, dict)
    ]
    for criterion in criteria_config.values():
        if not any(result.id == criterion.id for result in criteria_results):
            criteria_results.append(
                JudgeCriterionResult(
                    id=criterion.id,
                    passed=False,
                    score=0.0,
                    evidence="Judge did not return this criterion.",
                    required=criterion.required,
                )
            )

    score = clamp_score(
        output.get("score", _weighted_score(criteria_results, criteria_config))
    )
    criteria_failed = any(
        (config.require_all_criteria_pass or result.required) and not result.passed
        for result in criteria_results
    )
    passed = score >= config.pass_score and not criteria_failed
    return JudgeResult(
        provider=config.provider,
        model=config.model,
        score=score,
        pass_score=config.pass_score,
        passed=passed,
        reasoning=str(output.get("reasoning") or ""),
        criteria=criteria_results,
        latency_ms=metrics.get("latency_ms"),
        cost=metrics.get("cost"),
        tokens_total=metrics.get("tokens_total"),
        raw_output=output or raw_response,
    )


def build_judge_assertion(result: JudgeResult) -> AssertionResult:
    status = "passed" if result.passed else "failed"
    message = result.reasoning or (
        f"Judge score {result.score:.2f}; pass score {result.pass_score:.2f}."
    )
    return AssertionResult(
        id=result.assertion_id,
        status=status,
        code="judge_passed" if result.passed else "judge_failed",
        message=message,
        assertion_path="expectations.judge",
        observed={
            "score": result.score,
            "criteria": [
                criterion.model_dump(mode="json") for criterion in result.criteria
            ],
        },
        expected={"pass_score": result.pass_score},
    )


def normalized_criteria(config: JudgeExpectations) -> list[JudgeCriterion]:
    if config.criteria:
        return config.criteria
    return [
        JudgeCriterion(
            id=f"rubric_{index + 1}",
            description=item,
            required=config.require_all_criteria_pass,
        )
        for index, item in enumerate(config.rubric)
    ]


def normalize_criterion_result(
    item: dict[str, Any],
    criteria_config: dict[str, JudgeCriterion],
) -> JudgeCriterionResult:
    criterion_id = str(item.get("id") or "")
    config = criteria_config.get(criterion_id)
    return JudgeCriterionResult(
        id=criterion_id,
        passed=bool(item.get("passed")),
        score=clamp_score(item.get("score", 1.0 if item.get("passed") else 0.0)),
        evidence=str(item.get("evidence") or ""),
        required=config.required if config else False,
    )


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))


def _weighted_score(
    criteria_results: list[JudgeCriterionResult],
    criteria_config: dict[str, JudgeCriterion],
) -> float:
    if not criteria_results:
        return 0.0
    total_weight = 0.0
    total = 0.0
    for result in criteria_results:
        criterion = criteria_config.get(result.id)
        weight = criterion.weight if criterion is not None else 1.0
        total_weight += weight
        total += result.score * weight
    return total / total_weight if total_weight else 0.0
