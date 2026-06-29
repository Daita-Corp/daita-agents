"""Minimal deterministic fallback planner for DB agent loops."""

from __future__ import annotations

from typing import Any

from .planner_protocol import DbPlannerAction, DbPlannerDecision


class ContractFallbackDbAgentPlanner:
    """Small no-LLM adapter for contract facts when no model planner is injected."""

    def __init__(self, runtime_metadata: dict[str, Any]) -> None:
        self.runtime_metadata = dict(runtime_metadata)

    def decide(self, context: Any, observations: Any) -> DbPlannerDecision:
        ctx = dict(context or {})
        request = dict(ctx.get("request") or {})
        contract = dict(ctx.get("contract") or {})
        prompt = str(request.get("prompt") or "")
        lanes = set(str(item) for item in contract.get("granted_lanes") or ())
        evidence = _planner_evidence(ctx, observations)
        available = _available_capability_ids(ctx)

        if "none" in lanes:
            return _planner_finish("No deterministic DB work was authorized.")
        if "schema" in lanes:
            return self._schema_decision(prompt, request, evidence, available)
        if "read" in lanes:
            return self._read_decision(prompt, request, evidence, available)
        if "memory_answer" in lanes:
            return self._memory_answer_decision(prompt, evidence)
        if "memory_write" in lanes:
            return self._memory_write_decision(prompt, request, evidence)
        if "write_propose" in lanes:
            return self._write_decision(prompt, evidence, execute=False)
        if "write_execute" in lanes:
            return self._write_decision(prompt, evidence, execute=True)
        if "monitor_read" in lanes and not _has_evidence(evidence, "monitor.snapshot"):
            return DbPlannerDecision(
                actions=(DbPlannerAction(kind="inspect_monitor", payload={}),)
            )
        if "monitor_write" in lanes and not _has_monitor_write_evidence(evidence):
            return DbPlannerDecision(
                actions=(DbPlannerAction(kind="update_monitor", payload={}),)
            )
        if "monitor_execute" in lanes and not _has_monitor_execute_evidence(evidence):
            return DbPlannerDecision(
                actions=(DbPlannerAction(kind="execute_monitor", payload={}),)
            )
        return _planner_finish("DB planner completed.")

    def _schema_decision(
        self,
        prompt: str,
        request: dict[str, Any],
        evidence: tuple[dict[str, Any], ...],
        available: set[str],
    ) -> DbPlannerDecision:
        if (
            _database_schema_payload(evidence) is None
            and "db.schema.inspect" in available
        ):
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="inspect_schema",
                        payload={
                            "focus": _focus_for_prompt(prompt),
                            "source_scope": list(request.get("source_scope") or ()),
                        },
                    ),
                )
            )
        return _planner_finish("Schema inspection completed.")

    def _read_decision(
        self,
        prompt: str,
        request: dict[str, Any],
        evidence: tuple[dict[str, Any], ...],
        available: set[str],
    ) -> DbPlannerDecision:
        schema = _database_schema_payload(evidence)
        if schema is None:
            if "db.schema.inspect" in available:
                return DbPlannerDecision(
                    actions=(
                        DbPlannerAction(
                            kind="inspect_schema",
                            payload={
                                "focus": _focus_for_prompt(prompt),
                                "source_scope": list(request.get("source_scope") or ()),
                            },
                        ),
                    )
                )
            return _planner_finish("Schema evidence is unavailable.")

        validation_sql = _accepted_query_plan_sql(evidence)
        if (
            validation_sql
            and not _has_evidence(evidence, "query.result")
            and "db.sql.execute_read" in available
        ):
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="execute_validated_read",
                        payload={"sql": validation_sql},
                    ),
                )
            )
        if _has_rejected_query_plan_validation(evidence):
            return _planner_finish(
                "The fallback planner could not prepare a valid read query."
            )
        if (
            validation_sql is None
            and "db.query.prepare_read" in available
            and _simple_read_prompt(prompt)
        ):
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="propose_sql_read",
                        payload={
                            "prompt": prompt,
                            "schema_evidence_id": _latest_evidence_ref(
                                evidence,
                                "schema.asset_profile",
                            ),
                            "reason": "minimal_fallback_read_prepare",
                        },
                    ),
                )
            )
        return _planner_finish("A model planner is required for this DB read request.")

    def _memory_answer_decision(
        self,
        prompt: str,
        evidence: tuple[dict[str, Any], ...],
    ) -> DbPlannerDecision:
        if not _has_evidence(evidence, "memory.semantic.recall"):
            options = _memory_options(self.runtime_metadata)
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="recall_memory",
                        payload={
                            "query": prompt,
                            "prompt": prompt,
                            "category": "db_semantics",
                            "limit": int(options.get("limit") or 5) * 3,
                            "score_threshold": float(
                                options.get("score_threshold") or 0.45
                            ),
                            "retrieval_mode": options.get(
                                "retrieval_mode", "structured"
                            ),
                            "source_identity": options.get("source_identity"),
                        },
                    ),
                )
            )
        return _planner_finish("DB memory recall completed.")

    def _memory_write_decision(
        self,
        prompt: str,
        request: dict[str, Any],
        evidence: tuple[dict[str, Any], ...],
    ) -> DbPlannerDecision:
        if not _has_evidence(evidence, "memory.semantic.write"):
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="write_memory",
                        payload={
                            "prompt": prompt,
                            "request": {
                                "prompt": prompt,
                                "mode": request.get("mode"),
                                "metadata": dict(request.get("metadata") or {}),
                                "constraints": dict(request.get("constraints") or {}),
                                "source_scope": list(request.get("source_scope") or ()),
                            },
                        },
                    ),
                )
            )
        return _planner_finish("DB memory write completed.")

    @staticmethod
    def _write_decision(
        prompt: str,
        evidence: tuple[dict[str, Any], ...],
        *,
        execute: bool,
    ) -> DbPlannerDecision:
        if not _write_sql_prompt(prompt):
            return _planner_finish("A model planner is required for this write.")
        if execute and not _has_evidence(evidence, "write.execution"):
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="execute_validated_write",
                        payload={"sql": prompt},
                    ),
                )
            )
        if not execute and not _has_evidence(evidence, "sql.validation"):
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="propose_sql_write",
                        payload={"sql": prompt},
                    ),
                )
            )
        return _planner_finish("DB write handling completed.")


def _planner_finish(message: str) -> DbPlannerDecision:
    return DbPlannerDecision(
        actions=(DbPlannerAction(kind="finish", payload={"message": message}),)
    )


def _planner_evidence(
    context: dict[str, Any],
    observations: Any,
) -> tuple[dict[str, Any], ...]:
    items: list[dict[str, Any]] = []
    for item in context.get("evidence_observations") or ():
        if isinstance(item, dict):
            items.append(dict(item))
    for observation in observations or ():
        payload = getattr(observation, "payload", {}) or {}
        for item in payload.get("evidence") or ():
            if isinstance(item, dict):
                items.append(dict(item))
    return tuple(items)


def _available_capability_ids(context: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id"))
        for item in context.get("available_capabilities") or ()
        if isinstance(item, dict) and item.get("id")
    }


def _has_evidence(evidence: tuple[dict[str, Any], ...], kind: str) -> bool:
    return any(
        item.get("kind") == kind and item.get("accepted", True) for item in evidence
    )


def _latest_evidence_ref(
    evidence: tuple[dict[str, Any], ...],
    kind: str,
) -> str | None:
    for item in reversed(evidence):
        if item.get("kind") == kind and item.get("accepted", True) and item.get("id"):
            return str(item["id"])
    return None


def _database_schema_payload(
    evidence: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    for item in reversed(evidence):
        if item.get("kind") != "schema.asset_profile" or not item.get("accepted", True):
            continue
        payload = item.get("payload")
        if isinstance(payload, dict) and payload.get("tables"):
            return dict(payload)
    return None


def _accepted_query_plan_sql(evidence: tuple[dict[str, Any], ...]) -> str | None:
    for item in reversed(evidence):
        if item.get("kind") != "query.plan.validation" or not item.get(
            "accepted", True
        ):
            continue
        payload = item.get("payload")
        if (
            isinstance(payload, dict)
            and payload.get("valid") is True
            and (payload.get("accepted_sql") or payload.get("sql"))
        ):
            return str(payload.get("accepted_sql") or payload.get("sql"))
    return None


def _has_rejected_query_plan_validation(
    evidence: tuple[dict[str, Any], ...],
) -> bool:
    return any(
        item.get("kind") == "query.plan.validation" and not item.get("accepted", True)
        for item in evidence
    )


def _focus_for_prompt(prompt: str) -> str | None:
    lowered = prompt.lower()
    for token in ("table", "tables", "columns", "fields", "schema"):
        lowered = lowered.replace(token, " ")
    words = [
        word.strip(" ,.?;:'\"`")
        for word in lowered.split()
        if len(word.strip(" ,.?;:'\"`")) > 2
    ]
    stop = {"what", "which", "show", "list", "describe", "inspect", "are", "the", "in"}
    candidates = [word for word in words if word not in stop]
    return candidates[-1] if candidates else None


def _simple_read_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    if any(
        token in lowered
        for token in (
            " join ",
            "relationship",
            "related",
            " by ",
            " for ",
            " where ",
            " compare ",
            " trend ",
            " across ",
            " group ",
        )
    ):
        return False
    simple_tokens = (
        "how many",
        "count",
        "list",
        "show",
        "describe",
        "what columns",
        "which columns",
    )
    return any(token in lowered for token in simple_tokens)


def _write_sql_prompt(prompt: str) -> bool:
    return (
        prompt.lstrip()
        .lower()
        .startswith(("insert ", "update ", "delete ", "create ", "alter ", "drop "))
    )


def _memory_options(runtime_metadata: dict[str, Any]) -> dict[str, Any]:
    options = _from_db_options(runtime_metadata).get("memory")
    return dict(options) if isinstance(options, dict) else {}


def _from_db_options(runtime_metadata: dict[str, Any]) -> dict[str, Any]:
    options = runtime_metadata.get("from_db_options")
    return dict(options) if isinstance(options, dict) else {}


def _has_monitor_write_evidence(evidence: tuple[dict[str, Any], ...]) -> bool:
    return any(
        str(item.get("kind", "")).startswith("monitor.")
        and "proposal" in str(item.get("kind", ""))
        for item in evidence
    )


def _has_monitor_execute_evidence(evidence: tuple[dict[str, Any], ...]) -> bool:
    return any(
        str(item.get("kind", "")).startswith("monitor.")
        and "execution" in str(item.get("kind", ""))
        for item in evidence
    )
