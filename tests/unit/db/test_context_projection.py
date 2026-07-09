import json

from daita.db import DbRequest, DbRuntime, DbRuntimeConfig
from daita.db.context_projection import (
    ProjectionContext,
    ProjectionMode,
    project_memory_refs,
    project_memory_semantics,
    project_operation_evidence,
    project_session_context,
)
from daita.db.models import DbIntent, DbIntentKind
from daita.db.planning_context import DbPlanningContextBuilder
from daita.db.planner_protocol import DbPlannerDecision, DbPlannerDecisionStatus
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode, Evidence, Operation, OperationStatus


class _BlockedPlanner:
    async def plan(self, state):
        return DbPlannerDecision(
            status=DbPlannerDecisionStatus.BLOCKED,
            metadata={
                "reason": (
                    "blocked customers.loyalty_band = platinum should stay internal"
                )
            },
        )


def _projection(mode=ProjectionMode.PLANNER):
    return ProjectionContext(
        mode=mode,
        operation_intent="data.query",
        policy_summary={"blocked_columns": ["customers.loyalty_band"]},
        safety_frame={"max_access": "read"},
        session_id="session-1",
        user_id="user-1",
    )


def test_planner_session_projection_redacts_blocked_query_scope_values():
    projected = project_session_context(
        {
            "session_id": "session-1",
            "user_id": "user-1",
            "referents": {
                "tables": ["customers"],
                "columns": ["customers.loyalty_band", "customers.revenue"],
            },
            "query_scopes": [
                {
                    "operation_id": "op-prior",
                    "tables": ["customers"],
                    "filters": [
                        {
                            "column": "customers.loyalty_band",
                            "operator": "=",
                            "values": ["platinum"],
                        },
                        {
                            "column": "customers.region",
                            "operator": "=",
                            "values": ["west"],
                        },
                    ],
                    "selected_columns": [
                        "customers.loyalty_band",
                        "customers.revenue",
                    ],
                    "result_row_count": 4,
                }
            ],
        },
        _projection(),
    )

    dumped = json.dumps(projected, sort_keys=True)
    assert "customers.loyalty_band" not in dumped
    assert "platinum" not in dumped
    assert "customers.revenue" in dumped
    assert "customers.region" in dumped
    assert "west" in dumped
    assert projected["diagnostics"]["projection"]["mode"] == "planner"


def test_planner_memory_projection_redacts_blocked_refs_and_values():
    semantic_contract = {
        "version": 1,
        "contract_kind": "metric_definition",
        "subject": {"type": "metric", "key": "metric:loyalty", "aliases": []},
        "requirements": {
            "refs": [{"kind": "column", "ref": "customers.loyalty_band"}],
            "filters": [
                {
                    "ref": "customers.loyalty_band",
                    "operator": "=",
                    "value": "platinum",
                }
            ],
        },
    }
    refs = project_memory_refs(
        [
            {
                "chunk_id": "mem-1",
                "kind": "metric_definition",
                "key": "metric:loyalty",
                "text": "Loyalty metric uses customers.loyalty_band platinum rows.",
                "confidence": 0.95,
                "importance": 0.8,
                "evidence_refs": ["evidence-memory"],
                "semantic_contract": semantic_contract,
            }
        ],
        _projection(),
    )
    semantics = project_memory_semantics(
        [
            {
                "key": "metric:loyalty",
                "required_refs": ["customers.loyalty_band"],
                "required_filters": [
                    {"ref": "customers.loyalty_band", "value": "platinum"}
                ],
                "enforceable": True,
            }
        ],
        _projection(),
    )

    dumped = json.dumps({"refs": refs, "semantics": semantics}, sort_keys=True)
    assert "customers.loyalty_band" not in dumped
    assert "platinum" not in dumped
    assert refs[0]["projection"]["reason"] == "blocked_by_policy"
    assert semantics[0]["projection"]["reason"] == "blocked_by_policy"
    assert semantics[0]["enforceable"] is False


def test_public_diagnostic_and_audit_evidence_projection_modes():
    raw = Evidence(
        id="evidence-planning",
        kind="planning.context",
        owner="db_runtime",
        payload={
            "rendered_context": (
                "Session query scopes:\n" "- filters customers.loyalty_band = platinum"
            ),
            "included_sections": ["schema", "session_context"],
            "session_context": {
                "query_scopes": [
                    {
                        "filters": [
                            {
                                "column": "customers.loyalty_band",
                                "operator": "=",
                                "values": ["platinum"],
                            }
                        ]
                    }
                ]
            },
            "column_value_hints": [
                {
                    "table": "customers",
                    "column": "loyalty_band",
                    "observed_values": [{"value": "platinum"}],
                }
            ],
            "diagnostics": {
                "schema_table_count": 1,
                "db_memory_ref_count": 0,
            },
        },
    )

    public = project_operation_evidence(
        (raw,), _projection(ProjectionMode.PUBLIC_RESULT)
    )
    diagnostic = project_operation_evidence(
        (raw,),
        _projection(ProjectionMode.DIAGNOSTIC),
    )
    audit = project_operation_evidence((raw,), _projection(ProjectionMode.AUDIT))

    assert "customers.loyalty_band" not in json.dumps(public[0].payload)
    assert "platinum" not in json.dumps(public[0].payload)
    assert public[0].payload["redacted"] is True
    assert public[0].payload["included_sections"] == ["schema", "session_context"]
    assert "customers.loyalty_band" not in json.dumps(diagnostic[0].payload)
    assert diagnostic[0].payload["diagnostics"]["schema_table_count"] == 1
    assert audit[0].payload == raw.payload


def test_data_query_planning_context_prefers_catalog_asset_evidence():
    context = DbPlanningContextBuilder(DbRuntimeConfig()).build(
        request=DbRequest(prompt="Show customer revenue", source_scope=("sqlite",)),
        intent=DbIntent(
            kind=DbIntentKind.DATA_QUERY,
            confidence=1.0,
            access=AccessMode.READ,
        ),
        operation=Operation(id="op-catalog-first", operation_type="data.query"),
        schema_evidence=Evidence(
            id="connector-schema",
            kind="schema.asset_profile",
            owner="sqlite",
            accepted=True,
            payload={
                "database_type": "sqlite",
                "tables": [
                    {
                        "name": "orders",
                        "columns": [{"name": "id", "data_type": "INTEGER"}],
                    }
                ],
            },
            metadata={"payload_fingerprint": "fp-connector-schema"},
        ),
        catalog_evidence=(
            Evidence(
                id="catalog-customers",
                kind="schema.asset_profile",
                owner="catalog",
                accepted=True,
                payload={
                    "success": True,
                    "store_id": "store:sqlite",
                    "asset": {"name": "customers", "asset_ref": "customers"},
                    "fields": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "revenue", "type": "REAL"},
                    ],
                },
                metadata={"payload_fingerprint": "fp-catalog-customers"},
            ),
        ),
    )

    assert context.diagnostics["structural_schema_source"] == "catalog"
    assert [table["name"] for table in context.schema["tables"]] == ["customers"]
    assert context.schema_evidence_refs == ("connector-schema",)
    assert context.catalog_evidence_refs == ("catalog-customers",)
    assert context.source_fingerprints == {
        "connector-schema": "fp-connector-schema",
        "catalog-customers": "fp-catalog-customers",
    }
    assert "orders" not in context.rendered_context
    assert "customers" in context.rendered_context


def test_public_projection_redacts_same_turn_repair_deferred_diagnostics():
    raw = Evidence(
        id="planner-compilation",
        kind="planner.compilation",
        owner="db_runtime",
        accepted=False,
        payload={
            "compilation": {
                "rejected_action_summaries": [
                    {
                        "action_id": "execute_repair",
                        "kind": "execute_validated_read",
                        "error": "deferred_until_query_plan_proposal_available",
                        "deferred": {
                            "reason": "same_turn_repair_query_plan",
                            "producer_action_ids": ["repair_plan"],
                            "blocked_filter": ("customers.loyalty_band = platinum"),
                        },
                    }
                ]
            }
        },
    )

    public = project_operation_evidence(
        (raw,),
        _projection(ProjectionMode.PUBLIC_RESULT),
    )

    payload = public[0].payload
    dumped = json.dumps(payload, sort_keys=True)
    assert payload["redacted"] is True
    assert payload["source_kind"] == "planner.compilation"
    assert payload["payload_keys"] == ["compilation"]
    assert "deferred_until_query_plan_proposal_available" not in dumped
    assert "customers.loyalty_band" not in dumped
    assert "platinum" not in dumped


def test_public_planning_context_projection_strips_blocked_memory_details():
    raw = Evidence(
        id="evidence-planning",
        kind="planning.context",
        owner="db_runtime",
        payload={
            "rendered_context": (
                "Database memory:\n"
                "- metric metric:board_revenue: subtract refunds.amount."
            ),
            "included_sections": [
                "schema",
                "db_memory",
                "db_memory_semantics",
            ],
            "db_memory_refs": [
                {
                    "chunk_id": "mem-board-revenue",
                    "kind": "metric_definition",
                    "key": "metric:board_revenue",
                    "text": "Board revenue subtracts refunds.amount.",
                    "semantic_contract": {
                        "requirements": {
                            "refs": [
                                {
                                    "kind": "column",
                                    "ref": "refunds.amount",
                                }
                            ],
                            "filters": [
                                {
                                    "ref": "orders.status",
                                    "operator": "=",
                                    "value": "complete",
                                }
                            ],
                        }
                    },
                }
            ],
            "db_memory_semantics": [
                {
                    "key": "metric:board_revenue",
                    "required_refs": ["refunds.amount"],
                    "required_filters": [
                        {
                            "ref": "orders.status",
                            "operator": "=",
                            "value": "complete",
                        }
                    ],
                    "required_aggregations": [
                        {"function": "sum", "ref": "refunds.amount"},
                    ],
                    "result_shape": {"grain": "single_aggregate"},
                    "enforceable": False,
                }
            ],
            "db_memory_contract_diagnostics": {
                "enforced_count": 0,
                "advisory_count": 1,
                "omitted_reasons": {"blocked_by_policy": 1},
            },
            "diagnostics": {
                "schema_table_count": 2,
                "db_memory_ref_count": 1,
                "db_memory_contract_count": 1,
            },
        },
    )

    public = project_operation_evidence(
        (raw,),
        ProjectionContext(
            mode=ProjectionMode.PUBLIC_RESULT,
            operation_intent="data.query",
            policy_summary={"blocked_columns": ["refunds.amount"]},
            safety_frame={"max_access": "read"},
        ),
    )

    payload = public[0].payload
    dumped = json.dumps(payload, sort_keys=True)
    assert payload["redacted"] is True
    assert "db_memory_refs" not in payload
    assert "db_memory_semantics" not in payload
    assert "db_memory_contract_diagnostics" not in payload
    assert "rendered_context" not in payload
    assert "refunds.amount" not in dumped
    assert "complete" not in dumped
    assert "blocked_by_policy" not in dumped
    assert "semantic_contract" not in dumped
    assert "required_filters" not in dumped


def test_query_result_projection_redacts_blocked_error_scalars():
    raw = Evidence(
        id="query-result",
        kind="query.result",
        owner="sqlite",
        payload={
            "rows": [],
            "total_rows": 0,
            "success": False,
            "error": "SQL guardrail rejected customers.loyalty_band = platinum.",
        },
    )

    public = project_operation_evidence(
        (raw,),
        _projection(ProjectionMode.PUBLIC_RESULT),
    )
    diagnostic = project_operation_evidence(
        (raw,),
        _projection(ProjectionMode.DIAGNOSTIC),
    )

    dumped = json.dumps(
        [public[0].payload, diagnostic[0].payload],
        sort_keys=True,
    )
    assert "customers.loyalty_band" not in dumped
    assert "platinum" not in dumped
    assert public[0].payload["error"] == "<redacted>"
    assert diagnostic[0].payload["error"] == "<redacted>"
    assert public[0].payload["row_count"] == 0


def test_runtime_result_projection_helper_redacts_failed_result_evidence():
    sqlite = SQLitePlugin(path=":memory:", blocked_columns=["customers.loyalty_band"])
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(plugins=(sqlite,)),
    )
    raw = Evidence(
        id="query-result",
        kind="query.result",
        owner="sqlite",
        payload={
            "rows": [{"loyalty_band": "platinum"}],
            "success": False,
            "error": "SQL guardrail rejected customers.loyalty_band = platinum.",
        },
    )

    projected, diagnostics = runtime._project_result_evidence(
        (raw,),
        request=DbRequest("Summarize customer revenue."),
        intent=DbIntent(kind=DbIntentKind.DATA_QUERY),
        operation=Operation(
            id="operation-1",
            operation_type="db.run",
            metadata={
                "safety_frame": {
                    "blocked_columns": ["customers.loyalty_band"],
                }
            },
        ),
        safety_frame=None,
    )

    dumped = json.dumps(
        {
            "projected": [item.to_dict() for item in projected],
            "diagnostics": diagnostics,
        },
        sort_keys=True,
    )
    assert "customers.loyalty_band" not in dumped
    assert "platinum" not in dumped
    assert projected[0].payload["error"] == "<redacted>"
    assert diagnostics["public_mode"] == "public_result"
    assert diagnostics["diagnostic_mode"] == "diagnostic"


async def test_non_finished_runtime_result_returns_public_projected_evidence():
    sqlite = SQLitePlugin(path=":memory:", blocked_columns=["customers.loyalty_band"])
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(plugins=(sqlite,)),
        host_services={"db_agent_planner": _BlockedPlanner()},
    )

    result = await runtime.run(DbRequest("Summarize customer revenue."))
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert snapshot is not None
    assert any(
        "customers.loyalty_band" in json.dumps(item.payload)
        for item in snapshot.evidence
        if item.kind == "planner.decision"
    )
    assert result.evidence
    dumped = json.dumps([item.to_dict() for item in result.evidence], sort_keys=True)
    assert "customers.loyalty_band" not in dumped
    assert "platinum" not in dumped
    assert all(
        item.metadata["projection_mode"] == "public_result" for item in result.evidence
    )
    diagnostics_dumped = json.dumps(result.diagnostics["evidence_projection"])
    assert "customers.loyalty_band" not in diagnostics_dumped
    assert "platinum" not in diagnostics_dumped
