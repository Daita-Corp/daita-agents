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
    assert "rendered_context" not in payload
    assert "refunds.amount" not in dumped
    assert "complete" not in dumped
    assert "semantic_contract" not in dumped
    assert "required_filters" not in dumped


def test_public_planning_context_projection_exposes_safe_memory_provenance():
    raw = Evidence(
        id="planning-context",
        kind="planning.context",
        owner="db_runtime",
        payload={
            "included_sections": ["schema", "db_memory", "db_memory_semantics"],
            "db_memory_refs": [
                {
                    "chunk_id": "mem-recognized-revenue",
                    "kind": "metric_definition",
                    "key": "metric:recognized_revenue",
                    "text": "Recognized revenue uses complete orders.",
                    "semantic_contract": {"internal": "not public"},
                }
            ],
            "db_memory_semantics": [
                {
                    "memory_key": "metric:recognized_revenue",
                    "required_refs": ["orders.total", "orders.status"],
                    "required_filters": [
                        {"ref": "orders.status", "operator": "=", "value": "complete"}
                    ],
                    "enforceable": True,
                }
            ],
            "db_memory_diagnostics": {
                "candidate_count": 2,
                "included_count": 1,
                "used_chars": 42,
                "char_budget": 120,
            },
            "db_memory_contract_diagnostics": {
                "candidate_count": 1,
                "enforced_count": 1,
                "omitted_reasons": {},
            },
        },
    )

    public = project_operation_evidence(
        (raw,),
        ProjectionContext(mode=ProjectionMode.PUBLIC_RESULT),
    )

    payload = public[0].payload
    dumped = json.dumps(payload, sort_keys=True)
    assert payload["db_memory_refs"][0]["key"] == "metric:recognized_revenue"
    assert "text" not in payload["db_memory_refs"][0]
    assert "semantic_contract" not in dumped
    assert payload["db_memory_semantics"][0]["memory_key"] == (
        "metric:recognized_revenue"
    )
    assert payload["db_memory_diagnostics"]["included_count"] == 1
    assert payload["db_memory_diagnostics"]["used_chars"] == 42
    assert payload["db_memory_contract_diagnostics"]["enforced_count"] == 1


def test_memory_recall_projection_exposes_safe_retrieval_diagnostics():
    recall = Evidence(
        id="memory-recall",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [{"id": "one"}],
            "diagnostics": {
                "retrieval_mode": "structured",
                "embedding_available": False,
                "structured_candidate_count": 1,
                "embedding_candidate_count": 0,
            },
        },
    )

    public = project_operation_evidence(
        (recall,),
        ProjectionContext(mode=ProjectionMode.PUBLIC_RESULT),
    )

    assert public[0].payload["result_count"] == 1
    assert public[0].payload["diagnostics"] == {
        "retrieval_mode": "structured",
        "embedding_available": False,
        "structured_candidate_count": 1,
        "embedding_candidate_count": 0,
    }


def test_memory_artifact_projection_redacts_public_and_exposes_diagnostics():
    selection = Evidence(
        id="memory-selection",
        kind="db.memory.selection",
        owner="db_runtime",
        payload={
            "source_identity": "sqlite:from_db:source-a",
            "schema_fingerprint": "schema-a",
            "recall_evidence_refs": ["memory-recall"],
            "raw_candidate_count": 3,
            "included_count": 1,
            "included_refs": [
                {
                    "chunk_id": "mem-board-revenue",
                    "kind": "metric_definition",
                    "key": "metric:board_revenue",
                    "text": "Board revenue subtracts refunds.amount.",
                }
            ],
            "omitted_counts_by_reason": {"cross_source": 1, "unsafe": 1},
            "safe_diagnostic_omission_summaries": [
                {"reason": "cross_source", "count": 1},
                {"reason": "unsafe", "count": 1},
            ],
            "budget_usage": {
                "limit": 3,
                "char_budget": 120,
                "used_chars": 58,
            },
        },
    )
    contracts = Evidence(
        id="memory-contracts",
        kind="db.memory.contracts",
        owner="db_runtime",
        payload={
            "source_identity": "sqlite:from_db:source-a",
            "schema_fingerprint": "schema-a",
            "contracts": [
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
                    "enforceable": False,
                    "omission_reason": "blocked_by_policy",
                }
            ],
            "enforceable_contracts": [],
            "advisory_contracts": [
                {
                    "key": "metric:board_revenue",
                    "required_refs": ["refunds.amount"],
                    "enforceable": False,
                    "omission_reason": "blocked_by_policy",
                }
            ],
            "contract_omission_reasons": {"blocked_by_policy": 1},
            "source_schema_applicability": {
                "source_identity": "sqlite:from_db:source-a",
                "schema_fingerprint": "schema-a",
                "contract_candidate_count": 1,
            },
            "safe_diagnostic_summaries": [{"reason": "blocked_by_policy", "count": 1}],
        },
    )

    public = project_operation_evidence(
        (selection, contracts),
        _projection(ProjectionMode.PUBLIC_RESULT),
    )
    diagnostic = project_operation_evidence(
        (selection, contracts),
        _projection(ProjectionMode.DIAGNOSTIC),
    )

    public_dumped = json.dumps([item.payload for item in public], sort_keys=True)
    assert "Board revenue subtracts refunds.amount." not in public_dumped
    assert "metric:board_revenue" not in public_dumped
    assert "sqlite:from_db:source-a" not in public_dumped
    assert "schema-a" not in public_dumped
    assert public[0].payload["raw_candidate_count"] == 3
    assert public[0].payload["included_count"] == 1
    assert public[0].payload["omitted_count"] == 2
    assert public[1].payload["enforceable_count"] == 0
    assert public[1].payload["advisory_count"] == 1
    assert public[1].payload["omitted_count"] == 1

    assert diagnostic[0].payload["omitted_counts_by_reason"] == {
        "cross_source": 1,
        "unsafe": 1,
    }
    assert diagnostic[0].payload["safe_diagnostic_omission_summaries"] == [
        {"reason": "cross_source", "count": 1},
        {"reason": "unsafe", "count": 1},
    ]
    assert diagnostic[0].payload["budget_usage"]["char_budget"] == 120
    assert "included_refs" not in diagnostic[0].payload
    assert diagnostic[1].payload["contract_omission_reasons"] == {
        "blocked_by_policy": 1
    }
    assert diagnostic[1].payload["safe_diagnostic_summaries"] == [
        {"reason": "blocked_by_policy", "count": 1}
    ]
    assert (
        diagnostic[1].payload["source_schema_applicability"]["contract_candidate_count"]
        == 1
    )
    assert "contracts" not in diagnostic[1].payload


def test_session_scope_artifact_projection_redacts_public_and_summarizes_diagnostic():
    scope = Evidence(
        id="scope-evidence",
        kind="session.query_scope",
        owner="db_runtime",
        payload={
            "scope_id": "scope-prior",
            "source_operation_id": "op-prior",
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
            "joins": [
                {
                    "left_table": "customers",
                    "left_column": "region_id",
                    "right_table": "regions",
                    "right_column": "id",
                }
            ],
            "selected_columns": ["customers.revenue"],
            "result_row_count": 4,
        },
    )
    binding = Evidence(
        id="binding-evidence",
        kind="session.scope_binding",
        owner="db_runtime",
        payload={
            "binding_status": "bound",
            "source_scope_id": "scope-prior",
            "source_operation_id": "op-prior",
            "required_filters": [
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
            "required_joins": [
                {
                    "left_table": "customers",
                    "left_column": "region_id",
                    "right_table": "regions",
                    "right_column": "id",
                }
            ],
            "omitted_unsafe_referents": [{"reason": "blocked_by_policy", "count": 1}],
        },
    )

    public = project_operation_evidence(
        (scope, binding),
        _projection(ProjectionMode.PUBLIC_RESULT),
    )
    diagnostic = project_operation_evidence(
        (scope, binding),
        _projection(ProjectionMode.DIAGNOSTIC),
    )

    public_dumped = json.dumps([item.payload for item in public], sort_keys=True)
    assert "scope-prior" not in public_dumped
    assert "op-prior" not in public_dumped
    assert "customers.loyalty_band" not in public_dumped
    assert "platinum" not in public_dumped
    assert public[0].payload["table_count"] == 1
    assert public[0].payload["filter_count"] == 2
    assert public[0].payload["join_count"] == 1
    assert "binding_status" not in public[1].payload
    assert public[1].payload["required_filter_count"] == 2
    assert public[1].payload["required_join_count"] == 1

    diagnostic_dumped = json.dumps(
        [item.payload for item in diagnostic], sort_keys=True
    )
    assert "customers.loyalty_band" not in diagnostic_dumped
    assert "platinum" not in diagnostic_dumped
    assert diagnostic[0].payload["scope_id"] == "scope-prior"
    assert diagnostic[0].payload["filters"] == [
        {"column": "customers.region", "operator": "=", "values": ["west"]}
    ]
    assert diagnostic[1].payload["source_scope_id"] == "scope-prior"
    assert diagnostic[1].payload["binding_status"] == "bound"
    assert diagnostic[1].payload["required_filters"] == [
        {"column": "customers.region", "operator": "=", "values": ["west"]}
    ]
    assert diagnostic[1].payload["omitted_unsafe_referents"] == [
        {"reason": "blocked_by_policy", "count": 1}
    ]


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
