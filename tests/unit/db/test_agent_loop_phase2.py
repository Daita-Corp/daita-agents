import json

from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.loop import DbAgentLoop
from daita.db.loop.grounding import (
    _validation_grounding_runtime_continuation_action,
)
from daita.db.loop.summaries import _evidence_summary
from daita.db.llm_agent_planner import DbLLMAgentPlanner
from daita.db.llm_planner import DbLLMPlannerExecutor, DbLLMRepairExecutor
from daita.db.llm_service import DbLLMResponse, DbLLMService
from daita.db.planner_protocol import (
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionShapeError,
    DbPlannerDecisionStatus,
    validate_planner_decision_shape,
)
from daita.db.runtime.tasks.models import DbTaskSpec
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
    Task,
    TaskDependency,
)
from daita.plugins import MemoryPlugin
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin

import pytest

from tests.db_evidence_helpers import assert_no_invalid_accepted_query_plans


class PhaseTwoExecutor:
    def __init__(self, executor_id, capability_ids):
        self.id = executor_id
        self.capability_ids = frozenset(capability_ids)

    async def execute(self, task, operation, context):
        assert operation.metadata["latest_compiled_contract_snapshot"]
        if task.capability_id == "db.schema.inspect":
            return [
                Evidence(
                    kind="database.schema",
                    owner="phase_two",
                    payload={"tables": [{"name": "orders"}]},
                )
            ]
        if task.capability_id == "db.sql.validate":
            sql = task.input["sql"]
            return [
                Evidence(
                    kind="sql.validation",
                    owner="phase_two",
                    accepted=True,
                    payload={"valid": True, "sql": sql, "operation": "query"},
                )
            ]
        if task.capability_id == "db.sql.execute_read":
            return [
                Evidence(
                    kind="query.result",
                    owner="phase_two",
                    payload={
                        "rows": [{"answer": 1}],
                        "sql": task.input.get("sql"),
                        "validated_evidence_id": task.input.get(
                            "validated_evidence_id"
                        ),
                    },
                )
            ]
        if task.capability_id == "db.sql.execute_write":
            return [
                Evidence(
                    kind="write.execution",
                    owner="phase_two",
                    payload={"status": "executed"},
                )
            ]
        if task.capability_id == "db.memory.commit_update":
            return [
                Evidence(
                    kind="db.memory.definition",
                    owner="phase_two",
                    payload={"status": "committed"},
                )
            ]
        raise AssertionError(f"unexpected capability: {task.capability_id}")


class PhaseTwoPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="phase_two",
        display_name="Phase Two",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="db.schema.inspect",
                owner="phase_two",
                description="Inspect schema.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"database.schema"}),
                executor="phase_two.schema.inspect",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.validate",
                owner="phase_two",
                description="Validate SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"sql.validation"}),
                executor="phase_two.sql.validate",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.execute_read",
                owner="phase_two",
                description="Execute read.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="phase_two.sql.execute_read",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.execute_write",
                owner="phase_two",
                description="Execute write.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.HIGH,
                input_schema={"type": "object"},
                output_evidence=frozenset({"write.execution"}),
                executor="phase_two.sql.execute_write",
                runtime_only=True,
                side_effecting=True,
                replay_safe=False,
                idempotent=False,
            ),
            Capability(
                id="db.memory.commit_update",
                owner="phase_two",
                description="Commit a memory update.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"memory.update"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.MEDIUM,
                input_schema={"type": "object"},
                output_evidence=frozenset({"db.memory.definition"}),
                executor="phase_two.memory.commit_update",
                runtime_only=True,
                side_effecting=True,
                replay_safe=False,
                idempotent=False,
            ),
        ]

    def get_executors(self):
        return [
            PhaseTwoExecutor("phase_two.schema.inspect", {"db.schema.inspect"}),
            PhaseTwoExecutor("phase_two.sql.validate", {"db.sql.validate"}),
            PhaseTwoExecutor("phase_two.sql.execute_read", {"db.sql.execute_read"}),
            PhaseTwoExecutor("phase_two.sql.execute_write", {"db.sql.execute_write"}),
            PhaseTwoExecutor(
                "phase_two.memory.commit_update",
                {"db.memory.commit_update"},
            ),
        ]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="database.schema",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="sql.validation",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="query.result",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="write.execution",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="db.memory.definition",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
        ]


class FakePlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        return self.decisions.pop(0)


class FakeLLMService:
    available = True
    safe_metadata = {"provider": "fake", "model": "phase-two"}

    def __init__(self, content):
        self.contents = (
            list(content) if isinstance(content, (list, tuple)) else [content]
        )
        self.messages = None
        self.calls = []
        self.response_schemas = []

    async def generate_json(
        self,
        messages,
        *,
        response_schema=None,
        schema_name="db_json_response",
    ):
        self.messages = messages
        self.calls.append(messages)
        self.response_schemas.append((response_schema, schema_name))
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return DbLLMResponse(
            content=self.contents[index],
            diagnostics={"provider": "fake", "model": "phase-two"},
        )


async def test_loop_state_projects_bounded_evidence_backed_catalog_context():
    runtime, operation = await _runtime_and_operation(
        "phase-one-bounded-catalog-context"
    )
    secret = "TOP_SECRET_CATALOG_VALUE"
    assets = [
        {
            "store_id": "store:catalog",
            "name": f"asset_{asset_index:02d}",
            "asset_ref": f"asset_{asset_index:02d}",
            "matched_fields": [
                {
                    "name": f"column_{column_index:02d}",
                    "type": "TEXT",
                    "sample_values": [secret],
                }
                for column_index in reversed(range(25))
            ],
            "sample_rows": [{"unsafe": secret}],
        }
        for asset_index in reversed(range(12))
    ]
    try:
        await runtime.store.save_evidence(
            Evidence(
                id="catalog-search-related",
                kind="schema.search_result",
                owner="catalog",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "store_id": "store:catalog",
                    "query": "prompt scoped assets",
                    "total_matches": 12,
                    "truncated": True,
                    "assets": assets,
                    "sql": "select unrestricted_catalog_payload",
                },
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="catalog-search-rejected",
                kind="schema.search_result",
                owner="catalog",
                operation_id=operation.id,
                accepted=False,
                payload={
                    "store_id": "store:rejected",
                    "assets": [{"name": "rejected_asset"}],
                },
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="catalog-search-malformed",
                kind="schema.search_result",
                owner="catalog",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "store_id": "store:malformed",
                    "assets": [
                        {
                            "name": "malformed_asset",
                            "metadata": "not-a-mapping",
                        }
                    ],
                },
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="catalog-search-unrelated",
                kind="schema.search_result",
                owner="catalog",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "store_id": "store:unrelated",
                    "assets": [{"name": "unrelated_asset"}],
                },
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="planning-context-stale",
                kind="planning.context",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "schema_fingerprint": "stale-fingerprint",
                    "catalog_evidence_refs": ["catalog-search-unrelated"],
                    "diagnostics": {
                        "structural_schema_source": "catalog",
                        "catalog_structural_evidence_refs": [
                            "catalog-search-unrelated"
                        ],
                    },
                },
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="planning-context-authoritative",
                kind="planning.context",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "prompt": secret,
                    "schema_fingerprint": "schema-fingerprint-catalog",
                    "catalog_evidence_refs": [
                        "catalog-search-malformed",
                        "catalog-search-related",
                        "catalog-search-rejected",
                    ],
                    "diagnostics": {
                        "structural_schema_source": "catalog",
                        "catalog_structural_evidence_refs": [
                            "catalog-search-malformed",
                            "catalog-search-rejected",
                            "catalog-search-related",
                        ],
                    },
                    "schema": {
                        "tables": [
                            {
                                "name": "raw_planning_context_asset",
                                "columns": [
                                    {"name": "unsafe", "sample_values": [secret]}
                                ],
                            }
                        ],
                        "sample_rows": [{"unsafe": secret}],
                    },
                    "rendered_context": secret,
                },
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="planning-context-rejected-later",
                kind="planning.context",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=False,
                payload={
                    "schema_fingerprint": "rejected-fingerprint",
                    "catalog_evidence_refs": ["catalog-search-unrelated"],
                    "diagnostics": {
                        "structural_schema_source": "catalog",
                        "catalog_structural_evidence_refs": [
                            "catalog-search-unrelated"
                        ],
                    },
                },
            )
        )

        loop = DbAgentLoop(runtime, FakePlanner())
        state = await loop.build_loop_state(operation, turn=1, remaining_turns=2)
        rebuilt = await loop.build_loop_state(operation, turn=1, remaining_turns=2)
    finally:
        await runtime.teardown()

    context = state.catalog_context
    assert context == rebuilt.catalog_context
    assert context["planning_context_evidence_id"] == ("planning-context-authoritative")
    assert context["structural_source"] == "catalog"
    assert context["schema_fingerprint"] == "schema-fingerprint-catalog"
    assert context["catalog_store_id"] == "store:catalog"
    assert context["catalog_store_ids"] == ["store:catalog"]
    assert context["catalog_store_count"] == 1
    assert context["catalog_stores_truncated"] is False
    assert context["supporting_catalog_evidence_ids"] == ["catalog-search-related"]
    assert context["supporting_evidence_count"] == 1
    assert context["supporting_evidence_truncated"] is False
    assert context["referenced_evidence_count"] == 3
    assert context["omitted_evidence_count"] == 2
    assert context["omitted_evidence_reasons"] == [
        {"reason": "catalog_normalization_failed", "count": 1},
        {
            "reason": "not_accepted_catalog_structural_evidence",
            "count": 1,
        },
    ]
    assert context["candidate_count"] == 12
    assert context["included_candidate_count"] == 8
    assert context["truncated"] is True
    assert context["candidate_sources"] == [
        {
            "evidence_id": "catalog-search-related",
            "candidate_count": 12,
            "included_candidate_count": 12,
            "truncated": True,
            "catalog_store_id": "store:catalog",
            "catalog_store_ids": ["store:catalog"],
            "catalog_store_count": 1,
            "catalog_stores_truncated": False,
        }
    ]
    assert [item["name"] for item in context["assets"]] == [
        f"asset_{index:02d}" for index in range(8)
    ]
    first_asset = context["assets"][0]
    assert first_asset["asset_ref"] == "asset_00"
    assert first_asset["evidence_ids"] == ["catalog-search-related"]
    assert first_asset["catalog_store_id"] == "store:catalog"
    assert first_asset["catalog_store_ids"] == ["store:catalog"]
    assert first_asset["column_count"] == 25
    assert first_asset["included_column_count"] == 20
    assert first_asset["columns_truncated"] is True
    assert first_asset["columns"] == [
        {"name": f"column_{index:02d}", "type": "TEXT"} for index in range(20)
    ]
    serialized = json.dumps(context, sort_keys=True)
    assert secret not in serialized
    assert "sample_rows" not in serialized
    assert "sample_values" not in serialized
    assert "unrestricted_catalog_payload" not in serialized
    assert "raw_planning_context_asset" not in serialized
    assert "malformed_asset" not in serialized
    assert "rejected_asset" not in serialized
    assert "unrelated_asset" not in serialized
    assert DbLoopState.from_dict(state.to_dict()).catalog_context == context


async def test_catalog_context_preserves_multi_store_asset_provenance_and_counts():
    runtime, operation = await _runtime_and_operation(
        "phase-one-multi-store-catalog-context"
    )
    try:
        for evidence in (
            Evidence(
                id="catalog-search-a",
                kind="schema.search_result",
                owner="catalog",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "store_id": "store:a",
                    "total_matches": 2,
                    "assets": [{"name": "zeta"}],
                },
            ),
            Evidence(
                id="catalog-search-b",
                kind="schema.search_result",
                owner="catalog",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "store_id": "store:b",
                    "total_matches": 50,
                    "truncated": True,
                    "assets": [{"name": "alpha"}],
                },
            ),
            Evidence(
                id="catalog-profile-b",
                kind="catalog.profile",
                owner="catalog",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "store_id": "store:b",
                    "table_count": 999,
                    "assets": [{"name": "profile_only"}],
                },
            ),
            Evidence(
                id="planning-context-multi-store",
                kind="planning.context",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "catalog_evidence_refs": [
                        "catalog-search-a",
                        "catalog-search-b",
                        "catalog-profile-b",
                    ],
                    "diagnostics": {
                        "structural_schema_source": "catalog",
                        "catalog_structural_evidence_refs": [
                            "catalog-search-a",
                            "catalog-search-b",
                            "catalog-profile-b",
                        ],
                    },
                },
            ),
        ):
            await runtime.store.save_evidence(evidence)

        state = await DbAgentLoop(runtime, FakePlanner()).build_loop_state(
            operation,
            turn=1,
            remaining_turns=1,
        )
    finally:
        await runtime.teardown()

    context = state.catalog_context
    assert "catalog_store_id" not in context
    assert context["catalog_store_ids"] == ["store:a", "store:b"]
    assert context["catalog_store_count"] == 2
    assert context["catalog_stores_truncated"] is False
    assert context["candidate_count"] == 3
    assert [
        (item["evidence_id"], item["candidate_count"])
        for item in context["candidate_sources"]
    ] == [("catalog-search-a", 2), ("catalog-search-b", 50)]
    assert context["referenced_evidence_count"] == 3
    assert context["supporting_evidence_count"] == 3
    assert context["omitted_evidence_count"] == 0
    assert context["omitted_evidence_reasons"] == []
    assets = {item["name"]: item for item in context["assets"]}
    assert assets["alpha"]["catalog_store_id"] == "store:b"
    assert assets["profile_only"]["catalog_store_id"] == "store:b"
    assert assets["zeta"]["catalog_store_id"] == "store:a"


def test_monitor_and_failed_catalog_evidence_summaries_are_safe_and_bounded():
    secret = "TOP_SECRET_MONITOR_PAYLOAD"
    listing = _evidence_summary(
        Evidence(
            id="listing",
            kind="monitor.listing",
            owner="db_runtime",
            accepted=True,
            payload={
                "monitors": [
                    {
                        "id": f"monitor_{index:02d}",
                        "name": f"Monitor {index:02d}",
                        "status": "active",
                        "observation_plan": {"sql": secret},
                        "delivery": {"target": secret},
                    }
                    for index in reversed(range(25))
                ],
                "raw_prompt": secret,
            },
        )
    )
    inspection = _evidence_summary(
        Evidence(
            id="inspection",
            kind="monitor.inspection",
            owner="db_runtime",
            accepted=True,
            payload={
                "resolution": {
                    "monitor_id": "monitor_00",
                    "monitor_ref": "Monitor 00",
                    "resolution_source": "canonical_id",
                    "matches": [f"monitor_{index:02d}" for index in range(12)],
                    "warnings": [f"warning_{index:02d}" for index in range(12)],
                    "errors": [f"error_{index:02d}" for index in range(12)],
                    "metadata": {"secret": secret},
                },
                "inspection": {
                    "monitor": {
                        "id": "monitor_00",
                        "name": "Monitor 00",
                        "status": "active",
                        "observation_plan": {"sql": secret},
                    },
                    "runs": [{"rows": [secret]}],
                },
                "sql": secret,
            },
        )
    )
    proposal = _evidence_summary(
        Evidence(
            id="proposal",
            kind="monitor.proposal",
            owner="db_runtime",
            accepted=False,
            payload={
                "action": "pause",
                "monitor_id": "canonical-monitor",
                "validation": {
                    "accepted": False,
                    "errors": [f"validation_{index:02d}" for index in range(12)],
                    "governance": {"secret": secret},
                },
                "candidates": [
                    {
                        "monitor_id": f"candidate_{index:02d}",
                        "name": f"Candidate {index:02d}",
                        "definition": {"sql": secret},
                    }
                    for index in reversed(range(12))
                ],
                "before": {"observation_plan": {"sql": secret}},
                "after": {"delivery": {"target": secret}},
                "raw_prompt": secret,
                "sql": secret,
            },
        )
    )
    create_proposal = _evidence_summary(
        Evidence(
            id="create-proposal",
            kind="monitor.proposal",
            owner="db_runtime",
            accepted=True,
            payload={"monitor_id": "new-monitor", "validation": {"errors": []}},
        )
    )
    approval_state = _evidence_summary(
        Evidence(
            id="approval-state",
            kind="monitor.approval_state",
            owner="db_runtime",
            accepted=True,
            payload={
                "approvals": [
                    {
                        "approval_id": f"approval_{index:02d}",
                        "operation_id": f"operation_{index:02d}",
                        "status": "pending",
                        "requested_by_policy_id": "monitor-policy",
                        "context": {
                            "monitor_id": f"monitor_{index:02d}",
                            "governance": {"secret": secret},
                        },
                        "reason": secret,
                    }
                    for index in reversed(range(25))
                ]
            },
        )
    )
    approval_resolution = _evidence_summary(
        Evidence(
            id="approval-resolution",
            kind="monitor.approval_resolution",
            owner="db_runtime",
            accepted=True,
            payload={
                "status": "resolved",
                "approval_id": "approval_00",
                "approval_status": "approved",
                "operation_id": "operation_00",
                "matched_approvals": [
                    {
                        "approval_id": f"approval_{index:02d}",
                        "operation_id": f"operation_{index:02d}",
                        "status": "pending",
                        "requested_by_policy_id": "monitor-policy",
                        "context": {"monitor_id": f"monitor_{index:02d}"},
                        "governance": {"secret": secret},
                    }
                    for index in reversed(range(25))
                ],
                "governance": {"secret": secret},
            },
        )
    )
    failed_asset = _evidence_summary(
        Evidence(
            id="failed-asset",
            kind="schema.asset_profile",
            owner="catalog",
            accepted=False,
            payload={
                "success": False,
                "asset_ref": "pending orders",
                "candidates": [
                    {
                        "asset_ref": f"asset_{index:02d}",
                        "name": f"Asset {index:02d}",
                        "sample_rows": [{"secret": secret}],
                    }
                    for index in reversed(range(12))
                ],
                "sql": secret,
                "error": secret,
            },
        )
    )

    assert listing["monitor_count"] == 25
    assert listing["included_monitor_count"] == 20
    assert listing["monitors_truncated"] is True
    assert listing["monitors"][0] == {
        "id": "monitor_00",
        "name": "Monitor 00",
        "status": "active",
    }
    assert inspection["monitor_id"] == "monitor_00"
    assert inspection["monitor_name"] == "Monitor 00"
    assert inspection["monitor_status"] == "active"
    assert inspection["resolution"]["match_count"] == 12
    assert inspection["resolution"]["included_match_count"] == 10
    assert inspection["resolution"]["matches_truncated"] is True
    assert inspection["resolution"]["warning_count"] == 12
    assert inspection["resolution"]["included_warning_count"] == 10
    assert inspection["resolution"]["warnings_truncated"] is True
    assert proposal["action"] == "pause"
    assert proposal["monitor_id"] == "canonical-monitor"
    assert proposal["validation_error_count"] == 12
    assert proposal["included_validation_error_count"] == 10
    assert proposal["validation_errors_truncated"] is True
    assert proposal["candidate_count"] == 12
    assert proposal["included_candidate_count"] == 10
    assert proposal["candidates_truncated"] is True
    assert create_proposal["action"] == "create"
    assert approval_state["approval_count"] == 25
    assert approval_state["included_approval_count"] == 20
    assert approval_state["approvals_truncated"] is True
    assert approval_state["approvals"][0] == {
        "approval_id": "approval_00",
        "target_operation_id": "operation_00",
        "monitor_id": "monitor_00",
        "policy_id": "monitor-policy",
        "status": "pending",
    }
    assert approval_resolution["resolution_status"] == "resolved"
    assert approval_resolution["approval_id"] == "approval_00"
    assert approval_resolution["approval_status"] == "approved"
    assert approval_resolution["target_operation_id"] == "operation_00"
    assert approval_resolution["matched_approval_count"] == 25
    assert approval_resolution["included_matched_approval_count"] == 20
    assert approval_resolution["matched_approvals_truncated"] is True
    assert failed_asset["requested_asset"] == "pending orders"
    assert failed_asset["candidate_count"] == 12
    assert failed_asset["included_candidate_count"] == 10
    assert failed_asset["candidates_truncated"] is True
    assert failed_asset["candidates"][0] == {
        "asset_ref": "asset_00",
        "name": "Asset 00",
    }

    serialized = json.dumps(
        {
            "listing": listing,
            "inspection": inspection,
            "proposal": proposal,
            "approval_state": approval_state,
            "approval_resolution": approval_resolution,
            "failed_asset": failed_asset,
        },
        sort_keys=True,
    )
    assert secret not in serialized
    for forbidden in (
        "observation_plan",
        "delivery",
        "raw_prompt",
        "governance",
        "sample_rows",
        "sql",
    ):
        assert forbidden not in serialized


def test_monitor_evidence_summaries_bound_scalar_text_and_report_truncation():
    oversized = "x" * 100_000
    listing = _evidence_summary(
        Evidence(
            id="long-listing",
            kind="monitor.listing",
            owner="db_runtime",
            accepted=True,
            payload={
                "monitors": [{"id": oversized, "name": oversized, "status": oversized}]
            },
        )
    )
    inspection = _evidence_summary(
        Evidence(
            id="long-inspection",
            kind="monitor.inspection",
            owner="db_runtime",
            accepted=True,
            payload={
                "resolution": {
                    "monitor_id": "monitor_00",
                    "monitor_ref": oversized,
                    "warnings": [oversized],
                    "errors": [oversized],
                }
            },
        )
    )
    proposal = _evidence_summary(
        Evidence(
            id="long-proposal",
            kind="monitor.proposal",
            owner="db_runtime",
            accepted=False,
            payload={
                "monitor_id": oversized,
                "validation": {
                    "errors": [oversized, {"code": oversized}],
                },
            },
        )
    )

    monitor = listing["monitors"][0]
    assert len(monitor["id"]) == 48
    assert len(monitor["name"]) == 64
    assert len(monitor["status"]) == 256
    assert monitor["truncated_fields"] == ["id", "name", "status"]

    resolution = inspection["resolution"]
    assert len(resolution["monitor_ref"]) == 256
    assert resolution["truncated_fields"] == ["monitor_ref"]
    assert len(resolution["warnings"][0]) == 256
    assert resolution["warning_text_truncated_count"] == 1
    assert resolution["warnings_truncated"] is True
    assert len(resolution["errors"][0]) == 256
    assert resolution["error_text_truncated_count"] == 1
    assert resolution["errors_truncated"] is True

    assert len(proposal["monitor_id"]) == 48
    assert proposal["truncated_fields"] == ["monitor_id"]
    assert len(proposal["validation_errors"][0]) == 256
    assert proposal["validation_error_text_truncated_count"] == 2
    assert proposal["validation_errors_truncated"] is True


@pytest.mark.parametrize(
    ("kind", "payload", "accepted"),
    (
        (
            "monitor.listing",
            {"monitors": {"id": {"nested": "secret"}, "name": []}},
            True,
        ),
        (
            "monitor.inspection",
            {"resolution": "malformed", "inspection": {"monitor": []}},
            False,
        ),
        (
            "monitor.proposal",
            {
                "action": {"nested": "secret"},
                "validation": {"errors": {"nested": "secret"}},
                "candidates": {"id": {"nested": "secret"}},
            },
            False,
        ),
        (
            "monitor.approval_state",
            {"approvals": {"approval_id": {"nested": "secret"}}},
            True,
        ),
        (
            "monitor.approval_resolution",
            {"status": [], "matched_approvals": "malformed"},
            True,
        ),
        (
            "schema.asset_profile",
            {
                "success": False,
                "asset_ref": {"nested": "secret"},
                "candidates": {"name": {"nested": "secret"}},
            },
            False,
        ),
    ),
)
def test_bounded_evidence_summaries_tolerate_malformed_optional_fields(
    kind,
    payload,
    accepted,
):
    summary = _evidence_summary(
        Evidence(
            id=f"malformed-{kind}",
            kind=kind,
            owner="db_runtime",
            accepted=accepted,
            payload=payload,
        )
    )

    assert summary["kind"] == kind
    assert "secret" not in json.dumps(summary, sort_keys=True)


async def test_llm_query_plan_normalizes_sql_alias_and_accepts_executable_plan():
    sql = "select count(*) as order_count from orders"
    runtime, operation = await _runtime_and_operation(
        "phase-two-llm-plan-sql-alias",
        db_llm_service=FakeLLMService(
            json.dumps(
                {
                    "operation": "read",
                    "sql": sql,
                    "selected_tables": ["orders"],
                    "confidence": 0.91,
                }
            )
        ),
    )
    try:
        await runtime.store.save_evidence(_planning_context_evidence(operation.id))
        evidence = await DbLLMPlannerExecutor(runtime=runtime).execute(
            _llm_task(
                operation,
                capability_id="db.query.plan",
                executor_id="db_runtime.query.plan.llm",
                task_input={"planning_context_evidence_id": "planning-context"},
            ),
            operation,
            {},
        )
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    proposal = evidence[0]
    assert proposal.accepted is True
    assert proposal.payload["valid"] is True
    assert proposal.payload["sql"] == sql
    assert proposal.payload["structured_plan"]["selected_sql"] == sql
    assert "sql" not in proposal.payload["structured_plan"]
    assert proposal.payload["parse_diagnostics"]["normalized_aliases"] == {
        "sql": "selected_sql"
    }


async def test_llm_query_plan_without_sql_or_clarification_is_rejected():
    runtime, operation = await _runtime_and_operation(
        "phase-two-llm-plan-no-sql",
        db_llm_service=FakeLLMService(
            json.dumps(
                {
                    "operation": "read",
                    "selected_tables": ["orders"],
                    "confidence": 0.31,
                }
            )
        ),
    )
    try:
        await runtime.store.save_evidence(_planning_context_evidence(operation.id))
        evidence = await DbLLMPlannerExecutor(runtime=runtime).execute(
            _llm_task(
                operation,
                capability_id="db.query.plan",
                executor_id="db_runtime.query.plan.llm",
                task_input={"planning_context_evidence_id": "planning-context"},
            ),
            operation,
            {},
        )
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    proposal = evidence[0]
    assert proposal.accepted is False
    assert proposal.payload["valid"] is False
    assert proposal.payload["sql"] is None
    assert proposal.payload["raw_model_response"]


async def test_llm_query_plan_clarification_only_is_not_executable_evidence():
    runtime, operation = await _runtime_and_operation(
        "phase-two-llm-plan-clarification",
        db_llm_service=FakeLLMService(
            json.dumps(
                {
                    "operation": "read",
                    "clarification_question": "Which customer segment?",
                    "confidence": 0.25,
                }
            )
        ),
    )
    try:
        await runtime.store.save_evidence(_planning_context_evidence(operation.id))
        evidence = await DbLLMPlannerExecutor(runtime=runtime).execute(
            _llm_task(
                operation,
                capability_id="db.query.plan",
                executor_id="db_runtime.query.plan.llm",
                task_input={"planning_context_evidence_id": "planning-context"},
            ),
            operation,
            {},
        )
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    proposal = evidence[0]
    assert proposal.accepted is False
    assert proposal.payload["valid"] is False
    assert proposal.payload["sql"] is None
    assert proposal.payload["structured_plan"]["clarification_question"] == (
        "Which customer segment?"
    )


async def test_llm_repair_without_executable_sql_is_rejected():
    runtime, operation = await _repair_runtime_and_operation(
        "phase-two-llm-repair-no-sql",
        json.dumps({"operation": "read", "confidence": 0.4}),
    )
    try:
        task = _llm_task(
            operation,
            capability_id="db.query.repair",
            executor_id="db_runtime.query.repair.llm",
            task_input={
                "planning_context_evidence_id": "planning-context",
                "failure_evidence_id": "failure-validation",
                "prior_plan_evidence_id": "prior-plan",
            },
        )
        evidence = await DbLLMRepairExecutor(runtime=runtime).execute(
            task,
            operation,
            {},
        )
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    repair, proposal = evidence
    assert repair.accepted is False
    assert repair.payload["valid"] is True
    assert repair.payload["parse_succeeded"] is True
    assert repair.payload["proposal_accepted"] is False
    assert proposal.accepted is False
    assert proposal.payload["valid"] is False
    assert proposal.payload["sql"] is None


async def test_llm_repair_analysis_only_output_is_rejected_as_non_executable():
    runtime, operation = await _repair_runtime_and_operation(
        "phase-two-llm-repair-analysis-only",
        json.dumps(
            {
                "operation": "analysis",
                "revised_plan": {"steps": ["explain why the query failed"]},
                "confidence": 0.4,
            }
        ),
    )
    try:
        task = _llm_task(
            operation,
            capability_id="db.query.repair",
            executor_id="db_runtime.query.repair.llm",
            task_input={
                "planning_context_evidence_id": "planning-context",
                "failure_evidence_id": "failure-validation",
                "prior_plan_evidence_id": "prior-plan",
            },
        )
        evidence = await DbLLMRepairExecutor(runtime=runtime).execute(
            task,
            operation,
            {},
        )
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    repair, proposal = evidence
    assert repair.accepted is False
    assert repair.payload["failure"] == "repair_non_executable_plan"
    assert repair.payload["repair_rejection_reason"] == "operation_not_read:analysis"
    assert repair.payload["proposal_accepted"] is False
    assert proposal.accepted is False
    assert proposal.payload["failure"] == "repair_non_executable_plan"
    assert proposal.payload["valid"] is False


async def test_llm_repair_repeated_sql_is_rejected():
    sql = "select 1 as answer"
    runtime, operation = await _repair_runtime_and_operation(
        "phase-two-llm-repair-repeat",
        json.dumps(
            {
                "operation": "read",
                "selected_sql": sql,
                "confidence": 0.9,
            }
        ),
        prior_sql=sql,
    )
    try:
        task = _llm_task(
            operation,
            capability_id="db.query.repair",
            executor_id="db_runtime.query.repair.llm",
            task_input={
                "planning_context_evidence_id": "planning-context",
                "failure_evidence_id": "failure-validation",
                "prior_plan_evidence_id": "prior-plan",
            },
        )
        evidence = await DbLLMRepairExecutor(runtime=runtime).execute(
            task,
            operation,
            {},
        )
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    repair, proposal = evidence
    assert repair.accepted is False
    assert repair.payload["valid"] is True
    assert repair.payload["proposal_accepted"] is False
    assert repair.payload["repeated_sql_blocked"] is True
    assert proposal.accepted is False
    assert proposal.payload["valid"] is True
    assert proposal.payload["repeated_sql_blocked"] is True


async def test_llm_repair_repeated_sql_allowed_when_failure_context_changed():
    sql = "select 1 as answer"
    runtime, operation = await _repair_runtime_and_operation(
        "phase-two-llm-repair-repeat-context-changed",
        json.dumps(
            {
                "operation": "read",
                "selected_sql": sql,
                "confidence": 0.9,
            }
        ),
        prior_sql=sql,
        failure_payload={
            "valid": False,
            "planning_context_evidence_id": "previous-planning-context",
            "validation_facts": [
                {
                    "kind": "filter_literal_requires_grounding",
                    "table": "orders",
                    "column": "status",
                    "literal": "completed",
                }
            ],
        },
    )
    try:
        task = _llm_task(
            operation,
            capability_id="db.query.repair",
            executor_id="db_runtime.query.repair.llm",
            task_input={
                "planning_context_evidence_id": "planning-context",
                "failure_evidence_id": "failure-validation",
                "prior_plan_evidence_id": "prior-plan",
            },
        )
        evidence = await DbLLMRepairExecutor(runtime=runtime).execute(
            task,
            operation,
            {},
        )
        repair_messages = runtime.db_llm_service.messages
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    repair, proposal = evidence
    assert repair.accepted is True
    assert repair.payload["repeated_sql_blocked"] is False
    assert repair.payload["repeated_sql_allowed_context_changed"] is True
    assert proposal.accepted is True
    assert proposal.payload["repair_context_changed"] is True
    assert proposal.payload["repeated_sql_allowed_context_changed"] is True
    assert "current source of truth" in repair_messages[0]["content"]
    repair_request = json.loads(repair_messages[1]["content"])
    assert repair_request["repair_context_changed"] is True


async def test_llm_repair_missing_input_ids_produces_rejected_diagnostics():
    service = FakeLLMService(json.dumps({"operation": "read"}))
    runtime, operation = await _runtime_and_operation(
        "phase-two-llm-repair-missing-inputs",
        db_llm_service=service,
    )
    try:
        evidence = await DbLLMRepairExecutor(runtime=runtime).execute(
            _llm_task(
                operation,
                capability_id="db.query.repair",
                executor_id="db_runtime.query.repair.llm",
                task_input={},
            ),
            operation,
            {},
        )
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    repair, proposal = evidence
    assert service.messages is None
    assert repair.accepted is False
    assert proposal.accepted is False
    assert repair.payload["failure"] == "repair_inputs_missing"
    assert proposal.payload["failure"] == "repair_inputs_missing"
    assert set(repair.payload["missing_input_ids"]) == {
        "planning_context_evidence_id",
        "failure_evidence_id",
        "prior_plan_evidence_id",
    }


async def test_repair_query_plan_missing_input_ids_rejected_before_task_creation():
    runtime, operation = await _runtime_and_operation(
        "phase-two-repair-missing-input-compile",
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    loop = DbAgentLoop(runtime, FakePlanner())
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": "read"},
        turn=1,
        remaining_turns=1,
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="repair",
                kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
                input={"owner": "db_runtime"},
            ),
        ),
    )

    try:
        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.task_specs == ()
    assert (
        compilation.rejected_action_summaries[0]["error"]
        == "missing_repair_failure_evidence"
    )


async def test_repair_query_plan_binds_durable_inputs_and_dependencies():
    action = DbPlannerAction(
        action_id="repair",
        kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
        input={"owner": "db_runtime"},
        depends_on=("a3",),
    )

    compilation, runtime, _ = await _compile_single_action(
        "phase-three-repair-durable-inputs",
        action,
        plan_evidence=(
            {
                "evidence_id": "prior-plan",
                "sql": "select count(*) from orders where status = 'completed'",
            },
        ),
        query_plan_validation_evidence=(
            {
                "evidence_id": "failed-plan-validation",
                "plan_evidence_id": "prior-plan",
            },
        ),
        planning_context=True,
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    repair = next(
        spec
        for spec in compilation.task_specs
        if spec.capability_id == "db.query.repair"
    )
    assert repair.capability_id == "db.query.repair"
    assert repair.input["planning_context_evidence_id"] == "planning-context"
    assert repair.input["prior_plan_evidence_id"] == "prior-plan"
    assert repair.input["failure_evidence_id"] == "failed-plan-validation"
    assert "query_plan_ref" not in repair.input
    assert "plan_evidence_id" not in repair.input
    assert [dependency.evidence_id for dependency in repair.dependencies] == [
        "planning-context",
        "prior-plan",
        "failed-plan-validation",
    ]
    assert [dependency.evidence_kind for dependency in repair.dependencies] == [
        "planning.context",
        "query.plan.proposal",
        "query.plan.validation",
    ]
    assert [dependency.evidence_accepted for dependency in repair.dependencies] == [
        True,
        True,
        False,
    ]


async def test_same_decision_repair_execute_defers_until_repaired_plan_is_durable():
    stale_sql = "select count(*) from orders where status = 'completed'"
    runtime, operation = await _runtime_and_operation(
        "phase-seven-same-turn-repair-defers",
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    try:
        await runtime.store.save_evidence(_planning_context_evidence(operation.id))
        await runtime.store.save_evidence(
            _query_plan_evidence(
                operation.id,
                evidence_id="prior-plan",
                sql=stale_sql,
            )
        )
        await runtime.store.save_evidence(
            _query_plan_validation_evidence(
                operation.id,
                evidence_id="failed-plan-validation",
                plan_evidence_id="prior-plan",
                validation_facts=False,
            )
        )
        loop = DbAgentLoop(runtime, FakePlanner())
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "read"},
            turn=1,
            remaining_turns=1,
        )
        decision = _same_turn_repair_execute_decision()

        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.query.repair"
    ]
    repair = compilation.task_specs[0]
    assert repair.input["prior_plan_evidence_id"] == "prior-plan"
    assert repair.input["failure_evidence_id"] == "failed-plan-validation"
    rejected = compilation.rejected_action_summaries
    assert len(rejected) == 1
    assert rejected[0]["action_id"] == "execute_repair"
    assert rejected[0]["error"] == "deferred_until_query_plan_proposal_available"
    assert rejected[0]["deferred"]["producer_action_ids"] == ["repair_plan"]
    assert rejected[0]["deferred"]["non_terminal"] is True
    assert not any(
        spec.capability_id in {"db.query.plan.validate", "db.sql.validate"}
        for spec in compilation.task_specs
    )
    dumped_specs = json.dumps(
        [spec.to_dict() for spec in compilation.task_specs],
        sort_keys=True,
    )
    assert stale_sql not in dumped_specs


async def test_same_turn_repair_deferral_continues_after_repair_evidence():
    stale_sql = "select count(*) as order_count from orders where status = 'completed'"
    repaired_sql = (
        "select count(*) as order_count from orders where status = 'complete'"
    )
    runtime, operation = await _runtime_and_operation(
        "phase-seven-same-turn-repair-continues",
        db_llm_service=FakeLLMService(
            json.dumps(
                {
                    "operation": "read",
                    "selected_sql": repaired_sql,
                    "selected_tables": ["orders"],
                    "confidence": 0.92,
                }
            )
        ),
        required_evidence={"query.result"},
    )
    try:
        await runtime.store.save_evidence(_planning_context_evidence(operation.id))
        await runtime.store.save_evidence(
            _query_plan_evidence(
                operation.id,
                evidence_id="prior-plan",
                sql=stale_sql,
            )
        )
        await runtime.store.save_evidence(
            _query_plan_validation_evidence(
                operation.id,
                evidence_id="failed-plan-validation",
                plan_evidence_id="prior-plan",
                validation_facts=False,
            )
        )
        result = await DbAgentLoop(
            runtime,
            FakePlanner(
                _same_turn_repair_execute_decision(),
                _planner_decision(
                    DbPlannerAction(
                        action_id="execute_repaired_durable_plan",
                        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                        input={
                            "owner": "phase_two",
                            "query_plan_ref": "latest_accepted_query_plan",
                        },
                    )
                ),
            ),
        ).run(
            operation,
            safety_frame={"max_access": "read"},
            max_turns=3,
        )
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    assert result.status == "finished"
    compilation = next(
        item
        for item in evidence
        if item.kind == "planner.compilation"
        and any(
            rejected["error"] == "deferred_until_query_plan_proposal_available"
            for rejected in item.payload["compilation"]["rejected_action_summaries"]
        )
    )
    assert compilation.accepted is False
    assert [
        spec["capability_id"]
        for spec in compilation.payload["compilation"]["task_specs"]
    ] == ["db.query.repair"]
    sql_validation_tasks = [
        task for task in tasks if task.capability_id == "db.sql.validate"
    ]
    assert [task.input["sql"] for task in sql_validation_tasks] == [repaired_sql]
    assert all(task.input["sql"] != stale_sql for task in sql_validation_tasks)


async def test_repair_query_plan_falls_back_to_failed_sql_validation():
    action = DbPlannerAction(
        action_id="repair",
        kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
        input={"owner": "db_runtime"},
    )

    compilation, runtime, _ = await _compile_single_action(
        "phase-three-repair-failed-sql-validation",
        action,
        plan_evidence=(
            {"evidence_id": "prior-plan", "sql": "select * from missing_table"},
        ),
        sql_validation_evidence=(
            {
                "evidence_id": "failed-sql-validation",
                "sql": "select * from missing_table",
                "valid": False,
                "accepted": False,
            },
        ),
        planning_context=True,
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    repair = compilation.task_specs[0]
    assert repair.input["failure_evidence_id"] == "failed-sql-validation"
    assert repair.dependencies[-1].evidence_id == "failed-sql-validation"
    assert repair.dependencies[-1].evidence_kind == "sql.validation"
    assert repair.dependencies[-1].evidence_accepted is False


async def test_repair_query_plan_blocks_without_failed_validation_evidence():
    action = DbPlannerAction(
        action_id="repair",
        kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
        input={"owner": "db_runtime"},
    )

    compilation, runtime, _ = await _compile_single_action(
        "phase-three-repair-no-failure-evidence",
        action,
        plan_evidence=({"evidence_id": "prior-plan", "sql": "select 1"},),
        planning_context=True,
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    await runtime.teardown()

    assert compilation.task_specs == ()
    assert (
        compilation.rejected_action_summaries[0]["error"]
        == "missing_repair_failure_evidence"
    )


async def test_repair_query_plan_blocks_ambiguous_failure_evidence_id():
    action = DbPlannerAction(
        action_id="repair",
        kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
        input={
            "owner": "db_runtime",
            "failure_evidence_id": "ambiguous-failure",
        },
    )

    compilation, runtime, _ = await _compile_single_action(
        "phase-three-repair-ambiguous-failure",
        action,
        plan_evidence=({"evidence_id": "prior-plan", "sql": "select 1"},),
        query_plan_validation_evidence=(
            {
                "evidence_id": "ambiguous-failure",
                "plan_evidence_id": "prior-plan",
                "validation_facts": False,
            },
        ),
        sql_validation_evidence=(
            {
                "evidence_id": "ambiguous-failure",
                "sql": "select 1",
                "valid": False,
                "accepted": False,
            },
        ),
        planning_context=True,
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    await runtime.teardown()

    assert compilation.task_specs == ()
    assert (
        compilation.rejected_action_summaries[0]["error"]
        == "ambiguous_repair_failure_evidence"
    )


async def test_propose_sql_read_without_context_compiles_context_prerequisite():
    runtime, operation = await _runtime_and_operation(
        "phase-two-propose-builds-context",
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    loop = DbAgentLoop(runtime, FakePlanner())
    try:
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "metadata_read"},
            turn=1,
            remaining_turns=1,
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=(
                DbPlannerAction(
                    action_id="plan",
                    kind=DbPlannerActionKind.PROPOSE_SQL_READ,
                    input={"owner": "db_runtime"},
                ),
            ),
        )

        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.schema.inspect",
        "db.planning.context.build",
        "db.query.plan",
    ]
    context, query_plan = compilation.task_specs[1:]
    assert "planning_context_evidence_id" not in query_plan.input
    assert len(query_plan.dependencies) == 1
    dependency = query_plan.dependencies[0]
    assert dependency.evidence_kind == "planning.context"
    assert dependency.evidence_owner == "db_runtime"
    assert dependency.producer_task_id == context.task_id
    assert dependency.producer_capability_id == "db.planning.context.build"
    assert dependency.producer_executor_id == "db_runtime.planning.context.build"
    assert dependency.evidence_accepted is True
    assert dependency.evidence_id is None


async def test_propose_sql_read_strips_unsupported_plan_refs_before_query_plan():
    runtime, operation = await _runtime_and_operation(
        "phase-two-propose-strips-refs",
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    loop = DbAgentLoop(runtime, FakePlanner())
    try:
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "metadata_read"},
            turn=1,
            remaining_turns=1,
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=(
                DbPlannerAction(
                    action_id="plan",
                    kind=DbPlannerActionKind.PROPOSE_SQL_READ,
                    input={
                        "owner": "db_runtime",
                        "query_plan_ref": "latest_accepted_query_plan",
                        "plan_evidence_id": "prior-plan",
                    },
                ),
            ),
        )

        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.schema.inspect",
        "db.planning.context.build",
        "db.query.plan",
    ]
    query_plan = compilation.task_specs[-1]
    assert "query_plan_ref" not in query_plan.input
    assert "plan_evidence_id" not in query_plan.input
    assert query_plan.dependencies[0].evidence_kind == "planning.context"
    assert query_plan.dependencies[0].producer_task_id == (
        compilation.task_specs[-2].task_id
    )


async def test_propose_sql_read_with_existing_context_injects_context_evidence_id():
    action = DbPlannerAction(
        action_id="plan",
        kind=DbPlannerActionKind.PROPOSE_SQL_READ,
        input={
            "owner": "db_runtime",
            "query_plan_ref": "latest_accepted_query_plan",
            "plan_evidence_id": "prior-plan",
        },
    )

    compilation, runtime, _ = await _compile_single_action(
        "phase-two-propose-existing-context",
        action,
        planning_context=True,
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == ["db.query.plan"]
    query_plan = compilation.task_specs[0]
    assert query_plan.input == {"planning_context_evidence_id": "planning-context"}
    assert query_plan.dependencies == ()


async def test_propose_sql_read_does_not_reuse_catalog_hint_only_context():
    runtime, operation = await _catalog_runtime_and_operation(
        "phase-two-catalog-hint-context"
    )
    try:
        await runtime.store.save_evidence(
            Evidence(
                id="hint-only-context",
                kind="planning.context",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "schema": {
                        "database_type": "sqlite",
                        "tables": [
                            {
                                "name": "orders",
                                "columns": [{"name": "status"}],
                            }
                        ],
                    },
                    "catalog_evidence_refs": ["catalog-status-hints"],
                    "diagnostics": {
                        "structural_schema_source": "connector",
                        "catalog_structural_evidence_refs": [],
                    },
                },
                metadata={"payload_fingerprint": "fp-hint-only-context"},
            )
        )
        loop = DbAgentLoop(runtime, FakePlanner())
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "metadata_read"},
            turn=1,
            remaining_turns=1,
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=(
                DbPlannerAction(
                    action_id="plan",
                    kind=DbPlannerActionKind.PROPOSE_SQL_READ,
                    input={"owner": "db_runtime"},
                ),
            ),
        )

        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.schema.inspect",
        "catalog.schema.search",
        "db.planning.context.build",
        "db.query.plan",
    ]
    query_plan = compilation.task_specs[-1]
    assert "planning_context_evidence_id" not in query_plan.input
    assert query_plan.dependencies[0].producer_task_id == (
        compilation.task_specs[-2].task_id
    )


async def test_propose_sql_read_inserts_catalog_search_and_asset_prerequisites():
    runtime, operation = await _catalog_runtime_and_operation(
        "phase-two-catalog-prereqs"
    )
    loop = DbAgentLoop(runtime, FakePlanner())
    try:
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "metadata_read"},
            turn=1,
            remaining_turns=1,
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=(
                DbPlannerAction(
                    action_id="plan",
                    kind=DbPlannerActionKind.PROPOSE_SQL_READ,
                    input={
                        "owner": "db_runtime",
                        "source_owner": "sqlite",
                        "tables": ["orders"],
                    },
                ),
            ),
        )

        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.schema.inspect",
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.planning.context.build",
        "db.query.plan",
    ]
    search = compilation.task_specs[1]
    asset = compilation.task_specs[2]
    context = compilation.task_specs[3]
    query_plan = compilation.task_specs[4]
    assert search.input["query"] == "phase two catalog"
    assert asset.input["asset_ref"] == "orders"
    assert asset.dependencies[0].producer_task_id == search.task_id
    assert {dependency.evidence_kind for dependency in context.dependencies} == {
        "schema.asset_profile",
        "schema.search_result",
    }
    assert query_plan.dependencies[0].producer_task_id == context.task_id


async def test_relationship_path_action_does_not_guess_assets_from_prompt_text():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
    )
    await runtime.setup(agent_id="agent-phase-two")
    try:
        operation = await runtime.kernel.create_operation(
            operation_id="phase-two-no-prompt-relationship-guessing",
            operation_type="data.query",
            request={
                "prompt": "Join orders to customers using their relationship",
                "source_scope": ["sqlite"],
            },
            metadata={"source_scope": ["sqlite"]},
            evaluate_governance=False,
        )
        loop = DbAgentLoop(runtime, FakePlanner())
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "metadata_read"},
            turn=1,
            remaining_turns=1,
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=(
                DbPlannerAction(
                    action_id="relationships",
                    kind=DbPlannerActionKind.FIND_RELATIONSHIP_PATHS,
                    input={"owner": "catalog"},
                ),
            ),
        )

        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.task_specs == ()
    assert [item["error"] for item in compilation.rejected_action_summaries] == [
        "missing_from_assets",
        "missing_to_assets",
    ]


async def test_execute_joined_query_plan_inserts_relationship_path_prerequisite():
    sql = (
        "SELECT o.id, c.email FROM orders o " "JOIN customers c ON o.customer_id = c.id"
    )
    runtime, operation = await _catalog_runtime_and_operation(
        "phase-two-relationship-prereq"
    )
    try:
        await runtime.store.save_evidence(_planning_context_evidence(operation.id))
        await runtime.store.save_evidence(
            Evidence(
                id="join-plan",
                kind="query.plan.proposal",
                owner="db_runtime",
                operation_id=operation.id,
                task_id="task-join-plan",
                accepted=True,
                payload={
                    "valid": True,
                    "sql": sql,
                    "structured_plan": {
                        "operation": "read",
                        "selected_sql": sql,
                        "selected_tables": ["orders", "customers"],
                        "joins": [
                            {
                                "left_table": "orders",
                                "left_column": "customer_id",
                                "right_table": "customers",
                                "right_column": "id",
                            }
                        ],
                        "confidence": 0.95,
                    },
                },
                metadata={"payload_fingerprint": "fp-join-plan"},
            )
        )
        loop = DbAgentLoop(runtime, FakePlanner())
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "read"},
            turn=1,
            remaining_turns=1,
        )
        decision = _planner_decision(
            DbPlannerAction(
                action_id="execute_join",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "sqlite",
                    "plan_evidence_id": "join-plan",
                },
            )
        )

        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "catalog.schema.search",
        "catalog.relationship_paths.find",
        "db.planning.context.build",
        "db.query.plan.validate",
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    relationship = compilation.task_specs[1]
    refreshed_context = compilation.task_specs[2]
    plan_validation = compilation.task_specs[3]
    assert relationship.input["from_assets"] == ["orders"]
    assert relationship.input["to_assets"] == ["customers"]
    assert refreshed_context.dependencies[-1].producer_task_id == relationship.task_id
    assert "planning_context_evidence_id" not in plan_validation.input
    assert any(
        dependency.evidence_kind == "planning.context"
        and dependency.producer_task_id == refreshed_context.task_id
        for dependency in plan_validation.dependencies
    )


async def test_propose_sql_read_context_prerequisite_uses_memory_recall_chain():
    runtime = DbRuntime(
        plugins=(MemoryPlugin(),),
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.0,
                        "retrieval_mode": "structured",
                        "source_identity": "sqlite:from_db:propose-memory",
                    }
                }
            }
        ),
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    await runtime.setup(agent_id="agent-phase-two")
    try:
        state = DbLoopState(
            operation_id="op-propose-memory-context",
            normalized_user_request={
                "prompt": "Calculate recognized revenue from orders.total."
            },
            safety_frame={"max_access": "metadata_read"},
            available_action_kinds=tuple(DbPlannerActionKind),
            accepted_evidence_summaries=(
                {
                    "id": "schema-existing",
                    "kind": "schema.asset_profile",
                    "owner": "sqlite",
                    "accepted": True,
                    "task_id": "schema-task",
                },
            ),
            memory_context={
                "enabled": True,
                "source_identity": "sqlite:from_db:propose-memory",
                "retrieval_mode": "structured",
                "limit": 3,
                "score_threshold": 0.0,
                "recall_decision": {
                    "recall": True,
                    "reason": "semantic_prompt",
                    "query": "recognized revenue orders.total complete",
                },
            },
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=(
                DbPlannerAction(
                    action_id="plan",
                    kind=DbPlannerActionKind.PROPOSE_SQL_READ,
                    input={"owner": "db_runtime"},
                ),
            ),
        )

        compilation = DbAgentLoop(runtime, FakePlanner()).compile_actions(
            decision,
            state,
        )
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "memory.semantic.recall",
        "db.planning.context.build",
        "db.query.plan",
    ]
    recall, context, query_plan = compilation.task_specs
    assert context.dependencies[0].evidence_kind == "memory.semantic.recall"
    assert context.dependencies[0].producer_task_id == recall.task_id
    assert context.input["memory_recall_diagnostics"]["queried"] is True
    assert query_plan.dependencies[0].evidence_kind == "planning.context"
    assert query_plan.dependencies[0].producer_task_id == context.task_id


async def test_agent_loop_runs_schema_and_read_flow_through_task_specs():
    runtime, operation = await _runtime_and_operation("phase-two-read")
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "db.run"},
        actions=(
            DbPlannerAction(
                action_id="schema",
                kind=DbPlannerActionKind.INSPECT_SCHEMA,
                input={"owner": "phase_two"},
            ),
            DbPlannerAction(
                action_id="read",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={"owner": "phase_two", "sql": "select 1 as answer"},
            ),
        ),
    )

    result = await DbAgentLoop(runtime, FakePlanner(decision)).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )

    assert result.status == "finished"
    tasks = await runtime.store.list_tasks(operation.id)
    assert [task.capability_id for task in tasks] == [
        "db.schema.inspect",
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert [task.status.value for task in tasks] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    read_task = tasks[-1]
    assert read_task.dependencies[0].producer_task_id == tasks[1].id
    evidence = await runtime.store.list_evidence(operation.id)
    kinds = [item.kind for item in evidence]
    assert kinds.index("planner.decision") < kinds.index("database.schema")
    assert {"planner.decision", "planner.compilation", "planner.observation"} <= set(
        kinds
    )
    query_result = next(item for item in evidence if item.kind == "query.result")
    assert query_result.payload["sql"] == "select 1 as answer"


async def test_premature_finish_cannot_complete_a_data_query_without_result():
    runtime, operation = await _runtime_and_operation(
        "phase-zero-premature-finish",
        required_evidence=frozenset({"query.result"}),
    )
    planner = FakePlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.FINISH,
            intent={"operation_type": "data.query"},
            actions=(),
        )
    )
    try:
        result = await DbAgentLoop(runtime, planner).run(operation, max_turns=1)
    finally:
        await runtime.teardown()

    assert result.status != "finished"
    assert not any(
        item.kind == "query.result"
        for item in await runtime.store.list_evidence(operation.id)
    )


async def test_direct_sql_compiles_validation_and_read_task_specs():
    sql = "select 1 as answer"
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "sql": sql},
    )

    compilation, _, _ = await _compile_single_action("phase-four-direct", action)

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    validation, read = compilation.task_specs
    assert validation.input == {"sql": sql, "operation": "query"}
    assert validation.dependencies == ()
    assert validation.metadata["sql_provenance"]["provenance"] == "direct"
    assert "sql_ref" not in read.input
    assert read.input["params"] == []
    assert read.input["sql_fingerprint"]
    assert read.input["sql_validation_task_id"] == validation.task_id
    assert read.input["sql_validation_input_hash"]
    sql_dependency = next(
        dependency
        for dependency in read.dependencies
        if dependency.evidence_kind == "sql.validation"
    )
    assert sql_dependency.producer_task_id == validation.task_id
    assert sql_dependency.input_hash == read.input["sql_validation_input_hash"]


async def test_explicit_plan_evidence_id_attaches_dependency_to_validation():
    sql = "select count(*) as count from orders"
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-explicit"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-explicit-plan",
        action,
        plan_evidence=(
            {"evidence_id": "plan-explicit", "sql": sql, "task_id": "plan-task"},
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation, read = compilation.task_specs
    assert validation.input == {"sql": sql, "operation": "query"}
    assert "sql_ref" not in read.input
    assert read.input["plan_evidence_id"] == "plan-explicit"
    assert read.input["plan_payload_fingerprint"] == "fp-plan-explicit"
    assert read.input["sql_validation_task_id"] == validation.task_id
    assert len(validation.dependencies) == 1
    dependency = validation.dependencies[0]
    assert dependency.evidence_kind == "query.plan.proposal"
    assert dependency.evidence_id == "plan-explicit"
    assert dependency.evidence_owner == "phase_two"
    assert dependency.producer_task_id == "plan-task"
    assert dependency.payload_fingerprint == "fp-plan-explicit"
    provenance = validation.metadata["sql_provenance"]
    assert provenance["provenance"] == "plan_evidence_id"
    assert provenance["source_evidence_id"] == "plan-explicit"
    assert provenance["source_evidence_kind"] == "query.plan.proposal"
    assert provenance["source_evidence_owner"] == "phase_two"
    assert provenance["source_task_id"] == "plan-task"
    assert provenance["source_payload_fingerprint"] == "fp-plan-explicit"
    assert provenance["sql_fingerprint"]


async def test_latest_accepted_query_plan_ref_selects_latest_accepted_plan():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-latest-plan",
        action,
        plan_evidence=(
            {"evidence_id": "plan-old", "sql": "select 1 as answer"},
            {"evidence_id": "plan-new", "sql": "select 2 as answer"},
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 2 as answer"
    assert validation.dependencies[0].evidence_id == "plan-new"
    assert (
        validation.metadata["sql_provenance"]["provenance"]
        == "latest_accepted_query_plan"
    )
    continuation = validation.metadata["continuation_resolution"]
    assert continuation["source"] == "explicit_role"
    assert continuation["role"] == "latest_accepted_query_plan"
    assert continuation["evidence_id"] == "plan-new"


async def test_explicit_plan_id_matching_latest_plan_ref_is_not_ambiguous():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "plan_evidence_id": "plan-new",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-explicit-plan-and-latest-ref",
        action,
        plan_evidence=(
            {"evidence_id": "plan-old", "sql": "select 1 as answer"},
            {"evidence_id": "plan-new", "sql": "select 2 as answer"},
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 2 as answer"
    assert validation.dependencies[0].evidence_id == "plan-new"
    provenance = validation.metadata["sql_provenance"]
    assert provenance["provenance"] == "plan_evidence_id"
    assert provenance["source_evidence_id"] == "plan-new"


async def test_explicit_plan_id_conflicting_with_latest_plan_ref_is_ambiguous():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "plan_evidence_id": "plan-old",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-explicit-plan-conflicts-with-latest-ref",
        action,
        plan_evidence=(
            {"evidence_id": "plan-old", "sql": "select 1 as answer"},
            {"evidence_id": "plan-new", "sql": "select 2 as answer"},
        ),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == "ambiguous_sql_input"


async def test_prior_turn_query_plan_dependency_recovers_to_latest_plan():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two"},
        depends_on=("plan_previous_turn",),
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-prior-turn-plan-dependency",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"
    assert (
        validation.metadata["continuation_resolution"]["evidence_id"] == "plan-accepted"
    )
    assert (
        validation.metadata["sql_provenance"]["provenance"]
        == "latest_accepted_query_plan"
    )


async def test_prior_turn_latest_plan_ref_dependency_recovers_to_latest_plan():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
        depends_on=("plan_previous_turn",),
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-prior-turn-plan-ref-dependency",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"
    assert (
        validation.metadata["sql_provenance"]["provenance"]
        == "latest_accepted_query_plan"
    )
    assert validation.metadata["dependency_recovery"] == "latest_accepted_query_plan"
    continuation = validation.metadata["continuation_resolution"]
    assert continuation["source"] == "explicit_role"
    assert continuation["stale_dependency_ids"] == ["plan_previous_turn"]


async def test_prior_turn_dependency_with_explicit_plan_evidence_id_preserves_id():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-explicit"},
        depends_on=("plan_previous_turn",),
    )

    compilation, _, _ = await _compile_single_action(
        "phase-five-explicit-id-continuation",
        action,
        plan_evidence=(
            {
                "evidence_id": "plan-explicit",
                "sql": "select 1 as answer",
                "task_id": "plan-task",
            },
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-explicit"
    provenance = validation.metadata["sql_provenance"]
    assert provenance["provenance"] == "plan_evidence_id"
    assert provenance["source_evidence_id"] == "plan-explicit"
    continuation = validation.metadata["continuation_resolution"]
    assert continuation["source"] == "explicit_evidence_id"
    assert continuation["evidence_id"] == "plan-explicit"
    assert continuation["stale_dependency_ids"] == ["plan_previous_turn"]


async def test_prior_turn_query_plan_dependency_blocks_ambiguous_candidates():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two"},
        depends_on=("plan_previous_turn",),
    )

    compilation, _, _ = await _compile_single_action(
        "phase-five-ambiguous-plan-continuation",
        action,
        plan_evidence=(
            {"evidence_id": "plan-one", "sql": "select 1 as answer"},
            {"evidence_id": "plan-two", "sql": "select 2 as answer"},
        ),
    )

    assert compilation.task_specs == ()
    rejected = compilation.rejected_action_summaries[0]
    assert rejected["error"] == "ambiguous_continuation:latest_accepted_query_plan"
    assert rejected["continuation"]["status"] == "blocked"
    assert rejected["continuation"]["candidate_ids"] == ["plan-one", "plan-two"]
    assert rejected["continuation"]["stale_dependency_ids"] == ["plan_previous_turn"]


async def test_memory_proposal_role_ref_resolves_to_latest_accepted_proposal():
    action = DbPlannerAction(
        action_id="commit",
        kind=DbPlannerActionKind.COMMIT_MEMORY_UPDATE,
        input={
            "owner": "phase_two",
            "proposal_ref": "latest_accepted_memory_proposal",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-five-memory-role-ref",
        action,
        memory_proposal_evidence=(
            {
                "evidence_id": "proposal-old",
                "proposal_fingerprint": "proposal-fp-old",
            },
            {
                "evidence_id": "proposal-new",
                "proposal_fingerprint": "proposal-fp-new",
            },
        ),
        explicit_mode="memory.update",
        max_access="write",
    )

    assert compilation.rejected_action_summaries == ()
    commit = compilation.task_specs[0]
    assert commit.input == {
        "proposal_evidence_id": "proposal-new",
        "proposal_fingerprint": "proposal-fp-new",
    }
    assert commit.dependencies[0].evidence_id == "proposal-new"
    continuation = commit.metadata["continuation_resolution"]
    assert continuation["source"] == "explicit_role"
    assert continuation["role"] == "latest_accepted_memory_proposal"
    assert continuation["evidence_id"] == "proposal-new"


async def test_memory_commit_without_role_blocks_ambiguous_proposals():
    action = DbPlannerAction(
        action_id="commit",
        kind=DbPlannerActionKind.COMMIT_MEMORY_UPDATE,
        input={"owner": "phase_two"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-five-memory-ambiguous",
        action,
        memory_proposal_evidence=(
            {"evidence_id": "proposal-one"},
            {"evidence_id": "proposal-two"},
        ),
        explicit_mode="memory.update",
        max_access="write",
    )

    assert compilation.task_specs == ()
    rejected = compilation.rejected_action_summaries[0]
    assert rejected["error"] == "ambiguous_continuation:latest_accepted_memory_proposal"
    assert rejected["continuation"]["candidate_ids"] == [
        "proposal-one",
        "proposal-two",
    ]


async def test_memory_update_runtime_continuation_compiles_commit_without_action():
    runtime, operation = await _runtime_and_operation(
        "phase-six-memory-runtime-continuation",
        mode="memory.update",
    )
    await runtime.store.save_evidence(
        _memory_proposal_evidence(
            operation.id,
            evidence_id="proposal-runtime",
            proposal_fingerprint="proposal-fp-runtime",
        )
    )
    loop = DbAgentLoop(runtime, FakePlanner())
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": "write"},
        turn=1,
        remaining_turns=1,
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "memory.update"},
        actions=(),
    )

    compilation = loop.compile_actions(decision, state)

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.memory.commit_update"
    ]
    commit = compilation.task_specs[0]
    assert commit.input == {
        "proposal_evidence_id": "proposal-runtime",
        "proposal_fingerprint": "proposal-fp-runtime",
    }
    assert commit.dependencies[0].evidence_id == "proposal-runtime"
    assert commit.metadata["runtime_continuation"] is True
    continuation = commit.metadata["continuation_resolution"]
    assert continuation["source"] == "runtime_continuation"
    assert continuation["role"] == "latest_uncommitted_memory_proposal"
    assert continuation["evidence_id"] == "proposal-runtime"


async def test_memory_update_runtime_continuation_skips_committed_proposal():
    runtime, operation = await _runtime_and_operation(
        "phase-six-memory-runtime-committed",
        mode="memory.update",
    )
    await runtime.store.save_evidence(
        _memory_proposal_evidence(
            operation.id,
            evidence_id="proposal-committed",
            proposal_fingerprint="proposal-fp-committed",
        )
    )
    await runtime.store.save_evidence(
        _memory_definition_evidence(
            operation.id,
            evidence_id="definition-committed",
            proposal_evidence_id="proposal-committed",
            proposal_fingerprint="proposal-fp-committed",
        )
    )
    loop = DbAgentLoop(runtime, FakePlanner())
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": "write"},
        turn=1,
        remaining_turns=1,
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "memory.update"},
        actions=(),
    )

    compilation = loop.compile_actions(decision, state)

    assert compilation.rejected_action_summaries == ()
    assert compilation.task_specs == ()


async def test_memory_update_runtime_continuation_blocks_multiple_proposals():
    runtime, operation = await _runtime_and_operation(
        "phase-six-memory-runtime-ambiguous",
        mode="memory.update",
    )
    for evidence_id in ("proposal-one", "proposal-two"):
        await runtime.store.save_evidence(
            _memory_proposal_evidence(operation.id, evidence_id=evidence_id)
        )
    loop = DbAgentLoop(runtime, FakePlanner())
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": "write"},
        turn=1,
        remaining_turns=1,
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "memory.update"},
        actions=(),
    )

    compilation = loop.compile_actions(decision, state)

    assert compilation.task_specs == ()
    rejected = compilation.rejected_action_summaries[0]
    assert (
        rejected["error"] == "ambiguous_continuation:latest_uncommitted_memory_proposal"
    )
    assert rejected["continuation"]["source"] == "runtime_continuation"
    assert rejected["continuation"]["status"] == "blocked"
    assert rejected["continuation"]["candidate_ids"] == [
        "proposal-one",
        "proposal-two",
    ]


async def test_direct_sql_matching_latest_plan_ref_uses_plan_provenance():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "sql": "select 1 as answer;",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-direct-sql-matching-latest-plan",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"
    assert (
        validation.metadata["sql_provenance"]["provenance"]
        == "latest_accepted_query_plan"
    )


async def test_direct_sql_mismatching_latest_plan_ref_is_ambiguous():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "sql": "select 2 as answer",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-direct-sql-mismatching-latest-plan",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == "ambiguous_sql_input"


async def test_write_execute_with_direct_sql_and_plan_ref_uses_matching_validation():
    sql = "UPDATE orders SET status = 'approved' WHERE order_id = 101"
    action = DbPlannerAction(
        action_id="write",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
        input={
            "owner": "phase_two",
            "sql": f"{sql};",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-write-from-validation",
        action,
        sql_validation_evidence=(
            {
                "evidence_id": "validation-write",
                "sql": sql,
                "operation": "write",
                "task_id": "validation-task",
            },
        ),
        explicit_mode="write.execute",
        max_access="write",
    )

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.sql.execute_write",
    ]
    (write,) = compilation.task_specs
    assert "sql_ref" not in write.input
    assert write.input["sql_validation_evidence_id"] == "validation-write"
    assert write.input["sql_validation_payload_fingerprint"] == "fp-validation-write"
    dependency = write.dependencies[0]
    assert dependency.evidence_kind == "sql.validation"
    assert dependency.evidence_id == "validation-write"
    assert dependency.payload_fingerprint == "fp-validation-write"
    assert dependency.producer_task_id == "validation-task"
    provenance = write.metadata["sql_provenance"]
    assert provenance["provenance"] == "latest_accepted_sql_validation"
    assert provenance["source_evidence_id"] == "validation-write"
    assert provenance["source_evidence_kind"] == "sql.validation"
    assert provenance["source_task_id"] == "validation-task"


async def test_write_execute_plan_evidence_id_can_reference_sql_validation():
    sql = "UPDATE orders SET status = 'approved' WHERE order_id = 101"
    action = DbPlannerAction(
        action_id="write",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
        input={
            "owner": "phase_two",
            "sql": sql,
            "plan_evidence_id": "validation-write",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-write-validation-id",
        action,
        sql_validation_evidence=(
            {
                "evidence_id": "validation-write",
                "sql": sql,
                "operation": "write",
                "task_id": "validation-task",
            },
        ),
        explicit_mode="write.execute",
        max_access="write",
    )

    assert compilation.rejected_action_summaries == ()
    (write,) = compilation.task_specs
    assert "sql_ref" not in write.input
    assert write.input["sql_validation_evidence_id"] == "validation-write"
    dependency = write.dependencies[0]
    assert dependency.evidence_kind == "sql.validation"
    assert dependency.evidence_id == "validation-write"
    provenance = write.metadata["sql_provenance"]
    assert provenance["provenance"] == "validation_evidence_id"
    assert provenance["source_evidence_id"] == "validation-write"
    assert provenance["source_evidence_kind"] == "sql.validation"


async def test_repaired_sql_validation_creates_distinct_execute_task():
    old_sql = "select count(*) from orders where status = 'completed'"
    repaired_sql = "select count(*) from orders where status = 'complete'"
    runtime, operation = await _runtime_and_operation(
        "phase-one-distinct-repaired-execute"
    )
    try:
        await runtime.store.save_evidence(
            _sql_validation_evidence(
                operation.id,
                evidence_id="validation-old",
                sql=old_sql,
                task_id="validation-task-old",
            )
        )
        old_compilation = await _compile_action_for_runtime(
            runtime,
            operation,
            DbPlannerAction(
                action_id="execute_repaired",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "phase_two",
                    "plan_evidence_id": "validation-old",
                },
            ),
        )
        old_plan = await runtime.plan_task_specs(
            operation,
            old_compilation.task_specs,
        )

        await runtime.store.save_evidence(
            _sql_validation_evidence(
                operation.id,
                evidence_id="validation-repaired",
                sql=repaired_sql,
                task_id="validation-task-repaired",
            )
        )
        repaired_compilation = await _compile_action_for_runtime(
            runtime,
            operation,
            DbPlannerAction(
                action_id="execute_repaired",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "phase_two",
                    "plan_evidence_id": "validation-repaired",
                },
            ),
        )
        repaired_plan = await runtime.plan_task_specs(
            operation,
            repaired_compilation.task_specs,
        )
    finally:
        await runtime.teardown()

    old_execute = old_plan.tasks[0]
    repaired_execute = repaired_plan.tasks[0]
    assert old_execute.id != repaired_execute.id
    assert old_execute.input["sql_validation_evidence_id"] == "validation-old"
    assert repaired_execute.input["sql_validation_evidence_id"] == "validation-repaired"
    assert (
        old_execute.input["sql_fingerprint"]
        != repaired_execute.input["sql_fingerprint"]
    )
    assert old_execute.dependencies[0].evidence_id == "validation-old"
    assert repaired_execute.dependencies[0].evidence_id == "validation-repaired"


async def test_stale_blocked_execute_does_not_block_repaired_execution():
    old_sql = "select count(*) from orders where status = 'completed'"
    repaired_sql = "select count(*) from orders where status = 'complete'"
    runtime, operation = await _runtime_and_operation("phase-one-stale-blocked-execute")
    try:
        await runtime.store.save_evidence(
            _sql_validation_evidence(
                operation.id,
                evidence_id="validation-old",
                sql=old_sql,
                task_id="validation-task-old",
            )
        )
        old_compilation = await _compile_action_for_runtime(
            runtime,
            operation,
            DbPlannerAction(
                action_id="execute_repaired",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "phase_two",
                    "plan_evidence_id": "validation-old",
                },
            ),
        )
        old_plan = await runtime.plan_task_specs(
            operation,
            old_compilation.task_specs,
        )
        old_execute = old_plan.tasks[0]
        await runtime.kernel.block_task(old_execute.id, message="stale execution")
        old_blocked = await runtime.store.load_task(old_execute.id)

        await runtime.store.save_evidence(
            _sql_validation_evidence(
                operation.id,
                evidence_id="validation-repaired",
                sql=repaired_sql,
                task_id="validation-task-repaired",
            )
        )
        repaired_compilation = await _compile_action_for_runtime(
            runtime,
            operation,
            DbPlannerAction(
                action_id="execute_repaired",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "phase_two",
                    "plan_evidence_id": "validation-repaired",
                },
            ),
        )
        repaired_plan = await runtime.plan_task_specs(
            operation,
            repaired_compilation.task_specs,
        )
        repaired_execute = repaired_plan.tasks[0]
        repaired_readiness = await runtime.task_readiness(
            repaired_execute,
            operation,
        )
    finally:
        await runtime.teardown()

    assert old_execute.id != repaired_execute.id
    assert old_blocked is not None
    assert old_blocked.status.value == "blocked"
    assert repaired_execute.status.value == "pending"
    assert repaired_readiness["ready"] is True


async def test_execute_read_readiness_requires_exact_sql_validation_evidence():
    runtime, operation = await _runtime_and_operation(
        "phase-one-exact-validation-readiness"
    )
    try:
        await runtime.store.save_evidence(
            _sql_validation_evidence(
                operation.id,
                evidence_id="validation-other",
                sql="select 2 as answer",
                task_id="validation-task-other",
            )
        )
        exact_dependency = TaskDependency(
            kind="evidence",
            evidence_kind="sql.validation",
            evidence_id="validation-required",
            evidence_owner="phase_two",
            producer_task_id="validation-task-required",
            evidence_accepted=True,
            evidence_payload={"valid": True},
            payload_fingerprint="fp-validation-required",
            operation_id=operation.id,
        )
        plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="db.sql.execute_read",
                    owner="phase_two",
                    input={
                        "sql_validation_evidence_id": "validation-required",
                        "sql_validation_payload_fingerprint": (
                            "fp-validation-required"
                        ),
                        "sql_fingerprint": "sql-fp-required",
                        "params": [],
                    },
                    dependencies=(exact_dependency,),
                    reason="phase_1_exact_validation_readiness",
                    sequence=1,
                ),
            ),
        )
        task = plan.tasks[0]
        readiness = await runtime.task_readiness(task, operation)

        await runtime.store.save_evidence(
            _sql_validation_evidence(
                operation.id,
                evidence_id="validation-required",
                sql="select 1 as answer",
                task_id="validation-task-required",
            )
        )
        repaired_readiness = await runtime.task_readiness(task, operation)
    finally:
        await runtime.teardown()

    assert readiness["ready"] is False
    assert readiness["unsatisfied_dependencies"][0]["evidence_id"] == (
        "validation-required"
    )
    assert repaired_readiness["ready"] is True


async def test_write_execute_ignores_invalid_matching_validation():
    sql = "UPDATE orders SET status = 'approved' WHERE order_id = 101"
    action = DbPlannerAction(
        action_id="write",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
        input={
            "owner": "phase_two",
            "sql": sql,
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-write-invalid-validation",
        action,
        sql_validation_evidence=(
            {
                "evidence_id": "validation-invalid",
                "sql": sql,
                "operation": "write",
                "valid": False,
            },
        ),
        explicit_mode="write.execute",
        max_access="write",
    )

    assert compilation.task_specs == ()
    assert (
        compilation.rejected_action_summaries[0]["error"]
        == "missing_valid_sql_validation"
    )


async def test_plan_evidence_with_context_validates_plan_before_sql():
    sql = "select count(*) as count from orders"
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-explicit"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-plan-validation",
        action,
        plan_evidence=(
            {"evidence_id": "plan-explicit", "sql": sql, "task_id": "plan-task"},
        ),
        planning_context=True,
    )

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.query.plan.validate",
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    plan_validation, validation, read = compilation.task_specs
    assert plan_validation.input == {
        "plan_evidence_id": "plan-explicit",
        "planning_context_evidence_id": "planning-context",
    }
    assert validation.input == {"sql": sql, "operation": "query"}
    assert validation.dependencies[0].evidence_kind == "query.plan.validation"
    assert validation.dependencies[0].producer_task_id == plan_validation.task_id
    assert "sql_ref" not in read.input
    assert read.input["plan_evidence_id"] == "plan-explicit"
    assert read.input["sql_validation_task_id"] == validation.task_id
    sql_dependency = next(
        dependency
        for dependency in read.dependencies
        if dependency.evidence_kind == "sql.validation"
    )
    assert sql_dependency.producer_task_id == validation.task_id


async def test_explicit_write_execute_mode_overrides_planner_data_query_intent():
    action = DbPlannerAction(
        action_id="write",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
        input={"owner": "phase_two", "sql": "update orders set status = 'approved'"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-write-mode-contract",
        action,
        explicit_mode="write.execute",
        max_access="write",
    )

    assert compilation.rejected_action_summaries == ()
    assert compilation.compiled_contract_snapshot["operation_type"] == "write.execute"
    assert compilation.compiled_contract_snapshot["access"] == "write"
    assert (
        compilation.compiled_contract_snapshot["metadata"]["planner_intent"][
            "operation_type"
        ]
        == "write.execute"
    )
    assert (
        compilation.compiled_contract_snapshot["metadata"]["planner_raw_intent"][
            "operation_type"
        ]
        == "data.query"
    )


async def test_approval_state_uses_requested_policy_id():
    runtime, operation = await _runtime_and_operation("phase-two-approval-state")
    approval = ApprovalRequest(
        approval_id="approval-1",
        operation_id=operation.id,
        reason="Approve write execution.",
        risk=RiskLevel.HIGH,
        requested_by_policy_id="approval_required_for_writes",
        proposed_action={"approval": "human"},
        status=ApprovalStatus.PENDING,
    )
    await runtime.store.save_approval_request(approval)

    state = await DbAgentLoop(runtime, FakePlanner())._approval_state(operation.id)

    assert state["requests"] == [
        {
            "approval_id": "approval-1",
            "status": "pending",
            "policy_id": "approval_required_for_writes",
            "task_id": None,
        }
    ]


async def test_latest_accepted_query_plan_ref_ignores_rejected_plan_evidence():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-rejected-ignored",
        action,
        plan_evidence=(
            {"evidence_id": "plan-accepted", "sql": "select 1 as answer"},
            {
                "evidence_id": "plan-rejected",
                "sql": "select 999 as answer",
                "accepted": False,
            },
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"


async def test_latest_accepted_query_plan_ref_ignores_rejected_invalid_plan_evidence():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-invalid-plan-ignored",
        action,
        plan_evidence=(
            {"evidence_id": "plan-accepted", "sql": "select 1 as answer"},
            {
                "evidence_id": "plan-invalid",
                "sql": "select 999 as answer",
                "valid": False,
                "accepted": False,
            },
            {
                "evidence_id": "plan-rejected",
                "sql": "select 888 as answer",
                "accepted": False,
            },
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"


async def test_latest_accepted_query_plan_ref_ignores_rejected_plan_without_sql():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-latest-plan-skips-nosql",
        action,
        plan_evidence=(
            {"evidence_id": "plan-with-sql", "sql": "select 1 as answer"},
            {"evidence_id": "plan-without-sql", "sql": None, "accepted": False},
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-with-sql"


async def test_rejected_plan_evidence_id_is_rejected_clearly():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-rejected"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-rejected-explicit",
        action,
        plan_evidence=(
            {
                "evidence_id": "plan-rejected",
                "sql": "select 999 as answer",
                "accepted": False,
            },
        ),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == (
        "rejected_plan_evidence:plan-rejected"
    )


async def test_ambiguous_sql_inputs_are_rejected_clearly():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "sql": "select 1 as answer",
            "plan_evidence_id": "plan-explicit",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-ambiguous-sql-input",
        action,
        plan_evidence=({"evidence_id": "plan-explicit", "sql": "select 2 as answer"},),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == ("ambiguous_sql_input")


async def test_ambiguous_plan_evidence_id_is_rejected_clearly():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-ambiguous"},
    )
    runtime, operation = await _runtime_and_operation("phase-four-ambiguous-plan")
    state = DbLoopState(
        operation_id=operation.id,
        normalized_user_request={"prompt": "phase four"},
        safety_frame={"max_access": "read"},
        available_action_kinds=tuple(DbPlannerActionKind),
        accepted_evidence_summaries=(
            {
                "id": "plan-ambiguous",
                "kind": "query.plan.proposal",
                "owner": "phase_two",
                "accepted": True,
                "sql": "select 1 as answer",
            },
            {
                "id": "plan-ambiguous",
                "kind": "query.plan.proposal",
                "owner": "phase_two",
                "accepted": True,
                "sql": "select 2 as answer",
            },
        ),
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(action,),
    )

    compilation = DbAgentLoop(runtime, FakePlanner()).compile_actions(
        decision,
        state,
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == (
        "ambiguous_plan_evidence:plan-ambiguous"
    )


async def test_invalid_sql_reference_inputs_are_rejected_clearly():
    cases = (
        (
            "missing-plan-id",
            {"owner": "phase_two", "plan_evidence_id": ""},
            (),
            "missing_plan_evidence_id",
        ),
        (
            "unsupported-ref",
            {"owner": "phase_two", "query_plan_ref": "latest_sql"},
            (),
            "unsupported_query_plan_ref:latest_sql",
        ),
        (
            "rejected-plan-without-sql",
            {"owner": "phase_two", "plan_evidence_id": "plan-nosql"},
            ({"evidence_id": "plan-nosql", "sql": None, "accepted": False},),
            "rejected_plan_evidence:plan-nosql",
        ),
    )
    for suffix, action_input, plan_evidence, expected_error in cases:
        action = DbPlannerAction(
            action_id="read",
            kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
            input=action_input,
        )

        compilation, _, _ = await _compile_single_action(
            f"phase-four-{suffix}",
            action,
            plan_evidence=plan_evidence,
        )

        assert compilation.task_specs == ()
        assert compilation.rejected_action_summaries[0]["error"] == expected_error


async def test_prior_turn_depends_on_with_unsupported_ref_remains_missing_dependency_error():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_sql",
        },
        depends_on=("prior_turn_plan",),
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-prior-depends-on",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == (
        "missing_dependency:prior_turn_plan"
    )


async def test_missing_sql_no_longer_falls_back_to_latest_plan_silently():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-missing-sql",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == "missing_sql"


async def test_validation_grounding_repair_targets_only_validation_column():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
    )
    await runtime.setup(agent_id="agent-phase-two")
    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Repair the draft SQL using validation facts.",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        await runtime.store.save_evidence(
            Evidence(
                id="schema-targeted-repair",
                kind="schema.asset_profile",
                owner="sqlite",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "database_type": "sqlite",
                    "tables": [
                        {
                            "name": "orders",
                            "columns": [
                                {"name": "id", "data_type": "INTEGER"},
                                {"name": "status", "data_type": "TEXT"},
                                {"name": "channel", "data_type": "TEXT"},
                            ],
                        },
                        {
                            "name": "customers",
                            "columns": [
                                {"name": "id", "data_type": "INTEGER"},
                                {"name": "status", "data_type": "TEXT"},
                            ],
                        },
                    ],
                },
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="validation-unobserved-status",
                kind="sql.validation",
                owner="sqlite",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "valid": False,
                    "sql": (
                        "SELECT COUNT(*) FROM orders "
                        "WHERE orders.status = 'completed'"
                    ),
                    "operation": "query",
                    "warnings": [
                        {
                            "kind": "unobserved_filter_literal",
                            "table": "orders",
                            "column": "status",
                            "literal": "completed",
                            "candidates": ["complete", "pending"],
                        }
                    ],
                    "validation_facts": [
                        {
                            "kind": "unobserved_filter_literal",
                            "table": "orders",
                            "column": "status",
                            "literal": "completed",
                            "candidates": ["complete", "pending"],
                        }
                    ],
                },
            )
        )
        loop = DbAgentLoop(runtime, object())
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "read"},
            turn=1,
            remaining_turns=1,
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=(
                DbPlannerAction(
                    action_id="context",
                    kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
                    input={"source_owner": "sqlite"},
                ),
            ),
        )

        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    grounding_specs = [
        spec
        for spec in compilation.task_specs
        if spec.capability_id
        in {
            "catalog.value_grounding.plan",
            "catalog.column_value_hints.resolve",
        }
    ]
    assert grounding_specs
    targets = {
        (target["table"], target["column"])
        for spec in grounding_specs
        for target in _phase0_value_grounding_targets(spec.input)
    }
    assert ("orders", "status") in targets
    assert ("orders", "channel") not in targets
    assert ("customers", "status") not in targets


def test_validation_grounding_continues_from_refresh_to_repair_then_execution():
    grounding_fact = {
        "kind": "filter_literal_requires_grounding",
        "table": "orders",
        "column": "status",
        "literal": "completed",
        "candidates": ["complete", "pending"],
    }
    rejected_validation = {
        "id": "validation-rejected",
        "kind": "query.plan.validation",
        "owner": "db_runtime",
        "accepted": False,
        "valid": False,
        "plan_evidence_id": "plan-original",
        "validation_facts": [grounding_fact],
    }
    grounded_evidence = (
        {
            "id": "plan-original",
            "kind": "query.plan.proposal",
            "owner": "db_runtime",
            "accepted": True,
            "valid": True,
            "sql": "select count(*) from orders where status = 'completed'",
        },
        {
            "id": "hint-orders-status",
            "kind": "schema.column_value_hint",
            "owner": "catalog",
            "accepted": True,
            "hints": [
                {
                    "table": "orders",
                    "column": "status",
                    "observed_values": ["complete", "pending"],
                }
            ],
        },
        {
            "id": "planning-context-refreshed",
            "kind": "planning.context",
            "owner": "db_runtime",
            "accepted": True,
        },
    )
    state_after_refresh = DbLoopState(
        operation_id="phase-zero-grounding-repair",
        normalized_user_request={"prompt": "Count completed orders."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=grounded_evidence,
        rejected_evidence_summaries=(rejected_validation,),
        validation_summaries=(rejected_validation,),
    )

    repair = _validation_grounding_runtime_continuation_action(
        state_after_refresh,
        current_action_ids=set(),
    )

    assert repair is not None
    assert repair.kind is DbPlannerActionKind.REPAIR_QUERY_PLAN
    assert repair.input["failure_evidence_id"] == "validation-rejected"
    assert repair.input["prior_plan_evidence_id"] == "plan-original"
    assert repair.input["planning_context_evidence_id"] == (
        "planning-context-refreshed"
    )

    repaired_plan = {
        "id": "plan-repaired",
        "kind": "query.plan.proposal",
        "owner": "db_runtime",
        "accepted": True,
        "valid": True,
        "sql": "select count(*) from orders where status = 'complete'",
    }
    state_after_repair = DbLoopState(
        operation_id="phase-zero-grounding-execute",
        normalized_user_request={"prompt": "Count completed orders."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=(*grounded_evidence, repaired_plan),
        rejected_evidence_summaries=(rejected_validation,),
        validation_summaries=(rejected_validation,),
    )

    execute = _validation_grounding_runtime_continuation_action(
        state_after_repair,
        current_action_ids=set(),
    )

    assert execute is not None
    assert execute.kind is DbPlannerActionKind.EXECUTE_VALIDATED_READ
    assert execute.input["plan_evidence_id"] == "plan-repaired"


async def test_agent_loop_rejects_action_outside_contract_before_task_creation():
    runtime, operation = await _runtime_and_operation("phase-two-reject")
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "db.run"},
        actions=(
            DbPlannerAction(
                action_id="write",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
                input={
                    "owner": "phase_two",
                    "sql": "update orders set status = 'paid'",
                },
            ),
        ),
    )

    result = await DbAgentLoop(runtime, FakePlanner(decision)).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )

    assert result.status == "budget_exhausted"
    assert await runtime.store.list_tasks(operation.id) == []
    compilation = next(
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "planner.compilation"
    )
    rejected = compilation.payload["compilation"]["rejected_action_summaries"]
    assert rejected
    assert rejected[0]["error"].startswith("access_outside_contract")
    observation = next(
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "planner.observation"
    )
    assert observation.payload["observation"]["diagnostics"]["status"] == (
        "compilation_rejected"
    )


async def test_llm_agent_planner_emits_typed_decision_from_mocked_response():
    content = json.dumps(_llm_planner_payload())
    service = FakeLLMService(content)
    state = _loop_state()

    decision = await DbLLMAgentPlanner(service).plan(state)

    assert decision.status is DbPlannerDecisionStatus.CONTINUE
    assert decision.actions[0].kind is DbPlannerActionKind.EXECUTE_VALIDATED_READ
    assert decision.metadata["planner"] == "fake"
    assert decision.metadata["llm"]["model"] == "phase-two"
    assert service.messages is not None
    request_payload = json.loads(service.messages[-1]["content"])
    assert request_payload["state"]["operation_id"] == "op-loop"
    assert "finish" not in request_payload["available_action_kinds"]
    assert "clarify" not in request_payload["available_action_kinds"]
    assert "finish" not in request_payload["state"]["available_action_kinds"]
    assert "clarify" not in request_payload["state"]["available_action_kinds"]
    schema = request_payload["decision_schema"]
    assert "Executable actions require status='continue'" in schema["description"]
    branches = schema["properties"]["decision"]["anyOf"]
    assert len(branches) == 3
    action_kind_schema = branches[0]["properties"]["actions"]["items"]["properties"][
        "kind"
    ]
    assert "finish" not in action_kind_schema["enum"]
    assert "clarify" not in action_kind_schema["enum"]
    assert (
        "Executable actions require status='continue'" in service.messages[0]["content"]
    )
    native_schema, schema_name = service.response_schemas[0]
    assert native_schema == schema
    assert schema_name == "db_planner_decision"


async def test_llm_agent_planner_parses_fenced_json_at_boundary():
    content = f"```json\n{json.dumps(_llm_planner_payload())}\n```"

    decision = await DbLLMAgentPlanner(FakeLLMService(content)).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.CONTINUE
    diagnostics = decision.metadata["planner_json_normalization"]
    assert any(
        step["step"] == "json_fence_stripped"
        for step in diagnostics["normalization_steps"]
    )


async def test_llm_agent_planner_unwraps_common_decision_envelopes():
    for envelope_key in ("decision", "planner_decision", "DbPlannerDecision"):
        content = json.dumps({envelope_key: _llm_planner_payload()})

        decision = await DbLLMAgentPlanner(FakeLLMService(content)).plan(_loop_state())

        assert decision.status is DbPlannerDecisionStatus.CONTINUE
        diagnostics = decision.metadata["planner_json_normalization"]
        assert diagnostics["unwrapped_envelope"] == envelope_key


async def test_llm_agent_planner_normalizes_unknown_keys_and_tuple_fields():
    payload = _llm_planner_payload(
        stop_conditions={"name": "verified"},
        unexpected_decision_key="drop me",
    )
    payload["actions"][0]["depends_on"] = [{"action_id": "schema"}]
    payload["actions"][0]["unexpected_action_key"] = "drop me too"
    content = json.dumps({"planner_decision": payload, "wrapper_note": "ignored"})

    decision = await DbLLMAgentPlanner(FakeLLMService(content)).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.CONTINUE
    assert decision.actions[0].depends_on == ("schema",)
    assert decision.stop_conditions == ("verified",)
    assert decision.metadata["planner"] == "fake"

    diagnostics = decision.metadata["planner_json_normalization"]
    assert diagnostics["dropped_envelope_keys"] == ["wrapper_note"]
    assert "unexpected_decision_key" in diagnostics["dropped_decision_keys"]
    assert diagnostics["dropped_action_keys"] == [
        {
            "index": 0,
            "action_id": "read",
            "keys": ["unexpected_action_key"],
        }
    ]
    assert {item["path"] for item in diagnostics["coerced_fields"]} == {
        "actions[0].depends_on",
        "stop_conditions",
    }


def test_planner_decision_shape_accepts_continue_with_executable_action():
    decision = DbPlannerDecision.from_dict(_llm_planner_payload())

    assert validate_planner_decision_shape(decision) is decision


@pytest.mark.parametrize("status", ["finish", "clarify", "blocked", "failed"])
def test_planner_decision_shape_rejects_terminal_status_with_actions(status):
    decision = DbPlannerDecision.from_dict(
        _llm_planner_payload(
            status=status,
            clarification_question=("Which segment?" if status == "clarify" else None),
        )
    )

    with pytest.raises(DbPlannerDecisionShapeError) as raised:
        validate_planner_decision_shape(decision)

    assert raised.value.code == "terminal_status_mixed_with_executable_actions"


@pytest.mark.parametrize("status", ["finish", "clarify", "blocked", "failed"])
def test_planner_decision_shape_accepts_terminal_status_without_actions(status):
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus(status),
        actions=(),
        clarification_question=("Which segment?" if status == "clarify" else None),
    )

    assert validate_planner_decision_shape(decision) is decision


def test_persisted_legacy_terminal_actions_normalize_compatibly():
    for status, action_kind in (
        ("finish", "finish"),
        ("continue", "finish"),
        ("clarify", "clarify"),
        ("continue", "clarify"),
    ):
        payload = _llm_planner_payload(
            status=status,
            actions=[
                {
                    "action_id": "terminal",
                    "kind": action_kind,
                    "input": {},
                    "depends_on": [],
                    "metadata": {},
                }
            ],
            clarification_question=(
                "Which customer segment?" if action_kind == "clarify" else None
            ),
        )

        decision = DbPlannerDecision.from_persisted_dict(payload)

        assert decision.status.value == action_kind
        assert decision.actions == ()


@pytest.mark.parametrize(
    "case",
    ["terminal_status", "legacy_terminal_action"],
)
async def test_llm_agent_planner_rejects_mixed_terminal_and_executable_actions(
    case,
):
    if case == "terminal_status":
        payload = _llm_planner_payload(status="finish")
    else:
        payload = _llm_planner_payload(
            actions=[
                {
                    "action_id": "terminal",
                    "kind": "finish",
                    "input": {},
                    "depends_on": [],
                    "metadata": {},
                },
                {
                    "action_id": "read",
                    "kind": "execute_validated_read",
                    "input": {"owner": "phase_two", "sql": "select 1"},
                    "depends_on": [],
                    "metadata": {},
                },
            ]
        )
    service = FakeLLMService(json.dumps(payload))
    decision = await DbLLMAgentPlanner(service).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.FAILED
    assert decision.actions == ()
    assert decision.metadata["failure"] == "planner_decision_shape_invalid"
    assert "terminal" in decision.metadata["error"]
    assert len(service.calls) == 2


async def test_llm_agent_planner_rejects_empty_clarification_question():
    payload = _llm_planner_payload(
        status="clarify",
        actions=[],
        clarification_question="  ",
    )

    service = FakeLLMService(json.dumps(payload))
    decision = await DbLLMAgentPlanner(service).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.FAILED
    assert decision.metadata["error"] == "clarification_question_required"
    assert len(service.calls) == 2


async def test_llm_agent_planner_corrects_one_invalid_shape_with_same_state():
    invalid = _llm_planner_payload(status="finish")
    valid = _llm_planner_payload(status="continue")
    service = FakeLLMService([json.dumps(invalid), json.dumps(valid)])

    decision = await DbLLMAgentPlanner(service).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.CONTINUE
    assert len(decision.actions) == 1
    assert len(service.calls) == 2
    assert (
        json.loads(service.calls[0][-1]["content"])["state"]
        == json.loads(service.calls[1][1]["content"])["state"]
    )
    correction = json.loads(service.calls[1][-1]["content"])
    assert correction["validation_error"]["code"] == (
        "terminal_status_mixed_with_executable_actions"
    )
    diagnostics = decision.metadata["planner_private_diagnostics"]
    assert diagnostics["attempt_count"] == 2
    assert [item["correction_requested"] for item in diagnostics["attempts"]] == [
        False,
        True,
    ]


async def test_llm_agent_planner_corrects_non_object_action_array_item():
    invalid = _llm_planner_payload()
    invalid["actions"].extend(["rationale", "metadata"])
    valid = _llm_planner_payload()
    service = FakeLLMService([json.dumps(invalid), json.dumps(valid)])

    decision = await DbLLMAgentPlanner(service).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.CONTINUE
    assert len(service.calls) == 2
    correction = json.loads(service.calls[1][-1]["content"])
    assert correction["validation_error"]["code"] == "planner_action_not_object"
    diagnostics = decision.metadata["planner_private_diagnostics"]
    assert diagnostics["attempt_count"] == 2
    assert diagnostics["attempts"][0]["validation_error"]["code"] == (
        "planner_action_not_object"
    )


async def test_planner_shape_correction_has_no_tasks_or_extra_loop_turn():
    invalid = _llm_planner_payload(status="failed")
    valid = _llm_planner_payload(
        status="clarify",
        actions=[],
        clarification_question="Which customer segment?",
    )
    service = FakeLLMService([json.dumps(invalid), json.dumps(valid)])
    planner = DbLLMAgentPlanner(service)
    runtime, operation = await _runtime_and_operation(
        "phase-five-shape-correction",
    )
    try:
        result = await DbAgentLoop(runtime, planner).run(operation, max_turns=1)
        tasks = await runtime.store.list_tasks(operation.id)
        decisions = [
            item
            for item in await runtime.store.list_evidence(operation.id)
            if item.kind == "planner.decision"
        ]
    finally:
        await runtime.teardown()

    assert result.status == "clarification_required"
    assert tasks == []
    assert len(service.calls) == 2
    assert len(decisions) == 1
    assert decisions[0].payload["turn"] == 1


async def test_planner_private_attempt_diagnostics_are_redacted_and_persisted():
    secret = "sk-test-secret-123456"
    invalid = _llm_planner_payload(
        status="blocked",
        rationale=f"Contact ada@example.com using {secret}",
    )
    service = FakeLLMService(json.dumps(invalid))
    planner = DbLLMAgentPlanner(service)
    runtime, operation = await _runtime_and_operation(
        "phase-five-private-planner-diagnostics"
    )
    try:
        result = await DbAgentLoop(runtime, planner).run(operation, max_turns=1)
        persisted = next(
            item
            for item in await runtime.store.list_evidence(operation.id)
            if item.kind == "planner.decision"
        )
    finally:
        await runtime.teardown()

    assert result.status == "failed"
    dumped = json.dumps(persisted.payload, sort_keys=True)
    assert secret not in dumped
    assert "ada@example.com" not in dumped
    assert "[REDACTED_API_KEY]" in dumped
    assert "[REDACTED_EMAIL]" in dumped
    private = persisted.payload["decision"]["metadata"]["planner_private_diagnostics"]
    assert private["attempt_count"] == 2
    assert private["attempts"][0]["parsed_pre_normalization"] is not None


async def test_compiler_rejects_compatibility_terminal_action():
    runtime, operation = await _runtime_and_operation("phase-two-terminal-compiler")
    loop = DbAgentLoop(runtime, FakePlanner())
    state = await loop.build_loop_state(operation, turn=1, remaining_turns=1)
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        actions=(
            DbPlannerAction(
                action_id="legacy-finish",
                kind=DbPlannerActionKind.FINISH,
            ),
        ),
    )
    try:
        compilation = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == (
        "terminal_action_must_use_decision_status"
    )


async def test_repeated_premature_finish_has_unmet_finalization_reason():
    runtime, operation = await _runtime_and_operation(
        "phase-two-repeated-premature-finish",
        required_evidence={"query.result"},
    )
    finish = DbPlannerDecision(
        status=DbPlannerDecisionStatus.FINISH,
        intent={"operation_type": "data.query"},
    )
    try:
        result = await DbAgentLoop(runtime, FakePlanner(finish, finish)).run(
            operation,
            max_turns=2,
        )
    finally:
        await runtime.teardown()

    assert result.status == "failed"
    assert result.diagnostics["error"] == "db_agent_loop_unmet_finalization"
    assert "db_agent_loop_unmet_finalization" in result.warnings


async def test_premature_finish_progress_resets_repetition_fingerprint():
    runtime, operation = await _runtime_and_operation(
        "phase-two-premature-finish-progress",
        required_evidence={"query.result"},
    )
    finish = DbPlannerDecision(
        status=DbPlannerDecisionStatus.FINISH,
        intent={"operation_type": "data.query"},
    )
    inspect = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="inspect-after-finish",
                kind=DbPlannerActionKind.INSPECT_SCHEMA,
                input={"owner": "phase_two"},
            ),
        ),
    )
    try:
        result = await DbAgentLoop(
            runtime,
            FakePlanner(finish, inspect, finish),
        ).run(operation, max_turns=3)
    finally:
        await runtime.teardown()

    assert result.status == "budget_exhausted"
    assert "db_agent_loop_unmet_finalization" not in result.warnings


async def test_premature_finish_preserves_substantive_contract_for_resume():
    runtime, operation = await _runtime_and_operation(
        "phase-two-preserve-contract",
        required_evidence={"query.result"},
    )
    inspect = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="inspect",
                kind=DbPlannerActionKind.INSPECT_SCHEMA,
                input={"owner": "phase_two"},
            ),
        ),
    )
    finish = DbPlannerDecision(
        status=DbPlannerDecisionStatus.FINISH,
        intent={"operation_type": "data.query"},
    )
    loop = DbAgentLoop(runtime, FakePlanner(inspect, finish))
    try:
        result = await loop.run(operation, max_turns=2)
        persisted = await runtime.store.load_operation(operation.id)
        resumed_state = await loop.build_loop_state(
            persisted,
            turn=1,
            remaining_turns=1,
        )
    finally:
        await runtime.teardown()

    assert result.status == "budget_exhausted"
    snapshot = persisted.metadata["latest_compiled_contract_snapshot"]
    assert snapshot["required_capabilities"] == ["db.schema.inspect"]
    assert snapshot["required_evidence"] == ["database.schema"]
    assert resumed_state.latest_compiled_contract_snapshot == snapshot


def test_validation_continuation_rejects_stale_latest_plan():
    state = DbLoopState(
        operation_id="phase-two-stale-plan",
        normalized_user_request={"prompt": "Count completed orders."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=(
            {
                "id": "context-new",
                "kind": "planning.context",
                "accepted": True,
                "schema_fingerprint": "schema-new",
            },
            {
                "id": "plan-stale",
                "kind": "query.plan.proposal",
                "accepted": True,
                "valid": True,
                "sql": "select count(*) from orders",
                "planning_context_evidence_id": "context-old",
                "schema_fingerprint": "schema-old",
            },
        ),
    )

    action = _validation_grounding_runtime_continuation_action(
        state,
        current_action_ids=set(),
    )

    assert action is not None
    resolution = action.metadata["continuation_resolution"]
    assert resolution["status"] == "blocked"
    assert resolution["error"] == "stale_query_plan_planning_context"


def test_validation_grounding_fingerprint_is_bound_to_failed_plan():
    def state_for(plan_id, validation_id):
        validation = {
            "id": validation_id,
            "kind": "query.plan.validation",
            "accepted": False,
            "valid": False,
            "plan_evidence_id": plan_id,
            "validation_facts": [
                {
                    "kind": "unobserved_filter_literal",
                    "table": "orders",
                    "column": "status",
                    "literal": "completed",
                }
            ],
        }
        return DbLoopState(
            operation_id="phase-two-plan-bound-grounding",
            normalized_user_request={"prompt": "Count completed orders."},
            safety_frame={"max_access": "read"},
            accepted_evidence_summaries=(
                {
                    "id": plan_id,
                    "kind": "query.plan.proposal",
                    "accepted": True,
                    "valid": True,
                    "sql": "select count(*) from orders",
                },
            ),
            rejected_evidence_summaries=(validation,),
            validation_summaries=(validation,),
        )

    first = _validation_grounding_runtime_continuation_action(
        state_for("plan-a", "validation-a"),
        current_action_ids=set(),
    )
    second = _validation_grounding_runtime_continuation_action(
        state_for("plan-b", "validation-b"),
        current_action_ids=set(),
    )

    assert first is not None and second is not None
    assert first.metadata["validation_grounding_fingerprint"] != (
        second.metadata["validation_grounding_fingerprint"]
    )


def test_validation_continuation_ignores_result_from_older_plan():
    state = DbLoopState(
        operation_id="phase-two-old-result",
        normalized_user_request={"prompt": "Count customers."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=(
            {
                "id": "plan-old",
                "kind": "query.plan.proposal",
                "accepted": True,
                "valid": True,
                "sql": "select count(*) from orders",
            },
            {
                "id": "result-old",
                "kind": "query.result",
                "accepted": True,
                "plan_evidence_id": "plan-old",
            },
            {
                "id": "plan-new",
                "kind": "query.plan.proposal",
                "accepted": True,
                "valid": True,
                "sql": "select count(*) from customers",
            },
        ),
    )

    action = _validation_grounding_runtime_continuation_action(
        state,
        current_action_ids=set(),
    )

    assert action is not None
    assert action.kind is DbPlannerActionKind.EXECUTE_VALIDATED_READ
    assert action.input["plan_evidence_id"] == "plan-new"


def test_equivalent_planning_context_fingerprint_allows_new_evidence_id():
    state = DbLoopState(
        operation_id="phase-two-equivalent-context",
        normalized_user_request={"prompt": "Count orders."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=(
            {
                "id": "plan-current",
                "kind": "query.plan.proposal",
                "accepted": True,
                "valid": True,
                "sql": "select count(*) from orders",
                "planning_context_evidence_id": "context-old-id",
                "planning_context_fingerprint": "context-same",
            },
            {
                "id": "context-new-id",
                "kind": "planning.context",
                "accepted": True,
                "payload_fingerprint": "context-same",
            },
        ),
    )

    action = _validation_grounding_runtime_continuation_action(
        state,
        current_action_ids=set(),
    )

    assert action is not None
    assert "continuation_resolution" not in action.metadata


@pytest.mark.parametrize(
    ("plan_key", "state_key", "expected_error"),
    [
        (
            "session_context_fingerprint",
            "session_context_fingerprint",
            "stale_query_plan_session_binding",
        ),
        (
            "contract_fingerprint",
            "contract_fingerprint",
            "stale_query_plan_contract",
        ),
    ],
)
def test_validation_continuation_rejects_session_or_contract_change(
    plan_key,
    state_key,
    expected_error,
):
    plan = {
        "id": "plan-current",
        "kind": "query.plan.proposal",
        "accepted": True,
        "valid": True,
        "sql": "select count(*) from orders",
        "planning_context_fingerprint": "context-same",
        plan_key: "old-binding",
    }
    current_context = {
        "id": "context-current",
        "kind": "planning.context",
        "accepted": True,
        "payload_fingerprint": "context-same",
    }
    if state_key == "session_context_fingerprint":
        current_context[state_key] = "new-binding"
    state = DbLoopState(
        operation_id="phase-two-stale-binding",
        normalized_user_request={"prompt": "Count orders."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=(
            plan,
            current_context,
        ),
        diagnostics={
            state_key: (
                "new-binding"
                if state_key == "contract_fingerprint"
                else "raw-request-binding"
            )
        },
    )

    action = _validation_grounding_runtime_continuation_action(
        state,
        current_action_ids=set(),
    )

    assert action is not None
    assert action.metadata["continuation_resolution"]["error"] == expected_error


def test_validation_continuation_uses_bounded_session_context_binding():
    state = DbLoopState(
        operation_id="phase-two-bounded-session-binding",
        normalized_user_request={"prompt": "Count orders."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=(
            {
                "id": "plan-current",
                "kind": "query.plan.proposal",
                "accepted": True,
                "valid": True,
                "sql": "select count(*) from orders",
                "planning_context_fingerprint": "context-same",
                "session_context_fingerprint": "bounded-session",
            },
            {
                "id": "context-current",
                "kind": "planning.context",
                "accepted": True,
                "payload_fingerprint": "context-same",
                "session_context_fingerprint": "bounded-session",
            },
        ),
        diagnostics={"session_context_fingerprint": "raw-request-session"},
    )

    action = _validation_grounding_runtime_continuation_action(
        state,
        current_action_ids=set(),
    )

    assert action is not None
    assert "continuation_resolution" not in action.metadata


async def test_compiler_preserves_selected_runtime_continuation_action_id():
    runtime, operation = await _runtime_and_operation(
        "phase-two-runtime-continuation-identity"
    )
    state = DbLoopState(
        operation_id=operation.id,
        normalized_user_request={"prompt": "Count orders."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=(
            {
                "id": "plan-current",
                "kind": "query.plan.proposal",
                "accepted": True,
                "valid": True,
                "sql": "select count(*) from orders",
                "planning_context_fingerprint": "context-same",
                "contract_fingerprint": "old-contract",
            },
            {
                "id": "context-current",
                "kind": "planning.context",
                "accepted": True,
                "payload_fingerprint": "context-same",
            },
        ),
        diagnostics={"contract_fingerprint": "new-contract"},
    )
    action = _validation_grounding_runtime_continuation_action(
        state,
        current_action_ids=set(),
    )
    assert action is not None
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(action,),
    )
    try:
        compilation = DbAgentLoop(runtime, FakePlanner()).compile_actions(
            decision,
            state,
        )
    finally:
        await runtime.teardown()

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["action_id"] == action.action_id


def test_validation_grounding_repair_exhaustion_blocks_specifically():
    grounding_fact = {
        "kind": "unobserved_filter_literal",
        "table": "orders",
        "column": "status",
        "literal": "completed",
    }
    failed_validation = {
        "id": "validation-rejected",
        "kind": "query.plan.validation",
        "accepted": False,
        "valid": False,
        "plan_evidence_id": "plan-original",
        "validation_facts": [grounding_fact],
    }
    state = DbLoopState(
        operation_id="phase-two-repair-exhausted",
        normalized_user_request={"prompt": "Count completed orders."},
        safety_frame={"max_access": "read"},
        accepted_evidence_summaries=(
            {
                "id": "plan-original",
                "kind": "query.plan.proposal",
                "accepted": True,
                "valid": True,
                "sql": "select count(*) from orders where status = 'completed'",
            },
            {
                "id": "hint-orders-status",
                "kind": "schema.column_value_hint",
                "accepted": True,
                "hints": [{"table": "orders", "column": "status"}],
            },
            {
                "id": "planning-context-refreshed",
                "kind": "planning.context",
                "accepted": True,
            },
        ),
        rejected_evidence_summaries=(
            failed_validation,
            {
                "id": "repair-rejected",
                "kind": "query.plan.repair",
                "accepted": False,
                "failure_evidence_id": "validation-rejected",
                "prior_plan_evidence_id": "plan-original",
                "planning_context_evidence_id": "planning-context-refreshed",
            },
        ),
        validation_summaries=(failed_validation,),
    )

    action = _validation_grounding_runtime_continuation_action(
        state,
        current_action_ids=set(),
    )

    assert action is not None
    resolution = action.metadata["continuation_resolution"]
    assert resolution["status"] == "blocked"
    assert resolution["error"] == "validation_grounding_repair_exhausted"


async def test_llm_agent_planner_rejects_invalid_action_kind_without_tasks():
    content = json.dumps(
        {
            "status": "continue",
            "intent": {"operation_type": "db.run"},
            "actions": [
                {
                    "action_id": "bad",
                    "kind": "made_up_action",
                    "input": {"owner": "phase_two"},
                    "depends_on": [],
                    "rationale": "Invalid action should not compile.",
                    "metadata": {},
                }
            ],
            "stop_conditions": [],
            "clarification_question": None,
            "rationale": "Malformed planner action.",
            "metadata": {},
        }
    )
    planner = DbLLMAgentPlanner(FakeLLMService(content))

    decision = await planner.plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.FAILED
    assert decision.actions == ()
    assert decision.metadata["failure"] == "planner_decision_invalid"

    runtime, operation = await _runtime_and_operation("phase-two-invalid-action")
    result = await DbAgentLoop(runtime, planner).run(operation, max_turns=1)

    assert result.status == "failed"
    assert await runtime.store.list_tasks(operation.id) == []
    decision_evidence = next(
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "planner.decision"
    )
    persisted = decision_evidence.payload["decision"]
    assert persisted["actions"] == []
    assert persisted["metadata"]["failure"] == "planner_decision_invalid"


async def test_missing_db_llm_configuration_returns_no_planner_actions():
    decision = await DbLLMAgentPlanner(DbLLMService(None)).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.FAILED
    assert decision.actions == ()
    assert decision.metadata["configuration_required"] is True

    runtime, operation = await _runtime_and_operation("phase-two-missing-llm")
    result = await DbAgentLoop(runtime, DbLLMAgentPlanner(DbLLMService(None))).run(
        operation,
        max_turns=1,
    )

    assert result.status == "configuration_required"
    assert await runtime.store.list_tasks(operation.id) == []
    decision_evidence = next(
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "planner.decision"
    )
    persisted = decision_evidence.payload["decision"]
    assert persisted["actions"] == []
    assert persisted["metadata"]["configuration_required"] is True


async def _runtime_and_operation(
    operation_id,
    *,
    mode=None,
    db_llm_service=None,
    required_evidence=frozenset(),
):
    runtime = DbRuntime(
        config=DbRuntimeConfig(plugins=(PhaseTwoPlugin(),)),
        runtime_id="phase-two-runtime",
        db_llm_service=db_llm_service,
    )
    await runtime.setup(agent_id="agent-phase-two")
    operation = await runtime.kernel.create_operation(
        operation_id=operation_id,
        operation_type="db.run",
        request={
            "prompt": "phase two",
            "source_scope": ["orders"],
            **({"mode": mode} if mode else {}),
        },
        required_evidence=frozenset(required_evidence),
        metadata=({"mode": mode} if mode else {}),
        evaluate_governance=False,
    )
    return runtime, operation


async def _catalog_runtime_and_operation(operation_id):
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
            plugins=(CatalogPlugin(auto_persist=False), SQLitePlugin(path=":memory:")),
        ),
        runtime_id="phase-two-catalog-runtime",
        db_llm_service=FakeLLMService(json.dumps({"operation": "read"})),
    )
    await runtime.setup(agent_id="agent-phase-two")
    operation = await runtime.kernel.create_operation(
        operation_id=operation_id,
        operation_type="data.query",
        request={
            "prompt": "phase two catalog",
            "source_scope": ["sqlite"],
        },
        required_evidence=frozenset(),
        metadata={
            "source_scope": ["sqlite"],
            "safety_frame": {"max_access": "read"},
        },
        evaluate_governance=False,
    )
    return runtime, operation


async def _repair_runtime_and_operation(
    operation_id,
    content,
    *,
    prior_sql=None,
    failure_payload=None,
):
    prior_sql = prior_sql or "select 0 as answer"
    runtime, operation = await _runtime_and_operation(
        operation_id,
        db_llm_service=FakeLLMService(content),
    )
    await runtime.store.save_evidence(_planning_context_evidence(operation.id))
    await runtime.store.save_evidence(
        Evidence(
            id="prior-plan",
            kind="query.plan.proposal",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload={
                "valid": True,
                "sql": prior_sql,
                "structured_plan": {"selected_sql": prior_sql},
            },
        )
    )
    await runtime.store.save_evidence(
        Evidence(
            id="failure-validation",
            kind="query.plan.validation",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=False,
            payload=failure_payload
            or {"valid": False, "errors": ["validation failed"]},
        )
    )
    return runtime, operation


def _llm_task(operation, *, capability_id, executor_id, task_input):
    return Task(
        id=f"task-{capability_id.replace('.', '-')}",
        operation_id=operation.id,
        capability_id=capability_id,
        executor_id=executor_id,
        input=task_input,
    )


def _planner_decision(*actions):
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=actions,
    )


def _same_turn_repair_execute_decision():
    return _planner_decision(
        DbPlannerAction(
            action_id="repair_plan",
            kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
            input={"owner": "db_runtime"},
            depends_on=("failed_plan_validation",),
        ),
        DbPlannerAction(
            action_id="execute_repair",
            kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
            input={
                "owner": "phase_two",
                "query_plan_ref": "latest_accepted_query_plan",
            },
            depends_on=("repair_plan",),
        ),
    )


async def _compile_single_action(
    operation_id,
    action,
    *,
    plan_evidence=(),
    query_plan_validation_evidence=(),
    sql_validation_evidence=(),
    memory_proposal_evidence=(),
    planning_context=False,
    db_llm_service=None,
    explicit_mode=None,
    max_access="read",
):
    runtime, operation = await _runtime_and_operation(
        operation_id,
        mode=explicit_mode,
        db_llm_service=db_llm_service,
    )
    for item in plan_evidence:
        await runtime.store.save_evidence(
            _query_plan_evidence(operation.id, **dict(item))
        )
    for item in query_plan_validation_evidence:
        await runtime.store.save_evidence(
            _query_plan_validation_evidence(operation.id, **dict(item))
        )
    for item in sql_validation_evidence:
        await runtime.store.save_evidence(
            _sql_validation_evidence(operation.id, **dict(item))
        )
    for item in memory_proposal_evidence:
        await runtime.store.save_evidence(
            _memory_proposal_evidence(operation.id, **dict(item))
        )
    if planning_context:
        await runtime.store.save_evidence(_planning_context_evidence(operation.id))
    loop = DbAgentLoop(runtime, FakePlanner())
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": max_access},
        turn=1,
        remaining_turns=1,
    )
    assert_no_invalid_accepted_query_plans(
        await runtime.store.list_evidence(operation.id)
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(action,),
    )
    return loop.compile_actions(decision, state), runtime, operation


async def _compile_action_for_runtime(runtime, operation, action, *, max_access="read"):
    loop = DbAgentLoop(runtime, FakePlanner())
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": max_access},
        turn=1,
        remaining_turns=1,
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(action,),
    )
    compilation = loop.compile_actions(decision, state)
    assert compilation.rejected_action_summaries == ()
    return compilation


def _query_plan_evidence(
    operation_id,
    *,
    evidence_id,
    sql,
    accepted=True,
    valid=None,
    task_id=None,
):
    payload = {"valid": bool(sql) if valid is None else valid}
    if sql is not None:
        payload["sql"] = sql
    return Evidence(
        id=evidence_id,
        kind="query.plan.proposal",
        owner="phase_two",
        operation_id=operation_id,
        task_id=task_id or f"task-{evidence_id}",
        accepted=accepted,
        payload=payload,
        metadata={"payload_fingerprint": f"fp-{evidence_id}"},
    )


def _query_plan_validation_evidence(
    operation_id,
    *,
    evidence_id,
    plan_evidence_id="plan-accepted",
    valid=False,
    accepted=False,
    task_id=None,
    validation_facts=True,
):
    payload = {
        "valid": valid,
        "plan_evidence_id": plan_evidence_id,
    }
    if validation_facts:
        payload["errors"] = [
            "filter_literal_requires_grounding:orders.status=completed"
        ]
        payload["validation_facts"] = [
            {
                "kind": "filter_literal_requires_grounding",
                "table": "orders",
                "column": "status",
                "operator": "=",
                "literal": "completed",
                "source": "query.plan.validation",
            }
        ]
    return Evidence(
        id=evidence_id,
        kind="query.plan.validation",
        owner="db_runtime",
        operation_id=operation_id,
        task_id=task_id or f"task-{evidence_id}",
        accepted=accepted,
        payload=payload,
        metadata={"payload_fingerprint": f"fp-{evidence_id}"},
    )


def _sql_validation_evidence(
    operation_id,
    *,
    evidence_id,
    sql,
    operation="query",
    valid=True,
    accepted=True,
    task_id=None,
):
    return Evidence(
        id=evidence_id,
        kind="sql.validation",
        owner="phase_two",
        operation_id=operation_id,
        task_id=task_id or f"task-{evidence_id}",
        accepted=accepted,
        payload={"valid": valid, "sql": sql, "operation": operation},
        metadata={"payload_fingerprint": f"fp-{evidence_id}"},
    )


def _memory_proposal_evidence(
    operation_id,
    *,
    evidence_id,
    accepted=True,
    task_id=None,
    proposal_fingerprint=None,
):
    proposal_fingerprint = proposal_fingerprint or f"proposal-fp-{evidence_id}"
    return Evidence(
        id=evidence_id,
        kind="db.memory.proposal",
        owner="db_runtime",
        operation_id=operation_id,
        task_id=task_id or f"task-{evidence_id}",
        accepted=accepted,
        payload={"proposal_fingerprint": proposal_fingerprint},
        metadata={
            "payload_fingerprint": f"payload-fp-{evidence_id}",
            "proposal_fingerprint": proposal_fingerprint,
        },
    )


def _memory_definition_evidence(
    operation_id,
    *,
    evidence_id,
    proposal_evidence_id,
    proposal_fingerprint,
):
    return Evidence(
        id=evidence_id,
        kind="db.memory.definition",
        owner="db_runtime",
        operation_id=operation_id,
        task_id=f"task-{evidence_id}",
        accepted=True,
        payload={
            "proposal_evidence_id": proposal_evidence_id,
            "proposal_fingerprint": proposal_fingerprint,
            "committed": True,
        },
        metadata={
            "proposal_evidence_id": proposal_evidence_id,
            "proposal_fingerprint": proposal_fingerprint,
        },
    )


def _planning_context_evidence(operation_id):
    return Evidence(
        id="planning-context",
        kind="planning.context",
        owner="db_runtime",
        operation_id=operation_id,
        accepted=True,
        payload={
            "schema": {
                "database_type": "sqlite",
                "tables": [{"name": "orders", "columns": [{"name": "status"}]}],
            },
            "column_value_hints": [],
        },
        metadata={"payload_fingerprint": "fp-planning-context"},
    )


def _phase0_value_grounding_targets(task_input):
    targets = []
    for key in ("targets", "profile_pairs"):
        for item in task_input.get(key) or []:
            if isinstance(item, dict) and item.get("table") and item.get("column"):
                targets.append({"table": item["table"], "column": item["column"]})
    for key in ("validation_facts", "warnings", "validation_warnings"):
        for item in task_input.get(key) or []:
            if isinstance(item, dict) and item.get("table") and item.get("column"):
                targets.append({"table": item["table"], "column": item["column"]})
    return targets


def _loop_state():
    return DbLoopState(
        operation_id="op-loop",
        normalized_user_request={"prompt": "show one row"},
        safety_frame={"max_access": "read"},
        available_action_kinds=tuple(DbPlannerActionKind),
        capability_summaries=(
            {
                "id": "db.sql.execute_read",
                "owner": "phase_two",
                "access": "read",
            },
        ),
        runtime_limits={"max_tasks": 3},
        remaining_budget={"planner_turns": 1},
    )


def _llm_planner_payload(**overrides):
    payload = {
        "status": "continue",
        "intent": {"operation_type": "db.run"},
        "actions": [
            {
                "action_id": "read",
                "kind": "execute_validated_read",
                "input": {"owner": "phase_two", "sql": "select 1"},
                "depends_on": [],
                "rationale": "Need one row.",
                "metadata": {"source": "test"},
            }
        ],
        "stop_conditions": ["verified"],
        "clarification_question": None,
        "rationale": "Read from the database.",
        "metadata": {"planner": "fake"},
    }
    payload.update(overrides)
    return payload
