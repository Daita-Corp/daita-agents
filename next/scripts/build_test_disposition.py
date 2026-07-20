#!/usr/bin/env python3
"""Build the explicit v1-test disposition inventory for the v2 candidate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
from pathlib import Path
import subprocess

NEXT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = NEXT_ROOT.parent
OUTPUT_PATH = NEXT_ROOT / "TEST_DISPOSITION.csv"

ALLOWED_DISPOSITIONS = {
    "defer_documented",
    "preserve_acceptance",
    "port_leaf",
    "retire_internal",
}

POST_MVP_SUPPORT_TARGET = "next/tests/architecture/test_parity_matrix.py"

RETIRE_INTERNAL = {
    "tests/unit/catalog/test_catalog_discoverer.py": (
        "Phase 3",
        "next/tests/unit/adapters/test_contracts.py",
        "The v1 catalog-discoverer hierarchy is replaced by the narrow resource-adapter protocol.",
    ),
    "tests/unit/db/test_agent_loop_phase2.py": (
        "Phase 3",
        "next/tests/architecture/test_phase1_loop_architecture.py",
        "The database-specific planner/compiler loop is replaced by the single generic loop plus data-domain contracts.",
    ),
    "tests/unit/db/test_contract_builder.py": (
        "Phase 5",
        "next/tests/architecture/test_phase5_context_learning_architecture.py",
        "V1's database contract builder is replaced by capability declarations, context owners, and the generic loop.",
    ),
    "tests/unit/db/test_live_production_helpers.py": (
        "Phase 9",
        "next/tests/acceptance/test_sqlite_catalog_journey.py",
        "V1-only live-test helpers are not a product contract; replacement journeys assert the observable behavior directly.",
    ),
    "tests/unit/db/test_phase4_task_specs.py": (
        "Phase 6",
        "next/tests/architecture/test_phase6_monitor_architecture.py",
        "V1-path source scan is replaced by a v2 dependency-boundary assertion.",
    ),
    "tests/unit/db/test_planner_protocol_phase1.py": (
        "Phase 1",
        "next/tests/architecture/test_phase1_loop_architecture.py",
        "The v1 database planner protocol is replaced by canonical loop actions and persisted operation tasks.",
    ),
    "tests/unit/memory/test_working_memory.py": (
        "Phase 5",
        "next/tests/unit/context/test_budgeting.py",
        "The mutable v1 scratchpad is replaced by bounded context blocks and durable session checkpoints.",
    ),
    "tests/unit/plugins/test_plugin_base.py": (
        "Phase 8",
        "next/tests/contract/extensions/test_extension_registry.py",
        "Asserts the universal v1 plugin class hierarchy that ADR 0008 replaces.",
    ),
}

PORT_LEAF = {
    "tests/unit/catalog/test_catalog_normalizer.py": (
        "Phase 3",
        "next/tests/unit/catalog/test_models.py",
        "catalog identity normalization",
    ),
    "tests/unit/catalog/test_catalog_relationships.py": (
        "Phase 3",
        "next/tests/unit/catalog/test_sqlite_catalog_service.py",
        "bounded relationship normalization and traversal",
    ),
    "tests/unit/core/test_config.py": (
        "Phase 9",
        "next/tests/contract/config/test_agent_configuration.py",
        "public configuration validation",
    ),
    "tests/unit/core/test_exceptions.py": (
        "Phase 9",
        "next/tests/contract/test_errors.py",
        "typed public error behavior",
    ),
    "tests/unit/core/test_preprocessor.py": (
        "Phase 5",
        "next/tests/unit/memory/test_learning_service.py",
        "bounded learning-candidate preprocessing",
    ),
    "tests/unit/core/test_security.py": (
        "Phase 9",
        "next/tests/unit/domains/data/test_sql.py",
        "identifier, SQL, redaction, and untrusted-input safety",
    ),
    "tests/unit/db/test_context_projection.py": (
        "Phase 3",
        "next/tests/unit/domains/data/test_controller_context.py",
        "audit/model/public projection separation",
    ),
    "tests/unit/db/test_json_normalization.py": (
        "Phase 1",
        "next/tests/unit/test_json_values.py",
        "canonical JSON normalization",
    ),
    "tests/unit/db/test_loop_utils.py": (
        "Phase 1",
        "next/tests/unit/loop/test_loop_models.py",
        "loop fingerprint and bounded utility behavior",
    ),
    "tests/unit/db/test_plan_validation.py": (
        "Phase 3",
        "next/tests/unit/domains/data/test_sql.py",
        "SQL and scope validation",
    ),
    "tests/unit/db/test_query_planning.py": (
        "Phase 3",
        "next/tests/unit/domains/data/test_sql.py",
        "bounded query-plan normalization",
    ),
    "tests/unit/db/test_result_projection.py": (
        "Phase 3",
        "next/tests/unit/domains/data/test_results.py",
        "bounded result projection",
    ),
    "tests/unit/db/test_verification.py": (
        "Phase 3",
        "next/tests/unit/domains/data/test_controller_context.py",
        "evidence-grounded verification",
    ),
    "tests/unit/llm/test_llm_pricing.py": (
        "Phase 8",
        "next/tests/unit/llm/test_llm_models.py",
        "usage and cost normalization",
    ),
    "tests/unit/llm/test_provider_lifecycle.py": (
        "Phase 8",
        "next/tests/contract/models/test_provider_conformance.py",
        "lazy provider client lifecycle",
    ),
}

# Post-MVP rows intentionally share one executable documentation anchor. The
# parity-matrix architecture test proves that these surfaces are visibly
# deferred and cannot silently fall back to v1.
DEFER_DOCUMENTED = frozenset(
    {
        "tests/integration/catalog/test_catalog_aws_live.py",
        "tests/integration/catalog/test_catalog_azure_live.py",
        "tests/integration/catalog/test_catalog_gcp_live.py",
        "tests/integration/catalog/test_catalog_github_live.py",
        "tests/integration/catalog/test_catalog_mongodb_live.py",
        "tests/integration/catalog/test_catalog_mysql_live.py",
        "tests/integration/data/test_data_quality_integration.py",
        "tests/integration/data/test_lineage_live.py",
        "tests/integration/data/test_transformer_integration.py",
        "tests/integration/evals/test_from_db_evals_live.py",
        "tests/integration/evals/test_from_db_postgres_performance_live.py",
        "tests/integration/evals/test_from_db_postgres_profile_freshness_live.py",
        "tests/integration/evals/test_from_db_postgres_quality_benchmark_live.py",
        "tests/integration/evals/test_from_db_postgres_value_profiles_live.py",
        "tests/integration/evals/test_from_db_postgres_wide_schema_live.py",
        "tests/integration/evals/test_from_db_quality_benchmark_live.py",
        "tests/integration/focus/test_focus_live_llm.py",
        "tests/integration/focus/test_focus_sql_live.py",
        "tests/integration/from_db/test_schema_synthesis_specificity_live.py",
        "tests/integration/memory/test_memory_graph_live.py",
        "tests/integration/runtime/test_worker_agents_live.py",
        "tests/unit/catalog/test_catalog_azure.py",
        "tests/unit/catalog/test_plugin_catalog.py",
        "tests/unit/core/test_assertions.py",
        "tests/unit/data/test_data_quality.py",
        "tests/unit/data/test_transformer.py",
        "tests/unit/db/test_agent_loop_concurrency.py",
        "tests/unit/db/test_data_quality_runtime.py",
        "tests/unit/db/test_lineage_runtime.py",
        "tests/unit/db/test_multi_step_analysis.py",
        "tests/unit/evals/test_evals.py",
        "tests/unit/focus/test_focus.py",
        "tests/unit/focus/test_focus_real_world.py",
        "tests/unit/focus/test_focus_sql_backend.py",
        "tests/unit/llm/test_embeddings.py",
        "tests/unit/memory/test_memory_graph.py",
        "tests/unit/memory/test_memory_graph_eval.py",
        "tests/unit/memory/test_memory_graph_quality.py",
        "tests/unit/memory/test_memory_performance.py",
        "tests/unit/memory/test_memory_reinforcement.py",
        "tests/unit/plugins/test_mcp_runtime_declarations.py",
        "tests/unit/plugins/test_plugin_bigquery.py",
        "tests/unit/plugins/test_plugin_chroma.py",
        "tests/unit/plugins/test_plugin_elasticsearch.py",
        "tests/unit/plugins/test_plugin_email.py",
        "tests/unit/plugins/test_plugin_exa.py",
        "tests/unit/plugins/test_plugin_google_drive.py",
        "tests/unit/plugins/test_plugin_mongodb.py",
        "tests/unit/plugins/test_plugin_mysql.py",
        "tests/unit/plugins/test_plugin_neo4j.py",
        "tests/unit/plugins/test_plugin_pinecone.py",
        "tests/unit/plugins/test_plugin_qdrant.py",
        "tests/unit/plugins/test_plugin_redis.py",
        "tests/unit/plugins/test_plugin_redis_messaging.py",
        "tests/unit/plugins/test_plugin_rest.py",
        "tests/unit/plugins/test_plugin_s3.py",
        "tests/unit/plugins/test_plugin_slack.py",
        "tests/unit/plugins/test_plugin_snowflake.py",
        "tests/unit/plugins/test_plugin_websearch.py",
        "tests/unit/plugins/test_stage6_service_boundaries.py",
        "tests/unit/reference/test_phase0_domain_plugin_reference.py",
        "tests/unit/test_graph_tools.py",
    }
)


# Each retained black-box file names its real v2 owner and one executable
# anchor. Grouping keeps the inventory readable without falling back to a
# filename heuristic.
PRESERVE_GROUPS = (
    (
        "Phase 9",
        "next/tests/unit/llm/test_routing.py",
        "credential-gated retry behavior is owned by the canonical model router",
        ("tests/integration/agents/test_agent_retry_live.py",),
    ),
    (
        "Phase 2",
        "next/tests/acceptance/test_public_agent.py",
        "persistent Agent and session behavior",
        (
            "tests/integration/agents/test_conversation_history_live.py",
            "tests/unit/agents/test_agent_execution.py",
            "tests/unit/agents/test_agent_init.py",
            "tests/unit/agents/test_conversation_history.py",
            "tests/unit/db/test_session_context.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/live/test_postgresql_live.py",
        "real PostgreSQL catalog and bounded-read behavior crosses the least-privileged service boundary",
        (
            "tests/integration/catalog/test_catalog_postgres_live.py",
            "tests/integration/from_db/test_from_db_live_postgres.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/acceptance/test_postgresql_catalog_journey.py",
        "PostgreSQL scale behavior uses the same catalog/runtime journey",
        (
            "tests/performance/from_db/test_live_from_db_benchmarks.py",
            "tests/performance/from_db/test_postgres_catalog_profile_scale_live.py",
            "tests/performance/from_db/test_postgres_large_schema_load_live.py",
            "tests/performance/from_db/test_postgres_llm_planning_load_live.py",
            "tests/performance/from_db/test_postgres_warm_deterministic_load_live.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/acceptance/test_sqlite_catalog_journey.py",
        "credential-gated SQLite planning, grounding, and schema behavior",
        (
            "tests/integration/from_db/test_from_db_live_edge_cases.py",
            "tests/integration/from_db/test_from_db_live_production_contracts.py",
            "tests/integration/from_db/test_from_db_live_sqlite.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/acceptance/test_learning_journey.py",
        "credential-gated memory correctness and memory performance",
        (
            "tests/integration/from_db/test_from_db_live_memory_contracts.py",
            "tests/integration/from_db/test_from_db_memory_live.py",
            "tests/performance/from_db/test_db_memory_latency.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/acceptance/test_monitor_lifecycle.py",
        "credential/service-backed monitor and scheduler behavior",
        (
            "tests/integration/from_db/test_from_db_live_monitors.py",
            "tests/integration/from_db/test_monitor_intent_extraction_live.py",
            "tests/integration/runtime/test_monitor_db_live.py",
            "tests/integration/runtime/test_monitor_worker_pipeline_live.py",
            "tests/performance/from_db/test_runtime_worker_monitor_load_live.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/acceptance/test_sqlite_update_approval_journey.py",
        "credential-gated write governance and durable resume",
        ("tests/integration/from_db/test_from_db_live_resume_governance.py",),
    ),
    (
        "Phase 9",
        "next/tests/contract/telemetry/test_event_projection.py",
        "committed-event telemetry and observability projection",
        (
            "tests/integration/from_db/test_from_db_tracing_telemetry_live.py",
            "tests/performance/from_db/test_runtime_observability_contract.py",
            "tests/unit/core/test_tracing.py",
            "tests/unit/db/test_from_db_tracing_telemetry.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/live/test_model_providers_live.py",
        "retained-provider behavior crosses each real credential or service boundary",
        ("tests/integration/llm/test_llm_providers_live.py",),
    ),
    (
        "Phase 9",
        "next/tests/acceptance/test_loop_budgets.py",
        "candidate latency/token budgets remain bounded and observable",
        ("tests/performance/agents/test_live_agent_benchmarks.py",),
    ),
    (
        "Phase 9",
        "next/tests/contract/extensions/test_local_capability.py",
        "public local tools are projections of registered capabilities",
        (
            "tests/unit/agents/test_agent_tools.py",
            "tests/unit/core/test_tools.py",
            "tests/unit/reference/test_phase0_agent_plugin_reference.py",
            "tests/unit/runtime/test_tool_adapters.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/contract/test_errors.py",
        "public failures use the stable replacement error taxonomy",
        ("tests/unit/agents/test_error_paths.py",),
    ),
    (
        "Phase 9",
        "next/tests/acceptance/test_public_agent_stream_and_detach.py",
        "public streaming and detached operation ownership",
        (
            "tests/unit/agents/test_streaming.py",
            "tests/unit/db/test_agent_from_db.py",
        ),
    ),
    (
        "Phase 9",
        "next/tests/acceptance/test_public_agent.py",
        "replacement public Agent/inspection surface",
        ("tests/unit/db/test_public_api.py",),
    ),
    (
        "Phase 9",
        "next/tests/contract/extensions/test_extension_registry.py",
        "validated public extension declarations and collision diagnostics",
        ("tests/unit/plugins/test_extension_registry.py",),
    ),
    (
        "Phase 1",
        "next/tests/acceptance/test_text_only_loop.py",
        "single generic text/tool loop behavior",
        ("tests/unit/agents/test_chat_runtime.py",),
    ),
    (
        "Phase 1",
        "next/tests/acceptance/test_structured_action_repair.py",
        "bounded no-progress and structured repair behavior",
        ("tests/unit/db/test_agent_loop_completion_targets.py",),
    ),
    (
        "Phase 3",
        "next/tests/acceptance/test_structured_action_repair.py",
        "data-action validation and bounded repair behavior",
        ("tests/unit/db/test_agent_loop_live_hardening.py",),
    ),
    (
        "Phase 2",
        "next/tests/unit/operations/test_capability_execution_boundary.py",
        "accepted evidence and executor-boundary behavior",
        (
            "tests/unit/db/test_evidence.py",
            "tests/unit/runtime/test_executors.py",
        ),
    ),
    (
        "Phase 7",
        "next/tests/acceptance/test_fake_side_effect_approval.py",
        "shared governance, approval, and resume semantics",
        (
            "tests/unit/db/test_governance_runtime.py",
            "tests/unit/runtime/test_governance.py",
        ),
    ),
    (
        "Phase 8",
        "next/tests/unit/llm/test_routing.py",
        "model retry and configured planning route behavior",
        (
            "tests/unit/agents/test_agent_runtime_retry.py",
            "tests/unit/db/test_llm_native_planning.py",
        ),
    ),
    (
        "Phase 8",
        "next/tests/contract/models/test_provider_conformance.py",
        "canonical provider lifecycle and response behavior",
        (
            "tests/unit/db/test_llm_service_lifecycle.py",
            "tests/unit/llm/test_llm_base.py",
        ),
    ),
    (
        "Phase 1",
        "next/tests/unit/llm/test_mock_provider.py",
        "deterministic mock model behavior",
        ("tests/unit/llm/test_llm_mock.py",),
    ),
    (
        "Phase 3",
        "next/tests/unit/catalog/test_sqlite_catalog_service.py",
        "catalog declarations, search, inspection, and relationships",
        ("tests/unit/catalog/test_catalog_extensions.py",),
    ),
    (
        "Phase 5",
        "next/tests/unit/skills/test_skill_service.py",
        "bounded skill discovery and activation",
        ("tests/unit/core/test_skills.py",),
    ),
    (
        "Phase 3",
        "next/tests/unit/domains/data/test_controller_context.py",
        "evidence-grounded synthesis and readiness",
        (
            "tests/unit/db/test_llm_synthesis.py",
            "tests/unit/db/test_synthesis.py",
        ),
    ),
    (
        "Phase 5",
        "next/tests/unit/memory/test_learning_service.py",
        "explicit memory commands and provenance-backed learning",
        (
            "tests/unit/db/test_memory_commands.py",
            "tests/unit/db/test_memory_learning.py",
        ),
    ),
    (
        "Phase 5",
        "next/tests/unit/memory/test_service.py",
        "scoped durable memory lifecycle and retrieval",
        (
            "tests/unit/db/test_memory_runtime.py",
            "tests/unit/memory/test_memory_data_ops.py",
            "tests/unit/memory/test_memory_lifecycle.py",
        ),
    ),
    (
        "Phase 6",
        "next/tests/acceptance/test_monitor_lifecycle.py",
        "public monitor lifecycle and evidence linkage",
        (
            "tests/unit/db/test_db_monitor_commands.py",
            "tests/unit/db/test_monitor_evidence.py",
        ),
    ),
    (
        "Phase 6",
        "next/tests/unit/monitors/test_scheduler.py",
        "durable due-tick leases, cooldown, and catch-up",
        ("tests/unit/db/test_db_monitor_scheduler.py",),
    ),
    (
        "Phase 6",
        "next/tests/unit/monitors/test_service.py",
        "monitor records and service mutations",
        ("tests/unit/db/test_db_monitors.py",),
    ),
    (
        "Phase 3",
        "next/tests/acceptance/test_sqlite_catalog_journey.py",
        "read-only SQLite operation execution through catalog and runtime",
        (
            "tests/unit/db/test_operation_execution.py",
            "tests/unit/db/test_sqlite_vertical_slice.py",
        ),
    ),
    (
        "Phase 8",
        "next/tests/acceptance/test_postgresql_catalog_journey.py",
        "PostgreSQL data-domain execution through the shared runtime",
        ("tests/unit/db/test_postgresql_runtime.py",),
    ),
    (
        "Phase 2",
        "next/tests/unit/operations/test_runtime_recovery.py",
        "runtime lifecycle and restart recovery",
        (
            "tests/unit/db/test_runtime_lifecycle.py",
            "tests/unit/db/test_runtime_loop_phase5.py",
            "tests/unit/db/test_runtime_skeleton.py",
        ),
    ),
    (
        "Phase 1",
        "next/tests/architecture/test_phase1_loop_architecture.py",
        "the operation runtime remains the sole task materialization/execution boundary",
        ("tests/unit/db/test_task_materialization_guards.py",),
    ),
    (
        "Phase 8",
        "next/tests/unit/adapters/test_postgresql.py",
        "PostgreSQL adapter lifecycle and lazy dependency behavior",
        ("tests/unit/plugins/test_plugin_postgresql.py",),
    ),
    (
        "Phase 3",
        "next/tests/unit/adapters/test_sqlite.py",
        "SQLite adapter lifecycle and direct adapter contract",
        ("tests/unit/plugins/test_sqlite.py",),
    ),
    (
        "Phase 6",
        "next/tests/unit/hosting/test_host.py",
        "local host context and lifecycle",
        ("tests/unit/runtime/test_hosting.py",),
    ),
    (
        "Phase 2",
        "next/tests/contract/operations/test_in_memory_task_execution_lifecycle.py",
        "lease fencing, committed events, and runtime execution",
        ("tests/unit/runtime/test_kernel.py",),
    ),
    (
        "Phase 6",
        "next/tests/architecture/test_phase6_monitor_architecture.py",
        "monitors and workers reuse the ordinary runtime boundary",
        ("tests/unit/runtime/test_phase5_monitors_workers.py",),
    ),
    (
        "Phase 1",
        "next/tests/unit/operations/test_operation_models.py",
        "canonical runtime primitive validation and serialization",
        ("tests/unit/runtime/test_primitives.py",),
    ),
    (
        "Phase 2",
        "next/tests/contract/runtime/test_operation_store.py",
        "durable operation/task/event store semantics",
        ("tests/unit/runtime/test_store.py",),
    ),
)


def _preserve_acceptance() -> dict[str, tuple[str, str, str]]:
    classified: dict[str, tuple[str, str, str]] = {}
    for phase, target, behavior, paths in PRESERVE_GROUPS:
        for path in paths:
            if path in classified:
                raise ValueError(f"Duplicate preserve-acceptance disposition: {path}")
            classified[path] = (phase, target, behavior)
    return classified


PRESERVE_ACCEPTANCE = _preserve_acceptance()


@dataclass(frozen=True)
class DispositionRow:
    path: str
    disposition: str
    v2_phase: str
    v2_target: str
    rationale: str


def tracked_test_modules() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "tests"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = {
        line.strip()
        for line in result.stdout.splitlines()
        if Path(line.strip()).name.startswith("test_")
        and Path(line.strip()).suffix == ".py"
    }
    return tuple(sorted(paths))


def classify(path: str) -> DispositionRow:
    if path in DEFER_DOCUMENTED:
        return DispositionRow(
            path,
            "defer_documented",
            "post-MVP",
            POST_MVP_SUPPORT_TARGET,
            "Explicitly unsupported by the replacement candidate; the support matrix documents the deferral and prohibits a v1 fallback.",
        )

    if path in RETIRE_INTERNAL:
        phase, target, rationale = RETIRE_INTERNAL[path]
        return DispositionRow(path, "retire_internal", phase, target, rationale)

    if path in PORT_LEAF:
        phase, target, behavior = PORT_LEAF[path]
        return DispositionRow(
            path,
            "port_leaf",
            phase,
            target,
            f"Focused v1 leaf behavior retained under the v2 owner: {behavior}.",
        )

    preserved = PRESERVE_ACCEPTANCE.get(path)
    if preserved is not None:
        phase, target, behavior = preserved
        return DispositionRow(
            path,
            "preserve_acceptance",
            phase,
            target,
            f"Preserve observable behavior through the v2 owner: {behavior}; do not import v1.",
        )

    raise ValueError(f"Tracked v1 test has no explicit v2 disposition: {path}")


def build_rows() -> tuple[DispositionRow, ...]:
    tracked = set(tracked_test_modules())
    categories = {
        "defer_documented": set(DEFER_DOCUMENTED),
        "retire_internal": set(RETIRE_INTERNAL),
        "port_leaf": set(PORT_LEAF),
        "preserve_acceptance": set(PRESERVE_ACCEPTANCE),
    }
    seen: set[str] = set()
    duplicates: set[str] = set()
    for paths in categories.values():
        duplicates.update(seen.intersection(paths))
        seen.update(paths)

    if duplicates:
        raise ValueError(f"Tests have multiple dispositions: {sorted(duplicates)}")

    missing = tracked - seen
    stale = seen - tracked
    if missing or stale:
        raise ValueError(
            "Disposition tables do not match tracked v1 tests: "
            f"missing={sorted(missing)}, stale={sorted(stale)}"
        )

    return tuple(classify(path) for path in sorted(tracked))


def render_csv(rows: tuple[DispositionRow, ...] | None = None) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=("path", "disposition", "v2_phase", "v2_target", "rationale"),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows or build_rows():
        writer.writerow(
            {
                "path": row.path,
                "disposition": row.disposition,
                "v2_phase": row.v2_phase,
                "v2_target": row.v2_target,
                "rationale": row.rationale,
            }
        )
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--stdout", action="store_true")
    arguments = parser.parse_args()

    rendered = render_csv()
    if arguments.stdout:
        print(rendered, end="")
        return 0
    if arguments.check:
        if (
            not OUTPUT_PATH.exists()
            or OUTPUT_PATH.read_text(encoding="utf-8") != rendered
        ):
            raise SystemExit(f"{OUTPUT_PATH} is stale; run with --write")
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
