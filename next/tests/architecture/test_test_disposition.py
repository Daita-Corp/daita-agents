from __future__ import annotations

import csv
import io
from pathlib import Path
import subprocess
import sys

NEXT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = NEXT_ROOT.parent
INVENTORY = NEXT_ROOT / "TEST_DISPOSITION.csv"
GENERATOR = NEXT_ROOT / "scripts" / "build_test_disposition.py"

ALLOWED_DISPOSITIONS = {
    "defer_documented",
    "preserve_acceptance",
    "port_leaf",
    "retire_internal",
}
ALLOWED_PHASES = {*(f"Phase {number}" for number in range(1, 10)), "post-MVP"}
POST_MVP_SUPPORT_TARGET = "next/tests/architecture/test_parity_matrix.py"


def _inventory_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(INVENTORY.read_text(encoding="utf-8"))))


def _tracked_test_modules() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "tests"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if Path(line.strip()).name.startswith("test_")
        and Path(line.strip()).suffix == ".py"
    }


def test_inventory_covers_every_tracked_v1_test_module_once() -> None:
    rows = _inventory_rows()
    paths = [row["path"] for row in rows]

    assert len(paths) == len(set(paths))
    assert set(paths) == _tracked_test_modules()
    assert len(paths) == 164


def test_inventory_values_are_complete_and_conservative() -> None:
    rows = _inventory_rows()

    assert all(row["disposition"] in ALLOWED_DISPOSITIONS for row in rows)
    assert all(row["v2_phase"] in ALLOWED_PHASES for row in rows)
    assert all(row["v2_target"].startswith("next/tests/") for row in rows)
    assert all(row["rationale"].strip() for row in rows)

    retired = {row["path"] for row in rows if row["disposition"] == "retire_internal"}
    assert retired == {
        "tests/unit/catalog/test_catalog_discoverer.py",
        "tests/unit/db/test_agent_loop_phase2.py",
        "tests/unit/db/test_contract_builder.py",
        "tests/unit/db/test_live_production_helpers.py",
        "tests/unit/db/test_phase4_task_specs.py",
        "tests/unit/db/test_planner_protocol_phase1.py",
        "tests/unit/memory/test_working_memory.py",
        "tests/unit/plugins/test_plugin_base.py",
    }


def test_active_targets_exist_and_post_mvp_uses_one_support_anchor() -> None:
    rows = _inventory_rows()
    active = [row for row in rows if row["v2_phase"] != "post-MVP"]
    deferred = [row for row in rows if row["v2_phase"] == "post-MVP"]

    missing = sorted(
        row["v2_target"]
        for row in active
        if not (REPOSITORY_ROOT / row["v2_target"]).is_file()
    )
    assert missing == []
    assert all(row["disposition"] != "defer_documented" for row in active)
    assert deferred
    assert all(row["disposition"] == "defer_documented" for row in deferred)
    assert {row["v2_target"] for row in deferred} == {POST_MVP_SUPPORT_TARGET}
    assert (REPOSITORY_ROOT / POST_MVP_SUPPORT_TARGET).is_file()


def test_inventory_matches_deterministic_generator_output() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--stdout"],
        cwd=NEXT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == INVENTORY.read_text(encoding="utf-8")


def test_inventory_keeps_explicitly_deferred_surfaces_out_of_active_phases() -> None:
    phases = {row["path"]: row["v2_phase"] for row in _inventory_rows()}

    for path in (
        "tests/integration/catalog/test_catalog_aws_live.py",
        "tests/integration/catalog/test_catalog_azure_live.py",
        "tests/integration/catalog/test_catalog_gcp_live.py",
        "tests/integration/catalog/test_catalog_github_live.py",
        "tests/integration/catalog/test_catalog_mongodb_live.py",
        "tests/integration/catalog/test_catalog_mysql_live.py",
        "tests/integration/evals/test_from_db_evals_live.py",
        "tests/integration/evals/test_from_db_postgres_performance_live.py",
        "tests/integration/evals/test_from_db_quality_benchmark_live.py",
        "tests/integration/from_db/test_schema_synthesis_specificity_live.py",
        "tests/unit/catalog/test_catalog_azure.py",
        "tests/unit/db/test_data_quality_runtime.py",
        "tests/unit/db/test_lineage_runtime.py",
        "tests/unit/evals/test_evals.py",
        "tests/unit/llm/test_embeddings.py",
        "tests/unit/memory/test_memory_graph.py",
        "tests/unit/plugins/test_mcp_runtime_declarations.py",
        "tests/unit/plugins/test_plugin_bigquery.py",
        "tests/unit/plugins/test_plugin_google_drive.py",
        "tests/unit/plugins/test_plugin_mysql.py",
        "tests/unit/plugins/test_plugin_redis.py",
        "tests/unit/plugins/test_plugin_websearch.py",
    ):
        assert phases[path] == "post-MVP"

    for path in (
        "tests/integration/from_db/test_from_db_tracing_telemetry_live.py",
        "tests/unit/db/test_from_db_tracing_telemetry.py",
    ):
        assert phases[path] == "Phase 9"


def test_live_external_v1_modules_map_to_real_v2_live_acceptance() -> None:
    targets = {row["path"]: row["v2_target"] for row in _inventory_rows()}

    assert targets["tests/integration/llm/test_llm_providers_live.py"] == (
        "next/tests/live/test_model_providers_live.py"
    )
    for path in (
        "tests/integration/catalog/test_catalog_postgres_live.py",
        "tests/integration/from_db/test_from_db_live_postgres.py",
    ):
        assert targets[path] == "next/tests/live/test_postgresql_live.py"
