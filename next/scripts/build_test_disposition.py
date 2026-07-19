#!/usr/bin/env python3
"""Build the Phase 0 disposition inventory for tracked v1 test modules."""

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
    "preserve_acceptance",
    "port_leaf",
    "retire_internal",
}

RETIRE_INTERNAL = {
    "tests/unit/db/test_phase4_task_specs.py": (
        "Phase 6",
        "next/tests/architecture/test_monitor_boundaries.py",
        "V1-path source scan is replaced by a v2 dependency-boundary assertion.",
    ),
    "tests/unit/plugins/test_plugin_base.py": (
        "Phase 8",
        "next/tests/contract/extensions/test_registry.py",
        "Asserts the universal v1 plugin class hierarchy that ADR 0008 replaces.",
    ),
}

PORT_LEAF = {
    "tests/unit/catalog/test_catalog_normalizer.py": (
        "Phase 3",
        "catalog identity normalization",
    ),
    "tests/unit/catalog/test_catalog_relationships.py": (
        "Phase 3",
        "bounded relationship normalization and traversal",
    ),
    "tests/unit/core/test_config.py": ("Phase 2", "configuration validation"),
    "tests/unit/core/test_exceptions.py": ("Phase 1", "typed error behavior"),
    "tests/unit/core/test_preprocessor.py": ("Phase 4", "bounded input preprocessing"),
    "tests/unit/core/test_security.py": (
        "Phase 4",
        "redaction and untrusted-input safety",
    ),
    "tests/unit/db/test_context_projection.py": (
        "Phase 3",
        "audit/model/public projection separation",
    ),
    "tests/unit/db/test_json_normalization.py": (
        "Phase 1",
        "canonical JSON normalization",
    ),
    "tests/unit/db/test_loop_utils.py": (
        "Phase 1",
        "loop fingerprint and bounded utility behavior",
    ),
    "tests/unit/db/test_plan_validation.py": ("Phase 3", "SQL and scope validation"),
    "tests/unit/db/test_result_projection.py": ("Phase 3", "bounded result projection"),
    "tests/unit/db/test_verification.py": ("Phase 3", "evidence-grounded verification"),
    "tests/unit/llm/test_llm_pricing.py": ("Phase 8", "usage and cost normalization"),
    "tests/unit/llm/test_provider_lifecycle.py": (
        "Phase 8",
        "lazy provider client lifecycle",
    ),
    "tests/unit/memory/test_memory_graph_quality.py": (
        "Phase 5",
        "memory candidate quality scoring",
    ),
    "tests/unit/plugins/projection_helpers.py": (
        "Phase 3",
        "projection helper behavior",
    ),
}

# The Phase 0 filename heuristic is deliberately conservative, but Phase 3's
# SQLite-only scope is now concrete. These explicit dispositions prevent a
# path containing ``catalog`` or ``from_db`` from silently pulling deferred
# integrations or Phase 9 hardening into the data vertical slice.
PHASE_OVERRIDES = {
    "tests/integration/catalog/test_catalog_aws_live.py": "post-MVP",
    "tests/integration/catalog/test_catalog_azure_live.py": "post-MVP",
    "tests/integration/catalog/test_catalog_gcp_live.py": "post-MVP",
    "tests/integration/catalog/test_catalog_github_live.py": "post-MVP",
    "tests/integration/catalog/test_catalog_mongodb_live.py": "post-MVP",
    "tests/integration/catalog/test_catalog_mysql_live.py": "post-MVP",
    "tests/integration/evals/test_from_db_evals_live.py": "Phase 9",
    "tests/integration/evals/test_from_db_quality_benchmark_live.py": "Phase 9",
    "tests/integration/focus/test_focus_sql_live.py": "post-MVP",
    "tests/integration/from_db/test_from_db_tracing_telemetry_live.py": "Phase 9",
    "tests/integration/from_db/test_schema_synthesis_specificity_live.py": "Phase 9",
    "tests/unit/catalog/test_catalog_azure.py": "post-MVP",
    "tests/unit/db/test_from_db_tracing_telemetry.py": "Phase 9",
    "tests/unit/focus/test_focus_sql_backend.py": "post-MVP",
    "tests/unit/plugins/test_plugin_mysql.py": "post-MVP",
}


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


def _phase_for(path: str) -> str:
    overridden = PHASE_OVERRIDES.get(path)
    if overridden is not None:
        return overridden
    lowered = path.lower()
    if "/performance/" in lowered:
        return "Phase 9"
    if any(token in lowered for token in ("monitor", "scheduler")):
        return "Phase 6"
    if any(token in lowered for token in ("memory", "skill")):
        return "Phase 5"
    if any(
        token in lowered
        for token in (
            "anthropic",
            "gemini",
            "grok",
            "ollama",
            "provider",
            "postgres",
        )
    ):
        return "Phase 8"
    if any(token in lowered for token in ("governance", "approval", "write")):
        return "Phase 7"
    if any(token in lowered for token in ("catalog", "sqlite", "sql", "from_db")):
        return "Phase 3"
    if any(token in lowered for token in ("runtime", "store", "hosting", "stream")):
        return "Phase 2"
    if any(token in lowered for token in ("agent_loop", "chat_runtime", "tool")):
        return "Phase 1"
    if any(
        token in lowered
        for token in (
            "/evals/",
            "/focus/",
            "/data/",
            "data_quality",
            "lineage",
            "graph",
        )
    ):
        return "post-MVP"
    if "/plugins/" in lowered:
        return "Phase 9"
    return "Phase 9"


def _preservation_target(path: str) -> str:
    relative = Path(path).relative_to("tests")
    return str(Path("next/tests/acceptance/v1_oracles") / relative)


def _leaf_target(path: str) -> str:
    relative = Path(path).relative_to("tests")
    return str(Path("next/tests/unit/ported") / relative)


def classify(path: str) -> DispositionRow:
    if path in RETIRE_INTERNAL:
        phase, target, rationale = RETIRE_INTERNAL[path]
        return DispositionRow(path, "retire_internal", phase, target, rationale)

    if path in PORT_LEAF:
        phase, behavior = PORT_LEAF[path]
        return DispositionRow(
            path,
            "port_leaf",
            phase,
            _leaf_target(path),
            f"Focused v1 leaf behavior to port under the v2 owner: {behavior}.",
        )

    phase = _phase_for(path)
    return DispositionRow(
        path,
        "preserve_acceptance",
        phase,
        _preservation_target(path),
        "Conservatively preserve observable behavior through a v2 black-box or contract oracle; do not import v1.",
    )


def build_rows() -> tuple[DispositionRow, ...]:
    return tuple(classify(path) for path in tracked_test_modules())


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
