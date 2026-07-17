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
    "preserve_acceptance",
    "port_leaf",
    "retire_internal",
}
ALLOWED_PHASES = {*(f"Phase {number}" for number in range(1, 10)), "post-MVP"}


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
        "tests/unit/db/test_phase4_task_specs.py",
        "tests/unit/plugins/test_plugin_base.py",
    }


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
