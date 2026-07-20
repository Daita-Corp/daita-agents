from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = PROJECT_ROOT / "examples"
RETAINED = (
    "00_quickstart_sqlite_from_db.py",
    "01_inspectable_operation.py",
    "02_catalog_assisted_joins.py",
    "03_governed_reads_and_writes.py",
    "04_persistent_runtime_store.py",
    "06_memory_for_business_semantics.py",
    "07_monitor_orders.py",
    "09_custom_data_plugin_extension.py",
    "10_csv_to_sqlite_data_app.py",
)
DEFERRED = (
    "05_data_quality_and_lineage.py",
    "08_infrastructure_catalog.py",
)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    local_source = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        local_source if not existing else local_source + os.pathsep + existing
    )
    return environment


@pytest.mark.parametrize("filename", RETAINED)
def test_every_retained_example_has_an_executable_help_path(filename: str) -> None:
    completed = subprocess.run(
        (sys.executable, str(EXAMPLES / filename), "--help"),
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.casefold()


def test_quickstart_runs_safely_offline_with_an_explicit_fresh_root(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(EXAMPLES / RETAINED[0]),
            "--root",
            str(tmp_path / "fresh-v2-state"),
        ),
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "answer: There are 3 orders. [evidence:evidence-1]" in completed.stdout
    assert "(completed)" in completed.stdout


def test_retained_set_and_deferred_examples_are_explicit() -> None:
    present = {path.name for path in EXAMPLES.glob("[0-9][0-9]_*.py")}
    assert present == set(RETAINED)
    assert all(not (EXAMPLES / filename).exists() for filename in DEFERRED)

    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "deferred from the 2.0 MVP" in readme
    assert all(filename in readme for filename in DEFERRED)


def test_examples_use_v2_surfaces_without_legacy_imports_or_auto_approval() -> None:
    forbidden_imports = ("daita.agents", "daita.db", "daita.plugins")
    for filename in RETAINED:
        source = (EXAMPLES / filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(
            name == forbidden or name.startswith(forbidden + ".")
            for name in imported
            for forbidden in forbidden_imports
        )

    governed_write = (EXAMPLES / RETAINED[3]).read_text(encoding="utf-8")
    governed_tree = ast.parse(governed_write, filename=RETAINED[3])
    decision_calls = {
        node.func.attr
        for node in ast.walk(governed_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        if node.func.attr in {"approve", "reject"}
    }
    assert decision_calls == set()


def test_local_host_deployment_has_safe_help_and_dry_run(tmp_path: Path) -> None:
    deployment = EXAMPLES / "deployments" / "data-team-agent" / "run.py"
    help_result = subprocess.run(
        (sys.executable, str(deployment), "--help"),
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr

    dry_run = subprocess.run(
        (
            sys.executable,
            str(deployment),
            "--root",
            str(tmp_path / "dedicated-state"),
            "--dry-run",
        ),
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "mode: open" in dry_run.stdout
    assert "model: openai:gpt-4.1-mini" in dry_run.stdout
    assert not (tmp_path / "dedicated-state").exists()
