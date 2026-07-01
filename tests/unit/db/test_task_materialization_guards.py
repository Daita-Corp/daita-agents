import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DB_ROOT = REPO_ROOT / "daita" / "db"
TASK_OWNER = DB_ROOT / "runtime" / "tasks.py"
RAW_VALIDATED_READ_HELPER = "plan_validated_read" + "_tasks"
VALIDATED_SPEC_HELPERS = (
    "plan_validated_read" + "_spec",
    "plan_validated_write" + "_spec",
)


def _production_python_files():
    return tuple(path for path in DB_ROOT.rglob("*.py") if path != TASK_OWNER)


def test_private_task_materializers_stay_inside_task_owner():
    pattern = re.compile(r"\._task_for_(?:spec|capability)\(")
    matches = [
        f"{path.relative_to(REPO_ROOT)}:{line_no}"
        for path in _production_python_files()
        for line_no, line in enumerate(path.read_text().splitlines(), start=1)
        if pattern.search(line)
    ]

    assert matches == []


def test_raw_validated_read_helper_is_not_production_reachable():
    matches = [
        f"{path.relative_to(REPO_ROOT)}:{line_no}"
        for path in DB_ROOT.rglob("*.py")
        for line_no, line in enumerate(path.read_text().splitlines(), start=1)
        if RAW_VALIDATED_READ_HELPER in line
    ]

    assert matches == []


def test_validated_spec_helpers_are_not_production_materialization_paths():
    matches = [
        f"{path.relative_to(REPO_ROOT)}:{line_no}"
        for path in DB_ROOT.rglob("*.py")
        for line_no, line in enumerate(path.read_text().splitlines(), start=1)
        if any(helper in line for helper in VALIDATED_SPEC_HELPERS)
    ]

    assert matches == []


def test_workflow_code_does_not_plan_kernel_tasks_directly():
    matches = [
        f"{path.relative_to(REPO_ROOT)}:{line_no}"
        for path in _production_python_files()
        for line_no, line in enumerate(path.read_text().splitlines(), start=1)
        if "kernel.plan_task" in line
    ]

    assert matches == []
