import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_ROOT = REPO_ROOT / "daita" / "db"
TASK_OWNER = DB_ROOT / "runtime" / "tasks"
TASK_EXECUTION_OWNER = TASK_OWNER / "execution.py"
KERNEL_OWNER = REPO_ROOT / "daita" / "runtime" / "kernel.py"
RAW_VALIDATED_READ_HELPER = "plan_validated_read" + "_tasks"
VALIDATED_SPEC_HELPERS = (
    "plan_validated_read" + "_spec",
    "plan_validated_write" + "_spec",
)


def _production_python_files():
    return tuple(
        path for path in DB_ROOT.rglob("*.py") if TASK_OWNER not in path.parents
    )


def _call_matches(root, predicate):
    return [
        (path, node.lineno)
        for path in root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and predicate(node.func)
    ]


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


def test_runtime_kernel_is_the_only_executor_invocation_owner():
    matches = _call_matches(
        REPO_ROOT / "daita",
        lambda function: (
            isinstance(function, ast.Attribute)
            and function.attr == "execute"
            and isinstance(function.value, ast.Name)
            and function.value.id == "executor"
        ),
    )

    assert matches
    assert {path for path, _ in matches} == {KERNEL_OWNER}


def test_db_direct_kernel_execution_calls_stay_in_task_execution_owner():
    matches = _call_matches(
        DB_ROOT,
        lambda function: (
            isinstance(function, ast.Attribute)
            and function.attr in {"execute_task", "execute_claimed_task"}
            and (
                (isinstance(function.value, ast.Name) and function.value.id == "kernel")
                or (
                    isinstance(function.value, ast.Attribute)
                    and function.value.attr == "kernel"
                )
            )
        ),
    )

    assert matches
    assert {path for path, _ in matches} == {TASK_EXECUTION_OWNER}
