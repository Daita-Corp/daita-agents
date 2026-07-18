from __future__ import annotations

import ast
from pathlib import Path
import re

NEXT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = NEXT_ROOT / "src" / "daita"
CAPABILITIES = SOURCE_ROOT / "capabilities.py"
OPERATION_STORE = SOURCE_ROOT / "operations" / "store.py"
OPERATION_RUNTIME = SOURCE_ROOT / "operations" / "runtime.py"
OPERATION_LEASES = SOURCE_ROOT / "operations" / "leases.py"
SQLITE_ADAPTER = SOURCE_ROOT / "storage" / "sqlite.py"

TASK_EXECUTION_METHODS = {
    "claim_task",
    "renew_task_lease",
    "commit_fenced",
    "recover_expired_task",
}


def _production_trees() -> tuple[tuple[Path, ast.Module], ...]:
    return tuple(
        (
            path,
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
        )
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
    )


def _required_tree(path: Path) -> ast.Module:
    assert path.is_file(), f"missing required task-lifecycle owner: {path}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _relative_path(path: Path) -> str:
    return path.relative_to(SOURCE_ROOT).as_posix()


def _class_definition(tree: ast.Module, name: str) -> ast.ClassDef:
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(definitions) == 1, f"expected exactly one {name} definition"
    return definitions[0]


def _public_methods(class_definition: ast.ClassDef) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in class_definition.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _function_arguments(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }


def _assigned_self_attributes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    attributes: set[str] = set()
    for node in ast.walk(function):
        targets: tuple[ast.expr, ...]
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attributes.add(target.attr)
    return attributes


def _attribute_parts(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _docstring_constants(tree: ast.Module) -> set[ast.Constant]:
    constants: set[ast.Constant] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants.add(value)
    return constants


def test_both_operation_adapters_concretely_implement_task_execution_store() -> None:
    owners = (
        (OPERATION_STORE, "InMemoryOperationStore"),
        (SQLITE_ADAPTER, "SQLiteOperationStore"),
    )

    for path, class_name in owners:
        definition = _class_definition(_required_tree(path), class_name)
        methods = _public_methods(definition)
        assert (
            TASK_EXECUTION_METHODS <= methods.keys()
        ), f"{class_name} must concretely own every TaskExecutionStore method"
        assert all(
            isinstance(methods[method_name], ast.AsyncFunctionDef)
            for method_name in TASK_EXECUTION_METHODS
        )


def test_task_lease_clock_and_duration_configuration_belong_to_adapters() -> None:
    in_memory = _class_definition(
        _required_tree(OPERATION_STORE),
        "InMemoryOperationStore",
    )
    sqlite = _class_definition(
        _required_tree(SQLITE_ADAPTER),
        "SQLiteOperationStore",
    )

    in_memory_constructors = [
        node
        for node in in_memory.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    sqlite_constructors = [
        node
        for node in sqlite.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    sqlite_openers = [
        node
        for node in sqlite.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "open"
    ]
    assert len(in_memory_constructors) == 1
    assert len(sqlite_constructors) == 1
    assert len(sqlite_openers) == 1

    for owner, function in (
        ("InMemoryOperationStore.__init__", in_memory_constructors[0]),
        ("SQLiteOperationStore.__init__", sqlite_constructors[0]),
        ("SQLiteOperationStore.open", sqlite_openers[0]),
    ):
        arguments = _function_arguments(function)
        assert "clock" in arguments, f"{owner} must accept its authoritative clock"
        duration_arguments = {
            argument
            for argument in arguments
            if "lease" in argument and "duration" in argument
        }
        assert (
            len(duration_arguments) == 1
        ), f"{owner} must own one bounded lease-duration setting"

    for owner, constructor in (
        ("InMemoryOperationStore", in_memory_constructors[0]),
        ("SQLiteOperationStore", sqlite_constructors[0]),
    ):
        attributes = _assigned_self_attributes(constructor)
        assert "_clock" in attributes, f"{owner} must retain its injected clock"
        duration_attributes = {
            attribute
            for attribute in attributes
            if "lease" in attribute and "duration" in attribute
        }
        assert (
            len(duration_attributes) == 1
        ), f"{owner} must retain one bounded lease-duration setting"


def test_task_lifecycle_sql_remains_exclusive_to_the_sqlite_adapter() -> None:
    task_lifecycle_sql = re.compile(
        r"\b(?:CREATE\s+(?:TABLE|INDEX|TRIGGER)|ALTER\s+TABLE|SELECT|"
        r"INSERT\s+INTO|UPDATE|DELETE\s+FROM)\b[\s\S]{0,700}"
        r"\b(?:tasks|task_dependencies|task_leases)\b",
        re.IGNORECASE,
    )
    owners: set[str] = set()

    for path, tree in _production_trees():
        docstrings = _docstring_constants(tree)
        if any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node not in docstrings
            and task_lifecycle_sql.search(node.value) is not None
            for node in ast.walk(tree)
        ):
            owners.add(_relative_path(path))

    assert owners == {"storage/sqlite.py"}


def test_task_lifecycle_adds_no_parallel_store_runtime_or_lease_owner() -> None:
    method_owners: list[tuple[str, str]] = []
    suspicious_owner_names: list[tuple[str, str]] = []
    forbidden_owner_suffixes = (
        "LeaseManager",
        "LeaseRepository",
        "LeaseRuntime",
        "RecoveryRuntime",
        "TaskRuntime",
        "ExecutionKernel",
    )

    for path, tree in _production_trees():
        relative_path = _relative_path(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if methods & TASK_EXECUTION_METHODS:
                method_owners.append((relative_path, node.name))
            if node.name.endswith(forbidden_owner_suffixes):
                suspicious_owner_names.append((relative_path, node.name))

    assert method_owners == [
        ("operations/store.py", "TaskExecutionStore"),
        ("operations/store.py", "InMemoryOperationStore"),
        ("storage/sqlite.py", "SQLiteOperationStore"),
    ]
    assert suspicious_owner_names == []

    assert _class_definition(
        _required_tree(OPERATION_RUNTIME),
        "OperationRuntime",
    )
    assert _class_definition(_required_tree(OPERATION_LEASES), "TaskLease")


def test_operation_runtime_remains_the_sole_executor_invocation_boundary() -> None:
    callers: list[tuple[str, str]] = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and _attribute_parts(node.func) == ("executor", "execute")
            ):
                callers.append((_relative_path(path), ast.unparse(node.func)))

    assert callers == [("operations/runtime.py", "executor.execute")]


def test_operation_runtime_consumes_the_existing_fenced_store_contract() -> None:
    runtime_tree = _required_tree(OPERATION_RUNTIME)
    runtime = _class_definition(runtime_tree, "OperationRuntime")
    state = _class_definition(runtime_tree, "_OperationState")
    request = _class_definition(_required_tree(CAPABILITIES), "ExecutionRequest")

    state_fields = {
        node.target.id
        for node in state.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert {"task_dependencies", "task_leases"} <= state_fields

    request_fields = {
        node.target.id
        for node in request.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert {
        "executor_id",
        "attempt",
        "fencing_token",
        "idempotency_key",
    } <= request_fields

    constructors = [
        node
        for node in runtime.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    assert len(constructors) == 1
    store_argument = next(
        argument
        for argument in constructors[0].args.kwonlyargs
        if argument.arg == "store"
    )
    assert store_argument.annotation is not None
    assert "TaskExecutionStore" in ast.unparse(store_argument.annotation)

    submit = next(
        node
        for node in runtime.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "submit"
    )
    store_calls = {
        _attribute_parts(node.func)
        for node in ast.walk(submit)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert ("self", "_store", "claim_task") in store_calls
    runtime_calls = {
        _attribute_parts(node.func)
        for node in ast.walk(runtime)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert ("self", "_store", "commit_fenced") in runtime_calls
