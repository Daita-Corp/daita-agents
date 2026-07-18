from __future__ import annotations

import ast
from pathlib import Path
import re

NEXT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = NEXT_ROOT / "src" / "daita"
OPERATION_MODELS = SOURCE_ROOT / "operations" / "models.py"
OPERATION_LEASES = SOURCE_ROOT / "operations" / "leases.py"
OPERATION_CHECKPOINTS = SOURCE_ROOT / "operations" / "checkpoints.py"
OPERATION_STORE = SOURCE_ROOT / "operations" / "store.py"
OPERATION_RUNTIME = SOURCE_ROOT / "operations" / "runtime.py"
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
    assert path.is_file(), f"missing required task-persistence module: {path}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _relative_path(path: Path) -> str:
    return path.relative_to(SOURCE_ROOT).as_posix()


def _attribute_parts(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _base_name(node: ast.expr) -> str:
    return ast.unparse(node).split("[", maxsplit=1)[0].split(".")[-1]


def _imported_modules(tree: ast.Module) -> tuple[tuple[int, str], ...]:
    modules: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append((node.lineno, node.module))
    return tuple(modules)


def _class_locations(*class_names: str) -> dict[str, list[str]]:
    locations: dict[str, list[str]] = {class_name: [] for class_name in class_names}
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in locations:
                locations[node.name].append(_relative_path(path))
    return locations


def _class_definition(tree: ast.Module, name: str) -> ast.ClassDef:
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(definitions) == 1, f"expected exactly one {name} definition"
    return definitions[0]


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


def _migration_calls(tree: ast.Module) -> tuple[ast.Call, ...]:
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_MIGRATIONS"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, "SQLite must define one fixed _MIGRATIONS plan"
    value = assignments[0].value
    assert isinstance(value, (ast.Tuple, ast.List))
    calls = tuple(element for element in value.elts if isinstance(element, ast.Call))
    assert len(calls) == len(value.elts), "every migration entry must be explicit"
    return calls


def _keyword(call: ast.Call, name: str) -> ast.expr:
    values = [keyword.value for keyword in call.keywords if keyword.arg == name]
    assert len(values) == 1, f"migration must declare exactly one {name}"
    return values[0]


def _literal_assignment(tree: ast.Module, name: str) -> tuple[str, ...]:
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    assert len(values) == 1, f"missing one explicit {name} SQL tuple"
    value = values[0]
    assert isinstance(value, (ast.Tuple, ast.List))
    statements: list[str] = []
    for expression in value.elts:
        if (
            isinstance(expression, ast.Call)
            and not expression.args
            and not expression.keywords
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "strip"
        ):
            statement = ast.literal_eval(expression.func.value).strip()
        else:
            statement = ast.literal_eval(expression)
        assert isinstance(statement, str)
        statements.append(statement)
    return tuple(statements)


def _migration_statements(tree: ast.Module, call: ast.Call) -> tuple[str, ...]:
    expression = _keyword(call, "statements")
    if isinstance(expression, ast.Name):
        return _literal_assignment(tree, expression.id)
    literal = ast.literal_eval(expression)
    assert isinstance(literal, tuple)
    assert all(isinstance(statement, str) for statement in literal)
    return literal


def test_task_execution_records_have_exactly_one_canonical_owner() -> None:
    assert _class_locations("TaskExecutionFacts", "TaskDependency", "TaskLease") == {
        "TaskExecutionFacts": ["operations/models.py"],
        "TaskDependency": ["operations/models.py"],
        "TaskLease": ["operations/leases.py"],
    }


def test_portable_task_records_and_contracts_do_not_import_storage() -> None:
    forbidden = {"storage", "sqlite", "sqlite3", "aiosqlite", "sqlalchemy"}
    violations: list[tuple[str, int, str]] = []

    for path in (
        OPERATION_MODELS,
        OPERATION_LEASES,
        OPERATION_CHECKPOINTS,
        OPERATION_STORE,
    ):
        for lineno, module in _imported_modules(_required_tree(path)):
            root = module.split(".", maxsplit=1)[0]
            if root in forbidden or "sqlite" in module.lower():
                violations.append((_relative_path(path), lineno, module))

    assert violations == []


def test_only_the_sqlite_adapter_owns_task_projection_sql() -> None:
    task_sql = re.compile(
        r"\b(?:CREATE\s+(?:TABLE|INDEX)|ALTER\s+TABLE|INSERT\s+INTO|"
        r"UPDATE|DELETE\s+FROM|SELECT)\b[\s\S]{0,500}"
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
            and task_sql.search(node.value) is not None
            for node in ast.walk(tree)
        ):
            owners.add(_relative_path(path))

    assert owners == {"storage/sqlite.py"}


def test_migration_four_normalizes_task_dependencies_and_lease_history() -> None:
    tree = _required_tree(SQLITE_ADAPTER)
    migrations = _migration_calls(tree)
    versions = tuple(ast.literal_eval(_keyword(call, "version")) for call in migrations)

    assert versions[:4] == (1, 2, 3, 4)
    migration_four = migrations[versions.index(4)]
    statements = _migration_statements(tree, migration_four)
    dependency_tables = [
        statement
        for statement in statements
        if re.search(
            r"\bCREATE\s+TABLE\s+task_dependencies\b",
            statement,
            re.IGNORECASE,
        )
    ]
    lease_tables = [
        statement
        for statement in statements
        if re.search(
            r"\bCREATE\s+TABLE\s+task_leases\b",
            statement,
            re.IGNORECASE,
        )
    ]

    assert len(dependency_tables) == 1
    assert len(lease_tables) == 1

    dependency_sql = dependency_tables[0].lower()
    lease_sql = lease_tables[0].lower()
    assert "json" not in dependency_sql
    assert "json" not in lease_sql
    dependency_tokens = {
        "operation_id",
        "task_id",
        "prerequisite_task_id",
        "primary key",
        "foreign key",
    }
    lease_tokens = {
        "operation_id",
        "task_id",
        "attempt",
        "fencing_token",
        "holder_id",
        "acquired_at",
        "expires_at",
        "started_at",
        "renewed_at",
        "released_at",
        "release_reason",
        "foreign key",
    }
    assert all(token in dependency_sql for token in dependency_tokens)
    assert all(token in lease_sql for token in lease_tokens)


def test_task_execution_store_is_the_only_narrow_task_persistence_protocol() -> None:
    tree = _required_tree(OPERATION_STORE)
    task_store = _class_definition(tree, "TaskExecutionStore")
    base_names = {_base_name(base) for base in task_store.bases}
    public_methods = {
        node.name
        for node in task_store.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }

    assert {"OperationStore", "Protocol"} <= base_names
    assert public_methods == TASK_EXECUTION_METHODS
    assert all(
        isinstance(node, ast.AsyncFunctionDef)
        for node in task_store.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in TASK_EXECUTION_METHODS
    )

    protocol_owners: list[tuple[str, str]] = []
    for path, production_tree in _production_trees():
        for node in ast.walk(production_tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            bases = {_base_name(base) for base in node.bases}
            if methods & TASK_EXECUTION_METHODS and "Protocol" in bases:
                protocol_owners.append((_relative_path(path), node.name))

    assert protocol_owners == [("operations/store.py", "TaskExecutionStore")]


def test_task_persistence_adds_no_parallel_store_or_runtime_owner() -> None:
    assert _class_locations(
        "TaskExecutionStore",
        "TaskLeaseStore",
        "StateStore",
        "OperationRuntime",
    ) == {
        "TaskExecutionStore": ["operations/store.py"],
        "TaskLeaseStore": [],
        "StateStore": [],
        "OperationRuntime": ["operations/runtime.py"],
    }


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
