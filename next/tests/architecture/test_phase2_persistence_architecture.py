from __future__ import annotations

import ast
from pathlib import Path
import re

NEXT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = NEXT_ROOT / "src" / "daita"
CAPABILITIES = SOURCE_ROOT / "capabilities.py"
EVENT_MODELS = SOURCE_ROOT / "events" / "models.py"
OPERATION_MODELS = SOURCE_ROOT / "operations" / "models.py"
OPERATION_CHECKPOINTS = SOURCE_ROOT / "operations" / "checkpoints.py"
OPERATION_STORE = SOURCE_ROOT / "operations" / "store.py"
OPERATION_RUNTIME = SOURCE_ROOT / "operations" / "runtime.py"
LOOP_DRIVER = SOURCE_ROOT / "loop" / "driver.py"
SQLITE_ADAPTER = SOURCE_ROOT / "storage" / "sqlite.py"
BLOB_STORE = SOURCE_ROOT / "storage" / "blobs.py"


def _production_trees() -> tuple[tuple[Path, ast.Module], ...]:
    return tuple(
        (
            path,
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
        )
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
    )


def _required_tree(path: Path) -> ast.Module:
    assert path.is_file(), f"missing required Phase 2 module: {path}"
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


def _imported_modules(tree: ast.Module) -> tuple[tuple[int, str], ...]:
    modules: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append((node.lineno, node.module))
    return tuple(modules)


def _is_module_or_child(module: str, parent: str) -> bool:
    return module == parent or module.startswith(f"{parent}.")


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


def _is_executor_invocation(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _attribute_parts(node.func) == ("executor", "execute")
    )


def test_persistence_records_have_canonical_nonruntime_owners() -> None:
    assert _class_locations(
        "RuntimeEvent",
        "ModelCallStatus",
        "ModelCall",
        "OperationSnapshot",
    ) == {
        "RuntimeEvent": ["events/models.py"],
        "ModelCallStatus": ["operations/checkpoints.py"],
        "ModelCall": ["operations/checkpoints.py"],
        "OperationSnapshot": ["operations/checkpoints.py"],
    }


def test_runtime_reexports_records_from_their_canonical_owners() -> None:
    tree = _required_tree(OPERATION_RUNTIME)
    imports = {
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert {
        ("events.models", "RuntimeEvent"),
        ("checkpoints", "ModelCallStatus"),
        ("checkpoints", "ModelCall"),
        ("checkpoints", "OperationSnapshot"),
    } <= imports


def test_canonical_records_do_not_depend_on_runtime_or_storage() -> None:
    forbidden_parents = {
        "operations.runtime",
        "operations.store",
        "runtime",
        "store",
        "loop.driver",
        "storage",
        "sqlite3",
        "aiosqlite",
        "sqlalchemy",
    }
    violations: list[tuple[str, int, str]] = []

    for path in (EVENT_MODELS, OPERATION_CHECKPOINTS):
        tree = _required_tree(path)
        for lineno, module in _imported_modules(tree):
            if any(_is_module_or_child(module, parent) for parent in forbidden_parents):
                violations.append((_relative_path(path), lineno, module))

    assert violations == []


def test_operation_store_is_a_narrow_async_optimistic_contract() -> None:
    tree = _required_tree(OPERATION_STORE)
    operation_store = _class_definition(tree, "OperationStore")
    public_methods = {
        node.name: node
        for node in operation_store.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }

    assert set(public_methods) == {
        "create",
        "load",
        "load_nonterminal",
        "load_by_trigger",
        "load_by_approval",
        "commit",
    }
    assert all(
        isinstance(method, ast.AsyncFunctionDef) for method in public_methods.values()
    )
    nonterminal = public_methods["load_nonterminal"]
    assert isinstance(nonterminal, ast.AsyncFunctionDef)
    assert "agent_id" in {
        argument.arg
        for argument in (*nonterminal.args.args, *nonterminal.args.kwonlyargs)
    }
    commit = public_methods["commit"]
    assert isinstance(commit, ast.AsyncFunctionDef)
    assert "expected_revision" in {argument.arg for argument in commit.args.kwonlyargs}

    for path, class_name in (
        (OPERATION_STORE, "InMemoryOperationStore"),
        (SQLITE_ADAPTER, "SQLiteOperationStore"),
    ):
        concrete = _class_definition(_required_tree(path), class_name)
        methods = {
            node.name: node
            for node in concrete.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        }
        assert isinstance(methods.get("load_nonterminal"), ast.AsyncFunctionDef)

    assert _class_locations(
        "OperationStore",
        "InMemoryOperationStore",
        "VersionedOperation",
        "CommitResult",
        "OperationRevisionConflict",
        "InvalidOperationCheckpointError",
        "StateStore",
    ) == {
        "OperationStore": ["operations/store.py"],
        "InMemoryOperationStore": ["operations/store.py"],
        "VersionedOperation": ["operations/store.py"],
        "CommitResult": ["operations/store.py"],
        "OperationRevisionConflict": ["operations/store.py"],
        "InvalidOperationCheckpointError": ["operations/store.py"],
        "StateStore": [],
    }


def test_store_contract_does_not_import_runtime_execution_or_sql_owners() -> None:
    tree = _required_tree(OPERATION_STORE)
    forbidden_parents = {
        "capabilities",
        "loop.driver",
        "operations.runtime",
        "runtime",
        "storage",
        "sqlite3",
        "aiosqlite",
        "sqlalchemy",
        "asyncpg",
        "psycopg",
        "psycopg2",
    }
    violations = [
        (lineno, module)
        for lineno, module in _imported_modules(tree)
        if any(_is_module_or_child(module, parent) for parent in forbidden_parents)
    ]

    assert violations == []


def test_sqlite_adapter_imports_only_canonical_records_and_standard_library() -> None:
    tree = _required_tree(SQLITE_ADAPTER)
    imported_modules = {module for _, module in _imported_modules(tree)}

    assert imported_modules == {
        "__future__",
        "_json",
        "asyncio",
        "capabilities",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "events.models",
        "events.protocols",
        "hashlib",
        "identity",
        "json",
        "llm.models",
        "loop.models",
        "operations.checkpoints",
        "operations.governance",
        "operations.leases",
        "operations.models",
        "operations.store",
        "pathlib",
        "re",
        "sessions",
        "sqlite3",
        "typing",
    }


def test_blob_store_has_one_canonical_owner_and_no_generic_state_store() -> None:
    assert _class_locations("BlobStore", "StateStore") == {
        "BlobStore": ["storage/blobs.py"],
        "StateStore": [],
    }


def test_runtime_links_blob_evidence_through_portable_contracts_only() -> None:
    assert _class_locations("EvidenceArtifact", "Evidence") == {
        "EvidenceArtifact": ["capabilities.py"],
        "Evidence": ["operations/models.py"],
    }
    runtime = _class_definition(
        _required_tree(OPERATION_RUNTIME),
        "OperationRuntime",
    )
    constructor = next(
        node
        for node in runtime.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    blob_store_argument = next(
        argument
        for argument in constructor.args.kwonlyargs
        if argument.arg == "blob_store"
    )
    assert blob_store_argument.annotation is not None
    assert "BlobStore" in ast.unparse(blob_store_argument.annotation)

    calls = [
        _attribute_parts(node.func)
        for node in ast.walk(runtime)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls.count(("self", "_blob_store", "put")) == 1
    assert not any("LocalBlobStore" in ast.unparse(node) for node in ast.walk(runtime))

    for path in (CAPABILITIES, OPERATION_MODELS):
        assert all(
            not _is_module_or_child(module, "storage")
            for _, module in _imported_modules(_required_tree(path))
        )


def test_blob_store_does_not_import_runtime_execution_sql_or_provider_owners() -> None:
    tree = _required_tree(BLOB_STORE)
    forbidden_parents = {
        "aiosqlite",
        "anthropic",
        "asyncpg",
        "boto3",
        "botocore",
        "capabilities",
        "daita",
        "google.genai",
        "groq",
        "loop.driver",
        "openai",
        "operations.runtime",
        "psycopg",
        "psycopg2",
        "sqlalchemy",
        "sqlite3",
        "storage.sqlite",
        "xai",
    }
    violations = [
        (lineno, module)
        for lineno, module in _imported_modules(tree)
        if any(_is_module_or_child(module, parent) for parent in forbidden_parents)
    ]

    assert violations == []


def test_sqlite_adapter_has_no_opaque_snapshot_or_history_rewrite_sql() -> None:
    tree = _required_tree(SQLITE_ADAPTER)
    forbidden = re.compile(
        r"\bsnapshot_json\b|\bdelete\s+from\b|\breplace\s+into\b|"
        r"\binsert\s+or\s+(?:replace|ignore|abort|fail|rollback)\b|"
        r"\bon\s+conflict\b|\bupsert\b",
        re.IGNORECASE,
    )
    samples = (
        "snapshot_json TEXT NOT NULL",
        "DELETE FROM runtime_events",
        "REPLACE INTO operations VALUES (?)",
        "INSERT OR IGNORE INTO evidence VALUES (?)",
        "ON CONFLICT(id) DO UPDATE SET revision = 2",
        "UPSERT operation",
    )
    assert all(forbidden.search(sample) is not None for sample in samples)

    violations = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and forbidden.search(node.value) is not None
    ]

    assert violations == []


def test_loop_and_runtime_do_not_own_sql_or_import_storage_adapters() -> None:
    sql_modules = {
        "storage",
        "sqlite3",
        "aiosqlite",
        "sqlalchemy",
        "asyncpg",
        "psycopg",
        "psycopg2",
    }
    forbidden_loop_modules = {
        "events.store",
        "operations.store",
        *sql_modules,
    }
    forbidden_runtime_modules = {
        "storage.sqlite",
        "operations.sqlite",
        "sqlite",
        "sqlite3",
        "aiosqlite",
        "sqlalchemy",
        "asyncpg",
        "psycopg",
        "psycopg2",
    }
    runtime_tree = _required_tree(OPERATION_RUNTIME)
    runtime_storage_imports = [
        node
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and _is_module_or_child(node.module, "storage")
    ]
    assert len(runtime_storage_imports) == 1
    assert runtime_storage_imports[0].module == "storage.blobs"
    assert {alias.name for alias in runtime_storage_imports[0].names} == {
        "BlobMetadata",
        "BlobPut",
        "BlobStore",
    }
    sql_leader = re.compile(
        r"^(?:PRAGMA\b|CREATE\s+(?:TABLE|INDEX|TRIGGER|VIEW)\b|"
        r"ALTER\s+TABLE\b|DROP\s+(?:TABLE|INDEX|TRIGGER|VIEW)\b|SELECT\b|"
        r"INSERT\s+INTO\b|UPDATE\s+\S+\s+SET\b|DELETE\s+FROM\b|WITH\b|"
        r"VACUUM\b|ATTACH\b|DETACH\b|"
        r"BEGIN(?:\s+(?:IMMEDIATE|EXCLUSIVE|TRANSACTION))?\s*;?$|"
        r"COMMIT\s*;?$|ROLLBACK\s*;?$)",
        re.IGNORECASE,
    )
    violations: list[tuple[str, int, str]] = []

    for path, forbidden_modules in (
        (LOOP_DRIVER, forbidden_loop_modules),
        (OPERATION_RUNTIME, forbidden_runtime_modules),
    ):
        tree = _required_tree(path)
        docstrings = _docstring_constants(tree)
        relative_path = _relative_path(path)
        for lineno, module in _imported_modules(tree):
            if any(_is_module_or_child(module, parent) for parent in forbidden_modules):
                violations.append((relative_path, lineno, f"import {module}"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node not in docstrings
                and sql_leader.match(node.value.strip())
            ):
                continue
            violations.append(
                (relative_path, node.lineno, f"SQL literal {node.value[:32]!r}")
            )

    assert violations == []


def test_production_has_one_loop_one_runtime_and_one_executor_boundary() -> None:
    agent_loop_classes: list[tuple[str, str]] = []
    operation_runtime_classes: list[tuple[str, str]] = []
    executor_callers: list[tuple[str, str]] = []

    synthetic_executor_call = ast.parse("executor.execute(request)", mode="eval").body
    synthetic_sql_call = ast.parse("connection.execute(query)", mode="eval").body
    assert _is_executor_invocation(synthetic_executor_call)
    assert not _is_executor_invocation(synthetic_sql_call)

    for path, tree in _production_trees():
        relative_path = _relative_path(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = {ast.unparse(base).split(".")[-1] for base in node.bases}
                if node.name.endswith("AgentLoop") or "AgentLoop" in base_names:
                    agent_loop_classes.append((relative_path, node.name))
                if (
                    node.name.endswith("OperationRuntime")
                    or "OperationRuntime" in base_names
                ):
                    operation_runtime_classes.append((relative_path, node.name))
            elif _is_executor_invocation(node):
                assert isinstance(node, ast.Call)
                executor_callers.append((relative_path, ast.unparse(node.func)))

    assert agent_loop_classes == [("loop/driver.py", "AgentLoop")]
    assert operation_runtime_classes == [("operations/runtime.py", "OperationRuntime")]
    assert executor_callers == [("operations/runtime.py", "executor.execute")]


def test_runtime_uses_the_injected_store_as_authoritative_state() -> None:
    tree = _required_tree(OPERATION_RUNTIME)
    operation_runtime = _class_definition(tree, "OperationRuntime")
    constructors = [
        node
        for node in operation_runtime.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    assert len(constructors) == 1
    constructor = constructors[0]
    constructor_arguments = {
        argument.arg
        for argument in (
            *constructor.args.posonlyargs,
            *constructor.args.args,
            *constructor.args.kwonlyargs,
        )
    }
    assert "store" in constructor_arguments

    assigned_attributes = {
        _attribute_parts(target)
        for node in ast.walk(constructor)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
    }
    assert ("self", "_store") in assigned_attributes

    forbidden_state_attributes: list[tuple[int, str]] = []
    store_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts = _attribute_parts(node)
            if parts in {
                ("self", "_states"),
                ("self", "_operation_by_trigger"),
            }:
                forbidden_state_attributes.append((node.lineno, ast.unparse(node)))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _attribute_parts(node.func)[:2] == ("self", "_store")
        ):
            store_calls.add(node.func.attr)

    assert forbidden_state_attributes == []
    assert {"create", "load", "load_nonterminal", "commit"} <= store_calls
