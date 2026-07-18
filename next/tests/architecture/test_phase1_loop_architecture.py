from __future__ import annotations

import ast
from pathlib import Path

NEXT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = NEXT_ROOT / "src" / "daita"
LOOP_DRIVER = SOURCE_ROOT / "loop" / "driver.py"
LOOP_MODELS = SOURCE_ROOT / "loop" / "models.py"


def _production_trees() -> tuple[tuple[Path, ast.Module], ...]:
    return tuple(
        (
            path,
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
        )
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
    )


def _relative_package_import(node: ast.ImportFrom) -> str | None:
    if node.level == 0 or node.module is None:
        return None
    return node.module


def _attribute_parts(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _identity_branch_references(condition: ast.AST) -> list[str]:
    identity_names = {
        "capability_id",
        "domain_id",
        "executor_id",
        "provider_id",
        "tool_name",
    }
    component_roots = {("self", "_domain"), ("self", "_model")}
    component_identity_fields = {"__class__", "domain_id", "id", "kind", "name"}
    violations: list[str] = []

    direct_candidates = [condition]
    if isinstance(condition, ast.Compare):
        direct_candidates.extend((condition.left, *condition.comparators))
    elif isinstance(condition, ast.UnaryOp):
        direct_candidates.append(condition.operand)
    elif isinstance(condition, ast.BoolOp):
        direct_candidates.extend(condition.values)
    violations.extend(
        ast.unparse(candidate)
        for candidate in direct_candidates
        if _attribute_parts(candidate) in component_roots
    )

    for node in ast.walk(condition):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"isinstance", "type"} and any(
                _attribute_parts(argument)[:2] in component_roots
                for argument in node.args
            ):
                violations.append(ast.unparse(node))
        if isinstance(node, ast.Attribute):
            parts = _attribute_parts(node)
            if parts[:2] in component_roots and any(
                part in component_identity_fields for part in parts[2:]
            ):
                violations.append(ast.unparse(node))
            if node.attr in identity_names:
                violations.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in identity_names:
            violations.append(node.id)

    return violations


def test_operation_runtime_is_the_only_executor_invocation_boundary() -> None:
    callers: list[tuple[str, str]] = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and _attribute_parts(node.func) == ("executor", "execute")
            ):
                continue
            callers.append(
                (
                    path.relative_to(SOURCE_ROOT).as_posix(),
                    ast.unparse(node.func),
                )
            )

    assert callers == [("operations/runtime.py", "executor.execute")]


def test_production_contains_one_generic_agent_loop() -> None:
    agent_loop_classes: list[tuple[str, str]] = []
    model_generation_callers: list[tuple[str, str]] = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = {ast.unparse(base).split(".")[-1] for base in node.bases}
                if node.name.endswith("AgentLoop") or "AgentLoop" in base_names:
                    agent_loop_classes.append(
                        (path.relative_to(SOURCE_ROOT).as_posix(), node.name)
                    )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
            ):
                model_generation_callers.append(
                    (
                        path.relative_to(SOURCE_ROOT).as_posix(),
                        ast.unparse(node.func),
                    )
                )

    assert agent_loop_classes == [("loop/driver.py", "AgentLoop")]
    assert model_generation_callers == [("loop/driver.py", "self._model.generate")]


def test_operations_import_checkpoint_contracts_not_loop_implementation() -> None:
    loop_imports: list[tuple[str, str]] = []
    for path, tree in _production_trees():
        relative_path = path.relative_to(SOURCE_ROOT)
        if relative_path.parts[0] != "operations":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = _relative_package_import(node)
            if module is not None and (module == "loop" or module.startswith("loop.")):
                loop_imports.append((relative_path.as_posix(), module))

    assert loop_imports == [
        ("operations/checkpoints.py", "loop.models"),
        ("operations/runtime.py", "loop.models"),
    ]


def test_loop_checkpoint_contracts_are_an_implementation_free_leaf() -> None:
    tree = ast.parse(
        LOOP_MODELS.read_text(encoding="utf-8"),
        filename=str(LOOP_MODELS),
    )
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert imported_modules == {
        "__future__",
        "_json",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "math",
    }


def test_generic_loop_imports_contracts_not_domain_or_provider_implementations() -> (
    None
):
    tree = ast.parse(
        LOOP_DRIVER.read_text(encoding="utf-8"),
        filename=str(LOOP_DRIVER),
    )
    allowed_modules = {
        "__future__",
        "asyncio",
        "collections.abc",
        "llm.models",
        "llm.protocols",
        "models",
        "operations.checkpoints",
        "operations.models",
        "operations.runtime",
        "typing",
    }
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        modules: tuple[str, ...]
        if isinstance(node, ast.Import):
            modules = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules = (node.module,)
        else:
            continue
        for module in modules:
            if module not in allowed_modules:
                violations.append((node.lineno, module))

    assert violations == []


def test_generic_loop_never_branches_on_provider_or_domain_identity() -> None:
    tree = ast.parse(
        LOOP_DRIVER.read_text(encoding="utf-8"),
        filename=str(LOOP_DRIVER),
    )
    forbidden_examples = (
        "isinstance(self._domain, SqlDomain)",
        "type(self._model).__name__ == 'OpenAIProvider'",
        "self._model.provider_id == 'openai'",
    )
    assert all(
        _identity_branch_references(ast.parse(source, mode="eval").body)
        for source in forbidden_examples
    )
    allowed_contract_branch = ast.parse(
        "self._domain.evaluate_final_answer()",
        mode="eval",
    ).body
    assert _identity_branch_references(allowed_contract_branch) == []

    conditions: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            conditions.append(node.test)
        elif isinstance(node, ast.Match):
            conditions.append(node.subject)
        elif isinstance(node, ast.comprehension):
            conditions.extend(node.ifs)

    violations: list[tuple[int, str]] = []
    for condition in conditions:
        if (
            isinstance(condition, ast.Compare)
            and len(condition.ops) == 1
            and isinstance(condition.ops[0], (ast.Eq, ast.NotEq))
            and len(condition.comparators) == 1
        ):
            compared_parts = {
                _attribute_parts(condition.left),
                _attribute_parts(condition.comparators[0]),
            }
            if compared_parts == {
                ("model_call", "provider_id"),
                ("self", "_model", "provider_id"),
            }:
                continue
        violations.extend(
            (getattr(condition, "lineno", -1), reference)
            for reference in _identity_branch_references(condition)
        )

    assert violations == []
