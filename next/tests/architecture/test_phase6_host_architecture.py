from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[2] / "src" / "daita"
HOSTING = SOURCE / "hosting"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> tuple[str, ...]:
    values: list[str] = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            values.append(node.module.lstrip("."))
    return tuple(values)


def test_transport_contract_has_no_runtime_or_adapter_owner() -> None:
    imports = _imports(HOSTING / "local_protocol.py")
    forbidden = ("adapters", "llm", "loop", "monitors", "operations", "storage")

    assert not {
        module
        for module in imports
        if any(module == name or module.startswith(name + ".") for name in forbidden)
    }


def test_host_and_socket_dispatch_never_invoke_an_executor() -> None:
    violations: list[tuple[str, int]] = []
    for name in ("host.py", "local_server.py"):
        for node in ast.walk(_tree(HOSTING / name)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
            ):
                violations.append((name, node.lineno))

    assert violations == []
    assert "sqlite3" not in _imports(HOSTING / "local_server.py")


def test_only_explicit_host_start_creates_host_background_tasks() -> None:
    protocol_source = (HOSTING / "local_protocol.py").read_text(encoding="utf-8")
    server_source = (HOSTING / "local_server.py").read_text(encoding="utf-8")
    cli_source = (SOURCE / "cli.py").read_text(encoding="utf-8")

    assert "create_task(" not in protocol_source
    assert "create_task(" not in server_source
    assert "create_task(" not in cli_source
    assert "asyncio.create_task(" in (HOSTING / "host.py").read_text(encoding="utf-8")


def test_every_socket_mutation_requires_an_idempotency_key() -> None:
    source_path = HOSTING / "local_server.py"
    tree = _tree(source_path)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_MUTATING_METHODS"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    assert len(assignment.value.args) == 1
    methods = frozenset(ast.literal_eval(assignment.value.args[0]))
    assert methods == frozenset(
        (
            "chat.submit",
            "source.attach",
            "operation.cancel",
            "approval.approve",
            "approval.reject",
            "monitor.propose",
            "monitor.confirm",
            "monitor.pause",
            "monitor.resume",
            "monitor.run_now",
            "monitor.delete",
        )
    )
    assert "idempotency_required" in source_path.read_text(encoding="utf-8")
