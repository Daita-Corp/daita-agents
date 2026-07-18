from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "daita"


def test_public_agent_is_a_thin_embedded_facade() -> None:
    tree = ast.parse((SOURCE / "agent.py").read_text(encoding="utf-8"))
    agent = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Agent"
    )
    methods = {
        node.name: node
        for node in agent.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert {"create", "open", "run", "inspect", "resume"} <= methods.keys()
    forbidden_calls = {"generate", "execute", "evaluate", "sqlite3.connect"}
    calls = {
        ast.unparse(node.func) for node in ast.walk(agent) if isinstance(node, ast.Call)
    }
    assert not any(call.split(".")[-1] in forbidden_calls for call in calls)


def test_embedded_composition_does_not_add_a_loop_or_executor_boundary() -> None:
    loop_classes: list[str] = []
    executor_calls: list[str] = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("AgentLoop"):
                loop_classes.append(path.relative_to(SOURCE).as_posix())
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "executor"
                and node.func.attr == "execute"
            ):
                executor_calls.append(path.relative_to(SOURCE).as_posix())
    assert loop_classes == ["loop/driver.py"]
    assert executor_calls == ["operations/runtime.py"]
