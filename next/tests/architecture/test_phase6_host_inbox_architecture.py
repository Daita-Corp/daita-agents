from __future__ import annotations

import ast
from pathlib import Path

INBOX = Path(__file__).parents[2] / "src" / "daita" / "hosting" / "inbox.py"


def _tree() -> ast.Module:
    return ast.parse(INBOX.read_text(encoding="utf-8"), filename=str(INBOX))


def test_host_inbox_contract_has_no_execution_or_sqlite_dependency() -> None:
    forbidden = {
        "llm",
        "loop",
        "monitors.scheduler",
        "operations.runtime",
        "storage",
        "sqlite3",
    }
    imported: list[str] = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module.lstrip("."))

    assert not {
        module
        for module in imported
        if any(
            module == parent or module.startswith(parent + ".") for parent in forbidden
        )
    }


def test_host_inbox_is_a_narrow_single_writer_contract() -> None:
    protocol = next(
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == "HostInboxStore"
    )
    methods = {
        node.name for node in protocol.body if isinstance(node, ast.AsyncFunctionDef)
    }

    assert methods == {
        "admit_host_mutation",
        "enqueue_host_inbox",
        "list_pending_host_inbox",
        "complete_host_inbox",
    }
    source = INBOX.read_text(encoding="utf-8").lower()
    assert "lease" not in source
    assert "fenc" not in source
    assert "create_task" not in source
    assert "ensure_future" not in source
