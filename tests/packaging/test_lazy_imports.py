"""Fresh-process coverage for package and terminal integration imports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script, *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_importing_terminal_directly_keeps_textual_and_rich_lazy() -> None:
    script = r"""
import builtins
import sys

blocked = {"rich", "textual"}
original = builtins.__import__

def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] in blocked:
        raise AssertionError(f"eager integration import: {name}")
    return original(name, globals, locals, fromlist, level)

builtins.__import__ = guarded
import daita.terminal

loaded = sorted(name for name in sys.modules if name.split(".")[0] in blocked)
if loaded:
    raise AssertionError(f"eager integration modules loaded: {loaded}")
"""
    completed = _run(script)

    assert completed.returncode == 0, completed.stderr


def test_terminal_import_guard_rejects_an_in_memory_eager_import(
    tmp_path: Path,
) -> None:
    script = r"""
import builtins
from pathlib import Path

source = Path(__import__("sys").argv[1]).read_text(encoding="utf-8")
mutated = source.replace(
    "from __future__ import annotations\n",
    "from __future__ import annotations\nimport textual\n",
    1,
)
original = builtins.__import__

def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] == "textual":
        raise AssertionError(f"eager integration import: {name}")
    return original(name, globals, locals, fromlist, level)

builtins.__import__ = guarded
namespace = {"__name__": "daita._terminal_eager_probe", "__package__": "daita"}
exec(compile(mutated, "terminal-eager-probe.py", "exec"), namespace)
"""
    import daita.terminal

    assert daita.terminal.__file__ is not None
    completed = _run(script, daita.terminal.__file__)

    assert completed.returncode != 0
    assert "eager integration import: textual" in completed.stderr
    assert not tuple(tmp_path.iterdir())


def test_terminal_loader_normalizes_a_missing_textual_dependency() -> None:
    script = r"""
import builtins
import daita.terminal

original = builtins.__import__

def missing_textual(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] == "textual":
        raise ImportError("private missing dependency detail")
    return original(name, globals, locals, fromlist, level)

builtins.__import__ = missing_textual
try:
    daita.terminal._load_textual_app()
except ImportError as error:
    message = str(error)
    assert "pipx reinstall daita-agents" in message
    assert "private missing dependency detail" not in message
else:
    raise AssertionError("missing Textual dependency was accepted")
"""
    completed = _run(script)

    assert completed.returncode == 0, completed.stderr
