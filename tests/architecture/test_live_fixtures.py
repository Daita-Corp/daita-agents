"""Every live test's fixtures resolve, without running any live test."""

from __future__ import annotations

import os
import re
import subprocess
import sys

from tests.support.paths import REPO_ROOT

_GATE = re.compile(r"DAITA_RUN_[A-Z0-9_]+")


def _live_gates() -> set[str]:
    gates: set[str] = set()
    for folder in ("tests/live", "tests/support"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            gates.update(_GATE.findall(path.read_text(encoding="utf-8")))
    return gates


def test_every_live_test_resolves_its_fixtures_when_authorized():
    # --setup-plan resolves each fixture graph and reports a missing fixture as
    # an error, but executes no fixture or test body. Opening every gate keeps
    # authorization skips from hiding an unresolved fixture.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DAITA_") and not key.endswith("_API_KEY")
    }
    environment.update({gate: "1" for gate in _live_gates()})
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/live",
            "--setup-plan",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = completed.stdout + completed.stderr
    unresolved = sorted(set(re.findall(r"fixture '([^']+)' not found", output)))
    assert not unresolved, unresolved
    assert completed.returncode == 0, output[-4000:]
