from __future__ import annotations

from pathlib import Path

NEXT_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = NEXT_ROOT / ".github" / "workflows" / "candidate-ci.yml"
LIFECYCLE = NEXT_ROOT / "scripts" / "verify_candidate_lifecycle.py"


def test_candidate_ci_declares_both_supported_python_versions() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert 'python-version: "3.11"' in workflow
    assert "not requires_llm and not requires_db" in workflow
    assert "scripts/verify_candidate_lifecycle.py --python python" in workflow


def test_candidate_ci_keeps_broad_static_checks_in_one_job() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("black --check") == 1
    assert workflow.count("python -m mypy") == 1
    assert "--no-verify" not in workflow


def test_candidate_lifecycle_gate_is_stdlib_only_and_network_independent() -> None:
    source = LIFECYCLE.read_text(encoding="utf-8")

    assert '"--no-isolation"' in source
    assert '"--no-deps"' in source
    assert '"pip", "uninstall", "-y", "daita-agents"' in source
    assert "find_spec('daita') is None" in source
    assert "state_retained_after_uninstall" in source
    assert "requests" not in source
    assert "urllib" not in source
