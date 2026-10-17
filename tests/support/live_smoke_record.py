"""The committed live smoke record and its merge gate.

``release/live-smoke.json`` states which model-facing surface digest last
passed the fixed live smoke list on the subscription. It holds no machine,
user, path, account or credential facts. ``scripts/live_smoke.py`` writes it;
``tests/architecture/test_model_surface.py`` enforces it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Collection, Mapping
from datetime import date
from typing import Any

from tests.support.paths import REPO_ROOT

RECORD_PATH = REPO_ROOT / "release" / "live-smoke.json"
REQUIRE_ENV = "DAITA_REQUIRE_LIVE_SMOKE"
RUN_COMMAND = (
    ".venv/bin/python -m tests.support.live_billing login  # once\n"
    "DAITA_LIVE_BILLING=subscription .venv/bin/python scripts/live_smoke.py run"
)
STATUS_PENDING = "pending"
STATUS_RECORDED = "recorded"
CASE_PASSED = "passed"
CASE_PASSED_ON_RERUN = "passed_on_rerun"
_KEYS = frozenset(
    {"status", "surface_digest", "billing", "recorded_on", "cases", "summary"}
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def load_record() -> dict[str, Any] | None:
    try:
        text = RECORD_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("live smoke record must be a JSON object")
    return value


def record_text(record: Mapping[str, object]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def build_record(
    *,
    surface_digest: str,
    cases: Mapping[str, str],
    recorded_on: date,
) -> dict[str, object]:
    record: dict[str, object] = {
        "status": STATUS_RECORDED,
        "surface_digest": surface_digest,
        "billing": "subscription",
        "recorded_on": recorded_on.isoformat(),
        "cases": dict(sorted(cases.items())),
        "summary": {"passed": len(cases), "total": len(cases)},
    }
    problems = record_problems(record, expected_cases=tuple(cases))
    if problems:
        raise ValueError("; ".join(problems))
    return record


def record_problems(
    record: Mapping[str, object], *, expected_cases: Collection[str]
) -> list[str]:
    """Return shape problems; a pending record is well formed but unproven."""

    problems: list[str] = []
    if set(record) != _KEYS:
        problems.append(f"record keys must be exactly {sorted(_KEYS)}")
    status = record.get("status")
    if status not in {STATUS_PENDING, STATUS_RECORDED}:
        problems.append("status must be 'pending' or 'recorded'")
    digest = record.get("surface_digest")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        problems.append("surface_digest must be sha256:<64 hex>")
    if record.get("billing") != "subscription":
        problems.append("billing must be 'subscription'")
    cases = record.get("cases")
    summary = record.get("summary")
    if not isinstance(cases, dict) or not isinstance(summary, dict):
        problems.append("cases and summary must be objects")
        return problems
    if set(summary) != {"passed", "total"}:
        problems.append("summary keys must be passed and total")
    if status == STATUS_PENDING:
        if cases or summary.get("passed") != 0 or summary.get("total") != 0:
            problems.append("a pending record has no cases and a zero summary")
        if record.get("recorded_on") is not None:
            problems.append("a pending record has recorded_on null")
        return problems
    recorded_on = record.get("recorded_on")
    try:
        date.fromisoformat(str(recorded_on))
    except ValueError:
        problems.append("recorded_on must be YYYY-MM-DD")
    if set(cases) != set(expected_cases):
        missing = sorted(set(expected_cases) - set(cases))
        extra = sorted(set(cases) - set(expected_cases))
        problems.append(
            f"cases differ from the smoke list (missing {missing}, extra {extra})"
        )
    if any(
        value not in {CASE_PASSED, CASE_PASSED_ON_RERUN} for value in cases.values()
    ):
        problems.append("every recorded case must be passed or passed_on_rerun")
    if summary.get("passed") != len(cases) or summary.get("total") != len(cases):
        problems.append("summary must count every case as passed")
    return problems


def gate_failure(
    record: Mapping[str, object] | None,
    *,
    surface_digest: str,
    expected_cases: Collection[str],
    require_recorded: bool,
) -> str | None:
    """Explain why merge is blocked, or return None when the gate is satisfied."""

    how = f"Run the live smoke on the subscription:\n{RUN_COMMAND}"
    if record is None:
        return f"{RECORD_PATH.name} is missing. {how}"
    problems = record_problems(record, expected_cases=expected_cases)
    if problems:
        return f"{RECORD_PATH.name} is malformed: {'; '.join(problems)}"
    if record["surface_digest"] != surface_digest:
        return (
            "The model-facing surface changed since the last live smoke run "
            f"(record {record['surface_digest']}, snapshot {surface_digest}). {how}"
        )
    if record["status"] == STATUS_PENDING and require_recorded:
        return (
            f"{RECORD_PATH.name} is pending: no live smoke has passed for this "
            f"surface, and {REQUIRE_ENV}=1 requires one. {how}"
        )
    return None


def require_recorded() -> bool:
    return os.environ.get(REQUIRE_ENV) == "1"


__all__ = [
    "CASE_PASSED",
    "CASE_PASSED_ON_RERUN",
    "RECORD_PATH",
    "REQUIRE_ENV",
    "build_record",
    "gate_failure",
    "load_record",
    "record_problems",
    "record_text",
    "require_recorded",
]
