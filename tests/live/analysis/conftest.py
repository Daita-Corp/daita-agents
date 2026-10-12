"""Opt-in native E2E evidence, with finite paid-model admission."""

from __future__ import annotations

import json
import os
import platform
import signal
import time
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from daita._json import canonical_json

_STARTED = time.monotonic()
_RESERVED = Decimal(0)


def positive_decimal(name: str, default: str) -> Decimal:
    value = Decimal(os.environ.get(name, default))
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@pytest.fixture(autouse=True)
def analysis_live_gate(request):
    if os.environ.get("DAITA_RUN_LIVE_ANALYSIS") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_ANALYSIS=1 for authorized native E2E execution")
    request.node._analysis_evidence = {
        "nodeid": request.node.nodeid,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "started_monotonic": time.monotonic(),
        "measurements": None,
        "cleanup": None,
    }
    deadline = float(os.environ.get("DAITA_ANALYSIS_LIVE_DEADLINE_SECONDS", "900"))
    if not 0 < deadline < float("inf") or time.monotonic() - _STARTED >= deadline:
        pytest.fail("The finite whole-suite analysis deadline is exhausted")
    prior_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("The finite whole-suite analysis deadline is exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(
        signal.ITIMER_REAL, max(0.001, deadline - (time.monotonic() - _STARTED))
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)


@pytest.fixture
def analysis_report(request):
    return request.node._analysis_evidence


@pytest.fixture
def paid_case():
    global _RESERVED
    from dotenv import load_dotenv

    from daita.llm.factory import create_llm_provider
    from daita.llm.profiles import reviewed_model_profile

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    per_run = positive_decimal("DAITA_ANALYSIS_LIVE_MAX_COST_USD", "0.50")
    aggregate = positive_decimal("DAITA_ANALYSIS_LIVE_SUITE_MAX_COST_USD", "5.00")
    if _RESERVED + per_run > aggregate:
        pytest.fail("The finite aggregate model-cost ceiling is exhausted")
    _RESERVED += per_run
    model_id = os.environ.get("DAITA_ANALYSIS_LIVE_MODEL_ID", "openai:gpt-5.6-terra")
    profile = reviewed_model_profile(model_id)
    key_name = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "grok": "XAI_API_KEY",
    }.get(model_id.partition(":")[0])
    key = os.environ.get("DAITA_ANALYSIS_LIVE_LLM_API_KEY") or (
        os.environ.get(key_name) if key_name else None
    )
    if profile is None or not profile.supports_tools or not key:
        pytest.fail(
            "Live analysis needs a reviewed tool-capable model and configured credentials"
        )
    provider = create_llm_provider(
        model_id, api_key=key, max_output_tokens=min(profile.max_output_tokens, 2048)
    )
    return profile, provider, per_run


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()
    if result.when == "call" or (result.when == "setup" and result.failed):
        evidence = getattr(item, "_analysis_evidence", {"nodeid": item.nodeid})
        evidence.update(
            {
                "status": result.outcome,
                "duration_seconds": result.duration,
                "failure": None if result.passed else str(result.longrepr),
            }
        )
        root = Path(
            os.environ.get(
                "DAITA_ANALYSIS_LIVE_REPORT_DIR",
                "/private/tmp/daita-analysis-live-evidence",
            )
        )
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = root / (sha256(item.nodeid.encode()).hexdigest()[:16] + ".json")
        history = root / "history"
        history.mkdir(mode=0o700, exist_ok=True)
        if path.exists():
            (history / f"{path.stem}-{time.time_ns()}.json").write_bytes(
                path.read_bytes()
            )
        path.write_text(canonical_json(evidence) + "\n")
        path.chmod(0o600)
        reports = []
        for report in root.glob("*.json"):
            if report.name != "aggregate.json":
                reports.append(json.loads(report.read_text()))
        (root / "aggregate.json").write_text(
            json.dumps(
                {
                    "cases": [
                        {"nodeid": report["nodeid"], "status": report["status"]}
                        for report in reports
                    ],
                    "paid_reserved_usd": str(_RESERVED),
                },
                indent=2,
            )
            + "\n"
        )
