"""Qualify the candidate through the real managed installer and native worker."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from tests.support.paths import REPO_ROOT


async def test_actual_managed_installation_native_execution_and_lifecycle(
    analysis_report,
):
    wheel = Path(os.environ["DAITA_ANALYSIS_LIVE_WHEEL"]).resolve(strict=True)
    archive = Path(os.environ["DAITA_ANALYSIS_LIVE_UV_ARCHIVE"]).resolve(strict=True)
    runtime = json.loads((REPO_ROOT / "release/managed-installer.json").read_text())[
        "runtime"
    ]
    target = runtime["targets"]["macos-arm64"]
    assert sha256(archive.read_bytes()).hexdigest() == target["uv_sha256"]
    command = [
        sys.executable,
        "-m",
        "tests.packaging.managed_installer_lifecycle_smoke",
        "--candidate-wheel",
        str(wheel),
        "--real-uv-archive",
        str(archive),
        "--real-uv-version",
        runtime["uv_version"],
        "--real-uv-member",
        target["uv_member"],
        "--real-python-request",
        runtime["python_request"],
        "--real-python-identity",
        target["python_identity"],
    ]
    result = await asyncio.to_thread(
        subprocess.run, command, capture_output=True, text=True, timeout=420
    )
    analysis_report.update(
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("ANALYSIS_EVIDENCE=")
    )
    facts = json.loads(line.removeprefix("ANALYSIS_EVIDENCE="))
    analysis_report.update(
        measurements=facts["usage"], cleanup=facts["cleanup"], runtime=facts["runtime"]
    )
    assert facts["cleanup"]["process_reaped"] is True
    assert facts["cleanup"]["scratch_deleted"] is True
