"""Narrow live-harness fixtures; importing them performs no external I/O."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.support.mcp_routine_harness import model_ids, repeats, report_path


@pytest.fixture(params=model_ids())
def model_id(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=range(repeats()))
def repetition(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture
def evidence_path(
    request: pytest.FixtureRequest,
    record_property: Callable[[str, object], None],
) -> Path:
    path = report_path(request.node.nodeid).resolve()
    record_property("phase_f_evidence", str(path))
    return path
