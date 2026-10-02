"""Opt-in read evaluation limits and evidence locations; collection does no I/O."""

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest

from tests.support.mcp_read_harness import (
    COST_ENV,
    FAILURE_LIMIT_ENV,
    REPORT_ENV,
    TOTAL_COST_ENV,
    ReadBudget,
    positive_cost,
)

_READ_BUDGET = pytest.StashKey[ReadBudget]()


@pytest.fixture(scope="session")
def read_budget(request: pytest.FixtureRequest) -> ReadBudget:
    # One aggregate ceiling across both read modules in this pytest process.
    budget = ReadBudget(
        total=positive_cost(TOTAL_COST_ENV, "3.00"),
        per_run=positive_cost(COST_ENV, "0.50"),
        provider_failure_limit=int(os.environ.get(FAILURE_LIMIT_ENV, "2")),
    )
    request.session.stash[_READ_BUDGET] = budget
    return budget


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    result = yield
    # Run normal teardown/reporting before stopping: the failed attempt and
    # its reservation must survive, and every Agent/SDK owner must drain.
    budget = item.session.stash.get(_READ_BUDGET, None)
    if nextitem is not None and budget is not None and budget.stop_reason is not None:
        pytest.exit(
            f"Live MCP read evaluation stopped: {budget.stop_reason}. Completed and failed attempts were retained; remaining cases were not run.",
            returncode=1,
        )
    return result


@pytest.fixture
def read_report_path(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    record_property: Callable[[str, object], None],
) -> Path:
    directory = Path(os.environ.get(REPORT_ENV, str(tmp_path / "evidence")))
    identity = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]
    path = (directory / f"{identity}-{uuid4().hex[:8]}.json").resolve()
    record_property("mcp_read_evidence", str(path))
    return path
