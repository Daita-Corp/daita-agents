"""Protect the release-to-release agent-home compatibility gate."""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.check_home_release_contract import (
    DEFAULT_SNAPSHOT,
    HomeReleaseContractError,
    build_snapshot,
    check_snapshot,
    compare_snapshots,
    make_snapshot,
)
from tests.support.paths import REPO_ROOT


def _changed_snapshot(*, revision_offset: int) -> dict[str, object]:
    baseline = build_snapshot()
    contract = deepcopy(baseline["contract"])
    assert isinstance(contract, dict)
    record_fields = contract["sqlite_record_fields"]
    assert isinstance(record_fields, dict)
    identity_fields = record_fields["AgentIdentity"]
    assert isinstance(identity_fields, list)
    identity_fields.append("synthetic_release_field")
    revision = baseline["home_revision"]
    minimum = baseline["minimum_supported_home_revision"]
    assert isinstance(revision, int)
    assert isinstance(minimum, int)
    return make_snapshot(
        contract,
        home_revision=revision + revision_offset,
        minimum_supported_home_revision=minimum,
    )


def test_committed_agent_home_contract_matches_current_production_owners() -> None:
    check_snapshot(DEFAULT_SNAPSHOT)


def test_changed_release_contract_requires_a_new_home_revision() -> None:
    baseline = build_snapshot()

    with pytest.raises(
        HomeReleaseContractError,
        match="changed without a new home revision",
    ):
        compare_snapshots(baseline, _changed_snapshot(revision_offset=0))

    compare_snapshots(baseline, _changed_snapshot(revision_offset=1))


def test_release_workflows_enforce_and_publish_the_home_contract() -> None:
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = (REPO_ROOT / ".github/workflows/managed-release.yml").read_text(
        encoding="utf-8"
    )

    assert "check_home_release_contract.py check" in ci
    assert "check_home_release_contract.py compare" in ci
    assert "git tag --merged HEAD" in ci
    assert "check_home_release_contract.py check" in release
    assert "check_home_release_contract.py compare" in release
    assert release.count("agent-home-contract.json") >= 7
