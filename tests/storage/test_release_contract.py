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


@pytest.mark.parametrize("backend", ["postgresql", "s3_artifacts"])
def test_backend_contracts_use_the_same_home_revision_and_release_gate(backend) -> None:
    baseline = build_snapshot()
    changed = deepcopy(baseline)
    contracts = changed["backend_contracts"]
    assert isinstance(contracts, dict)
    contracts[backend] = "0" * 64
    with pytest.raises(HomeReleaseContractError, match="without a new home revision"):
        compare_snapshots(baseline, changed)
    revision = baseline["home_revision"]
    assert isinstance(revision, int)
    changed["home_revision"] = revision + 1
    compare_snapshots(baseline, changed)


def test_first_backend_snapshot_preserves_the_released_sqlite_contract() -> None:
    candidate = build_snapshot()
    released = deepcopy(candidate)
    released.pop("backend_contracts")
    compare_snapshots(released, candidate)
    with pytest.raises(HomeReleaseContractError, match="backend contract changed"):
        compare_snapshots(candidate, released)
    malformed = deepcopy(candidate)
    malformed["backend_contracts"] = {"postgresql": "not-a-checksum"}
    with pytest.raises(HomeReleaseContractError, match="invalid backend contracts"):
        compare_snapshots(released, malformed)


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
