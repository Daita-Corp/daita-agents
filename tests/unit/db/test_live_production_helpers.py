import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from tests.integration.from_db.live_production_helpers import (
    assert_sql_is_read_only,
    sql_from_result,
    write_failure_artifacts,
)


@dataclass(frozen=True)
class FakeEvidence:
    kind: str
    payload: dict[str, Any]
    accepted: bool = True
    id: str = "ev-test"
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "accepted": self.accepted,
            "task_id": self.task_id,
            "payload": self.payload,
        }


class FakeSnapshot(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": {"id": self.operation.id},
            "evidence": [item.to_dict() for item in self.evidence],
            "tasks": [],
        }


def _snapshot(*evidence: FakeEvidence) -> FakeSnapshot:
    return FakeSnapshot(
        operation=SimpleNamespace(id="op-test"),
        evidence=evidence,
        tasks=(),
    )


def test_sql_from_result_falls_back_to_accepted_query_plan_proposal():
    snapshot = _snapshot(
        FakeEvidence(
            kind="query.plan.proposal",
            payload={
                "structured_plan": {
                    "selected_sql": " select count(*) as count from customers "
                }
            },
        )
    )

    assert sql_from_result(snapshot) == "select count(*) as count from customers"


def test_sql_from_result_prefers_existing_executed_validation_order_over_plan():
    snapshot = _snapshot(
        FakeEvidence(
            kind="query.plan.proposal",
            payload={"sql": "select count(*) from planned_customers"},
        ),
        FakeEvidence(
            kind="query.result",
            payload={"sql": "select count(*) from result_customers"},
        ),
        FakeEvidence(
            kind="sql.validation",
            payload={"sql": "select count(*) from validated_customers"},
        ),
    )

    assert sql_from_result(snapshot) == "select count(*) from validated_customers"


def test_sql_from_result_ignores_rejected_query_plan_proposal():
    snapshot = _snapshot(
        FakeEvidence(
            kind="query.plan.proposal",
            accepted=False,
            payload={"sql": "select count(*) from customers"},
        )
    )

    assert sql_from_result(snapshot) == ""


def test_assert_sql_is_read_only_catches_destructive_verbs():
    with pytest.raises(AssertionError, match="unsafe verbs"):
        assert_sql_is_read_only("select * from customers; drop table customers")


def test_failure_artifacts_split_planner_decisions_and_compilations(tmp_path):
    snapshot = _snapshot(
        FakeEvidence(
            kind="planner.decision",
            payload={"actions": [{"kind": "query"}]},
            id="ev-decision",
        ),
        FakeEvidence(
            kind="planner.compilation",
            payload={"compiled_tasks": [{"capability_id": "db.sql.validate"}]},
            id="ev-compilation",
        ),
        FakeEvidence(
            kind="query.plan.proposal",
            payload={"sql": "select count(*) from customers"},
            id="ev-plan",
        ),
    )

    artifact_dir = write_failure_artifacts(tmp_path, snapshot=snapshot)

    decisions = json.loads((artifact_dir / "planner_decisions.json").read_text())
    compilations = json.loads((artifact_dir / "planner_compilations.json").read_text())
    assert decisions[0]["id"] == "ev-decision"
    assert compilations[0]["id"] == "ev-compilation"
    assert (artifact_dir / "sql.txt").read_text().strip() == (
        "select count(*) from customers"
    )
