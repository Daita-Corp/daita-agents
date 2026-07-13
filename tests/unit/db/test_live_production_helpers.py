import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from daita.db.context_projection import (
    ProjectionContext,
    ProjectionMode,
    project_operation_result,
)
from daita.db.models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from daita.runtime import (
    AccessMode,
    Evidence,
    Operation,
    OperationSnapshot,
    OperationStatus,
    Task,
    TaskStatus,
)
from tests.integration.from_db.live_production_helpers import (
    assert_no_unexpected_write_execution,
    assert_scalar_answer_fact,
    assert_synthesized_answer,
    assert_sql_is_read_only,
    query_rows,
    latest_evidence,
    row_values,
    sql_from_result,
    task_capabilities,
    write_failure_artifacts,
)
from tests.integration.from_db.test_from_db_memory_live import (
    _public_planning_context,
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


@dataclass(frozen=True)
class FakeTask:
    capability_id: str
    input: dict[str, Any]
    id: str = "task-test"
    executor_id: str = "executor-test"
    status: Any = "pending"


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


def _projected_result_and_snapshot(
    *,
    capability_id: str = "db.sql.execute_read",
) -> tuple[DbOperationResult, OperationSnapshot]:
    operation_id = "op-projected-helper-contract"
    task = Task(
        id="task-projected-read",
        operation_id=operation_id,
        capability_id=capability_id,
        executor_id="sqlite.read",
        input={"sql": "select count(*) as count from customers"},
        status=TaskStatus.SUCCEEDED,
    )
    raw_evidence = (
        Evidence(
            id="ev-projected-query",
            kind="query.result",
            owner="sqlite",
            operation_id=operation_id,
            task_id=task.id,
            payload={
                "sql": "select count(*) as count from customers",
                "rows": [{"count": 4}],
            },
        ),
        Evidence(
            id="ev-projected-synthesis",
            kind="answer.synthesis",
            owner="db_runtime",
            operation_id=operation_id,
            payload={"answer": "There are 4 customers."},
        ),
    )
    raw_result = DbOperationResult(
        operation_id=operation_id,
        request=DbRequest("How many customers are there?"),
        intent=DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        contract=DbOperationContract(operation_type="db.run", access=AccessMode.READ),
        status=OperationStatus.SUCCEEDED,
        answer="There are 4 customers.",
        evidence=raw_evidence,
        diagnostics={
            "execution": {
                "tasks": [task.to_dict()],
                "task_count": 1,
            }
        },
    )
    public_result = project_operation_result(
        raw_result,
        ProjectionContext(mode=ProjectionMode.PUBLIC_RESULT),
    )
    snapshot = OperationSnapshot(
        operation=Operation(
            id=operation_id,
            operation_type="db.run",
            status=OperationStatus.SUCCEEDED,
        ),
        tasks=(task,),
        evidence=raw_evidence,
    )
    return public_result, snapshot


def test_sql_from_result_reads_query_result_sql():
    snapshot = _snapshot(
        FakeEvidence(
            kind="query.result",
            payload={"rows": [{"count": 4}], "sql": "select count(*) from customers"},
        )
    )

    assert sql_from_result(snapshot) == "select count(*) from customers"


def test_sql_from_result_reads_query_plan_validation_sql():
    snapshot = _snapshot(
        FakeEvidence(
            kind="query.plan.validation",
            payload={"valid": True, "query": "select customer_id from customers"},
        )
    )

    assert sql_from_result(snapshot) == "select customer_id from customers"


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


def test_sql_from_result_falls_back_to_planner_compilation_sql():
    snapshot = _snapshot(
        FakeEvidence(
            kind="planner.compilation",
            payload={
                "compilation": {
                    "task_specs": [
                        {
                            "capability_id": "db.sql.validate",
                            "input": {"sql": "select count(*) from compiled_customers"},
                        }
                    ]
                }
            },
        )
    )

    assert sql_from_result(snapshot) == "select count(*) from compiled_customers"


def test_sql_from_result_falls_back_to_snapshot_task_input():
    snapshot = FakeSnapshot(
        operation=SimpleNamespace(id="op-test"),
        evidence=(),
        tasks=(
            FakeTask(
                capability_id="db.sql.validate",
                input={"sql": "select count(*) from task_customers"},
            ),
        ),
    )

    assert sql_from_result(snapshot) == "select count(*) from task_customers"


def test_sql_from_result_falls_back_to_diagnostics_task_input():
    result = SimpleNamespace(
        evidence=(),
        diagnostics={
            "execution": {
                "tasks": [
                    {
                        "capability_id": "db.sql.validate",
                        "input": {"sql": "select count(*) from diagnostics_customers"},
                    }
                ]
            }
        },
    )

    assert sql_from_result(result) == "select count(*) from diagnostics_customers"


def test_projected_result_and_snapshot_helpers_respect_the_observation_boundary():
    public_result, snapshot = _projected_result_and_snapshot()

    assert task_capabilities(public_result) == ["db.sql.execute_read"]
    assert task_capabilities(snapshot) == ["db.sql.execute_read"]
    assert query_rows(snapshot) == [{"count": 4}]
    assert sql_from_result(snapshot) == "select count(*) as count from customers"

    public_payload = json.dumps(
        {
            "diagnostics": public_result.diagnostics,
            "evidence": [item.to_dict() for item in public_result.evidence],
        },
        sort_keys=True,
    )
    assert "select count(*) as count from customers" not in public_payload
    assert '"count": 4' not in public_payload
    with pytest.raises(AssertionError, match="requires an OperationSnapshot"):
        query_rows(public_result)
    with pytest.raises(AssertionError, match="requires an OperationSnapshot"):
        row_values(public_result)
    with pytest.raises(AssertionError, match="requires an OperationSnapshot"):
        sql_from_result(public_result)
    with pytest.raises(AssertionError, match="requires an OperationSnapshot"):
        assert_scalar_answer_fact(public_result, value=4)


def test_synthesized_answer_helper_validates_raw_and_public_surfaces_together():
    public_result, snapshot = _projected_result_and_snapshot()

    assert_synthesized_answer(snapshot, public_result=public_result)


def test_latest_planning_context_selection_uses_latest_accepted_evidence():
    snapshot = _snapshot(
        FakeEvidence(
            kind="planning.context",
            payload={"rendered_context": "first accepted"},
            id="ev-context-first",
        ),
        FakeEvidence(
            kind="planning.context",
            payload={"rendered_context": "rejected newest"},
            accepted=False,
            id="ev-context-rejected",
        ),
        FakeEvidence(
            kind="planning.context",
            payload={"rendered_context": "latest accepted"},
            id="ev-context-latest",
        ),
    )

    context = latest_evidence(snapshot, "planning.context")
    payload = _public_planning_context(snapshot)

    assert context is not None
    assert context.id == "ev-context-latest"
    assert payload["rendered_context"] == "latest accepted"


def test_unexpected_write_checks_use_public_task_refs_and_snapshot_tasks():
    public_result, snapshot = _projected_result_and_snapshot(
        capability_id="db.sql.execute_write"
    )

    with pytest.raises(AssertionError, match="Unexpected write capabilities"):
        assert_no_unexpected_write_execution(public_result)
    with pytest.raises(AssertionError, match="Unexpected write capabilities"):
        assert_no_unexpected_write_execution(snapshot)


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
