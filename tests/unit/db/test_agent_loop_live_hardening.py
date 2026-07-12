from daita.db import DbRuntime
from daita.db.loop import DbAgentLoop
from daita.db.loop.progress import _LoopProgressGuard
from daita.db.planner_protocol import (
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.plugins.catalog import CatalogPlugin


async def test_relationship_path_action_requires_structured_assets():
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False),))
    await runtime.setup(agent_id="agent-loop-live-hardening")

    try:
        compilation = DbAgentLoop(runtime, object()).compile_actions(
            DbPlannerDecision(
                status=DbPlannerDecisionStatus.CONTINUE,
                intent={"operation_type": "data.query"},
                actions=(
                    DbPlannerAction(
                        action_id="find_join",
                        kind=DbPlannerActionKind.FIND_RELATIONSHIP_PATHS,
                        input={"owner": "catalog"},
                        metadata={
                            "target_hint": "relationship between orders and customers"
                        },
                    ),
                ),
            ),
            _loop_state(prompt="Find relationship paths for the selected assets"),
        )
    finally:
        await runtime.teardown()

    assert compilation.task_specs == ()
    assert {item["error"] for item in compilation.rejected_action_summaries} == {
        "missing_from_assets",
        "missing_to_assets",
    }


async def test_relationship_path_action_normalizes_common_metadata_pair():
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False),))
    await runtime.setup(agent_id="agent-loop-live-hardening")

    try:
        compilation = DbAgentLoop(runtime, object()).compile_actions(
            DbPlannerDecision(
                status=DbPlannerDecisionStatus.CONTINUE,
                intent={"operation_type": "data.query"},
                actions=(
                    DbPlannerAction(
                        action_id="find_join",
                        kind=DbPlannerActionKind.FIND_RELATIONSHIP_PATHS,
                        input={"owner": "catalog"},
                        metadata={"from_assets": ["orders", "customers"]},
                    ),
                ),
            ),
            _loop_state(),
        )
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert len(compilation.task_specs) == 1
    assert compilation.task_specs[0].input == {
        "from_assets": ["orders"],
        "to_assets": ["customers"],
    }


async def test_relationship_path_action_normalizes_explicit_input_pair():
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False),))
    await runtime.setup(agent_id="agent-loop-live-hardening")

    try:
        compilation = DbAgentLoop(runtime, object()).compile_actions(
            DbPlannerDecision(
                status=DbPlannerDecisionStatus.CONTINUE,
                intent={"operation_type": "data.query"},
                actions=(
                    DbPlannerAction(
                        action_id="find_join",
                        kind=DbPlannerActionKind.FIND_RELATIONSHIP_PATHS,
                        input={
                            "owner": "catalog",
                            "tables": ["orders", "customers"],
                        },
                    ),
                ),
            ),
            _loop_state(prompt="Find relationship paths for the selected assets"),
        )
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert compilation.task_specs[0].input == {
        "from_assets": ["orders"],
        "to_assets": ["customers"],
    }


def test_repeated_sql_failure_is_terminal_on_second_occurrence():
    guard = _LoopProgressGuard()
    facts = {
        "sql_error_fingerprints": ["same-sql-error"],
        "failed_action": True,
        "new_accepted_evidence_refs": [],
        "compiled_action_fingerprints": ["execute-latest-plan"],
        "no_progress": False,
        "progress_fingerprint": "same-progress",
    }

    first = guard.evaluate(facts)
    second = guard.evaluate(facts)

    assert first.terminal_status is None
    assert second.terminal_status == "failed"
    assert "db_agent_loop_repeated_sql_failure" in second.warnings
    assert any(
        item.get("warning") == "db_agent_loop_repeated_sql_failure"
        and item.get("count") == 2
        for item in second.retry_facts
    )


def _loop_state(
    *,
    prompt: str = "Join orders to customers using their relationship",
) -> DbLoopState:
    return DbLoopState(
        operation_id="db-op-live-hardening",
        normalized_user_request={"prompt": prompt},
        safety_frame={"max_access": "metadata_read"},
        available_action_kinds=tuple(DbPlannerActionKind),
    )
