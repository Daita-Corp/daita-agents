import json
from pathlib import Path

import pytest

from daita.db.safety import DbCapabilityLane, DbSafetyVerifier


FIXTURE = Path(__file__).parents[2] / "fixtures/db/safety_lane_replay/core.json"


def _frame(prompt, *, requested_capabilities=()):
    return DbSafetyVerifier().verify(
        prompt,
        requested_capabilities=requested_capabilities,
    )


def test_write_execute_frame_records_exact_permission_fields():
    frame = _frame("execute update orders set status = 'closed'")
    actual = frame.to_dict()

    assert set(actual) == {
        "prompt",
        "normalized_prompt",
        "explicit_schema_only",
        "direct_memory_operation",
        "sql_statement_type",
        "has_db_target",
        "has_mutation_payload",
        "explicit_execution",
        "destructive",
        "admin",
        "monitor_operation",
        "requested_capabilities",
        "granted_lanes",
        "forbidden_capabilities",
        "rewrites",
        "assumptions",
        "lane_grants",
        "required_capabilities",
        "approval_required",
        "blocked_actions",
    }
    assert actual["prompt"] == "execute update orders set status = 'closed'"
    assert actual["normalized_prompt"] == "execute update orders set status = 'closed'"
    assert actual["explicit_schema_only"] is False
    assert actual["direct_memory_operation"] is None
    assert actual["sql_statement_type"] == "update"
    assert actual["has_db_target"] is True
    assert actual["has_mutation_payload"] is True
    assert actual["explicit_execution"] is True
    assert actual["destructive"] is False
    assert actual["admin"] is False
    assert actual["monitor_operation"] is None
    assert actual["requested_capabilities"] == []
    assert actual["granted_lanes"] == ["write_execute"]
    assert actual["forbidden_capabilities"] == []
    assert actual["rewrites"] == []
    assert actual["assumptions"] == []
    assert actual["required_capabilities"] == [
        "db.sql.validate",
        "db.sql.execute_write",
    ]
    assert actual["approval_required"] is True
    assert actual["blocked_actions"] == ["row_read", "admin", "monitor_action"]
    assert actual["lane_grants"] == [
        {
            "lane": "write_execute",
            "required_capabilities": [
                "db.sql.validate",
                "db.sql.execute_write",
            ],
            "forbidden_capabilities": [],
            "approval_required": True,
            "reason": "explicit_write_execution",
        }
    ]


@pytest.mark.parametrize(
    ("prompt", "expected_lanes"),
    [
        ("write up a summary", (DbCapabilityLane.NONE,)),
        ("write me a report", (DbCapabilityLane.NONE,)),
        ("update me on the schema", (DbCapabilityLane.SCHEMA,)),
        ("delete this from my notes", (DbCapabilityLane.NONE,)),
        ("drop me a summary", (DbCapabilityLane.NONE,)),
    ],
)
def test_phrase_traps_never_grant_sql_write(prompt, expected_lanes):
    frame = _frame(prompt)

    assert frame.granted_lanes == expected_lanes
    assert DbCapabilityLane.WRITE_PROPOSE not in frame.granted_lanes
    assert DbCapabilityLane.WRITE_EXECUTE not in frame.granted_lanes
    assert "db.sql.execute_write" not in frame.required_capabilities
    assert "write_execute" in frame.blocked_actions


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "what columns are in orders",
            {
                "lanes": ("schema",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": None,
                "required": [
                    "catalog.schema.search",
                    "catalog.asset.inspect",
                    "db.schema.inspect",
                ],
            },
        ),
        (
            "schema evidence only; do not query rows",
            {
                "lanes": ("schema",),
                "schema_only": True,
                "sql": None,
                "memory": None,
                "monitor": None,
                "required": [
                    "catalog.schema.search",
                    "catalog.asset.inspect",
                    "db.schema.inspect",
                ],
            },
        ),
        (
            "remember that revenue excludes tax",
            {
                "lanes": ("memory_write",),
                "schema_only": False,
                "sql": None,
                "memory": "update",
                "monitor": None,
                "required": [
                    "db.memory.plan_update",
                    "db.memory.commit_update",
                ],
            },
        ),
        (
            "remember this: revenue excludes tax",
            {
                "lanes": ("memory_write",),
                "schema_only": False,
                "sql": None,
                "memory": "update",
                "monitor": None,
                "required": [
                    "db.memory.plan_update",
                    "db.memory.commit_update",
                ],
            },
        ),
        (
            "note revenue excludes tax",
            {
                "lanes": ("memory_write",),
                "schema_only": False,
                "sql": None,
                "memory": "update",
                "monitor": None,
                "required": [
                    "db.memory.plan_update",
                    "db.memory.commit_update",
                ],
            },
        ),
        (
            "what do you remember about revenue",
            {
                "lanes": ("memory_answer",),
                "schema_only": False,
                "sql": None,
                "memory": "recall",
                "monitor": None,
                "required": [
                    "memory.semantic.recall",
                    "db.memory.answer_context.build",
                ],
            },
        ),
        (
            "list memories for orders",
            {
                "lanes": ("memory_answer",),
                "schema_only": False,
                "sql": None,
                "memory": "list",
                "monitor": None,
                "required": [
                    "memory.semantic.recall",
                    "db.memory.answer_context.build",
                ],
            },
        ),
        (
            "how many orders are there",
            {
                "lanes": ("read",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": None,
                "required": ["db.sql.validate", "db.sql.execute_read"],
            },
        ),
        (
            "show recent orders",
            {
                "lanes": ("read",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": None,
                "required": ["db.sql.validate", "db.sql.execute_read"],
            },
        ),
        (
            "list order rows",
            {
                "lanes": ("read",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": None,
                "required": ["db.sql.validate", "db.sql.execute_read"],
            },
        ),
        (
            "list monitors",
            {
                "lanes": ("monitor_read",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": "read",
                "required": ["db.monitor.inspect"],
            },
        ),
        (
            "create a monitor for revenue drops",
            {
                "lanes": ("monitor_write",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": "write",
                "required": [
                    "db.monitor.plan_create",
                    "db.monitor.plan_lifecycle",
                ],
            },
        ),
        (
            "run the revenue monitor now",
            {
                "lanes": ("monitor_execute",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": "execute",
                "required": ["db.monitor.execute"],
            },
        ),
        (
            "alert me when revenue drops",
            {
                "lanes": ("monitor_write",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": "write",
                "required": [
                    "db.monitor.plan_create",
                    "db.monitor.plan_lifecycle",
                ],
            },
        ),
        (
            "watch revenue daily and alert me",
            {
                "lanes": ("monitor_write",),
                "schema_only": False,
                "sql": None,
                "memory": None,
                "monitor": "write",
                "required": [
                    "db.monitor.plan_create",
                    "db.monitor.plan_lifecycle",
                ],
            },
        ),
    ],
)
def test_expected_examples_are_permission_lanes_not_semantic_intents(
    prompt,
    expected,
):
    frame = _frame(prompt)

    assert tuple(lane.value for lane in frame.granted_lanes) == expected["lanes"]
    assert frame.explicit_schema_only is expected["schema_only"]
    assert frame.sql_statement_type == expected["sql"]
    assert frame.direct_memory_operation == expected["memory"]
    assert frame.monitor_operation == expected["monitor"]
    assert list(frame.required_capabilities) == expected["required"]


@pytest.mark.parametrize(
    ("prompt", "expected_lane", "approval"),
    [
        ("update orders set status = 'closed'", "write_propose", True),
        ("execute update orders set status = 'closed'", "write_execute", True),
        ("delete from orders where id = 10", "write_propose", True),
        ("run delete from orders where id = 10", "write_execute", True),
    ],
)
def test_write_lanes_require_sql_shape_and_execution_words(
    prompt,
    expected_lane,
    approval,
):
    frame = _frame(prompt)

    assert tuple(lane.value for lane in frame.granted_lanes) == (expected_lane,)
    assert frame.has_db_target is True
    assert frame.has_mutation_payload is True
    assert frame.approval_required is approval


def test_polite_prefix_still_extracts_explicit_sql_shape():
    frame = _frame("please execute update orders set status = closed")

    assert frame.sql_statement_type == "update"
    assert frame.granted_lanes == (DbCapabilityLane.WRITE_EXECUTE,)
    assert frame.required_capabilities == (
        "db.sql.validate",
        "db.sql.execute_write",
    )


@pytest.mark.parametrize(
    ("prompt", "statement", "from_lanes"),
    [
        ("schema only: select * from orders", "select", ["read"]),
        (
            "metadata only: execute update orders set status = closed",
            "update",
            ["write_execute"],
        ),
    ],
)
def test_schema_only_rewrites_embedded_sql_to_schema_lane(
    prompt,
    statement,
    from_lanes,
):
    frame = _frame(prompt)

    assert frame.explicit_schema_only is True
    assert frame.sql_statement_type == statement
    assert frame.granted_lanes == (DbCapabilityLane.SCHEMA,)
    assert frame.rewrites[0].to_dict() == {
        "rule": "schema_only_forbids_sql_execution",
        "from_lanes": from_lanes,
        "to_lanes": ["schema"],
        "reason": "hard_schema_only_negation",
    }
    assert "db.sql.execute_read" in frame.forbidden_capabilities
    assert "db.sql.execute_write" in frame.forbidden_capabilities


def test_stricter_control_plane_lane_keeps_embedded_sql_shape_blocked():
    frame = _frame("show monitor status and execute update orders set status = closed")

    assert frame.granted_lanes == (DbCapabilityLane.MONITOR_READ,)
    assert frame.sql_statement_type == "update"
    assert frame.monitor_operation == "read"
    assert "db.sql.execute_write" in frame.forbidden_capabilities
    assert "db.sql.execute_write" not in frame.required_capabilities


def test_unsafe_or_underspecified_write_downgrades_with_rewrite():
    frame = _frame("update orders")

    assert frame.granted_lanes == (DbCapabilityLane.NONE,)
    assert frame.rewrites[0].to_dict() == {
        "rule": "write_requires_explicit_sql_shape",
        "from_lanes": ["write_propose"],
        "to_lanes": ["none"],
        "reason": "natural_language_write_not_enough",
    }
    assert "write_downgraded_without_explicit_sql_shape" in frame.assumptions
    assert "db.sql.execute_write" in frame.forbidden_capabilities


def test_requested_capabilities_cannot_override_forbids():
    frame = _frame(
        "schema only; do not query rows",
        requested_capabilities=("db.sql.execute_read", "db.sql.execute_write"),
    )

    assert frame.granted_lanes == (DbCapabilityLane.SCHEMA,)
    assert frame.requested_capabilities == (
        "db.sql.execute_read",
        "db.sql.execute_write",
    )
    assert "db.sql.execute_read" in frame.forbidden_capabilities
    assert "db.sql.execute_write" in frame.forbidden_capabilities
    assert "db.sql.execute_read" not in frame.required_capabilities
    assert "db.sql.execute_write" not in frame.required_capabilities
    assert len(frame.rewrites) == 2
    assert {rewrite.rule for rewrite in frame.rewrites} == {
        "requested_capability_blocked_by_forbid"
    }
    assert "requested_capabilities_cannot_override_forbids" in frame.assumptions


def test_admin_requires_explicit_destructive_sql_structure():
    trap = _frame("drop me a summary")
    admin = _frame("drop table orders")
    truncate = _frame("truncate orders")

    assert trap.granted_lanes == (DbCapabilityLane.NONE,)
    assert trap.admin is False
    assert admin.granted_lanes == (DbCapabilityLane.ADMIN,)
    assert admin.admin is True
    assert admin.destructive is True
    assert admin.approval_required is True
    assert truncate.granted_lanes == (DbCapabilityLane.ADMIN,)
    assert truncate.admin is True
    assert truncate.destructive is True


def test_monitor_lanes_require_explicit_control_plane_structure():
    vague = _frame("monitor revenue trends")
    write = _frame("pause the revenue monitor")
    execute = _frame("send the monitor report now")

    assert vague.granted_lanes == (DbCapabilityLane.NONE,)
    assert vague.monitor_operation is None
    assert write.granted_lanes == (DbCapabilityLane.MONITOR_WRITE,)
    assert write.monitor_operation == "write"
    assert execute.granted_lanes == (DbCapabilityLane.MONITOR_EXECUTE,)
    assert execute.monitor_operation == "execute"


def test_safety_lane_replay_core_fixture():
    cases = json.loads(FIXTURE.read_text())
    verifier = DbSafetyVerifier()

    for case in cases:
        frame = verifier.verify(
            case["prompt"],
            requested_capabilities=case.get("requested_capabilities", ()),
        )

        assert [lane.value for lane in frame.granted_lanes] == case["lanes"]
        assert list(frame.forbidden_capabilities) == case["forbidden_capabilities"]
        assert [rewrite.rule for rewrite in frame.rewrites] == case["rewrite_rules"]
        assert list(frame.assumptions) == case["assumptions"]
        assert frame.approval_required is case["approval_required"]
        assert list(frame.blocked_actions) == case["blocked_actions"]
