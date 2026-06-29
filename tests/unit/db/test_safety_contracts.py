from daita.db.contracts import DbContractBuilder
from daita.db.models import DbRequest, DbRuntimeConfig
from daita.db.runtime import DbRuntime
from daita.db.safety import DbSafetyVerifier
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.memory.memory_plugin import MemoryPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode
from daita.skills import SkillRuntimeEffects


def _runtime(*, plugins=True) -> DbRuntime:
    if not plugins:
        return DbRuntime()
    return DbRuntime(
        plugins=(
            CatalogPlugin(auto_persist=False),
            SQLitePlugin(path=":memory:"),
            MemoryPlugin(workspace="safety-contract-tests"),
        )
    )


def _contract(prompt, *, requested_capabilities=(), plugins=True):
    request = DbRequest(prompt, requested_capabilities=tuple(requested_capabilities))
    frame = DbSafetyVerifier().verify(request)
    runtime = _runtime(plugins=plugins)
    contract = DbContractBuilder(runtime.registry, DbRuntimeConfig()).build(
        request,
        frame,
    )
    return frame, contract


def _blocked_ids(contract):
    return [item["id"] for item in contract.metadata["blocked_capabilities"]]


def test_skill_contract_metadata_cannot_overwrite_safety_fields():
    request = DbRequest("schema only; do not query rows")
    frame = DbSafetyVerifier().verify(request)
    runtime = _runtime()
    effect = SkillRuntimeEffects(
        skill_id="unsafe_metadata",
        contract_metadata={
            "forbidden_capabilities": [],
            "required_capabilities": ["db.sql.execute_write"],
            "approval_required": False,
            "safety_frame": {"forged": True},
            "custom": {"hint": "kept"},
        },
    )

    contract = DbContractBuilder(runtime.registry, DbRuntimeConfig()).build(
        request,
        frame,
        skill_effects=(effect,),
    )

    assert contract.metadata["forbidden_capabilities"] == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert contract.metadata["required_capabilities"] == [
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.schema.inspect",
    ]
    assert contract.metadata["approval_required"] is False
    assert contract.metadata["safety_frame"] == frame.to_dict()
    assert contract.metadata["skill_contract_metadata"] == {
        "forbidden_capabilities": [],
        "required_capabilities": ["db.sql.execute_write"],
        "approval_required": False,
        "safety_frame": {"forged": True},
        "custom": {"hint": "kept"},
    }
    assert "custom" not in contract.metadata


def test_none_lane_forbids_sql_without_required_capabilities():
    frame, contract = _contract("write me a report")

    assert [lane.value for lane in frame.granted_lanes] == ["none"]
    assert contract.operation_type == "db.none"
    assert contract.access is AccessMode.NONE
    assert contract.required_capabilities == ()
    assert contract.required_evidence == ()
    assert contract.metadata["granted_lanes"] == ["none"]
    assert contract.metadata["required_capabilities"] == []
    assert contract.metadata["forbidden_capabilities"] == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert _blocked_ids(contract) == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert contract.metadata["missing_capabilities"] == []
    assert contract.metadata["approval_required"] is False
    assert contract.metadata["safety_frame"] == frame.to_dict()


def test_schema_contract_uses_lane_requirements_and_blocks_requested_sql():
    frame, contract = _contract(
        "schema only; do not query rows",
        requested_capabilities=("db.sql.execute_read", "db.sql.execute_write"),
    )

    assert [lane.value for lane in frame.granted_lanes] == ["schema"]
    assert contract.operation_type == "schema.query"
    assert contract.access is AccessMode.METADATA_READ
    assert contract.required_capabilities == (
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.schema.inspect",
    )
    assert contract.metadata["required_capabilities"] == [
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.schema.inspect",
    ]
    assert contract.metadata["forbidden_capabilities"] == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert _blocked_ids(contract) == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert contract.metadata["missing_capabilities"] == []
    assert "db.sql.execute_read" not in contract.required_capabilities
    assert "db.sql.execute_write" not in contract.required_capabilities
    assert contract.metadata["diagnostics"]["requested_capabilities"] == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]


def test_requested_capability_can_add_a_lane_but_forbids_still_win():
    frame, contract = _contract(
        "show recent orders",
        requested_capabilities=("catalog.schema.search", "db.sql.execute_write"),
    )

    assert [lane.value for lane in frame.granted_lanes] == ["schema", "read"]
    assert contract.metadata["forbidden_capabilities"] == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert contract.metadata["required_capabilities"] == [
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.schema.inspect",
        "db.sql.validate",
    ]
    assert contract.required_capabilities == (
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.schema.inspect",
        "db.sql.validate",
    )
    assert "db.sql.execute_read" not in contract.required_capabilities
    assert "db.sql.execute_write" not in contract.required_capabilities
    assert contract.metadata["missing_capabilities"] == []


def test_missing_capabilities_report_required_lane_ids_without_forbidden_ids():
    frame, contract = _contract("what columns are in orders", plugins=False)

    assert [lane.value for lane in frame.granted_lanes] == ["schema"]
    assert contract.required_capabilities == ()
    assert contract.metadata["required_capabilities"] == [
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.schema.inspect",
    ]
    assert contract.metadata["missing_capabilities"] == [
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.schema.inspect",
    ]
    assert contract.metadata["forbidden_capabilities"] == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert "db.sql.execute_read" not in contract.metadata["missing_capabilities"]
    assert "db.sql.execute_write" not in contract.metadata["missing_capabilities"]


def test_memory_answer_contract_forbids_sql_validation_and_execution():
    frame, contract = _contract("what do you remember about revenue")

    assert [lane.value for lane in frame.granted_lanes] == ["memory_answer"]
    assert contract.operation_type == "memory.recall"
    assert contract.required_capabilities == (
        "memory.semantic.recall",
        "db.memory.answer_context.build",
    )
    assert contract.metadata["forbidden_capabilities"] == [
        "db.sql.validate",
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert _blocked_ids(contract) == [
        "db.sql.validate",
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert contract.metadata["missing_capabilities"] == []
    assert contract.metadata["approval_required"] is False


def test_memory_write_contract_uses_plan_and_commit_lane_requirements():
    frame, contract = _contract("remember that revenue excludes tax")

    assert [lane.value for lane in frame.granted_lanes] == ["memory_write"]
    assert contract.operation_type == "memory.update"
    assert contract.access is AccessMode.WRITE
    assert contract.required_capabilities == (
        "db.memory.plan_update",
        "db.memory.commit_update",
    )
    assert contract.metadata["forbidden_capabilities"] == [
        "db.sql.validate",
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert contract.metadata["missing_capabilities"] == []
    assert contract.metadata["approval_required"] is False


def test_read_contract_requires_sql_validate_and_read_but_forbids_write():
    frame, contract = _contract("how many orders are there")

    assert [lane.value for lane in frame.granted_lanes] == ["read"]
    assert contract.operation_type == "data.query"
    assert contract.access is AccessMode.READ
    assert contract.required_capabilities == (
        "db.sql.validate",
        "db.sql.execute_read",
    )
    assert contract.metadata["forbidden_capabilities"] == ["db.sql.execute_write"]
    assert _blocked_ids(contract) == ["db.sql.execute_write"]
    assert contract.metadata["missing_capabilities"] == []
    assert contract.metadata["approval_required"] is False


def test_write_propose_contract_requires_validation_and_approval_only():
    frame, contract = _contract("update orders set status = 'closed'")

    assert [lane.value for lane in frame.granted_lanes] == ["write_propose"]
    assert contract.operation_type == "write.propose"
    assert contract.required_capabilities == ("db.sql.validate",)
    assert contract.metadata["forbidden_capabilities"] == ["db.sql.execute_write"]
    assert _blocked_ids(contract) == ["db.sql.execute_write"]
    assert contract.metadata["missing_capabilities"] == []
    assert contract.metadata["approval_required"] is True
    assert contract.policy_ids == ("runtime:approval_required_for_safety_lane",)


def test_write_execute_contract_requires_validation_write_and_approval():
    frame, contract = _contract("execute update orders set status = 'closed'")

    assert [lane.value for lane in frame.granted_lanes] == ["write_execute"]
    assert contract.operation_type == "write.execute"
    assert contract.access is AccessMode.WRITE
    assert contract.required_capabilities == (
        "db.sql.validate",
        "db.sql.execute_write",
    )
    assert contract.metadata["forbidden_capabilities"] == []
    assert contract.metadata["blocked_capabilities"] == []
    assert contract.metadata["missing_capabilities"] == []
    assert contract.metadata["approval_required"] is True
    assert contract.policy_ids == ("runtime:approval_required_for_safety_lane",)


def test_admin_contract_reports_admin_declaration_missing_and_requires_approval():
    frame, contract = _contract("drop table orders")

    assert [lane.value for lane in frame.granted_lanes] == ["admin"]
    assert contract.operation_type == "admin"
    assert contract.access is AccessMode.ADMIN
    assert contract.required_capabilities == ()
    assert contract.metadata["required_capabilities"] == ["db.admin.propose"]
    assert contract.metadata["missing_capabilities"] == ["db.admin.propose"]
    assert contract.metadata["forbidden_capabilities"] == [
        "db.sql.execute_read",
        "db.sql.execute_write",
    ]
    assert contract.metadata["approval_required"] is True


def test_monitor_read_contract_blocks_mutation_execution_and_sql_write():
    frame, contract = _contract("list monitors")

    assert [lane.value for lane in frame.granted_lanes] == ["monitor_read"]
    assert contract.operation_type == "monitor.read"
    assert contract.access is AccessMode.METADATA_READ
    assert contract.required_capabilities == ()
    assert contract.metadata["required_capabilities"] == ["db.monitor.inspect"]
    assert contract.metadata["missing_capabilities"] == ["db.monitor.inspect"]
    assert contract.metadata["forbidden_capabilities"] == [
        "db.monitor.commit_create",
        "db.monitor.commit_lifecycle",
        "monitor.delivery.local",
        "monitor.delivery.in_app",
        "db.sql.execute_write",
    ]
    assert _blocked_ids(contract) == [
        "db.monitor.commit_create",
        "db.monitor.commit_lifecycle",
        "monitor.delivery.local",
        "monitor.delivery.in_app",
        "db.sql.execute_write",
    ]
    assert contract.metadata["approval_required"] is False


def test_monitor_write_contract_requires_plan_capabilities_and_approval():
    frame, contract = _contract("create a monitor for revenue drops")

    assert [lane.value for lane in frame.granted_lanes] == ["monitor_write"]
    assert contract.operation_type == "monitor.write"
    assert contract.access is AccessMode.WRITE
    assert contract.required_capabilities == (
        "db.monitor.plan_create",
        "db.monitor.plan_lifecycle",
    )
    assert contract.metadata["forbidden_capabilities"] == ["db.sql.execute_write"]
    assert _blocked_ids(contract) == ["db.sql.execute_write"]
    assert contract.metadata["missing_capabilities"] == []
    assert contract.metadata["approval_required"] is True


def test_monitor_execute_contract_reports_missing_executor_capability():
    frame, contract = _contract("run the revenue monitor now")

    assert [lane.value for lane in frame.granted_lanes] == ["monitor_execute"]
    assert contract.operation_type == "monitor.execute"
    assert contract.access is AccessMode.WRITE
    assert contract.required_capabilities == ()
    assert contract.metadata["required_capabilities"] == ["db.monitor.execute"]
    assert contract.metadata["missing_capabilities"] == ["db.monitor.execute"]
    assert contract.metadata["forbidden_capabilities"] == [
        "db.monitor.commit_create",
        "db.monitor.commit_lifecycle",
        "db.sql.execute_write",
    ]
    assert _blocked_ids(contract) == [
        "db.monitor.commit_create",
        "db.monitor.commit_lifecycle",
        "db.sql.execute_write",
    ]
    assert contract.metadata["approval_required"] is True
