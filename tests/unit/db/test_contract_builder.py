from daita.db import DbIntentKind, DbRequest, DbRuntime, DbRuntimeConfig
from daita.core.exceptions import SkillError
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode
from daita.skills import Skill, SkillActivationRules, SkillRuntimeEffects
import pytest


def _runtime() -> DbRuntime:
    return DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "contract-builder:sqlite",
                    "catalog_profile_key": "contract-builder:sqlite",
                    "source_options": {"include_sample_values": False},
                }
            }
        ),
        plugins=(
            CatalogPlugin(auto_persist=False),
            SQLitePlugin(path=":memory:"),
        ),
    )


def test_missing_capability_diagnostics_explain_unavailable_services():
    runtime = DbRuntime(plugins=(SQLitePlugin(path=":memory:"),))
    request = DbRequest("Check data quality for the orders table", mode="quality.check")

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.QUALITY_CHECK
    assert contract.required_capabilities == ()
    assert contract.required_evidence == ()
    assert contract.metadata["missing_capabilities"] == ["quality.profile"]


def test_conversational_contract_requires_no_capabilities_or_evidence():
    runtime = _runtime()

    contract = runtime.build_contract(DbRequest("Hello there"))

    assert contract.operation_type == "conversational"
    assert contract.access is AccessMode.NONE
    assert contract.required_capabilities == ()
    assert contract.required_evidence == ()
    assert contract.metadata["missing_capabilities"] == []


def test_requested_capabilities_are_included_when_allowed():
    runtime = _runtime()
    request = DbRequest(
        "How many orders are there?",
        requested_capabilities=("catalog.schema.search",),
    )

    contract = runtime.build_contract(request)

    assert "catalog.schema.search" in contract.required_capabilities
    selected = {
        (item["id"], item["reason"])
        for item in contract.metadata["selected_capabilities"]
    }
    assert ("catalog.schema.search", "requested") in selected


def test_skill_requested_capabilities_and_evidence_are_included_in_contract():
    skill = Skill(
        name="finance",
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_finance",
            requested_capabilities=("catalog.schema.search",),
            required_evidence=("schema.asset_profile",),
            policy_ids=("skill_finance:aggregate_only",),
            contract_metadata={"planning_hints": {"prefer_aggregate_queries": True}},
            verifier_metadata={"checks": ["sql.validation"]},
            synthesis_metadata={"style": "finance_summary"},
        ),
    )
    runtime = DbRuntime(
        plugins=(
            CatalogPlugin(auto_persist=False),
            SQLitePlugin(path=":memory:"),
            skill,
        )
    )
    request = DbRequest("How many orders are there?")

    skill_resolution = runtime._resolve_skills(request)
    contract = runtime.build_contract(request, skill_resolution=skill_resolution)

    assert "catalog.schema.search" in contract.required_capabilities
    assert "schema.asset_profile" in contract.required_evidence
    assert "skill_finance:aggregate_only" in contract.policy_ids
    assert contract.metadata["planning_hints"]["prefer_aggregate_queries"] is True
    assert contract.metadata["skill_verifier_metadata"]["checks"] == ["sql.validation"]
    assert contract.metadata["skill_synthesis_metadata"]["style"] == "finance_summary"
    selected = {
        (item["id"], item["reason"])
        for item in contract.metadata["selected_capabilities"]
    }
    assert ("catalog.schema.search", "skill_requested") in selected


def test_build_contract_includes_registered_skill_effects_by_default():
    skill = Skill(
        name="finance",
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_finance",
            requested_capabilities=("catalog.schema.search",),
            contract_metadata={"planning_hints": {"finance": True}},
        ),
    )
    runtime = DbRuntime(
        plugins=(
            CatalogPlugin(auto_persist=False),
            SQLitePlugin(path=":memory:"),
            skill,
        )
    )

    contract = runtime.build_contract(DbRequest("How many orders are there?"))

    assert "catalog.schema.search" in contract.required_capabilities
    assert contract.metadata["planning_hints"] == {"finance": True}


def test_build_contract_include_skills_false_produces_baseline_contract():
    skill = Skill(
        name="finance",
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_finance",
            requested_capabilities=("catalog.schema.search",),
            contract_metadata={"planning_hints": {"finance": True}},
        ),
    )
    runtime = DbRuntime(
        plugins=(
            CatalogPlugin(auto_persist=False),
            SQLitePlugin(path=":memory:"),
            skill,
        )
    )

    contract = runtime.build_contract(
        DbRequest("How many orders are there?"),
        include_skills=False,
    )

    assert "catalog.schema.search" not in contract.required_capabilities
    assert "planning_hints" not in contract.metadata


async def test_run_and_build_contract_agree_on_skill_contract_metadata():
    skill = Skill(
        name="finance",
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_finance",
            contract_metadata={"planning_hints": {"finance": True}},
        ),
    )
    runtime = DbRuntime(plugins=(skill,))
    request = DbRequest("Hello there")

    try:
        built = runtime.build_contract(request)
        result = await runtime.run(request)
    finally:
        await runtime.teardown()

    assert built.metadata["skill_contract_metadata"] == (
        result.contract.metadata["skill_contract_metadata"]
    )


def test_unknown_db_skill_selection_raises_clear_error():
    runtime = DbRuntime(plugins=(Skill(name="finance"),))

    with pytest.raises(SkillError, match="Unknown skill selection\\(s\\): finacne"):
        runtime.build_contract(
            DbRequest("Hello there", metadata={"skills": ["finacne"]})
        )


def test_db_skill_mode_rules_gate_contract_effects():
    skill = Skill(
        name="schema_style",
        activation_rules=SkillActivationRules(
            always_on=True,
            modes=("schema.query",),
        ),
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_schema_style",
            contract_metadata={"planning_hints": {"schema_style": True}},
        ),
    )
    runtime = DbRuntime(plugins=(skill,))

    selected = runtime.build_contract(
        DbRequest("What columns are available?", mode="schema.query")
    )
    skipped = runtime.build_contract(
        DbRequest("How many orders are there?", mode="data.query")
    )

    assert selected.metadata["planning_hints"] == {"schema_style": True}
    assert "planning_hints" not in skipped.metadata


async def test_db_skill_activation_records_events_and_creates_no_activation_tasks():
    skill = Skill(
        name="quality_gate",
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_quality_gate",
            requested_capabilities=("quality.profile",),
            required_evidence=("quality.profile",),
            contract_metadata={"planning_hints": {"quality_first": True}},
        ),
    )
    runtime = _runtime()
    runtime.register_plugin(skill)

    try:
        result = await runtime.run("Hello there")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.contract.metadata["missing_capabilities"] == ["quality.profile"]
    assert snapshot is not None
    assert snapshot.tasks == ()
    assert snapshot.operation.metadata["skills"]["selected"][0]["skill_id"] == (
        "skill_quality_gate"
    )
    diagnostics = [event.payload.get("diagnostic") for event in snapshot.events]
    assert "skill.selected" in diagnostics
    assert "skill.contract_modified" in diagnostics


async def test_prompt_only_db_skill_records_selection_without_contract_modified():
    runtime = _runtime()
    runtime.register_plugin(
        Skill(
            name="prompt_only",
            instructions="Answer using the house style.",
            context_mode="always",
        )
    )

    try:
        result = await runtime.run("Hello there")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    diagnostics = [event.payload.get("diagnostic") for event in snapshot.events]
    assert "skill.selected" in diagnostics
    assert "skill.contract_modified" not in diagnostics
