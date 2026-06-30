from daita.agents.agent import Agent
from daita.plugins.sqlite import SQLitePlugin


async def test_from_db_model_config_registers_llm_planning_capability(tmp_path):
    db_path = tmp_path / "from_db_llm.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script("CREATE TABLE customers (id INTEGER PRIMARY KEY)")
    await sqlite.disconnect()

    agent = await Agent.from_db(str(db_path), model="mock-model", llm_provider="mock")
    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert "db_runtime:db.query.plan" in inspection.capability_ids
    assert "db_runtime:query.plan.proposal" in inspection.evidence_schema_kinds
    assert inspection.metadata["from_db_options"]["model"] == "mock-model"
    assert "api_key" not in inspection.metadata["from_db_options"]
