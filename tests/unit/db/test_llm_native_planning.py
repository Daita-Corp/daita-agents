from daita.agents.agent import Agent
from daita.db import DbLLMConfig
from daita.plugins.sqlite import SQLitePlugin


async def test_from_db_model_config_uses_closed_slim_capabilities(tmp_path):
    db_path = tmp_path / "from_db_llm.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script("CREATE TABLE customers (id INTEGER PRIMARY KEY)")
    await sqlite.disconnect()

    agent = await Agent.from_db(
        str(db_path), llm=DbLLMConfig(provider="mock", model="mock-model")
    )
    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert "db_runtime:db.query.plan" not in inspection.capability_ids
    assert "db_runtime:query.plan.proposal" not in inspection.evidence_schema_kinds
    assert inspection.metadata["from_db_options"]["llm"]["model"] == "mock-model"
    assert "api_key" not in inspection.metadata["from_db_options"]
