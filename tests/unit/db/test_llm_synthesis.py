import json
from uuid import uuid4

import pytest

from daita.agents.agent import Agent
from daita.db import DbIntent, DbIntentKind, DbOperationContract, DbRequest, DbRuntime
from daita.db.llm_service import DbLLMResponse
from daita.db.synthesis import build_synthesis_context, validate_synthesis_payload
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode, Evidence, OperationStatus, TaskStatus


class FakeSynthesisLLMService:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.safe_metadata = {"provider": "fake", "model": "synthesis-test"}

    @property
    def available(self):
        return True

    async def generate_json(self, messages):
        return await self.generate_synthesis_json(messages)

    async def generate_synthesis_json(self, messages):
        self.calls.append(messages)
        content = self.responder(messages)
        return DbLLMResponse(
            content=content,
            diagnostics={
                "provider": "fake",
                "model": "synthesis-test",
                "tokens": {"prompt_tokens": 12, "completion_tokens": 8},
                "estimated_cost_usd": 0.002,
                "latency_ms": 2.5,
            },
        )


async def _runtime(tmp_path, llm_service=None, **sqlite_options):
    db_path = tmp_path / f"llm_synthesis_{uuid4().hex}.sqlite"
    sqlite = SQLitePlugin(path=str(db_path), **sqlite_options)
    await sqlite.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            email TEXT
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_email TEXT,
            total REAL NOT NULL
        );
        INSERT INTO orders (id, customer_email, total)
        VALUES
            (1, 'ada@example.com', 10.0),
            (2, 'linus@example.com', 20.0),
            (3, 'grace@example.com', 30.0);
        """)
    runtime = DbRuntime(
        plugins=(CatalogPlugin(auto_persist=False), sqlite),
        db_llm_service=llm_service,
    )
    await runtime.setup()
    return runtime, sqlite


def _context_from_messages(messages):
    return json.loads(messages[-1]["content"])["context"]


def _citations(context, *kinds):
    citations = []
    for kind in kinds:
        evidence = next(item for item in context["evidence"] if item["kind"] == kind)
        citations.append(
            {
                "id": evidence["id"],
                "kind": kind,
                "purpose": f"cite {kind}",
            }
        )
    return citations


def _data_intent():
    return DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ)


def _data_contract():
    return DbOperationContract(
        operation_type="data.query",
        required_evidence=("sql.validation", "query.result"),
    )


def _query_result(rows, *, sql="SELECT COUNT(*) AS customer_count FROM customers"):
    return Evidence(
        id="query-result-1",
        kind="query.result",
        accepted=True,
        task_id="task-2",
        payload={"rows": rows, "sql": sql, "truncated": False},
    )


def _verification_result():
    return Evidence(
        id="verification-1",
        kind="verification.result",
        accepted=True,
        task_id="task-3",
        payload={
            "passed": True,
            "warnings": [],
            "evidence_details": [
                {
                    "id": "query-result-1",
                    "kind": "query.result",
                    "owner": None,
                    "task_id": "task-2",
                }
            ],
        },
    )


def _data_synthesis_context(*, char_budget=16000):
    evidence = (
        _query_result([{"customer_count": 4}]),
        _verification_result(),
    )
    context = build_synthesis_context(
        request=DbRequest("How many customers are there?"),
        intent=_data_intent(),
        contract=_data_contract(),
        evidence=evidence,
        char_budget=char_budget,
    )
    return context, evidence


def _parsed_synthesis_response(context, answer, *, sufficiency="answered"):
    return {
        "answer": answer,
        "reasoning_summary": "Used the verified query result.",
        "cited_evidence_refs": _citations(
            context, "query.result", "verification.result"
        ),
        "assumptions": [],
        "limitations": [],
        "warnings": [],
        "follow_up_questions": [],
        "sufficiency": sufficiency,
        "confidence": 0.91,
        "truncation": context["truncation"],
        "grounding": {"all_claims_from_evidence": True},
        "answer_facts": context["answer_facts"],
    }


def _valid_response(answer):
    def responder(messages):
        context = _context_from_messages(messages)
        caveats = context.get("required_caveats") or []
        return json.dumps(
            {
                "answer": answer,
                "reasoning_summary": "Used the verified query result.",
                "cited_evidence_refs": _citations(
                    context, "query.result", "verification.result"
                ),
                "assumptions": [],
                "limitations": caveats,
                "warnings": caveats,
                "follow_up_questions": [],
                "sufficiency": "partial" if caveats else "answered",
                "confidence": 0.91,
                "truncation": context["truncation"],
                "grounding": {"all_claims_from_evidence": True},
            }
        )

    return responder


def _valid_schema_response(answer):
    def responder(messages):
        context = _context_from_messages(messages)
        caveats = context.get("required_caveats") or []
        return json.dumps(
            {
                "answer": answer,
                "reasoning_summary": "Used accepted schema evidence.",
                "cited_evidence_refs": _citations(
                    context, "schema.asset_profile", "verification.result"
                ),
                "assumptions": [],
                "limitations": caveats,
                "warnings": caveats,
                "follow_up_questions": [],
                "sufficiency": "partial" if caveats else "answered",
                "confidence": 0.88,
                "truncation": context["truncation"],
                "grounding": {"all_claims_from_evidence": True},
            }
        )

    return responder


async def test_from_db_model_registers_answer_synthesis_capability(tmp_path):
    db_path = tmp_path / "from_db_answer_synthesis.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    await sqlite.disconnect()

    agent = await Agent.from_db(str(db_path), model="mock-model", llm_provider="mock")
    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert "db_runtime:db.answer.synthesize" in inspection.capability_ids
    assert "db_runtime:answer.synthesis" in inspection.evidence_schema_kinds
    assert "db_runtime:verification.result" in inspection.evidence_schema_kinds


def test_llm_synthesis_validation_rejects_answer_missing_scalar_fact():
    context, evidence = _data_synthesis_context()
    parsed = _parsed_synthesis_response(context.payload, "Returned 1 row.")

    with pytest.raises(
        ValueError,
        match="synthesis_missing_answer_fact:customer_count",
    ):
        validate_synthesis_payload(
            parsed,
            dependency_evidence=evidence,
            context_metadata=context.metadata,
            llm_diagnostics={"provider": "fake", "model": "synthesis-test"},
        )


def test_llm_synthesis_validation_accepts_scalar_with_context_only_truncation():
    context, evidence = _data_synthesis_context(char_budget=1)
    parsed = _parsed_synthesis_response(
        context.payload,
        "customer_count is 4.",
        sufficiency="partial",
    )

    payload = validate_synthesis_payload(
        parsed,
        dependency_evidence=evidence,
        context_metadata=context.metadata,
        llm_diagnostics={"provider": "fake", "model": "synthesis-test"},
    )

    assert context.metadata["truncation"]["context_chars_truncated"] is True
    assert payload.sufficiency == "answered"
    assert payload.answer_facts["primary_scalar"]["value"] == 4
    assert "synthesis_context_truncated" not in payload.limitations
    assert "synthesis_context_truncated" not in payload.warnings
