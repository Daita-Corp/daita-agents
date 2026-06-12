import json
from uuid import uuid4

from daita.agents.agent import Agent
from daita.db import DbRuntime
from daita.db.llm_service import DbLLMResponse
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import Evidence, OperationStatus, TaskStatus


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


async def test_verified_query_persists_synthesis_task_and_final_answer_from_evidence(
    tmp_path,
):
    llm = FakeSynthesisLLMService(_valid_response("There are 3 orders."))
    runtime, sqlite = await _runtime(tmp_path, llm)
    calls = []
    original = runtime.execute_task

    async def spy(task, operation, context=None):
        calls.append(task.capability_id)
        return await original(task, operation, context)

    runtime.execute_task = spy
    try:
        result = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert "db.answer.synthesize" in calls
    assert result.answer == "There are 3 orders."
    synthesis = next(
        item for item in result.evidence if item.kind == "answer.synthesis"
    )
    assert synthesis.accepted
    assert result.answer == synthesis.payload["answer"]
    assert snapshot is not None
    task = next(
        item for item in snapshot.tasks if item.capability_id == "db.answer.synthesize"
    )
    assert task.status is TaskStatus.SUCCEEDED


async def test_synthesis_task_depends_on_accepted_evidence_fingerprints(tmp_path):
    llm = FakeSynthesisLLMService(_valid_response("There are 3 orders."))
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert snapshot is not None
    task = next(
        item for item in snapshot.tasks if item.capability_id == "db.answer.synthesize"
    )
    dependencies = {
        dependency.evidence_kind: dependency for dependency in task.dependencies
    }
    assert {
        "query.result",
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "verification.result",
    } <= set(dependencies)
    evidence_by_id = {item.id: item for item in snapshot.evidence}
    for dependency in dependencies.values():
        evidence = evidence_by_id[dependency.evidence_id]
        assert evidence.accepted
        assert evidence.operation_id == result.operation_id
        assert (
            dependency.payload_fingerprint == evidence.metadata["payload_fingerprint"]
        )


async def test_llm_context_uses_accepted_dependency_evidence_only(tmp_path):
    llm = FakeSynthesisLLMService(_valid_response("There are 3 orders."))
    runtime, sqlite = await _runtime(tmp_path, llm)
    original = runtime._execute_answer_synthesis

    async def inject_noise(*, operation, intent, outcome_evidence):
        await runtime.store.save_evidence(
            Evidence(
                id="rejected-noise",
                kind="query.result",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=False,
                payload={"rows": [{"count": 999}], "sql": "SELECT 999"},
                metadata={"payload_fingerprint": "rejected"},
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="other-operation-noise",
                kind="query.result",
                owner="db_runtime",
                operation_id="other-operation",
                accepted=True,
                payload={"rows": [{"count": 123}], "sql": "SELECT 123"},
                metadata={"payload_fingerprint": "other"},
            )
        )
        return await original(
            operation=operation,
            intent=intent,
            outcome_evidence=outcome_evidence,
        )

    runtime._execute_answer_synthesis = inject_noise
    try:
        await runtime.run("How many orders are there?")
    finally:
        await sqlite.disconnect()

    context = _context_from_messages(llm.calls[-1])
    evidence_ids = {item["id"] for item in context["evidence"]}
    assert "rejected-noise" not in evidence_ids
    assert "other-operation-noise" not in evidence_ids
    assert {item["operation_id"] for item in context["evidence"]} != {"other-operation"}


async def test_no_llm_uses_deterministic_fallback_through_synthesis_evidence(tmp_path):
    runtime, sqlite = await _runtime(tmp_path, None)
    try:
        result = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.answer == "The count is 3."
    synthesis = next(
        item for item in result.evidence if item.kind == "answer.synthesis"
    )
    assert synthesis.payload["diagnostics"]["mode"] == "deterministic_fallback"
    assert synthesis.payload["diagnostics"]["fallback_reason"] == (
        "db_llm_service_unavailable"
    )
    assert snapshot is not None
    assert any(task.capability_id == "db.answer.synthesize" for task in snapshot.tasks)


async def test_invalid_llm_outputs_fallback_with_reason(tmp_path):
    cases = [
        ("not json", "JSONDecodeError"),
        (
            lambda messages: json.dumps(
                {
                    "answer": "Unknown citation.",
                    "reasoning_summary": "",
                    "cited_evidence_refs": [
                        {"id": "missing", "kind": "query.result", "purpose": "bad"}
                    ],
                    "assumptions": [],
                    "limitations": [],
                    "warnings": [],
                    "follow_up_questions": [],
                    "sufficiency": "answered",
                    "confidence": 0.5,
                    "truncation": _context_from_messages(messages)["truncation"],
                    "grounding": {"all_claims_from_evidence": True},
                }
            ),
            "unknown_citation",
        ),
        (
            lambda messages: json.dumps(
                {
                    "answer": "I need to execute SQL to answer this.",
                    "reasoning_summary": "",
                    "cited_evidence_refs": _citations(
                        _context_from_messages(messages),
                        "query.result",
                        "verification.result",
                    ),
                    "assumptions": [],
                    "limitations": [],
                    "warnings": [],
                    "follow_up_questions": [],
                    "sufficiency": "answered",
                    "confidence": 0.5,
                    "truncation": _context_from_messages(messages)["truncation"],
                    "grounding": {"all_claims_from_evidence": True},
                }
            ),
            "requests_db_work",
        ),
        (
            lambda messages: json.dumps(
                {
                    "answer": "Orders are joined through a customer relationship.",
                    "reasoning_summary": "",
                    "cited_evidence_refs": _citations(
                        _context_from_messages(messages),
                        "query.result",
                        "verification.result",
                    ),
                    "assumptions": [],
                    "limitations": [],
                    "warnings": [],
                    "follow_up_questions": [],
                    "sufficiency": "answered",
                    "confidence": 0.5,
                    "truncation": _context_from_messages(messages)["truncation"],
                    "grounding": {"all_claims_from_evidence": True},
                }
            ),
            "ungrounded_relationship_claim",
        ),
    ]
    for responder, expected_reason in cases:
        llm = FakeSynthesisLLMService(
            responder
            if callable(responder)
            else lambda messages, value=responder: value
        )
        runtime, sqlite = await _runtime(tmp_path, llm)
        try:
            result = await runtime.run("How many orders are there?")
        finally:
            await sqlite.disconnect()
        synthesis = next(
            item for item in result.evidence if item.kind == "answer.synthesis"
        )
        assert synthesis.payload["diagnostics"]["mode"] == "deterministic_fallback"
        assert expected_reason in synthesis.payload["diagnostics"]["fallback_reason"]


async def test_dropped_caveats_and_redaction_force_fallback(tmp_path):
    def dropped_caveats(messages):
        context = _context_from_messages(messages)
        return json.dumps(
            {
                "answer": "Returned a sample of orders.",
                "reasoning_summary": "Used rows.",
                "cited_evidence_refs": _citations(
                    context, "query.result", "verification.result"
                ),
                "assumptions": [],
                "limitations": [],
                "warnings": [],
                "follow_up_questions": [],
                "sufficiency": "answered",
                "confidence": 0.7,
                "truncation": context["truncation"],
                "grounding": {"all_claims_from_evidence": True},
            }
        )

    llm = FakeSynthesisLLMService(dropped_caveats)
    runtime, sqlite = await _runtime(tmp_path, llm, query_max_rows=1)
    try:
        result = await runtime.run("List orders")
    finally:
        await sqlite.disconnect()

    context = _context_from_messages(llm.calls[-1])
    assert "query_result_truncated" in context["required_caveats"]
    assert "sensitive_values_redacted" in context["required_caveats"]
    assert "[REDACTED]" in json.dumps(context)
    synthesis = next(
        item for item in result.evidence if item.kind == "answer.synthesis"
    )
    assert synthesis.payload["diagnostics"]["mode"] == "deterministic_fallback"
    assert "dropped_caveat" in synthesis.payload["diagnostics"]["fallback_reason"]


async def test_simple_answers_stay_deterministic_for_empty_and_single_row(tmp_path):
    runtime, sqlite = await _runtime(tmp_path, None)
    try:
        count = await runtime.run("How many orders are there?")
        empty = await runtime.run("List orders where id > 99")
        single = await runtime.run("List orders where id = 1")
    finally:
        await sqlite.disconnect()

    assert count.answer == "The count is 3."
    assert empty.answer == "The query returned no rows."
    assert single.answer == "Returned 1 row."


async def test_synthesis_diagnostics_and_resume_skip_completed_task(tmp_path):
    llm = FakeSynthesisLLMService(_valid_response("There are 3 orders."))
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(result.operation_id)
        resumed = await runtime.resume_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    synthesis = next(
        item for item in result.evidence if item.kind == "answer.synthesis"
    )
    diagnostics = synthesis.payload["diagnostics"]
    assert diagnostics["mode"] == "llm"
    assert diagnostics["provider"] == "fake"
    assert diagnostics["model"] == "synthesis-test"
    assert diagnostics["input_tokens"] == 12
    assert diagnostics["output_tokens"] == 8
    assert diagnostics["estimated_cost"] == 0.002
    assert diagnostics["latency_ms"] == 2.5
    assert diagnostics["evidence_refs"]
    assert diagnostics["context"]["context_budget"]["row_budget"] == 25
    assert diagnostics["sufficiency"] == "answered"
    assert diagnostics["fallback_reason"] is None
    assert len(llm.calls) == 1
    assert snapshot is not None
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    before = [
        task.id
        for task in snapshot.tasks
        if task.capability_id == "db.answer.synthesize"
    ]
    after = [
        task.id
        for task in resumed.tasks
        if task.capability_id == "db.answer.synthesize"
    ]
    assert before == after
