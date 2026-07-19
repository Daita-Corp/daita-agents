from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject
from daita.catalog import ResourceKind, catalog_resource_id
from daita.learning import LearningProposalState, LearningRejectionCategory
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.loop.models import LoopExitKind
from daita.memory import (
    ExplicitCorrectionNotEligibleError,
    MemoryCreator,
    MemoryInspectionRequest,
    MemoryListRequest,
    MemoryQualification,
    MemoryRecallRequest,
    MemoryRestoreRequest,
    MemoryScope,
    ResourceAliasCorrection,
)

NOW = datetime(2026, 7, 19, 6, 0, tzinfo=timezone.utc)
PROFILE = ModelProfile(
    id="mock:learning-journey",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
)


class AdvancingClock:
    def __init__(self) -> None:
        self._value = NOW

    def __call__(self) -> datetime:
        current = self._value
        self._value += timedelta(milliseconds=1)
        return current


class JourneyProvider:
    provider_id = "mock:learning-journey"

    def __init__(self) -> None:
        self.script: list[ModelResponse] = []
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("unexpected model call")
        return self.script.pop(0)

    def assert_consumed(self) -> None:
        assert self.script == []


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _final(text: str, evidence_number: int) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        text=f"{text} [evidence:evidence-{evidence_number}]",
    )


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL
            );
            INSERT INTO customers (name, status) VALUES
                ('Ada', 'complete'),
                ('Grace', 'complete'),
                ('Linus', 'pending');
            """)


def _script(source_id: str) -> tuple[ModelResponse, ...]:
    count_sql = "SELECT COUNT(*) AS customer_count FROM customers WHERE status = ?"
    return (
        _tool(
            "call-initial-search",
            "catalog_search",
            {"query": "customers", "source_id": source_id, "limit": 5},
        ),
        _tool(
            "call-initial-query",
            "data_query_sqlite",
            {
                "source_id": source_id,
                "sql": count_sql,
                "parameters": ["completed"],
            },
        ),
        _final("There are 0 customers stored as completed.", 2),
        _tool(
            "call-first-correction-query",
            "data_query_sqlite",
            {
                "source_id": source_id,
                "sql": count_sql,
                "parameters": ["complete"],
            },
        ),
        _final("The exact stored value complete matches 2 customers.", 3),
        _tool(
            "call-recalled-query",
            "data_query_sqlite",
            {
                "source_id": source_id,
                "sql": count_sql,
                "parameters": ["complete"],
            },
        ),
        _final("There are 2 completed customers.", 4),
        _tool(
            "call-second-correction-query",
            "data_query_sqlite",
            {
                "source_id": source_id,
                "sql": count_sql,
                "parameters": ["closed"],
            },
        ),
        _final("No customers currently use the exact stored value closed.", 5),
        _tool(
            "call-refresh-search",
            "catalog_search",
            {"query": "customers", "source_id": source_id, "limit": 5},
        ),
        _tool(
            "call-refresh-query",
            "data_query_sqlite",
            {
                "source_id": source_id,
                "sql": count_sql,
                "parameters": ["complete"],
            },
        ),
        _final("There are still 2 complete customer rows after refresh.", 7),
        _tool(
            "call-stale-correction-query",
            "data_query_sqlite",
            {
                "source_id": source_id,
                "sql": count_sql,
                "parameters": ["complete"],
            },
        ),
        _final("The refreshed table still has 2 complete customer rows.", 8),
        _tool(
            "call-pii-correction-query",
            "data_query_sqlite",
            {
                "source_id": source_id,
                "sql": count_sql,
                "parameters": ["complete"],
            },
        ),
        _final("The current table still has 2 complete customer rows.", 9),
    )


def _request_text(request: ModelRequest) -> str:
    return "\n".join(
        block.text
        for message in request.messages
        for block in message.content
        if isinstance(block, TextBlock)
    )


def _memory_context_payloads(text: str) -> tuple[dict[str, object], ...]:
    start = "BEGIN_MEMORY_CONTEXT_JSON"
    end = "END_MEMORY_CONTEXT_JSON"
    payloads: list[dict[str, object]] = []
    for remainder in text.split(start)[1:]:
        encoded, separator, _ = remainder.partition(end)
        assert separator == end
        payload = json.loads(encoded.strip())
        assert isinstance(payload, dict)
        payloads.append(payload)
    return tuple(payloads)


def _query_parameters(snapshot: object) -> tuple[object, ...]:
    tasks = getattr(snapshot, "tasks")
    query = next(task for task in tasks if task.capability_id == "data.sqlite.query")
    parameters = query.arguments["parameters"]
    assert isinstance(parameters, tuple)
    return parameters


def _search_revision(snapshot: object) -> str:
    evidence = getattr(snapshot, "evidence")[0]
    hits = evidence.payload["hits"]
    assert isinstance(hits, tuple) and len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, FrozenJsonObject)
    revision = hit["revision"]
    assert isinstance(revision, str)
    return revision


async def test_public_exact_correction_learning_and_revision_journey(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    root = tmp_path / "state"
    source = SQLiteSource(database, name="Customers")
    provider = JourneyProvider()
    clock = AdvancingClock()
    id_factory = _ids()
    agent = await Agent.create(
        "atlas",
        root=root,
        model=provider,
        model_profile=PROFILE,
        clock=clock,
        id_factory=id_factory,
    )
    registration = await agent.attach(source)
    source_id = registration.id
    resource_id = catalog_resource_id(
        source_id,
        ResourceKind.TABLE,
        "main.customers",
    )
    provider.script.extend(_script(source_id))

    initial = await agent.run(
        "How many completed customers are there?",
        session_id="session-before-correction",
    )
    initial_snapshot = await agent.inspect(initial.operation_id)
    old_revision = _search_revision(initial_snapshot)
    assert initial.kind is LoopExitKind.COMPLETED
    assert _query_parameters(initial_snapshot) == ("completed",)
    assert initial_snapshot.evidence[-1].accepted is True
    assert initial.final_text is not None
    assert "[evidence:evidence-2]" in initial.final_text

    scope = MemoryScope(
        agent_id=agent.id,
        source_id=source_id,
        resource_id=resource_id,
    )
    first_message = ResourceAliasCorrection(
        source_id=source_id,
        resource_id=resource_id,
        resource_revision=old_revision,
        field="status",
        business_term="completed",
        stored_value="complete",
    ).to_trigger_message()
    first_correction = await agent.run(
        first_message,
        session_id="session-first-correction",
    )
    first_snapshot = await agent.inspect(first_correction.operation_id)
    learned = await agent.list_memories(
        MemoryListRequest(
            scope=scope,
            current_resource_revision=old_revision,
        )
    )

    assert first_correction.kind is LoopExitKind.COMPLETED
    assert _query_parameters(first_snapshot) == ("complete",)
    assert first_snapshot.evidence[-1].accepted is True
    assert first_correction.final_text is not None
    assert "[evidence:evidence-3]" in first_correction.final_text
    assert len(learned.items) == 1
    first_memory = learned.items[0]
    assert first_memory.qualification is MemoryQualification.CURRENT
    assert first_memory.snapshot.record.current_version == 1
    assert first_memory.snapshot.version.attributes["stored_value"] == "complete"
    memory_id = first_memory.snapshot.record.id

    recall_request_index = len(provider.requests)
    recalled = await agent.run(
        "How many completed customers are there?",
        session_id="session-after-correction",
    )
    recalled_snapshot = await agent.inspect(recalled.operation_id)
    recalled_context = _request_text(provider.requests[recall_request_index])
    recalled_payloads = _memory_context_payloads(recalled_context)

    assert recalled.kind is LoopExitKind.COMPLETED
    assert _query_parameters(recalled_snapshot) == ("complete",)
    assert _query_parameters(recalled_snapshot) != _query_parameters(initial_snapshot)
    assert recalled_snapshot.evidence[-1].accepted is True
    assert recalled.final_text is not None
    assert "[evidence:evidence-4]" in recalled.final_text
    assert len(recalled_payloads) == 1
    recalled_payload = recalled_payloads[0]
    assert recalled_payload["memory_id"] == memory_id
    assert recalled_payload["attributes"] == {
        "business_term": "completed",
        "field": "status",
        "stored_value": "complete",
    }
    assert recalled_payload["scope"] == {
        "agent_id": agent.id,
        "resource_id": resource_id,
        "session_id": None,
        "source_id": source_id,
        "user_id": None,
    }
    recalled_provenance = recalled_payload["provenance"]
    assert isinstance(recalled_provenance, dict)
    assert recalled_provenance == {
        "content_hash": first_memory.snapshot.version.provenance.content_hash,
        "evidence_id": None,
        "external_ref": None,
        "kind": "user_statement",
        "operation_id": first_correction.operation_id,
        "session_id": "session-first-correction",
        "trigger_id": first_snapshot.trigger.id,
    }
    origin_operation_id = recalled_provenance["operation_id"]
    origin_trigger_id = recalled_provenance["trigger_id"]
    assert isinstance(origin_operation_id, str)
    assert isinstance(origin_trigger_id, str)
    recalled_origin = await agent.inspect(origin_operation_id)
    assert recalled_origin.operation.id == first_correction.operation_id
    assert recalled_origin.trigger.id == origin_trigger_id

    second_message = ResourceAliasCorrection(
        source_id=source_id,
        resource_id=resource_id,
        resource_revision=old_revision,
        field="status",
        business_term="completed",
        stored_value="closed",
    ).to_trigger_message()
    second_correction = await agent.run(
        second_message,
        session_id="session-second-correction",
    )
    second_snapshot = await agent.inspect(second_correction.operation_id)
    version_two = await agent.inspect_memory(
        MemoryInspectionRequest(
            agent_id=agent.id,
            memory_id=memory_id,
            current_resource_revision=old_revision,
        )
    )

    assert second_correction.kind is LoopExitKind.COMPLETED
    assert _query_parameters(second_snapshot) == ("closed",)
    assert second_snapshot.evidence[-1].accepted is True
    assert version_two.qualification is MemoryQualification.CURRENT
    assert version_two.history.record.current_version == 2
    assert [version.version for version in version_two.history.versions] == [1, 2]
    assert [
        version.attributes["stored_value"] for version in version_two.history.versions
    ] == ["complete", "closed"]
    assert version_two.history.versions[1].supersedes_version == 1
    assert (
        version_two.history.versions[0].provenance.operation_id
        == first_correction.operation_id
    )
    assert (
        version_two.history.versions[1].provenance.operation_id
        == second_correction.operation_id
    )

    original = version_two.history.versions[0]
    restored_version = replace(
        original,
        version=3,
        creator=MemoryCreator.USER,
        created_at=clock(),
        supersedes_version=2,
    )
    restored = await agent.restore_memory(
        MemoryRestoreRequest(
            agent_id=agent.id,
            memory_id=memory_id,
            expected_version=2,
            restore_version=1,
            replacement=restored_version,
        )
    )
    assert restored.history.record.current_version == 3
    assert [version.version for version in restored.history.versions] == [1, 2, 3]
    assert restored.history.current.creator is MemoryCreator.USER
    assert restored.history.current.provenance == original.provenance
    for field_name in (
        "content",
        "attributes",
        "confidence",
        "sensitivity",
        "expires_at",
        "resource_revision",
    ):
        assert getattr(restored.history.current, field_name) == getattr(
            original,
            field_name,
        )

    await agent.close()
    agent = await Agent.open(
        "atlas",
        root=root,
        model=provider,
        clock=clock,
        id_factory=id_factory,
    )
    reopened_history = await agent.inspect_memory(
        MemoryInspectionRequest(
            agent_id=agent.id,
            memory_id=memory_id,
            current_resource_revision=old_revision,
        )
    )
    assert reopened_history.history == restored.history
    assert reopened_history.qualification is MemoryQualification.CURRENT

    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE customers ADD COLUMN note TEXT")
    refreshed = await agent.attach(SQLiteSource(database, name="Customers"))
    assert refreshed == registration

    refresh_request_index = len(provider.requests)
    after_refresh = await agent.run(
        "How many completed customers are there after the catalog refresh?",
        session_id="session-after-refresh",
    )
    refresh_snapshot = await agent.inspect(after_refresh.operation_id)
    new_revision = _search_revision(refresh_snapshot)
    stale = await agent.inspect_memory(
        MemoryInspectionRequest(
            agent_id=agent.id,
            memory_id=memory_id,
            current_resource_revision=new_revision,
        )
    )
    excluded = await agent.recall_memory(
        MemoryRecallRequest(
            query="completed customer status",
            scope=scope,
            current_resource_revision=new_revision,
        )
    )

    assert after_refresh.kind is LoopExitKind.COMPLETED
    assert new_revision != old_revision
    assert stale.qualification is MemoryQualification.STALE_REVISION
    assert stale.history == restored.history
    assert excluded.hits == ()
    assert excluded.omitted_by_revision == 1
    refreshed_context = _request_text(provider.requests[refresh_request_index])
    assert "BEGIN_MEMORY_CONTEXT_JSON" not in refreshed_context
    assert memory_id not in refreshed_context

    stale_message = ResourceAliasCorrection(
        source_id=source_id,
        resource_id=resource_id,
        resource_revision=old_revision,
        field="status",
        business_term="completed",
        stored_value="archived",
    ).to_trigger_message()
    stale_operation = await agent.run(
        stale_message,
        session_id="session-stale-correction",
    )
    assert stale_operation.kind is LoopExitKind.COMPLETED
    assert stale_operation.post_operation_notices == ("learning.correction_failed",)
    with pytest.raises(ExplicitCorrectionNotEligibleError, match="not current"):
        await agent.learn_correction(stale_operation.operation_id)

    private_value = "ada@example.com"
    pii_message = ResourceAliasCorrection(
        source_id=source_id,
        resource_id=resource_id,
        resource_revision=new_revision,
        field="status",
        business_term="completed",
        stored_value=private_value,
    ).to_trigger_message()
    pii_operation = await agent.run(
        pii_message,
        session_id="session-pii-correction",
    )
    pii_snapshot = await agent.inspect(pii_operation.operation_id)
    rejected = await agent.learn_correction(pii_operation.operation_id)

    assert pii_operation.kind is LoopExitKind.COMPLETED
    assert pii_operation.post_operation_notices == ()
    assert pii_snapshot.evidence[-1].accepted is True
    assert rejected.replayed is True
    assert rejected.proposal.state is LearningProposalState.REJECTED
    assert (
        rejected.proposal.rejection_category is LearningRejectionCategory.RAW_ROW_OR_PII
    )
    assert rejected.proposal.candidate_payload is None
    assert rejected.memory is None
    assert private_value not in repr(rejected.proposal)

    await agent.close()
    reopened = await Agent.open("atlas", root=root, clock=clock)
    durable_rejection = await reopened.learn_correction(pii_operation.operation_id)
    durable_history = await reopened.inspect_memory(
        MemoryInspectionRequest(
            agent_id=reopened.id,
            memory_id=memory_id,
            current_resource_revision=new_revision,
        )
    )
    await reopened.close()

    assert durable_rejection.replayed is True
    assert durable_rejection.proposal == rejected.proposal
    assert durable_rejection.proposal.candidate_payload is None
    assert durable_rejection.memory is None
    assert durable_history.history == restored.history
    provider.assert_consumed()
