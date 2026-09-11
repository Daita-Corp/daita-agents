__all__ = (
    "NOW",
    "_HISTORY_OMISSION_MARKER",
    "_MAXIMUM_PRIOR_UTF8_BYTES",
    "Agent",
    "AgentContextBuilder",
    "AgentLoop",
    "CanonicalMessage",
    "CatalogSpy",
    "EmbeddedAgent",
    "FinishReason",
    "FreshQueryTools",
    "InMemoryTranscriptStore",
    "LoopExit",
    "LoopExitKind",
    "Mapping",
    "MessageRole",
    "MockModelProvider",
    "ModelProfile",
    "ModelProviderError",
    "ModelRequest",
    "ModelResponse",
    "NoTools",
    "ProviderErrorCode",
    "ReplayTools",
    "RunInput",
    "SQLiteStateStore",
    "TextBlock",
    "ToolCall",
    "ToolDefinition",
    "ToolResultBlock",
    "TranscriptContext",
    "_analytical_conversation_record",
    "_conversation_record",
    "_neutral_message",
    "_prepared_request",
    "_profile",
    "_project_completed_history",
    "_request_text",
    "_simple_conversation_record",
    "_stop",
    "_tool_response",
    "canonical_json",
    "fields",
    "inspect",
    "pytest",
    "workspace_for",
)

import inspect
from collections.abc import Mapping
from dataclasses import fields
from datetime import UTC, datetime

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.context import (
    _HISTORY_OMISSION_MARKER,
    _MAXIMUM_PRIOR_UTF8_BYTES,
    AgentContextBuilder,
    _neutral_message,
    _project_completed_history,
)
from daita.hosting.embedded import EmbeddedAgent
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop import (
    AgentLoop,
    ConversationRun,
    InMemoryTranscriptStore,
    LoopExit,
    LoopExitKind,
    RunInput,
    ToolBatchOutcome,
    Transcript,
)
from daita.storage.sqlite import SQLiteStateStore
from tests.support.capability_runtime import ContextToolProjectionAdapter
from tests.support.workspace import workspace_for

NOW = datetime(2026, 7, 21, tzinfo=UTC)


async def _prepared_request(
    builder: AgentContextBuilder,
    run: RunInput,
    messages: tuple[CanonicalMessage, ...],
    tools: tuple[ToolDefinition, ...],
    *,
    step: int,
) -> ModelRequest:
    current_start = max(
        index
        for index, message in enumerate(messages)
        if message.role is MessageRole.USER
    )
    projection = ContextToolProjectionAdapter(tools)
    catalog = await projection.prepare_run(run)
    snapshot = await builder.prepare(
        run,
        messages[: current_start + 1],
        catalog,
    )
    return builder.project(
        snapshot,
        messages[current_start:],
        step=step,
        tool_context=projection.project(catalog, messages[current_start:]),
    )


class TranscriptContext:
    async def prepare(self, run, messages, tool_context, *, max_total_tokens=None):
        del run
        return messages[:-1], tool_context.initial_provider_definitions

    def project(
        self,
        snapshot,
        messages,
        *,
        step,
        tool_context,
        previous_request_input_tokens=None,
        remaining_tokens=None,
        request_input_growth_tokens=None,
        remaining_steps=None,
    ):
        del step, previous_request_input_tokens, tool_context
        static, tools = snapshot
        return ModelRequest(
            messages=(*static, *messages),
            tools=tools,
        )


class NoTools:
    def __init__(self) -> None:
        self._projection = ContextToolProjectionAdapter(())

    async def prepare_run(self, run):
        return await self._projection.prepare_run(run)

    def project(self, catalog, messages):
        return self._projection.project(catalog, messages)

    async def execute_all(self, run, calls, *, projection, messages, sensitivity):
        del run, projection, messages, sensitivity
        assert calls == ()
        return ToolBatchOutcome(())


class ReplayTools:
    def __init__(self) -> None:
        self._projection = ContextToolProjectionAdapter(
            tuple(
                ToolDefinition(
                    name=name,
                    description=name,
                    input_schema={"type": "object"},
                )
                for name in ("memory_set", "skill_save", "skill_delete", "skill_view")
            )
        )

    async def prepare_run(self, run):
        return await self._projection.prepare_run(run)

    def project(self, catalog, messages):
        return self._projection.project(catalog, messages)

    async def execute_all(self, run, calls, *, projection, messages, sensitivity):
        del run, projection, messages, sensitivity
        return ToolBatchOutcome(
            tuple(
                ToolResultBlock(
                    call_id=call.id,
                    output=(
                        {
                            "kind": "skill.document",
                            "data": {
                                "name": "secret-skill",
                                "instructions": "SECRET SKILL BODY",
                            },
                        }
                        if call.name == "skill_view"
                        else {
                            "kind": {
                                "memory_set": "memory.replacement",
                                "skill_save": "skill.saved",
                                "skill_delete": "skill.deleted",
                            }[call.name],
                            "data": {"ok": True},
                        }
                    ),
                )
                for call in calls
            )
        )


class FreshQueryTools:
    def __init__(self):
        self.calls = []
        self._projection = ContextToolProjectionAdapter(
            (
                ToolDefinition(
                    name="data_query",
                    description="Run a fresh read-only PostgreSQL query.",
                    input_schema={"type": "object"},
                ),
            )
        )

    async def prepare_run(self, run):
        return await self._projection.prepare_run(run)

    def project(self, catalog, messages):
        return self._projection.project(catalog, messages)

    async def execute_all(self, run, calls, *, projection, messages, sensitivity):
        del run, projection, messages, sensitivity
        self.calls.extend(calls)
        return ToolBatchOutcome(
            tuple(
                ToolResultBlock(
                    call_id=call.id,
                    output={
                        "kind": "data.query_result",
                        "data": {
                            "columns": ["segment", "paid_revenue"],
                            "rows": [{"segment": "enterprise", "paid_revenue": 321}],
                            "total_rows": 1,
                            "returned_rows": 1,
                            "truncated": False,
                            "resource_revisions": [
                                {
                                    "resource_id": "orders",
                                    "revision": "sha256:" + ("d" * 64),
                                }
                            ],
                            "source_id": "warehouse",
                            "source_revision": "catalog:current",
                        },
                    },
                )
                for call in calls
            )
        )


class CatalogSpy:
    async def source_routing_facts(self, agent_id, source_ids=()):
        ids = {item.get("source_id", "source-history") for item in self.resources} or {
            "source-history"
        }
        return tuple(
            {"source_id": item, "adapter_id": "sqlite"}
            for item in ids
            if not source_ids or item in source_ids
        )

    async def readable_resource_ids(self, agent_id, source_ids=()):
        return frozenset(item["resource_id"] for item in self.resources) or frozenset(
            {"resource-unmatched"}
        )

    def __init__(self, resources=(), sources=()):
        self.queries = []
        self.resources = resources
        self.sources = sources

    async def admitted_model_sensitivity(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> ModelSensitivity:
        del agent_id, source_ids
        return ModelSensitivity.PUBLIC

    async def catalog_context(
        self,
        agent_id,
        query,
        *,
        prior_query=None,
        limit,
        source_ids=(),
        resource_ids=(),
    ):
        del agent_id, limit, source_ids, resource_ids
        from daita._json import FrozenJsonObject

        self.queries.append((query, prior_query))
        return FrozenJsonObject.from_mapping(
            {
                "resources": self.resources,
                "sources": self.sources,
                "total_matches": len(self.resources),
                "returned_count": len(self.resources),
                "truncated": False,
                "trust_classification": "untrusted_external_data",
            }
        )


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _tool_response(run_number: int) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(
            ToolCall(
                id=f"shared-{name}",
                name=name,
                arguments={"content": f"SECRET {name} {run_number}"},
                provider_call_id=f"native-{name}",
            )
            for name in ("memory_set", "skill_save", "skill_delete", "skill_view")
        ),
        provider_id="mock:scripted",
        provider_metadata={"continuation": {"run": run_number}},
    )


def _conversation_record(
    index: int,
    messages: tuple[CanonicalMessage, ...],
    *,
    kind: LoopExitKind = LoopExitKind.COMPLETED,
) -> ConversationRun:
    run = RunInput(
        id=f"history-run-{index}",
        agent_id="agent-history",
        message=f"history user {index}",
        created_at=NOW,
        conversation_id="history-conversation",
    )
    result = LoopExit(
        run_id=run.id,
        conversation_id="history-conversation",
        kind=kind,
        reason=kind.value,
        created_at=NOW,
        final_text=(
            f"history answer {index}" if kind is LoopExitKind.COMPLETED else None
        ),
    )
    return ConversationRun(
        turn_index=index,
        transcript=Transcript(run=run, messages=messages),
        result=result,
    )


def _simple_conversation_record(index: int, answer: str | None = None):
    return _conversation_record(
        index,
        (
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(f"history user {index}"),),
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock(answer or f"history answer {index}"),),
            ),
        ),
    )


def _analytical_conversation_record(
    *,
    oversized: bool,
) -> tuple[ConversationRun, tuple[CanonicalMessage, ...]]:
    user_text = (
        "Analyze captured payments across all dates, grouped by region. Define paid "
        "revenue as tax-exclusive merchandise revenue from captured payments, paid "
        "order count as distinct paid orders, AOV as paid revenue divided by paid "
        "order count, COGS as merchandise unit cost times quantity, and gross margin "
        "as paid revenue minus COGS divided by paid revenue."
    )
    catalog_call = ToolCall(
        id="catalog-native-call",
        name="catalog_inspect",
        arguments={"resource_id": "orders"},
        provider_call_id="provider-catalog-call",
    )
    query_call = ToolCall(
        id="query-native-call",
        name="data_query",
        arguments={
            "source_id": "warehouse",
            "resource_ids": ("warehouse:captured_payments",),
            "sql": (
                "SELECT region, "
                "SUM(quantity * merchandise_unit_price) AS paid_revenue, "
                "COUNT(DISTINCT order_id) AS paid_order_count, "
                "SUM(quantity * merchandise_unit_price) / "
                "COUNT(DISTINCT order_id) AS aov, "
                "SUM(quantity * merchandise_unit_cost) AS cogs, "
                "(SUM(quantity * merchandise_unit_price) - "
                "SUM(quantity * merchandise_unit_cost)) / "
                "SUM(quantity * merchandise_unit_price) AS gross_margin "
                "FROM captured_payments JOIN order_items USING (order_id) "
                "GROUP BY region"
            ),
        },
        provider_call_id="provider-query-call",
    )
    raw_rows = [
        {
            "region": f"region-{index}",
            "paid_revenue": 123_456 + index,
            "paid_order_count": 50 + index,
            "padding": "raw-row-sentinel-" + ("x" * 700),
        }
        for index in range(40 if oversized else 1)
    ]
    messages = (
        CanonicalMessage(role=MessageRole.USER, content=(TextBlock(user_text),)),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(catalog_call,),
            provider_id="mock:history",
            provider_metadata={"native": "catalog"},
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=catalog_call.id,
                    output={
                        "kind": "catalog.resource_snapshot",
                        "data": {
                            "resource_id": "orders",
                            "schema": "catalog-snapshot-sentinel-" + ("z" * 30_000),
                        },
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(query_call,),
            provider_id="mock:history",
            provider_metadata={"native": "query"},
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=query_call.id,
                    output={
                        "kind": "data.query_result",
                        "data": {
                            "columns": [
                                "region",
                                "paid_revenue",
                                "paid_order_count",
                                "aov",
                                "cogs",
                                "gross_margin",
                            ],
                            "rows": raw_rows,
                            "total_rows": len(raw_rows),
                            "returned_rows": len(raw_rows),
                            "truncated": False,
                            "resource_revisions": [
                                {
                                    "resource_id": "orders",
                                    "revision": "sha256:" + ("a" * 64),
                                },
                                {
                                    "resource_id": "order_items",
                                    "revision": "sha256:" + ("b" * 64),
                                },
                            ],
                            "source_id": "warehouse",
                            "source_revision": "catalog:history",
                            "trust_classification": "untrusted_external_data",
                        },
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(
                TextBlock(
                    "Overall results use captured-payment scope across all dates with "
                    "regional grouping. Paid revenue is tax-exclusive merchandise "
                    "revenue; paid order count is distinct paid orders; AOV is paid "
                    "revenue divided by paid order count; COGS is merchandise unit "
                    "cost times quantity; gross margin is (paid revenue - COGS) / "
                    "paid revenue. Overall paid revenue was 123456. A segment follow-up "
                    "must compare enterprise customers with the overall results."
                ),
            ),
            provider_id="mock:history",
            provider_metadata={"native": "answer"},
        ),
    )
    return _conversation_record(0, messages), messages


def _request_text(request: ModelRequest) -> tuple[str, ...]:
    return tuple(
        block.text
        for message in request.messages
        if message.role.value != "system"
        for block in message.content
        if isinstance(block, TextBlock)
    )
