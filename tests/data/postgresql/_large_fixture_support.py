from __future__ import annotations

__all__ = (
    "ATTACHED_SCHEMAS",
    "FIXTURE",
    "Agent",
    "ApprovalDecision",
    "ApprovalRequest",
    "Decimal",
    "FinishReason",
    "Mapping",
    "MockModelProvider",
    "ModelResponse",
    "Path",
    "SecretReference",
    "ToolCall",
    "_BulkUpdateProvider",
    "_Secrets",
    "_profile",
    "_restore_bulk_priority",
    "_tool_results",
    "os",
    "pytest",
    "update_constraints",
    "workspace_for",
)

import os
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from daita import Agent, ApprovalDecision, ApprovalRequest
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.protocols import ModelProvider
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference
from tests.support.paths import FIXTURES_ROOT, REPO_ROOT
from tests.support.relational_writes import update_constraints
from tests.support.workspace import workspace_for

ROOT = REPO_ROOT
FIXTURE = FIXTURES_ROOT / "postgres-large"
ATTACHED_SCHEMAS = (
    "analytics",
    "archive",
    "billing",
    "catalog",
    "core",
    "sales",
    "support",
)


class _Secrets:
    def __init__(self, password: str) -> None:
        self.password = password
        self.references: list[SecretReference] = []

    async def resolve(self, reference: SecretReference) -> str:
        self.references.append(reference)
        return self.password


def _profile(provider: ModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=4_096,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _tool_results(provider: MockModelProvider) -> tuple[ToolResultBlock, ...]:
    return tuple(
        block
        for request in provider.requests
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )


class _BulkUpdateProvider:
    def __init__(self) -> None:
        self._source_id = ""
        self._resource_id = ""
        self._desired_priority = ""
        self._phase = 0
        self._matched_rows = 0
        self._requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return "mock:postgres-large-bulk-update"

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        return tuple(self._requests)

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    def configure(
        self,
        *,
        source_id: str,
        resource_id: str,
        desired_priority: str,
    ) -> None:
        self._source_id = source_id
        self._resource_id = resource_id
        self._desired_priority = desired_priority
        self._phase = 0
        self._matched_rows = 0

    def _plan(self) -> dict[str, object]:
        return {
            "source_id": self._source_id,
            "resource_id": self._resource_id,
            "where": [
                {
                    "column": "ticket_status",
                    "operator": "eq",
                    "value": "waiting",
                },
                {"column": "category", "operator": "eq", "value": "billing"},
            ],
            "assignments": [{"column": "priority", "value": self._desired_priority}],
        }

    @staticmethod
    def _latest_result(request: ModelRequest, call_id: str) -> ToolResultBlock:
        for message in reversed(request.messages):
            for block in reversed(message.content):
                if isinstance(block, ToolResultBlock) and block.call_id == call_id:
                    return block
        raise AssertionError(f"missing tool result {call_id}")

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self._requests.append(request)
        if self._phase == 0:
            self._phase = 1
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bulk-load-update-tools",
                        name="toolbox_load",
                        arguments={
                            "tool_names": [
                                "data_preview_update_rows",
                                "data_update_rows",
                            ]
                        },
                    ),
                ),
            )
        if self._phase == 1:
            loaded = self._latest_result(request, "bulk-load-update-tools")
            assert loaded.is_error is False
            self._phase = 2
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bulk-preview",
                        name="data_preview_update_rows",
                        arguments=self._plan(),
                    ),
                ),
            )
        if self._phase == 2:
            preview = self._latest_result(request, "bulk-preview")
            assert preview.is_error is False
            data = preview.output["data"]
            assert isinstance(data, Mapping)
            matched_rows = data["matched_rows"]
            preview_fingerprint = data["preview_fingerprint"]
            assert isinstance(matched_rows, int)
            assert matched_rows > 1
            assert isinstance(preview_fingerprint, str)
            self._matched_rows = matched_rows
            self._phase = 3
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bulk-update",
                        name="data_update_rows",
                        arguments={
                            **self._plan(),
                            "preview_fingerprint": preview_fingerprint,
                            "expected_affected_rows": matched_rows,
                        },
                    ),
                ),
            )
        if self._phase == 3:
            update = self._latest_result(request, "bulk-update")
            assert update.is_error is False
            data = update.output["data"]
            assert isinstance(data, Mapping)
            assert data["outcome"] == "committed"
            assert data["affected_rows"] == self._matched_rows
            self._phase = 4
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bulk-readback",
                        name="data_query",
                        arguments={
                            "source_id": self._source_id,
                            "resource_ids": (self._resource_id,),
                            "sql": (
                                "SELECT priority, COUNT(*) AS matched_rows "
                                "FROM support.tickets "
                                "WHERE ticket_status = $1 AND category = $2 "
                                "GROUP BY priority ORDER BY priority"
                            ),
                            "parameters": ["waiting", "billing"],
                        },
                    ),
                ),
            )
        if self._phase == 4:
            readback = self._latest_result(request, "bulk-readback")
            assert readback.is_error is False
            data = readback.output["data"]
            assert isinstance(data, Mapping)
            rows = data["rows"]
            assert isinstance(rows, tuple)
            assert len(rows) == 1
            row = rows[0]
            assert isinstance(row, Mapping)
            assert row["priority"] == self._desired_priority
            assert row["matched_rows"] == self._matched_rows
            self._phase = 5
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                text=(f"Committed and verified {self._matched_rows} ticket updates."),
            )
        raise AssertionError("bulk update provider received an unexpected model call")


async def _restore_bulk_priority(*, port: int, password: str) -> None:
    import asyncpg  # type: ignore[import-untyped]

    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=port,
        database="daita_large_fixture",
        user="daita_large_writer",
        password=password,
        ssl=False,
    )
    try:
        await connection.execute(
            "UPDATE support.tickets SET priority = 'low' "
            "WHERE ticket_status = 'waiting' AND category = 'billing'"
        )
    finally:
        await connection.close()
