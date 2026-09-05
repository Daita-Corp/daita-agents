from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest
from _mcp_fixtures import MCPConformanceTransport, conformance_identities
from _toolbox_model_support import ToolboxAwareMockModelProvider
from _workspace_support import workspace_for

from daita import Agent, MCPToolSelection, SQLiteSource
from daita.adapters.mcp import StreamableHTTPMCPClientFactory
from daita.artifacts.models import ArtifactAuthorship, ArtifactError
from daita.artifacts.renderers import (
    HTML_MEDIA_TYPE,
    MAX_MODEL_TABULAR_ROWS,
    XLSX_MEDIA_TYPE,
    read_exact_xlsx_data,
    render_model_tabular,
)
from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    OperationalEffect,
    ToolLoadMode,
)
from daita.domains.data.export_capabilities import (
    ARTIFACT_CREATE_TABULAR_CAPABILITY_ID,
    ARTIFACT_CREATE_TABULAR_TOOL_NAME,
    ARTIFACT_READ_TOOL_NAME,
    DOCUMENT_CREATE_TOOL_NAME,
    artifact_capability_declarations,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import Transcript
from daita.storage.sqlite_codecs import decode_message, encode_message

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def _ids():
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        if prefix in {"run", "conversation", "artifact", "destination"}:
            return f"{prefix}-{counts[prefix]:032x}"
        return f"{prefix}-{counts[prefix]}"

    return create


def _profile(provider: ToolboxAwareMockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=4_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _call(call_id: str, name: str, arguments: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _tool_result(output: Transcript, call_id: str) -> ToolResultBlock:
    matches = tuple(
        block
        for message in output.messages
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )
    assert len(matches) == 1
    return matches[0]


def _error_code(result: ToolResultBlock) -> str:
    error = result.output.get("error")
    assert isinstance(error, Mapping)
    code = error.get("code")
    assert isinstance(code, str)
    return code


class _LineageTamperingProvider(ToolboxAwareMockModelProvider):
    def __init__(self) -> None:
        super().__init__(
            (
                _call(
                    "evidence",
                    DOCUMENT_CREATE_TOOL_NAME,
                    {"format": "txt", "content": "alpha\n"},
                ),
                _call(
                    "table",
                    ARTIFACT_CREATE_TABULAR_TOOL_NAME,
                    {
                        "columns": ["answer"],
                        "rows": [["alpha"]],
                        "format": "csv",
                        "evidence_call_ids": ["evidence"],
                    },
                ),
                ModelResponse(finish_reason=FinishReason.STOP, text="done"),
            ),
            provider_id="mock:derived-tabular-tampered",
        )
        self.state_path: Path | None = None
        self.tampered = False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not self.tampered and any(
            isinstance(block, ToolResultBlock) and block.call_id == "evidence"
            for message in request.messages
            for block in message.content
        ):
            assert self.state_path is not None
            with sqlite3.connect(self.state_path) as connection:
                rows = connection.execute(
                    "SELECT run_id, position, data FROM messages ORDER BY run_id, position"
                ).fetchall()
                for run_id, position, data in rows:
                    message = decode_message(data)
                    if not any(
                        isinstance(block, ToolResultBlock)
                        and block.call_id == "evidence"
                        for block in message.content
                    ):
                        continue
                    changed = replace(
                        message,
                        content=tuple(
                            (
                                replace(block, executor_id="tampered.executor")
                                if isinstance(block, ToolResultBlock)
                                and block.call_id == "evidence"
                                else block
                            )
                            for block in message.content
                        ),
                    )
                    connection.execute(
                        "UPDATE messages SET data = ? WHERE run_id = ? AND position = ?",
                        (encode_message(changed), run_id, position),
                    )
                    self.tampered = True
                    break
            assert self.tampered
        return await super().generate(request)


def test_artifact_create_tabular_is_one_bounded_source_neutral_tool() -> None:
    declarations = artifact_capability_declarations()
    views = tuple(
        item
        for item in declarations.tool_views
        if item.name == ARTIFACT_CREATE_TABULAR_TOOL_NAME
    )
    assert len(views) == 1
    (view,) = views
    capability = next(
        item
        for item in declarations.capabilities
        if item.id == ARTIFACT_CREATE_TABULAR_CAPABILITY_ID
    )

    assert view.capability_id == capability.id
    assert view.presentation.load_mode is ToolLoadMode.ON_DEMAND
    assert capability.access_mode is AccessMode.NONE
    assert capability.operational_effect is OperationalEffect.NONE
    assert capability.automation_eligibility is AutomationEligibility.INTERACTIVE_ONLY
    properties = capability.input_schema.get("properties")
    assert isinstance(properties, Mapping)
    assert set(properties) == {
        "columns",
        "rows",
        "format",
        "filename",
        "evidence_call_ids",
    }
    assert capability.input_schema.get("required") == (
        "columns",
        "rows",
        "format",
        "evidence_call_ids",
    )
    assert capability.input_schema.get("additionalProperties") is False
    assert "source_id" not in properties
    assert "adapter_id" not in properties
    assert "executor_id" not in properties
    assert cast(Mapping[str, object], properties["format"])["enum"] == (
        "csv",
        "xlsx",
        "html",
    )
    evidence = cast(Mapping[str, object], properties["evidence_call_ids"])
    assert evidence["minItems"] == 1
    assert evidence["maxItems"] == 16
    assert evidence["uniqueItems"] is True
    rows = cast(Mapping[str, object], properties["rows"])
    columns = cast(Mapping[str, object], properties["columns"])
    assert isinstance(rows.get("maxItems"), int)
    assert isinstance(columns.get("maxItems"), int)


async def test_artifact_create_tabular_commits_derived_csv_from_current_run_evidence(
    tmp_path: Path,
) -> None:
    provider = ToolboxAwareMockModelProvider(
        (
            _call(
                "evidence",
                DOCUMENT_CREATE_TOOL_NAME,
                {"format": "txt", "content": "alpha\n"},
            ),
            _call(
                "table",
                ARTIFACT_CREATE_TABULAR_TOOL_NAME,
                {
                    "columns": ["answer", "count"],
                    "rows": [["alpha", 1]],
                    "format": "csv",
                    "filename": "findings.csv",
                    "evidence_call_ids": ["evidence"],
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        ),
        provider_id="mock:derived-tabular-csv",
    )
    agent = await Agent.create(
        "derived-tabular-csv",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Create a table from the gathered finding.")
        assert len(result.artifacts) == 2
        table_ref = result.artifacts[1]
        payload = await agent.read_artifact(table_ref.artifact_id)
    finally:
        await agent.close()

    assert table_ref.capability_id == ARTIFACT_CREATE_TABULAR_CAPABILITY_ID
    assert table_ref.filename == "findings.csv"
    assert table_ref.media_type == "text/csv"
    assert table_ref.provenance.authorship is ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
    assert table_ref.provenance.evidence_call_ids == ("evidence",)
    assert table_ref.provenance.result_binding is None
    assert payload.content == b'"answer","count"\r\n"alpha",1\r\n'


async def test_mcp_result_can_feed_tabular_and_document_artifacts_with_inherited_lineage(
    tmp_path: Path,
) -> None:
    alpha, _beta = conformance_identities()
    factory = StreamableHTTPMCPClientFactory(
        http_transport=httpx.MockTransport(MCPConformanceTransport(alpha))
    )
    agent = await Agent.create(
        "mcp-derived-artifacts",
        root=tmp_path,
        clock=lambda: NOW,
        mcp_client_factory=factory,
        workspace=workspace_for(tmp_path),
    )
    status = await agent.attach_mcp_server(
        endpoint=alpha.endpoint,
        selections=(
            MCPToolSelection(
                remote_name="lookup",
                local_alias="lookup",
                description="Read one admitted fixture finding.",
                result_sensitivity=ModelSensitivity.CONFIDENTIAL,
            ),
        ),
    )
    mcp_tool_name = status.binding.tools[0].local_name
    await agent.close()

    provider = ToolboxAwareMockModelProvider(
        (
            _call("mcp-result", mcp_tool_name, {"query": "alpha"}),
            _call(
                "table",
                ARTIFACT_CREATE_TABULAR_TOOL_NAME,
                {
                    "columns": ["answer"],
                    "rows": [["alpha"]],
                    "format": "csv",
                    "evidence_call_ids": ["mcp-result"],
                },
            ),
            _call(
                "document",
                DOCUMENT_CREATE_TOOL_NAME,
                {
                    "format": "txt",
                    "content": "The admitted result was alpha.\n",
                    "evidence_call_ids": ["mcp-result"],
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        ),
        provider_id="mock:mcp-derived-artifacts",
    )
    reopened = await Agent.open(
        "mcp-derived-artifacts",
        root=tmp_path,
        clock=lambda: NOW,
        model=provider,
        model_profile=_profile(provider),
        mcp_client_factory=factory,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await reopened.run("Gather the finding and export it.")
        assert len(result.artifacts) == 2
        table_ref, document_ref = result.artifacts
        table_payload = await reopened.read_artifact(table_ref.artifact_id)
        transcript = await reopened.transcript(result.run_id)
    finally:
        await reopened.close()

    assert alpha.calls == [("lookup", {"query": "alpha"})]
    assert table_payload.content == b'"answer"\r\n"alpha"\r\n'
    for ref in (table_ref, document_ref):
        assert ref.provenance.authorship is ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
        assert ref.provenance.evidence_call_ids == ("mcp-result",)
        assert ref.provenance.resource_bindings == ()
        assert ref.sensitivity.value == "confidential"
    assert (
        _tool_result(transcript, "table").sensitivity is ModelSensitivity.CONFIDENTIAL
    )
    assert (
        _tool_result(transcript, "document").sensitivity
        is ModelSensitivity.CONFIDENTIAL
    )


async def test_artifact_create_tabular_rejects_cross_run_evidence(
    tmp_path: Path,
) -> None:
    provider = ToolboxAwareMockModelProvider(
        (
            _call(
                "evidence",
                DOCUMENT_CREATE_TOOL_NAME,
                {"format": "txt", "content": "alpha\n"},
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="first done"),
            _call(
                "table",
                ARTIFACT_CREATE_TABULAR_TOOL_NAME,
                {
                    "columns": ["answer"],
                    "rows": [["alpha"]],
                    "format": "csv",
                    "evidence_call_ids": ["evidence"],
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="second done"),
        ),
        provider_id="mock:derived-tabular-cross-run",
    )
    agent = await Agent.create(
        "derived-tabular-cross-run",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await agent.run("Gather a finding.")
        second = await agent.run(
            "Try to reuse prior-run evidence.",
            conversation_id=first.conversation_id,
        )
        transcript = await agent.transcript(second.run_id)
    finally:
        await agent.close()

    assert len(first.artifacts) == 1
    assert second.artifacts == ()
    table_result = _tool_result(transcript, "table")
    assert table_result.is_error is True
    assert _error_code(table_result) == "artifact_evidence_invalid"


async def test_artifact_create_tabular_rejects_tampered_execution_lineage(
    tmp_path: Path,
) -> None:
    provider = _LineageTamperingProvider()
    agent = await Agent.create(
        "derived-tabular-tampered",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    provider.state_path = agent._embedded._store.path
    try:
        result = await agent.run("Create a table from the prior result.")
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert provider.tampered is True
    assert len(result.artifacts) == 1
    table_result = _tool_result(transcript, "table")
    assert table_result.is_error is True
    assert _error_code(table_result) == "artifact_evidence_invalid"


async def test_relational_evidence_preserves_exact_current_resource_bindings(
    tmp_path: Path,
) -> None:
    database = tmp_path / "findings.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE findings (answer TEXT, count INTEGER)")
        connection.execute("INSERT INTO findings VALUES ('alpha', 1)")
    provider = ToolboxAwareMockModelProvider(
        (), provider_id="mock:derived-tabular-relational"
    )
    agent = await Agent.create(
        "derived-tabular-relational",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database))
    resource = (await agent.list_catalog_resources(source_id=source.id))[0]
    provider.replace_script(
        (
            _call(
                "query",
                "data_query",
                {
                    "source_id": source.id,
                    "resource_ids": [resource.id],
                    "sql": "SELECT answer, count FROM findings",
                },
            ),
            _call(
                "table",
                ARTIFACT_CREATE_TABULAR_TOOL_NAME,
                {
                    "columns": ["answer", "count"],
                    "rows": [["alpha", 1]],
                    "format": "html",
                    "evidence_call_ids": ["query"],
                },
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    try:
        result = await agent.run("Query the source and make a findings table.")
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert len(result.artifacts) == 1
    (table_ref,) = result.artifacts
    query_data = _tool_result(transcript, "query").output.get("data")
    assert isinstance(query_data, Mapping)
    revisions = query_data.get("resource_revisions")
    assert isinstance(revisions, tuple)
    assert len(table_ref.provenance.resource_bindings) == 1
    binding = table_ref.provenance.resource_bindings[0]
    assert binding.source_id == source.id
    assert binding.source_revision == query_data["source_revision"]
    assert binding.resource_id == resource.id
    assert binding.resource_revision == revisions[0]["revision"]


async def test_artifact_read_previews_model_authored_xlsx_data(
    tmp_path: Path,
) -> None:
    provider = ToolboxAwareMockModelProvider(
        (
            _call(
                "evidence",
                DOCUMENT_CREATE_TOOL_NAME,
                {"format": "txt", "content": "alpha\n"},
            ),
            _call(
                "table",
                ARTIFACT_CREATE_TABULAR_TOOL_NAME,
                {
                    "columns": ["formula", "count"],
                    "rows": [["=1+1", 2]],
                    "format": "xlsx",
                    "evidence_call_ids": ["evidence"],
                },
            ),
            _call(
                "read",
                ARTIFACT_READ_TOOL_NAME,
                {"artifact_id": "artifact-00000000000000000000000000000002"},
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        ),
        provider_id="mock:derived-tabular-xlsx-read",
    )
    agent = await Agent.create(
        "derived-tabular-xlsx-read",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Create and preview an XLSX findings table.")
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert len(result.artifacts) == 2
    read = _tool_result(transcript, "read")
    assert read.is_error is False
    data = read.output.get("data")
    assert isinstance(data, Mapping)
    assert data["representation"] == "xlsx_data"
    assert data["columns"] == ("formula", "count")
    assert data["rows"] == (("=1+1", 2),)


def test_model_tabular_html_is_fixed_escaped_and_deterministic() -> None:
    first = render_model_tabular(
        columns=("answer", "active", "missing"),
        rows=(("<script>alert(1)</script>", True, None),),
        format="html",
        filename=None,
        evidence_call_ids=("mcp-call",),
        created_at=NOW,
    )
    second = render_model_tabular(
        columns=("answer", "active", "missing"),
        rows=(("<script>alert(1)</script>", True, None),),
        format="html",
        filename=None,
        evidence_call_ids=("mcp-call",),
        created_at=NOW,
    )

    assert first == second
    assert first.media_type == HTML_MEDIA_TYPE
    assert first.suggested_filename == "findings.html"
    assert b"<script>" not in first.content
    assert b"&lt;script&gt;alert(1)&lt;/script&gt;" in first.content
    assert b'<td data-daita-null="true"></td>' in first.content
    assert b"default-src 'none'" in first.content


def test_model_tabular_xlsx_is_literal_only_readable_and_authorship_bound() -> None:
    draft = render_model_tabular(
        columns=("formula", "count"),
        rows=(("=1+1", 2),),
        format="xlsx",
        filename="findings.xlsx",
        evidence_call_ids=("mcp-call",),
        created_at=NOW,
    )

    assert draft.media_type == XLSX_MEDIA_TYPE
    data = read_exact_xlsx_data(
        draft.content,
        expected_authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
    )
    assert data.columns == ("formula", "count")
    assert data.rows == (("=1+1", 2),)
    with pytest.raises(ArtifactError, match="XLSX package"):
        read_exact_xlsx_data(draft.content)


@pytest.mark.parametrize(
    "columns,rows",
    (
        (("a", "b"), ((1,),)),
        (("a",), (({"nested": "not a scalar"},),)),
        (("a",), tuple((index,) for index in range(MAX_MODEL_TABULAR_ROWS + 1))),
    ),
)
def test_model_tabular_rejects_invalid_shape_nested_values_and_excess_rows(
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> None:
    with pytest.raises(ArtifactError):
        render_model_tabular(
            columns=columns,
            rows=rows,
            format="csv",
            filename=None,
            evidence_call_ids=("source-call",),
            created_at=NOW,
        )
