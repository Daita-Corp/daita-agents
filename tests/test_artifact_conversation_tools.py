from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from _toolbox_model_support import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)
from _workspace_support import workspace_for

import daita.domains.data.export_capabilities as artifact_capabilities
from daita import Agent, SQLiteSource
from daita.artifacts.models import (
    ArtifactAuthorship,
    ArtifactError,
    ArtifactProvenance,
    ArtifactRef,
)
from daita.artifacts.renderers import XLSX_MEDIA_TYPE, ExactXlsxData
from daita.artifacts.store import AgentHomeArtifactStore
from daita.capabilities import ToolExecution
from daita.catalog.models import Sensitivity
from daita.domains.data.export_capabilities import (
    ARTIFACT_CONVERT_TOOL_NAME,
    ARTIFACT_LIST_TOOL_NAME,
    ARTIFACT_READ_TOOL_NAME,
    ARTIFACT_SAVE_LOCAL_TOOL_NAME,
    DATA_EXPORT_TABULAR_TOOL_NAME,
    DOCUMENT_CREATE_TOOL_NAME,
    ArtifactListExecutor,
    ArtifactReadExecutor,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)


def _ids():
    counts: defaultdict[str, int] = defaultdict(int)

    def create(prefix: str) -> str:
        counts[prefix] += 1
        if prefix in {"run", "conversation", "artifact", "destination"}:
            return f"{prefix}-{counts[prefix]:032x}"
        return f"{prefix}-{counts[prefix]}"

    return create


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _tools(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


async def _result(agent: Agent, run_id: str, call_id: str) -> ToolResultBlock:
    transcript = await agent.transcript(run_id)
    return next(
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )


async def test_artifact_list_is_bounded_newest_first_and_metadata_only() -> None:
    created = datetime(2026, 8, 3, tzinfo=UTC)
    refs = tuple(
        ArtifactRef(
            artifact_id=f"artifact-{index:032x}",
            run_id=f"run-{index:032x}",
            conversation_id="conversation-00000000000000000000000000000001",
            call_id=f"call-{index}",
            capability_id="artifact.create_document",
            filename=f"file-{index}.txt",
            media_type="text/plain",
            byte_size=1,
            sha256="sha256:" + "a" * 64,
            sensitivity=Sensitivity.INTERNAL,
            provenance=ArtifactProvenance(
                authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
            ),
            created_at=created + timedelta(seconds=index),
        )
        for index in range(1, 52)
    )

    class _Store:
        async def list_refs(self, **_kwargs: object) -> tuple[ArtifactRef, ...]:
            return refs

    output = await ArtifactListExecutor(cast(AgentHomeArtifactStore, _Store())).execute(
        ToolExecution(
            run_id="run-current",
            call_id="call-current",
            capability_id="artifact.list",
            conversation_id="conversation-00000000000000000000000000000001",
        )
    )
    summaries = output.data["artifacts"]
    assert isinstance(summaries, tuple)
    assert len(summaries) == 50
    assert summaries[0]["artifact_id"] == "artifact-00000000000000000000000000000033"
    assert summaries[-1]["artifact_id"] == "artifact-00000000000000000000000000000002"
    assert output.data["truncated"] is True
    for summary in summaries:
        assert set(summary) <= {
            "artifact_id",
            "filename",
            "media_type",
            "byte_size",
            "sha256",
            "created_at",
            "derived_from_artifact_id",
        }


async def test_model_lists_reads_and_converts_the_current_conversation_xlsx_snapshot(
    tmp_path: Path,
) -> None:
    database = tmp_path / "records.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE records (label TEXT NOT NULL, number INTEGER)")
    connection.executemany(
        "INSERT INTO records VALUES (?, ?)",
        (("alpha", 1), ("beta", 2)),
    )
    connection.commit()
    connection.close()
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider((), provider_id="mock:artifact-conversation")
    agent = await Agent.create(
        "artifact-conversation",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database))
    resource = (await agent.list_catalog_resources(source_id=source.id))[0]
    xlsx_id = "artifact-00000000000000000000000000000001"
    other_id = "artifact-00000000000000000000000000000002"
    csv_id = "artifact-00000000000000000000000000000003"
    provider.replace_script(
        (
            _tools(
                ToolCall(
                    id="xlsx-export",
                    name=DATA_EXPORT_TABULAR_TOOL_NAME,
                    arguments={
                        "source_id": source.id,
                        "resource_ids": (resource.id,),
                        "sql": "SELECT label, number FROM records ORDER BY number",
                        "format": "xlsx",
                        "filename": "records.xlsx",
                    },
                )
            ),
            _tools(
                ToolCall(
                    id="xlsx-save",
                    name=ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                    arguments={
                        "artifact_id": xlsx_id,
                        "mode": "create_new",
                        "destination_id": "default",
                    },
                )
            ),
            _stop("saved workbook"),
            _tools(
                ToolCall(
                    id="other-document",
                    name=DOCUMENT_CREATE_TOOL_NAME,
                    arguments={
                        "format": "txt",
                        "filename": "other.txt",
                        "content": "not part of the first conversation",
                    },
                )
            ),
            _stop("created other artifact"),
            _tools(ToolCall(id="list", name=ARTIFACT_LIST_TOOL_NAME, arguments={})),
            _tools(
                ToolCall(
                    id="read",
                    name=ARTIFACT_READ_TOOL_NAME,
                    arguments={"artifact_id": xlsx_id},
                )
            ),
            _tools(
                ToolCall(
                    id="convert",
                    name=ARTIFACT_CONVERT_TOOL_NAME,
                    arguments={
                        "artifact_id": xlsx_id,
                        "format": "csv",
                        "filename": "records-converted.csv",
                    },
                )
            ),
            _tools(
                ToolCall(
                    id="csv-save",
                    name=ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                    arguments={
                        "artifact_id": csv_id,
                        "mode": "create_new",
                        "destination_id": "default",
                    },
                )
            ),
            _stop("converted and saved"),
            _tools(
                ToolCall(
                    id="cross-list",
                    name=ARTIFACT_LIST_TOOL_NAME,
                    arguments={},
                ),
                ToolCall(
                    id="cross-read",
                    name=ARTIFACT_READ_TOOL_NAME,
                    arguments={"artifact_id": xlsx_id},
                ),
            ),
            _stop("read exact prior artifact"),
            _tools(
                ToolCall(
                    id="cross-convert",
                    name=ARTIFACT_CONVERT_TOOL_NAME,
                    arguments={"artifact_id": xlsx_id, "format": "csv"},
                )
            ),
            _stop("conversion remained conversation scoped"),
        )
    )
    try:
        first = await agent.run("Export the current records as XLSX.")
        assert first.artifacts[0].artifact_id == xlsx_id
        assert first.artifacts[0].media_type == XLSX_MEDIA_TYPE

        other = await agent.run("Create a separate TXT artifact.")
        assert other.artifacts[0].artifact_id == other_id

        connection = sqlite3.connect(database)
        connection.execute("DELETE FROM records")
        connection.execute("INSERT INTO records VALUES ('changed', 99)")
        connection.commit()
        connection.close()

        follow_up_request = len(provider.requests)
        converted = await agent.run(
            "Convert the workbook we just made to CSV and save it.",
            conversation_id=first.conversation_id,
        )
        assert xlsx_id not in str(provider.requests[follow_up_request])

        listed = await _result(agent, converted.run_id, "list")
        listed_data = listed.output["data"]
        assert isinstance(listed_data, Mapping)
        summaries = listed_data["artifacts"]
        assert isinstance(summaries, tuple)
        assert tuple(item["artifact_id"] for item in summaries) == (xlsx_id,)
        assert other_id not in repr(listed)

        read = await _result(agent, converted.run_id, "read")
        read_data = read.output["data"]
        assert isinstance(read_data, Mapping)
        assert read_data["representation"] == "xlsx_data"
        assert read_data["columns"] == ("label", "number")
        assert read_data["rows"] == (("alpha", 1), ("beta", 2))
        assert read_data["truncated"] is False

        assert tuple(ref.artifact_id for ref in converted.artifacts) == (csv_id,)
        csv_ref = converted.artifacts[0]
        assert csv_ref.provenance.derived_from_artifact_id == xlsx_id
        assert csv_ref.provenance.resource_bindings == (
            first.artifacts[0].provenance.resource_bindings
        )
        assert csv_ref.sensitivity is first.artifacts[0].sensitivity
        expected = b'"label","number"\r\n"alpha",1\r\n"beta",2\r\n'
        assert (await agent.read_artifact(csv_id)).content == expected
        assert len(converted.artifact_deliveries) == 1
        assert (
            Path(converted.artifact_deliveries[0].saved_path).read_bytes() == expected
        )

        cross_conversation = await agent.run(
            "Read the exact earlier workbook by its artifact ID."
        )
        assert cross_conversation.conversation_id != first.conversation_id
        cross_list = await _result(agent, cross_conversation.run_id, "cross-list")
        cross_list_data = cross_list.output["data"]
        assert isinstance(cross_list_data, Mapping)
        assert cross_list_data["artifacts"] == ()
        cross_read = await _result(agent, cross_conversation.run_id, "cross-read")
        assert cross_read.is_error is False
        cross_read_data = cross_read.output["data"]
        assert isinstance(cross_read_data, Mapping)
        assert cross_read_data["rows"] == (("alpha", 1), ("beta", 2))

        cross_convert = await agent.run(
            "Convert the earlier workbook from another conversation."
        )
        conversion = await _result(agent, cross_convert.run_id, "cross-convert")
        assert conversion.is_error is True
        conversion_error = conversion.output["error"]
        assert isinstance(conversion_error, Mapping)
        assert conversion_error["code"] == "artifact_missing"
    finally:
        await agent.close()


async def test_exact_artifact_read_does_not_cross_agent_homes(tmp_path: Path) -> None:
    provider = MockModelProvider(
        (
            _tools(
                ToolCall(
                    id="create",
                    name=DOCUMENT_CREATE_TOOL_NAME,
                    arguments={"format": "txt", "content": "private agent artifact"},
                )
            ),
            _stop("created"),
        ),
        provider_id="mock:artifact-agent-isolation",
    )
    first = await Agent.create(
        "artifact-agent-one",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        workspace=workspace_for(tmp_path),
    )
    second = await Agent.create(
        "artifact-agent-two", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        created = await first.run("Create a private artifact.")
        artifact_id = created.artifacts[0].artifact_id
        with pytest.raises(ArtifactError, match="not available"):
            await second.read_artifact(artifact_id)
        with pytest.raises(ArtifactError, match="not available"):
            await ArtifactReadExecutor(second._embedded._artifact_store).execute(
                ToolExecution(
                    run_id="run-foreign-artifact",
                    call_id="read-foreign-artifact",
                    capability_id="artifact.read",
                    arguments={"artifact_id": artifact_id},
                    conversation_id=created.conversation_id,
                )
            )
    finally:
        await first.close()
        await second.close()


async def test_artifact_convert_rejects_non_xlsx_without_creating_a_child(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider((), provider_id="mock:artifact-invalid-conversion")
    agent = await Agent.create(
        "artifact-invalid-conversion",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    text_id = "artifact-00000000000000000000000000000001"
    provider.replace_script(
        (
            _tools(
                ToolCall(
                    id="create",
                    name=DOCUMENT_CREATE_TOOL_NAME,
                    arguments={"format": "txt", "content": "hello"},
                )
            ),
            _stop("created"),
            _tools(
                ToolCall(
                    id="convert",
                    name=ARTIFACT_CONVERT_TOOL_NAME,
                    arguments={"artifact_id": text_id, "format": "csv"},
                )
            ),
            _stop("not converted"),
        )
    )
    try:
        first = await agent.run("Create a text file.")
        second = await agent.run(
            "Convert that artifact to CSV.",
            conversation_id=first.conversation_id,
        )
        result = await _result(agent, second.run_id, "convert")
        assert result.is_error
        error = result.output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "artifact_invalid_format"
        assert second.artifacts == ()
        assert second.artifact_deliveries == ()
    finally:
        await agent.close()


async def test_artifact_conversion_cancellation_leaves_no_child_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "cancel.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE records (label TEXT NOT NULL)")
    connection.execute("INSERT INTO records VALUES ('alpha')")
    connection.commit()
    connection.close()
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    provider = MockModelProvider((), provider_id="mock:artifact-cancel-conversion")
    agent = await Agent.create(
        "artifact-cancel-conversion",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        id_factory=_ids(),
        downloads_directory=downloads,
        workspace=workspace_for(tmp_path),
    )
    source = await agent.attach(SQLiteSource(database))
    resource = (await agent.list_catalog_resources(source_id=source.id))[0]
    xlsx_id = "artifact-00000000000000000000000000000001"
    provider.replace_script(
        (
            _tools(
                ToolCall(
                    id="export",
                    name=DATA_EXPORT_TABULAR_TOOL_NAME,
                    arguments={
                        "source_id": source.id,
                        "resource_ids": (resource.id,),
                        "sql": "SELECT label FROM records",
                        "format": "xlsx",
                    },
                )
            ),
            _stop("created"),
            _tools(
                ToolCall(
                    id="convert",
                    name=ARTIFACT_CONVERT_TOOL_NAME,
                    arguments={"artifact_id": xlsx_id, "format": "csv"},
                )
            ),
        )
    )
    started = threading.Event()
    release = threading.Event()
    original = artifact_capabilities.read_exact_xlsx_data

    def blocked(content: bytes) -> ExactXlsxData:
        started.set()
        assert release.wait(2)
        return original(content)

    monkeypatch.setattr(artifact_capabilities, "read_exact_xlsx_data", blocked)
    try:
        first = await agent.run("Create an XLSX workbook.")
        running = asyncio.create_task(
            agent.run(
                "Convert it to CSV.",
                conversation_id=first.conversation_id,
            )
        )
        assert await asyncio.to_thread(started.wait, 2)
        running.cancel()
        release.set()
        try:
            await running
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("conversion run did not propagate cancellation")
        refs = await agent._embedded._artifact_store.list_refs(
            conversation_id=first.conversation_id
        )
        assert tuple(ref.artifact_id for ref in refs) == (xlsx_id,)
        assert not tuple((agent.home / "artifacts" / ".staging").iterdir())
    finally:
        release.set()
        await agent.close()
