"""Production composition acceptance with deterministic model, MCP and database I/O."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from daita.capabilities import ApprovalDecision, EffectEvidenceBasis, EffectOutcome
from daita.capability_runtime import APPROVED_TARGET_CHANGED
from daita.llm.models import (
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.tui.projection import approval_review_document
from tests.support.data_write_approval import (
    PREVIEW_KEYS,
    TARGET_KEYS,
    data_write_approval,
)
from tests.support.native_writes import (
    UPSERT_SQL,
    ScriptedResearchModel,
    create_fixture,
    response,
)


def _script(provider: ScriptedResearchModel, batch: Mapping[str, object]) -> None:
    provider.replace_script(
        (
            response(
                ToolCall("load", "toolbox_load", {"tool_names": ["data_write"]}),
            ),
            response(
                ToolCall(
                    "write",
                    "data_write",
                    {
                        "source": "Company research",
                        "sql": batch["sql"],
                        "parameters": batch["parameters"],
                    },
                )
            ),
            response(text="Saved."),
        )
    )


def _write_result(transcript: Any) -> ToolResultBlock:
    (block,) = (
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == "write"
    )
    return block


def _error(block: ToolResultBlock) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], block.output["error"])


async def test_the_write_tool_is_named_for_loading_only_while_a_table_allows_writes(
    tmp_path, monkeypatch
):
    agent, provider, _db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    try:
        provider.replace_script((response(text="Inspection only."),))
        await agent.run("Save the requested company.")
        request = provider.requests[-1]
        system = "\n".join(
            b.text for b in request.messages[0].content if isinstance(b, TextBlock)
        )
        assert "data_write" in system
        assert "load the tool with toolbox_load" in system
        assert "data_write" not in {tool.name for tool in request.tools}
        permission = await agent.preview_source_permissions(
            source_id=batch["source_id"],
            read_mode="all",
            read_resource_ids=(),
            relational_write_scopes={},
        )
        await agent.apply_source_permissions(
            source_id=batch["source_id"],
            confirmation_fingerprint=permission.confirmation_fingerprint,
        )
        provider.replace_script((response(text="Inspection only."),))
        await agent.run("Save the requested company.")
        system = "\n".join(
            b.text
            for b in provider.requests[-1].messages[0].content
            if isinstance(b, TextBlock)
        )
        assert "Native writes are unavailable" in system
        assert "data_write" not in system
        assert not await agent.list_effects()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("query", "expected", "rank"),
    (
        (
            "Insert one row into a database table with exact supplied fields and "
            "return effect evidence.",
            "data_write",
            3,
        ),
        ("catalog schema columns", "catalog_describe", 1),
        ("export all database rows csv", "data_query", 1),
    ),
)
async def test_search_finds_the_write_tool_apart_from_catalog_and_export_tools(
    tmp_path, monkeypatch, query, expected, rank
):
    agent, provider, db, *_ = await create_fixture(tmp_path, monkeypatch)
    provider.replace_script(
        (
            response(ToolCall("search", "toolbox_search", {"query": query})),
            response(text="Discovery only."),
        )
    )
    try:
        result = await agent.run("Discover the tools without executing work.")
        transcript = await agent.transcript(result.run_id)
        block = dict((call.id, item) for call, item in transcript.tool_pairs)["search"]
        names = [item["tool_name"] for item in block.output["data"]["matches"]]
        assert expected in names[:rank], (query, names)
        assert not db.rows and not await agent.list_effects()
    finally:
        await agent.close()


async def test_a_foreground_write_shows_the_exact_sql_and_keeps_one_receipt(
    tmp_path, monkeypatch
):
    agent, provider, db, *_rest, batch, _clock, approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    try:
        _script(provider, batch)
        result = await agent.run("Save the company.")
        assert result.reason == "completed"
        (receipt,) = await agent._embedded._store.list_effect_receipts(
            agent.id, run_id=result.run_id
        )
        assert receipt.call_id == "write"
        assert receipt.receipt_kind == "data.write"
        assert receipt.outcome is EffectOutcome.SUCCEEDED
        assert receipt.evidence_basis is EffectEvidenceBasis.ADAPTER_VERIFIED
        assert db.rows["new.test"]["evidence_url"] == "https://source.test/new"
        assert db.inserts() == 1

        transcript = await agent._embedded._store.load(result.run_id)
        block = _write_result(transcript)
        assert not block.is_error
        assert cast(Mapping[str, Any], block.output["data"])["outcome"] == "committed"
        # The model addressed the source by name; the call record keeps that.
        raw = next(
            call
            for message in transcript.messages
            for call in message.tool_calls
            if call.id == "write"
        )
        assert raw.arguments["source"] == "Company research"
        assert "source_id" not in raw.arguments

        (approval,) = approvals
        assert approval.capability_id == "data.write"
        document = data_write_approval(approval)
        assert document is not None
        assert set(document["target"]) == TARGET_KEYS
        assert set(document["preview"]) == PREVIEW_KEYS
        assert document["target"]["name"] == "public.companies"
        assert document["target"]["source_name"] == "Company research"
        assert document["arguments"]["sql"] == UPSERT_SQL
        assert document["preview"]["inserted_rows"] == 1
        assert document["preview"]["samples"][0]["values"]["domain"] == "new.test"
        text, reviewable = approval_review_document(
            tool_name=approval.tool_name,
            capability_id=approval.capability_id,
            arguments_text=approval.render_arguments_for_review(),
            reason=approval.reason,
        )
        assert reviewable and text is not None
        assert "Table: public.companies" in text
        assert "Connection: Company research" in text
        assert "SQL: INSERT INTO public.companies" in text
        assert "1 insert, 0 update, 0 delete" in text
    finally:
        await agent.close()


async def test_a_write_above_the_source_classification_is_refused_before_dispatch(
    tmp_path, monkeypatch
):
    from daita.catalog.models import Sensitivity

    agent, provider, db, *_rest, batch, _clock, approvals = await create_fixture(
        tmp_path, monkeypatch, sensitivity=Sensitivity.INTERNAL
    )
    try:
        # Restricted context makes the whole request restricted; the internal
        # source cannot receive it.
        await agent.set_memory(
            "A retained private note.", sensitivity=ModelSensitivity.RESTRICTED
        )
        _script(provider, batch)
        result = await agent.run("Save the company.")
        block = _write_result(await agent._embedded._store.load(result.run_id))
        assert block.is_error
        assert _error(block)["code"] == "write_sensitivity_denied"
        assert not approvals and not db.rows
        assert not await agent.list_effects()
    finally:
        await agent.close()


async def test_a_change_during_approval_is_final_with_no_mutation(
    tmp_path, monkeypatch
):
    agent, provider, db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    concurrent = {
        "id": 42,
        "domain": "new.test",
        "name": "Concurrent writer",
        "evidence_url": "https://source.test/concurrent",
        "notes": None,
        "__xmin": "7",
    }
    approvals = []

    async def approve_after_concurrent_write(request):
        approvals.append(request)
        db.rows["new.test"] = dict(concurrent)
        return ApprovalDecision.APPROVE

    try:
        agent._embedded._capability_runtime._approval_handler = (
            approve_after_concurrent_write
        )
        _script(provider, batch)
        result = await agent.run("Save the company.")
        block = _write_result(await agent._embedded._store.load(result.run_id))
        assert block.is_error
        assert _error(block)["code"] == "state_changed"
        assert dict(_error(block)["details"]["recovery"]) == (
            APPROVED_TARGET_CHANGED.to_details()
        )
        assert len(approvals) == 1
        assert not await agent.list_effects()
        assert db.rows == {"new.test": concurrent}
        assert db.inserts() == 0
    finally:
        await agent.close()
