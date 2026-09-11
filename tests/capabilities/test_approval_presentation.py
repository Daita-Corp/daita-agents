"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

from tests.support.product_workflows import (
    FrozenJsonObject,
    approval_review_document,
    json,
    pytest,
    pytestmark,
    replace,
)


@pytest.mark.parametrize("field", ("password", "api_key", "access_token"))
@pytest.mark.parametrize(
    "capability_id", ("mcp.tool", "data.update_rows", "data.upsert_rows")
)
def test_approval_rejects_secret_fields_without_confusing_token_budgets(
    field, capability_id
):
    safe = json.dumps(
        {
            "routine": {"per_run_max_tokens": 5000},
            "authorization_fingerprint": "sha256:" + "0" * 64,
        }
    )
    assert approval_review_document(
        tool_name="routine_create", capability_id="routines.create", arguments_text=safe
    )[1]
    unsafe = json.dumps({"nested": {field: "sensitive"}})
    assert not approval_review_document(
        tool_name="action", capability_id=capability_id, arguments_text=unsafe
    )[1]


@pytest.mark.parametrize("operation", ("update", "upsert"))
def test_native_review_sanitizes_preview_text_and_denies_oversized_details(operation):
    from daita.capabilities import MAX_APPROVAL_DOCUMENT_CHARACTERS, ApprovalRequest

    arguments = {
        "arguments": {"assignments": [{"column": "name", "value": "\x1b[2JNew"}]},
        "target": {"name": "companies\x1b[2J", "source_name": "Owner's database"},
        "preview": {"matched_rows": 1, "samples": []},
    }
    request = ApprovalRequest(
        "run",
        "call",
        f"data_{operation}_rows",
        f"data.{operation}_rows",
        FrozenJsonObject.from_mapping(arguments),
        "Exact review",
    )
    document, reviewable = approval_review_document(
        tool_name=request.tool_name,
        capability_id=request.capability_id,
        arguments_text=request.render_arguments_for_review(),
        reason=request.reason,
    )
    assert reviewable and document is not None and "\x1b" not in document
    assert "Exact validated details:" in document and "\\u001b" in document
    oversized = replace(
        request,
        arguments=FrozenJsonObject.from_mapping(
            {
                **arguments,
                "arguments": {"value": "x" * MAX_APPROVAL_DOCUMENT_CHARACTERS},
            }
        ),
    )
    assert oversized.render_arguments_for_review() is None
    assert not approval_review_document(
        tool_name=oversized.tool_name,
        capability_id=oversized.capability_id,
        arguments_text=oversized.render_arguments_for_review(),
        reason=oversized.reason,
    )[1]
