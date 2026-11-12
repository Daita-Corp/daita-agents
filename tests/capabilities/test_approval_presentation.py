"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

from tests.support.product_workflows import (
    FrozenJsonObject,
    approval_review_document,
    json,
    pytest,
    pytestmark as _pytestmark,
    replace,
)

pytestmark = _pytestmark


@pytest.mark.parametrize("field", ("password", "api_key", "access_token"))
@pytest.mark.parametrize("capability_id", ("mcp.tool", "data.write"))
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
        tool_name="routine_save", capability_id="routines.create", arguments_text=safe
    )[1]
    unsafe = json.dumps({"nested": {field: "sensitive"}})
    assert not approval_review_document(
        tool_name="action", capability_id=capability_id, arguments_text=unsafe
    )[1]


def test_native_review_sanitizes_preview_text_and_denies_oversized_details():
    from daita.capabilities import MAX_APPROVAL_DOCUMENT_CHARACTERS, ApprovalRequest

    arguments = {
        "arguments": {"sql": "UPDATE t SET name = '\x1b[2JNew' WHERE id = 1"},
        "target": {"name": "companies\x1b[2J", "source_name": "Owner's database"},
        "preview": {"sql": "UPDATE\x1b[2J", "affected_rows": 1, "samples": []},
    }
    request = ApprovalRequest(
        "run",
        "call",
        "data_write",
        "data.write",
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
    assert "data_write\n  Arguments:" in document and "\\u001b" in document
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


def test_readable_details_keep_every_value_and_cannot_fake_a_field():
    from daita.tui.projection import readable_arguments

    text = readable_arguments(
        json.dumps(
            {
                "max_rows": 1000,
                "note": "line one\nmax_rows: 1",
                "count": "42",
                "city": "Montr\u00e9al",
                "columns": ["priority"],
            }
        )
    )
    assert "Max rows: 1000" in text and "  - priority" in text
    assert 'Count: "42"' in text and "Montr\\u00e9al" in text
    assert "\n    max_rows: 1" in text  # indented under Note, not a field
