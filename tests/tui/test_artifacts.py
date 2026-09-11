"""Component-owned tests split from ``test_public_surfaces.py``."""

from __future__ import annotations

from tests.artifacts._public_surface_support import (
    CanonicalMessage,
    MessageRole,
    Path,
    RunInput,
    ToolCall,
    ToolResultBlock,
    Transcript,
    _surface_records,
    artifact_delivery_messages,
    delivery_module,
)


def test_terminal_renders_authoritative_saved_path_and_truthful_delivery_failure() -> (
    None
):
    ref, receipt, result = _surface_records()
    assert any(
        receipt.filename in getattr(item, "filename", "")
        or getattr(item, "saved_path", "") == receipt.saved_path
        for item in result.artifact_deliveries
    )
    run = RunInput(
        id=ref.run_id,
        agent_id="agent-one",
        message="save a file",
        created_at=ref.created_at,
        conversation_id=ref.conversation_id,
    )
    failed = Transcript(
        run=run,
        messages=(
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="save",
                        name="artifact_save_local",
                        arguments={
                            "artifact_id": ref.artifact_id,
                            "mode": "create_new",
                            "destination_id": "default",
                        },
                    ),
                ),
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id="save",
                        is_error=True,
                        output={
                            "error": {
                                "code": "artifact_downloads_unavailable",
                                "message": "Downloads is unavailable.",
                            }
                        },
                    ),
                ),
            ),
        ),
    )
    failed_messages = artifact_delivery_messages(failed.tool_pairs)
    assert any(
        "remains available; local delivery failed" in text for text in failed_messages
    )

    uncertain_edit = Transcript(
        run=run,
        messages=(
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="replace",
                        name="artifact_save_local",
                        arguments={
                            "artifact_id": ref.artifact_id,
                            "mode": "replace_bound_file",
                        },
                    ),
                ),
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id="replace",
                        output={
                            "kind": "artifact.delivery_receipt",
                            "data": {
                                "artifact_id": ref.artifact_id,
                                "mode": "replace_bound_file",
                                "outcome": "uncertain",
                                "relative_path": "config.yaml",
                            },
                        },
                    ),
                ),
            ),
        ),
    )
    uncertain_messages = artifact_delivery_messages(uncertain_edit.tool_pairs)
    assert any(
        "update outcome for workspace file config.yaml is uncertain" in text
        for text in uncertain_messages
    )

    not_delivered = Transcript(
        run=run,
        messages=(
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    ToolCall(
                        id="create",
                        name="artifact_create_document",
                        arguments={"format": "txt", "content": "report"},
                    ),
                ),
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id="create",
                        output={
                            "kind": "artifact.document",
                            "data": {"format": "txt", "character_count": 6},
                            "artifact": {"artifact_id": ref.artifact_id},
                            "delivery_status": "not_delivered",
                        },
                    ),
                ),
            ),
        ),
    )
    not_delivered_messages = artifact_delivery_messages(not_delivered.tool_pairs)
    assert any(
        "was created internally but was not saved locally" in text
        for text in not_delivered_messages
    )


def test_open_reveal_and_folder_picker_are_user_actions_not_model_or_shell_tools() -> (
    None
):
    from daita.domains.data.export_capabilities import artifact_capability_declarations

    tool_names = {item.name for item in artifact_capability_declarations().tool_views}
    assert tool_names.isdisjoint(
        {"artifact_open", "artifact_reveal", "artifact_pick_folder"}
    )
    assert delivery_module.__file__ is not None
    delivery_source = Path(delivery_module.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in delivery_source
    assert "shell=True" not in delivery_source
