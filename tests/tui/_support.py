"""Pure and Pilot tests for the Textual interactive presentation."""

from __future__ import annotations

__all__ = (
    "BUILTIN_SLASH_COMMANDS",
    "BUILTIN_SLASH_COMMAND_ROOTS",
    "CAPABILITY_LABELS",
    "MAX_CLIPBOARD_UTF8_BYTES",
    "MAX_COMPOSER_CHARACTERS",
    "MIN_READY_ROWS",
    "MIN_USABLE_COLUMNS",
    "SLASH_COMMAND_COMPLETIONS",
    "UTC",
    "ActivityBar",
    "Agent",
    "AgentCreateScreen",
    "AgentEvent",
    "AgentEventKind",
    "App",
    "ApprovalDecision",
    "ApprovalPanel",
    "ApprovalRequest",
    "Button",
    "CatalogScreen",
    "ChatScreen",
    "ClipboardResult",
    "CollapsibleTitle",
    "CompletionPopup",
    "ComposeResult",
    "Composer",
    "ConfirmScreen",
    "CredentialSession",
    "DaitaApp",
    "DeliveryState",
    "DeliverySubjectKind",
    "FinishReason",
    "Footer",
    "FrozenJsonObject",
    "InboxScreen",
    "InboxView",
    "Input",
    "Iterator",
    "GraphState",
    "JobsScreen",
    "LoopExit",
    "LoopExitKind",
    "MockModelProvider",
    "ModelResponse",
    "ModelSensitivity",
    "ModelSetupScreen",
    "ObserverEvent",
    "Offset",
    "OptionList",
    "OutcomeConclusionKind",
    "OutcomeState",
    "Path",
    "PermissionsScreen",
    "PickerOption",
    "ReviewCostScreen",
    "RoutineState",
    "RoutinesScreen",
    "SQLiteSource",
    "ScheduleKind",
    "ScheduledRoutineInspection",
    "ScheduledRoutineSummary",
    "SecretReference",
    "SecretResolutionError",
    "Select",
    "Selection",
    "SelectionScreen",
    "SimpleNamespace",
    "SourceEditScreen",
    "SourceSetupScreen",
    "Static",
    "Text",
    "ToolCall",
    "ToolCard",
    "ToolCardDetails",
    "ToolCardState",
    "ToolResultBlock",
    "TranscriptBlock",
    "TranscriptView",
    "Tree",
    "UserInputError",
    "WelcomeView",
    "_create_sqlite_source",
    "_mock_profile",
    "_tui_inbox_item",
    "_tui_job_inspection",
    "_tui_job_summary",
    "approval_review_document",
    "asyncio",
    "clipboard_mechanism",
    "context_window_text",
    "contextmanager",
    "datetime",
    "deliver_clipboard",
    "format_token_count",
    "learning_invocation_message",
    "os",
    "osc52_sequence",
    "parse_postgresql_connection_url",
    "parse_source_override",
    "project_tool_details",
    "pytest",
    "redact_presentation_value",
    "render_inbox_item",
    "replace",
    "run_failure_notice",
    "sanitize_terminal_text",
    "sqlite3",
    "workspace_for",
)

import asyncio
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import (
    Button,
    Footer,
    Input,
    OptionList,
    Select,
    Static,
    Tree,
)
from textual.widgets._collapsible import CollapsibleTitle

from daita import (
    Agent,
    AgentEvent,
    AgentEventKind,
    ApprovalDecision,
    ApprovalRequest,
    DeliveryState,
    DeliverySubjectKind,
    GraphState,
    InboxView,
    LoopExit,
    LoopExitKind,
    OutcomeConclusionKind,
    OutcomeState,
    RoutineState,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    SQLiteSource,
)
from daita._json import FrozenJsonObject
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.routines.models import ScheduleKind
from daita.security import CredentialSession, SecretReference, SecretResolutionError
from daita.tui.app import DaitaApp
from daita.tui.clipboard import (
    MAX_CLIPBOARD_UTF8_BYTES,
    ClipboardResult,
    clipboard_mechanism,
    deliver_clipboard,
    osc52_sequence,
)
from daita.tui.commands import (
    BUILTIN_SLASH_COMMAND_ROOTS,
    BUILTIN_SLASH_COMMANDS,
    SLASH_COMMAND_COMPLETIONS,
    learning_invocation_message,
    parse_postgresql_connection_url,
    parse_source_override,
)
from daita.tui.models import (
    MAX_COMPOSER_CHARACTERS,
    MIN_READY_ROWS,
    MIN_USABLE_COLUMNS,
    PickerOption,
    ToolCardDetails,
    ToolCardState,
    TranscriptBlock,
    UserInputError,
)
from daita.tui.observer import ObserverEvent
from daita.tui.projection import (
    CAPABILITY_LABELS,
    approval_review_document,
    project_tool_details,
    redact_presentation_value,
    run_failure_notice,
)
from daita.tui.sanitization import sanitize_terminal_text
from daita.tui.screens.catalog import CatalogScreen
from daita.tui.screens.chat import ChatScreen
from daita.tui.screens.confirm import ConfirmScreen
from daita.tui.screens.editing import ReviewCostScreen
from daita.tui.screens.inbox import InboxScreen, render_inbox_item
from daita.tui.screens.jobs import JobsScreen
from daita.tui.screens.onboarding import (
    AgentCreateScreen,
    ModelSetupScreen,
    SourceSetupScreen,
)
from daita.tui.screens.permissions import PermissionsScreen
from daita.tui.screens.routines import RoutinesScreen
from daita.tui.screens.selection import SelectionScreen
from daita.tui.screens.source_edit import SourceEditScreen
from daita.tui.widgets.approval import ApprovalPanel
from daita.tui.widgets.composer import CompletionPopup, Composer
from daita.tui.widgets.status import (
    ActivityBar,
    context_window_text,
    format_token_count,
)
from daita.tui.widgets.tool_card import ToolCard
from daita.tui.widgets.transcript import TranscriptView
from daita.tui.widgets.welcome import WelcomeView
from tests.support.workspace import workspace_for


def _tui_inbox_item(
    delivery_id: str = "delivery-stage-c",
    *,
    report: str = "The durable profile completed successfully.",
) -> InboxView:
    observed = datetime(2026, 8, 23, 14, 5, tzinfo=UTC)
    return InboxView(
        delivery_id=delivery_id,
        conversation_id="conversation-inbox",
        subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
        subject_id="occurrence-inbox",
        conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
        conclusion_state=OutcomeState.SUCCEEDED,
        conclusion_digest="sha256:" + "2" * 64,
        conclusion_preview=report,
        conclusion_preview_truncated=False,
        resulting_run_id="run-routine",
        artifact_references=(),
        effective_sensitivity=ModelSensitivity.INTERNAL,
        provenance_digest="sha256:" + "3" * 64,
        destination_id="conversation_inbox:conversation-inbox",
        state=DeliveryState.AVAILABLE,
        created_at=observed,
        updated_at=observed,
        acknowledged_at=None,
        blocked_reason_code=None,
        failure_code=None,
    )


def _tui_job_summary(
    job_id: str,
    status: GraphState,
    *,
    result_available: bool,
) -> SimpleNamespace:
    observed = datetime(2026, 8, 23, 14, 0, tzinfo=UTC)
    return SimpleNamespace(
        job_id=job_id,
        conversation_id="conversation-jobs",
        state=status,
        desired_state=SimpleNamespace(value="run"),
        finalizer_task_id="task-finalizer",
        deadline_at=observed + timedelta(minutes=5),
        specification=SimpleNamespace(
            objective="Profile the selected data.",
            authority=SimpleNamespace(
                source_ids=("source-one",),
                resource_ids=("resource-one",),
            ),
        ),
        created_at=observed,
        updated_at=observed,
        terminal_at=(observed if result_available else None),
        terminal_result_id=("result-final" if result_available else None),
        failure_code=None,
    )


def _tui_job_inspection(summary: SimpleNamespace) -> SimpleNamespace:
    observed = datetime(2026, 8, 23, 14, 0, tzinfo=UTC)
    return SimpleNamespace(
        job=summary,
        graph=SimpleNamespace(
            revision=1,
            task_count=0,
            edge_count=0,
            active_attempt_count=0,
        ),
        tasks=(),
        dependencies=(),
        attempts=(
            SimpleNamespace(
                ordinal=1,
                fencing_epoch=1,
                state=SimpleNamespace(value="running"),
                started_at=observed,
                ended_at=None,
                error_code=None,
            ),
        ),
        results=(),
        controls=(),
        checkpoints=(),
        comments=(),
        budget_ledgers=(),
        events=(),
        delivery_ids=(),
    )


def _mock_profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=32_000,
        max_output_tokens=1_024,
        supports_tools=True,
        supports_parallel_tools=True,
        supports_streaming=True,
    )


def _create_sqlite_source(path: Path, table: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)')
        connection.commit()
    finally:
        connection.close()
