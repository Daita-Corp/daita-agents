"""Format headless CLI output without importing the Textual application."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import TextIO

from daita import Agent, LearningCandidateStatus, LoopExit
from daita.learning_candidates import (
    LearningCandidateView,
    LearningReviewResult,
    learning_candidate_content_from_mapping,
    learning_candidate_content_to_mapping,
)
from daita.artifacts.models import ArtifactDeliveryMode, ArtifactDeliveryOutcome
from daita.semantics import SemanticAnnotationState, SemanticAnnotationView

from .tui.projection import artifact_delivery_messages, completed_tool_pairs
from .tui.sanitization import MAX_DISPLAY_CHARACTERS, render_model_answer, safe_display

_render_model_answer = render_model_answer


async def _write_artifact_outcomes(
    agent: Agent,
    result: LoopExit,
    output_stream: TextIO,
) -> None:
    for receipt in result.artifact_deliveries:
        filename = getattr(receipt, "filename", None)
        saved_path = getattr(receipt, "saved_path", None)
        if not isinstance(filename, str) or not isinstance(saved_path, str):
            continue
        if receipt.mode is ArtifactDeliveryMode.REPLACE_BOUND_FILE:
            if receipt.outcome is ArtifactDeliveryOutcome.SUCCEEDED:
                message = "Updated workspace file " + safe_display(
                    saved_path, fallback="the bound file"
                )
            elif receipt.outcome is ArtifactDeliveryOutcome.UNCERTAIN:
                message = (
                    "Workspace file update outcome is uncertain for "
                    + safe_display(saved_path, fallback="the bound file")
                )
            else:
                message = "Workspace file was not updated: " + safe_display(
                    saved_path, fallback="the bound file"
                )
            print(message, file=output_stream)
            continue
        print(
            "Saved "
            + safe_display(filename, fallback="artifact")
            + " to "
            + safe_display(saved_path, fallback="the selected destination"),
            file=output_stream,
        )
    try:
        transcript = await agent.transcript(result.run_id)
        for message in artifact_delivery_messages(completed_tool_pairs(transcript)):
            print(message, file=output_stream)
    except Exception:
        return


def _write_learning_candidate_list(
    views: tuple[LearningCandidateView, ...],
    output_stream: TextIO,
) -> None:
    print("Learning candidates", file=output_stream)
    if not views:
        print("  (none)", file=output_stream)
        return
    for view in views:
        print(
            "  "
            + safe_display(view.candidate.id, fallback="candidate")
            + f" [{view.status.value}/{view.candidate.target.value}]",
            file=output_stream,
        )


def _write_learning_candidate_view(
    view: LearningCandidateView,
    output_stream: TextIO,
) -> None:
    candidate = view.candidate
    print(f"Learning candidate: {safe_display(candidate.id)}", file=output_stream)
    print(f"Status: {view.status.value}", file=output_stream)
    print(f"Target: {candidate.target.value}", file=output_stream)
    content = learning_candidate_content_to_mapping(candidate.content)
    print("Proposed content:", file=output_stream)
    print(
        safe_display(
            json.dumps(content.to_dict(), indent=2, sort_keys=True),
            fallback="(invalid)",
            maximum=MAX_DISPLAY_CHARACTERS,
        ),
        file=output_stream,
    )


def _write_learning_review_result(
    result: LearningReviewResult,
    output_stream: TextIO,
) -> None:
    print("Learning review", file=output_stream)
    print(f"  Status: {result.status.value}", file=output_stream)
    print(f"  Reviewed runs: {len(result.reviewed_run_ids)}", file=output_stream)
    print(f"  New candidates: {len(result.candidates)}", file=output_stream)
    print(f"  Model calls: {result.model_calls}", file=output_stream)
    if result.skipped_run_count:
        print(
            f"  Skipped unreadable runs: {result.skipped_run_count}",
            file=output_stream,
        )


async def _write_memory_surface(
    agent: Agent,
    memory_text: str,
    output_stream: TextIO,
) -> None:
    print("Memory", file=output_stream)
    print(file=output_stream)
    print("Global memory:", file=output_stream)
    print(
        safe_display(memory_text, fallback="(empty)", maximum=MAX_DISPLAY_CHARACTERS),
        file=output_stream,
    )
    candidates = await agent.list_learning_candidates()
    print(file=output_stream)
    print("Pending candidates:", file=output_stream)
    if not candidates:
        print("  (none)", file=output_stream)
    else:
        for view in candidates[:12]:
            print(
                "  "
                + safe_display(view.candidate.id, fallback="candidate")
                + f" [{view.status.value}/{view.candidate.target.value}]",
                file=output_stream,
            )
    views = await agent.list_semantic_annotations()
    for heading, state in (
        ("Active data semantics", SemanticAnnotationState.ACTIVE),
        ("Exact duplicates", SemanticAnnotationState.DUPLICATE),
        ("Stale definitions", SemanticAnnotationState.STALE),
        ("Conflicts", SemanticAnnotationState.CONFLICTING),
        ("Superseded definitions", SemanticAnnotationState.SUPERSEDED),
    ):
        print(file=output_stream)
        print(f"{heading}:", file=output_stream)
        selected = tuple(item for item in views if item.state is state)
        if not selected:
            print("  (none)", file=output_stream)
            continue
        for semantic_view in selected:
            print(
                "  "
                + safe_display(
                    semantic_view.annotation.id,
                    fallback="annotation",
                )
                + f" [{semantic_view.annotation.kind.value}] "
                + safe_display(
                    semantic_view.annotation.statement,
                    fallback="definition",
                    maximum=512,
                ),
                file=output_stream,
            )


def _write_semantic_view(view: SemanticAnnotationView, output_stream: TextIO) -> None:
    annotation = view.annotation
    print(f"Semantic annotation: {annotation.id}", file=output_stream)
    print(f"State: {view.state.value}", file=output_stream)
    print(f"Kind: {annotation.kind.value}", file=output_stream)
    print("Statement:", file=output_stream)
    print(
        safe_display(
            annotation.statement,
            fallback="(empty)",
            maximum=MAX_DISPLAY_CHARACTERS,
        ),
        file=output_stream,
    )


def _edit_document(seed: str, *, agent_home: Path) -> str:
    editor = os.environ.get("EDITOR")
    if editor is None or not editor.strip():
        raise RuntimeError("$EDITOR is not set; set it to an available editor command")
    try:
        command = shlex.split(editor)
    except ValueError as error:
        raise RuntimeError("$EDITOR is malformed") from error
    if not command:
        raise RuntimeError("$EDITOR is empty")
    home = agent_home.resolve(strict=True)
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
    if temporary_root == home or home in temporary_root.parents:
        raise RuntimeError("no temporary directory is available outside the agent home")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="daita-edit-",
            suffix=".md",
            delete=False,
            dir=temporary_root,
        ) as temporary:
            temporary.write(seed)
            temporary_path = Path(temporary.name)
        try:
            completed = subprocess.run([*command, str(temporary_path)], check=False)
        except FileNotFoundError as error:
            raise RuntimeError(
                f"$EDITOR command is unavailable: {command[0]}"
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"$EDITOR exited with status {completed.returncode}; "
                "no changes were saved"
            )
        return temporary_path.read_text(encoding="utf-8")
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


async def _edit_learning_candidate(agent: Agent, candidate_id: str) -> None:
    view = await agent.read_learning_candidate(candidate_id)
    if view is None:
        raise ValueError(f"learning candidate not found: {candidate_id}")
    if view.status is not LearningCandidateStatus.AWAITING_REVIEW:
        raise ValueError(f"learning candidate is not editable: {view.status.value}")
    mapping = learning_candidate_content_to_mapping(view.candidate.content)
    current = json.dumps(mapping.to_dict(), indent=2, sort_keys=True) + "\n"
    edited = _edit_document(current, agent_home=agent.home)
    try:
        value = json.loads(edited)
    except json.JSONDecodeError as error:
        raise ValueError("edited candidate content must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("edited candidate content must be one JSON object")
    content = learning_candidate_content_from_mapping(view.candidate.target, value)
    await agent.edit_learning_candidate(candidate_id, content)
