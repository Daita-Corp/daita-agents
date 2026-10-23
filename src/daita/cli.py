"""Implement headless command-line workflows through the public Agent API."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import getpass
import json
import os
import shlex
import subprocess
import sys
import tempfile
import typing
from collections.abc import Mapping, Sequence
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from . import (
    Agent,
    AgentConfig,
    AgentEvent,
    AgentEventKind,
    AgentObserver,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    ArtifactError,
    EffectReceipt,
    EffectResolutionDecision,
    FolderSource,
    LocalFileAccess,
    LocalWorkspace,
    LoopLimits,
    MCPAuthentication,
    MCPBindingStatus,
    MCPServerInspection,
    MCPToolSelection,
    PostgreSQLSource,
    RoutineIntent,
    RoutineState,
    SourceTypeRegistry,
    SQLiteSource,
    __version__,
    create_llm_provider,
)
from ._json import FrozenJsonObject
from .artifacts.models import (
    artifact_delivery_receipt_to_mapping,
    artifact_destination_to_mapping,
    artifact_ref_to_mapping,
)
from .distribution import delivery_inspection_projection, delivery_view_projection
from .errors import StateCompatibilityError
from .jobs.graph.models import GraphInspection, GraphState, TaskState
from .jobs.owner import GraphBlockerProjection
from .llm import (
    ModelProfile,
    ModelSensitivity,
)
from .llm.profiles import reviewed_model_profile
from .llm.protocols import ManagedModelProvider
from .observation import summarize_diagnostics
from .routines.capabilities import routine_inspection_projection, routine_projection
from .security import SecretReference
from .terminal import run_terminal_application
from .tui.projection import run_failure_notice, tool_outcome_summary
from .work_board import work_board_projection
from .workspace import working_directory


def _graph_inspection_mapping(inspection: GraphInspection) -> dict[str, object]:
    """Project one bounded current graph inspection for the CLI."""

    if not isinstance(inspection, GraphInspection):
        raise TypeError("graph inspection projection requires GraphInspection")
    attempts_by_task: dict[str, list[dict[str, object]]] = {}
    for attempt in inspection.attempts:
        attempts_by_task.setdefault(attempt.task_id, []).append(
            {
                "attempt_id": attempt.attempt_id,
                "ordinal": attempt.ordinal,
                "state": attempt.state.value,
                "fencing_epoch": attempt.fencing_epoch,
                "heartbeat_at": (
                    None
                    if attempt.heartbeat_at is None
                    else attempt.heartbeat_at.isoformat()
                ),
                "checkpoint_ids": attempt.checkpoint_ids,
                "artifact_ids": attempt.artifact_ids,
                "error_code": attempt.error_code,
            }
        )
    results = {item.task_id: item for item in inspection.results}
    return {
        "job_id": inspection.job.job_id,
        "state": inspection.job.state.value,
        "desired_state": inspection.job.desired_state.value,
        "specification_digest": inspection.job.specification_digest,
        "topology_revision": inspection.graph.revision,
        "task_count": inspection.graph.task_count,
        "edge_count": inspection.graph.edge_count,
        "active_attempt_count": inspection.graph.active_attempt_count,
        "terminal_result_id": inspection.job.terminal_result_id,
        "tasks": [
            {
                "task_id": task.task_id,
                "role": task.role.value,
                "state": task.state.value,
                "task_revision": task.task_revision,
                "attempts": attempts_by_task.get(task.task_id, []),
                "accepted_result": (
                    None
                    if task.task_id not in results
                    else {
                        "result_id": results[task.task_id].result_id,
                        "result_digest": results[task.task_id].result_digest,
                        "artifact_ids": results[task.task_id].artifact_ids,
                    }
                ),
            }
            for task in inspection.tasks
        ],
        "dependencies": [
            {
                "upstream_task_id": edge.upstream_task_id,
                "downstream_task_id": edge.downstream_task_id,
                "edge_kind": edge.edge_kind.value,
            }
            for edge in inspection.dependencies
        ],
        "controls": [
            {
                "control_id": control.control_id,
                "task_id": control.task_id,
                "kind": control.kind.value,
                "state": control.state.value,
                "payload_digest": control.payload_digest,
                "payload": control.payload,
                "resolution": control.resolution,
                "resolved_by_kind": control.resolved_by_kind,
                "resolved_by_id": control.resolved_by_id,
            }
            for control in inspection.controls
        ],
        "checkpoints": [
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "task_id": checkpoint.task_id,
                "attempt_id": checkpoint.attempt_id,
                "ordinal": checkpoint.ordinal,
                "milestone": checkpoint.milestone,
                "payload_digest": checkpoint.payload_digest,
            }
            for checkpoint in inspection.checkpoints
        ],
        "comments": [
            {
                "comment_id": comment.comment_id,
                "task_id": comment.task_id,
                "author_kind": comment.author_kind,
                "author_id": comment.author_id,
                "body_digest": comment.body_digest,
            }
            for comment in inspection.comments
        ],
        "budgets": [
            {
                "dimension": item.dimension,
                "task_id": item.task_id,
                "ceiling": item.ceiling,
                "settled": item.settled,
                "reserved": item.reserved,
                "control_reserved": item.control_reserved,
            }
            for item in inspection.budget_ledgers
        ],
        "events": [
            {
                "event_id": item.event_id,
                "kind": item.kind,
                "task_id": item.task_id,
                "attempt_id": item.attempt_id,
                "created_at": item.created_at.isoformat(),
            }
            for item in inspection.events
        ],
        "delivery_ids": inspection.delivery_ids,
    }


def _graph_blockers_mapping(
    projection: GraphBlockerProjection,
) -> dict[str, object]:
    """Project the bounded typed blocker view without adding graph state ownership."""

    if not isinstance(projection, GraphBlockerProjection):
        raise TypeError("graph blocker projection requires GraphBlockerProjection")
    return {
        "job_id": projection.job_id,
        "graph_state": projection.graph_state.value,
        "blockers": projection.blockers,
    }


def _effect_receipt_mapping(receipt: EffectReceipt) -> dict[str, object]:
    """Project operator evidence without exposing a Textual dependency."""

    resolution = receipt.resolution
    return FrozenJsonObject.from_mapping(
        {
            **receipt.material(),
            "receipt_digest": receipt.receipt_digest,
            "resolution": (
                None
                if resolution is None
                else {
                    "decision": resolution.decision.value,
                    "note": resolution.note,
                    "evidence_references": resolution.evidence_references,
                    "approving_principal_id": resolution.approving_principal_id,
                    "control_id": resolution.control_id,
                    "resolved_at": resolution.resolved_at.isoformat(),
                    "receipt_digest": resolution.receipt_digest,
                }
            ),
        }
    ).to_dict()


def _mcp_authentication(
    bearer_env: str | None, bearer_ref: str | None = None
) -> MCPAuthentication:
    if bearer_ref is not None:
        return MCPAuthentication.bearer(SecretReference.parse(bearer_ref))
    return (
        MCPAuthentication.no_auth()
        if bearer_env is None
        else MCPAuthentication.bearer(SecretReference.environment(bearer_env))
    )


def _mcp_inspection_mapping(inspection: MCPServerInspection) -> dict[str, object]:
    return {
        "endpoint": inspection.endpoint,
        "protocol_version": inspection.protocol_version,
        "server_name": inspection.server_name,
        "server_version": inspection.server_version,
        "observed_at": inspection.observed_at.isoformat(),
        "tools": [
            {
                "remote_name": tool.remote_name,
                "supported": tool.supported,
                "unsupported_reason": tool.unsupported_reason,
                "input_schema": (
                    None if tool.input_schema is None else tool.input_schema.to_dict()
                ),
                "input_schema_digest": tool.input_schema_digest,
                "output_schema": (
                    None if tool.output_schema is None else tool.output_schema.to_dict()
                ),
                "output_schema_digest": tool.output_schema_digest,
            }
            for tool in inspection.tools
        ],
    }


def _mcp_status_mapping(status: MCPBindingStatus) -> dict[str, object]:
    binding = status.binding
    return {
        "binding_id": binding.binding_id,
        "endpoint": binding.endpoint,
        "protocol_version": binding.protocol_version,
        "server_name": binding.server_name,
        "server_version": binding.server_version,
        "local_label": binding.local_label,
        "maximum_outbound_sensitivity": (binding.maximum_outbound_sensitivity.value),
        "state": binding.state.value,
        "revision": binding.revision,
        "activated_revision": status.activated_revision,
        "active_in_runtime": status.active_in_runtime,
        "reopen_required": status.reopen_required,
        "stale_reason": binding.stale_reason,
        "admitted_at": binding.admitted_at.isoformat(),
        "last_checked_at": binding.last_checked_at.isoformat(),
        "revoked_at": (
            None if binding.revoked_at is None else binding.revoked_at.isoformat()
        ),
        "tools": [
            {
                "local_name": tool.local_name,
                "remote_name": tool.remote_name,
                "capability_id": tool.capability_id,
                "input_schema_digest": tool.input_schema_digest,
                "output_schema_digest": tool.output_schema_digest,
                "result_sensitivity": tool.result_sensitivity.value,
            }
            for tool in binding.tools
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daita")
    parser.add_argument("--version", action="version", version=f"daita {__version__}")
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--workspace",
        type=Path,
        help="default working directory for foreground local file tools",
    )
    parser.add_argument(
        "--workspace-sensitivity",
        choices=("internal", "confidential", "restricted"),
        default="internal",
    )
    parser.add_argument("--agent", help="agent to open in terminal mode")
    parser.add_argument(
        "--no-host",
        action="store_true",
        help="open the agent in this process instead of the Daita host; "
        "fails if the host holds it",
    )
    commands = parser.add_subparsers(dest="command")

    create = commands.add_parser("create", help="create an agent")
    create.add_argument("name")

    attach = commands.add_parser("attach", help="attach a read-only source")
    attach.add_argument("name")
    attach.add_argument(
        "kind",
        choices=tuple(item.adapter_id for item in SourceTypeRegistry().types()),
    )
    attach.add_argument("path", type=Path, nargs="?")
    attach.add_argument("--host")
    attach.add_argument("--port", type=int, default=5432)
    attach.add_argument("--database")
    attach.add_argument("--username")
    attach.add_argument(
        "--password-env",
        help="environment variable containing the PostgreSQL password",
    )
    attach.add_argument(
        "--schema",
        action="append",
        dest="schemas",
        help="PostgreSQL schema to attach; repeat for multiple schemas",
    )
    attach.add_argument(
        "--ssl-mode",
        choices=("disable", "prefer", "allow", "require", "verify-ca", "verify-full"),
        default="require",
    )
    attach.add_argument("--source-name")

    sources = commands.add_parser("sources", help="list attached sources")
    sources.add_argument("name")

    state = commands.add_parser("state", help="inspect agent-home persistence state")
    state_commands = state.add_subparsers(dest="state_command", required=True)
    state_status = state_commands.add_parser(
        "status",
        help="show revision and pending-recovery facts without changing the home",
    )
    state_status.add_argument("name")

    mcp = commands.add_parser("mcp", help="manage admitted remote MCP tools")
    mcp_commands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_inspect = mcp_commands.add_parser(
        "inspect",
        help="inspect one endpoint without admitting tools",
    )
    mcp_inspect.add_argument("name")
    mcp_inspect.add_argument("endpoint")
    inspect_auth = mcp_inspect.add_mutually_exclusive_group()
    inspect_auth.add_argument("--bearer-env")
    inspect_auth.add_argument("--bearer-ref", help="env:NAME or keychain:ACCOUNT")
    inspect_auth.add_argument(
        "--bearer-prompt", action="store_true", help="enter a masked token"
    )
    mcp_attach = mcp_commands.add_parser(
        "attach",
        help="admit and activate exact read-only tools",
    )
    mcp_attach.add_argument("name")
    mcp_attach.add_argument("endpoint")
    attach_auth = mcp_attach.add_mutually_exclusive_group()
    attach_auth.add_argument("--bearer-env")
    attach_auth.add_argument("--bearer-ref", help="env:NAME or keychain:ACCOUNT")
    attach_auth.add_argument(
        "--bearer-prompt",
        action="store_true",
        help="enter and save a masked token in keychain",
    )
    mcp_attach.add_argument("--binding-id")
    mcp_attach.add_argument(
        "--maximum-outbound-sensitivity",
        choices=tuple(item.value for item in ModelSensitivity),
        default=ModelSensitivity.INTERNAL.value,
    )
    mcp_attach.add_argument(
        "--tool",
        action="append",
        nargs="+",
        required=True,
        metavar="REMOTE",
        help="repeat --tool REMOTE; optionally supply ALIAS DESCRIPTION RESULT_SENSITIVITY",
    )
    mcp_status = mcp_commands.add_parser("status", help="show bounded binding status")
    mcp_status.add_argument("name")
    mcp_status.add_argument("binding_id", nargs="?")
    mcp_refresh = mcp_commands.add_parser("refresh", help="refresh one exact binding")
    mcp_refresh.add_argument("name")
    mcp_refresh.add_argument("binding_id")
    mcp_revoke = mcp_commands.add_parser("revoke", help="revoke one exact binding")
    mcp_revoke.add_argument("name")
    mcp_revoke.add_argument("binding_id")
    mcp_revoke.add_argument("--yes", action="store_true")

    readiness = commands.add_parser(
        "relational-update-readiness",
        help="inspect one resource and assignment-column update scope",
    )
    readiness.add_argument("name")
    readiness.add_argument("source_id")
    readiness.add_argument("resource_id")
    readiness.add_argument(
        "--assignment-column",
        action="append",
        dest="assignment_columns",
        required=True,
        help="exact catalog column proposed for assignment; repeat as needed",
    )

    detach = commands.add_parser("detach", help="detach a source")
    detach.add_argument("name")
    detach.add_argument("source_id")
    detach.add_argument(
        "--yes",
        action="store_true",
        help="confirm source detachment and owned-credential deletion",
    )

    conversations = commands.add_parser(
        "conversations",
        help="manage persisted conversation history",
    )
    conversation_commands = conversations.add_subparsers(
        dest="conversations_command",
        required=True,
    )
    conversation_clear = conversation_commands.add_parser(
        "clear",
        help="clear all conversation history",
    )
    conversation_clear.add_argument("name")
    conversation_clear.add_argument(
        "--yes",
        action="store_true",
        help="confirm transcript deletion",
    )

    artifacts = commands.add_parser("artifacts", help="save a known run artifact by ID")
    artifact_commands = artifacts.add_subparsers(
        dest="artifacts_command",
        required=True,
    )
    artifact_save = artifact_commands.add_parser("save")
    artifact_save.add_argument("name")
    artifact_save.add_argument("artifact_id")
    artifact_save.add_argument("--destination", type=Path)
    artifact_save.add_argument("--filename")

    export_location = commands.add_parser(
        "export-location",
        help="get or change the persistent local export location",
    )
    export_location_commands = export_location.add_subparsers(
        dest="export_location_command",
        required=True,
    )
    export_location_get = export_location_commands.add_parser("get")
    export_location_get.add_argument("name")
    export_location_set = export_location_commands.add_parser("set")
    export_location_set.add_argument("name")
    export_location_set.add_argument("directory", type=Path)
    export_location_reset = export_location_commands.add_parser("reset")
    export_location_reset.add_argument("name")

    delete = commands.add_parser("delete", help="permanently delete an agent")
    delete.add_argument("name")
    delete.add_argument(
        "--yes",
        action="store_true",
        help="confirm agent-home and owned-credential deletion",
    )

    run = commands.add_parser("run", help="run one agent request")
    run.add_argument("name")
    run.add_argument("message")
    run.add_argument(
        "--model",
        help="provider:model override for this invocation only",
    )
    run.add_argument("--base-url")
    run.add_argument("--context-window", type=int)
    run.add_argument("--max-output", type=int)
    run.add_argument("--conversation-id")
    run.add_argument("--files-only", action="store_true")
    run.add_argument("--events-jsonl", action="store_true")
    run.add_argument(
        "--diagnostics-path",
        type=Path,
        help="append payload-free per-call diagnostics to this private JSONL file",
    )
    run.add_argument(
        "--max-cost-usd",
        type=Decimal,
        help="finite positive estimated model-cost ceiling for this run",
    )

    chat = commands.add_parser(
        "chat",
        help="open the same interactive Textual app as `daita`",
    )
    chat.add_argument("name")
    chat.add_argument("--model", help="ignored; model setup happens inside the app")
    chat.add_argument("--conversation")

    host = commands.add_parser(
        "host",
        help="run the background Daita host that owns agents, runs and routines",
    )
    host.add_argument(
        "--agent", dest="host_agent", help="deprecated; runs `daita host serve`"
    )
    host_commands = host.add_subparsers(dest="host_command")
    host_serve = host_commands.add_parser("serve", help="run the host in this process")
    host_serve.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    host_serve.add_argument(
        "--foreground", action="store_true", help="also log to standard error"
    )
    host_serve.add_argument(
        "--log-level", choices=("debug", "info", "warning", "error"), default="info"
    )
    # Set by `daita host start` and the login service: losing the singleton race
    # exits quietly, and the host exits when idle unless the service is installed.
    host_serve.add_argument("--auto", action="store_true", help=argparse.SUPPRESS)
    host_commands.add_parser("start", help="start the host in the background")
    host_commands.add_parser("stop", help="stop the host gracefully")
    host_status = host_commands.add_parser("status", help="show whether the host runs")
    host_status.add_argument("--json", action="store_true", dest="as_json")
    host_logs = host_commands.add_parser("logs", help="print the host log")
    host_logs.add_argument("-f", "--follow", action="store_true")
    host_install = host_commands.add_parser(
        "install", help="run the host as a login service (launchd or systemd)"
    )
    host_install.add_argument(
        "--no-start", action="store_true", help="start it at the next login"
    )
    host_commands.add_parser("uninstall", help="remove the login service")

    diagnostics = commands.add_parser(
        "diagnostics",
        help="summarize a private per-call diagnostics file",
    )
    diagnostic_commands = diagnostics.add_subparsers(
        dest="diagnostics_command", required=True
    )
    diagnostic_summary = diagnostic_commands.add_parser(
        "summary",
        help="group model attempts, tokens and largest tool results by run",
    )
    diagnostic_summary.add_argument("path", type=Path)

    jobs = commands.add_parser(
        "jobs",
        help="inspect and resolve bounded durable-job controls",
    )
    job_commands = jobs.add_subparsers(dest="jobs_command", required=True)
    job_list = job_commands.add_parser("list")
    job_list.add_argument("name")
    job_list.add_argument(
        "--state", action="append", choices=tuple(item.value for item in GraphState)
    )
    job_list.add_argument("--limit", type=int, default=50)
    job_inspect = job_commands.add_parser("inspect")
    job_inspect.add_argument("name")
    job_inspect.add_argument("job_id")
    job_tasks = job_commands.add_parser("tasks")
    job_tasks.add_argument("name")
    job_tasks.add_argument("job_id")
    job_tasks.add_argument(
        "--state", action="append", choices=tuple(item.value for item in TaskState)
    )
    job_tasks.add_argument("--limit", type=int, default=64)
    job_dependencies = job_commands.add_parser("dependencies")
    job_dependencies.add_argument("name")
    job_dependencies.add_argument("job_id")
    job_dependencies.add_argument("--task-id")
    job_dependencies.add_argument("--limit", type=int, default=100)
    job_attempts = job_commands.add_parser("attempts")
    job_attempts.add_argument("name")
    job_attempts.add_argument("job_id")
    job_attempts.add_argument("task_id")
    job_attempts.add_argument("--limit", type=int, default=3)
    job_result = job_commands.add_parser("result")
    job_result.add_argument("name")
    job_result.add_argument("job_id")
    job_result.add_argument("task_id")
    job_checkpoints = job_commands.add_parser("checkpoints")
    job_checkpoints.add_argument("name")
    job_checkpoints.add_argument("job_id")
    job_checkpoints.add_argument("task_id")
    job_checkpoints.add_argument("--limit", type=int, default=8)
    job_artifacts = job_commands.add_parser("artifacts")
    job_artifacts.add_argument("name")
    job_artifacts.add_argument("job_id")
    job_artifacts.add_argument("--task-id")
    job_artifacts.add_argument("--limit", type=int, default=64)
    job_timeline = job_commands.add_parser("timeline")
    job_timeline.add_argument("name")
    job_timeline.add_argument("job_id")
    job_timeline.add_argument("--task-id")
    job_timeline.add_argument("--after", type=int, default=0)
    job_timeline.add_argument("--limit", type=int, default=100)
    job_board = job_commands.add_parser("board")
    job_board.add_argument("name")
    job_board.add_argument("job_id")
    job_blockers = job_commands.add_parser("blockers")
    job_blockers.add_argument("name")
    job_blockers.add_argument("job_id")
    job_answer = job_commands.add_parser("answer")
    job_answer.add_argument("name")
    job_answer.add_argument("job_id")
    job_answer.add_argument("task_id")
    job_answer.add_argument("control_id")
    job_answer.add_argument("--principal-id", required=True)
    job_answer.add_argument("--answer-json", required=True)
    job_answer.add_argument("--idempotency-key")
    job_accept = job_commands.add_parser("accept-review")
    job_accept.add_argument("name")
    job_accept.add_argument("job_id")
    job_accept.add_argument("task_id")
    job_accept.add_argument("control_id")
    job_accept.add_argument("--principal-id", required=True)
    job_accept.add_argument("--rationale", required=True)
    job_accept.add_argument("--idempotency-key", required=True)
    job_changes = job_commands.add_parser("request-changes")
    job_changes.add_argument("name")
    job_changes.add_argument("job_id")
    job_changes.add_argument("task_id")
    job_changes.add_argument("control_id")
    job_changes.add_argument("--principal-id", required=True)
    job_changes.add_argument("--rationale", required=True)
    job_changes.add_argument("--guidance", required=True)
    job_changes.add_argument("--idempotency-key", required=True)
    job_retry = job_commands.add_parser("retry-control")
    job_retry.add_argument("name")
    job_retry.add_argument("job_id")
    job_retry.add_argument("task_id")
    job_retry.add_argument("control_id")
    job_retry.add_argument("--principal-id", required=True)
    job_retry.add_argument("--note", required=True)
    job_retry.add_argument("--idempotency-key", required=True)
    job_reject = job_commands.add_parser("reject")
    job_reject.add_argument("name")
    job_reject.add_argument("job_id")
    job_reject.add_argument("task_id")
    job_reject.add_argument("control_id")
    job_reject.add_argument("--principal-id", required=True)
    job_reject.add_argument("--reason", required=True)
    job_reject.add_argument("--idempotency-key")
    job_cancel = job_commands.add_parser("cancel-graph")
    job_cancel.add_argument("name")
    job_cancel.add_argument("job_id")
    job_cancel.add_argument("--principal-id", required=True)
    job_replace = job_commands.add_parser("replace-task")
    job_replace.add_argument("name")
    job_replace.add_argument("job_id")
    job_replace.add_argument("task_id")
    job_replace.add_argument("expected_revision", type=int)
    job_replace.add_argument("--principal-id", required=True)
    job_replace.add_argument("--idempotency-key", required=True)
    job_replace.add_argument("--note", required=True)

    memory = commands.add_parser("memory", help="manage agent memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    for command in ("read", "edit"):
        action = memory_commands.add_parser(command)
        action.add_argument("name")
        action.add_argument(
            "--target",
            choices=("memory", "user"),
            default="memory",
        )
    memory_inspect = memory_commands.add_parser(
        "inspect",
        help="inspect bounded global and resource-scoped memory state",
    )
    memory_inspect.add_argument("name")
    memory_set = memory_commands.add_parser("set")
    memory_set.add_argument("name")
    memory_set.add_argument("--target", choices=("memory", "user"), required=True)
    memory_set.add_argument("--file", required=True)

    skills = commands.add_parser("skills", help="manage agent skills")
    skill_commands = skills.add_subparsers(dest="skills_command", required=True)
    skill_list = skill_commands.add_parser("list")
    skill_list.add_argument("name")
    for command in ("show", "edit"):
        action = skill_commands.add_parser(command)
        action.add_argument("name")
        action.add_argument("skill_name")
    skill_save = skill_commands.add_parser("save")
    skill_save.add_argument("name")
    skill_save.add_argument("skill_name")
    skill_save.add_argument("--description", required=True)
    skill_save.add_argument("--instructions-file", required=True)
    skill_delete_parser = skill_commands.add_parser("delete")
    skill_delete_parser.add_argument("name")
    skill_delete_parser.add_argument("skill_name")

    effects = commands.add_parser(
        "effects",
        help="inspect action receipts and record human recovery without replay",
    )
    effect_commands = effects.add_subparsers(dest="effects_command", required=True)
    effect_list = effect_commands.add_parser("list")
    effect_list.add_argument("name")
    effect_list.add_argument("--unresolved", action="store_true")
    effect_list.add_argument("--limit", type=int, default=20)
    effect_list.add_argument("--offset", type=int, default=0)
    effect_inspect = effect_commands.add_parser("inspect")
    effect_inspect.add_argument("name")
    effect_inspect.add_argument("receipt_id")
    effect_resolve = effect_commands.add_parser(
        "resolve", help="review one exact human decision; performs no action"
    )
    effect_resolve.add_argument("name")
    effect_resolve.add_argument("receipt_id")
    effect_resolve.add_argument("--expected-digest", required=True)
    effect_resolve.add_argument(
        "--decision",
        required=True,
        choices=tuple(item.value for item in EffectResolutionDecision),
    )
    effect_resolve.add_argument("--note", required=True)
    effect_resolve.add_argument("--evidence", action="append", default=[])

    routines = commands.add_parser(
        "routines",
        help=(
            "scheduled routines: a title, a prompt and a schedule; each run asks "
            "before any write (execution requires an open host)"
        ),
    )
    routine_commands = routines.add_subparsers(
        dest="routines_command",
        required=True,
    )
    routine_list = routine_commands.add_parser("list", help="list routines")
    routine_list.add_argument("name")
    routine_list.add_argument(
        "--state",
        action="append",
        choices=tuple(item.value for item in RoutineState),
    )
    routine_inspect = routine_commands.add_parser(
        "inspect", help="show a routine and its recent occurrences"
    )
    routine_inspect.add_argument("name")
    routine_inspect.add_argument("routine_id")
    for command in ("create", "update"):
        action = routine_commands.add_parser(
            command,
            help=(
                "save a new routine after confirmation; results go to a new "
                "conversation's Inbox"
                if command == "create"
                else "change a routine after confirmation; omitted fields keep "
                "their values"
            ),
        )
        action.add_argument("name")
        if command == "update":
            action.add_argument("routine_id")
            action.add_argument(
                "--expected-revision",
                type=int,
                help="refuse the change if the routine moved past this revision",
            )
        action.add_argument("--title")
        action.add_argument("--prompt", help="what each run should do")
        action.add_argument(
            "--schedule",
            help=(
                'JSON schedule, e.g. \'{"kind": "once", "after_seconds": 480}\', '
                '\'{"kind": "interval", "interval_seconds": 3600, '
                '"anchor_after_seconds": 60}\' or \'{"kind": "calendar", '
                '"timezone": "America/Chicago", "hour": 9, "minute": 0}\''
            ),
        )
        action.add_argument(
            "--source",
            action="append",
            help="source name or id to use; repeat for more (default: all sources)",
        )
        action.add_argument(
            "--spec",
            type=Path,
            help="JSON file with title, prompt, schedule, sources and "
            "run_immediately; flags override it",
        )
        if command == "create":
            action.add_argument(
                "--run-immediately",
                action="store_true",
                help="also run once now (interval and calendar schedules)",
            )
    for command in ("pause", "resume", "run-now", "disable"):
        action = routine_commands.add_parser(command, help=f"{command} a routine")
        action.add_argument("name")
        action.add_argument("routine_id")
        action.add_argument("expected_revision", type=int)

    work = commands.add_parser(
        "work",
        help="show the work board and inspect or dismiss results",
    )
    work_commands = work.add_subparsers(dest="work_command", required=True)
    work_show = work_commands.add_parser("show")
    work_show.add_argument("name")
    work_show.add_argument(
        "--done-hours",
        type=int,
        default=24,
        help="hours finished work stays in Done (1 to 168)",
    )
    for command in ("inspect", "dismiss"):
        action = work_commands.add_parser(command)
        action.add_argument("name")
        action.add_argument("delivery_id")
    return parser


def _absolute_path(path: Path | None) -> Path | None:
    return None if path is None else path.expanduser().absolute()


def _write_event_jsonl(event: AgentEvent) -> None:
    print(
        json.dumps(
            {
                "kind": event.kind.value,
                "occurred_at": event.occurred_at.isoformat(),
                "run_id": event.run_id,
                "conversation_id": event.conversation_id,
                "run_origin": event.run_origin,
                "data": event.data.to_dict(),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


class _RunEvents:
    """Collect user-run events from the host stream and write one run's in order.

    The stream carries every run of the agent, and this run's id is known only
    once it returns, so its events are written then, before the result.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[AgentEvent]] = {}
        self._completed: set[str] = set()
        self._changed = asyncio.Event()

    def observe(self, event: AgentEvent) -> None:
        if event.run_origin != "user":
            return
        self._events.setdefault(event.run_id, []).append(event)
        if event.kind is AgentEventKind.RUN_COMPLETED:
            self._completed.add(event.run_id)
            self._changed.set()

    async def write(self, run_id: str, timeout: float = 10.0) -> None:
        # run() returns on the run's state, which can overtake its last events;
        # a stream that never delivers them still leaves what did arrive.
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(timeout):
                while run_id not in self._completed:
                    self._changed.clear()
                    await self._changed.wait()
        for event in self._events.pop(run_id, ()):
            _write_event_jsonl(event)


def _model_configuration(
    model_id: str,
    *,
    base_url: str | None = None,
    context_window: int | None = None,
    max_output: int | None = None,
) -> tuple[ManagedModelProvider, ModelProfile]:
    if (context_window is None) != (max_output is None):
        raise ValueError("--context-window and --max-output must be supplied together")
    reviewed_profile = reviewed_model_profile(model_id)
    provider_max_output = (
        max_output
        if max_output is not None
        else (
            reviewed_profile.max_output_tokens
            if reviewed_profile is not None
            else 1_024
        )
    )
    provider = create_llm_provider(
        model_id,
        base_url=base_url,
        max_output_tokens=provider_max_output,
    )
    profile = _provider_profile(provider)
    if context_window is None:
        return provider, profile
    assert max_output is not None
    if (
        context_window > profile.context_window_tokens
        or max_output > profile.max_output_tokens
    ):
        raise ValueError(
            "explicit token limits cannot exceed the reviewed or provider-owned "
            "model profile"
        )
    return provider, ModelProfile(
        id=profile.id,
        context_window_tokens=context_window,
        max_output_tokens=max_output,
        supports_tools=profile.supports_tools,
        supports_parallel_tools=profile.supports_parallel_tools,
        supports_streaming=profile.supports_streaming,
    )


def _provider_profile(provider: ManagedModelProvider) -> ModelProfile:
    """Resolve only reviewed or explicit provider-owned model facts."""

    profile = reviewed_model_profile(provider.provider_id)
    if profile is None:
        owned_profile = getattr(provider, "model_profile", None)
        if isinstance(owned_profile, ModelProfile):
            profile = owned_profile
    if profile is None or profile.id != provider.provider_id:
        raise ValueError(
            "unreviewed models must be configured in the interactive terminal "
            "to prove tool support and establish explicit hard token limits"
        )
    return profile


async def _close_cli_resources(
    *,
    agent: Agent | None,
    provider: ManagedModelProvider | None,
) -> None:
    """Release caller-owned CLI resources while preserving the first failure."""

    first_error: BaseException | None = None
    if agent is not None:
        try:
            await agent.close()
        except BaseException as error:
            first_error = error
    if provider is not None:
        try:
            await provider.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _require_chat_terminal() -> None:
    streams = (sys.stdin, sys.stdout, sys.stderr)
    if not all(stream.isatty() for stream in streams):
        raise RuntimeError(
            "chat requires an interactive terminal on stdin, stdout, and stderr"
        )


def _require_terminal_application() -> None:
    streams = (sys.stdin, sys.stdout, sys.stderr)
    if not all(stream.isatty() for stream in streams):
        raise RuntimeError("daita requires interactive stdin, stdout, and stderr")


def _read_input_document(location: str) -> str:
    if location == "-":
        return sys.stdin.read()
    return Path(location).read_text(encoding="utf-8")


def _editor_command() -> list[str]:
    value = os.environ.get("EDITOR")
    if value is None or not value.strip():
        raise RuntimeError("$EDITOR is not set; set it to an available editor command")
    try:
        command = shlex.split(value)
    except ValueError as error:
        raise RuntimeError(
            "$EDITOR is malformed; set it to a valid editor command"
        ) from error
    if not command:
        raise RuntimeError("$EDITOR is empty; set it to an available editor command")
    return command


def _editor_temporary_directory(agent_home: Path) -> Path:
    home = agent_home.resolve(strict=True)
    candidates = (Path(tempfile.gettempdir()), Path("/tmp"))
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved != home and home not in resolved.parents:
            return resolved
    raise RuntimeError("no temporary directory is available outside the agent home")


def _edit_document(seed: str, *, agent_home: Path) -> str:
    command = _editor_command()
    temporary_directory = _editor_temporary_directory(agent_home)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="daita-edit-",
            suffix=".md",
            delete=False,
            dir=temporary_directory,
        ) as temporary:
            temporary.write(seed)
            temporary_path = Path(temporary.name)
        try:
            completed = subprocess.run(
                [*command, str(temporary_path)],
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"$EDITOR command is unavailable: {command[0]}"
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"$EDITOR exited with status {completed.returncode}; no changes were saved"
            )
        return temporary_path.read_text(encoding="utf-8")
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _render_skill_editor_document(
    name: str,
    description: str,
    instructions: str,
) -> str:
    return f"# {name}\n\n{description}\n\n## Instructions\n\n{instructions}\n"


def _parse_skill_editor_document(name: str, text: str) -> tuple[str, str]:
    prefix = f"# {name}\n\n"
    marker = "\n\n## Instructions\n\n"
    if not text.startswith(prefix) or not text.endswith("\n"):
        raise ValueError(
            "edited skill must keep the exact '# <name>' header and final newline"
        )
    body = text[len(prefix) : -1]
    if body.count(marker) != 1:
        raise ValueError(
            "edited skill must contain exactly one '## Instructions' section"
        )
    description, instructions = body.split(marker, 1)
    return description, instructions


async def _read_memory_target(agent: Agent, target: str) -> str:
    if target == "memory":
        return await agent.read_memory()
    return await agent.read_user_profile()


async def _set_memory_target(agent: Agent, target: str, content: str) -> None:
    if target == "memory":
        await agent.set_memory(content)
    else:
        await agent.set_user_profile(content)


async def _edit_memory_target(agent: Agent, target: str) -> None:
    current = await _read_memory_target(agent, target)
    edited = _edit_document(current, agent_home=agent.home)
    await _set_memory_target(agent, target, edited)


async def _edit_skill(agent: Agent, name: str) -> bool:
    skill = await agent.read_skill(name)
    if skill is None:
        raise ValueError(f"skill not found: {name}")
    document = _render_skill_editor_document(
        name,
        skill.description,
        skill.instructions,
    )
    edited = _edit_document(document, agent_home=agent.home)
    description, instructions = _parse_skill_editor_document(name, edited)
    return await agent.save_skill(name, description, instructions)


async def _prompt_for_exact_approval(
    request: ApprovalRequest,
) -> ApprovalDecision:
    from .tui.projection import approval_review_document

    document, reviewable = approval_review_document(
        tool_name=request.tool_name,
        capability_id=request.capability_id,
        arguments_text=request.render_arguments_for_review(),
        reason=request.reason,
    )
    if not reviewable or document is None:
        print("Approval unavailable: exact details cannot be safely reviewed.")
        return ApprovalDecision.DENY
    print("Approval required")
    print()
    print(document)
    while True:
        try:
            # Off the event loop, so a host client keeps reading its stream.
            answer = await asyncio.to_thread(
                input, "Approve this exact change once? [y/n]"
            )
        except EOFError:
            print()
            return ApprovalDecision.DENY
        normalized = answer.strip().casefold()
        if normalized == "y":
            return ApprovalDecision.APPROVE
        if normalized == "n":
            return ApprovalDecision.DENY
        print("Enter y to approve or n to deny.")


def _resolve_cli_workspace(args: argparse.Namespace) -> LocalWorkspace:
    sensitivity = ModelSensitivity(args.workspace_sensitivity)
    explicit = args.workspace
    if explicit is not None:
        return LocalWorkspace(
            explicit,
            sensitivity=sensitivity,
            access=LocalFileAccess.COMPUTER,
        )

    state_root = (
        (Path.home() / ".daita")
        if args.root is None
        else Path(os.path.abspath(os.fspath(args.root))).resolve(strict=False)
    )
    state_root = state_root.resolve(strict=False)
    return LocalWorkspace(
        working_directory(Path.cwd().resolve(strict=True), state_root),
        sensitivity=sensitivity,
        access=LocalFileAccess.COMPUTER,
    )


def _host_workspace(workspace: LocalWorkspace) -> None:
    if workspace.sensitivity is not LocalWorkspace(workspace.root).sensitivity:
        raise ValueError(
            f"--workspace-sensitivity {workspace.sensitivity.value} needs --no-host: "
            "the Daita host opens folders at the default sensitivity"
        )


async def _open_agent(
    args: argparse.Namespace,
    workspace: LocalWorkspace,
    library: Mapping[str, typing.Any],
    *,
    observer: AgentObserver | None = None,
    approval_handler: ApprovalHandler | None = None,
) -> Agent:
    """Open the agent through the Daita host, or in this process with --no-host.

    ``library`` holds the in-process composition; it never crosses the socket.
    """

    if args.no_host:
        return await Agent.open(
            args.name, workspace=workspace, root=args.root, **library
        )
    from .host.client import RemoteAgent

    _host_workspace(workspace)
    remote = await RemoteAgent.open(
        args.name,
        workspace=workspace,
        root=args.root,
        observer=observer,
        approval_handler=approval_handler,
    )
    # RemoteAgent mirrors every Agent member the CLI calls.
    return typing.cast(Agent, remote)


async def _deny_without_prompting(request: ApprovalRequest) -> ApprovalDecision:
    # A one-shot run never prompts or writes; in process it has no handler.
    del request
    return ApprovalDecision.DENY


def _routine_intent(args: argparse.Namespace) -> RoutineIntent:
    """Read one routine from a small spec file and flags; flags win."""

    document: dict[str, object] = {}
    if args.spec is not None:
        try:
            value = json.loads(args.spec.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"routine spec is not valid JSON: {error.msg}") from error
        document.update(_object_mapping(value, "routine spec"))
    if args.schedule is not None:
        try:
            document["schedule"] = json.loads(args.schedule)
        except json.JSONDecodeError as error:
            raise ValueError(f"--schedule is not valid JSON: {error.msg}") from error
    for name in ("title", "prompt"):
        if getattr(args, name) is not None:
            document[name] = getattr(args, name)
    if args.source:
        document["sources"] = args.source
    if getattr(args, "run_immediately", False):
        document["run_immediately"] = True
    return RoutineIntent.from_mapping(document)


def _object_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value


async def _execute(args: argparse.Namespace) -> object:
    if args.command == "state":
        status = await Agent.inspect_home(args.name, root=args.root)
        return {
            "current_revision": status.current_revision,
            "found_revision": status.found_revision,
            "minimum_supported_revision": status.minimum_supported_revision,
            "recovery_required": status.recovery_required,
            "source_kind": status.source_kind,
            "upgrade_required": status.upgrade_required,
        }
    if args.command == "delete":
        if not args.yes:
            raise ValueError("delete requires --yes")
        if args.no_host:
            await Agent.delete(args.name, root=args.root)
        else:
            from .host.client import RemoteAgent

            # The host closes the agent first if it holds it.
            await RemoteAgent.delete(args.name, root=args.root)
        return {"name": args.name, "deleted": True}
    if args.command == "diagnostics":
        return {"runs": summarize_diagnostics(args.path)}
    workspace = _resolve_cli_workspace(args)
    if args.command == "create":
        if args.no_host:
            agent = await Agent.create(args.name, workspace=workspace, root=args.root)
        else:
            from .host.client import RemoteAgent

            _host_workspace(workspace)
            created = await RemoteAgent.create(
                args.name, workspace=workspace, root=args.root
            )
            agent = typing.cast(Agent, created)
        try:
            return {"agent_id": agent.id, "name": agent.name, "home": str(agent.home)}
        finally:
            await agent.close()
    if args.command == "host":
        raise ValueError("choose a host command: serve, start, stop, status or logs")
    if args.command == "jobs":
        agent = await _open_agent(args, workspace, {"config": AgentConfig()})
        try:
            if args.jobs_command == "list":
                job_states = frozenset(GraphState(item) for item in (args.state or ()))
                jobs = await agent.list_jobs(states=job_states, limit=args.limit)
                return [
                    {
                        "job_id": job.job_id,
                        "state": job.state.value,
                        "desired_state": job.desired_state.value,
                        "updated_at": job.updated_at.isoformat(),
                        "deadline_at": job.deadline_at.isoformat(),
                    }
                    for job in jobs
                ]
            if args.jobs_command == "inspect":
                job_inspection = await agent.inspect_job(args.job_id)
                return (
                    None
                    if job_inspection is None
                    else _graph_inspection_mapping(job_inspection)
                )
            if args.jobs_command == "tasks":
                task_states = frozenset(TaskState(item) for item in (args.state or ()))
                tasks = await agent.list_job_tasks(
                    args.job_id, states=task_states, limit=args.limit
                )
                return [
                    {
                        "task_id": task.task_id,
                        "role": task.role.value,
                        "state": task.state.value,
                        "task_revision": task.task_revision,
                        "current_attempt_id": task.current_attempt_id,
                        "latest_result_id": task.latest_result_id,
                        "latest_control_id": task.latest_control_id,
                    }
                    for task in tasks
                ]
            if args.jobs_command == "dependencies":
                edges = await agent.list_job_dependencies(
                    args.job_id, task_id=args.task_id, limit=args.limit
                )
                return [
                    {
                        "upstream_task_id": edge.upstream_task_id,
                        "downstream_task_id": edge.downstream_task_id,
                        "edge_kind": edge.edge_kind.value,
                    }
                    for edge in edges
                ]
            if args.jobs_command == "attempts":
                attempts = await agent.list_task_attempts(
                    args.job_id, args.task_id, limit=args.limit
                )
                return [
                    {
                        "attempt_id": attempt.attempt_id,
                        "ordinal": attempt.ordinal,
                        "state": attempt.state.value,
                        "fencing_epoch": attempt.fencing_epoch,
                        "run_id": attempt.run_id,
                        "checkpoint_ids": attempt.checkpoint_ids,
                        "artifact_ids": attempt.artifact_ids,
                        "error_code": attempt.error_code,
                    }
                    for attempt in attempts
                ]
            if args.jobs_command == "result":
                task_result = await agent.read_task_result(args.job_id, args.task_id)
                return (
                    None
                    if task_result is None
                    else {
                        **task_result.candidate_material(),
                        "completed_at": task_result.completed_at.isoformat(),
                    }
                )
            if args.jobs_command == "checkpoints":
                checkpoints = await agent.list_task_checkpoints(
                    args.job_id, args.task_id, limit=args.limit
                )
                return [
                    {
                        "checkpoint_id": item.checkpoint_id,
                        "attempt_id": item.attempt_id,
                        "ordinal": item.ordinal,
                        "milestone": item.milestone,
                        "payload": item.payload,
                        "payload_digest": item.payload_digest,
                    }
                    for item in checkpoints
                ]
            if args.jobs_command == "artifacts":
                return await agent.list_task_artifacts(
                    args.job_id, task_id=args.task_id, limit=args.limit
                )
            if args.jobs_command == "timeline":
                timeline = await agent.job_timeline(
                    args.job_id,
                    after_event_id=args.after,
                    limit=args.limit,
                    task_id=args.task_id,
                )
                if timeline is None:
                    return None
                return {
                    "job_id": timeline.job_id,
                    "graph_state": timeline.graph_state.value,
                    "graph_revision": timeline.graph_revision,
                    "events": [
                        {
                            "event_id": item.event_id,
                            "kind": item.kind,
                            "task_id": item.task_id,
                            "attempt_id": item.attempt_id,
                            "created_at": item.created_at.isoformat(),
                            "payload": item.payload,
                        }
                        for item in timeline.events
                    ],
                    "next_cursor": timeline.next_cursor,
                    "diagnostics": {
                        "blockers": timeline.diagnostics.blockers,
                        "exhausted_budgets": timeline.diagnostics.exhausted_budgets,
                        "deadlocked": timeline.diagnostics.deadlocked,
                        "deadlock_reason": timeline.diagnostics.deadlock_reason,
                    },
                }
            if args.jobs_command == "board":
                board = await agent.job_board(args.job_id)
                if board is None:
                    return None
                return {
                    "job_id": board.job_id,
                    "graph_state": board.graph_state.value,
                    "graph_revision": board.graph_revision,
                    "columns": [
                        {"name": item.name, "task_ids": item.task_ids}
                        for item in board.columns
                    ],
                    "dependencies": [
                        {
                            "upstream_task_id": item.upstream_task_id,
                            "downstream_task_id": item.downstream_task_id,
                            "edge_kind": item.edge_kind,
                            "satisfied": item.satisfied,
                        }
                        for item in board.dependencies
                    ],
                    "diagnostics": {
                        "blockers": board.diagnostics.blockers,
                        "exhausted_budgets": board.diagnostics.exhausted_budgets,
                        "deadlocked": board.diagnostics.deadlocked,
                        "deadlock_reason": board.diagnostics.deadlock_reason,
                    },
                }
            if args.jobs_command == "blockers":
                blockers = await agent.graph_blockers(args.job_id)
                return None if blockers is None else _graph_blockers_mapping(blockers)
            if args.jobs_command == "answer":
                answer = _object_mapping(
                    json.loads(args.answer_json), "task input answer"
                )
                control = await agent.answer_task_input(
                    args.job_id,
                    args.task_id,
                    args.control_id,
                    principal_id=args.principal_id,
                    answer=answer,
                    idempotency_key=args.idempotency_key,
                )
                return (
                    None
                    if control is None
                    else {
                        "control_id": control.control_id,
                        "state": control.state.value,
                        "resolved_by_id": control.resolved_by_id,
                    }
                )
            if args.jobs_command == "accept-review":
                reviewed_task_result = await agent.accept_task_review(
                    args.job_id,
                    args.task_id,
                    args.control_id,
                    principal_id=args.principal_id,
                    rationale=args.rationale,
                    idempotency_key=args.idempotency_key,
                )
                return {
                    "result_id": reviewed_task_result.result_id,
                    "result_digest": reviewed_task_result.result_digest,
                    "task_id": reviewed_task_result.task_id,
                }
            if args.jobs_command == "request-changes":
                control = await agent.request_task_review_changes(
                    args.job_id,
                    args.task_id,
                    args.control_id,
                    principal_id=args.principal_id,
                    rationale=args.rationale,
                    replacement_guidance=args.guidance,
                    idempotency_key=args.idempotency_key,
                )
                return {
                    "control_id": control.control_id,
                    "kind": control.kind.value,
                    "state": control.state.value,
                }
            if args.jobs_command == "retry-control":
                control = await agent.retry_task_control(
                    args.job_id,
                    args.task_id,
                    args.control_id,
                    principal_id=args.principal_id,
                    advisory_note=args.note,
                    idempotency_key=args.idempotency_key,
                )
                return (
                    None
                    if control is None
                    else {
                        "control_id": control.control_id,
                        "state": control.state.value,
                    }
                )
            if args.jobs_command == "reject":
                control = await agent.reject_task_control(
                    args.job_id,
                    args.task_id,
                    args.control_id,
                    principal_id=args.principal_id,
                    reason=args.reason,
                    idempotency_key=args.idempotency_key,
                )
                return (
                    None
                    if control is None
                    else {
                        "control_id": control.control_id,
                        "state": control.state.value,
                        "resolved_by_id": control.resolved_by_id,
                    }
                )
            if args.jobs_command == "cancel-graph":
                job = await agent.cancel_graph_job(
                    args.job_id, principal_id=args.principal_id
                )
                return (
                    None
                    if job is None
                    else {
                        "job_id": job.job_id,
                        "state": job.state.value,
                        "desired_state": job.desired_state.value,
                    }
                )
            mutation = await agent.replace_graph_task_by_policy(
                args.job_id,
                args.task_id,
                principal_id=args.principal_id,
                advisory_note=args.note,
                idempotency_key=args.idempotency_key,
                expected_revision=args.expected_revision,
            )
            return {
                "mutation_id": mutation.mutation_id,
                "decision": mutation.decision.value,
                "failure_code": mutation.failure_code,
                "committed_revision": mutation.committed_revision,
                "task_ids": mutation.resulting_task_ids,
            }
        finally:
            await agent.close()
    if args.command == "detach":
        if not args.yes:
            raise ValueError("detach requires --yes")
        agent = await _open_agent(args, workspace, {"config": AgentConfig()})
        try:
            detached_source = await agent.detach(args.source_id)
            return {
                "source_id": detached_source.id,
                "name": detached_source.display_name,
                "detached": True,
            }
        finally:
            await agent.close()
    if args.command == "conversations":
        if not args.yes:
            raise ValueError("conversations clear requires --yes")
        agent = await _open_agent(args, workspace, {"config": AgentConfig()})
        try:
            return {
                "name": args.name,
                "cleared_runs": await agent.clear_conversations(),
            }
        finally:
            await agent.close()
    if args.command == "run":
        if args.model is None and any(
            value is not None
            for value in (args.base_url, args.context_window, args.max_output)
        ):
            raise ValueError(
                "--base-url, --context-window, and --max-output require --model"
            )
        local = tuple(
            flag
            for flag, value in (
                ("--model", args.model),
                ("--max-cost-usd", args.max_cost_usd),
                ("--diagnostics-path", args.diagnostics_path),
            )
            if value is not None
        )
        if local and not args.no_host:
            raise ValueError(
                f"{', '.join(local)}: only with --no-host; the Daita host runs "
                "the agent with its saved model settings"
            )
        provider: ManagedModelProvider | None = None
        run_agent: Agent | None = None
        events = _RunEvents() if args.events_jsonl and not args.no_host else None
        try:
            if not args.no_host:
                run_agent = await _open_agent(
                    args,
                    workspace,
                    {},
                    observer=None if events is None else events.observe,
                    approval_handler=_deny_without_prompting,
                )
            else:
                library: dict[str, object] = {
                    "observer": _write_event_jsonl if args.events_jsonl else None,
                    "diagnostics_path": _absolute_path(args.diagnostics_path),
                    "limits": (
                        None
                        if args.max_cost_usd is None
                        else LoopLimits(max_estimated_cost_usd=args.max_cost_usd)
                    ),
                }
                if args.model is not None:
                    provider, profile = _model_configuration(
                        args.model,
                        base_url=args.base_url,
                        context_window=args.context_window,
                        max_output=args.max_output,
                    )
                    library.update(model=provider, model_profile=profile)
                run_agent = await _open_agent(args, workspace, library)
            try:
                result = (
                    await run_agent.run(
                        args.message,
                        conversation_id=args.conversation_id,
                        files_only=True,
                    )
                    if args.files_only
                    else await run_agent.run(
                        args.message,
                        conversation_id=args.conversation_id,
                    )
                )
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is None or task.cancelling():
                    raise
                # The host ended the run: it shut down, or a question expired.
                raise RuntimeError(
                    "the run was interrupted in the Daita host"
                ) from None
            if events is not None:
                await events.write(result.run_id)
            transcript = await run_agent.transcript(result.run_id)
            analysis = await run_agent.analysis_usage(result.run_id)
            return {
                "run_id": result.run_id,
                "conversation_id": result.conversation_id,
                "status": result.kind.value,
                "reason": result.reason,
                "text": result.final_text,
                "steps": result.steps,
                "analysis_usage": analysis,
                "model_usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "cost_usd": (
                        None
                        if result.usage.cost_estimate.amount_usd is None
                        else str(result.usage.cost_estimate.amount_usd)
                    ),
                },
                "notice": (
                    None
                    if result.kind.value == "completed"
                    else run_failure_notice(result, transcript)
                ),
                "tool_results": tuple(
                    {
                        "call_id": call.id,
                        "tool_name": call.name,
                        "is_error": None if outcome is None else outcome.is_error,
                        "summary": (
                            None if outcome is None else tool_outcome_summary(outcome)
                        ),
                    }
                    for call, outcome in transcript.tool_pairs
                ),
                "artifacts": tuple(
                    artifact_ref_to_mapping(item) for item in result.artifacts
                ),
                "artifact_deliveries": tuple(
                    artifact_delivery_receipt_to_mapping(item)
                    for item in result.artifact_deliveries
                ),
            }
        finally:
            await _close_cli_resources(agent=run_agent, provider=provider)
    if args.command == "chat":
        return await run_terminal_application(
            root=args.root,
            workspace=workspace,
            agent_name=args.name,
            conversation_id=args.conversation,
        )
    agent = await _open_agent(
        args,
        workspace,
        {"approval_handler": _prompt_for_exact_approval},
        approval_handler=_prompt_for_exact_approval,
    )
    owned_mcp_credential: SecretReference | None = None
    try:
        if args.command == "effects":
            if args.effects_command == "list":
                return [
                    _effect_receipt_mapping(item)
                    for item in await agent.list_effects(
                        unresolved_only=args.unresolved,
                        limit=args.limit,
                        offset=args.offset,
                    )
                ]
            effect: EffectReceipt | None
            if args.effects_command == "resolve":
                effect = await agent.resolve_effect(
                    args.receipt_id,
                    expected_digest=args.expected_digest,
                    decision=EffectResolutionDecision(args.decision),
                    note=args.note,
                    evidence_references=tuple(args.evidence),
                )
            else:
                effect = await agent.inspect_effect(args.receipt_id)
                if effect is None:
                    raise ValueError("effect receipt not found")
            return _effect_receipt_mapping(effect)
        if args.command == "artifacts":
            receipt = await agent.save_artifact(
                args.artifact_id,
                args.destination,
                filename=args.filename,
            )
            return artifact_delivery_receipt_to_mapping(receipt)
        if args.command == "export-location":
            if args.export_location_command == "get":
                destination = await agent.export_destination()
            elif args.export_location_command == "set":
                destination = await agent.set_export_destination(args.directory)
            else:
                destination = await agent.reset_export_destination()
            return artifact_destination_to_mapping(destination)
        if args.command == "attach":
            source = _source_from_attach_args(args)
            registration = await agent.attach(source)
            return {
                "source_id": registration.id,
                "adapter": registration.adapter_id,
                "name": registration.display_name,
            }
        if args.command == "mcp":
            if args.mcp_command in {"inspect", "attach"}:
                mcp_authentication = _mcp_authentication(
                    args.bearer_env, args.bearer_ref
                )
                if args.bearer_prompt:
                    credential = await asyncio.to_thread(
                        getpass.getpass, "MCP bearer token: "
                    )
                    owned_mcp_credential = await agent.store_mcp_bearer(credential)
                    credential = ""
                    mcp_authentication = MCPAuthentication.bearer(owned_mcp_credential)
            if args.mcp_command == "inspect":
                inspection = await agent.inspect_mcp_server(
                    endpoint=args.endpoint,
                    authentication=mcp_authentication,
                )
                return _mcp_inspection_mapping(inspection)
            if args.mcp_command == "attach":
                if any(len(item) not in {1, 4} for item in args.tool):
                    raise ValueError(
                        "--tool requires REMOTE or REMOTE ALIAS DESCRIPTION RESULT_SENSITIVITY"
                    )
                selections = tuple(
                    (
                        MCPToolSelection(item[0])
                        if len(item) == 1
                        else MCPToolSelection(
                            item[0],
                            item[1],
                            item[2],
                            result_sensitivity=ModelSensitivity(item[3]),
                        )
                    )
                    for item in args.tool
                )
                attached_status = await agent.attach_mcp_server(
                    endpoint=args.endpoint,
                    authentication=mcp_authentication,
                    maximum_outbound_sensitivity=ModelSensitivity(
                        args.maximum_outbound_sensitivity
                    ),
                    selections=selections,
                    binding_id=args.binding_id,
                )
                return _mcp_status_mapping(attached_status)
            if args.mcp_command == "status":
                statuses = await agent.list_mcp_servers()
                if args.binding_id is not None:
                    statuses = tuple(
                        status
                        for status in statuses
                        if status.binding.binding_id == args.binding_id
                    )
                    if not statuses:
                        raise ValueError("MCP binding does not exist")
                return [_mcp_status_mapping(status) for status in statuses]
            if args.mcp_command == "refresh":
                return _mcp_status_mapping(
                    await agent.refresh_mcp_server(args.binding_id)
                )
            if not args.yes:
                raise ValueError("mcp revoke requires --yes")
            return _mcp_status_mapping(await agent.revoke_mcp_server(args.binding_id))
        if args.command == "relational-update-readiness":
            readiness_result = await agent.relational_update_readiness(
                args.source_id,
                args.resource_id,
                tuple(args.assignment_columns),
            )
            return {
                "readiness": readiness_result.to_mapping(),
                "guidance": (
                    "Apply any role or grant remediation externally; Daita does "
                    "not accept administrator credentials."
                ),
            }
        if args.command == "memory":
            if args.memory_command == "read":
                return {
                    "target": args.target,
                    "content": await _read_memory_target(agent, args.target),
                }
            if args.memory_command == "inspect":
                return {
                    "global_memory": await agent.read_memory(),
                    "annotations": [
                        {
                            "id": view.annotation.id,
                            "kind": view.annotation.kind.value,
                            "state": view.state.value,
                            "source_ids": view.annotation.subject.source_ids,
                            "resource_ids": view.annotation.subject.resource_ids,
                            "field_count": len(view.annotation.subject.fields),
                            "stale_reasons": view.stale_reasons,
                            "conflicting_ids": view.conflicting_ids,
                            "duplicate_ids": view.duplicate_ids,
                            "duplicate_of_id": view.duplicate_of_id,
                            "superseded_by_id": view.superseded_by_id,
                            "requires_revalidation": view.requires_revalidation,
                        }
                        for view in await agent.list_semantic_annotations()
                    ],
                }
            if args.memory_command == "edit":
                await _edit_memory_target(agent, args.target)
                return {"target": args.target, "updated": True}
            content = _read_input_document(args.file)
            await _set_memory_target(agent, args.target, content)
            return {"target": args.target, "updated": True}
        if args.command == "skills":
            if args.skills_command == "list":
                return [
                    {"name": skill.name, "description": skill.description}
                    for skill in await agent.list_skills()
                ]
            if args.skills_command == "show":
                skill = await agent.read_skill(args.skill_name)
                if skill is None:
                    raise ValueError(f"skill not found: {args.skill_name}")
                return {
                    "name": skill.name,
                    "description": skill.description,
                    "instructions": skill.instructions,
                }
            if args.skills_command == "edit":
                changed = await _edit_skill(agent, args.skill_name)
                return {"name": args.skill_name, "changed": changed}
            if args.skills_command == "save":
                instructions = _read_input_document(args.instructions_file)
                changed = await agent.save_skill(
                    args.skill_name,
                    args.description,
                    instructions,
                )
                return {"name": args.skill_name, "changed": changed}
            deleted = await agent.delete_skill(args.skill_name)
            return {"name": args.skill_name, "deleted": deleted}
        if args.command == "work":
            if args.work_command == "show":
                return work_board_projection(
                    await agent.work_board(done_window=timedelta(hours=args.done_hours))
                )
            if args.work_command == "inspect":
                delivery_inspection = await agent.inspect_delivery(args.delivery_id)
                if delivery_inspection is None:
                    raise ValueError("delivery not found")
                return delivery_inspection_projection(delivery_inspection)
            dismissed = await agent.acknowledge_delivery(args.delivery_id)
            if dismissed is None:
                raise ValueError("delivery not found")
            return delivery_view_projection(dismissed)
        if args.command == "routines":
            if args.routines_command == "list":
                states = frozenset(RoutineState(item) for item in (args.state or ()))
                return [
                    {
                        "routine_id": item.routine_id,
                        "title": item.title,
                        "state": item.state.value,
                        "schedule_kind": item.schedule_kind.value,
                        "next_due_at": (
                            None
                            if item.next_due_at is None
                            else item.next_due_at.isoformat()
                        ),
                        "revision": item.revision,
                        "occurrence_count": item.occurrence_count,
                        "consecutive_failures": item.consecutive_failures,
                    }
                    for item in await agent.list_routines(states=states)
                ]
            if args.routines_command == "inspect":
                routine_inspection = await agent.inspect_routine(args.routine_id)
                if routine_inspection is None:
                    raise ValueError("routine not found")
                return routine_inspection_projection(routine_inspection)
            if args.routines_command == "create":
                return routine_projection(
                    await agent.create_routine(
                        _routine_intent(args),
                        confirmation_handler=_prompt_for_exact_approval,
                    )
                )
            if args.routines_command == "update":
                return routine_projection(
                    await agent.update_routine(
                        args.routine_id,
                        _routine_intent(args),
                        expected_revision=args.expected_revision,
                        confirmation_handler=_prompt_for_exact_approval,
                    )
                )
            routine_control = {
                "pause": agent.pause_routine,
                "resume": agent.resume_routine,
                "run-now": agent.run_routine_now,
                "disable": agent.disable_routine,
            }[args.routines_command]
            return routine_projection(
                await routine_control(
                    args.routine_id,
                    expected_revision=args.expected_revision,
                )
            )
        return [
            {
                "source_id": item.id,
                "adapter": item.adapter_id,
                "name": item.display_name,
                "active": item.active,
            }
            for item in await agent.list_sources()
        ]
    finally:
        try:
            if owned_mcp_credential is not None and not any(
                status.binding.authentication.secret_reference == owned_mcp_credential
                for status in await agent.list_mcp_servers()
            ):
                await agent.delete_mcp_bearer(owned_mcp_credential)
        finally:
            await agent.close()


def _source_from_attach_args(
    args: argparse.Namespace,
) -> SQLiteSource | PostgreSQLSource | FolderSource:
    if args.kind == "sqlite":
        if args.path is None:
            raise ValueError("attach sqlite requires a path")
        return SQLiteSource(args.path)
    if args.kind == "folder":
        if args.path is None:
            raise ValueError("attach folder requires a path")
        return FolderSource(args.path, name=args.source_name)
    if args.kind != "postgresql":
        raise ValueError(f"attach {args.kind} has no command-line form")
    if args.path is not None:
        raise ValueError("attach postgresql does not accept a path")
    required = {
        "--host": args.host,
        "--database": args.database,
        "--username": args.username,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError("attach postgresql requires " + ", ".join(missing))
    credential = (
        None
        if args.password_env is None
        else SecretReference.environment(args.password_env)
    )
    return PostgreSQLSource(
        host=args.host,
        port=args.port,
        database=args.database,
        username=args.username,
        credential=credential,
        schemas=tuple(args.schemas or ("public",)),
        ssl_mode=args.ssl_mode,
        name=args.source_name,
    )


def _host_command(args: argparse.Namespace) -> int:
    from .host import service
    from .host.paths import host_paths
    from .host.supervise import host_status, log_tail, start_host, stop_host

    paths = host_paths(args.root)
    if args.host_command == "serve":
        return _serve_host(args, paths)
    if args.host_command == "install":
        for note in service.install(paths, start=not args.no_start):
            print(note)
        return 0
    if args.host_command == "uninstall":
        print(
            "Daita host service removed."
            if service.uninstall(paths)
            else "The Daita host service was not installed."
        )
        return 0
    if args.host_command == "start":
        status = start_host(paths)
        print(f"Daita host {'started' if status['started'] else 'already running'}.")
        return 0
    if args.host_command == "stop":
        print(
            "Daita host stopped." if stop_host(paths) else "Daita host is not running."
        )
        return 0
    if args.host_command == "status":
        status = host_status(paths) or {"running": False}
        status["service_installed"] = service.installed()
        if args.as_json:
            print(json.dumps(status, default=str, sort_keys=True))
        elif not status["running"]:
            print("Daita host: stopped")
        else:
            print(
                f"Daita host: running (pid {status.get('pid', 'unknown')}, "
                f"version {status['version']})"
            )
            if status.get("version_mismatch"):
                print(f"This command is version {__version__}; restart the host.")
            else:
                print(f"Agents open: {', '.join(status['agents']) or 'none'}")  # type: ignore[arg-type]
                print(f"Socket: {status['socket']}")
        if not args.as_json:
            print(
                "Service: installed (starts at login)"
                if status["service_installed"]
                else "Service: not installed (daita host install)"
            )
        print(f"Log: {paths.log}")
        return 0
    lines = log_tail(paths)
    print("".join(lines), end="")
    if args.follow:
        _follow_log(paths.log)
    return 0


def _serve_host(args: argparse.Namespace, paths: object) -> int:
    import logging
    import signal
    from logging.handlers import RotatingFileHandler

    from .host.notify import desktop_sender
    from .host.paths import HostPaths
    from .host.server import HostAlreadyRunning, serve_host

    assert isinstance(paths, HostPaths)
    os.makedirs(paths.log_dir, mode=0o700, exist_ok=True)
    handler = RotatingFileHandler(paths.log, maxBytes=5 * 1024 * 1024, backupCount=5)
    handlers: list[logging.Handler] = [handler]
    if args.foreground:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    def reopen_log() -> None:
        # A FileHandler reopens its file on the next record once its stream is None.
        handler.acquire()
        try:
            if handler.stream is not None:
                handler.stream.close()
            handler.stream = None
        finally:
            handler.release()

    async def serve() -> None:
        asyncio.get_running_loop().add_signal_handler(signal.SIGHUP, reopen_log)
        # Started by a client or the service: exit when idle unless installed.
        await serve_host(
            paths, handle_signals=True, idle_exit=args.auto, notify=desktop_sender()
        )

    try:
        asyncio.run(serve())
    except HostAlreadyRunning as error:
        print(str(error), file=sys.stderr)
        return 0 if args.auto else 1
    return 0


def _follow_log(path: Path) -> None:
    import time

    with open(path, encoding="utf-8", errors="replace") as file:
        file.seek(0, os.SEEK_END)
        while True:
            line = file.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.5)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        result: object
        if args.no_host and args.command in (None, "chat"):
            raise ValueError("the terminal app always uses the Daita host")
        if args.command == "host" and args.host_agent is not None:
            print(
                "daita host --agent is deprecated; the Daita host now serves every "
                "agent. Running daita host serve.",
                file=sys.stderr,
            )
            root = [] if args.root is None else ["--root", str(args.root)]
            args = build_parser().parse_args([*root, "host", "serve"])
        if args.command == "host" and args.host_command is not None:
            return _host_command(args)
        if args.command is None:
            _require_terminal_application()
            result = asyncio.run(
                run_terminal_application(
                    root=args.root,
                    workspace=_resolve_cli_workspace(args),
                    agent_name=args.agent,
                    conversation_id=None,
                )
            )
        else:
            if args.agent is not None:
                raise ValueError("--agent is only valid without a subcommand")
            if args.command == "chat":
                _require_chat_terminal()
            result = asyncio.run(_execute(args))
    except KeyboardInterrupt:
        print("Chat interrupted.", file=sys.stderr)
        return 130
    except ArtifactError as error:
        print(
            json.dumps(
                {
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details.to_dict(),
                    }
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except StateCompatibilityError as error:
        if args.command is None:
            print("Daita could not open this local agent state.", file=sys.stderr)
            print(file=sys.stderr)
            print(str(error), file=sys.stderr)
            print(file=sys.stderr)
            print(f"State: {error.path}", file=sys.stderr)
            print("Local data changed: no", file=sys.stderr)
        else:
            print(
                json.dumps({"error": error.to_mapping()}, sort_keys=True),
                file=sys.stderr,
            )
        return 1
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    if args.command is None or args.command == "chat":
        assert isinstance(result, int)
        return result
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
