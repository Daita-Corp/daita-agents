"""Small CLI that calls the same public Agent API as Python users."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from . import (
    Agent,
    AgentConfig,
    AgentEvent,
    ApprovalDecision,
    ApprovalRequest,
    ArtifactError,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    LearningCandidateView,
    LocalDirectorySource,
    LoopExit,
    MCPAuthentication,
    MCPBindingStatus,
    MCPServerInspection,
    MCPToolSelection,
    PostgreSQLSource,
    Skill,
    SkillSummary,
    SQLiteSource,
    __version__,
    create_llm_provider,
)
from .artifacts.models import (
    artifact_delivery_receipt_to_mapping,
    artifact_destination_to_mapping,
    artifact_ref_to_mapping,
)
from .cli_text import (
    _edit_learning_candidate,
    _render_model_answer,
    _write_learning_candidate_list,
    _write_learning_candidate_view,
    _write_learning_review_result,
    _write_memory_surface,
    _write_semantic_view,
)
from .errors import StateCompatibilityError
from .learning_candidates import (
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    learning_candidate_content_to_mapping,
)
from .llm import (
    CostEstimate,
    ModelProfile,
    ModelProvider,
    ModelSensitivity,
    aggregate_cost_estimates,
)
from .llm.profiles import reviewed_model_profile
from .security import SecretReference
from .skills import validate_skill_name
from .terminal import run_terminal_application
from .tui.models import (
    validate_candidate_review_cost_limit as _validate_candidate_review_cost_limit,
)

_SKILL_DESCRIPTION_PLACEHOLDER = "Describe when the agent should use this skill."
_SKILL_INSTRUCTIONS_PLACEHOLDER = "Write the reusable procedure here."
_CANDIDATE_REVIEW_COST_LIMIT_ENV = "DAITA_CANDIDATE_REVIEW_MAX_COST_USD"
_CANDIDATE_REVIEWER_MAX_OUTPUT_TOKENS = LEARNING_REVIEW_MAX_TOTAL_TOKENS // 4
_BUILTIN_CHAT_COMMANDS = frozenset(
    {
        "/catalog",
        "/conversation",
        "/exit",
        "/help",
        "/learn",
        "/memory",
        "/mcp",
        "/model",
        "/new",
        "/resume",
        "/settings",
        "/source",
        "/sources",
        "/skills",
        "/status",
        "/user",
    }
)


class _SourceSummary(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def active(self) -> bool: ...


def _candidate_reviewer_profile(profile: ModelProfile) -> ModelProfile:
    """Project one explicit CLI model into the fixed review token budget."""

    if not isinstance(profile, ModelProfile):
        raise TypeError("candidate reviewer profile must be ModelProfile")
    return replace(
        profile,
        max_output_tokens=min(
            profile.max_output_tokens,
            _CANDIDATE_REVIEWER_MAX_OUTPUT_TOKENS,
        ),
    )


def _learning_candidate_mapping(
    view: LearningCandidateView,
    *,
    include_content: bool = False,
) -> dict[str, object]:
    candidate = view.candidate
    value: dict[str, object] = {
        "id": candidate.id,
        "target": candidate.target.value,
        "status": view.status.value,
        "source_ids": candidate.source_ids,
        "supporting_run_ids": candidate.supporting_run_ids,
        "obsolete_reasons": view.obsolete_reasons,
        "rejection_reason": (
            None
            if candidate.rejection_reason is None
            else candidate.rejection_reason.value
        ),
    }
    if include_content:
        value["content"] = learning_candidate_content_to_mapping(
            candidate.content
        ).to_dict()
    return value


def _mcp_authentication(bearer_env: str | None) -> MCPAuthentication:
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
    parser.add_argument("--agent", help="agent to open in terminal mode")
    commands = parser.add_subparsers(dest="command")

    create = commands.add_parser("create", help="create an agent")
    create.add_argument("name")

    attach = commands.add_parser("attach", help="attach a read-only source")
    attach.add_argument("name")
    attach.add_argument("kind", choices=("sqlite", "files", "postgresql"))
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

    mcp = commands.add_parser("mcp", help="manage admitted remote MCP read tools")
    mcp_commands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_inspect = mcp_commands.add_parser(
        "inspect",
        help="inspect one endpoint without admitting tools",
    )
    mcp_inspect.add_argument("name")
    mcp_inspect.add_argument("endpoint")
    mcp_inspect.add_argument("--bearer-env")
    mcp_attach = mcp_commands.add_parser(
        "attach",
        help="admit exact read-only tools for activation on reopen",
    )
    mcp_attach.add_argument("name")
    mcp_attach.add_argument("endpoint")
    mcp_attach.add_argument("--bearer-env")
    mcp_attach.add_argument("--binding-id")
    mcp_attach.add_argument(
        "--maximum-outbound-sensitivity",
        choices=tuple(item.value for item in ModelSensitivity),
        default=ModelSensitivity.INTERNAL.value,
    )
    mcp_attach.add_argument(
        "--tool",
        action="append",
        nargs=4,
        required=True,
        metavar=("REMOTE", "ALIAS", "DESCRIPTION", "RESULT_SENSITIVITY"),
        help="repeat for each exact read tool; quote descriptions containing spaces",
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
        "postgresql-update-readiness",
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
        help="confirm transcript and candidate-record deletion",
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
    run.add_argument("--events-jsonl", action="store_true")

    chat = commands.add_parser(
        "chat",
        help="open the same interactive Textual app as `daita`",
    )
    chat.add_argument("name")
    chat.add_argument("--model", help="ignored; model setup happens inside the app")
    chat.add_argument("--conversation")

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
    memory_review = memory_commands.add_parser("review")
    memory_review.add_argument("name")
    memory_review.add_argument("--model", required=True, help="provider:model")
    memory_review.add_argument(
        "--cost-limit",
        type=Decimal,
        required=True,
        help="maximum estimated USD cost for the one reviewer request",
    )
    memory_candidates = memory_commands.add_parser("list-candidates")
    memory_candidates.add_argument("name")
    memory_candidates.add_argument(
        "--status",
        choices=tuple(item.value for item in LearningCandidateStatus),
    )
    memory_show_candidate = memory_commands.add_parser("show-candidate")
    memory_show_candidate.add_argument("name")
    memory_show_candidate.add_argument("candidate_id")
    memory_edit_candidate = memory_commands.add_parser("edit-candidate")
    memory_edit_candidate.add_argument("name")
    memory_edit_candidate.add_argument("candidate_id")
    memory_accept_candidate = memory_commands.add_parser("accept-candidate")
    memory_accept_candidate.add_argument("name")
    memory_accept_candidate.add_argument("candidate_id")
    memory_accept_candidate.add_argument(
        "--model", required=True, help="provider:model"
    )
    memory_reject_candidate = memory_commands.add_parser("reject-candidate")
    memory_reject_candidate.add_argument("name")
    memory_reject_candidate.add_argument("candidate_id")
    memory_reject_candidate.add_argument(
        "--reason",
        choices=tuple(item.value for item in LearningCandidateRejectionReason),
        default=LearningCandidateRejectionReason.USER_DECLINED.value,
    )
    memory_clear_rejected = memory_commands.add_parser("clear-rejected")
    memory_clear_rejected.add_argument("name")

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
    skill_delete = skill_commands.add_parser("delete")
    skill_delete.add_argument("name")
    skill_delete.add_argument("skill_name")
    return parser


def _write_event_jsonl(event: AgentEvent) -> None:
    print(
        json.dumps(
            {
                "kind": event.kind.value,
                "occurred_at": event.occurred_at.isoformat(),
                "run_id": event.run_id,
                "conversation_id": event.conversation_id,
                "data": event.data.to_dict(),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def _model_configuration(
    model_id: str,
    *,
    base_url: str | None = None,
    context_window: int | None = None,
    max_output: int | None = None,
) -> tuple[ModelProvider, ModelProfile]:
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


def _reviewer_model_configuration(
    model_id: str,
) -> tuple[ModelProvider, ModelProfile]:
    """Construct one direct provider with the fixed candidate-review output bound."""

    _, profile = _model_configuration(model_id)
    reviewer_profile = _candidate_reviewer_profile(profile)
    return (
        create_llm_provider(
            model_id,
            max_output_tokens=reviewer_profile.max_output_tokens,
        ),
        reviewer_profile,
    )


def _candidate_review_cost_limit_from_environment() -> Decimal | None:
    raw = os.environ.get(_CANDIDATE_REVIEW_COST_LIMIT_ENV)
    if raw is None:
        return None
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError):
        raise ValueError(
            f"{_CANDIDATE_REVIEW_COST_LIMIT_ENV} must be a finite "
            "non-negative decimal"
        ) from None
    _validate_candidate_review_cost_limit(value)
    return value


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


def _source_lines(sources: Sequence[_SourceSummary]) -> tuple[str, ...]:
    if not sources:
        return ("  (none)",)
    return tuple(
        f"  {source.display_name} ({source.adapter_id}, "
        f"{'active' if source.active else 'inactive'}) [{source.id}]"
        for source in sources
    )


def _resume_command(
    state_root: Path,
    agent_name: str,
    model_id: str,
    conversation_id: str,
) -> str:
    return shlex.join(
        (
            "daita",
            "--root",
            str(state_root),
            "chat",
            agent_name,
            "--model",
            model_id,
            "--conversation",
            conversation_id,
        )
    )


def _write_sources(sources: Sequence[_SourceSummary]) -> None:
    print("Sources:")
    for line in _source_lines(sources):
        print(line)


def _write_resume(
    state_root: Path,
    agent_name: str,
    model_id: str,
    conversation_id: str,
) -> None:
    print(f"Conversation: {conversation_id}")
    print("Resume with:")
    print(_resume_command(state_root, agent_name, model_id, conversation_id))


@dataclass(slots=True)
class _ChatTotals:
    turns: int = 0
    steps: int = 0
    tokens: int = 0
    cost_estimate: CostEstimate = field(
        default_factory=lambda: CostEstimate.unavailable("no_model_attempts")
    )

    def add(self, result: LoopExit) -> None:
        prior_turns = self.turns
        self.turns += 1
        self.steps += result.steps
        self.tokens += result.usage.total_tokens
        self.cost_estimate = (
            result.usage.cost_estimate
            if prior_turns == 0
            else aggregate_cost_estimates(
                (self.cost_estimate, result.usage.cost_estimate)
            )
        )


def _write_startup(
    agent: Agent,
    model_id: str,
    sources: Sequence[_SourceSummary],
    conversation_id: str | None,
) -> None:
    profile = agent.model_profile
    print("Daita chat")
    print(f"Agent: {agent.name}")
    print(f"Model: {profile.id if profile is not None else model_id}")
    _write_sources(sources)
    print(f"Conversation: {conversation_id or 'new'}")
    print()
    print("Type /help for commands. Ctrl-D exits; Ctrl-C interrupts.")
    print()


def _write_help() -> None:
    print("Commands:")
    print("  /help")
    print("  /status")
    print("  /conversation")
    print("  /new")
    print("  /resume <conversation-id>")
    print("  /sources")
    print("  /learn <material>")
    print("  /memory")
    print("  /memory edit")
    print("  /memory list")
    print("  /review")
    print("  /memory show <candidate-or-annotation-id>")
    print("  /memory edit <candidate-id>")
    print("  /memory accept <candidate-id>")
    print("  /memory reject <candidate-id> [reason]")
    print("  /memory clear-rejected")
    print("  /memory delete <annotation-id>")
    print("  /user")
    print("  /user edit")
    print("  /skills")
    print("  /skills show <name>")
    print("  /skills create [name]")
    print("  /skills edit <name>")
    print("  /skills delete <name>")
    print("  /skills use <name> [request]")
    print("  /<skill-name> [request]")
    print("  /exit")


def _write_local_diagnostic(message: str) -> None:
    bounded = message if len(message) <= 512 else f"{message[:509]}..."
    print(bounded, file=sys.stderr)


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


def _write_memory(target: str, content: str) -> None:
    print(f"{target.capitalize()}:")
    print(content if content else "(empty)")


def _write_skills(skills: Sequence[SkillSummary]) -> None:
    print("Skills:")
    if not skills:
        print("  (none)")
        return
    for skill in skills:
        print(f"  /{skill.name}: {skill.description}")


def _write_skill(skill: Skill) -> None:
    print(f"Skill: {skill.name}")
    print(f"Description: {skill.description}")
    print("Instructions:")
    print(skill.instructions)


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


async def _create_skill(agent: Agent, name: str) -> bool:
    validate_skill_name(name)
    if await agent.read_skill(name) is not None:
        raise ValueError(f"skill already exists: {name}")
    seed = _render_skill_editor_document(
        name,
        _SKILL_DESCRIPTION_PLACEHOLDER,
        _SKILL_INSTRUCTIONS_PLACEHOLDER,
    )
    draft = seed
    while True:
        edited = _edit_document(draft, agent_home=agent.home)
        if edited == seed:
            print("Skill creation cancelled; template was unchanged.")
            return False
        try:
            description, instructions = _parse_skill_editor_document(name, edited)
            if description == _SKILL_DESCRIPTION_PLACEHOLDER:
                raise ValueError("replace the description placeholder")
            if instructions == _SKILL_INSTRUCTIONS_PLACEHOLDER:
                raise ValueError("replace the instructions placeholder")
            Skill(name, description, instructions)
        except ValueError as error:
            _write_local_diagnostic(f"Skill document is invalid: {error}")
            draft = edited
            try:
                answer = input("Reopen editor? [Y/n]")
            except EOFError:
                print()
                answer = "n"
            if answer.strip().casefold() in {"n", "no"}:
                print("Skill creation cancelled.")
                return False
            continue
        if await agent.read_skill(name) is not None:
            raise ValueError(f"skill already exists: {name}")
        changed = await agent.save_skill(name, description, instructions)
        if not changed:
            raise RuntimeError("new skill was not persisted")
        print(f"Skill {name!r} created. Invoke it with /{name}.")
        return True


async def _create_skill_wizard(agent: Agent) -> bool:
    print("Create skill")
    print("Enter /cancel at any prompt to stop.")
    while True:
        try:
            name = input("Name:").strip()
        except EOFError:
            print()
            print("Skill creation cancelled.")
            return False
        if name.casefold() == "/cancel":
            print("Skill creation cancelled.")
            return False
        try:
            validate_skill_name(name)
            if await agent.read_skill(name) is not None:
                raise ValueError(f"skill already exists: {name}")
        except ValueError as error:
            _write_local_diagnostic(f"Invalid name: {error}")
            continue
        break

    while True:
        try:
            description = input("Description:").strip()
        except EOFError:
            print()
            print("Skill creation cancelled.")
            return False
        if description.casefold() == "/cancel":
            print("Skill creation cancelled.")
            return False
        try:
            Skill(name, description, _SKILL_INSTRUCTIONS_PLACEHOLDER)
        except ValueError as error:
            _write_local_diagnostic(f"Invalid description: {error}")
            continue
        break

    while True:
        print("Instructions (finish with a single . on its own line):")
        instruction_lines: list[str] = []
        while True:
            try:
                line = input(">")
            except EOFError:
                print()
                print("Skill creation cancelled.")
                return False
            if line.casefold() == "/cancel":
                print("Skill creation cancelled.")
                return False
            if line == ".":
                break
            instruction_lines.append(line)
        instructions = "\n".join(instruction_lines).strip()
        try:
            Skill(name, description, instructions)
        except ValueError as error:
            _write_local_diagnostic(f"Invalid instructions: {error}")
            print("Re-enter the instructions body.")
            continue
        break

    if await agent.read_skill(name) is not None:
        raise ValueError(f"skill already exists: {name}")
    changed = await agent.save_skill(name, description, instructions)
    if not changed:
        raise RuntimeError("new skill was not persisted")
    print(f"Skill {name!r} created. Invoke it with /{name}.")
    return True


async def _confirm_skill_deletion(name: str) -> bool:
    try:
        answer = input(f"Delete skill {name!r}? [y/N]")
    except EOFError:
        print()
        return False
    return answer.strip().lower() == "y"


async def _handle_knowledge_chat_command(parts: list[str], agent: Agent) -> bool:
    name = parts[0] if parts else ""
    if name == "/review":
        if len(parts) != 1:
            _write_local_diagnostic("Usage: /review")
            return True
        _write_learning_review_result(
            await agent.review_learning_candidates(),
            sys.stdout,
        )
        return True
    if name in {"/memory", "/user"}:
        target = "memory" if name == "/memory" else "user"
        if len(parts) == 1:
            content = await _read_memory_target(agent, target)
            if target == "memory":
                await _write_memory_surface(agent, content, sys.stdout)
            else:
                _write_memory(target, content)
        elif target == "memory" and parts[1:] == ["list"]:
            _write_learning_candidate_list(
                await agent.list_learning_candidates(),
                sys.stdout,
            )
        elif target == "memory" and len(parts) == 3 and parts[1] == "show":
            candidate = await agent.read_learning_candidate(parts[2])
            if candidate is not None:
                _write_learning_candidate_view(candidate, sys.stdout)
            else:
                view = await agent.read_semantic_annotation(parts[2])
                if view is None:
                    raise ValueError(f"memory record not found: {parts[2]}")
                _write_semantic_view(view, sys.stdout)
        elif target == "memory" and len(parts) == 3 and parts[1] == "edit":
            await _edit_learning_candidate(agent, parts[2])
            print(f"Learning candidate {parts[2]!r} updated.")
        elif target == "memory" and len(parts) == 3 and parts[1] == "accept":
            result = await agent.accept_learning_candidate(parts[2])
            print(
                _render_model_answer(
                    result.final_text,
                    fallback=f"{result.kind.value}: {result.reason}",
                )
            )
        elif target == "memory" and len(parts) in {3, 4} and parts[1] == "reject":
            reason = (
                LearningCandidateRejectionReason.USER_DECLINED
                if len(parts) == 3
                else LearningCandidateRejectionReason(parts[3])
            )
            rejected = await agent.reject_learning_candidate(parts[2], reason)
            print(f"Learning candidate {rejected.candidate.id!r} rejected.")
        elif target == "memory" and parts[1:] == ["clear-rejected"]:
            cleared = await agent.clear_rejected_learning_candidates()
            print(f"Cleared {cleared} rejected candidate(s).")
        elif target == "memory" and len(parts) == 3 and parts[1] == "delete":
            view = await agent.read_semantic_annotation(parts[2])
            if view is None:
                raise ValueError(f"semantic annotation not found: {parts[2]}")
            try:
                answer = input(f"Delete semantic annotation {parts[2]!r}? [y/N]")
            except EOFError:
                print()
                answer = ""
            if answer.strip().lower() != "y":
                print("Deletion cancelled.")
                return True
            await agent.delete_semantic_annotation(
                parts[2],
                expected_sha256=view.sha256,
            )
            print(f"Semantic annotation {parts[2]!r} deleted.")
        elif len(parts) == 2 and parts[1] == "edit":
            await _edit_memory_target(agent, target)
            print(f"{target.capitalize()} updated.")
        else:
            usage = (
                "/memory [list|show <id>|edit [id]|accept <id>|"
                "reject <id> [reason]|clear-rejected|delete <semantic-id>]"
                if target == "memory"
                else "/user [edit]"
            )
            _write_local_diagnostic(f"Usage: {usage}")
        return True
    if name != "/skills":
        return False
    if len(parts) == 1:
        _write_skills(await agent.list_skills())
        return True
    if len(parts) == 3 and parts[1] == "show":
        skill = await agent.read_skill(parts[2])
        if skill is None:
            raise ValueError(f"skill not found: {parts[2]}")
        _write_skill(skill)
        return True
    if len(parts) == 2 and parts[1] == "create":
        await _create_skill_wizard(agent)
        return True
    if len(parts) == 3 and parts[1] == "create":
        await _create_skill(agent, parts[2])
        return True
    if len(parts) == 3 and parts[1] == "edit":
        changed = await _edit_skill(agent, parts[2])
        print(f"Skill {parts[2]!r} {'updated' if changed else 'unchanged'}.")
        return True
    if len(parts) == 3 and parts[1] == "delete":
        if not await _confirm_skill_deletion(parts[2]):
            print("Deletion cancelled.")
            return True
        deleted = await agent.delete_skill(parts[2])
        print(f"Skill {parts[2]!r} {'deleted' if deleted else 'not found'}.")
        return True
    _write_local_diagnostic(
        "Usage: /skills [show <name>|create [name]|edit <name>|"
        "delete <name>|use <name> [request]]"
    )
    return True


async def _skill_invocation_message(agent: Agent, message: str) -> str | None:
    parts = message.split()
    if not parts:
        return None
    command = parts[0]
    if command == "/skills" and len(parts) >= 2 and parts[1] == "use":
        if len(parts) < 3:
            raise ValueError("usage: /skills use <name> [request]")
        skill_name = parts[2]
        try:
            skill = await agent.read_skill(skill_name)
        except ValueError as error:
            raise ValueError(f"invalid skill name {skill_name!r}: {error}") from error
        if skill is None:
            raise ValueError(f"skill not found: {skill_name}")
        return message
    if command in _BUILTIN_CHAT_COMMANDS or not command.startswith("/"):
        return None
    skill_name = command[1:]
    if not skill_name:
        return None
    try:
        skill = await agent.read_skill(skill_name)
    except ValueError:
        return None
    return message if skill is not None else None


async def _prompt_for_exact_approval(
    request: ApprovalRequest,
) -> ApprovalDecision:
    print("Approval required")
    print()
    print(f"Tool:       {request.tool_name}")
    print(f"Capability: {request.capability_id}")
    print("Arguments:")
    print(
        json.dumps(
            request.arguments.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    while True:
        try:
            answer = input("Approve this exact change once? [y/n]")
        except EOFError:
            print()
            return ApprovalDecision.DENY
        normalized = answer.strip().casefold()
        if normalized == "y":
            return ApprovalDecision.APPROVE
        if normalized == "n":
            return ApprovalDecision.DENY
        print("Enter y to approve or n to deny.")


async def _execute(args: argparse.Namespace) -> object:
    if args.command == "create":
        agent = await Agent.create(args.name, root=args.root)
        try:
            return {"agent_id": agent.id, "name": agent.name, "home": str(agent.home)}
        finally:
            await agent.close()
    if args.command == "delete":
        if not args.yes:
            raise ValueError("delete requires --yes")
        await Agent.delete(args.name, root=args.root)
        return {"name": args.name, "deleted": True}
    if args.command == "detach":
        if not args.yes:
            raise ValueError("detach requires --yes")
        agent = await Agent.open(
            args.name,
            root=args.root,
            config=AgentConfig(),
        )
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
        agent = await Agent.open(
            args.name,
            root=args.root,
            config=AgentConfig(),
        )
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
        if args.model is None:
            agent = await Agent.open(
                args.name,
                root=args.root,
                observer=_write_event_jsonl if args.events_jsonl else None,
            )
        else:
            provider, profile = _model_configuration(
                args.model,
                base_url=args.base_url,
                context_window=args.context_window,
                max_output=args.max_output,
            )
            agent = await Agent.open(
                args.name,
                root=args.root,
                model=provider,
                model_profile=profile,
                observer=_write_event_jsonl if args.events_jsonl else None,
            )
        try:
            result = await agent.run(
                args.message,
                conversation_id=args.conversation_id,
            )
            return {
                "run_id": result.run_id,
                "conversation_id": result.conversation_id,
                "status": result.kind.value,
                "reason": result.reason,
                "text": result.final_text,
                "steps": result.steps,
                "artifacts": tuple(
                    artifact_ref_to_mapping(item) for item in result.artifacts
                ),
                "artifact_deliveries": tuple(
                    artifact_delivery_receipt_to_mapping(item)
                    for item in result.artifact_deliveries
                ),
            }
        finally:
            await agent.close()
    if args.command == "chat":
        return await run_terminal_application(
            root=args.root,
            agent_name=args.name,
            reviewer_max_estimated_cost_usd=(
                _candidate_review_cost_limit_from_environment()
            ),
        )
    if args.command == "memory" and args.memory_command == "review":
        _validate_candidate_review_cost_limit(args.cost_limit)
        provider, profile = _reviewer_model_configuration(args.model)
        agent = await Agent.open(
            args.name,
            root=args.root,
            reviewer_model=provider,
            reviewer_profile=profile,
            reviewer_max_estimated_cost_usd=args.cost_limit,
        )
        try:
            review_result = await agent.review_learning_candidates()
            return {
                "status": review_result.status.value,
                "reviewed_run_ids": review_result.reviewed_run_ids,
                "candidate_ids": tuple(
                    item.candidate.id for item in review_result.candidates
                ),
                "model_calls": review_result.model_calls,
                "skipped_run_count": review_result.skipped_run_count,
                "duplicate_proposals_suppressed": (
                    review_result.duplicate_proposals_suppressed
                ),
                "total_tokens": review_result.usage.total_tokens,
            }
        finally:
            await agent.close()
    if args.command == "memory" and args.memory_command == "accept-candidate":
        provider, profile = _model_configuration(args.model)
        agent = await Agent.open(
            args.name,
            root=args.root,
            model=provider,
            model_profile=profile,
            approval_handler=_prompt_for_exact_approval,
        )
        try:
            result = await agent.accept_learning_candidate(args.candidate_id)
            return {
                "candidate_id": args.candidate_id,
                "run_id": result.run_id,
                "status": result.kind.value,
                "reason": result.reason,
                "text": result.final_text,
            }
        finally:
            await agent.close()
    agent = await Agent.open(args.name, root=args.root)
    try:
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
            if args.mcp_command == "inspect":
                inspection = await agent.inspect_mcp_server(
                    endpoint=args.endpoint,
                    authentication=_mcp_authentication(args.bearer_env),
                )
                return _mcp_inspection_mapping(inspection)
            if args.mcp_command == "attach":
                selections = tuple(
                    MCPToolSelection(
                        remote_name=remote_name,
                        local_alias=local_alias,
                        description=description,
                        result_sensitivity=ModelSensitivity(result_sensitivity),
                    )
                    for remote_name, local_alias, description, result_sensitivity in (
                        args.tool
                    )
                )
                status = await agent.attach_mcp_server(
                    endpoint=args.endpoint,
                    authentication=_mcp_authentication(args.bearer_env),
                    maximum_outbound_sensitivity=ModelSensitivity(
                        args.maximum_outbound_sensitivity
                    ),
                    selections=selections,
                    binding_id=args.binding_id,
                )
                return _mcp_status_mapping(status)
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
        if args.command == "postgresql-update-readiness":
            readiness_result = await agent.postgresql_update_readiness(
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
                    "candidates": [
                        _learning_candidate_mapping(view)
                        for view in await agent.list_learning_candidates()
                    ],
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
            if args.memory_command == "list-candidates":
                candidate_status = (
                    None
                    if args.status is None
                    else LearningCandidateStatus(args.status)
                )
                return [
                    _learning_candidate_mapping(view)
                    for view in await agent.list_learning_candidates(
                        status=candidate_status
                    )
                ]
            if args.memory_command == "show-candidate":
                view = await agent.read_learning_candidate(args.candidate_id)
                if view is None:
                    raise ValueError(
                        f"learning candidate not found: {args.candidate_id}"
                    )
                return _learning_candidate_mapping(view, include_content=True)
            if args.memory_command == "edit-candidate":
                await _edit_learning_candidate(agent, args.candidate_id)
                view = await agent.read_learning_candidate(args.candidate_id)
                assert view is not None
                return _learning_candidate_mapping(view, include_content=True)
            if args.memory_command == "reject-candidate":
                view = await agent.reject_learning_candidate(
                    args.candidate_id,
                    LearningCandidateRejectionReason(args.reason),
                )
                return _learning_candidate_mapping(view)
            if args.memory_command == "clear-rejected":
                return {"cleared": await agent.clear_rejected_learning_candidates()}
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
        await agent.close()


def _source_from_attach_args(
    args: argparse.Namespace,
) -> SQLiteSource | LocalDirectorySource | PostgreSQLSource:
    if args.kind in {"sqlite", "files"}:
        if args.path is None:
            raise ValueError(f"attach {args.kind} requires a path")
        return (
            SQLiteSource(args.path)
            if args.kind == "sqlite"
            else LocalDirectorySource(args.path)
        )
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


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        result: object
        if args.command is None:
            _require_terminal_application()
            result = asyncio.run(
                run_terminal_application(
                    root=args.root,
                    agent_name=args.agent,
                    reviewer_max_estimated_cost_usd=(
                        _candidate_review_cost_limit_from_environment()
                    ),
                    input_stream=sys.stdin,
                    output_stream=sys.stdout,
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
