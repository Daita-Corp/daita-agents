"""Small CLI that calls the same public Agent API as Python users."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Protocol, Sequence

from . import (
    Agent,
    AgentEvent,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    LocalDirectorySource,
    LoopExit,
    PostgreSQLSource,
    SQLiteSource,
    Skill,
    SkillSummary,
    create_llm_provider,
)
from .llm import ModelProfile, ModelProvider
from .llm.profiles import reviewed_model_profile
from .security import SecretReference
from .terminal import run_terminal_application


class _SourceSummary(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def active(self) -> bool: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daita")
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

    run = commands.add_parser("run", help="run one agent request")
    run.add_argument("name")
    run.add_argument("message")
    run.add_argument("--model", required=True, help="provider:model")
    run.add_argument("--base-url")
    run.add_argument("--context-window", type=int)
    run.add_argument("--max-output", type=int)
    run.add_argument("--conversation-id")
    run.add_argument("--events-jsonl", action="store_true")

    chat = commands.add_parser("chat", help="chat interactively with an agent")
    chat.add_argument("name")
    chat.add_argument("--model", required=True, help="provider:model")
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
    )


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
    estimated_cost_usd: Decimal = Decimal("0")

    def add(self, result: LoopExit) -> None:
        self.turns += 1
        self.steps += result.steps
        self.tokens += result.usage.total_tokens
        self.estimated_cost_usd += result.usage.estimated_cost_usd


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
    print("  /memory")
    print("  /memory edit")
    print("  /user")
    print("  /user edit")
    print("  /skills")
    print("  /skills show <name>")
    print("  /skills edit <name>")
    print("  /skills delete <name>")
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
        print(f"  {skill.name}: {skill.description}")


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


async def _confirm_skill_deletion(name: str) -> bool:
    try:
        answer = input(f"Delete skill {name!r}? [y/N]")
    except EOFError:
        print()
        return False
    return answer.strip().lower() == "y"


async def _handle_knowledge_chat_command(parts: list[str], agent: Agent) -> bool:
    name = parts[0] if parts else ""
    if name in {"/memory", "/user"}:
        target = "memory" if name == "/memory" else "user"
        if len(parts) == 1:
            _write_memory(target, await _read_memory_target(agent, target))
        elif len(parts) == 2 and parts[1] == "edit":
            await _edit_memory_target(agent, target)
            print(f"{target.capitalize()} updated.")
        else:
            _write_local_diagnostic(f"Usage: {name} [edit]")
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
    _write_local_diagnostic("Usage: /skills [show <name>|edit <name>|delete <name>]")
    return True


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
    try:
        answer = input("Approve this exact change once? [y/N]")
    except EOFError:
        print()
        return ApprovalDecision.DENY
    if answer.strip().lower() == "y":
        return ApprovalDecision.APPROVE
    return ApprovalDecision.DENY


async def _handle_chat_command(
    command: str,
    *,
    agent: Agent,
    model_id: str,
    state_root: Path,
    conversation_id: str | None,
    totals: _ChatTotals,
) -> tuple[bool, str | None]:
    parts = command.split()
    name = parts[0] if parts else ""
    if name == "/exit" and len(parts) == 1:
        return True, conversation_id
    if name == "/help" and len(parts) == 1:
        _write_help()
        return False, conversation_id
    try:
        if await _handle_knowledge_chat_command(parts, agent):
            return False, conversation_id
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        _write_local_diagnostic(f"Knowledge command failed: {error}")
        return False, conversation_id
    if name == "/sources" and len(parts) == 1:
        _write_sources(await agent.list_sources())
        return False, conversation_id
    if name == "/status" and len(parts) == 1:
        print(f"Agent: {agent.name}")
        profile = agent.model_profile
        print(f"Model: {profile.id if profile is not None else model_id}")
        _write_sources(await agent.list_sources())
        print(f"Conversation: {conversation_id or 'new'}")
        print(
            f"This process: {totals.turns} turns, {totals.steps} steps, "
            f"{totals.tokens} tokens, ${totals.estimated_cost_usd}"
        )
        return False, conversation_id
    if name == "/conversation" and len(parts) == 1:
        if conversation_id is None:
            print("Conversation: new")
        else:
            _write_resume(
                state_root,
                agent.name,
                model_id,
                conversation_id,
            )
        return False, conversation_id
    if name == "/new" and len(parts) == 1:
        print("Conversation: new")
        return False, None
    if name == "/resume" and len(parts) == 2:
        candidate = parts[1]
        try:
            await agent.conversation_runs(candidate)
        except (TypeError, ValueError) as error:
            _write_local_diagnostic(f"Cannot resume conversation: {error}")
            return False, conversation_id
        print(f"Conversation: {candidate}")
        return False, candidate
    if name == "/resume":
        _write_local_diagnostic("Usage: /resume <conversation-id>")
    elif name in {
        "/exit",
        "/help",
        "/sources",
        "/status",
        "/conversation",
        "/new",
    }:
        _write_local_diagnostic(f"Usage: {name}")
    else:
        _write_local_diagnostic("Unknown command. Type /help for commands.")
    return False, conversation_id


async def _chat(args: argparse.Namespace) -> int:
    provider, profile = _model_configuration(args.model)
    approval_handler: ApprovalHandler = _prompt_for_exact_approval
    agent = await Agent.open(
        args.name,
        root=args.root,
        model=provider,
        model_profile=profile,
        approval_handler=approval_handler,
    )
    conversation_id: str | None = args.conversation
    last_completed_conversation_id: str | None = None
    totals = _ChatTotals()
    interrupted = False
    try:
        state_root = agent.home.parent.parent
        _write_startup(
            agent,
            args.model,
            await agent.list_sources(),
            conversation_id,
        )
        while True:
            try:
                message = input("You › ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                interrupted = True
                break
            message = message.strip()
            if not message:
                continue
            if message.startswith("/"):
                should_exit, conversation_id = await _handle_chat_command(
                    message,
                    agent=agent,
                    model_id=args.model,
                    state_root=state_root,
                    conversation_id=conversation_id,
                    totals=totals,
                )
                if should_exit:
                    break
                continue

            creates_conversation = conversation_id is None
            result = await agent.run(message, conversation_id=conversation_id)
            conversation_id = result.conversation_id
            last_completed_conversation_id = result.conversation_id
            totals.add(result)
            if creates_conversation:
                print(f"Conversation: {conversation_id}")
            print()
            print("Daita")
            if result.final_text is not None:
                print(result.final_text)
            else:
                print(f"{result.kind.value}: {result.reason}")
            print()
            print(
                f"{result.steps} steps · {result.usage.total_tokens} tokens · "
                f"${result.usage.estimated_cost_usd}"
            )
            print()

        resume_id = conversation_id or last_completed_conversation_id
        if resume_id is None:
            print("No conversation was created.")
        else:
            print(f"Conversation saved as {resume_id}.")
            print()
            print("Resume with:")
            print(_resume_command(state_root, agent.name, args.model, resume_id))
        return 130 if interrupted else 0
    finally:
        await agent.close()


async def _execute(args: argparse.Namespace) -> object:
    if args.command == "create":
        agent = await Agent.create(args.name, root=args.root)
        try:
            return {"agent_id": agent.id, "name": agent.name, "home": str(agent.home)}
        finally:
            await agent.close()
    if args.command == "run":
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
            }
        finally:
            await agent.close()
    if args.command == "chat":
        return await _chat(args)
    agent = await Agent.open(args.name, root=args.root)
    try:
        if args.command == "attach":
            source = _source_from_attach_args(args)
            registration = await agent.attach(source)
            return {
                "source_id": registration.id,
                "adapter": registration.adapter_id,
                "name": registration.display_name,
            }
        if args.command == "memory":
            if args.memory_command == "read":
                return {
                    "target": args.target,
                    "content": await _read_memory_target(agent, args.target),
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
