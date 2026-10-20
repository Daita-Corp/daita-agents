"""Parse slash commands, completions, source overrides, and editor documents."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from .models import (
    MAX_POSTGRESQL_CONNECTION_URL_BYTES,
    SSL_MODES,
    UserInputError,
    source_type_label,
)
from .sanitization import safe_display

SKILL_DESCRIPTION_PLACEHOLDER = "Describe when the agent should use this skill."
SKILL_INSTRUCTIONS_PLACEHOLDER = "Write the reusable procedure here."

# Advertised commands in menu order, grouped by what the user is doing.
# Each group's commands stay together so the popup can label the group once.
COMMAND_GROUPS = (
    (
        "Answer",
        (
            ("/save", "/save", "Save the latest answer as a document"),
            (
                "/schedule ",
                "/schedule [when]",
                "Run the latest question again on a schedule",
            ),
            ("/teach ", "/teach [note]", "Remember what worked in the latest answer"),
        ),
    ),
    (
        "Data",
        (
            (
                "/sources",
                "/sources",
                "Connect databases and folders; browse what is in them",
            ),
            ("/files ", "/files <question>", "Ask using only local files"),
            ("/workspace", "/workspace", "Show which local folder Daita can read"),
            ("/mcp", "/mcp", "Connect external tools (MCP servers)"),
        ),
    ),
    (
        "Work",
        (
            ("/work", "/work", "See scheduled, running, waiting and finished work"),
            ("/routines", "/routines", "View, pause and run your routines"),
            (
                "/routines create ",
                "/routines create <instruction>",
                "Describe a new routine in plain words",
            ),
            (
                "/routines update ",
                "/routines update <routine-id> <instruction>",
                "Change one routine's instruction",
            ),
            ("/jobs", "/jobs", "Multi-step background jobs"),
            ("/artifacts", "/artifacts", "Files Daita created: preview, save, delete"),
        ),
    ),
    (
        "Learn",
        (
            ("/learn ", "/learn <material>", "Teach a fact, preference or procedure"),
            (
                "/review",
                "/review [cost-usd]",
                "Suggest memories and skills from recent runs",
            ),
            (
                "/memory",
                "/memory",
                "What Daita remembers, and suggestions to review (duplicate, "
                "stale, conflicting, and superseded)",
            ),
            ("/user", "/user", "Your profile and preferences"),
            ("/skills", "/skills", "List reusable procedures"),
            ("/skills create", "/skills create", "Write a new skill"),
            ("/skills use ", "/skills use <name> [request]", "Run a skill by name"),
        ),
    ),
    (
        "Session",
        (
            ("/new", "/new", "Start a new conversation"),
            ("/resume ", "/resume <id>", "Return to an earlier conversation"),
            ("/conversation", "/conversation", "Show this conversation's ID"),
            (
                "/conversation clear",
                "/conversation clear",
                "Delete all conversation history",
            ),
            ("/help", "/help", "Keys and commands"),
            ("/exit", "/exit", "Quit Daita"),
        ),
    ),
    (
        "Agent",
        (
            ("/model", "/model", "Choose the AI model"),
            ("/settings", "/settings", "Agent and model settings"),
            ("/status", "/status", "Agent, sources and host status"),
            ("/agent delete", "/agent delete", "Permanently delete this agent"),
        ),
    ),
)
SLASH_COMMAND_COMPLETIONS = tuple(
    command for _group, commands in COMMAND_GROUPS for command in commands
)
_COMMAND_GROUP_BY_DISPLAY = {
    display: group
    for group, commands in COMMAND_GROUPS
    for _insertion, display, _description in commands
}
# Answer actions offered under each completed answer, in display order.
NEXT_STEPS = (
    ("/save", "Save as document"),
    ("/schedule", "Run on a schedule"),
    ("/teach", "Teach this"),
)


def completion_group(display: str) -> str:
    """Name the menu group of one completion row."""

    if display.startswith("@"):
        return "Source"
    return _COMMAND_GROUP_BY_DISPLAY.get(display, "Skill")


UNADVERTISED_SOURCE_COMMANDS = frozenset(
    {
        "/catalog",
        "/source",
        "/source add",
        "/source edit",
        "/source refresh <source-id>",
        "/source detach <source>",
        "/source permissions",
    }
)
BUILTIN_SLASH_COMMAND_ROOTS = frozenset(
    display.split(maxsplit=1)[0]
    for _insertion, display, _description in SLASH_COMMAND_COMPLETIONS
) | frozenset(command.split(maxsplit=1)[0] for command in UNADVERTISED_SOURCE_COMMANDS)
BUILTIN_SLASH_COMMANDS = BUILTIN_SLASH_COMMAND_ROOTS


def suggested_questions(resource_names: tuple[str, ...]) -> tuple[str, ...]:
    """Starter questions for an empty chat, named after real catalog tables."""

    names = tuple(
        safe_display(name, fallback="", maximum=60) for name in resource_names
    )
    names = tuple(name for name in names if name)
    if not names:
        return (
            "What files are in this folder?",
            "Summarize the most important file in this folder",
        )
    first = names[0]
    questions = [f"What is in {first}?", f"Show the 10 most recent rows of {first}"]
    if len(names) > 1:
        questions.append(f"How does {first} relate to {names[1]}?")
    return tuple(questions)


class LearningInvocation(str):
    """Process-local marker for the explicit /learn control-plane action."""


def slash_completion_maps(
    skill_completions: tuple[tuple[str, str], ...] = (),
) -> tuple[dict[str, str], dict[str, str]]:
    display = {
        insertion: shown for insertion, shown, _description in SLASH_COMMAND_COMPLETIONS
    }
    descriptions = {
        insertion: description
        for insertion, _shown, description in SLASH_COMMAND_COMPLETIONS
    }
    for name, description in skill_completions:
        command = f"/{name}"
        if command in BUILTIN_SLASH_COMMAND_ROOTS:
            continue
        insertion = f"{command} "
        display[insertion] = command
        descriptions[insertion] = description
    return display, descriptions


def matching_completions(
    prefix: str,
    *,
    skill_completions: tuple[tuple[str, str], ...] = (),
    source_completions: tuple[tuple[str, str, str], ...] = (),
) -> tuple[tuple[str, str, str], ...]:
    display, descriptions = slash_completion_maps(skill_completions)
    matches: list[tuple[str, str, str]] = []
    for insertion, shown in display.items():
        if insertion.startswith(prefix) or shown.startswith(prefix):
            matches.append((insertion, shown, descriptions[insertion]))
    for insertion, shown, description in source_completions:
        if insertion.startswith(prefix) or shown.startswith(prefix):
            matches.append((insertion, shown, description))
    return tuple(matches)


def learning_invocation_message(message: str) -> str | None:
    parts = message.split(maxsplit=1)
    if not parts or parts[0] != "/learn":
        return None
    if len(parts) == 1 or not parts[1].strip():
        raise ValueError("usage: /learn <material>")
    material = parts[1].strip()
    return LearningInvocation(
        "Treat the following as an explicit teaching request. Determine whether it "
        "belongs in stable user preferences, agent-wide business memory, a current "
        "resource/field-scoped semantic annotation, or a reusable procedural skill. "
        "Apply the normal foreground-learning safety rules and inspect current catalog "
        "scope and existing memory, semantics, or skills when needed. Call the smallest "
        "fitting learning tool immediately; its approval card is the only confirmation, "
        "so never ask the user for a typed approval phrase. Do not claim that a workflow "
        "is verified unless it was executed successfully or the user explicitly "
        "confirmed it.\n\nTeaching material:\n" + material
    )


def quoted_source_override(selector: str) -> str:
    escaped = selector.replace("\\", "\\\\").replace('"', '\\"')
    return f'@"{escaped}"'


def source_override_completions(
    sources: tuple[Any, ...],
    source_types: tuple[Any, ...],
) -> tuple[tuple[str, str, str], ...]:
    active_sources = tuple(source for source in sources if source.active)
    folded_name_counts: dict[str, int] = {}
    for source in active_sources:
        folded = source.display_name.casefold()
        folded_name_counts[folded] = folded_name_counts.get(folded, 0) + 1

    completions: list[tuple[str, str, str]] = []
    for source in active_sources:
        display_name = safe_display(
            source.display_name,
            fallback="source",
            maximum=512,
        )
        use_display_name = (
            display_name == source.display_name
            and len(source.display_name.encode("utf-8")) <= 1_024
            and folded_name_counts[source.display_name.casefold()] == 1
        )
        selector = source.display_name if use_display_name else source.id
        adapter_id = getattr(source, "adapter_id", "source")
        source_type = source_type_label(source_types, adapter_id)
        description = f"Ask one question using {source_type}"
        if not use_display_name:
            description += f" · {source.id[-8:]}"
        completions.append(
            (
                quoted_source_override(selector) + " ",
                f"@{display_name}",
                description,
            )
        )
    return tuple(completions)


def parse_source_override(message: str) -> tuple[str, str] | None:
    if not message.startswith("@"):
        return None
    if len(message) == 1 or message[1].isspace():
        raise UserInputError("Choose a source after @, then enter a question.")

    if not message.startswith('@"'):
        selector_end = 1
        while selector_end < len(message) and not message[selector_end].isspace():
            selector_end += 1
        selector = message[1:selector_end]
        question = message[selector_end:].strip()
        return selector, question

    selector_characters: list[str] = []
    position = 2
    while position < len(message):
        character = message[position]
        if character == '"':
            position += 1
            if position < len(message) and not message[position].isspace():
                raise UserInputError("Put a space after the quoted @ source name.")
            return "".join(selector_characters), message[position:].strip()
        if character == "\\" and position + 1 < len(message):
            escaped = message[position + 1]
            if escaped in {'"', "\\"}:
                selector_characters.append(escaped)
                position += 2
                continue
        selector_characters.append(character)
        position += 1
    raise UserInputError("Close the quoted @ source name before entering a question.")


def parse_postgresql_connection_url(
    value: str,
) -> tuple[str, int, str, str, str | None, str]:
    if not isinstance(value, str):
        raise ValueError("PostgreSQL connection URL must be text")
    try:
        encoded_length = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError("PostgreSQL connection URL is invalid") from None
    if (
        not value
        or encoded_length > MAX_POSTGRESQL_CONNECTION_URL_BYTES
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("PostgreSQL connection URL is invalid")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = 5432 if parsed.port is None else parsed.port
        raw_username = parsed.username
        raw_password = parsed.password
    except (TypeError, ValueError):
        raise ValueError("PostgreSQL connection URL is invalid") from None
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or parsed.fragment
        or host is None
        or not host
        or "," in host
        or not 1 <= port <= 65_535
        or raw_username is None
        or not raw_username
        or not parsed.path.startswith("/")
        or parsed.path == "/"
    ):
        raise ValueError("PostgreSQL connection URL is invalid")
    username = _decode_postgresql_url_component(raw_username)
    database = _decode_postgresql_url_component(parsed.path[1:])
    password = (
        None
        if raw_password is None or not raw_password
        else _decode_postgresql_url_component(raw_password)
    )
    ssl_mode = "require"
    if parsed.query:
        try:
            parameters = parse_qsl(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=2,
            )
        except ValueError:
            raise ValueError("PostgreSQL connection URL is invalid") from None
        if (
            len(parameters) != 1
            or parameters[0][0] != "sslmode"
            or parameters[0][1] not in SSL_MODES
        ):
            raise ValueError("PostgreSQL connection URL is invalid")
        ssl_mode = parameters[0][1]
    return host, port, database, username, password, ssl_mode


def _decode_postgresql_url_component(value: str) -> str:
    position = value.find("%")
    while position >= 0:
        if (
            position + 2 >= len(value)
            or value[position + 1] not in "0123456789abcdefABCDEF"
            or value[position + 2] not in "0123456789abcdefABCDEF"
        ):
            raise ValueError("PostgreSQL connection URL is invalid")
        position = value.find("%", position + 3)
    try:
        decoded = unquote(value, encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("PostgreSQL connection URL is invalid") from None
    if not decoded:
        raise ValueError("PostgreSQL connection URL is invalid")
    return decoded


def render_skill_editor_document(name: str, description: str, instructions: str) -> str:
    return f"# {name}\n\n{description}\n\n## Instructions\n\n{instructions}\n"


def parse_skill_editor_document(name: str, text: str) -> tuple[str, str]:
    prefix = f"# {name}\n\n"
    marker = "\n\n## Instructions\n\n"
    if not text.startswith(prefix) or not text.endswith("\n"):
        raise ValueError(
            "edited skill must keep the exact '# <name>' header and final newline"
        )
    rest = text[len(prefix) :]
    split_at = rest.find(marker)
    if split_at < 0:
        raise ValueError("edited skill must keep the '## Instructions' section")
    description = rest[:split_at].strip()
    instructions = rest[split_at + len(marker) :].strip()
    if not description or not instructions:
        raise ValueError("skill description and instructions must be non-empty")
    return description, instructions
