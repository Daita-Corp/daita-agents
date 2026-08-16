"""Slash-command parsing, completions, and one-run source selectors."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from .models import (
    MAX_POSTGRESQL_CONNECTION_URL_BYTES,
    SSL_MODES,
    SOURCE_TYPE_LABELS,
    UserInputError,
)
from .sanitization import safe_display

SKILL_DESCRIPTION_PLACEHOLDER = "Describe when the agent should use this skill."
SKILL_INSTRUCTIONS_PLACEHOLDER = "Write the reusable procedure here."

SLASH_COMMAND_COMPLETIONS = (
    ("/model", "/model", "Choose or validate the active model"),
    ("/sources", "/sources", "List registered data sources"),
    ("/source", "/source", "Choose the active query source"),
    ("/source use ", "/source use <name>", "Use a source for new conversations"),
    ("/source add", "/source add", "Add a data source"),
    ("/source edit", "/source edit", "Edit the active source connection"),
    ("/source refresh ", "/source refresh <id>", "Refresh a source catalog"),
    (
        "/source detach ",
        "/source detach <source>",
        "Detach a source and delete its Daita-owned credential",
    ),
    (
        "/source permissions",
        "/source permissions",
        "Configure read and PostgreSQL update access",
    ),
    ("/catalog", "/catalog", "Show the current catalog summary"),
    ("/settings", "/settings", "Show agent and model settings"),
    ("/new", "/new", "Start a new conversation"),
    ("/resume ", "/resume <id>", "Resume a previous conversation"),
    (
        "/conversation clear",
        "/conversation clear",
        "Delete all conversation history",
    ),
    ("/learn ", "/learn <material>", "Teach durable knowledge or a procedure"),
    (
        "/review",
        "/review [cost-usd]",
        "Review recent runs for memory or skill suggestions",
    ),
    (
        "/memory",
        "/memory",
        "Inspect global memory, semantics, learning candidates, and "
        "duplicate, stale, conflicting, and superseded states",
    ),
    ("/user", "/user", "View or edit the user profile"),
    ("/skills", "/skills", "List available skills"),
    ("/skills create", "/skills create", "Start guided skill creation"),
    (
        "/skills use ",
        "/skills use <name> [request]",
        "Invoke a skill by name",
    ),
    ("/status", "/status", "Show current agent status"),
    ("/conversation", "/conversation", "Show the current conversation ID"),
    ("/agent delete", "/agent delete", "Permanently delete this agent"),
    ("/help", "/help", "Show controls and usage help"),
    ("/exit", "/exit", "Exit Daita"),
)
BUILTIN_SLASH_COMMAND_ROOTS = frozenset(
    display.split(maxsplit=1)[0]
    for _insertion, display, _description in SLASH_COMMAND_COMPLETIONS
)
BUILTIN_SLASH_COMMANDS = frozenset(
    {
        "/agent",
        "/catalog",
        "/conversation",
        "/exit",
        "/help",
        "/learn",
        "/memory",
        "/model",
        "/new",
        "/review",
        "/resume",
        "/settings",
        "/source",
        "/sources",
        "/skills",
        "/status",
        "/user",
    }
)
HELP_TEXT = (
    "Type / to browse commands and their descriptions.\n"
    'Use @"source name" <question> to ask another source directly.\n'
    "Enter submit · Ctrl-J newline · Esc Esc clear input · Ctrl-D exit\n"
    "Wheel or Page Up/Page Down review · Ctrl-Home start · Ctrl-End latest\n"
    "Ctrl-O show/hide tool results · Ctrl-C copy selection or cancel the run\n"
    "Approvals accept only Yes or No; other input does not decide\n"
    "OSC 52 reports only that a copy request was sent, not that it succeeded"
)


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
        source_type = SOURCE_TYPE_LABELS.get(adapter_id, adapter_id)
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
