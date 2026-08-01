"""Focused terminal launcher through model and source onboarding."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import getpass
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any, cast, TextIO
import unicodedata
from urllib.parse import parse_qsl, unquote, urlsplit

from . import (
    ApprovalDecision,
    ApprovalRequest,
    SemanticAnnotationState,
    SemanticAnnotationView,
    __version__,
)
from .agent import (
    Agent,
    AgentAlreadyExistsError,
    AgentModelConfigurationError,
    AgentNameError,
    AgentNotFoundError,
    PostgreSQLProbeResult,
    PostgreSQLSourceError,
    SourceSelectionError,
)
from .config import AgentConfig
from .errors import ConfigError, LLMError
from .learning_candidates import (
    LEARNING_CANDIDATE_MAX_RECORDS,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    LearningCandidateView,
    LearningReviewResult,
    LearningReviewStatus,
    learning_candidate_content_from_mapping,
    learning_candidate_content_to_mapping,
)
from .security import KeychainStore
from .skills import Skill, validate_skill_name
from . import terminal_tui
from .terminal_selection import (
    SelectionCancelled,
    SelectionOption,
    select_many,
    select_one,
)

_PROVIDERS = (
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("gemini", "Gemini"),
    ("grok", "Grok"),
    ("ollama", "Ollama"),
    ("custom", "Custom OpenAI-compatible"),
)
_BUILTIN_PROVIDER_IDS = frozenset(provider for provider, _ in _PROVIDERS[:-1])


@dataclass(frozen=True, slots=True)
class _ModelSuggestion:
    provider_id: str
    model_id: str
    label: str
    description: str
    recommendation: str | None = None


_MODEL_SUGGESTIONS = {
    "openai": (
        _ModelSuggestion(
            "openai",
            "gpt-5.6-sol",
            "GPT-5.6 Sol",
            "Frontier capability for complex data-agent work",
            "Recommended",
        ),
        _ModelSuggestion(
            "openai",
            "gpt-5.6-terra",
            "GPT-5.6 Terra",
            "Balanced intelligence and cost for everyday workflows",
            "Balanced",
        ),
        _ModelSuggestion(
            "openai",
            "gpt-5.6-luna",
            "GPT-5.6 Luna",
            "Efficient model for high-volume bounded tasks",
            "Fast",
        ),
    ),
    "anthropic": (
        _ModelSuggestion(
            "anthropic",
            "claude-opus-4-8",
            "Claude Opus 4.8",
            "Complex agentic and enterprise work",
            "Strong",
        ),
        _ModelSuggestion(
            "anthropic",
            "claude-sonnet-5",
            "Claude Sonnet 5",
            "Fast balance of speed and intelligence",
            "Recommended",
        ),
        _ModelSuggestion(
            "anthropic",
            "claude-haiku-4-5-20251001",
            "Claude Haiku 4.5",
            "Low-latency near-frontier model",
            "Fast",
        ),
    ),
    "gemini": (
        _ModelSuggestion(
            "gemini",
            "gemini-3.6-flash",
            "Gemini 3.6 Flash",
            "Stable agentic model balancing speed and intelligence",
            "Recommended",
        ),
        _ModelSuggestion(
            "gemini",
            "gemini-3.5-flash",
            "Gemini 3.5 Flash",
            "Sustained performance for long-running agent work",
            "Strong",
        ),
        _ModelSuggestion(
            "gemini",
            "gemini-3.5-flash-lite",
            "Gemini 3.5 Flash-Lite",
            "Low-latency model for high-volume agent tasks",
            "Fast",
        ),
    ),
    "grok": (
        _ModelSuggestion(
            "grok",
            "grok-4.5",
            "Grok 4.5",
            "Agentic tool calling for general and code workflows",
            "Recommended",
        ),
    ),
    "ollama": (
        _ModelSuggestion(
            "ollama",
            "qwen3",
            "Qwen 3",
            "Common local model with tool and reasoning support",
            "Recommended",
        ),
        _ModelSuggestion(
            "ollama",
            "llama3.1",
            "Llama 3.1",
            "Common local model with tool-use support",
            "Lightweight",
        ),
        _ModelSuggestion(
            "ollama",
            "mistral-small3.2",
            "Mistral Small 3.2",
            "Local model improved for function calling",
            "Tool use",
        ),
    ),
}
_MANUAL_MODEL = "manual"
_BACK_TO_PROVIDERS = "back"
_VALIDATION_ERRORS = {
    "authentication_error": "The API key was rejected. Replace it and retry.",
    "model_not_found": "This account cannot access {model}.",
    "rate_limit_error": "The provider rate-limited the validation request.",
    "provider_unavailable": "The provider could not be reached.",
    "timeout": "The provider did not respond before the timeout.",
    "invalid_request": "The provider rejected this model configuration.",
    "output_limit": (
        "The model exhausted its validation output budget before calling the tool."
    ),
}
_MODEL_SETUP_ERRORS = {
    "secret_provider_unavailable": (
        "The API key could not be saved to the OS keychain. "
        "Check keychain access and retry."
    ),
    "secret_not_found": (
        "The saved API key could not be read from the OS keychain. "
        "Replace it and retry."
    ),
    "secret_provider_invalid_response": (
        "The OS keychain returned an invalid API key. Replace it and retry."
    ),
    "secret_scheme_unsupported": (
        "The configured credential store is not supported. "
        "Choose the model configuration again."
    ),
}
_SOURCE_TYPES = (
    ("sqlite", "SQLite file"),
    ("directory", "Local CSV/JSON directory"),
    ("postgresql", "PostgreSQL"),
)
_POSTGRESQL_CONNECTION_METHODS = (
    ("url", "Connection URL"),
    ("fields", "Individual fields"),
)
_POSTGRESQL_ERRORS = {
    "postgresql_connect_failed": (
        "Could not connect to PostgreSQL. Check that the database is running and "
        "verify the host, port, database, username, password, and SSL mode."
    ),
    "postgresql_credential_unavailable": (
        "The saved database password could not be read from the OS keychain. "
        "Replace it and retry."
    ),
    "postgresql_credential_invalid": (
        "The saved database password is empty or invalid. Replace it and retry."
    ),
    "postgresql_probe_failed": (
        "Connected to PostgreSQL, but schemas could not be inspected. "
        "Check the reader role's catalog permissions and retry."
    ),
    "postgresql_probe_result_invalid": (
        "PostgreSQL returned an invalid schema listing. Retry or check the "
        "server's catalog compatibility."
    ),
}
_SOURCE_SETUP_ERRORS = {
    "secret_provider_unavailable": (
        "The database password could not be saved to the OS keychain. "
        "Check keychain access and retry."
    ),
    "secret_scheme_unsupported": (
        "The configured credential store cannot save a database password."
    ),
}
_SSL_MODES = frozenset(
    {"disable", "prefer", "allow", "require", "verify-ca", "verify-full"}
)
_MAX_POSTGRESQL_CONNECTION_URL_BYTES = 64 * 1_024
_POSTGRESQL_CONNECTION_URL_ERROR = (
    "Enter a valid postgres:// or postgresql:// URL with a username, host, "
    "and database. Only the sslmode query parameter is supported."
)
_MAX_CHAT_INPUT_CHARACTERS = 16_384
_MAX_DISPLAY_CHARACTERS = 16_384
_MAX_APPROVAL_DOCUMENT_CHARACTERS = 64 * 1_024
_MAX_SOURCE_PICKER_OPTIONS = 128
_MAX_CATALOG_PREVIEW = 12
_MAX_SOURCE_PREVIEW = 8
_DEFAULT_CANDIDATE_REVIEW_COST_LIMIT_USD = Decimal("0.05")
_SKILL_DESCRIPTION_PLACEHOLDER = "Describe when the agent should use this skill."
_SKILL_INSTRUCTIONS_PLACEHOLDER = "Write the reusable procedure here."
_BUILTIN_SLASH_COMMANDS = frozenset(
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
_SOURCE_TYPE_LABELS = {
    "local-directory": "CSV/JSON",
    "postgresql": "PostgreSQL",
    "sqlite": "SQLite",
}
_SOURCE_READ_CAPABILITIES = {
    "local-directory": "CSV/JSON reads",
    "postgresql": "PostgreSQL queries",
    "sqlite": "SQLite queries",
}


def _validate_candidate_review_cost_limit(value: Decimal | None) -> None:
    if value is not None and (
        not isinstance(value, Decimal) or not value.is_finite() or value < 0
    ):
        raise ValueError(
            "candidate review cost ceiling must be finite and non-negative"
        )


async def run_terminal_application(
    *,
    root: str | Path | None = None,
    agent_name: str | None = None,
    reviewer_max_estimated_cost_usd: Decimal | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    hidden_input: Callable[[str], str] | None = None,
    keychain: KeychainStore | None = None,
    model_validator: Any = None,
    selection_input: Any = None,
    selection_output: Any = None,
    tui_input: Any = None,
    tui_output: Any = None,
) -> int:
    """Select one agent, derive readiness, and run integrated terminal chat."""

    _validate_candidate_review_cost_limit(reviewer_max_estimated_cost_usd)
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    assert input_stream is not None
    assert output_stream is not None
    if (tui_input is None) != (tui_output is None):
        raise ValueError("TUI input and output must be supplied together")
    read_hidden = (
        (
            lambda prompt: getpass.getpass(
                terminal_tui._setup_prompt_text(prompt, output_stream),
                stream=output_stream,
            )
        )
        if hidden_input is None
        else hidden_input
    )
    suspend_bridge = terminal_tui.TerminalSuspendBridge()
    observer_bridge = terminal_tui.TerminalObserverBridge()
    approval_bridge = terminal_tui.TerminalApprovalBridge(
        lambda request: suspend_bridge.run(
            lambda: _prompt_for_exact_approval(
                request,
                input_stream=input_stream,
                output_stream=output_stream,
            )
        )
    )
    approval_handler = approval_bridge
    agent: Agent | None = None
    try:
        agent = await _select_agent(
            root=root,
            requested_name=agent_name,
            input_stream=input_stream,
            output_stream=output_stream,
            keychain=keychain,
            model_validator=model_validator,
            approval_handler=approval_handler,
            observer_bridge=observer_bridge,
            selection_input=selection_input,
            selection_output=selection_output,
        )
        validated = False
        if agent.model_route is None:
            await _onboard_model(
                agent,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=read_hidden,
                selection_input=selection_input,
                selection_output=selection_output,
            )
            validated = True
            name = agent.name
            await agent.close()
            agent = await Agent.open(
                name,
                root=root,
                keychain=keychain,
                model_validator=model_validator,
                approval_handler=approval_handler,
                observer=observer_bridge,
            )
        if reviewer_max_estimated_cost_usd is not None:
            agent = await _reopen_with_candidate_reviewer(
                agent,
                root=root,
                keychain=keychain,
                model_validator=model_validator,
                approval_handler=approval_handler,
                observer_bridge=observer_bridge,
                max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
            )
        sources = tuple(
            source for source in await agent.list_sources() if source.active
        )
        if not sources:
            await _onboard_source(
                agent,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=read_hidden,
                selection_input=selection_input,
                selection_output=selection_output,
            )
            summary = await agent.catalog_summary()
            _write_catalog_phase(summary, output_stream)
            sources = tuple(
                source for source in await agent.list_sources() if source.active
            )
        else:
            summary = await agent.catalog_summary()

        if len(sources) > 1 and await agent.active_source() is None:
            await _select_query_source(
                agent,
                input_stream,
                output_stream,
                selection_input=selection_input,
                selection_output=selection_output,
            )

        conversation_id: str | None = None
        while True:
            while summary.is_empty:
                _write_stage_four_status(
                    agent,
                    sources,
                    summary,
                    output_stream,
                    validated=validated,
                )
                if (
                    await _select_catalog_repair(
                        input_stream,
                        output_stream,
                        selection_input=selection_input,
                        selection_output=selection_output,
                    )
                    == "exit"
                ):
                    return 0
                await _onboard_source(
                    agent,
                    input_stream=input_stream,
                    output_stream=output_stream,
                    hidden_input=read_hidden,
                    selection_input=selection_input,
                    selection_output=selection_output,
                )
                summary = await agent.catalog_summary()
                _write_catalog_phase(summary, output_stream)
                sources = tuple(
                    source for source in await agent.list_sources() if source.active
                )

            _write_stage_four_status(
                agent,
                sources,
                summary,
                output_stream,
                validated=validated,
            )
            agent, conversation_id, action = await _chat(
                agent,
                root=root,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=read_hidden,
                keychain=keychain,
                model_validator=model_validator,
                approval_handler=approval_handler,
                conversation_id=conversation_id,
                validated=validated,
                selection_input=selection_input,
                selection_output=selection_output,
                tui_input=tui_input,
                tui_output=tui_output,
                suspend_bridge=suspend_bridge,
                observer_bridge=observer_bridge,
                approval_bridge=approval_bridge,
            )
            if action == "exit":
                return 0
            if action == "deleted":
                return 0
            if action == "model":
                validated = True
                if reviewer_max_estimated_cost_usd is not None:
                    agent = await _reopen_with_candidate_reviewer(
                        agent,
                        root=root,
                        keychain=keychain,
                        model_validator=model_validator,
                        approval_handler=approval_handler,
                        observer_bridge=observer_bridge,
                        max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
                    )
            summary = await agent.catalog_summary()
            sources = tuple(
                source for source in await agent.list_sources() if source.active
            )
    except EOFError:
        print(file=output_stream)
        print("Setup cancelled.", file=output_stream)
        return 0
    except KeyboardInterrupt:
        print(file=output_stream)
        print("Setup interrupted.", file=output_stream)
        return 130
    finally:
        if agent is not None:
            await agent.close()


async def _select_agent(
    *,
    root: str | Path | None,
    requested_name: str | None,
    input_stream: TextIO,
    output_stream: TextIO,
    keychain: KeychainStore | None,
    model_validator: Any,
    approval_handler: Any,
    observer_bridge: terminal_tui.TerminalObserverBridge | None = None,
    selection_input: Any = None,
    selection_output: Any = None,
) -> Agent:
    if requested_name is not None:
        try:
            return await _open_selected_agent(
                requested_name,
                root=root,
                output_stream=output_stream,
                keychain=keychain,
                model_validator=model_validator,
                approval_handler=approval_handler,
                observer_bridge=observer_bridge,
            )
        except AgentNotFoundError as error:
            raise AgentNotFoundError(
                f"agent does not exist: {requested_name}. "
                "Run daita without --agent to select or create an agent."
            ) from error
    names = await Agent.list(root=root)
    if not names:
        return await _create_agent(
            root=root,
            input_stream=input_stream,
            output_stream=output_stream,
            keychain=keychain,
            model_validator=model_validator,
            approval_handler=approval_handler,
            observer_bridge=observer_bridge,
        )
    if len(names) == 1:
        return await _open_selected_agent(
            names[0],
            root=root,
            output_stream=output_stream,
            keychain=keychain,
            model_validator=model_validator,
            approval_handler=approval_handler,
            observer_bridge=observer_bridge,
        )

    options: tuple[SelectionOption[tuple[str, str]], ...] = tuple(
        SelectionOption[tuple[str, str]](("agent", name), name) for name in names
    ) + (
        SelectionOption[tuple[str, str]](
            ("create", ""),
            "Create a new agent",
        ),
    )
    action, selected_name = await select_one(
        "Select an agent",
        options,
        input_stream=input_stream,
        output_stream=output_stream,
        enhanced_input=selection_input,
        enhanced_output=selection_output,
    )
    if action == "agent":
        return await _open_selected_agent(
            selected_name,
            root=root,
            output_stream=output_stream,
            keychain=keychain,
            model_validator=model_validator,
            approval_handler=approval_handler,
            observer_bridge=observer_bridge,
        )
    return await _create_agent(
        root=root,
        input_stream=input_stream,
        output_stream=output_stream,
        keychain=keychain,
        model_validator=model_validator,
        approval_handler=approval_handler,
        observer_bridge=observer_bridge,
    )


async def _open_selected_agent(
    name: str,
    *,
    root: str | Path | None,
    output_stream: TextIO,
    keychain: KeychainStore | None,
    model_validator: Any,
    approval_handler: Any,
    observer_bridge: terminal_tui.TerminalObserverBridge | None = None,
) -> Agent:
    try:
        return await Agent.open(
            name,
            root=root,
            keychain=keychain,
            model_validator=model_validator,
            approval_handler=approval_handler,
            observer=observer_bridge,
        )
    except AgentModelConfigurationError:
        print(
            "The saved model configuration no longer meets current safety "
            "checks. Choose and validate a model again to replace it.",
            file=output_stream,
        )
        print(file=output_stream)
        return await Agent.open(
            name,
            root=root,
            config=AgentConfig(),
            keychain=keychain,
            model_validator=model_validator,
            approval_handler=approval_handler,
            observer=observer_bridge,
        )


async def _create_agent(
    *,
    root: str | Path | None,
    input_stream: TextIO,
    output_stream: TextIO,
    keychain: KeychainStore | None,
    model_validator: Any,
    approval_handler: Any,
    observer_bridge: terminal_tui.TerminalObserverBridge | None = None,
) -> Agent:
    while True:
        name = _read_line("Agent name: ", input_stream, output_stream).strip()
        try:
            return await Agent.create(
                name,
                root=root,
                keychain=keychain,
                model_validator=model_validator,
                approval_handler=approval_handler,
                observer=observer_bridge,
            )
        except (AgentNameError, AgentAlreadyExistsError) as error:
            print(str(error), file=output_stream)


async def _reopen_with_candidate_reviewer(
    agent: Agent,
    *,
    root: str | Path | None,
    keychain: KeychainStore | None,
    model_validator: Any,
    approval_handler: Any,
    observer_bridge: terminal_tui.TerminalObserverBridge | None,
    max_estimated_cost_usd: Decimal,
) -> Agent:
    """Reopen one selected agent with an explicitly enabled direct reviewer."""

    route = agent.model_route
    if route is None:
        raise RuntimeError("candidate review requires a configured model route")
    name = agent.name
    await agent.close()
    return await Agent.open(
        name,
        root=root,
        keychain=keychain,
        model_validator=model_validator,
        reviewer_max_estimated_cost_usd=max_estimated_cost_usd,
        approval_handler=approval_handler,
        observer=observer_bridge,
    )


def _read_line(
    prompt: str,
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    themed: bool = True,
) -> str:
    if themed:
        terminal_tui._write_setup_prompt(output_stream, prompt)
    else:
        print(prompt, end="", flush=True, file=output_stream)
    value = input_stream.readline()
    if value == "":
        raise EOFError
    return value.rstrip("\r\n")


async def _onboard_model(
    agent: Agent,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    hidden_input: Callable[[str], str],
    selection_input: Any = None,
    selection_output: Any = None,
) -> None:
    while True:
        provider, label = await _select_provider(
            input_stream,
            output_stream,
            selection_input=selection_input,
            selection_output=selection_output,
        )
        custom_provider = provider == "custom"
        if custom_provider:
            while True:
                provider = _read_required(
                    "Provider identifier: ",
                    input_stream,
                    output_stream,
                ).lower()
                if provider not in _BUILTIN_PROVIDER_IDS:
                    break
                print(
                    "Choose the matching built-in provider option instead.",
                    file=output_stream,
                )
            model = _read_required(
                "Model identifier: ",
                input_stream,
                output_stream,
            )
            if await _configure_selected_model(
                agent,
                provider=provider,
                model=model,
                custom_provider=True,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=hidden_input,
            ):
                return
            continue

        while True:
            selected_model = await _select_model(
                provider,
                label,
                input_stream=input_stream,
                output_stream=output_stream,
                selection_input=selection_input,
                selection_output=selection_output,
            )
            if selected_model is None:
                break
            if await _configure_selected_model(
                agent,
                provider=provider,
                model=selected_model,
                custom_provider=False,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=hidden_input,
            ):
                return


async def _select_model(
    provider: str,
    provider_label: str,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    selection_input: Any = None,
    selection_output: Any = None,
) -> str | None:
    suggestions = _MODEL_SUGGESTIONS[provider]
    options: tuple[SelectionOption[tuple[str, str]], ...] = tuple(
        SelectionOption[tuple[str, str]](
            ("model", suggestion.model_id),
            suggestion.label,
            " · ".join(
                item
                for item in (suggestion.recommendation, suggestion.description)
                if item
            ),
            search_terms=(suggestion.model_id,),
        )
        for suggestion in suggestions
    ) + (
        SelectionOption[tuple[str, str]](
            (_MANUAL_MODEL, ""),
            "Enter a model ID manually…",
            "Use any exact model ID",
            search_terms=("custom", "unlisted"),
        ),
        SelectionOption[tuple[str, str]](
            (_BACK_TO_PROVIDERS, ""),
            "Back to provider selection",
            "Choose a different provider",
        ),
    )
    print(file=output_stream)
    article = "an" if provider_label[:1].casefold() in "aeiou" else "a"
    try:
        action, model = await select_one(
            f"Select {article} {provider_label} model",
            options,
            input_stream=input_stream,
            output_stream=output_stream,
            enhanced_input=selection_input,
            enhanced_output=selection_output,
        )
    except SelectionCancelled:
        return None
    if action == _BACK_TO_PROVIDERS:
        return None
    if action == _MANUAL_MODEL:
        return _read_required(
            "Model identifier: ",
            input_stream,
            output_stream,
        )
    return model


async def _configure_selected_model(
    agent: Agent,
    *,
    provider: str,
    model: str,
    custom_provider: bool,
    input_stream: TextIO,
    output_stream: TextIO,
    hidden_input: Callable[[str], str],
) -> bool:
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    if agent.model_requires_explicit_limits(provider=provider, model=model):
        context_window_tokens, max_output_tokens = _read_explicit_model_limits(
            input_stream,
            output_stream,
        )
    base_url: str | None = None
    if provider == "ollama":
        base_url = (
            _read_line(
                "Base URL (optional; Enter for local Ollama): ",
                input_stream,
                output_stream,
            ).strip()
            or None
        )
    elif custom_provider:
        base_url = _read_required(
            "Base URL: ",
            input_stream,
            output_stream,
        )
    api_key: str | None = None
    if provider != "ollama":
        while not api_key:
            api_key = hidden_input("API key: ")
            if not api_key:
                print("API key cannot be empty.", file=output_stream)
    print(file=output_stream)
    print(
        "Validation contacts the provider and may incur a tiny API charge.",
        file=output_stream,
    )
    terminal_tui._write_setup_status(
        output_stream,
        "◐ Validating model configuration",
        role="progress",
    )
    try:
        try:
            await agent.configure_model(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
            )
            terminal_tui._write_setup_status(
                output_stream,
                "✓ Model configuration validated",
                role="success",
            )
            return True
        finally:
            api_key = None
    except LLMError as error:
        template = _VALIDATION_ERRORS.get(error.error_code)
        if template is None:
            print(
                "The provider could not validate this configuration.",
                file=output_stream,
            )
        else:
            print(
                template.format(
                    model=_safe_display(model, fallback="this model"),
                ),
                file=output_stream,
            )
    except ConfigError as error:
        print(
            _MODEL_SETUP_ERRORS.get(
                error.error_code,
                "The model configuration is invalid. "
                "Check the provider, model, and endpoint, then retry.",
            ),
            file=output_stream,
        )
    except ImportError as error:
        print(
            _safe_display(
                str(error),
                fallback="A required optional model dependency is unavailable.",
                maximum=512,
            ),
            file=output_stream,
        )
    except ValueError:
        print(
            "The model configuration is invalid. "
            "Check the provider, model, and endpoint, then retry.",
            file=output_stream,
        )
    except OSError:
        print(
            "The model configuration could not be saved. "
            "Check agent-home permissions and retry.",
            file=output_stream,
        )
    terminal_tui._write_setup_status(
        output_stream,
        "! Model validation failed",
        role="failure",
    )
    print("Choose the model configuration again.", file=output_stream)
    return False


def _read_explicit_model_limits(
    input_stream: TextIO,
    output_stream: TextIO,
) -> tuple[int, int]:
    print(
        "This unreviewed model requires explicit hard token limits.",
        file=output_stream,
    )
    context_window = _read_positive_integer(
        "Context window tokens: ",
        input_stream,
        output_stream,
    )
    while True:
        maximum_output = _read_positive_integer(
            "Maximum output tokens: ",
            input_stream,
            output_stream,
        )
        if maximum_output < context_window:
            return context_window, maximum_output
        print(
            "Maximum output tokens must be less than the context window.",
            file=output_stream,
        )


def _read_positive_integer(
    prompt: str,
    input_stream: TextIO,
    output_stream: TextIO,
) -> int:
    while True:
        raw = _read_line(prompt, input_stream, output_stream).strip()
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
        print("Enter a positive integer.", file=output_stream)


async def _select_provider(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    selection_input: Any = None,
    selection_output: Any = None,
) -> tuple[str, str]:
    print(file=output_stream)
    return await select_one(
        "Select a model provider",
        tuple(
            SelectionOption((provider, label), label) for provider, label in _PROVIDERS
        ),
        input_stream=input_stream,
        output_stream=output_stream,
        enhanced_input=selection_input,
        enhanced_output=selection_output,
    )


def _read_required(
    prompt: str,
    input_stream: TextIO,
    output_stream: TextIO,
) -> str:
    while True:
        value = _read_line(prompt, input_stream, output_stream).strip()
        if value:
            return value
        print("A value is required.", file=output_stream)


async def _onboard_source(
    agent: Agent,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    hidden_input: Callable[[str], str],
    selection_input: Any = None,
    selection_output: Any = None,
) -> Any:
    while True:
        source_type = await _select_source_type(
            input_stream,
            output_stream,
            selection_input=selection_input,
            selection_output=selection_output,
        )
        try:
            if source_type == "sqlite":
                path = _read_required(
                    "SQLite file: ",
                    input_stream,
                    output_stream,
                )
                name = (
                    _read_line(
                        "Display name (optional): ",
                        input_stream,
                        output_stream,
                    ).strip()
                    or None
                )
                terminal_tui._write_setup_status(
                    output_stream,
                    "… Discovering tables and relationships",
                    role="progress",
                )
                return await agent.attach_sqlite(
                    _absolute_user_path(path),
                    name=name,
                )
            if source_type == "directory":
                path = _read_required(
                    "CSV/JSON directory: ",
                    input_stream,
                    output_stream,
                )
                name = (
                    _read_line(
                        "Display name (optional): ",
                        input_stream,
                        output_stream,
                    ).strip()
                    or None
                )
                terminal_tui._write_setup_status(
                    output_stream,
                    "… Discovering tables and relationships",
                    role="progress",
                )
                return await agent.attach_local_directory(
                    _absolute_user_path(path),
                    name=name,
                )
            return await _onboard_postgresql(
                agent,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=hidden_input,
                selection_input=selection_input,
                selection_output=selection_output,
            )
        except EOFError:
            raise
        except PostgreSQLSourceError as error:
            print(
                _POSTGRESQL_ERRORS.get(
                    error.code,
                    "PostgreSQL setup failed without changing attached sources.",
                ),
                file=output_stream,
            )
        except ImportError as error:
            print(
                _safe_display(
                    str(error),
                    fallback="A required optional source dependency is unavailable.",
                    maximum=512,
                ),
                file=output_stream,
            )
        except ConfigError as error:
            print(
                _SOURCE_SETUP_ERRORS.get(
                    error.error_code,
                    "Source setup failed. Check the path or connection fields "
                    "and retry.",
                ),
                file=output_stream,
            )
        except ValueError:
            print(
                "Source setup failed. Check the path or connection fields and retry.",
                file=output_stream,
            )
        except OSError:
            print(
                "The source path could not be read. "
                "Check that it exists and is readable, then retry.",
                file=output_stream,
            )
        except Exception:
            print(
                "Source setup failed without changing committed catalog truth.",
                file=output_stream,
            )
        terminal_tui._write_setup_status(
            output_stream,
            "! Source validation failed",
            role="failure",
        )


async def _select_source_type(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    selection_input: Any = None,
    selection_output: Any = None,
) -> str:
    print(file=output_stream)
    return await select_one(
        "Select a data source",
        tuple(
            SelectionOption(source_type, label) for source_type, label in _SOURCE_TYPES
        ),
        input_stream=input_stream,
        output_stream=output_stream,
        enhanced_input=selection_input,
        enhanced_output=selection_output,
    )


async def _select_query_source(
    agent: Agent,
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    conversation_id: str | None = None,
    selection_input: Any = None,
    selection_output: Any = None,
) -> Any:
    sources = tuple(source for source in await agent.list_sources() if source.active)
    if not sources:
        raise SourceSelectionError(
            "No data sources are attached. Use /source add first."
        )
    current = await _active_source_for(
        agent,
        conversation_id=conversation_id,
    )
    ordered_sources = tuple(
        sorted(
            sources,
            key=lambda source: (
                current is None or source.id != current.id,
                source.display_name.casefold(),
                source.id,
            ),
        )
    )
    if len(ordered_sources) > _MAX_SOURCE_PICKER_OPTIONS:
        print(
            f"Showing {_MAX_SOURCE_PICKER_OPTIONS} sources. Use "
            "/source use <name> for another source.",
            file=output_stream,
        )
    selected_id = await select_one(
        "Select query source",
        tuple(
            SelectionOption(
                source.id,
                source.display_name,
                description=(
                    f"{_SOURCE_TYPE_LABELS.get(source.adapter_id, source.adapter_id)}"
                    + (
                        " · current"
                        if current is not None and source.id == current.id
                        else ""
                    )
                ),
                search_terms=(source.adapter_id, source.id),
            )
            for source in ordered_sources[:_MAX_SOURCE_PICKER_OPTIONS]
        ),
        input_stream=input_stream,
        output_stream=output_stream,
        enhanced_input=selection_input,
        enhanced_output=selection_output,
    )
    return await agent.select_source(selected_id)


async def _active_source_for(
    agent: Agent,
    *,
    conversation_id: str | None,
    sources: tuple[Any, ...] | None = None,
) -> Any:
    active_source = getattr(agent, "active_source", None)
    if callable(active_source):
        return await active_source(conversation_id=conversation_id)
    resolved_sources = (
        tuple(source for source in await agent.list_sources() if source.active)
        if sources is None
        else sources
    )
    return resolved_sources[0] if len(resolved_sources) == 1 else None


async def _source_status_label(
    agent: Agent,
    *,
    conversation_id: str | None,
    sources: tuple[Any, ...] | None = None,
) -> str:
    selected = await _active_source_for(
        agent,
        conversation_id=conversation_id,
        sources=sources,
    )
    if selected is not None:
        return _safe_display(selected.display_name, fallback="source")
    resolved_sources = (
        tuple(source for source in await agent.list_sources() if source.active)
        if sources is None
        else sources
    )
    if len(resolved_sources) == 1:
        return _safe_display(resolved_sources[0].display_name, fallback="source")
    return "select source" if resolved_sources else "none"


def _quoted_source_override(selector: str) -> str:
    escaped = selector.replace("\\", "\\\\").replace('"', '\\"')
    return f'@"{escaped}"'


def _source_override_completions(
    sources: tuple[Any, ...],
) -> tuple[tuple[str, str, str], ...]:
    active_sources = tuple(source for source in sources if source.active)
    folded_name_counts: dict[str, int] = {}
    for source in active_sources:
        folded = source.display_name.casefold()
        folded_name_counts[folded] = folded_name_counts.get(folded, 0) + 1

    completions: list[tuple[str, str, str]] = []
    for source in active_sources:
        display_name = _safe_display(
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
        source_type = _SOURCE_TYPE_LABELS.get(adapter_id, adapter_id)
        description = f"Ask one question using {source_type}"
        if not use_display_name:
            description += f" · {source.id[-8:]}"
        completions.append(
            (
                _quoted_source_override(selector) + " ",
                f"@{display_name}",
                description,
            )
        )
    return tuple(completions)


def _parse_source_override(message: str) -> tuple[str, str] | None:
    if not message.startswith("@"):
        return None
    if len(message) == 1 or message[1].isspace():
        raise terminal_tui.TerminalUserInputError(
            "Choose a source after @, then enter a question."
        )

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
                raise terminal_tui.TerminalUserInputError(
                    "Put a space after the quoted @ source name."
                )
            return "".join(selector_characters), message[position:].strip()
        if character == "\\" and position + 1 < len(message):
            escaped = message[position + 1]
            if escaped in {'"', "\\"}:
                selector_characters.append(escaped)
                position += 2
                continue
        selector_characters.append(character)
        position += 1
    raise terminal_tui.TerminalUserInputError(
        "Close the quoted @ source name before entering a question."
    )


async def _message_source_override(
    agent: Agent,
    message: str,
) -> tuple[str, str | None]:
    parsed = _parse_source_override(message)
    if parsed is None:
        return message, None
    selector, question = parsed
    try:
        source = await agent.resolve_source(selector)
    except SourceSelectionError as error:
        raise terminal_tui.TerminalUserInputError(str(error)) from error
    if not question:
        raise terminal_tui.TerminalUserInputError(
            "A source override must be followed by a question."
        )
    return question, source.id


async def _onboard_postgresql(
    agent: Agent,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    hidden_input: Callable[[str], str],
    selection_input: Any = None,
    selection_output: Any = None,
) -> Any:
    connection_method = await _select_postgresql_connection_method(
        input_stream,
        output_stream,
        selection_input=selection_input,
        selection_output=selection_output,
    )
    name = _read_required("Display name: ", input_stream, output_stream)
    password: str | None
    if connection_method == "url":
        host, port, database, username, password, ssl_mode = (
            _read_postgresql_connection_url(
                hidden_input,
                output_stream,
            )
        )
    else:
        host = _read_required("Host: ", input_stream, output_stream)
        port = _read_postgresql_port(input_stream, output_stream)
        database = _read_required("Database: ", input_stream, output_stream)
        username = _read_required("Username: ", input_stream, output_stream)
        password = None
        ssl_mode = _read_ssl_mode(input_stream, output_stream)
    while not password:
        password = hidden_input("Password: ")
        if not password:
            print("Password cannot be empty.", file=output_stream)
    reference = None
    attached = False
    try:
        credential = password
        password = None
        reference = await agent.store_postgresql_password(credential)
        credential = ""
        terminal_tui._write_setup_status(
            output_stream,
            "◐ Validating PostgreSQL connection",
            role="progress",
        )
        probe = await agent.probe_postgresql(
            host=host,
            port=port,
            database=database,
            username=username,
            credential=reference,
            ssl_mode=ssl_mode,
        )
        terminal_tui._write_setup_status(
            output_stream,
            "✓ Connection validated",
            role="success",
        )
        if probe.truncated:
            print(
                "Warning: more than 100 schemas exist; showing the first 100.",
                file=output_stream,
            )
        schemas = await _select_postgresql_schemas(
            probe,
            input_stream=input_stream,
            output_stream=output_stream,
            selection_input=selection_input,
            selection_output=selection_output,
        )
        terminal_tui._write_setup_status(
            output_stream,
            "✓ Schemas selected: "
            + ", ".join(_safe_display(schema, fallback="schema") for schema in schemas),
            role="success",
        )
        terminal_tui._write_setup_status(
            output_stream,
            "… Discovering tables and relationships",
            role="progress",
        )
        registration = await agent.attach_postgresql(
            host=host,
            port=port,
            database=database,
            username=username,
            credential=reference,
            schemas=schemas,
            ssl_mode=ssl_mode,
            name=name,
        )
        attached = True
        return registration
    finally:
        password = None
        if reference is not None and not attached:
            await agent.delete_postgresql_password(reference)


async def _select_postgresql_connection_method(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    selection_input: Any = None,
    selection_output: Any = None,
) -> str:
    print(file=output_stream)
    return await select_one(
        "How do you want to connect?",
        tuple(
            SelectionOption(method, label)
            for method, label in _POSTGRESQL_CONNECTION_METHODS
        ),
        input_stream=input_stream,
        output_stream=output_stream,
        enhanced_input=selection_input,
        enhanced_output=selection_output,
    )


def _read_postgresql_connection_url(
    hidden_input: Callable[[str], str],
    output_stream: TextIO,
) -> tuple[str, int, str, str, str | None, str]:
    while True:
        value = hidden_input("Connection URL: ")
        try:
            return _parse_postgresql_connection_url(value)
        except ValueError:
            print(_POSTGRESQL_CONNECTION_URL_ERROR, file=output_stream)
        finally:
            value = ""


def _parse_postgresql_connection_url(
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
        or encoded_length > _MAX_POSTGRESQL_CONNECTION_URL_BYTES
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
            or parameters[0][1] not in _SSL_MODES
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


def _read_postgresql_port(input_stream: TextIO, output_stream: TextIO) -> int:
    while True:
        value = _read_line(
            "Port (Enter for 5432): ",
            input_stream,
            output_stream,
        ).strip()
        try:
            port = 5432 if not value else int(value)
        except ValueError:
            port = 0
        if 1 <= port <= 65_535:
            return port
        print("Port must be from 1 through 65535.", file=output_stream)


def _read_ssl_mode(input_stream: TextIO, output_stream: TextIO) -> str:
    while True:
        value = (
            _read_line(
                "SSL mode (Enter for require): ",
                input_stream,
                output_stream,
            ).strip()
            or "require"
        )
        if value in _SSL_MODES:
            return value
        print(
            "SSL mode must be disable, prefer, allow, require, verify-ca, or verify-full.",
            file=output_stream,
        )


async def _select_postgresql_schemas(
    probe: PostgreSQLProbeResult,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    selection_input: Any = None,
    selection_output: Any = None,
) -> tuple[str, ...]:
    if not probe.schemas:
        raise PostgreSQLSourceError(
            "postgresql_probe_result_invalid",
            "PostgreSQL returned no selectable schemas.",
        )
    print(file=output_stream)
    return await select_many(
        "Select one or more schemas",
        tuple(
            SelectionOption(
                schema.name,
                schema.name,
                "base tables" if schema.has_base_tables else "no base tables",
            )
            for schema in probe.schemas
        ),
        input_stream=input_stream,
        output_stream=output_stream,
        enhanced_input=selection_input,
        enhanced_output=selection_output,
        maximum=32,
        empty_message="Select at least one schema.",
        maximum_message="Select at most 32 schemas.",
        invalid_message=(
            f"Choose 1 to {min(32, len(probe.schemas))} distinct schema numbers."
        ),
        fallback_prompt="Schemas (comma-separated numbers): ",
    )


def _absolute_user_path(value: str) -> Path:
    return Path(value).expanduser().absolute()


async def _chat(
    agent: Agent,
    *,
    root: str | Path | None,
    input_stream: TextIO,
    output_stream: TextIO,
    hidden_input: Callable[[str], str],
    keychain: KeychainStore | None,
    model_validator: Any,
    approval_handler: Any,
    conversation_id: str | None,
    validated: bool,
    selection_input: Any = None,
    selection_output: Any = None,
    tui_input: Any = None,
    tui_output: Any = None,
    suspend_bridge: terminal_tui.TerminalSuspendBridge | None = None,
    observer_bridge: terminal_tui.TerminalObserverBridge | None = None,
    approval_bridge: terminal_tui.TerminalApprovalBridge | None = None,
) -> tuple[Agent, str | None, str]:
    """Own only terminal-local selection and presentation around Agent.run()."""

    observer_bridge = observer_bridge or terminal_tui.TerminalObserverBridge()
    if terminal_tui.supports_terminal_tui(
        input_stream,
        output_stream,
        enhanced_input=tui_input,
        enhanced_output=tui_output,
    ):
        bridge = suspend_bridge or terminal_tui.TerminalSuspendBridge()
        try:
            return await _chat_tui(
                agent,
                root=root,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=hidden_input,
                keychain=keychain,
                model_validator=model_validator,
                approval_handler=approval_handler,
                conversation_id=conversation_id,
                validated=validated,
                selection_input=selection_input,
                selection_output=selection_output,
                tui_input=tui_input,
                tui_output=tui_output,
                suspend_bridge=bridge,
                observer_bridge=observer_bridge,
                approval_bridge=approval_bridge,
            )
        except terminal_tui.TerminalTUIUnavailable:
            pass

    await _write_ready_screen(
        agent,
        output_stream,
        conversation_id=conversation_id,
        validated=validated,
    )
    while True:
        try:
            capabilities = terminal_tui._terminal_capabilities(
                text_stream=output_stream
            )
            prompt = f"You {terminal_tui._terminal_glyphs(capabilities).prompt} "
            message = _read_line(
                prompt,
                input_stream,
                output_stream,
                themed=False,
            )
        except EOFError:
            print(file=output_stream)
            _write_resume_hint(conversation_id, output_stream)
            return agent, conversation_id, "exit"
        except KeyboardInterrupt:
            print(file=output_stream)
            print("Input interrupted.", file=output_stream)
            continue
        message = message.strip()
        if not message:
            continue
        if len(message) > _MAX_CHAT_INPUT_CHARACTERS:
            print(
                f"Input exceeds {_MAX_CHAT_INPUT_CHARACTERS} characters.",
                file=output_stream,
            )
            continue
        if message.startswith("/"):
            try:
                learning_invocation = _learning_invocation_message(message)
            except ValueError as error:
                print(
                    "Learning command failed: "
                    + _safe_display(str(error), fallback="invalid teaching request"),
                    file=output_stream,
                )
                continue
            if learning_invocation is not None:
                message = learning_invocation
            else:
                try:
                    skill_invocation = await _skill_invocation_message(agent, message)
                except ValueError as error:
                    print(
                        "Skill invocation failed: "
                        + _safe_display(str(error), fallback="invalid invocation"),
                        file=output_stream,
                    )
                    continue
                if skill_invocation is None:
                    agent, conversation_id, action = await _handle_local_command(
                        message,
                        agent=agent,
                        root=root,
                        input_stream=input_stream,
                        output_stream=output_stream,
                        hidden_input=hidden_input,
                        keychain=keychain,
                        model_validator=model_validator,
                        approval_handler=approval_handler,
                        conversation_id=conversation_id,
                        validated=validated,
                        selection_input=selection_input,
                        selection_output=selection_output,
                        observer_bridge=observer_bridge,
                    )
                    if action is not None:
                        return agent, conversation_id, action
                    continue

        creates_conversation = conversation_id is None
        try:
            result = await _run_message(
                agent,
                message,
                conversation_id=conversation_id,
                output_stream=output_stream,
            )
        except terminal_tui.TerminalUserInputError as error:
            print(
                _safe_display(str(error), fallback="Invalid source selection."),
                file=output_stream,
            )
            continue
        if result is None:
            continue
        conversation_id = result.conversation_id
        if creates_conversation:
            print(
                f"Conversation  {_safe_display(conversation_id, fallback='new')}",
                file=output_stream,
            )
        print(file=output_stream)
        print("Daita", file=output_stream)
        if result.final_text is not None:
            print(
                _render_model_answer(
                    result.final_text,
                    fallback="(empty response)",
                    maximum=None,
                ),
                file=output_stream,
            )
        else:
            print(
                f"{result.kind.value}: "
                f"{_safe_display(result.reason, fallback='failed')}",
                file=output_stream,
            )
        print(file=output_stream)
        print(
            f"{result.steps} steps · {result.usage.total_tokens} tokens · "
            f"{result.usage.cost_estimate.render()}",
            file=output_stream,
        )
        print(file=output_stream)


class _TerminalCommandOutput:
    def __init__(
        self,
        output_stream: TextIO,
        *,
        passthrough: bool,
    ) -> None:
        self._output_stream = output_stream
        self._passthrough = passthrough
        self._recording = io.StringIO()

    def write(self, value: str) -> int:
        written = self._output_stream.write(value) if self._passthrough else None
        self._recording.write(value)
        return len(value) if written is None else written

    def flush(self) -> None:
        if self._passthrough:
            self._output_stream.flush()

    def isatty(self) -> bool:
        return self._output_stream.isatty()

    def fileno(self) -> int:
        return self._output_stream.fileno()

    @property
    def value(self) -> str:
        return self._recording.getvalue()


def _command_uses_terminal_prompts(command: str) -> bool:
    parts = command.split()
    if not parts:
        return False
    if parts[0] == "/model" and len(parts) == 1:
        return True
    if parts[0] == "/source" and parts[1:] in ([], ["use"]):
        return True
    if parts[0] == "/source" and parts[1:] == ["add"]:
        return True
    if parts[0] == "/source" and len(parts) >= 3 and parts[1] == "detach":
        return True
    if parts[0] == "/conversation" and parts[1:] == ["clear"]:
        return True
    if parts[0] == "/agent" and parts[1:] == ["delete"]:
        return True
    if parts[0] in {"/memory", "/user"} and parts[1:] == ["edit"]:
        return True
    if parts[0] == "/review" and len(parts) == 1:
        return True
    if parts[0] == "/memory" and len(parts) == 3 and parts[1] == "edit":
        return True
    if parts[0] == "/skills" and parts[1:] == ["create"]:
        return True
    return (
        parts[0] == "/skills"
        and len(parts) == 3
        and parts[1] in {"create", "edit", "delete"}
    )


async def _chat_tui(
    agent: Agent,
    *,
    root: str | Path | None,
    input_stream: TextIO,
    output_stream: TextIO,
    hidden_input: Callable[[str], str],
    keychain: KeychainStore | None,
    model_validator: Any,
    approval_handler: Any,
    conversation_id: str | None,
    validated: bool,
    selection_input: Any,
    selection_output: Any,
    tui_input: Any,
    tui_output: Any,
    suspend_bridge: terminal_tui.TerminalSuspendBridge,
    observer_bridge: terminal_tui.TerminalObserverBridge,
    approval_bridge: terminal_tui.TerminalApprovalBridge | None,
) -> tuple[Agent, str | None, str]:
    sources = tuple(source for source in await agent.list_sources() if source.active)
    state = await _ready_view_state(
        agent,
        conversation_id=conversation_id,
        validated=validated,
        sources=sources,
    )

    async def run_message(message: str, selected_conversation: str | None) -> Any:
        return await _run_message(
            agent,
            message,
            conversation_id=selected_conversation,
            output_stream=None,
        )

    async def load_transcript(run_id: str) -> Any:
        return await agent.transcript(run_id)

    async def load_skill_completions() -> tuple[tuple[str, str], ...]:
        list_skills = getattr(agent, "list_skills", None)
        if not callable(list_skills):
            return ()
        return tuple(
            (summary.name, summary.description) for summary in await list_skills()
        )

    async def load_source_completions() -> tuple[tuple[str, str, str], ...]:
        return _source_override_completions(tuple(await agent.list_sources()))

    async def handle_command(
        command: str,
        selected_conversation: str | None,
    ) -> terminal_tui.TerminalCommandResult:
        nonlocal agent
        try:
            learning_invocation = _learning_invocation_message(command)
        except ValueError as error:
            return terminal_tui.TerminalCommandResult(
                conversation_id=selected_conversation,
                output=(
                    "Learning command failed: "
                    + _safe_display(
                        str(error),
                        fallback="invalid teaching request",
                    )
                    + "\n"
                ),
                source_summary=await _source_status_label(
                    agent,
                    conversation_id=selected_conversation,
                ),
            )
        if learning_invocation is not None:
            return terminal_tui.TerminalCommandResult(
                conversation_id=selected_conversation,
                model_message=learning_invocation,
                source_summary=await _source_status_label(
                    agent,
                    conversation_id=selected_conversation,
                ),
            )
        try:
            skill_invocation = await _skill_invocation_message(agent, command)
        except ValueError as error:
            return terminal_tui.TerminalCommandResult(
                conversation_id=selected_conversation,
                output=(
                    "Skill invocation failed: "
                    + _safe_display(str(error), fallback="invalid invocation")
                    + "\n"
                ),
                source_summary=await _source_status_label(
                    agent,
                    conversation_id=selected_conversation,
                ),
            )
        if skill_invocation is not None:
            return terminal_tui.TerminalCommandResult(
                conversation_id=selected_conversation,
                model_message=skill_invocation,
                source_summary=await _source_status_label(
                    agent,
                    conversation_id=selected_conversation,
                ),
            )
        passthrough = _command_uses_terminal_prompts(command)
        captured = _TerminalCommandOutput(
            output_stream,
            passthrough=passthrough,
        )
        enhanced_selection_input = (
            suspend_bridge.enhanced_input
            if suspend_bridge.enhanced_input is not None
            else selection_input
        )
        enhanced_selection_output = (
            suspend_bridge.enhanced_output
            if suspend_bridge.enhanced_output is not None
            else selection_output
        )
        agent, selected_conversation, action = await _handle_local_command(
            command,
            agent=agent,
            root=root,
            input_stream=input_stream,
            output_stream=cast(TextIO, captured),
            hidden_input=hidden_input,
            keychain=keychain,
            model_validator=model_validator,
            approval_handler=approval_handler,
            conversation_id=selected_conversation,
            validated=validated,
            selection_input=enhanced_selection_input,
            selection_output=enhanced_selection_output,
            observer_bridge=observer_bridge,
        )
        return terminal_tui.TerminalCommandResult(
            conversation_id=selected_conversation,
            action=action,
            output="" if passthrough else captured.value,
            presentation={
                "/status": "status",
                "/sources": "sources",
                "/catalog": "catalog",
                "/settings": "settings",
            }.get(command.split(maxsplit=1)[0], "local"),
            source_summary=(
                None
                if action == "deleted"
                else await _source_status_label(
                    agent,
                    conversation_id=selected_conversation,
                )
            ),
        )

    result = await terminal_tui.run_terminal_tui(
        state,
        run_message=run_message,
        load_transcript=load_transcript,
        handle_command=handle_command,
        command_requires_suspension=_command_uses_terminal_prompts,
        skill_completions=await load_skill_completions(),
        load_skill_completions=load_skill_completions,
        source_completions=_source_override_completions(sources),
        load_source_completions=load_source_completions,
        input_stream=input_stream,
        output_stream=output_stream,
        suspend_bridge=suspend_bridge,
        observer_bridge=observer_bridge,
        approval_bridge=approval_bridge,
        enhanced_input=tui_input,
        enhanced_output=tui_output,
    )
    conversation_id = result.conversation_id
    if result.action == "exit":
        _write_resume_hint(conversation_id, output_stream)
    return agent, conversation_id, result.action


async def _run_message(
    agent: Agent,
    message: str,
    *,
    conversation_id: str | None,
    output_stream: TextIO | None,
) -> Any:
    effective_message, override_source_id = await _message_source_override(
        agent,
        message,
    )
    run_arguments: dict[str, Any] = {"conversation_id": conversation_id}
    if override_source_id is not None:
        run_arguments["source_id"] = override_source_id
    run_request = agent.run(effective_message, **run_arguments)
    run = asyncio.create_task(run_request)
    try:
        return await run
    except SourceSelectionError as error:
        raise terminal_tui.TerminalUserInputError(str(error)) from error
    except (asyncio.CancelledError, KeyboardInterrupt):
        run.cancel()
        while not run.done():
            try:
                await asyncio.shield(run)
            except (asyncio.CancelledError, Exception):
                pass
        current = asyncio.current_task()
        if current is not None:
            while current.cancelling():
                current.uncancel()
        if output_stream is not None:
            print(file=output_stream)
            print("Run interrupted; returning to the prompt.", file=output_stream)
        return None


async def _handle_local_command(
    command: str,
    *,
    agent: Agent,
    root: str | Path | None,
    input_stream: TextIO,
    output_stream: TextIO,
    hidden_input: Callable[[str], str],
    keychain: KeychainStore | None,
    model_validator: Any,
    approval_handler: Any,
    conversation_id: str | None,
    validated: bool,
    selection_input: Any = None,
    selection_output: Any = None,
    observer_bridge: terminal_tui.TerminalObserverBridge | None = None,
) -> tuple[Agent, str | None, str | None]:
    parts = command.split()
    name = parts[0] if parts else ""
    if name == "/exit" and len(parts) == 1:
        return agent, conversation_id, "exit"
    if name == "/help" and len(parts) == 1:
        _write_chat_help(output_stream)
        return agent, conversation_id, None
    if name == "/new" and len(parts) == 1:
        print("Conversation  new", file=output_stream)
        return agent, None, None
    if name == "/resume" and len(parts) == 2:
        candidate = parts[1]
        try:
            exists = await agent.conversation_exists(candidate)
        except (TypeError, ValueError) as error:
            print(
                "Cannot resume conversation: "
                + _safe_display(str(error), fallback="invalid conversation"),
                file=output_stream,
            )
            return agent, conversation_id, None
        if not exists:
            print(
                "Cannot resume conversation: unknown conversation for this agent",
                file=output_stream,
            )
            return agent, conversation_id, None
        print(
            f"Conversation  {_safe_display(candidate, fallback='new')}",
            file=output_stream,
        )
        return agent, candidate, None
    if name == "/sources" and len(parts) == 1:
        await _write_sources(
            agent,
            output_stream,
            conversation_id=conversation_id,
        )
        return agent, conversation_id, None
    if name == "/catalog" and len(parts) == 1:
        await _write_catalog_preview(agent, output_stream)
        return agent, conversation_id, None
    if name == "/settings" and len(parts) == 1:
        _write_settings(agent, output_stream)
        return agent, conversation_id, None
    if name == "/status" and len(parts) == 1:
        await _write_ready_screen(
            agent,
            output_stream,
            conversation_id=conversation_id,
            validated=validated,
        )
        return agent, conversation_id, None
    if name == "/conversation" and len(parts) == 1:
        print(
            "Conversation  " f"{_safe_display(conversation_id, fallback='new')}",
            file=output_stream,
        )
        return agent, conversation_id, None
    if name == "/conversation" and parts[1:] == ["clear"]:
        confirmation = _read_line(
            "Clear all conversation history and learning candidate records? [y/N]: ",
            input_stream,
            output_stream,
        )
        if confirmation.strip().lower() != "y":
            print("Conversation history was not changed.", file=output_stream)
            return agent, conversation_id, None
        cleared = await agent.clear_conversations()
        print(
            f"Cleared {cleared} persisted conversation "
            f"{'run' if cleared == 1 else 'runs'}.",
            file=output_stream,
        )
        print("Approved memory and skills were preserved.", file=output_stream)
        return agent, None, None
    if name == "/agent" and parts[1:] == ["delete"]:
        selected_name = agent.name
        confirmation = _read_line(
            f"Type {selected_name} to permanently delete this agent: ",
            input_stream,
            output_stream,
        )
        if confirmation != selected_name:
            print("Agent was not deleted.", file=output_stream)
            return agent, conversation_id, None
        await agent.close()
        try:
            await Agent.delete(selected_name, root=root, keychain=keychain)
        except Exception as error:
            print(
                "Agent deletion failed: "
                + _safe_display(str(error), fallback="deletion failed"),
                file=output_stream,
            )
            replacement = await Agent.open(
                selected_name,
                root=root,
                keychain=keychain,
                model_validator=model_validator,
                approval_handler=approval_handler,
                observer=observer_bridge,
            )
            return replacement, conversation_id, None
        print(f"Deleted agent {selected_name}.", file=output_stream)
        return agent, None, "deleted"
    if name == "/source" and (len(parts) == 1 or parts[1:] == ["use"]):
        prior = await _active_source_for(
            agent,
            conversation_id=conversation_id,
        )
        try:
            selected = await _select_query_source(
                agent,
                input_stream,
                output_stream,
                conversation_id=conversation_id,
                selection_input=selection_input,
                selection_output=selection_output,
            )
        except SelectionCancelled:
            print("Source selection cancelled; returning to chat.", file=output_stream)
            return agent, conversation_id, None
        except SourceSelectionError as error:
            print(
                _safe_display(str(error), fallback="Source selection failed."),
                file=output_stream,
            )
            return agent, conversation_id, None
        changed = prior is None or prior.id != selected.id
        print(
            f"Source  {_safe_display(selected.display_name, fallback='source')}",
            file=output_stream,
        )
        if changed and conversation_id is not None:
            print(
                "Started a new conversation to keep source context isolated.",
                file=output_stream,
            )
            conversation_id = None
        return agent, conversation_id, None
    if name == "/source" and len(parts) >= 3 and parts[1] == "use":
        prior = await _active_source_for(
            agent,
            conversation_id=conversation_id,
        )
        try:
            selected = await agent.select_source(" ".join(parts[2:]))
        except SourceSelectionError as error:
            print(
                _safe_display(str(error), fallback="Source selection failed."),
                file=output_stream,
            )
            return agent, conversation_id, None
        changed = prior is None or prior.id != selected.id
        print(
            f"Source  {_safe_display(selected.display_name, fallback='source')}",
            file=output_stream,
        )
        if changed and conversation_id is not None:
            print(
                "Started a new conversation to keep source context isolated.",
                file=output_stream,
            )
            conversation_id = None
        return agent, conversation_id, None
    if name == "/source" and parts[1:] == ["add"]:
        try:
            await _onboard_source(
                agent,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=hidden_input,
                selection_input=selection_input,
                selection_output=selection_output,
            )
        except SelectionCancelled:
            print("Source setup cancelled; returning to chat.", file=output_stream)
            return agent, conversation_id, None
        summary = await agent.catalog_summary()
        _write_catalog_phase(summary, output_stream)
        if summary.is_empty:
            return agent, conversation_id, "repair"
        await _write_ready_screen(
            agent,
            output_stream,
            conversation_id=conversation_id,
            validated=validated,
        )
        return agent, conversation_id, None
    if name == "/source" and len(parts) >= 3 and parts[1] == "detach":
        try:
            selected = await agent.resolve_source(" ".join(parts[2:]))
        except SourceSelectionError as error:
            print(
                _safe_display(str(error), fallback="Source selection failed."),
                file=output_stream,
            )
            return agent, conversation_id, None
        active = await _active_source_for(
            agent,
            conversation_id=conversation_id,
        )
        confirmation = _read_line(
            "Detach "
            f"{_safe_display(selected.display_name, fallback='this source')} "
            "and delete its Daita-owned credential? [y/N]: ",
            input_stream,
            output_stream,
        )
        if confirmation.strip().lower() != "y":
            print("Source was not detached.", file=output_stream)
            return agent, conversation_id, None
        detached_succeeded = False
        try:
            await agent.detach(selected.id)
        except Exception as error:
            current = next(
                (
                    source
                    for source in await agent.list_sources()
                    if source.id == selected.id
                ),
                None,
            )
            detached_succeeded = current is not None and not current.active
            print(
                "Source detachment needs attention: "
                + _safe_display(str(error), fallback="credential cleanup failed"),
                file=output_stream,
            )
        else:
            detached_succeeded = True
            print(
                f"Detached source "
                f"{_safe_display(selected.display_name, fallback='source')}.",
                file=output_stream,
            )
        if detached_succeeded and active is not None and active.id == selected.id:
            conversation_id = None
            print(
                "Started a new conversation because its source was detached.",
                file=output_stream,
            )
        return agent, conversation_id, "sources" if detached_succeeded else None
    if name == "/source" and len(parts) == 3 and parts[1] == "refresh":
        try:
            await agent.refresh_source(parts[2])
        except Exception:
            print(
                "Source refresh failed without replacing committed catalog truth.",
                file=output_stream,
            )
            return agent, conversation_id, None
        summary = await agent.catalog_summary()
        _write_catalog_phase(summary, output_stream)
        if summary.is_empty:
            return agent, conversation_id, "repair"
        await _write_ready_screen(
            agent,
            output_stream,
            conversation_id=conversation_id,
            validated=validated,
        )
        return agent, conversation_id, None
    if name == "/model" and len(parts) == 1:
        _write_model_configuration(agent, output_stream)
        change = _read_line(
            "Change model? [y/N]: ",
            input_stream,
            output_stream,
        ).strip()
        if change.lower() != "y":
            return agent, conversation_id, None
        try:
            await _onboard_model(
                agent,
                input_stream=input_stream,
                output_stream=output_stream,
                hidden_input=hidden_input,
                selection_input=selection_input,
                selection_output=selection_output,
            )
        except SelectionCancelled:
            print("Model setup cancelled; returning to chat.", file=output_stream)
            return agent, conversation_id, None
        selected_name = agent.name
        await agent.close()
        replacement = await Agent.open(
            selected_name,
            root=root,
            keychain=keychain,
            model_validator=model_validator,
            approval_handler=approval_handler,
            observer=observer_bridge,
        )
        try:
            print(
                "Model configuration updated for subsequent runs.", file=output_stream
            )
        except BaseException:
            await replacement.close()
            raise
        return replacement, conversation_id, "model"
    try:
        if await _handle_knowledge_command(
            parts,
            agent=agent,
            input_stream=input_stream,
            output_stream=output_stream,
        ):
            return agent, conversation_id, None
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(
            "Knowledge command failed: "
            + _safe_display(str(error), fallback="command failed"),
            file=output_stream,
        )
        return agent, conversation_id, None

    if name == "/resume":
        print("Usage: /resume <conversation-id>", file=output_stream)
    elif name == "/source":
        print(
            "Usage: /source | /source use <name> | /source add | "
            "/source refresh <source-id> | /source detach <source>",
            file=output_stream,
        )
    elif name == "/conversation":
        print("Usage: /conversation | /conversation clear", file=output_stream)
    elif name == "/agent":
        print("Usage: /agent delete", file=output_stream)
    elif name in {
        "/exit",
        "/help",
        "/new",
        "/sources",
        "/catalog",
        "/settings",
        "/status",
        "/model",
    }:
        print(f"Usage: {name}", file=output_stream)
    else:
        print("Unknown command. Type /help for commands.", file=output_stream)
    return agent, conversation_id, None


async def _write_ready_screen(
    agent: Agent,
    output_stream: TextIO,
    *,
    conversation_id: str | None,
    validated: bool,
) -> None:
    state = await _ready_view_state(
        agent,
        conversation_id=conversation_id,
        validated=validated,
    )
    interactive = terminal_tui._stream_is_interactive(output_stream)
    detected = terminal_tui._terminal_capabilities(text_stream=output_stream)
    capabilities = (
        detected
        if interactive
        else terminal_tui.TerminalCapabilities("none", detected.unicode)
    )
    output_stream.write(
        terminal_tui._render_startup_text(
            state,
            width=terminal_tui._text_stream_width(output_stream),
            capabilities=capabilities,
        )
    )
    output_stream.flush()


async def _ready_view_state(
    agent: Agent,
    *,
    conversation_id: str | None,
    validated: bool,
    sources: tuple[Any, ...] | None = None,
) -> terminal_tui.TerminalViewState:
    """Assemble only safe, public runtime facts for terminal presentation."""

    route = agent.model_route
    if route is None:
        raise RuntimeError("ready chat requires a configured model")
    candidate = route.candidates[0]
    provider, _, model = candidate.provider_id.partition(":")
    provider_label = dict(_PROVIDERS).get(provider, provider)
    sources = (
        tuple(source for source in await agent.list_sources() if source.active)
        if sources is None
        else tuple(source for source in sources if source.active)
    )
    summary = await agent.catalog_summary()
    source_summary = await _source_status_label(
        agent,
        conversation_id=conversation_id,
        sources=sources,
    )
    adapter_counts: dict[str, int] = {}
    for source in sources:
        adapter_id = _safe_display(
            getattr(source, "adapter_id", None),
            fallback="source",
        )
        adapter_counts[adapter_id] = adapter_counts.get(adapter_id, 0) + 1
    source_types = tuple(
        (
            f"{count} {_SOURCE_TYPE_LABELS.get(adapter_id, adapter_id)}"
            if count > 1
            else _SOURCE_TYPE_LABELS.get(adapter_id, adapter_id)
        )
        for adapter_id, count in sorted(adapter_counts.items())
    )
    read_capabilities: list[str] = []
    if sources:
        read_capabilities.append("Catalog search & inspection")
    read_capabilities.extend(
        _SOURCE_READ_CAPABILITIES[adapter_id]
        for adapter_id in sorted(adapter_counts)
        if adapter_id in _SOURCE_READ_CAPABILITIES
    )
    warnings: list[str] = []
    if not sources:
        warnings.append("No data sources. Use /source add to attach one.")
    elif source_summary == "select source":
        warnings.append("Select a query source with /source before asking a question.")
    elif getattr(summary, "is_empty", summary.resource_count == 0):
        warnings.append(
            "Catalog is empty. Use /source refresh <id> after checking source access."
        )
    agent_home = getattr(agent, "home", None)
    model_profile = getattr(agent, "model_profile", None)
    context_capacity_tokens = getattr(
        model_profile,
        "maximum_input_tokens",
        None,
    )
    if (
        not isinstance(context_capacity_tokens, int)
        or isinstance(context_capacity_tokens, bool)
        or context_capacity_tokens < 1
    ):
        context_capacity_tokens = None
    return terminal_tui.TerminalViewState(
        agent_label=_safe_display(agent.name, fallback="agent"),
        model_label=_safe_display(model, fallback="model"),
        source_summary=source_summary,
        conversation_id=conversation_id,
        context_capacity_tokens=context_capacity_tokens,
        startup=terminal_tui.TerminalStartupInfo(
            version=__version__,
            provider_label=_safe_display(provider_label, fallback="provider"),
            model_status="validated" if validated else "configured",
            agent_home=_safe_display(
                (str(agent_home) if isinstance(agent_home, (str, Path)) else None),
                fallback="unavailable",
            ),
            source_count=len(sources),
            source_types=source_types,
            source_names=tuple(
                _safe_display(source.display_name, fallback="source")
                for source in sources[:3]
            ),
            resource_count=summary.resource_count,
            relationship_count=summary.relationship_count,
            read_capabilities=tuple(read_capabilities),
            warnings=tuple(warnings),
        ),
    )


async def _write_sources(
    agent: Agent,
    output_stream: TextIO,
    *,
    conversation_id: str | None,
) -> None:
    sources = tuple(source for source in await agent.list_sources() if source.active)
    selected = await _active_source_for(
        agent,
        conversation_id=conversation_id,
    )
    summary = await agent.catalog_summary()
    print("Sources", file=output_stream)
    if not sources:
        print("  (none)", file=output_stream)
    for source in sources[:_MAX_SOURCE_PREVIEW]:
        print(
            f"  {'●' if selected is not None and source.id == selected.id else '○'} "
            f"{_safe_display(source.display_name)} · "
            f"{_safe_display(source.adapter_id, fallback='adapter')} · "
            f"{_safe_display(source.id, fallback='source')}",
            file=output_stream,
        )
    if len(sources) > _MAX_SOURCE_PREVIEW:
        print(
            f"  +{len(sources) - _MAX_SOURCE_PREVIEW} more",
            file=output_stream,
        )
    print(
        "Catalog  "
        f"{_count_label(summary.resource_count, 'resource', 'resources')} · "
        f"{_count_label(summary.relationship_count, 'relationship', 'relationships')}",
        file=output_stream,
    )


async def _write_catalog_preview(agent: Agent, output_stream: TextIO) -> None:
    summary = await agent.catalog_summary()
    resources = await agent.catalog_preview(limit=_MAX_CATALOG_PREVIEW)
    print("Catalog preview", file=output_stream)
    if not resources:
        print("  (empty)", file=output_stream)
    for resource in resources:
        print(
            "  "
            f"{_safe_display(resource.name, fallback='resource')} · "
            f"{_safe_display(resource.kind.value, fallback='kind')}",
            file=output_stream,
        )
    if summary.resource_count > len(resources):
        print(
            f"  +{summary.resource_count - len(resources)} more resources",
            file=output_stream,
        )
    print(
        "Totals  "
        f"{_count_label(summary.resource_count, 'resource', 'resources')} · "
        f"{_count_label(summary.relationship_count, 'relationship', 'relationships')}",
        file=output_stream,
    )


def _write_settings(agent: Agent, output_stream: TextIO) -> None:
    print("Settings", file=output_stream)
    print(
        f"  Agent      {_safe_display(agent.name, fallback='agent')}",
        file=output_stream,
    )
    _write_model_configuration(agent, output_stream, indent="  ")


def _write_model_configuration(
    agent: Agent,
    output_stream: TextIO,
    *,
    indent: str = "",
) -> None:
    route = agent.model_route
    if route is None:
        print(f"{indent}Model      (not configured)", file=output_stream)
        return
    candidate = route.candidates[0]
    provider, _, model = candidate.provider_id.partition(":")
    label = dict(_PROVIDERS).get(provider, provider)
    print(
        f"{indent}Provider   {_safe_display(label, fallback='provider')}",
        file=output_stream,
    )
    print(
        f"{indent}Model      {_safe_display(model, fallback='model')}",
        file=output_stream,
    )
    print(
        f"{indent}Endpoint   "
        f"{'configured' if candidate.base_url is not None else 'provider default'}",
        file=output_stream,
    )
    print(
        f"{indent}Credential "
        f"{'configured' if candidate.secret_reference is not None else 'not required'}",
        file=output_stream,
    )


def _write_chat_help(output_stream: TextIO) -> None:
    print("Commands", file=output_stream)
    for line in (
        "/model",
        "/sources",
        "/source",
        "/source use <name>",
        "/source add",
        "/source refresh <id>",
        "/source detach <source>",
        '@"source name" <question>',
        "/catalog",
        "/settings",
        "/new",
        "/resume <id>",
        "/conversation clear",
        "/learn <material>",
        "/review [cost-usd]",
        "/memory [list|show|edit|accept|reject <id>|clear-rejected]",
        "/user [edit]",
        "/skills [show|edit|delete <name>]",
        "/skills create [name]",
        "/skills use <name> [request]",
        "/<skill-name> [request]",
        "/status",
        "/conversation",
        "/agent delete",
        "/help",
        "/exit",
    ):
        print(f"  {line}", file=output_stream)


def _write_resume_hint(
    conversation_id: str | None,
    output_stream: TextIO,
) -> None:
    if conversation_id is None:
        return
    safe_id = _safe_display(conversation_id, fallback="conversation")
    print(
        f"Resume conversation {safe_id} with /resume {safe_id} on the next launch.",
        file=output_stream,
    )


async def _prompt_for_exact_approval(
    request: ApprovalRequest,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
) -> ApprovalDecision:
    print("Approval required", file=output_stream)
    print(file=output_stream)
    print(
        f"Tool:       {_safe_display(request.tool_name, fallback='tool')}",
        file=output_stream,
    )
    print(
        "Capability: " f"{_safe_display(request.capability_id, fallback='capability')}",
        file=output_stream,
    )
    print("Arguments:", file=output_stream)
    rendered = _render_approval_arguments(request)
    if rendered is None:
        print(
            "Approval denied: exact arguments exceed the terminal review bound.",
            file=output_stream,
        )
        return ApprovalDecision.DENY
    print(rendered, file=output_stream)
    try:
        answer = _read_line(
            "Approve this exact change once? [y/N] ",
            input_stream,
            output_stream,
        )
    except EOFError:
        print(file=output_stream)
        return ApprovalDecision.DENY
    except KeyboardInterrupt as error:
        raise asyncio.CancelledError from error
    if answer.strip().lower() == "y":
        return ApprovalDecision.APPROVE
    return ApprovalDecision.DENY


def _parse_candidate_review_cost_limit(value: str) -> Decimal:
    try:
        limit = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(
            "candidate review cost ceiling must be a finite non-negative decimal"
        ) from None
    _validate_candidate_review_cost_limit(limit)
    return limit


async def _review_learning_candidates_from_terminal(
    agent: Agent,
    *,
    requested_cost_limit: str | None,
    input_stream: TextIO,
    output_stream: TextIO,
) -> None:
    if requested_cost_limit is None:
        result = await agent.review_learning_candidates()
        if result.status not in {
            LearningReviewStatus.DISABLED,
            LearningReviewStatus.COST_LIMIT_REQUIRED,
        }:
            _write_learning_review_result(result, output_stream)
            return

        print("Candidate review needs one-time authorization.", file=output_stream)
        print(
            "It can make one model call and only adds suggestions to your "
            "review inbox; memory and skills do not change until you accept one.",
            file=output_stream,
        )
        print(
            "The limit is checked against the model's reported estimate after "
            "the call; provider charges can still apply.",
            file=output_stream,
        )
        while True:
            try:
                answer = _read_line(
                    "Maximum accepted estimated cost in USD "
                    f"[{_DEFAULT_CANDIDATE_REVIEW_COST_LIMIT_USD}] "
                    "(or /cancel): ",
                    input_stream,
                    output_stream,
                ).strip()
            except EOFError:
                print(file=output_stream)
                print("Learning review cancelled.", file=output_stream)
                return
            if answer.lower() in {"/cancel", "cancel", "n", "no", "q", "quit"}:
                print("Learning review cancelled.", file=output_stream)
                return
            try:
                cost_limit = _parse_candidate_review_cost_limit(
                    answer or str(_DEFAULT_CANDIDATE_REVIEW_COST_LIMIT_USD)
                )
            except ValueError:
                print(
                    "Enter a finite non-negative USD amount, or /cancel.",
                    file=output_stream,
                )
                continue
            break
    else:
        cost_limit = _parse_candidate_review_cost_limit(requested_cost_limit)

    result = await agent.review_learning_candidates(
        max_estimated_cost_usd=cost_limit,
    )
    _write_learning_review_result(result, output_stream)
    if result.status is LearningReviewStatus.DISABLED:
        print(
            "Review is unavailable for this agent's current model configuration.",
            file=output_stream,
        )


async def _handle_knowledge_command(
    parts: list[str],
    *,
    agent: Agent,
    input_stream: TextIO,
    output_stream: TextIO,
) -> bool:
    name = parts[0] if parts else ""
    if name == "/review":
        if len(parts) not in {1, 2}:
            print("Usage: /review [cost-usd]", file=output_stream)
            return True
        await _review_learning_candidates_from_terminal(
            agent,
            requested_cost_limit=parts[1] if len(parts) == 2 else None,
            input_stream=input_stream,
            output_stream=output_stream,
        )
        return True
    if name in {"/memory", "/user"}:
        target = "memory" if name == "/memory" else "user"
        if len(parts) == 1:
            content = (
                await agent.read_memory()
                if target == "memory"
                else await agent.read_user_profile()
            )
            if target == "memory":
                await _write_memory_surface(agent, content, output_stream)
            else:
                print("User:", file=output_stream)
                print(
                    _safe_display(
                        content,
                        fallback="(empty)",
                        maximum=_MAX_DISPLAY_CHARACTERS,
                    ),
                    file=output_stream,
                )
        elif target == "memory" and parts[1:] == ["list"]:
            _write_learning_candidate_list(
                await agent.list_learning_candidates(),
                output_stream,
            )
        elif target == "memory" and len(parts) == 3 and parts[1] == "show":
            candidate = await agent.read_learning_candidate(parts[2])
            if candidate is not None:
                _write_learning_candidate_view(candidate, output_stream)
            else:
                view = await agent.read_semantic_annotation(parts[2])
                if view is None:
                    raise ValueError(f"memory record not found: {parts[2]}")
                _write_semantic_view(view, output_stream)
        elif target == "memory" and len(parts) == 3 and parts[1] == "edit":
            await _edit_learning_candidate(agent, parts[2])
            print(f"Learning candidate {parts[2]!r} updated.", file=output_stream)
        elif target == "memory" and len(parts) == 3 and parts[1] == "accept":
            result = await agent.accept_learning_candidate(parts[2])
            print("Candidate acceptance run:", file=output_stream)
            print(
                _render_model_answer(
                    result.final_text,
                    fallback=f"{result.kind.value}: {result.reason}",
                ),
                file=output_stream,
            )
        elif target == "memory" and len(parts) in {3, 4} and parts[1] == "reject":
            reason = (
                LearningCandidateRejectionReason.USER_DECLINED
                if len(parts) == 3
                else LearningCandidateRejectionReason(parts[3])
            )
            rejected = await agent.reject_learning_candidate(parts[2], reason)
            print(
                f"Learning candidate {rejected.candidate.id!r} rejected.",
                file=output_stream,
            )
        elif target == "memory" and parts[1:] == ["clear-rejected"]:
            cleared = await agent.clear_rejected_learning_candidates()
            print(
                f"Cleared {_count_label(cleared, 'rejected candidate', 'rejected candidates')}.",
                file=output_stream,
            )
        elif target == "memory" and len(parts) == 3 and parts[1] == "delete":
            view = await agent.read_semantic_annotation(parts[2])
            if view is None:
                raise ValueError(f"semantic annotation not found: {parts[2]}")
            answer = _read_line(
                f"Delete semantic annotation {parts[2]!r}? [y/N] ",
                input_stream,
                output_stream,
            )
            if answer.strip().lower() != "y":
                print("Deletion cancelled.", file=output_stream)
                return True
            await agent.delete_semantic_annotation(
                parts[2],
                expected_sha256=view.sha256,
            )
            print(f"Semantic annotation {parts[2]!r} deleted.", file=output_stream)
        elif len(parts) == 2 and parts[1] == "edit":
            current = (
                await agent.read_memory()
                if target == "memory"
                else await agent.read_user_profile()
            )
            edited = _edit_document(current, agent_home=agent.home)
            if target == "memory":
                await agent.set_memory(edited)
            else:
                await agent.set_user_profile(edited)
            print(f"{target.capitalize()} updated.", file=output_stream)
        else:
            usage = (
                "/memory [list|show <id>|edit [id]|accept <id>|"
                "reject <id> [reason]|clear-rejected|delete <semantic-id>]"
                if target == "memory"
                else "/user [edit]"
            )
            print(f"Usage: {usage}", file=output_stream)
        return True
    if name != "/skills":
        return False
    if len(parts) == 1:
        skills = await agent.list_skills()
        print("Skills:", file=output_stream)
        if not skills:
            print("  (none)", file=output_stream)
        for summary in skills[:50]:
            print(
                "  "
                f"/{_safe_display(summary.name, fallback='skill')}: "
                f"{_safe_display(summary.description, fallback='description', maximum=512)}",
                file=output_stream,
            )
        if len(skills) > 50:
            print(f"  +{len(skills) - 50} more", file=output_stream)
        return True
    if len(parts) == 2 and parts[1] == "create":
        await _create_skill_wizard(
            agent,
            input_stream=input_stream,
            output_stream=output_stream,
        )
        return True
    if len(parts) == 3 and parts[1] == "create":
        await _create_skill(
            agent,
            parts[2],
            input_stream=input_stream,
            output_stream=output_stream,
        )
        return True
    if len(parts) == 3 and parts[1] == "show":
        selected_skill = await agent.read_skill(parts[2])
        if selected_skill is None:
            raise ValueError(f"skill not found: {parts[2]}")
        print(
            f"Skill: {_safe_display(selected_skill.name, fallback='skill')}",
            file=output_stream,
        )
        print(
            "Description: "
            f"{_safe_display(selected_skill.description, fallback='description', maximum=512)}",
            file=output_stream,
        )
        print("Instructions:", file=output_stream)
        print(
            _safe_display(
                selected_skill.instructions,
                fallback="(empty)",
                maximum=_MAX_DISPLAY_CHARACTERS,
            ),
            file=output_stream,
        )
        return True
    if len(parts) == 3 and parts[1] == "edit":
        selected_skill = await agent.read_skill(parts[2])
        if selected_skill is None:
            raise ValueError(f"skill not found: {parts[2]}")
        document = _render_skill_editor_document(
            selected_skill.name,
            selected_skill.description,
            selected_skill.instructions,
        )
        edited = _edit_document(document, agent_home=agent.home)
        description, instructions = _parse_skill_editor_document(
            selected_skill.name,
            edited,
        )
        changed = await agent.save_skill(
            selected_skill.name,
            description,
            instructions,
        )
        print(
            f"Skill {selected_skill.name!r} "
            f"{'updated' if changed else 'unchanged'}.",
            file=output_stream,
        )
        return True
    if len(parts) == 3 and parts[1] == "delete":
        try:
            answer = _read_line(
                f"Delete skill {parts[2]!r}? [y/N] ",
                input_stream,
                output_stream,
            )
        except EOFError:
            print(file=output_stream)
            answer = ""
        if answer.strip().lower() != "y":
            print("Deletion cancelled.", file=output_stream)
            return True
        deleted = await agent.delete_skill(parts[2])
        print(
            f"Skill {parts[2]!r} {'deleted' if deleted else 'not found'}.",
            file=output_stream,
        )
        return True
    print(
        "Usage: /skills [show <name>|create [name]|edit <name>|"
        "delete <name>|use <name> [request]]",
        file=output_stream,
    )
    return True


async def _write_memory_surface(
    agent: Agent,
    memory_text: str,
    output_stream: TextIO,
) -> None:
    print("Memory", file=output_stream)
    print(file=output_stream)
    print("Global memory:", file=output_stream)
    print(
        _safe_display(
            memory_text,
            fallback="(empty)",
            maximum=_MAX_DISPLAY_CHARACTERS,
        ),
        file=output_stream,
    )
    candidates = await agent.list_learning_candidates()
    print(file=output_stream)
    print("Pending candidates:", file=output_stream)
    if not candidates:
        print("  (none)", file=output_stream)
    else:
        counts = {
            status: sum(item.status is status for item in candidates)
            for status in LearningCandidateStatus
        }
        print(
            "  "
            + " · ".join(
                f"{status.value}={counts[status]}"
                for status in LearningCandidateStatus
                if counts[status]
            ),
            file=output_stream,
        )
        for view in candidates[:12]:
            print(
                "  "
                f"{_safe_display(view.candidate.id, fallback='candidate')} "
                f"[{view.status.value}/{view.candidate.target.value}] "
                f"{_learning_candidate_summary(view)}",
                file=output_stream,
            )
        if len(candidates) > 12:
            print(f"  +{len(candidates) - 12} more", file=output_stream)
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
        selected = tuple(
            semantic_view for semantic_view in views if semantic_view.state is state
        )
        if not selected:
            print("  (none)", file=output_stream)
            continue
        for semantic_view in selected:
            annotation = semantic_view.annotation
            detail = ""
            if semantic_view.stale_reasons:
                detail = " · " + ", ".join(semantic_view.stale_reasons)
            elif semantic_view.conflicting_ids:
                detail = " · conflicts with " + ", ".join(semantic_view.conflicting_ids)
            elif semantic_view.duplicate_of_id is not None:
                detail = " · duplicate of " + semantic_view.duplicate_of_id
            elif semantic_view.duplicate_ids:
                detail = " · duplicates collapsed: " + ", ".join(
                    semantic_view.duplicate_ids
                )
            elif semantic_view.superseded_by_id is not None:
                detail = " · superseded by " + semantic_view.superseded_by_id
            print(
                "  "
                f"{_safe_display(annotation.id, fallback='annotation')} "
                f"[{annotation.kind.value}] "
                f"{_safe_display(annotation.statement, fallback='definition', maximum=512)}"
                f"{detail}",
                file=output_stream,
            )


def _write_learning_candidate_list(
    views: tuple[LearningCandidateView, ...],
    output_stream: TextIO,
) -> None:
    print("Learning candidates", file=output_stream)
    if not views:
        print("  (none)", file=output_stream)
        return
    for view in views[:LEARNING_CANDIDATE_MAX_RECORDS]:
        print(
            "  "
            f"{_safe_display(view.candidate.id, fallback='candidate')} "
            f"[{view.status.value}/{view.candidate.target.value}] "
            f"{_learning_candidate_summary(view)}",
            file=output_stream,
        )


def _write_learning_candidate_view(
    view: LearningCandidateView,
    output_stream: TextIO,
) -> None:
    candidate = view.candidate
    print(f"Learning candidate: {_safe_display(candidate.id)}", file=output_stream)
    print(f"Status: {view.status.value}", file=output_stream)
    print(f"Target: {candidate.target.value}", file=output_stream)
    print(
        "Source scope: "
        + (
            ", ".join(_safe_display(item) for item in candidate.source_ids)
            if candidate.source_ids
            else "(global)"
        ),
        file=output_stream,
    )
    print(
        "Supporting runs: "
        + ", ".join(_safe_display(item) for item in candidate.supporting_run_ids),
        file=output_stream,
    )
    if view.obsolete_reasons:
        print(
            "Obsolete: "
            + ", ".join(
                _safe_display(item, maximum=256) for item in view.obsolete_reasons
            ),
            file=output_stream,
        )
    content = learning_candidate_content_to_mapping(candidate.content)
    print("Proposed content:", file=output_stream)
    print(
        _safe_display(
            json.dumps(content.to_dict(), indent=2, sort_keys=True),
            fallback="(invalid)",
            maximum=_MAX_DISPLAY_CHARACTERS,
        ),
        file=output_stream,
    )


def _write_learning_review_result(
    result: LearningReviewResult,
    output_stream: TextIO,
) -> None:
    print("Learning review", file=output_stream)
    print(f"  Status: {result.status.value}", file=output_stream)
    print(
        f"  Reviewed runs: {len(result.reviewed_run_ids)}",
        file=output_stream,
    )
    print(
        f"  New candidates: {len(result.candidates)}",
        file=output_stream,
    )
    print(f"  Model calls: {result.model_calls}", file=output_stream)
    if result.skipped_run_count:
        print(
            f"  Skipped unreadable runs: {result.skipped_run_count}",
            file=output_stream,
        )
    if result.candidates:
        _write_learning_candidate_list(result.candidates, output_stream)


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
    content = learning_candidate_content_from_mapping(
        view.candidate.target,
        value,
    )
    await agent.edit_learning_candidate(candidate_id, content)


def _learning_candidate_summary(view: LearningCandidateView) -> str:
    content = view.candidate.content
    values = learning_candidate_content_to_mapping(content)
    for key in ("text", "statement", "description", "name", "annotation_id"):
        value = values.get(key)
        if isinstance(value, str) and value:
            return _safe_display(value, fallback="candidate", maximum=256)
    return "(bounded proposal)"


def _write_semantic_view(
    view: SemanticAnnotationView,
    output_stream: TextIO,
) -> None:
    annotation = view.annotation
    print(f"Semantic annotation: {annotation.id}", file=output_stream)
    print(f"State: {view.state.value}", file=output_stream)
    print(f"Kind: {annotation.kind.value}", file=output_stream)
    print(
        f"Resources: {', '.join(annotation.subject.resource_ids)}",
        file=output_stream,
    )
    fields = ", ".join(
        f"{field.resource_id}.{field.field_name}" for field in annotation.subject.fields
    )
    print(f"Fields: {fields or '(resource scoped)'}", file=output_stream)
    print(
        "Verified revisions: "
        + ", ".join(
            f"{binding.resource_id}@{binding.revision}"
            for binding in annotation.catalog_revisions
        ),
        file=output_stream,
    )
    print(
        f"Confirmed: {annotation.confirmed_at.isoformat()} "
        f"by {annotation.confirmed_by}",
        file=output_stream,
    )
    print(f"Current SHA-256: {view.sha256}", file=output_stream)
    if view.stale_reasons:
        print(f"Stale reasons: {', '.join(view.stale_reasons)}", file=output_stream)
    if view.conflicting_ids:
        print(
            f"Conflicts with: {', '.join(view.conflicting_ids)}",
            file=output_stream,
        )
    if view.duplicate_ids:
        print(
            f"Exact duplicates: {', '.join(view.duplicate_ids)}",
            file=output_stream,
        )
    if view.duplicate_of_id is not None:
        print(f"Duplicate of: {view.duplicate_of_id}", file=output_stream)
    if view.superseded_by_id is not None:
        print(f"Superseded by: {view.superseded_by_id}", file=output_stream)
    print("Statement:", file=output_stream)
    print(
        _safe_display(
            annotation.statement,
            fallback="(empty)",
            maximum=_MAX_DISPLAY_CHARACTERS,
        ),
        file=output_stream,
    )
    print("Evidence:", file=output_stream)
    for evidence in annotation.evidence:
        tool = (
            f", tool call {evidence.tool_call_id}"
            if evidence.tool_call_id is not None
            else ""
        )
        note = (
            f": {_safe_display(evidence.note, fallback='note', maximum=512)}"
            if evidence.note is not None
            else ""
        )
        print(
            f"  {evidence.kind.value} in run {evidence.run_id}, "
            f"message {evidence.message_position}{tool}{note}",
            file=output_stream,
        )


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
    if command in _BUILTIN_SLASH_COMMANDS or not command.startswith("/"):
        return None
    skill_name = command[1:]
    if not skill_name:
        return None
    try:
        skill = await agent.read_skill(skill_name)
    except ValueError:
        return None
    return message if skill is not None else None


def _learning_invocation_message(message: str) -> str | None:
    parts = message.split(maxsplit=1)
    if not parts or parts[0] != "/learn":
        return None
    if len(parts) == 1 or not parts[1].strip():
        raise ValueError("usage: /learn <material>")
    material = parts[1].strip()
    return (
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


async def _create_skill(
    agent: Agent,
    name: str,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
) -> bool:
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
            print(
                "Skill creation cancelled; template was unchanged.", file=output_stream
            )
            return False
        try:
            description, instructions = _parse_skill_editor_document(name, edited)
            if description == _SKILL_DESCRIPTION_PLACEHOLDER:
                raise ValueError("replace the description placeholder")
            if instructions == _SKILL_INSTRUCTIONS_PLACEHOLDER:
                raise ValueError("replace the instructions placeholder")
            Skill(name, description, instructions)
        except ValueError as error:
            print(
                "Skill document is invalid: "
                + _safe_display(str(error), fallback="invalid document"),
                file=output_stream,
            )
            draft = edited
            try:
                answer = _read_line(
                    "Reopen editor? [Y/n] ",
                    input_stream,
                    output_stream,
                )
            except EOFError:
                print(file=output_stream)
                answer = "n"
            if answer.strip().casefold() in {"n", "no"}:
                print("Skill creation cancelled.", file=output_stream)
                return False
            continue
        if await agent.read_skill(name) is not None:
            raise ValueError(f"skill already exists: {name}")
        changed = await agent.save_skill(name, description, instructions)
        if not changed:
            raise RuntimeError("new skill was not persisted")
        print(f"Skill {name!r} created. Invoke it with /{name}.", file=output_stream)
        return True


async def _create_skill_wizard(
    agent: Agent,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
) -> bool:
    print("Create skill", file=output_stream)
    print("Enter /cancel at any prompt to stop.", file=output_stream)
    while True:
        try:
            name = _read_line(
                "Name: ",
                input_stream,
                output_stream,
            ).strip()
        except EOFError:
            print(file=output_stream)
            print("Skill creation cancelled.", file=output_stream)
            return False
        if name.casefold() == "/cancel":
            print("Skill creation cancelled.", file=output_stream)
            return False
        try:
            validate_skill_name(name)
            if await agent.read_skill(name) is not None:
                raise ValueError(f"skill already exists: {name}")
        except ValueError as error:
            print(
                "Invalid name: "
                + _safe_display(str(error), fallback="invalid skill name"),
                file=output_stream,
            )
            continue
        break

    while True:
        try:
            description = _read_line(
                "Description: ",
                input_stream,
                output_stream,
            ).strip()
        except EOFError:
            print(file=output_stream)
            print("Skill creation cancelled.", file=output_stream)
            return False
        if description.casefold() == "/cancel":
            print("Skill creation cancelled.", file=output_stream)
            return False
        try:
            Skill(name, description, _SKILL_INSTRUCTIONS_PLACEHOLDER)
        except ValueError as error:
            print(
                "Invalid description: "
                + _safe_display(str(error), fallback="invalid description"),
                file=output_stream,
            )
            continue
        break

    while True:
        print(
            "Instructions (finish with a single . on its own line):",
            file=output_stream,
        )
        instruction_lines: list[str] = []
        while True:
            try:
                line = _read_line(
                    "> ",
                    input_stream,
                    output_stream,
                )
            except EOFError:
                print(file=output_stream)
                print("Skill creation cancelled.", file=output_stream)
                return False
            if line.casefold() == "/cancel":
                print("Skill creation cancelled.", file=output_stream)
                return False
            if line == ".":
                break
            instruction_lines.append(line)
        instructions = "\n".join(instruction_lines).strip()
        try:
            Skill(name, description, instructions)
        except ValueError as error:
            print(
                "Invalid instructions: "
                + _safe_display(str(error), fallback="invalid instructions"),
                file=output_stream,
            )
            print("Re-enter the instructions body.", file=output_stream)
            continue
        break

    if await agent.read_skill(name) is not None:
        raise ValueError(f"skill already exists: {name}")
    changed = await agent.save_skill(name, description, instructions)
    if not changed:
        raise RuntimeError("new skill was not persisted")
    print(f"Skill {name!r} created. Invoke it with /{name}.", file=output_stream)
    return True


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


def _write_catalog_phase(summary: Any, output_stream: TextIO) -> None:
    resources = _count_label(summary.resource_count, "table", "tables")
    relationships = _count_label(
        summary.relationship_count,
        "relationship",
        "relationships",
    )
    if summary.is_empty:
        terminal_tui._write_setup_status(
            output_stream,
            f"! Catalog contains {resources} · {relationships}",
            role="warning",
        )
    else:
        terminal_tui._write_setup_status(
            output_stream,
            f"✓ Catalog ready: {resources} · {relationships}",
            role="success",
        )


def _write_stage_four_status(
    agent: Agent,
    sources: tuple[Any, ...],
    summary: Any,
    output_stream: TextIO,
    *,
    validated: bool,
) -> None:
    if not summary.is_empty:
        return
    route = agent.model_route
    if route is None:
        raise RuntimeError("model configuration was not loaded after replacement")
    candidate = route.candidates[0]
    provider = candidate.provider_id.partition(":")[0]
    model = candidate.provider_id.partition(":")[2]
    labels = dict(_PROVIDERS)
    label = labels.get(provider, provider)
    print(file=output_stream)
    print("Daita", file=output_stream)
    print(file=output_stream)
    print(
        f"Agent     {_safe_display(agent.name, fallback='agent')}",
        file=output_stream,
    )
    print(
        "Model     "
        f"{_safe_display(label, fallback='provider')} · "
        f"{_safe_display(model, fallback='model')} · "
        f"{'validated' if validated else 'configured'}",
        file=output_stream,
    )
    if len(sources) == 1:
        print(
            "Source    "
            f"{_safe_display(sources[0].display_name)} · no supported tables",
            file=output_stream,
        )
    else:
        source_labels = ", ".join(
            _safe_display(source.display_name) for source in sources[:3]
        )
        if len(sources) > 3:
            source_labels += f", +{len(sources) - 3} more"
        suffix = f" · {source_labels}" if source_labels else ""
        print(f"Sources   {len(sources)} active{suffix}", file=output_stream)
    print(
        "Catalog   "
        f"{_count_label(summary.resource_count, 'table', 'tables')} · "
        f"{_count_label(summary.relationship_count, 'relationship', 'relationships')}",
        file=output_stream,
    )
    if summary.latest_successful_sync_completed_at is not None:
        print(
            "Sync      " f"{summary.latest_successful_sync_completed_at.isoformat()}",
            file=output_stream,
        )
    print(file=output_stream)
    print("Not ready", file=output_stream)
    print(file=output_stream)
    print(
        "No supported tables or resources were discovered in the current "
        "active catalog.",
        file=output_stream,
    )


async def _select_catalog_repair(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    selection_input: Any = None,
    selection_output: Any = None,
) -> str:
    print(
        "Re-enter PostgreSQL with different schemas, retry a local source, "
        "or add another supported source.",
        file=output_stream,
    )
    print(file=output_stream)
    return await select_one(
        "Select a repair action",
        (
            SelectionOption("source", "Add or retry a supported source"),
            SelectionOption("exit", "Exit"),
        ),
        input_stream=input_stream,
        output_stream=output_stream,
        enhanced_input=selection_input,
        enhanced_output=selection_output,
        invalid_message="Enter 1 or 2.",
        show_title_in_fallback=False,
    )


def _count_label(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _safe_display(
    value: object,
    *,
    fallback: str = "source",
    maximum: int = 128,
) -> str:
    if not isinstance(value, str):
        return fallback
    projected = "".join(
        (
            character
            if character.isprintable()
            and unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
            else "?"
        )
        for character in value
    )
    if len(projected) > maximum:
        projected = projected[: max(0, maximum - 3)] + "..."
    return projected or fallback


def _render_approval_arguments(request: ApprovalRequest) -> str | None:
    rendered = json.dumps(
        request.arguments.to_dict(),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    if len(rendered) > _MAX_APPROVAL_DOCUMENT_CHARACTERS:
        return None
    return rendered


def _render_model_answer(
    value: object,
    *,
    fallback: str = "(empty response)",
    maximum: int | None = _MAX_DISPLAY_CHARACTERS,
) -> str:
    if not isinstance(value, str):
        return fallback
    projected = "".join(
        (
            character
            if character == "\n"
            or (
                character.isprintable()
                and unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
            )
            else "?"
        )
        for character in value
    )
    if maximum is not None and len(projected) > maximum:
        projected = projected[: max(0, maximum - 3)] + "..."
    return projected or fallback


__all__ = ["run_terminal_application"]
