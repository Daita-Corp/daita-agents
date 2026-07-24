"""Focused terminal launcher through model and source onboarding."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
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

from . import ApprovalDecision, ApprovalRequest
from .agent import (
    Agent,
    AgentAlreadyExistsError,
    AgentModelConfigurationError,
    AgentNameError,
    AgentNotFoundError,
    PostgreSQLProbeResult,
    PostgreSQLSourceError,
)
from .config import AgentConfig
from .errors import ConfigError, LLMError
from .security import KeychainStore
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
_MAX_CHAT_INPUT_CHARACTERS = 16_384
_MAX_DISPLAY_CHARACTERS = 16_384
_MAX_APPROVAL_DOCUMENT_CHARACTERS = 64 * 1_024
_MAX_CATALOG_PREVIEW = 12
_MAX_SOURCE_PREVIEW = 8


async def run_terminal_application(
    *,
    root: str | Path | None = None,
    agent_name: str | None = None,
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

    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    assert input_stream is not None
    assert output_stream is not None
    if (tui_input is None) != (tui_output is None):
        raise ValueError("TUI input and output must be supplied together")
    read_hidden = (
        (lambda prompt: getpass.getpass(prompt, stream=output_stream))
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
        _write_selected(agent.name, output_stream)
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
        _write_model_status(agent, output_stream, validated=validated)
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
            if action == "model":
                validated = True
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


def _read_line(prompt: str, input_stream: TextIO, output_stream: TextIO) -> str:
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
                print("… Discovering tables and relationships", file=output_stream)
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
                print("… Discovering tables and relationships", file=output_stream)
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


async def _onboard_postgresql(
    agent: Agent,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    hidden_input: Callable[[str], str],
    selection_input: Any = None,
    selection_output: Any = None,
) -> Any:
    name = _read_required("Display name: ", input_stream, output_stream)
    host = _read_required("Host: ", input_stream, output_stream)
    port = _read_postgresql_port(input_stream, output_stream)
    database = _read_required("Database: ", input_stream, output_stream)
    username = _read_required("Username: ", input_stream, output_stream)
    password: str | None = None
    while not password:
        password = hidden_input("Password: ")
        if not password:
            print("Password cannot be empty.", file=output_stream)
    ssl_mode = _read_ssl_mode(input_stream, output_stream)
    reference = None
    attached = False
    try:
        credential = password
        password = None
        reference = await agent.store_postgresql_password(credential)
        credential = ""
        probe = await agent.probe_postgresql(
            host=host,
            port=port,
            database=database,
            username=username,
            credential=reference,
            ssl_mode=ssl_mode,
        )
        print("✓ Connection validated", file=output_stream)
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
        print(
            "✓ Schemas selected: "
            + ", ".join(_safe_display(schema, fallback="schema") for schema in schemas),
            file=output_stream,
        )
        print("… Discovering tables and relationships", file=output_stream)
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
            message = _read_line("You › ", input_stream, output_stream)
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
        result = await _run_message(
            agent,
            message,
            conversation_id=conversation_id,
            output_stream=output_stream,
        )
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
            f"${result.usage.estimated_cost_usd}",
            file=output_stream,
        )
        print(file=output_stream)


class _TerminalCommandOutput:
    def __init__(self, output_stream: TextIO) -> None:
        self._output_stream = output_stream
        self._recording = io.StringIO()

    def write(self, value: str) -> int:
        written = self._output_stream.write(value)
        self._recording.write(value)
        return len(value) if written is None else written

    def flush(self) -> None:
        self._output_stream.flush()

    def isatty(self) -> bool:
        return self._output_stream.isatty()

    def fileno(self) -> int:
        return self._output_stream.fileno()

    @property
    def value(self) -> str:
        return self._recording.getvalue()


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
    route = agent.model_route
    if route is None:
        raise RuntimeError("ready chat requires a configured model")
    candidate = route.candidates[0]
    _provider, _, model = candidate.provider_id.partition(":")
    sources = tuple(source for source in await agent.list_sources() if source.active)
    if len(sources) == 1:
        source_summary = _safe_display(sources[0].display_name, fallback="1 source")
    else:
        source_summary = _count_label(len(sources), "source", "sources")
    state = terminal_tui.TerminalViewState(
        agent_label=_safe_display(agent.name, fallback="agent"),
        model_label=_safe_display(model, fallback="model"),
        source_summary=source_summary,
        conversation_id=conversation_id,
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

    async def handle_command(
        command: str,
        selected_conversation: str | None,
    ) -> terminal_tui.TerminalCommandResult:
        nonlocal agent
        captured = _TerminalCommandOutput(output_stream)
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
            output=captured.value,
            presentation={
                "/status": "status",
                "/sources": "sources",
                "/catalog": "catalog",
                "/settings": "settings",
            }.get(command.split(maxsplit=1)[0], "local"),
        )

    result = await terminal_tui.run_terminal_tui(
        state,
        run_message=run_message,
        load_transcript=load_transcript,
        handle_command=handle_command,
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
    run = asyncio.create_task(
        agent.run(
            message,
            conversation_id=conversation_id,
        )
    )
    try:
        return await run
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
        await _write_sources(agent, output_stream)
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
        print("Usage: /source add | /source refresh <source-id>", file=output_stream)
    elif name in {
        "/exit",
        "/help",
        "/new",
        "/sources",
        "/catalog",
        "/settings",
        "/status",
        "/conversation",
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
    route = agent.model_route
    if route is None:
        raise RuntimeError("ready chat requires a configured model")
    candidate = route.candidates[0]
    provider, _, model = candidate.provider_id.partition(":")
    label = dict(_PROVIDERS).get(provider, provider)
    sources = tuple(source for source in await agent.list_sources() if source.active)
    summary = await agent.catalog_summary()
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
            "Source    " f"{_safe_display(sources[0].display_name)} · cataloged",
            file=output_stream,
        )
    else:
        labels = ", ".join(_safe_display(source.display_name) for source in sources[:3])
        if len(sources) > 3:
            labels += f", +{len(sources) - 3} more"
        source_text = f"{len(sources)} cataloged"
        if labels:
            source_text += f" · {labels}"
        print(f"Sources   {source_text}", file=output_stream)
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
    print(
        "Conversation  " f"{_safe_display(conversation_id, fallback='new')}",
        file=output_stream,
    )
    print(file=output_stream)
    print("Ready", file=output_stream)
    print(file=output_stream)


async def _write_sources(agent: Agent, output_stream: TextIO) -> None:
    sources = tuple(source for source in await agent.list_sources() if source.active)
    summary = await agent.catalog_summary()
    print("Sources", file=output_stream)
    if not sources:
        print("  (none)", file=output_stream)
    for source in sources[:_MAX_SOURCE_PREVIEW]:
        print(
            "  "
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
        "/source add",
        "/source refresh <id>",
        "/catalog",
        "/settings",
        "/new",
        "/resume <id>",
        "/memory [edit]",
        "/user [edit]",
        "/skills [show|edit|delete <name>]",
        "/status",
        "/conversation",
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


async def _handle_knowledge_command(
    parts: list[str],
    *,
    agent: Agent,
    input_stream: TextIO,
    output_stream: TextIO,
) -> bool:
    name = parts[0] if parts else ""
    if name in {"/memory", "/user"}:
        target = "memory" if name == "/memory" else "user"
        if len(parts) == 1:
            content = (
                await agent.read_memory()
                if target == "memory"
                else await agent.read_user_profile()
            )
            print(f"{target.capitalize()}:", file=output_stream)
            print(
                _safe_display(
                    content,
                    fallback="(empty)",
                    maximum=_MAX_DISPLAY_CHARACTERS,
                ),
                file=output_stream,
            )
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
            print(f"Usage: {name} [edit]", file=output_stream)
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
                f"{_safe_display(summary.name, fallback='skill')}: "
                f"{_safe_display(summary.description, fallback='description', maximum=512)}",
                file=output_stream,
            )
        if len(skills) > 50:
            print(f"  +{len(skills) - 50} more", file=output_stream)
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
        "Usage: /skills [show <name>|edit <name>|delete <name>]",
        file=output_stream,
    )
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


def _write_selected(name: str, output_stream: TextIO) -> None:
    print(file=output_stream)
    print("Daita", file=output_stream)
    print(file=output_stream)
    print(
        f"Agent     {_safe_display(name, fallback='agent')}",
        file=output_stream,
    )


def _write_model_status(
    agent: Agent,
    output_stream: TextIO,
    *,
    validated: bool,
) -> None:
    route = agent.model_route
    if route is None:
        raise RuntimeError("model configuration was not loaded after replacement")
    candidate = route.candidates[0]
    provider = candidate.provider_id.partition(":")[0]
    model = candidate.provider_id.partition(":")[2]
    labels = dict(_PROVIDERS)
    label = labels.get(provider, provider)
    print(
        "Model     "
        f"{_safe_display(label, fallback='provider')} · "
        f"{_safe_display(model, fallback='model')} · "
        f"{'validated' if validated else 'configured'}",
        file=output_stream,
    )
    if not validated:
        print(
            "Note      provider health was not checked this launch",
            file=output_stream,
        )


def _write_catalog_phase(summary: Any, output_stream: TextIO) -> None:
    resources = _count_label(summary.resource_count, "table", "tables")
    relationships = _count_label(
        summary.relationship_count,
        "relationship",
        "relationships",
    )
    if summary.is_empty:
        print(
            f"! Catalog contains {resources} · {relationships}",
            file=output_stream,
        )
    else:
        print(f"✓ Catalog ready: {resources} · {relationships}", file=output_stream)


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
    maximum: int = _MAX_DISPLAY_CHARACTERS,
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
    if len(projected) > maximum:
        projected = projected[: max(0, maximum - 3)] + "..."
    return projected or fallback


__all__ = ["run_terminal_application"]
