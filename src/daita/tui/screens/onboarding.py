"""Agent, model, and source onboarding screens over the public Agent API."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Select, Static

from ..commands import parse_postgresql_connection_url
from ..models import (
    BUILTIN_PROVIDER_IDS,
    MODEL_SUGGESTIONS,
    POSTGRESQL_CONNECTION_URL_ERROR,
    PROVIDERS,
    SOURCE_TYPES,
    SSL_MODES,
    SUBSCRIPTION_PROVIDER_IDS,
    PickerOption,
)
from ..sanitization import sanitize_terminal_text
from .selection import SelectionScreen


class AgentCreateScreen(Screen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="onboard"):
            yield Label("Create an agent", id="onboard-title", markup=False)
            yield Input(placeholder="Agent name", id="agent-name")
            yield Label("", id="onboard-error", markup=False)
            yield Button("Create", id="create-agent", variant="primary")
            yield Footer()

    def on_mount(self) -> None:
        self.query_one("#agent-name", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "create-agent":
            return
        name = self.query_one("#agent-name", Input).value.strip()
        if not name:
            self.query_one("#onboard-error", Label).update("Enter an agent name.")
            return
        try:
            await self.app.create_named_agent(name)  # type: ignore[attr-defined]
        except Exception as error:
            self.query_one("#onboard-error", Label).update(
                sanitize_terminal_text(
                    str(error),
                    maximum=512,
                    preserve_lines=False,
                    fallback="Could not create agent.",
                )
            )
            return
        self.dismiss(name)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "agent-name":
            await self.on_button_pressed(
                Button.Pressed(self.query_one("#create-agent", Button))
            )


class ModelSetupScreen(Screen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._provider: str | None = None
        self._model: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="onboard"):
            yield Label("Configure a model", id="onboard-title", markup=False)
            yield Static(
                "Choose a provider, then a model.", id="model-help", markup=False
            )
            yield Button("Choose provider", id="choose-provider", variant="primary")
            yield Input(
                placeholder="Custom provider identifier", id="model-provider-id"
            )
            yield Input(placeholder="Model ID (optional override)", id="model-id")
            yield Input(placeholder="API key", id="model-secret", password=True)
            yield Input(
                placeholder="Base URL (custom provider only)", id="model-base-url"
            )
            yield Input(
                placeholder="Context window tokens (if required)", id="model-context"
            )
            yield Input(
                placeholder="Max output tokens (if required)", id="model-output"
            )
            yield Label("", id="onboard-error", markup=False)
            yield Button("Save model", id="save-model", variant="success")
            yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(False)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "choose-provider":
            await self._choose_provider()
            return
        if event.button.id != "save-model":
            return
        if self._provider is None:
            self.query_one("#onboard-error", Label).update("Choose a provider first.")
            return
        selected_provider = self._provider
        provider = selected_provider
        model = self.query_one("#model-id", Input).value.strip() or self._model
        if selected_provider == "custom":
            provider = self.query_one("#model-provider-id", Input).value.strip().lower()
            if not provider or provider in BUILTIN_PROVIDER_IDS:
                self.query_one("#onboard-error", Label).update(
                    "Enter a non-built-in custom provider identifier."
                )
                return
        if model is None or not model:
            self.query_one("#onboard-error", Label).update(
                "Enter or choose a model ID."
            )
            return
        secret_input = self.query_one("#model-secret", Input)
        api_key = secret_input.value or None
        subscription_credential: str | None = None
        try:
            context = _optional_int(self.query_one("#model-context", Input).value)
            output = _optional_int(self.query_one("#model-output", Input).value)
            if self.app.controller.model_requires_explicit_limits(  # type: ignore[attr-defined]
                provider=provider,
                model=model,
            ) and (
                context is None or output is None
            ):
                raise ValueError(
                    "This model requires explicit context-window and output-token limits."
                )
            base_url = self.query_one("#model-base-url", Input).value.strip() or None
            if selected_provider == "custom" and base_url is None:
                raise ValueError("A custom provider requires a base URL.")
            if provider == "codex":
                api_key = None
                subscription_credential = await self.app.controller.authenticate_model_subscription(  # type: ignore[attr-defined]
                    provider=provider,
                    on_verification=self._show_subscription_verification,
                    on_progress=self._show_subscription_progress,
                )
            elif provider in SUBSCRIPTION_PROVIDER_IDS or provider == "ollama":
                api_key = None
            else:
                if api_key is None:
                    raise ValueError("An API key is required for this provider.")
            await self.app.controller.configure_model(  # type: ignore[attr-defined]
                provider=provider,
                model=model,
                api_key=api_key,
                subscription_credential=subscription_credential,
                base_url=base_url,
                context_window_tokens=context,
                max_output_tokens=output,
            )
        except Exception as error:
            self.query_one("#onboard-error", Label).update(
                sanitize_terminal_text(
                    str(error),
                    maximum=512,
                    preserve_lines=False,
                    fallback="Model configuration failed.",
                )
            )
            return
        finally:
            api_key = None
            subscription_credential = None
            secret_input.clear()
        self.dismiss(True)

    def _show_subscription_verification(self, prompt: object) -> None:
        verification_url = getattr(prompt, "verification_url", "")
        user_code = getattr(prompt, "user_code", "")
        self.query_one("#model-help", Static).update(
            sanitize_terminal_text(
                f"Open {verification_url}\nEnter code: {user_code}",
                maximum=512,
                preserve_lines=True,
                fallback="Complete ChatGPT authorization in your browser.",
            )
        )

    def _show_subscription_progress(self, message: str) -> None:
        self.query_one("#model-help", Static).update(
            sanitize_terminal_text(
                message,
                maximum=240,
                preserve_lines=False,
                fallback="Connecting subscription.",
            )
        )

    async def _choose_provider(self) -> None:
        options = tuple(PickerOption(provider, label) for provider, label in PROVIDERS)
        selected = await self.app._await_modal(  # type: ignore[attr-defined]
            SelectionScreen(title="Select a model provider", options=options)
        )
        if selected is None:
            return
        provider = selected[0]
        if not isinstance(provider, str):
            return
        self._provider = provider
        suggestions = MODEL_SUGGESTIONS.get(provider, ())
        model_options = tuple(
            PickerOption(
                item.model_id,
                item.label,
                (
                    item.description
                    if item.recommendation is None
                    else f"{item.recommendation} · {item.description}"
                ),
            )
            for item in suggestions
        )
        if self._provider == "custom":
            self._model = None
            self.query_one("#model-help", Static).update(
                "Enter the custom provider identifier, model ID, base URL, and API key."
            )
            return
        if model_options:
            chosen = await self.app._await_modal(  # type: ignore[attr-defined]
                SelectionScreen(title="Select a model", options=model_options)
            )
            if chosen is None:
                return
            model = chosen[0]
            if not isinstance(model, str):
                return
            self._model = model
            self.query_one("#model-id", Input).value = model
        self.query_one("#model-help", Static).update(
            sanitize_terminal_text(
                f"{self._provider}:{self._model}",
                maximum=240,
                preserve_lines=False,
                fallback="model",
            )
        )


class SourceSetupScreen(Screen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="onboard"):
            yield Label("Attach a source", id="onboard-title", markup=False)
            yield Select(
                ((label, key) for key, label in SOURCE_TYPES),
                prompt="Source type",
                id="source-type",
            )
            yield Input(placeholder="Display name", id="source-name")
            yield Input(placeholder="Path or PostgreSQL URL", id="source-path")
            yield Input(placeholder="Host", id="pg-host")
            yield Input(placeholder="Port", id="pg-port", value="5432")
            yield Input(placeholder="Database", id="pg-database")
            yield Input(placeholder="Username", id="pg-username")
            yield Input(placeholder="Password", id="pg-password", password=True)
            yield Input(
                placeholder="Schemas (comma-separated)", id="pg-schemas", value="public"
            )
            yield Select(
                ((mode, mode) for mode in sorted(SSL_MODES)),
                prompt="SSL mode",
                value="require",
                id="pg-ssl",
            )
            yield Label("", id="onboard-error", markup=False)
            yield Button("Attach", id="attach-source", variant="primary")
            yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(False)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "attach-source":
            return
        kind = self.query_one("#source-type", Select).value
        name = self.query_one("#source-name", Input).value.strip() or None
        path_value = self.query_one("#source-path", Input).value.strip()
        try:
            if kind == "sqlite":
                if not path_value:
                    raise ValueError("SQLite requires a path")
                await self.app.controller.attach_sqlite(  # type: ignore[attr-defined]
                    Path(path_value).expanduser(),
                    name=name,
                )
            elif kind == "directory":
                if not path_value:
                    raise ValueError("A local directory requires a path")
                await self.app.controller.attach_directory(  # type: ignore[attr-defined]
                    Path(path_value).expanduser(),
                    name=name,
                )
            elif kind == "postgresql":
                await self._attach_postgresql(name, path_value)
            else:
                raise ValueError("Choose a source type")
        except Exception as error:
            self.query_one("#onboard-error", Label).update(
                sanitize_terminal_text(
                    str(error),
                    maximum=512,
                    preserve_lines=False,
                    fallback="Source setup failed.",
                )
            )
            return
        self.dismiss(True)

    async def _attach_postgresql(self, name: str | None, path_value: str) -> None:
        controller = self.app.controller  # type: ignore[attr-defined]
        if path_value:
            try:
                host, port, database, username, password, ssl_mode = (
                    parse_postgresql_connection_url(path_value)
                )
            except ValueError as error:
                raise ValueError(POSTGRESQL_CONNECTION_URL_ERROR) from error
        else:
            host = self.query_one("#pg-host", Input).value.strip()
            database = self.query_one("#pg-database", Input).value.strip()
            username = self.query_one("#pg-username", Input).value.strip()
            password = self.query_one("#pg-password", Input).value
            ssl_mode = str(self.query_one("#pg-ssl", Select).value or "require")
            try:
                port = int(self.query_one("#pg-port", Input).value or "5432")
            except ValueError as error:
                raise ValueError("Port must be an integer") from error
        if not all((host, database, username, password)):
            raise ValueError(
                "PostgreSQL requires host, database, username, and password"
            )
        credential = await controller.store_postgresql_password(password)
        schemas = tuple(
            item.strip()
            for item in self.query_one("#pg-schemas", Input).value.split(",")
            if item.strip()
        ) or ("public",)
        await controller.attach_postgresql(
            host=host,
            database=database,
            username=username,
            credential=credential,
            schemas=schemas,
            port=port,
            ssl_mode=ssl_mode,
            name=name,
        )


class CatalogRepairScreen(Screen[str]):
    BINDINGS = [Binding("escape", "exit", "Exit")]

    def compose(self) -> ComposeResult:
        with Vertical(id="onboard"):
            yield Label("Catalog is empty", id="onboard-title", markup=False)
            yield Static(
                "Attach a source that Daita can catalog, or exit.",
                markup=False,
            )
            yield Button("Add a source", id="repair-add", variant="primary")
            yield Button("Exit", id="repair-exit")
            yield Footer()

    def action_exit(self) -> None:
        self.dismiss("exit")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "repair-add":
            attached = await self.app._await_modal(SourceSetupScreen())  # type: ignore[attr-defined]
            self.dismiss("added" if attached else "exit")
            return
        self.dismiss("exit")


def _optional_int(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    return int(text)
