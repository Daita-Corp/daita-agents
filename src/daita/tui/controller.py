"""Presentation adapter over the public Agent API."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from daita import (
    Agent,
    ApprovalHandler,
    ConversationRun,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    LearningReviewStatus,
    LoopExit,
    Transcript,
)
from daita.agent import (
    AgentAlreadyExistsError,
    AgentModelConfigurationError,
    AgentNameError,
    SourceRefreshError,
    SourceSelectionError,
)
from daita.observation import AgentObserver
from daita.security import (
    CredentialSession,
    KeychainSecretProvider,
    KeychainStore,
    SecretReference,
)
from daita.skills import Skill, validate_skill_name
from daita.learning_candidates import (
    learning_candidate_content_from_mapping,
    learning_candidate_content_to_mapping,
)

from .commands import (
    BUILTIN_SLASH_COMMANDS,
    HELP_TEXT,
    learning_invocation_message,
    parse_source_override,
)
from .models import (
    PROVIDERS,
    SOURCE_TYPE_LABELS,
    CommandOutcome,
    UserInputError,
    parse_candidate_review_cost_limit,
)
from .projection import artifact_delivery_messages, completed_tool_pairs
from .sanitization import MAX_DISPLAY_CHARACTERS, render_model_answer, safe_display

VALIDATION_ERRORS = {
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
SUBSCRIPTION_VALIDATION_ERRORS = {
    "authentication_error": (
        "The {client} subscription login was rejected. Sign in again and retry."
    ),
    "configuration_error": (
        "The official {client} client could not start: it may be missing, "
        "incompatible, or out of date. Install or update the client, run "
        "{login_command}, and retry."
    ),
    "local_access_error": (
        "The official {client} client cannot access its local login/state files. "
        "Exit Daita and launch it from macOS Terminal or iTerm, not from a "
        "sandboxed terminal inside another application, then retry."
    ),
    "rate_limit_error": "The {client} subscription allowance is currently exhausted.",
    "model_not_found": "This {client} subscription cannot access {model}.",
    "provider_unavailable": "The signed-in {client} client could not complete validation.",
    "timeout": "The signed-in {client} client did not respond before the timeout.",
    "invalid_request": "The {client} client rejected this model configuration.",
    "output_limit": (
        "The model exhausted its validation output budget before proposing the tool."
    ),
}
SUBSCRIPTION_CLIENTS = {
    "codex": ("ChatGPT", "sign in through Daita"),
    "claude-code": ("Claude Code", "claude auth login"),
    "grok-build": ("Grok Build", "grok login"),
}
MODEL_SETUP_ERRORS = {
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
POSTGRESQL_ERRORS = {
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
SOURCE_REFRESH_ERRORS = {
    **POSTGRESQL_ERRORS,
    "postgresql_discovery_failed": (
        "PostgreSQL was reached, but its current catalog could not be read. "
        "Check the saved role's schema and catalog permissions, then retry."
    ),
    "postgresql_metadata_invalid": (
        "PostgreSQL returned catalog metadata this Daita release cannot admit. "
        "Check server compatibility, then retry."
    ),
    "sqlite_open_failed": (
        "The saved SQLite source could not be opened read-only. "
        "Check that it is a valid SQLite database, then retry."
    ),
    "sqlite_path_invalid": (
        "The saved SQLite file is unavailable or its path is no longer safe. "
        "Check that the source is available at its original path, then retry."
    ),
    "local_root_invalid": (
        "The saved local source directory is unavailable or its path is no longer "
        "safe. Check that the source is available at its original path, then retry."
    ),
    "local_discovery_failed": (
        "The saved local source could not be cataloged. "
        "Check its read permissions and supported files, then retry."
    ),
}


class PresentationController:
    """Thin adapter from UI actions to public Agent methods."""

    def __init__(
        self,
        *,
        root: str | Path | None,
        keychain: KeychainStore | None = None,
        model_validator: Any = None,
        reviewer_max_estimated_cost_usd: Decimal | None = None,
    ) -> None:
        self.root = root
        if isinstance(keychain, CredentialSession):
            self.keychain = keychain
            self._owns_credential_session = False
        else:
            self.keychain = CredentialSession(keychain or KeychainSecretProvider())
            self._owns_credential_session = True
        self.model_validator = model_validator
        self.reviewer_max_estimated_cost_usd = reviewer_max_estimated_cost_usd
        self.model: Any = None
        self.model_profile: Any = None
        self.agent: Agent | None = None
        self.conversation_id: str | None = None
        self.validated_model = False

    def require_agent(self) -> Agent:
        if self.agent is None:
            raise RuntimeError("no agent is open")
        return self.agent

    async def list_agents(self) -> tuple[str, ...]:
        return await Agent.list(root=self.root)

    async def close_agent(self) -> None:
        agent = self.agent
        self.agent = None
        if agent is not None:
            await agent.close()

    async def close(self) -> None:
        await self.close_agent()
        if self._owns_credential_session:
            await self.keychain.close()

    async def open_agent(
        self,
        name: str,
        *,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> Agent:
        await self.close_agent()
        opened = await Agent.open(
            name,
            root=self.root,
            model=self.model,
            model_profile=self.model_profile,
            keychain=self.keychain,
            model_validator=self.model_validator,
            reviewer_max_estimated_cost_usd=self.reviewer_max_estimated_cost_usd,
            observer=observer,
            approval_handler=approval_handler,
        )
        try:
            await self._preload_active_credentials(opened)
        except BaseException:
            await opened.close()
            raise
        self.agent = opened
        return opened

    async def _preload_active_credentials(self, agent: Agent) -> None:
        """Move native secret authorization out of the query hot path."""

        references: list[SecretReference] = []
        route = agent.model_route
        if route is not None:
            references.extend(
                reference
                for candidate in route.candidates
                if (reference := candidate.secret_reference) is not None
                and reference.scheme == "keychain"
            )
        for source in await agent.list_sources():
            if not source.active:
                continue
            raw_reference = source.configuration.get("credential_ref")
            if not isinstance(raw_reference, str):
                continue
            reference = SecretReference.parse(raw_reference)
            if reference.scheme == "keychain":
                references.append(reference)
        await self.keychain.preload(references)

    async def create_agent(
        self,
        name: str,
        *,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> Agent:
        await self.close_agent()
        try:
            created = await Agent.create(
                name,
                root=self.root,
                model=self.model,
                model_profile=self.model_profile,
                keychain=self.keychain,
                model_validator=self.model_validator,
                reviewer_max_estimated_cost_usd=self.reviewer_max_estimated_cost_usd,
                observer=observer,
                approval_handler=approval_handler,
            )
        except (AgentAlreadyExistsError, AgentNameError, ValueError) as error:
            raise UserInputError(str(error)) from error
        self.agent = created
        return created

    async def delete_open_agent(self) -> None:
        agent = self.require_agent()
        name = agent.name
        await self.close_agent()
        await Agent.delete(name, root=self.root, keychain=self.keychain)

    async def reopen_agent(
        self,
        *,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> Agent:
        agent = self.require_agent()
        name = agent.name
        return await self.open_agent(
            name,
            observer=observer,
            approval_handler=approval_handler,
        )

    def model_label(self) -> str:
        agent = self.require_agent()
        route = agent.model_route
        if route is None:
            return "model not configured"
        candidate = route.candidates[0]
        _provider, _sep, model = candidate.provider_id.partition(":")
        return model or candidate.provider_id

    def provider_label(self) -> str:
        agent = self.require_agent()
        route = agent.model_route
        if route is None:
            return "not configured"
        candidate = route.candidates[0]
        provider, _sep, _model = candidate.provider_id.partition(":")
        return dict(PROVIDERS).get(provider, provider)

    async def source_summary(self) -> str:
        agent = self.require_agent()
        active = await agent.active_source(conversation_id=self.conversation_id)
        if active is not None:
            return safe_display(active.display_name, fallback="source")
        sources = tuple(
            source for source in await agent.list_sources() if source.active
        )
        if not sources:
            return "no sources"
        if len(sources) == 1:
            return safe_display(sources[0].display_name, fallback="source")
        return f"{len(sources)} sources"

    async def configure_model(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        subscription_credential: str | None = None,
        base_url: str | None = None,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        agent = self.require_agent()
        try:
            await agent.configure_model(
                provider=provider,
                model=model,
                api_key=api_key,
                subscription_credential=subscription_credential,
                base_url=base_url,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
            )
        except AgentModelConfigurationError as error:
            raise UserInputError(
                self._model_error_text(error, provider=provider, model=model)
            ) from error
        self.validated_model = True

    def model_requires_explicit_limits(self, *, provider: str, model: str) -> bool:
        return self.require_agent().model_requires_explicit_limits(
            provider=provider,
            model=model,
        )

    async def authenticate_model_subscription(
        self,
        *,
        provider: str,
        on_verification: Callable[[Any], None],
        on_progress: Callable[[str], None] | None = None,
    ) -> str:
        try:
            return await self.require_agent().authenticate_model_subscription(
                provider=provider,
                on_verification=on_verification,
                on_progress=on_progress,
            )
        except Exception as error:
            code = getattr(error, "error_code", "") or ""
            if code == "timeout":
                message = "ChatGPT authorization timed out. Start again to retry."
            elif code == "provider_unavailable":
                message = "Daita could not reach ChatGPT login. Check your connection."
            else:
                message = "ChatGPT authorization failed. Start again to retry."
            raise UserInputError(message) from error

    def _model_error_text(
        self,
        error: AgentModelConfigurationError,
        *,
        provider: str,
        model: str,
    ) -> str:
        code = getattr(error, "code", "") or ""
        if provider in SUBSCRIPTION_CLIENTS and code in SUBSCRIPTION_VALIDATION_ERRORS:
            client, login_command = SUBSCRIPTION_CLIENTS[provider]
            return SUBSCRIPTION_VALIDATION_ERRORS[code].format(
                client=client,
                login_command=login_command,
                model=model,
            )
        if code in VALIDATION_ERRORS:
            return VALIDATION_ERRORS[code].format(model=model)
        if code in MODEL_SETUP_ERRORS:
            return MODEL_SETUP_ERRORS[code]
        return safe_display(
            str(error), fallback="Model configuration failed.", maximum=512
        )

    async def attach_sqlite(self, path: Path, *, name: str | None) -> Any:
        return await self.require_agent().attach_sqlite(path, name=name)

    async def attach_directory(self, path: Path, *, name: str | None) -> Any:
        return await self.require_agent().attach_local_directory(path, name=name)

    async def store_postgresql_password(self, password: str) -> SecretReference:
        return await self.require_agent().store_postgresql_password(password)

    async def probe_postgresql(self, **kwargs: Any) -> Any:
        return await self.require_agent().probe_postgresql(**kwargs)

    async def attach_postgresql(self, **kwargs: Any) -> Any:
        return await self.require_agent().attach_postgresql(**kwargs)

    async def active_source(self) -> Any:
        return await self.require_agent().active_source(
            conversation_id=self.conversation_id
        )

    def source_edit_defaults(self, source: Any) -> dict[str, Any]:
        """Return safe editable connection fields from one public registration."""

        configuration = source.configuration
        defaults: dict[str, Any] = {
            "source_id": source.id,
            "adapter_id": source.adapter_id,
            "name": source.display_name,
        }
        if source.adapter_id == "sqlite":
            defaults["path"] = _configuration_text(configuration, "path")
        elif source.adapter_id == "local-directory":
            defaults["path"] = _configuration_text(configuration, "root")
        elif source.adapter_id == "postgresql":
            defaults.update(
                host=_configuration_text(configuration, "host"),
                port=_configuration_port(configuration),
                database=_configuration_text(configuration, "database"),
                username=_configuration_text(configuration, "username"),
                schemas=_configuration_schemas(configuration),
                ssl_mode=_configuration_text(configuration, "ssl_mode"),
            )
        else:
            raise UserInputError(
                f"Source type {source.adapter_id!r} cannot be edited here."
            )
        return defaults

    async def edit_source_connection(
        self,
        source: Any,
        *,
        name: str,
        path: Path | None = None,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        schemas: tuple[str, ...] = (),
        ssl_mode: str | None = None,
        confirmation_handler: Callable[[Any], Any],
    ) -> Any:
        """Validate, review, and atomically replace one source connection."""

        agent = self.require_agent()
        if source.adapter_id == "sqlite":
            if path is None:
                raise UserInputError("SQLite requires an absolute file path.")
            return await agent.edit_sqlite_source(
                source.id,
                path,
                name=name,
                confirmation_handler=confirmation_handler,
            )
        if source.adapter_id == "local-directory":
            if path is None:
                raise UserInputError(
                    "A local source requires an absolute directory path."
                )
            configuration = source.configuration
            return await agent.edit_local_directory_source(
                source.id,
                path,
                name=name,
                confirmation_handler=confirmation_handler,
                max_depth=_configuration_integer(configuration, "max_depth"),
                max_files=_configuration_integer(configuration, "max_files"),
                max_file_bytes=_configuration_integer(configuration, "max_file_bytes"),
                max_columns=_configuration_integer(configuration, "max_columns"),
                max_rows=_configuration_integer(configuration, "max_rows"),
                max_json_nodes=_configuration_integer(configuration, "max_json_nodes"),
                max_json_depth=_configuration_integer(configuration, "max_json_depth"),
                max_key_bytes=_configuration_integer(configuration, "max_key_bytes"),
                max_string_bytes=_configuration_integer(
                    configuration, "max_string_bytes"
                ),
                max_cell_bytes=_configuration_integer(configuration, "max_cell_bytes"),
            )
        if source.adapter_id != "postgresql":
            raise UserInputError(
                f"Source type {source.adapter_id!r} cannot be edited here."
            )
        if (
            not host
            or not database
            or not username
            or not schemas
            or not ssl_mode
            or port is None
        ):
            raise UserInputError("Complete all PostgreSQL connection fields.")
        reference_text = source.configuration.get("credential_ref")
        credential = (
            SecretReference.parse(reference_text)
            if isinstance(reference_text, str)
            else None
        )
        created_credential: SecretReference | None = None
        result: Any = None
        try:
            if password:
                created_credential = await agent.store_postgresql_password(password)
                credential = created_credential
            if credential is None:
                raise UserInputError(
                    "This PostgreSQL source has no saved password; enter a new password."
                )
            result = await agent.edit_postgresql_source(
                source.id,
                host=host,
                port=port,
                database=database,
                username=username,
                credential=credential,
                schemas=schemas,
                ssl_mode=ssl_mode,
                name=name,
                confirmation_handler=confirmation_handler,
            )
            return result
        finally:
            password = None
            if created_credential is not None and result is None:
                await agent.delete_postgresql_password(created_credential)

    async def catalog_summary(self) -> Any:
        return await self.require_agent().catalog_summary()

    async def list_catalog_resources(
        self, *, source_id: str | None = None
    ) -> tuple[Any, ...]:
        return await self.require_agent().list_catalog_resources(source_id=source_id)

    async def skill_completions(self) -> tuple[tuple[str, str], ...]:
        summaries = await self.require_agent().list_skills()
        return tuple(
            (
                summary.name,
                safe_display(
                    summary.description,
                    fallback="Reusable skill",
                    maximum=240,
                ),
            )
            for summary in summaries
        )

    async def list_sources(self) -> tuple[Any, ...]:
        return await self.require_agent().list_sources()

    async def select_source(self, selector: str) -> Any:
        try:
            return await self.require_agent().select_source(selector)
        except SourceSelectionError as error:
            raise UserInputError(str(error)) from error

    async def resolve_source(self, selector: str) -> Any:
        try:
            return await self.require_agent().resolve_source(selector)
        except SourceSelectionError as error:
            raise UserInputError(str(error)) from error

    async def refresh_source(self, source_id: str) -> Any:
        try:
            return await self.require_agent().refresh_source(source_id)
        except SourceRefreshError as error:
            guidance = SOURCE_REFRESH_ERRORS.get(
                error.code,
                "The saved source could not be refreshed. Check its availability "
                "and read permissions, then retry.",
            )
            raise UserInputError(
                guidance + " The existing catalog is still available."
            ) from error

    async def detach_source(self, source_id: str) -> Any:
        return await self.require_agent().detach(source_id)

    async def inspect_source_permissions(self, source_id: str) -> Any:
        return await self.require_agent().inspect_source_permissions(source_id)

    async def preview_source_permissions(self, **kwargs: Any) -> Any:
        return await self.require_agent().preview_source_permissions(**kwargs)

    async def apply_source_permissions(self, **kwargs: Any) -> Any:
        return await self.require_agent().apply_source_permissions(**kwargs)

    async def prepare_message(self, message: str) -> tuple[str, str | None, bool]:
        """Return (run text, one-run source id, is_learn). Does not call Agent.run()."""

        parsed = parse_source_override(message)
        if parsed is None:
            return message, None, False
        selector, question = parsed
        source = await self.resolve_source(selector)
        if not question:
            raise UserInputError("A source override must be followed by a question.")
        return question, source.id, False

    async def skill_invocation_message(self, message: str) -> str | None:
        agent = self.require_agent()
        parts = message.split()
        if not parts:
            return None
        command = parts[0]
        if command == "/skills" and len(parts) >= 2 and parts[1] == "use":
            if len(parts) < 3:
                raise ValueError("usage: /skills use <name> [request]")
            skill_name = parts[2]
            skill = await agent.read_skill(skill_name)
            if skill is None:
                raise ValueError(f"skill not found: {skill_name}")
            return message
        if command in BUILTIN_SLASH_COMMANDS or not command.startswith("/"):
            return None
        skill_name = command[1:]
        if not skill_name:
            return None
        try:
            skill = await agent.read_skill(skill_name)
        except ValueError:
            return None
        return message if skill is not None else None

    async def dispatch_command(self, command: str) -> CommandOutcome:
        parts = command.split()
        name = parts[0] if parts else ""
        conversation_id = self.conversation_id
        if name == "/exit" and len(parts) == 1:
            return CommandOutcome("exit", conversation_id=conversation_id)
        if name == "/help" and len(parts) == 1:
            return CommandOutcome("notice", HELP_TEXT, conversation_id=conversation_id)
        if name == "/new" and len(parts) == 1:
            self.conversation_id = None
            return CommandOutcome("notice", "Conversation  new", conversation_id=None)
        if name == "/resume" and len(parts) == 2:
            candidate = parts[1]
            try:
                exists = await self.require_agent().conversation_exists(candidate)
            except (TypeError, ValueError) as error:
                return CommandOutcome(
                    "notice",
                    "Cannot resume conversation: "
                    + safe_display(str(error), fallback="invalid conversation"),
                    conversation_id=conversation_id,
                )
            if not exists:
                return CommandOutcome(
                    "notice",
                    "Cannot resume conversation: unknown conversation for this agent",
                    conversation_id=conversation_id,
                )
            self.conversation_id = candidate
            return CommandOutcome(
                "notice",
                f"Conversation  {safe_display(candidate, fallback='new')}",
                conversation_id=candidate,
            )
        if name == "/sources" and len(parts) == 1:
            return CommandOutcome(
                "notice",
                await self._sources_text(),
                conversation_id=conversation_id,
            )
        if name == "/catalog" and len(parts) == 1:
            return CommandOutcome(
                "screen",
                screen="catalog",
                conversation_id=conversation_id,
            )
        if name == "/settings" and len(parts) == 1:
            return CommandOutcome(
                "notice",
                self._settings_text(),
                conversation_id=conversation_id,
            )
        if name == "/status" and len(parts) == 1:
            return CommandOutcome(
                "notice",
                await self._status_text(),
                conversation_id=conversation_id,
            )
        if name == "/conversation" and len(parts) == 1:
            return CommandOutcome(
                "notice",
                "Conversation  " + safe_display(conversation_id, fallback="new"),
                conversation_id=conversation_id,
            )
        if name == "/conversation" and parts[1:] == ["clear"]:
            return CommandOutcome(
                "confirm",
                "Clear all conversation history and learning candidate records?",
                conversation_id=conversation_id,
                screen="confirm_clear_conversations",
            )
        if name == "/agent" and parts[1:] == ["delete"]:
            return CommandOutcome(
                "confirm",
                f"Type {self.require_agent().name} to permanently delete this agent.",
                conversation_id=conversation_id,
                screen="confirm_delete_agent",
                payload={"name": self.require_agent().name},
            )
        if name == "/source" and (len(parts) == 1 or parts[1:] == ["use"]):
            return CommandOutcome(
                "screen",
                screen="source_picker",
                conversation_id=conversation_id,
            )
        if name == "/source" and len(parts) >= 3 and parts[1] == "use":
            return await self._use_source(" ".join(parts[2:]))
        if name == "/source" and parts[1:] == ["add"]:
            return CommandOutcome(
                "screen",
                screen="source_setup",
                conversation_id=conversation_id,
            )
        if name == "/source" and parts[1:] == ["edit"]:
            return CommandOutcome(
                "screen",
                screen="source_edit",
                conversation_id=conversation_id,
            )
        if name == "/source" and len(parts) >= 3 and parts[1] == "detach":
            source = await self.resolve_source(" ".join(parts[2:]))
            return CommandOutcome(
                "confirm",
                "Detach "
                + safe_display(source.display_name, fallback="this source")
                + " and delete its Daita-owned credential?",
                conversation_id=conversation_id,
                screen="confirm_detach_source",
                payload={"source_id": source.id, "display_name": source.display_name},
            )
        if name == "/source" and len(parts) == 3 and parts[1] == "refresh":
            await self.refresh_source(parts[2])
            summary = await self.catalog_summary()
            if summary.is_empty:
                return CommandOutcome(
                    "screen",
                    screen="catalog_repair",
                    conversation_id=conversation_id,
                )
            return CommandOutcome(
                "screen",
                screen="catalog",
                conversation_id=conversation_id,
            )
        if name == "/source" and parts[1:] == ["permissions"]:
            return CommandOutcome(
                "screen",
                screen="permissions",
                conversation_id=conversation_id,
            )
        if name == "/model" and len(parts) == 1:
            return CommandOutcome(
                "screen",
                self._settings_text(),
                screen="model_setup",
                conversation_id=conversation_id,
            )
        if name == "/learn":
            invocation = learning_invocation_message(command)
            assert invocation is not None
            return CommandOutcome(
                "run",
                run_message=invocation,
                conversation_id=conversation_id,
            )
        knowledge = await self._dispatch_knowledge(parts)
        if knowledge is not None:
            return knowledge
        if name == "/resume":
            return CommandOutcome("notice", "Usage: /resume <conversation-id>")
        if name == "/source":
            return CommandOutcome(
                "notice",
                "Usage: /source | /source use <name> | /source add | /source edit | "
                "/source refresh <source-id> | /source detach <source> | "
                "/source permissions",
            )
        if name == "/conversation":
            return CommandOutcome(
                "notice",
                "Usage: /conversation | /conversation clear",
            )
        if name == "/agent":
            return CommandOutcome("notice", "Usage: /agent delete")
        if name in BUILTIN_SLASH_COMMANDS:
            return CommandOutcome("notice", f"Usage: {name}")
        return CommandOutcome("notice", "Unknown command. Type / to browse commands.")

    async def _use_source(self, selector: str) -> CommandOutcome:
        prior = await self.require_agent().active_source(
            conversation_id=self.conversation_id
        )
        selected = await self.select_source(selector)
        conversation_id = self.conversation_id
        message = f"Source  {safe_display(selected.display_name, fallback='source')}"
        if (prior is None or prior.id != selected.id) and conversation_id is not None:
            self.conversation_id = None
            conversation_id = None
            message += "\nStarted a new conversation to keep source context isolated."
        return CommandOutcome("notice", message, conversation_id=conversation_id)

    async def clear_conversations(self) -> CommandOutcome:
        cleared = await self.require_agent().clear_conversations()
        self.conversation_id = None
        noun = "run" if cleared == 1 else "runs"
        return CommandOutcome(
            "notice",
            f"Cleared {cleared} persisted conversation {noun}.\n"
            "Approved memory and skills were preserved.",
            conversation_id=None,
        )

    async def _dispatch_knowledge(self, parts: list[str]) -> CommandOutcome | None:
        name = parts[0] if parts else ""
        conversation_id = self.conversation_id
        agent = self.require_agent()
        if name == "/review":
            if len(parts) not in {1, 2}:
                return CommandOutcome("notice", "Usage: /review [cost-usd]")
            requested = None if len(parts) == 1 else parts[1]
            return await self.review_candidates(requested)
        if name in {"/memory", "/user"}:
            return await self._memory_command(parts)
        if name != "/skills":
            return None
        if len(parts) == 1:
            skills = await agent.list_skills()
            if not skills:
                return CommandOutcome("notice", "Skills:\n  (none)")
            lines = ["Skills:"]
            for summary in skills[:50]:
                lines.append(
                    "  /"
                    + safe_display(summary.name, fallback="skill")
                    + ": "
                    + safe_display(
                        summary.description,
                        fallback="description",
                        maximum=512,
                    )
                )
            if len(skills) > 50:
                lines.append(f"  +{len(skills) - 50} more")
            return CommandOutcome("notice", "\n".join(lines))
        if len(parts) == 2 and parts[1] == "create":
            return CommandOutcome(
                "screen",
                screen="skill_create",
                conversation_id=conversation_id,
            )
        if len(parts) == 3 and parts[1] == "create":
            return CommandOutcome(
                "screen",
                screen="skill_create",
                conversation_id=conversation_id,
                payload={"name": parts[2]},
            )
        if len(parts) == 3 and parts[1] == "show":
            skill = await agent.read_skill(parts[2])
            if skill is None:
                raise ValueError(f"skill not found: {parts[2]}")
            return CommandOutcome(
                "notice",
                "Skill: "
                + safe_display(skill.name, fallback="skill")
                + "\nDescription: "
                + safe_display(skill.description, fallback="description", maximum=512)
                + "\nInstructions:\n"
                + safe_display(
                    skill.instructions,
                    fallback="(empty)",
                    maximum=MAX_DISPLAY_CHARACTERS,
                ),
            )
        if len(parts) == 3 and parts[1] == "edit":
            return CommandOutcome(
                "screen",
                screen="skill_edit",
                payload={"name": parts[2]},
            )
        if len(parts) == 3 and parts[1] == "delete":
            return CommandOutcome(
                "confirm",
                f"Delete skill {parts[2]!r}?",
                screen="confirm_delete_skill",
                payload={"name": parts[2]},
            )
        if len(parts) >= 3 and parts[1] == "use":
            invocation = await self.skill_invocation_message(" ".join(parts))
            return CommandOutcome(
                "run",
                run_message=invocation,
                conversation_id=conversation_id,
            )
        return CommandOutcome(
            "notice",
            "Usage: /skills [show <name>|create [name]|edit <name>|"
            "delete <name>|use <name> [request]]",
        )

    async def _memory_command(self, parts: list[str]) -> CommandOutcome:
        name = parts[0]
        target = "memory" if name == "/memory" else "user"
        agent = self.require_agent()
        if len(parts) == 1:
            if target == "user":
                content = await agent.read_user_profile()
                return CommandOutcome(
                    "notice",
                    "User:\n"
                    + safe_display(
                        content,
                        fallback="(empty)",
                        maximum=MAX_DISPLAY_CHARACTERS,
                    ),
                )
            content = await agent.read_memory()
            return CommandOutcome("notice", await self._memory_surface(content))
        if target == "memory" and parts[1:] == ["list"]:
            return CommandOutcome(
                "notice",
                self._candidate_list_text(await agent.list_learning_candidates()),
            )
        if target == "user" and parts[1:] == ["edit"]:
            return CommandOutcome("screen", screen="edit_user")
        if target == "memory" and parts[1:] == ["edit"]:
            return CommandOutcome("screen", screen="edit_memory")
        if target == "memory" and len(parts) == 3 and parts[1] == "show":
            candidate = await agent.read_learning_candidate(parts[2])
            if candidate is not None:
                return CommandOutcome("notice", self._candidate_view_text(candidate))
            view = await agent.read_semantic_annotation(parts[2])
            if view is None:
                raise ValueError(f"memory record not found: {parts[2]}")
            return CommandOutcome("notice", self._semantic_view_text(view))
        if target == "memory" and len(parts) == 3 and parts[1] == "edit":
            return CommandOutcome(
                "screen",
                screen="edit_candidate",
                payload={"candidate_id": parts[2]},
            )
        if target == "memory" and len(parts) == 3 and parts[1] == "accept":
            result = await agent.accept_learning_candidate(parts[2])
            return CommandOutcome(
                "notice",
                "Candidate acceptance run:\n"
                + render_model_answer(
                    result.final_text,
                    fallback=f"{result.kind.value}: {result.reason}",
                ),
            )
        if target == "memory" and len(parts) in {3, 4} and parts[1] == "reject":
            reason = (
                LearningCandidateRejectionReason.USER_DECLINED
                if len(parts) == 3
                else LearningCandidateRejectionReason(parts[3])
            )
            rejected = await agent.reject_learning_candidate(parts[2], reason)
            return CommandOutcome(
                "notice",
                f"Learning candidate {rejected.candidate.id!r} rejected.",
            )
        if target == "memory" and parts[1:] == ["clear-rejected"]:
            cleared = await agent.clear_rejected_learning_candidates()
            noun = "rejected candidate" if cleared == 1 else "rejected candidates"
            return CommandOutcome("notice", f"Cleared {cleared} {noun}.")
        if target == "memory" and len(parts) == 3 and parts[1] == "delete":
            return CommandOutcome(
                "confirm",
                f"Delete semantic annotation {parts[2]!r}?",
                screen="confirm_delete_semantic",
                payload={"annotation_id": parts[2]},
            )
        usage = (
            "/memory [list|show <id>|edit [id]|accept <id>|"
            "reject <id> [reason]|clear-rejected|delete <semantic-id>]"
            if target == "memory"
            else "/user [edit]"
        )
        return CommandOutcome("notice", f"Usage: {usage}")

    async def review_candidates(self, requested: str | None) -> CommandOutcome:
        agent = self.require_agent()
        if requested is None:
            result = await agent.review_learning_candidates()
            if result.status in {
                LearningReviewStatus.DISABLED,
                LearningReviewStatus.COST_LIMIT_REQUIRED,
            }:
                return CommandOutcome(
                    "screen",
                    "Candidate review needs one-time authorization.",
                    screen="review_cost",
                )
            return CommandOutcome("notice", self._review_text(result))
        limit = parse_candidate_review_cost_limit(requested)
        result = await agent.review_learning_candidates(max_estimated_cost_usd=limit)
        return CommandOutcome("notice", self._review_text(result))

    async def save_skill(self, name: str, description: str, instructions: str) -> bool:
        validate_skill_name(name)
        Skill(name, description, instructions)
        return await self.require_agent().save_skill(name, description, instructions)

    async def read_skill(self, name: str) -> Skill | None:
        validate_skill_name(name)
        return await self.require_agent().read_skill(name)

    async def delete_skill(self, name: str) -> bool:
        return await self.require_agent().delete_skill(name)

    async def set_memory(self, text: str) -> None:
        await self.require_agent().set_memory(text)

    async def read_memory(self) -> str:
        return await self.require_agent().read_memory()

    async def set_user_profile(self, text: str) -> None:
        await self.require_agent().set_user_profile(text)

    async def read_user_profile(self) -> str:
        return await self.require_agent().read_user_profile()

    async def candidate_editor_document(self, candidate_id: str) -> str:
        view = await self.require_agent().read_learning_candidate(candidate_id)
        if view is None:
            raise ValueError(f"learning candidate not found: {candidate_id}")
        if view.status is not LearningCandidateStatus.AWAITING_REVIEW:
            raise ValueError(f"learning candidate is not editable: {view.status.value}")
        mapping = learning_candidate_content_to_mapping(view.candidate.content)
        return json.dumps(mapping.to_dict(), indent=2, sort_keys=True) + "\n"

    async def save_candidate_document(self, candidate_id: str, text: str) -> None:
        view = await self.require_agent().read_learning_candidate(candidate_id)
        if view is None:
            raise ValueError(f"learning candidate not found: {candidate_id}")
        if view.status is not LearningCandidateStatus.AWAITING_REVIEW:
            raise ValueError(f"learning candidate is not editable: {view.status.value}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("edited candidate content must be valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("edited candidate content must be one JSON object")
        content = learning_candidate_content_from_mapping(view.candidate.target, value)
        await self.require_agent().edit_learning_candidate(candidate_id, content)

    async def delete_semantic_annotation(self, annotation_id: str) -> bool:
        view = await self.require_agent().read_semantic_annotation(annotation_id)
        if view is None:
            raise ValueError(f"semantic annotation not found: {annotation_id}")
        return await self.require_agent().delete_semantic_annotation(
            annotation_id,
            expected_sha256=view.sha256,
        )

    def edit_document(self, seed: str) -> str:
        """Run the configured external editor for one bounded local document."""

        editor = os.environ.get("EDITOR")
        if editor is None or not editor.strip():
            raise RuntimeError(
                "$EDITOR is not set; set it to an available editor command"
            )
        try:
            command = shlex.split(editor)
        except ValueError as error:
            raise RuntimeError("$EDITOR is malformed") from error
        if not command:
            raise RuntimeError("$EDITOR is empty")
        agent_home = self.require_agent().home.resolve(strict=True)
        temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
        if temporary_root == agent_home or agent_home in temporary_root.parents:
            raise RuntimeError(
                "no temporary directory is available outside the agent home"
            )
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

    async def transcript(self, run_id: str) -> Transcript:
        return await self.require_agent().transcript(run_id)

    async def conversation_runs(
        self,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        return await self.require_agent().conversation_runs(conversation_id)

    async def artifact_notices(self, result: LoopExit) -> tuple[str, ...]:
        notices: list[str] = []
        for receipt in result.artifact_deliveries:
            filename = getattr(receipt, "filename", None)
            saved_path = getattr(receipt, "saved_path", None)
            if isinstance(filename, str) and isinstance(saved_path, str):
                notices.append(
                    "Saved "
                    + safe_display(filename, fallback="artifact")
                    + " to "
                    + safe_display(saved_path, fallback="the selected destination")
                )
        try:
            transcript = await self.require_agent().transcript(result.run_id)
            notices.extend(artifact_delivery_messages(completed_tool_pairs(transcript)))
        except Exception:
            pass
        return tuple(notices)

    async def _sources_text(self) -> str:
        sources = await self.list_sources()
        if not sources:
            return "Sources\n  (none)"
        lines = ["Sources"]
        for source in sources:
            label = SOURCE_TYPE_LABELS.get(source.adapter_id, source.adapter_id)
            state = "active" if source.active else "inactive"
            lines.append(
                f"  {safe_display(source.display_name, fallback='source')} "
                f"({label}, {state}) [{source.id}]"
            )
        return "\n".join(lines)

    def _settings_text(self) -> str:
        agent = self.require_agent()
        route = agent.model_route
        lines = [
            "Settings",
            f"  Agent      {safe_display(agent.name, fallback='agent')}",
        ]
        if route is None:
            lines.append("  Model      (not configured)")
            return "\n".join(lines)
        candidate = route.candidates[0]
        provider, _sep, model = candidate.provider_id.partition(":")
        lines.extend(
            [
                f"  Provider   {safe_display(dict(PROVIDERS).get(provider, provider), fallback='provider')}",
                f"  Model      {safe_display(model, fallback='model')}",
                "  Endpoint   "
                + (
                    "configured"
                    if candidate.base_url is not None
                    else "provider default"
                ),
                "  Credential "
                + (
                    "configured"
                    if candidate.secret_reference is not None
                    else "not required"
                ),
            ]
        )
        return "\n".join(lines)

    async def _status_text(self) -> str:
        agent = self.require_agent()
        source = await self.source_summary()
        return (
            f"Agent      {safe_display(agent.name, fallback='agent')}\n"
            f"Model      {safe_display(self.model_label(), fallback='model')}\n"
            f"Source     {source}\n"
            "Conversation  " + safe_display(self.conversation_id, fallback="new")
        )

    async def _memory_surface(self, memory_text: str) -> str:
        from daita import SemanticAnnotationState

        agent = self.require_agent()
        lines = [
            "Memory",
            "",
            "Global memory:",
            safe_display(
                memory_text, fallback="(empty)", maximum=MAX_DISPLAY_CHARACTERS
            ),
            "",
            "Pending candidates:",
        ]
        candidates = await agent.list_learning_candidates()
        if not candidates:
            lines.append("  (none)")
        else:
            for view in candidates[:12]:
                lines.append(
                    "  "
                    + safe_display(view.candidate.id, fallback="candidate")
                    + f" [{view.status.value}/{view.candidate.target.value}]"
                )
        views = await agent.list_semantic_annotations()
        for heading, state in (
            ("Active data semantics", SemanticAnnotationState.ACTIVE),
            ("Exact duplicates", SemanticAnnotationState.DUPLICATE),
            ("Stale definitions", SemanticAnnotationState.STALE),
            ("Conflicts", SemanticAnnotationState.CONFLICTING),
            ("Superseded definitions", SemanticAnnotationState.SUPERSEDED),
        ):
            lines.append("")
            lines.append(f"{heading}:")
            selected = tuple(item for item in views if item.state is state)
            if not selected:
                lines.append("  (none)")
                continue
            for semantic_view in selected:
                lines.append(
                    "  "
                    + safe_display(semantic_view.annotation.id, fallback="annotation")
                    + f" [{semantic_view.annotation.kind.value}] "
                    + safe_display(
                        semantic_view.annotation.statement,
                        fallback="definition",
                        maximum=512,
                    )
                )
        return "\n".join(lines)

    def _candidate_list_text(self, views: tuple[Any, ...]) -> str:
        if not views:
            return "Learning candidates\n  (none)"
        lines = ["Learning candidates"]
        for view in views:
            lines.append(
                "  "
                + safe_display(view.candidate.id, fallback="candidate")
                + f" [{view.status.value}/{view.candidate.target.value}]"
            )
        return "\n".join(lines)

    def _candidate_view_text(self, view: Any) -> str:
        candidate = view.candidate
        return (
            f"Learning candidate: {safe_display(candidate.id)}\n"
            f"Status: {view.status.value}\n"
            f"Target: {candidate.target.value}"
        )

    def _semantic_view_text(self, view: Any) -> str:
        annotation = view.annotation
        lines = [
            f"Semantic annotation: {annotation.id}",
            f"State: {view.state.value}",
            f"Kind: {annotation.kind.value}",
            "Verified revisions: "
            + ", ".join(
                f"{binding.resource_id}@{binding.revision}"
                for binding in annotation.catalog_revisions
            ),
            f"Confirmed: {annotation.confirmed_at.isoformat()} by {annotation.confirmed_by}",
            f"Current SHA-256: {view.sha256}",
            "Statement:",
            safe_display(
                annotation.statement,
                fallback="(empty)",
                maximum=MAX_DISPLAY_CHARACTERS,
            ),
            "Evidence:",
        ]
        for evidence in annotation.evidence:
            lines.append(f"  {evidence.kind.value} in run {evidence.run_id}")
        return "\n".join(lines)

    def _review_text(self, result: Any) -> str:
        return (
            "Learning review\n"
            f"  Status: {result.status.value}\n"
            f"  Reviewed runs: {len(result.reviewed_run_ids)}\n"
            f"  New candidates: {len(result.candidates)}\n"
            f"  Model calls: {result.model_calls}"
        )


def _configuration_text(configuration: Any, key: str) -> str:
    value = configuration.get(key)
    if not isinstance(value, str) or not value:
        raise UserInputError(f"saved source {key} is invalid")
    return value


def _configuration_port(configuration: Any) -> int:
    value = configuration.get("port")
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 65_535
    ):
        raise UserInputError("saved source port is invalid")
    return value


def _configuration_schemas(configuration: Any) -> tuple[str, ...]:
    value = configuration.get("schemas")
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(schema, str) or not schema for schema in value)
    ):
        raise UserInputError("saved source schemas are invalid")
    return tuple(value)


def _configuration_integer(configuration: Any, key: str) -> int:
    value = configuration.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise UserInputError(f"saved source {key} is invalid")
    return value
