"""Adapt the public Agent API into presentation-safe operations for Textual screens."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from daita import (
    ApprovalHandler,
    ConversationRun,
    DeliveryInspection,
    DeliveryView,
    FolderSource,
    LocalFileAccess,
    LocalWorkspace,
    LoopExit,
    MCPAdmissionError,
    MCPAuthentication,
    MCPBindingState,
    MCPBindingStatus,
    MCPServerInspection,
    MCPToolSelection,
    ScheduledRoutine,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    SourceType,
    SourceTypeRegistry,
    Transcript,
    WorkBoard,
)
from daita.agent import (
    AgentAlreadyExistsError,
    AgentHomeError,
    AgentModelConfigurationError,
    AgentNameError,
    HostActiveError,
    SourceRefreshError,
)
from daita.artifacts import (
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactPayload,
    ArtifactRef,
)
from daita.errors import StateCompatibilityError
from daita.host import service
from daita.host.client import RemoteAgent
from daita.host.paths import host_paths
from daita.host.settings import read_setting, write_setting
from daita.jobs import (
    GraphBoardProjection,
    GraphInspection,
    GraphJob,
    GraphTask,
    GraphTimelinePage,
    TaskControl,
    TaskResult,
)
from daita.llm import ModelSensitivity
from daita.llm.models import ToolCall, ToolResultBlock
from daita.llm.provider_definitions import SUBSCRIPTION_CLIENTS
from daita.memory.capabilities import MEMORY_SET_TOOL_NAME
from daita.observation import AgentObserver
from daita.routines.capabilities import ROUTINE_CREATE_CAPABILITY_ID
from daita.security import SecretReference
from daita.semantics import SEMANTIC_SAVE_TOOL_NAME
from daita.skills import Skill, validate_skill_name
from daita.skills.capabilities import SKILL_SAVE_TOOL_NAME

from .commands import (
    BUILTIN_SLASH_COMMANDS,
    learning_invocation_message,
    parse_source_override,
)
from .models import (
    HOST_GUIDANCE,
    PROVIDERS,
    CommandOutcome,
    UserInputError,
)
from .projection import artifact_delivery_messages
from .sanitization import MAX_DISPLAY_CHARACTERS, safe_display

_LEARNING_WRITE_TOOLS = frozenset(
    {MEMORY_SET_TOOL_NAME, SEMANTIC_SAVE_TOOL_NAME, SKILL_SAVE_TOOL_NAME}
)
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
}


class PresentationController:
    """Thin adapter from UI actions to the agent the Daita host owns."""

    def __init__(
        self,
        *,
        root: str | Path | None,
        workspace: LocalWorkspace,
    ) -> None:
        self.root = root
        if not isinstance(workspace, LocalWorkspace):
            raise TypeError("workspace must be LocalWorkspace")
        self.workspace = workspace
        self.agent: RemoteAgent | None = None
        # Called when the host could not replay missed events (RemoteAgent.on_reset).
        self.on_reset: Callable[[], None] | None = None
        self.conversation_id: str | None = None
        # (conversation_id, run_id, question) of the latest answered question.
        self.last_answer: tuple[str, str, str] | None = None
        self.validated_model = False

    @property
    def model_configuration_requires_replacement(self) -> bool:
        return (
            self.agent is not None
            and self.agent.model_configuration_requires_replacement
        )

    def require_agent(self) -> RemoteAgent:
        if self.agent is None:
            raise RuntimeError("no agent is open")
        return self.agent

    async def list_agents(self) -> tuple[str, ...]:
        return await RemoteAgent.list(root=self.root)

    async def close_agent(self) -> None:
        """Detach from the agent; its runs and schedulers keep going in the host."""

        agent = self.agent
        self.agent = None
        if agent is not None:
            await agent.close()

    async def close(self) -> None:
        await self.close_agent()

    def _host_workspace(self) -> LocalWorkspace:
        # The terminal works in its folder with computer access; the host refuses
        # any other sensitivity rather than silently lowering it.
        return LocalWorkspace(
            self.workspace.root,
            sensitivity=self.workspace.sensitivity,
            access=LocalFileAccess.COMPUTER,
        )

    async def open_agent(
        self,
        name: str,
        *,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> RemoteAgent:
        await self.close_agent()
        opened = await RemoteAgent.open(
            name,
            workspace=self._host_workspace(),
            root=self.root,
            observer=observer,
            approval_handler=approval_handler,
        )
        opened.on_reset = self.on_reset
        self.agent = opened
        return opened

    async def create_agent(
        self,
        name: str,
        *,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> RemoteAgent:
        await self.close_agent()
        try:
            created = await RemoteAgent.create(
                name,
                workspace=self._host_workspace(),
                root=self.root,
                observer=observer,
                approval_handler=approval_handler,
            )
        except (AgentAlreadyExistsError, AgentNameError, ValueError) as error:
            raise UserInputError(str(error)) from error
        created.on_reset = self.on_reset
        self.agent = created
        return created

    async def delete_open_agent(self) -> None:
        agent = self.require_agent()
        name = agent.name
        await self.close_agent()
        await self.delete_agent(name)

    async def delete_agent(self, name: str) -> None:
        """Delete one agent through the host, which closes it first."""

        try:
            await RemoteAgent.delete(name, root=self.root)
        except HostActiveError as error:
            raise UserInputError(
                "That agent is active in another Daita process. Close it, then retry."
            ) from error
        except (AgentHomeError, StateCompatibilityError) as error:
            raise UserInputError(str(error)) from error

    async def reopen_agent(
        self,
        *,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> RemoteAgent:
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

    def has_saved_model_login(self, *, provider: str) -> bool:
        """Project the current primary credential reference without resolving it."""
        route = self.agent.model_route if self.agent is not None else None
        if route is None:
            return False
        candidate = route.candidates[0]
        return (
            candidate.provider_id.partition(":")[0] == provider
            and candidate.secret_reference is not None
        )

    async def source_summary(self) -> str:
        return summarize_sources(await self.require_agent().list_sources())

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

    async def attach_folder(self, path: Path, *, name: str | None) -> Any:
        return await self.require_agent().attach(FolderSource(path, name=name))

    async def store_postgresql_password(self, password: str) -> SecretReference:
        return await self.require_agent().store_postgresql_password(password)

    async def delete_postgresql_password(self, reference: SecretReference) -> None:
        await self.require_agent().delete_postgresql_password(reference)

    async def probe_postgresql(self, **kwargs: Any) -> Any:
        return await self.require_agent().probe_postgresql(**kwargs)

    async def probe_postgresql_source(
        self,
        source: Any,
        *,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str | None,
        ssl_mode: str,
    ) -> Any:
        """Probe edited PostgreSQL fields without changing the saved source."""

        if source.adapter_id != "postgresql":
            raise UserInputError("Only PostgreSQL sources expose schema discovery.")
        agent = self.require_agent()
        reference_text = source.configuration.get("credential_ref")
        credential = (
            SecretReference.parse(reference_text)
            if isinstance(reference_text, str)
            else None
        )
        temporary_credential: SecretReference | None = None
        try:
            if password:
                temporary_credential = await agent.store_postgresql_password(password)
                credential = temporary_credential
            if credential is None:
                raise UserInputError(
                    "This PostgreSQL source has no saved password; enter a new password."
                )
            return await agent.probe_postgresql(
                host=host,
                port=port,
                database=database,
                username=username,
                credential=credential,
                ssl_mode=ssl_mode,
            )
        finally:
            password = None
            if temporary_credential is not None:
                await agent.delete_postgresql_password(temporary_credential)

    async def attach_postgresql(self, **kwargs: Any) -> Any:
        return await self.require_agent().attach_postgresql(**kwargs)

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

    async def inspect_catalog_resource(self, resource_id: str) -> Any:
        """Read one bounded relationship neighborhood through the public API."""

        return await self.require_agent().inspect_catalog_resource(resource_id)

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

    def source_types(self) -> tuple[SourceType, ...]:
        # Matches this controller's AgentConfig(): built-in types only.
        return SourceTypeRegistry().types()

    async def list_jobs(self) -> tuple[GraphJob, ...]:
        return await self.require_agent().list_jobs(limit=50)

    async def work_board(self) -> WorkBoard:
        return await self.require_agent().work_board()

    async def list_artifacts(
        self, *, limit: int, offset: int, newest_first: bool = False
    ) -> tuple[ArtifactRef, ...]:
        return await self.require_agent().list_artifacts(
            limit=limit, offset=offset, newest_first=newest_first
        )

    async def read_artifact(self, artifact_id: str) -> ArtifactPayload:
        return await self.require_agent().read_artifact(artifact_id)

    async def save_artifact(self, artifact_id: str) -> ArtifactDeliveryReceipt:
        return await self.require_agent().save_artifact(artifact_id)

    async def artifact_export_destination(self) -> ArtifactDestination:
        return await self.require_agent().export_destination()

    async def delete_artifact(self, artifact_id: str) -> bool:
        return await self.require_agent().delete_artifact(artifact_id)

    async def inspect_delivery(
        self,
        delivery_id: str,
    ) -> DeliveryInspection | None:
        return await self.require_agent().inspect_delivery(delivery_id)

    async def acknowledge_delivery(self, delivery_id: str) -> DeliveryView | None:
        return await self.require_agent().acknowledge_delivery(delivery_id)

    async def inspect_job(self, job_id: str) -> GraphInspection | None:
        return await self.require_agent().inspect_job(job_id)

    async def job_board(self, job_id: str) -> GraphBoardProjection | None:
        return await self.require_agent().job_board(job_id)

    async def job_timeline(
        self,
        job_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 100,
    ) -> GraphTimelinePage | None:
        return await self.require_agent().job_timeline(
            job_id,
            after_event_id=after_event_id,
            limit=limit,
        )

    async def inspect_job_task(self, job_id: str, task_id: str) -> GraphTask | None:
        return await self.require_agent().inspect_job_task(job_id, task_id)

    async def read_job_result(self, job_id: str) -> TaskResult | None:
        return await self.require_agent().read_job_result(job_id)

    async def cancel_job(self, job_id: str) -> GraphJob | None:
        return await self.require_agent().cancel_job(job_id)

    async def answer_job_input(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        answer: dict[str, object],
        idempotency_key: str,
    ) -> TaskControl | None:
        inspection = await self._owned_job(job_id)
        return await self.require_agent().answer_task_input(
            job_id,
            task_id,
            control_id,
            principal_id=inspection.job.specification.principal_id,
            answer=answer,
            idempotency_key=idempotency_key,
        )

    async def accept_job_review(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        rationale: str,
        idempotency_key: str,
    ) -> TaskResult:
        inspection = await self._owned_job(job_id)
        return await self.require_agent().accept_task_review(
            job_id,
            task_id,
            control_id,
            principal_id=inspection.job.specification.principal_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
        )

    async def request_job_review_changes(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        rationale: str,
        replacement_guidance: str,
        idempotency_key: str,
    ) -> TaskControl:
        inspection = await self._owned_job(job_id)
        return await self.require_agent().request_task_review_changes(
            job_id,
            task_id,
            control_id,
            principal_id=inspection.job.specification.principal_id,
            rationale=rationale,
            replacement_guidance=replacement_guidance,
            idempotency_key=idempotency_key,
        )

    async def retry_job_control(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        advisory_note: str,
        idempotency_key: str,
    ) -> TaskControl | None:
        inspection = await self._owned_job(job_id)
        return await self.require_agent().retry_task_control(
            job_id,
            task_id,
            control_id,
            principal_id=inspection.job.specification.principal_id,
            advisory_note=advisory_note,
            idempotency_key=idempotency_key,
        )

    async def reject_job_control(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        reason: str,
        idempotency_key: str,
    ) -> TaskControl | None:
        inspection = await self._owned_job(job_id)
        return await self.require_agent().reject_task_control(
            job_id,
            task_id,
            control_id,
            principal_id=inspection.job.specification.principal_id,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    async def replace_job_task(
        self,
        job_id: str,
        task_id: str,
        *,
        advisory_note: str,
        idempotency_key: str,
        expected_revision: int,
    ) -> object:
        inspection = await self._owned_job(job_id)
        return await self.require_agent().replace_graph_task_by_policy(
            job_id,
            task_id,
            principal_id=inspection.job.specification.principal_id,
            advisory_note=advisory_note,
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
        )

    async def _owned_job(self, job_id: str) -> GraphInspection:
        inspection = await self.require_agent().inspect_job(job_id)
        if inspection is None:
            raise ValueError("No durable job with that ID belongs to this agent.")
        return inspection

    async def list_routines(self) -> tuple[ScheduledRoutineSummary, ...]:
        return await self.require_agent().list_routines(limit=50)

    async def inspect_routine(
        self, routine_id: str
    ) -> ScheduledRoutineInspection | None:
        return await self.require_agent().inspect_routine(routine_id)

    async def control_routine(
        self,
        routine_id: str,
        *,
        expected_revision: int,
        action: str,
    ) -> ScheduledRoutine:
        operations = {
            "pause": self.require_agent().pause_routine,
            "resume": self.require_agent().resume_routine,
            "run_now": self.require_agent().run_routine_now,
            "disable": self.require_agent().disable_routine,
        }
        try:
            operation = operations[action]
        except KeyError:
            raise ValueError("unknown routine control action") from None
        return await operation(routine_id, expected_revision=expected_revision)

    async def resolve_source(self, selector: str) -> Any:
        try:
            return await self.require_agent().resolve_source(selector)
        except ValueError as error:
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
            return CommandOutcome(
                "screen", screen="help", conversation_id=conversation_id
            )
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
                "screen",
                screen="catalog",
                conversation_id=conversation_id,
            )
        if name == "/mcp":
            return await self._mcp_command(parts)
        if name == "/jobs":
            return await self._jobs_command(parts)
        if name == "/routines":
            return await self._routines_command(parts, command)
        if name == "/work":
            if len(parts) == 1:
                return CommandOutcome(
                    "screen",
                    screen="work",
                    conversation_id=conversation_id,
                )
            return CommandOutcome(
                "notice",
                "Usage: /work",
                conversation_id=conversation_id,
            )
        if name == "/artifacts":
            return CommandOutcome(
                "screen" if len(parts) == 1 else "notice",
                "" if len(parts) == 1 else "Usage: /artifacts",
                screen="artifacts" if len(parts) == 1 else None,
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
        if name == "/workspace" and len(parts) == 1:
            return CommandOutcome(
                "notice",
                "Working directory  "
                + safe_display(str(self.workspace.root), fallback="admitted")
                + f" [{self.workspace.access.value} access; {self.workspace.sensitivity.value}]",
                conversation_id=conversation_id,
            )
        if name == "/files":
            return CommandOutcome(
                "notice",
                "Usage: /files <question>",
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
                "Clear all conversation history?",
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
        if name == "/source" and len(parts) == 1:
            return CommandOutcome(
                "screen", screen="catalog", conversation_id=conversation_id
            )
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
            refreshed = await self.refresh_source(parts[2])
            resources = await self.list_catalog_resources(source_id=refreshed.id)
            noun = "resource" if len(resources) == 1 else "resources"
            if resources:
                message = (
                    "Catalog refresh succeeded · "
                    + safe_display(refreshed.display_name, fallback="source")
                    + f" · {len(resources)} {noun}"
                )
            else:
                message = (
                    "Catalog refresh completed, but found no resources · "
                    + safe_display(refreshed.display_name, fallback="source")
                    + " · open /sources and choose Edit to review its schemas or path"
                )
            return CommandOutcome(
                "screen",
                message,
                screen="catalog",
                conversation_id=conversation_id,
                payload={"catalog_notice_warning": not resources},
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
        if name in {"/save", "/schedule"} or parts == ["/learn"]:
            return await self._next_step_command(name, command)
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
                "Usage: /source | /source add | /source edit | "
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

    async def _routines_command(self, parts: list[str], command: str) -> CommandOutcome:
        conversation_id = self.conversation_id
        if len(parts) == 1:
            return CommandOutcome(
                "screen",
                screen="routines",
                conversation_id=conversation_id,
            )
        if len(parts) >= 3 and parts[1] == "create":
            instruction = command.split(maxsplit=2)[2].strip()
            return CommandOutcome(
                "run",
                run_message=(
                    "Create a saved assignment for this self-contained "
                    "instruction, eliciting any missing schedule or scope details and "
                    "using the routine management tools: " + instruction
                ),
                conversation_id=conversation_id,
            )
        if len(parts) >= 4 and parts[1] == "promote":
            instruction = command.split(maxsplit=3)[3].strip()
            return CommandOutcome(
                "run",
                run_message=(
                    "Promote completed run "
                    + parts[2]
                    + " into a saved assignment with this self-contained "
                    "instruction, using exact promotion evidence: " + instruction
                ),
                conversation_id=conversation_id,
            )
        if len(parts) >= 4 and parts[1] == "update":
            instruction = command.split(maxsplit=3)[3].strip()
            return CommandOutcome(
                "run",
                run_message=(
                    "Inspect and update saved assignment "
                    + parts[2]
                    + " using its exact current revision and the routine management "
                    "tools. The replacement self-contained instruction is: "
                    + instruction
                ),
                conversation_id=conversation_id,
            )
        return CommandOutcome(
            "notice",
            "Usage: /routines | /routines create <instruction> | "
            "/routines promote <basis-run-id> <instruction> | "
            "/routines update <routine-id> <instruction>",
            conversation_id=conversation_id,
        )

    async def _next_step_command(self, name: str, command: str) -> CommandOutcome:
        """Act on the latest answer without the user handling run identities."""

        conversation_id = self.conversation_id
        answer = self.last_answer
        if answer is None or answer[0] != conversation_id:
            return CommandOutcome(
                "notice",
                (
                    "Usage: /learn <material>, or /learn alone right after an "
                    "answer to remember what worked."
                    if name == "/learn"
                    else f"{name} acts on the latest answer. Ask a question first."
                ),
                conversation_id=conversation_id,
            )
        _conversation, run_id, question = answer
        argument = command.removeprefix(name).strip()
        if name == "/save":
            return CommandOutcome(
                "run",
                run_message=(
                    "Save your answer to the previous question in this conversation "
                    "as a document artifact, keeping its content and tables as "
                    "answered. Do not query sources again."
                ),
                conversation_id=conversation_id,
            )
        if name == "/schedule":
            if not argument:
                return CommandOutcome(
                    "screen",
                    screen="schedule_prompt",
                    payload={"question": question},
                    conversation_id=conversation_id,
                )
            promote = f"/routines promote {run_id} {question}\nSchedule: {argument}"
            return await self._routines_command(promote.split(), promote)
        material = (
            f"The approach and facts behind my question {question!r} and your answer "
            f"(run {run_id}) are correct and worth reusing."
        )
        return CommandOutcome(
            "run",
            run_message=learning_invocation_message("/learn " + material),
            conversation_id=conversation_id,
        )

    async def _jobs_command(self, parts: list[str]) -> CommandOutcome:
        conversation_id = self.conversation_id
        if len(parts) == 1:
            return CommandOutcome(
                "screen",
                screen="jobs",
                conversation_id=conversation_id,
            )
        return CommandOutcome("notice", "Usage: /jobs", conversation_id=conversation_id)

    async def _mcp_command(self, parts: list[str]) -> CommandOutcome:
        agent = self.require_agent()
        conversation_id = self.conversation_id
        if len(parts) == 1:
            return CommandOutcome(
                "screen",
                screen="mcp_management",
                conversation_id=conversation_id,
            )
        if parts[1:] == ["add"]:
            return CommandOutcome(
                "screen",
                screen="mcp_setup",
                conversation_id=conversation_id,
            )
        if parts[1:] == ["status"]:
            statuses = await agent.list_mcp_servers()
            if not statuses:
                message = "MCP binding details  none"
            else:
                lines = ["MCP binding details"]
                for status in statuses:
                    binding = status.binding
                    display_state = (
                        "needs refresh"
                        if binding.state is MCPBindingState.STALE
                        else (
                            "revoked"
                            if binding.state is MCPBindingState.REVOKED
                            else (
                                "restart required"
                                if status.reopen_required
                                else (
                                    "ready"
                                    if status.active_in_runtime
                                    else "unavailable"
                                )
                            )
                        )
                    )
                    lines.append(
                        "  "
                        + safe_display(binding.binding_id, fallback="binding")
                        + "  "
                        + display_state
                        + "  "
                        + safe_display(
                            binding.endpoint, fallback="endpoint unavailable"
                        )
                    )
                message = "\n".join(lines)
            return CommandOutcome(
                "notice",
                message,
                conversation_id=conversation_id,
            )
        if len(parts) == 3 and parts[1] == "inspect":
            inspection = await self.inspect_mcp_server(parts[2])
            lines = [
                "MCP inspection  "
                + safe_display(inspection.server_name, fallback="unknown server")
                + "  "
                + safe_display(inspection.server_version, fallback="unknown version")
            ]
            lines.extend(
                "  "
                + safe_display(tool.remote_name, fallback="tool")
                + ("  supported" if tool.supported else "  unsupported")
                + (
                    ""
                    if tool.unsupported_reason is None
                    else "  "
                    + safe_display(tool.unsupported_reason, fallback="rejected")
                )
                for tool in inspection.tools
            )
            return CommandOutcome(
                "notice",
                "\n".join(lines),
                conversation_id=conversation_id,
            )
        if len(parts) == 5 and parts[1] == "attach":
            endpoint, remote_name, local_alias = parts[2:]
            status = await self.attach_mcp_tools(
                endpoint,
                (
                    MCPToolSelection(
                        remote_name=remote_name,
                        local_alias=local_alias,
                        description=(
                            "Read the explicitly admitted MCP tool " + remote_name + "."
                        ),
                    ),
                ),
            )
            return CommandOutcome(
                "notice",
                "MCP binding "
                + safe_display(status.binding.binding_id, fallback="created")
                + " attached; reopen Daita to activate it.",
                conversation_id=conversation_id,
            )
        if len(parts) == 3 and parts[1] == "refresh":
            status = await self.refresh_mcp_server(parts[2])
            return CommandOutcome(
                "notice",
                "MCP binding "
                + safe_display(status.binding.binding_id, fallback="binding")
                + " refreshed; state "
                + safe_display(status.binding.state.value, fallback="unknown")
                + ".",
                conversation_id=conversation_id,
            )
        if len(parts) == 3 and parts[1] == "revoke":
            return CommandOutcome(
                "confirm",
                "Revoke MCP binding "
                + safe_display(parts[2], fallback="this binding")
                + "?",
                conversation_id=conversation_id,
                screen="confirm_revoke_mcp",
                payload={"binding_id": parts[2]},
            )
        return CommandOutcome(
            "notice",
            "Use /mcp to open the server manager and add, refresh, or revoke tool access.",
            conversation_id=conversation_id,
        )

    async def list_mcp_servers(self) -> tuple[MCPBindingStatus, ...]:
        return await self.require_agent().list_mcp_servers()

    async def store_mcp_bearer(self, credential: str) -> SecretReference:
        return await self.require_agent().store_mcp_bearer(credential)

    async def delete_mcp_bearer(self, reference: SecretReference) -> None:
        await self.require_agent().delete_mcp_bearer(reference)

    async def inspect_mcp_server(
        self, endpoint: str, *, authentication: MCPAuthentication | None = None
    ) -> MCPServerInspection:
        return await self.require_agent().inspect_mcp_server(
            endpoint=endpoint, authentication=authentication
        )

    async def attach_mcp_tools(
        self,
        endpoint: str,
        selections: tuple[MCPToolSelection, ...],
        *,
        maximum_outbound_sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
        authentication: MCPAuthentication | None = None,
    ) -> MCPBindingStatus:
        try:
            return await self.require_agent().attach_mcp_server(
                endpoint=endpoint,
                selections=selections,
                maximum_outbound_sensitivity=maximum_outbound_sensitivity,
                authentication=authentication,
            )
        except MCPAdmissionError as error:
            reason = error.details.get("reason")
            if error.code != "mcp_schema_unsupported" or not isinstance(reason, str):
                raise
            raise UserInputError(
                "Cannot attach MCP tool: "
                + safe_display(
                    reason,
                    fallback="unsupported schema",
                    maximum=512,
                )
            ) from error

    async def refresh_mcp_server(self, binding_id: str) -> MCPBindingStatus:
        return await self.require_agent().refresh_mcp_server(binding_id)

    async def revoke_mcp_server(self, binding_id: str) -> str:
        status = await self.require_agent().revoke_mcp_server(binding_id)
        return (
            "MCP binding "
            + safe_display(status.binding.binding_id, fallback="binding")
            + " revoked."
        )

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
        if target == "user" and parts[1:] == ["edit"]:
            return CommandOutcome("screen", screen="edit_user")
        if target == "memory" and parts[1:] == ["edit"]:
            return CommandOutcome("screen", screen="edit_memory")
        if target == "memory" and len(parts) == 3 and parts[1] == "show":
            view = await agent.read_semantic_annotation(parts[2])
            if view is None:
                raise ValueError(f"memory record not found: {parts[2]}")
            return CommandOutcome("notice", self._semantic_view_text(view))
        if target == "memory" and len(parts) == 3 and parts[1] == "delete":
            return CommandOutcome(
                "confirm",
                f"Delete semantic annotation {parts[2]!r}?",
                screen="confirm_delete_semantic",
                payload={"annotation_id": parts[2]},
            )
        usage = (
            "/memory [show <semantic-id>|edit|delete <semantic-id>]"
            if target == "memory"
            else "/user [edit]"
        )
        return CommandOutcome("notice", f"Usage: {usage}")

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

    async def tool_evidence(
        self, run_id: str
    ) -> tuple[tuple[ToolCall, ToolResultBlock | None], ...]:
        return await self.require_agent().tool_evidence(run_id)

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
            notices.extend(
                artifact_delivery_messages(
                    await self.require_agent().tool_evidence(result.run_id)
                )
            )
        except Exception:
            pass
        return tuple(notices)

    async def learning_saved(self, run_id: str) -> bool:
        """Whether a run changed memory, semantics or skills, not what it claimed."""

        return any(
            call.name in _LEARNING_WRITE_TOOLS
            and output is not None
            and not output.is_error
            for call, output in await self.require_agent().tool_evidence(run_id)
        )

    async def created_routine(self, run_id: str) -> bool:
        """Whether a run committed a new routine."""

        return any(
            output is not None
            and not output.is_error
            and output.capability_id == ROUTINE_CREATE_CAPABILITY_ID
            for _call, output in await self.require_agent().tool_evidence(run_id)
        )

    def background_offer_due(self) -> bool:
        """Offer the login service once, and only while it is not installed."""

        try:
            return not service.installed() and read_setting(
                host_paths(self.root), "background_offer"
            )
        except ValueError:
            return False  # a malformed host.toml is never rewritten

    async def answer_background_offer(self, accepted: bool) -> list[str]:
        """Never ask again; with yes, install and start the login service."""

        paths = host_paths(self.root)
        write_setting(paths, "background_offer", False)
        if not accepted:
            return []
        return await asyncio.to_thread(service.install, paths)

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
            f"Sources    {source}\n"
            f"Host       {agent.connection}. {HOST_GUIDANCE}\n"
            f"Working dir  {safe_display(str(self.workspace.root), fallback='admitted')} "
            f"[{self.workspace.access.value} access; {self.workspace.sensitivity.value}]\n"
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
        ]
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


def summarize_sources(sources: tuple[Any, ...]) -> str:
    """Name the one active source, or count them, for the status bar."""

    active = tuple(source for source in sources if source.active)
    if not active:
        return "no sources"
    if len(active) == 1:
        return safe_display(active[0].display_name, fallback="source")
    return f"{len(active)} sources"


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
