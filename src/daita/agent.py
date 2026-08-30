"""Expose the public facade for configuring and running a persistent data agent."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Self

from ._json import FrozenJsonObject
from .adapters.job_profiles import ConnectedJobProfile
from .adapters.mcp import (
    MCPAuthentication,
    MCPBindingStatus,
    MCPClientFactory,
    MCPServerInspection,
    MCPToolSelection,
)
from .adapters.models import SourceRegistration
from .adapters.postgresql import (
    PostgreSQLProbeResult,
    PostgreSQLSourceError,
)
from .adapters.postgresql_write import PostgreSQLUpdateReadiness
from .adapters.protocols import (
    ResourceAdapterError as SourceRefreshError,
    ResourceSource,
)
from .artifacts.models import (
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactPayload,
)
from .distribution import DeliveryInspection, DistributionDestination, InboxView
from .capabilities import ApprovalHandler
from .catalog.models import (
    CatalogResource,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSummary,
)
from .config import AgentConfig
from .hosting.embedded import (
    AgentAlreadyExistsError,
    AgentHomeError,
    AgentIdentityMismatchError,
    AgentModelConfigurationError,
    AgentNameError,
    AgentNotConfiguredError,
    AgentNotFoundError,
    EmbeddedAgent,
    HostActiveError,
    SourceEditConfirmationHandler,
    SourceEditPreview as SourceEditPreview,
    SourceEditResult,
    SourceSelectionError,
)
from .jobs.models import (
    JobExecutionMode,
    JobInspection,
    JobResultView,
    JobStatus,
    JobSummary,
)
from .learning_candidates import (
    LearningCandidateContent,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    LearningCandidateView,
    LearningReviewResult,
)
from .llm.models import ModelProfile, ModelSensitivity
from .llm.protocols import ModelProvider
from .llm.routing import ModelRoute
from .llm.subscription_auth import CodexDevicePrompt
from .loop.models import ConversationRun, LoopExit, LoopLimits, Transcript
from .observation import AgentObserver
from .routines import (
    RoutineState,
    ScheduledRoutineDraft,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    ScheduledRoutine,
)
from .security import KeychainStore, SecretProvider, SecretReference
from .semantics import (
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticAnnotationView,
    SemanticKind,
)
from .skills import Skill, SkillSummary
from .storage.sqlite_records import (
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    SourceReadMode,
)
from .workspace import LocalWorkspace


class Agent:
    """Persistent identity with one transcript-driven run path."""

    def __init__(self, embedded: EmbeddedAgent) -> None:
        self._embedded = embedded

    @classmethod
    async def list(
        cls,
        *,
        root: str | Path | None = None,
    ) -> tuple[str, ...]:
        """Return valid agent names beneath one admitted Daita root."""

        return await EmbeddedAgent.list(root=root)

    @classmethod
    async def delete(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        keychain: KeychainStore | None = None,
    ) -> None:
        """Permanently delete one inactive agent home and its owned credentials."""

        await EmbeddedAgent.delete(name, root=root, keychain=keychain)

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        workspace: LocalWorkspace,
        root: str | Path | None = None,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        mcp_client_factory: MCPClientFactory | None = None,
        keychain: KeychainStore | None = None,
        model_validator: ModelProvider | None = None,
        reviewer_model: ModelProvider | None = None,
        reviewer_profile: ModelProfile | None = None,
        reviewer_max_estimated_cost_usd: Decimal | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        downloads_directory: Path | None = None,
        connected_job_profiles: tuple[ConnectedJobProfile, ...] = (),
    ) -> Self:
        _validate_downloads_directory(downloads_directory)
        return cls(
            await EmbeddedAgent.create(
                name,
                workspace=workspace,
                root=root,
                config=config,
                model=model,
                model_profile=model_profile,
                limits=limits,
                clock=clock,
                id_factory=id_factory,
                secret_provider=secret_provider,
                mcp_client_factory=mcp_client_factory,
                keychain=keychain,
                model_validator=model_validator,
                reviewer_model=reviewer_model,
                reviewer_profile=reviewer_profile,
                reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
                observer=observer,
                approval_handler=approval_handler,
                downloads_directory=downloads_directory,
                connected_job_profiles=connected_job_profiles,
            )
        )

    @classmethod
    async def open(
        cls,
        name: str,
        *,
        workspace: LocalWorkspace,
        root: str | Path | None = None,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        mcp_client_factory: MCPClientFactory | None = None,
        keychain: KeychainStore | None = None,
        model_validator: ModelProvider | None = None,
        reviewer_model: ModelProvider | None = None,
        reviewer_profile: ModelProfile | None = None,
        reviewer_max_estimated_cost_usd: Decimal | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        downloads_directory: Path | None = None,
        connected_job_profiles: tuple[ConnectedJobProfile, ...] = (),
    ) -> Self:
        _validate_downloads_directory(downloads_directory)
        return cls(
            await EmbeddedAgent.open(
                name,
                workspace=workspace,
                root=root,
                config=config,
                model=model,
                model_profile=model_profile,
                limits=limits,
                clock=clock,
                id_factory=id_factory,
                secret_provider=secret_provider,
                mcp_client_factory=mcp_client_factory,
                keychain=keychain,
                model_validator=model_validator,
                reviewer_model=reviewer_model,
                reviewer_profile=reviewer_profile,
                reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
                observer=observer,
                approval_handler=approval_handler,
                downloads_directory=downloads_directory,
                connected_job_profiles=connected_job_profiles,
            )
        )

    @property
    def id(self) -> str:
        return self._embedded.identity.id

    @property
    def name(self) -> str:
        return self._embedded.identity.display_name

    @property
    def home(self) -> Path:
        return self._embedded.home

    @property
    def workspace(self) -> LocalWorkspace:
        return self._embedded.workspace

    @property
    def model_profile(self) -> ModelProfile | None:
        return self._embedded.model_profile

    @property
    def model_route(self) -> ModelRoute | None:
        return self._embedded.model_route

    def model_requires_explicit_limits(self, *, provider: str, model: str) -> bool:
        """Return whether onboarding must collect hard token limits."""

        return self._embedded.model_requires_explicit_limits(
            provider=provider,
            model=model,
        )

    async def authenticate_model_subscription(
        self,
        *,
        provider: str,
        on_verification: Callable[[CodexDevicePrompt], None],
        on_progress: Callable[[str], None] | None = None,
    ) -> str:
        """Return an opaque subscription credential for model configuration."""

        return await self._embedded.authenticate_model_subscription(
            provider=provider,
            on_verification=on_verification,
            on_progress=on_progress,
        )

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
    ) -> ModelRoute:
        """Validate and persist one model route for the next open."""

        credential = api_key
        subscription_secret = subscription_credential
        api_key = None
        subscription_credential = None
        try:
            return await self._embedded.configure_model(
                provider=provider,
                model=model,
                api_key=credential,
                subscription_credential=subscription_secret,
                base_url=base_url,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
            )
        finally:
            credential = None
            subscription_secret = None

    async def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_id: str | None = None,
        files_only: bool = False,
        job_executor_profile_id: str | None = None,
    ) -> LoopExit:
        return await self._embedded.run(
            message,
            conversation_id=conversation_id,
            source_id=source_id,
            files_only=files_only,
            job_executor_profile_id=job_executor_profile_id,
        )

    async def learn(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_id: str | None = None,
    ) -> LoopExit:
        """Run one explicit user-authorized foreground learning action."""

        return await self._embedded.learn(
            message,
            conversation_id=conversation_id,
            source_id=source_id,
        )

    async def transcript(self, run_id: str) -> Transcript:
        return await self._embedded.transcript(run_id)

    async def inbox(
        self,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = 50,
    ) -> tuple[InboxView, ...]:
        """Inspect bounded durable autonomous results for this agent."""

        return await self._embedded.inbox(
            conversation_id=conversation_id,
            include_acknowledged=include_acknowledged,
            limit=limit,
        )

    async def distribution_destinations(
        self,
        conversation_id: str,
        *,
        sensitivity_ceiling: ModelSensitivity = ModelSensitivity.RESTRICTED,
    ) -> tuple[DistributionDestination, ...]:
        """List exact destinations currently selectable for one conversation."""

        return await self._embedded.distribution_destinations(
            conversation_id,
            sensitivity_ceiling=sensitivity_ceiling,
        )

    async def inspect_delivery(
        self,
        delivery_id: str,
    ) -> DeliveryInspection | None:
        """Inspect immutable safe facts for one exact logical delivery."""

        return await self._embedded.inspect_delivery(delivery_id)

    async def acknowledge_inbox(self, delivery_id: str) -> InboxView | None:
        """Idempotently acknowledge one exact inbox result."""

        return await self._embedded.acknowledge_inbox(delivery_id)

    async def conversation_runs(
        self,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        return await self._embedded.conversation_runs(conversation_id)

    async def conversation_exists(self, conversation_id: str) -> bool:
        """Return whether one conversation ID belongs to this agent."""

        return await self._embedded.conversation_exists(conversation_id)

    async def list_jobs(
        self,
        *,
        statuses: frozenset[JobStatus] = frozenset(),
        limit: int = 50,
    ) -> tuple[JobSummary, ...]:
        return await self._embedded.list_jobs(statuses=statuses, limit=limit)

    async def inspect_job(self, job_id: str) -> JobInspection | None:
        return await self._embedded.inspect_job(job_id)

    async def read_job_result(self, job_id: str) -> JobResultView | None:
        return await self._embedded.read_job_result(job_id)

    async def cancel_job(self, job_id: str) -> JobInspection | None:
        return await self._embedded.cancel_job(job_id)

    async def propose_routine(self, draft: ScheduledRoutineDraft) -> ScheduledRoutine:
        return await self._embedded.propose_routine(draft)

    async def promote_routine(
        self,
        draft: ScheduledRoutineDraft,
        *,
        basis_run_id: str,
    ) -> ScheduledRoutine:
        return await self._embedded.promote_routine(
            draft,
            basis_run_id=basis_run_id,
        )

    async def create_routine(self, proposal: ScheduledRoutine) -> ScheduledRoutine:
        return await self._embedded.create_routine(proposal)

    async def list_routines(
        self,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = 50,
    ) -> tuple[ScheduledRoutineSummary, ...]:
        return await self._embedded.list_routines(states=states, limit=limit)

    async def inspect_routine(
        self, routine_id: str
    ) -> ScheduledRoutineInspection | None:
        return await self._embedded.inspect_routine(routine_id)

    async def update_routine(
        self,
        routine_id: str,
        *,
        expected_revision: int,
        draft: ScheduledRoutineDraft,
        basis_run_id: str | None = None,
    ) -> ScheduledRoutine:
        return await self._embedded.update_routine(
            routine_id,
            expected_revision=expected_revision,
            draft=draft,
            basis_run_id=basis_run_id,
        )

    async def pause_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._embedded.pause_routine(
            routine_id,
            expected_revision=expected_revision,
        )

    async def resume_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._embedded.resume_routine(
            routine_id,
            expected_revision=expected_revision,
        )

    async def run_routine_now(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._embedded.run_routine_now(
            routine_id,
            expected_revision=expected_revision,
        )

    async def disable_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._embedded.disable_routine(
            routine_id,
            expected_revision=expected_revision,
        )

    async def clear_conversations(self) -> int:
        """Delete transcripts and candidate records, not approved knowledge."""

        return await self._embedded.clear_conversations()

    async def read_artifact(self, artifact_id: str) -> ArtifactPayload:
        _validate_artifact_id(artifact_id)
        return await self._embedded.read_artifact(artifact_id)

    async def save_artifact(
        self,
        artifact_id: str,
        destination: Path | None = None,
        *,
        filename: str | None = None,
    ) -> ArtifactDeliveryReceipt:
        _validate_artifact_id(artifact_id)
        if destination is not None and not isinstance(destination, Path):
            raise TypeError("destination must be pathlib.Path or None")
        if filename is not None and (
            not isinstance(filename, str) or not filename or len(filename) > 120
        ):
            raise ValueError("filename must be 1 through 120 characters or None")
        return await self._embedded.save_artifact(
            artifact_id,
            destination,
            filename=filename,
        )

    async def export_destination(self) -> ArtifactDestination:
        return await self._embedded.export_destination()

    async def set_export_destination(self, directory: Path) -> ArtifactDestination:
        if not isinstance(directory, Path):
            raise TypeError("directory must be pathlib.Path")
        return await self._embedded.set_export_destination(directory)

    async def reset_export_destination(self) -> ArtifactDestination:
        return await self._embedded.reset_export_destination()

    async def active_source(
        self,
        *,
        conversation_id: str | None = None,
    ) -> SourceRegistration | None:
        """Return the default or conversation-pinned active source."""

        return await self._embedded.active_source(conversation_id=conversation_id)

    async def resolve_source(self, selector: str) -> SourceRegistration:
        """Resolve one active source ID, display name, or display-name alias."""

        return await self._embedded.resolve_source(selector)

    async def select_source(self, selector: str) -> SourceRegistration:
        """Persist one source as the default for subsequent conversations."""

        return await self._embedded.select_source(selector)

    async def read_memory(self) -> str:
        return await self._embedded.read_memory()

    async def set_memory(self, text: str) -> None:
        await self._embedded.set_memory(text)

    async def read_user_profile(self) -> str:
        return await self._embedded.read_user_profile()

    async def set_user_profile(self, text: str) -> None:
        await self._embedded.set_user_profile(text)

    async def review_learning_candidates(
        self,
        *,
        max_estimated_cost_usd: Decimal | None = None,
    ) -> LearningReviewResult:
        if max_estimated_cost_usd is not None and (
            not isinstance(max_estimated_cost_usd, Decimal)
            or not max_estimated_cost_usd.is_finite()
            or max_estimated_cost_usd < 0
        ):
            raise ValueError(
                "candidate review cost ceiling must be finite and non-negative"
            )
        return await self._embedded.review_learning_candidates(
            max_estimated_cost_usd=max_estimated_cost_usd,
        )

    async def list_learning_candidates(
        self,
        *,
        status: LearningCandidateStatus | None = None,
    ) -> tuple[LearningCandidateView, ...]:
        return await self._embedded.list_learning_candidates(status=status)

    async def read_learning_candidate(
        self,
        candidate_id: str,
    ) -> LearningCandidateView | None:
        return await self._embedded.read_learning_candidate(candidate_id)

    async def edit_learning_candidate(
        self,
        candidate_id: str,
        content: LearningCandidateContent,
    ) -> LearningCandidateView:
        return await self._embedded.edit_learning_candidate(candidate_id, content)

    async def reject_learning_candidate(
        self,
        candidate_id: str,
        reason: LearningCandidateRejectionReason,
    ) -> LearningCandidateView:
        return await self._embedded.reject_learning_candidate(candidate_id, reason)

    async def accept_learning_candidate(
        self,
        candidate_id: str,
        *,
        conversation_id: str | None = None,
        source_id: str | None = None,
    ) -> LoopExit:
        return await self._embedded.accept_learning_candidate(
            candidate_id,
            conversation_id=conversation_id,
            source_id=source_id,
        )

    async def clear_rejected_learning_candidates(self) -> int:
        return await self._embedded.clear_rejected_learning_candidates()

    async def list_semantic_annotations(
        self,
        *,
        source_id: str | None = None,
        resource_id: str | None = None,
        kind: SemanticKind | None = None,
        state: SemanticAnnotationState | None = None,
    ) -> tuple[SemanticAnnotationView, ...]:
        return await self._embedded.list_semantic_annotations(
            source_id=source_id,
            resource_id=resource_id,
            kind=kind,
            state=state,
        )

    async def read_semantic_annotation(
        self,
        annotation_id: str,
    ) -> SemanticAnnotationView | None:
        return await self._embedded.read_semantic_annotation(annotation_id)

    async def save_semantic_annotation(
        self,
        annotation: SemanticAnnotation,
        *,
        expected_sha256: str | None = None,
    ) -> bool:
        return await self._embedded.save_semantic_annotation(
            annotation,
            expected_sha256=expected_sha256,
        )

    async def delete_semantic_annotation(
        self,
        annotation_id: str,
        *,
        expected_sha256: str,
    ) -> bool:
        return await self._embedded.delete_semantic_annotation(
            annotation_id,
            expected_sha256=expected_sha256,
        )

    async def list_skills(self) -> tuple[SkillSummary, ...]:
        return await self._embedded.list_skills()

    async def read_skill(self, name: str) -> Skill | None:
        return await self._embedded.read_skill(name)

    async def save_skill(
        self,
        name: str,
        description: str,
        instructions: str,
    ) -> bool:
        return await self._embedded.save_skill(name, description, instructions)

    async def delete_skill(self, name: str) -> bool:
        return await self._embedded.delete_skill(name)

    async def inspect_mcp_server(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication | None = None,
    ) -> MCPServerInspection:
        return await self._embedded.inspect_mcp_server(
            endpoint=endpoint,
            authentication=authentication,
        )

    async def attach_mcp_server(
        self,
        *,
        endpoint: str,
        selections: tuple[MCPToolSelection, ...],
        authentication: MCPAuthentication | None = None,
        maximum_outbound_sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
        local_label: str | None = None,
        binding_id: str | None = None,
    ) -> MCPBindingStatus:
        return await self._embedded.attach_mcp_server(
            endpoint=endpoint,
            selections=selections,
            authentication=authentication,
            maximum_outbound_sensitivity=maximum_outbound_sensitivity,
            local_label=local_label,
            binding_id=binding_id,
        )

    async def list_mcp_servers(self) -> tuple[MCPBindingStatus, ...]:
        return await self._embedded.list_mcp_servers()

    async def refresh_mcp_server(self, binding_id: str) -> MCPBindingStatus:
        return await self._embedded.refresh_mcp_server(binding_id)

    async def revoke_mcp_server(self, binding_id: str) -> MCPBindingStatus:
        return await self._embedded.revoke_mcp_server(binding_id)

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._embedded.attach(source)

    async def edit_source(
        self,
        source_id: str,
        source: ResourceSource,
        *,
        confirmation_handler: SourceEditConfirmationHandler,
    ) -> SourceEditResult | None:
        """Validate, review, and atomically edit one active source connection."""

        return await self._embedded.edit_source(
            source_id,
            source,
            confirmation_handler=confirmation_handler,
        )

    async def attach_sqlite(
        self,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> SourceRegistration:
        return await self._embedded.attach_sqlite(path, name=name)

    async def edit_sqlite_source(
        self,
        source_id: str,
        path: str | Path,
        *,
        confirmation_handler: SourceEditConfirmationHandler,
        name: str | None = None,
    ) -> SourceEditResult | None:
        return await self._embedded.edit_sqlite_source(
            source_id,
            path,
            confirmation_handler=confirmation_handler,
            name=name,
        )

    async def store_postgresql_password(self, password: str) -> SecretReference:
        return await self._embedded.store_postgresql_password(password)

    async def delete_postgresql_password(
        self,
        reference: SecretReference,
    ) -> None:
        await self._embedded.delete_postgresql_password(reference)

    async def probe_postgresql(
        self,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        port: int = 5432,
        ssl_mode: str = "require",
    ) -> PostgreSQLProbeResult:
        return await self._embedded.probe_postgresql(
            host=host,
            database=database,
            username=username,
            credential=credential,
            port=port,
            ssl_mode=ssl_mode,
        )

    async def attach_postgresql(
        self,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        schemas: tuple[str, ...],
        port: int = 5432,
        ssl_mode: str = "require",
        name: str | None = None,
    ) -> SourceRegistration:
        """Attach PostgreSQL with all reads and zero exact update scopes."""

        return await self._embedded.attach_postgresql(
            host=host,
            database=database,
            username=username,
            credential=credential,
            schemas=schemas,
            port=port,
            ssl_mode=ssl_mode,
            name=name,
        )

    async def edit_postgresql_source(
        self,
        source_id: str,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        schemas: tuple[str, ...],
        confirmation_handler: SourceEditConfirmationHandler,
        port: int = 5432,
        ssl_mode: str = "require",
        name: str | None = None,
    ) -> SourceEditResult | None:
        """Edit PostgreSQL with a reviewed atomic connection handoff."""

        return await self._embedded.edit_postgresql_source(
            source_id,
            host=host,
            database=database,
            username=username,
            credential=credential,
            schemas=schemas,
            confirmation_handler=confirmation_handler,
            port=port,
            ssl_mode=ssl_mode,
            name=name,
        )

    async def inspect_source_permissions(
        self,
        source_id: str,
    ) -> SourcePermissionsInspection:
        """Return exact scopes and safe complete-catalog permission choices."""

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        return await self._embedded.inspect_source_permissions(source_id)

    async def preview_source_permissions(
        self,
        *,
        source_id: str,
        read_mode: SourceReadMode | str,
        read_resource_ids: tuple[str, ...],
        postgresql_update_scopes: Mapping[str, Sequence[str]],
    ) -> SourcePermissionsPreview:
        """Preview one exact final scope state without changing durable state."""

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if isinstance(read_mode, str):
            try:
                read_mode = SourceReadMode(read_mode)
            except ValueError:
                raise ValueError("read_mode must be all, selected, or none") from None
        if not isinstance(read_mode, SourceReadMode):
            raise TypeError("read_mode must be SourceReadMode or string")
        if not isinstance(read_resource_ids, tuple) or any(
            not isinstance(resource_id, str) or not resource_id
            for resource_id in read_resource_ids
        ):
            raise TypeError("read_resource_ids must be a tuple of non-empty strings")
        if not isinstance(postgresql_update_scopes, Mapping):
            raise TypeError("postgresql_update_scopes must be a mapping")
        normalized_updates: dict[str, tuple[str, ...]] = {}
        for resource_id, columns in postgresql_update_scopes.items():
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError("update scope resource ids must be non-empty strings")
            if isinstance(columns, (str, bytes)) or not isinstance(
                columns,
                (list, tuple),
            ):
                raise TypeError("update scope columns must be lists or tuples")
            normalized = tuple(columns)
            if any(not isinstance(column, str) or not column for column in normalized):
                raise TypeError("update scope columns must be non-empty strings")
            normalized_updates[resource_id] = normalized
        return await self._embedded.preview_source_permissions(
            source_id=source_id,
            read_mode=read_mode,
            read_resource_ids=read_resource_ids,
            postgresql_update_scopes=normalized_updates,
        )

    async def apply_source_permissions(
        self,
        *,
        source_id: str,
        confirmation_fingerprint: str,
    ) -> SourcePermissionsInspection:
        """Apply one exact confirmed preview after locked current-state checks."""

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(confirmation_fingerprint, str) or not (
            confirmation_fingerprint.startswith("sha256:")
            and len(confirmation_fingerprint) == 71
        ):
            raise ValueError("confirmation_fingerprint must be a sha256 fingerprint")
        return await self._embedded.apply_source_permissions(
            source_id=source_id,
            confirmation_fingerprint=confirmation_fingerprint,
        )

    async def postgresql_update_readiness(
        self,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
    ) -> PostgreSQLUpdateReadiness:
        """Return bounded non-mutating readiness for one exact update scope."""

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource_id must be a non-empty string")
        if not isinstance(assignment_columns, tuple):
            raise TypeError("assignment_columns must be a tuple")
        return await self._embedded.postgresql_update_readiness(
            source_id,
            resource_id,
            assignment_columns,
        )

    async def detach(self, source_id: str) -> SourceRegistration:
        return await self._embedded.detach(source_id)

    async def refresh_source(self, source_id: str) -> SourceRegistration:
        """Refresh one registered source through its persisted admitted config."""

        return await self._embedded.refresh_source(source_id)

    async def list_sources(self) -> tuple[SourceRegistration, ...]:
        return await self._embedded.list_sources()

    async def list_catalog_resources(
        self, *, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        return await self._embedded.list_catalog_resources(source_id=source_id)

    async def catalog_summary(self) -> CatalogSummary:
        """Return compact facts from active current committed catalog snapshots."""

        return await self._embedded.catalog_summary()

    async def catalog_preview(
        self,
        *,
        limit: int = 12,
    ) -> tuple[CatalogResource, ...]:
        """Return a bounded deterministic preview of current catalog resources."""

        return await self._embedded.catalog_preview(limit=limit)

    async def search_catalog(
        self, request: CatalogSearchRequest
    ) -> CatalogSearchResult:
        return await self._embedded.search_catalog(request)

    async def inspect_catalog_resource(self, resource_id: str) -> FrozenJsonObject:
        return await self._embedded.inspect_catalog_resource(resource_id)

    async def close(self) -> None:
        await self._embedded.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


def _validate_artifact_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"artifact-[0-9a-f]{32}", value) is None
    ):
        raise ValueError("artifact_id must use artifact-<32 lowercase hex>")


def _validate_downloads_directory(value: Path | None) -> None:
    if value is not None and not isinstance(value, Path):
        raise TypeError("downloads_directory must be pathlib.Path or None")


__all__ = [
    "Agent",
    "AgentAlreadyExistsError",
    "AgentHomeError",
    "AgentIdentityMismatchError",
    "AgentModelConfigurationError",
    "AgentNameError",
    "AgentNotConfiguredError",
    "AgentNotFoundError",
    "HostActiveError",
    "InboxView",
    "JobExecutionMode",
    "JobInspection",
    "JobResultView",
    "JobStatus",
    "JobSummary",
    "PostgreSQLProbeResult",
    "PostgreSQLSourceError",
    "SourceRefreshError",
    "SourceSelectionError",
    "CatalogSummary",
]
