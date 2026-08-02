"""Focused public API for the MVP data agent."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import re
from typing import Self

from ._json import FrozenJsonObject
from .artifacts.models import (
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactPayload,
    ArtifactRef,
)
from .adapters.models import SourceRegistration
from .adapters.postgresql import (
    PostgreSQLProbeResult,
    PostgreSQLSourceError,
)
from .adapters.protocols import ResourceSource
from .catalog.models import (
    CatalogResource,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSummary,
)
from .capabilities import ApprovalHandler
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
    SourceSelectionError,
)
from .llm.models import ModelProfile
from .llm.protocols import ModelProvider
from .llm.routing import ModelRoute
from .learning_candidates import (
    LearningCandidateContent,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    LearningCandidateView,
    LearningReviewResult,
)
from .loop.driver import ContextBuilder, ToolRuntime
from .loop.models import ConversationRun, LoopExit, LoopLimits, Transcript
from .observation import AgentObserver
from .security import KeychainStore, SecretProvider, SecretReference
from .semantics import (
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticAnnotationView,
    SemanticKind,
)
from .skills import Skill, SkillSummary


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
        root: str | Path | None = None,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        tools: ToolRuntime | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        keychain: KeychainStore | None = None,
        model_validator: ModelProvider | None = None,
        reviewer_model: ModelProvider | None = None,
        reviewer_profile: ModelProfile | None = None,
        reviewer_max_estimated_cost_usd: Decimal | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        downloads_directory: Path | None = None,
    ) -> Self:
        _validate_downloads_directory(downloads_directory)
        return cls(
            await EmbeddedAgent.create(
                name,
                root=root,
                config=config,
                model=model,
                model_profile=model_profile,
                context_builder=context_builder,
                tools=tools,
                limits=limits,
                clock=clock,
                id_factory=id_factory,
                secret_provider=secret_provider,
                keychain=keychain,
                model_validator=model_validator,
                reviewer_model=reviewer_model,
                reviewer_profile=reviewer_profile,
                reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
                observer=observer,
                approval_handler=approval_handler,
                downloads_directory=downloads_directory,
            )
        )

    @classmethod
    async def open(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        tools: ToolRuntime | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        keychain: KeychainStore | None = None,
        model_validator: ModelProvider | None = None,
        reviewer_model: ModelProvider | None = None,
        reviewer_profile: ModelProfile | None = None,
        reviewer_max_estimated_cost_usd: Decimal | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        downloads_directory: Path | None = None,
    ) -> Self:
        _validate_downloads_directory(downloads_directory)
        return cls(
            await EmbeddedAgent.open(
                name,
                root=root,
                config=config,
                model=model,
                model_profile=model_profile,
                context_builder=context_builder,
                tools=tools,
                limits=limits,
                clock=clock,
                id_factory=id_factory,
                secret_provider=secret_provider,
                keychain=keychain,
                model_validator=model_validator,
                reviewer_model=reviewer_model,
                reviewer_profile=reviewer_profile,
                reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
                observer=observer,
                approval_handler=approval_handler,
                downloads_directory=downloads_directory,
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

    async def configure_model(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelRoute:
        """Validate and persist one model route for the next open."""

        credential = api_key
        api_key = None
        try:
            return await self._embedded.configure_model(
                provider=provider,
                model=model,
                api_key=credential,
                base_url=base_url,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
            )
        finally:
            credential = None

    async def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_id: str | None = None,
    ) -> LoopExit:
        return await self._embedded.run(
            message,
            conversation_id=conversation_id,
            source_id=source_id,
        )

    async def transcript(self, run_id: str) -> Transcript:
        return await self._embedded.transcript(run_id)

    async def conversation_runs(
        self,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        return await self._embedded.conversation_runs(conversation_id)

    async def conversation_exists(self, conversation_id: str) -> bool:
        """Return whether one conversation ID belongs to this agent."""

        return await self._embedded.conversation_exists(conversation_id)

    async def clear_conversations(self) -> int:
        """Delete transcripts and candidate records, not approved knowledge."""

        return await self._embedded.clear_conversations()

    async def list_artifacts(
        self,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        for value, name in (
            (run_id, "run_id"),
            (conversation_id, "conversation_id"),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > 256
            ):
                raise ValueError(f"{name} must be bounded non-empty text or None")
        return await self._embedded.list_artifacts(
            run_id=run_id,
            conversation_id=conversation_id,
        )

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

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._embedded.attach(source)

    async def attach_sqlite(
        self,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> SourceRegistration:
        return await self._embedded.attach_sqlite(path, name=name)

    async def attach_local_directory(
        self,
        root: str | Path,
        *,
        name: str | None = None,
    ) -> SourceRegistration:
        return await self._embedded.attach_local_directory(root, name=name)

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
    "PostgreSQLProbeResult",
    "PostgreSQLSourceError",
    "SourceSelectionError",
    "CatalogSummary",
]
