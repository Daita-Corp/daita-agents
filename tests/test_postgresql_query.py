from __future__ import annotations

import traceback
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from _capability_runtime_support import (
    StaticTestDomain,
    discovery_metadata,
    execute_projected,
    static_registry,
)

from daita.adapters import postgresql_query as postgresql_query_module
from daita.adapters.models import SourceRegistration
from daita.adapters.postgresql import PostgreSQLSourceError
from daita.capabilities import Capability, ToolExecution, ToolOutput, ToolView
from daita.capability_runtime import CapabilityRuntime
from daita.domains.data.sql import ResourceSchema
from daita.llm.models import ToolCall
from daita.loop.models import RunInput
from daita.security import EmptySecretProvider


class _SourceStore:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

    async def register_source(
        self,
        registration: SourceRegistration,
    ) -> SourceRegistration:
        self.registration = registration
        return registration

    async def load_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceRegistration | None:
        if self.registration.agent_id == agent_id and self.registration.id == source_id:
            return self.registration
        return None

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        return (self.registration,) if self.registration.agent_id == agent_id else ()

    async def detach_source(
        self,
        agent_id: str,
        source_id: str,
        detached_at: datetime,
    ) -> SourceRegistration:
        assert self.registration.agent_id == agent_id
        assert self.registration.id == source_id
        self.registration = self.registration.detach(detached_at)
        return self.registration


class _CatalogSchemas:
    def __init__(self, resources: tuple[ResourceSchema, ...]) -> None:
        self.resources = resources

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        del agent_id
        return tuple(item for item in self.resources if item.source_id == source_id)

    async def readable_resource_ids(self, agent_id: str, source_ids=()):
        del agent_id
        return frozenset(
            item.resource_id
            for item in self.resources
            if not source_ids or item.source_id in source_ids
        )


def _backend() -> tuple[
    postgresql_query_module.PostgreSQLQueryBackend,
    SourceRegistration,
]:
    agent_id = "agent-postgresql"
    registration = SourceRegistration.build(
        agent_id=agent_id,
        adapter_id="postgresql",
        native_identity="offline-query-contract",
        display_name="Offline PostgreSQL",
        configuration={},
        attached_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    source_revision = "sha256:" + "3" * 64
    resource = ResourceSchema(
        resource_id="resource-orders",
        source_id=registration.id,
        name="orders",
        aliases=("public.orders",),
        columns=("amount",),
        revision="sha256:" + "2" * 64,
        source_revision=source_revision,
        resource_kind="table",
    )
    return (
        postgresql_query_module.PostgreSQLQueryBackend(
            _SourceStore(registration),
            _CatalogSchemas((resource,)),
            EmptySecretProvider(),
        ),
        registration,
    )


@pytest.mark.parametrize("operation", ("query", "export"))
async def test_postgresql_backend_preserves_safe_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend, registration = _backend()
    diagnostic = "host=db.internal password=database-secret"

    async def failed_connect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        error = PostgreSQLSourceError(
            "postgresql_connect_failed",
            "PostgreSQL source could not be opened.",
            source_id=registration.id,
        )
        error.add_note(diagnostic)
        raise error

    monkeypatch.setattr(postgresql_query_module, "_connect", failed_connect)

    with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
        if operation == "query":
            await backend.execute_read(
                agent_id=registration.agent_id,
                source_id=registration.id,
                sql="SELECT amount FROM public.orders",
                parameters=(),
                max_rows=100,
                max_bytes=65_536,
            )
        else:
            await backend.execute_exact_tabular(
                agent_id=registration.agent_id,
                source_id=registration.id,
                sql="SELECT amount FROM public.orders",
                parameters=(),
                format_name="csv",
                parameters_sha256="sha256:" + "0" * 64,
                created_at=datetime(2026, 8, 5, tzinfo=UTC),
                max_rows=100_000,
                max_columns=256,
                max_bytes=64 * 1024 * 1024,
                timeout_seconds=60,
            )

    assert raised.value.code == "postgresql_connect_failed"
    assert str(raised.value) == "PostgreSQL source could not be opened."
    assert raised.value.__context__ is None
    assert diagnostic not in "".join(traceback.format_exception(raised.value))


async def test_postgresql_backend_preserves_credential_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, registration = _backend()

    async def failed_connect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PostgreSQLSourceError(
            "postgresql_credential_unavailable",
            "PostgreSQL credential resolution failed.",
            source_id=registration.id,
        )

    monkeypatch.setattr(postgresql_query_module, "_connect", failed_connect)

    with pytest.raises(postgresql_query_module.PostgreSQLQueryError) as raised:
        await backend.execute_read(
            agent_id=registration.agent_id,
            source_id=registration.id,
            sql="SELECT amount FROM public.orders",
            parameters=(),
            max_rows=100,
            max_bytes=65_536,
        )

    assert raised.value.code == "postgresql_credential_unavailable"
    assert str(raised.value) == "PostgreSQL credential resolution failed."


class _FailingExecutor:
    executor_id = "test.postgresql.failure.executor"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        raise self.error


async def _failure_result(error: BaseException):
    capability = Capability(
        id="test.postgresql.failure",
        description="Exercise normalized executor failures.",
        input_schema={"type": "object", "properties": {}},
        output_kind="test.postgresql.failure.output",
        output_schema={"type": "object", "properties": {}},
        executor_id=_FailingExecutor.executor_id,
    )
    view = ToolView(
        name="test_postgresql_failure",
        capability_id=capability.id,
        description=capability.description,
        discovery=discovery_metadata(),
    )
    domain = StaticTestDomain((capability,), (view,))
    runtime = CapabilityRuntime(
        static_registry(domain, (_FailingExecutor(error),)),
        (domain,),
    )
    outcome = await execute_projected(
        runtime,
        RunInput(
            id="run-postgresql-failure",
            agent_id="agent-postgresql-failure",
            message="exercise failure normalization",
            created_at=datetime(2026, 8, 19, tzinfo=UTC),
        ),
        (ToolCall(id="call-postgresql", name=view.name),),
    )
    return outcome.ordered_results[0]


async def test_postgresql_query_error_remains_structured_at_tool_result_boundary() -> (
    None
):
    result = await _failure_result(
        postgresql_query_module.PostgreSQLQueryError(
            "postgresql_connect_failed",
            "PostgreSQL source could not be opened.",
        ),
    )

    assert result.is_error is True
    error = result.output["error"]
    assert isinstance(error, Mapping)
    assert error["code"] == "postgresql_connect_failed"
    assert error["message"] == "PostgreSQL source could not be opened."
    details = error["details"]
    assert isinstance(details, Mapping)
    assert dict(details) == {}


async def test_unexpected_executor_error_is_redacted_at_tool_result_boundary() -> None:
    result = await _failure_result(
        RuntimeError("secret=/tmp/private-connection-fragment"),
    )

    assert result.is_error is True
    serialized = repr(result.output)
    assert "secret=" not in serialized
    assert "/tmp/private-connection-fragment" not in serialized
    error = result.output["error"]
    assert isinstance(error, Mapping)
    assert error["code"] == "tool_execution_failed"
    assert error["message"] == (
        "The tool could not complete because of an unexpected internal error."
    )
