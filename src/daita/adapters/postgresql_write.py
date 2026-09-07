"""Preview and transactionally execute admitted native PostgreSQL row writes."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, Protocol, cast
from uuid import UUID

from .._json import (
    FrozenJsonObject,
    FrozenJsonValue,
    canonical_json,
    freeze_json,
    thaw_json,
)
from ..capabilities import (
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ToolExecution,
)
from ..domains.data.capabilities import (
    RelationalUpsertResult,
    RelationalPreviewFingerprint,
    RelationalUpdatePreview,
    RelationalUpdatePreviewChecks,
    RelationalUpdateResult,
    RelationalUpdateSample,
)
from ..domains.data.controller import (
    RELATIONAL_UPDATE_CAPABILITY_ID,
    RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID,
    RelationalWriteCatalogReader,
)
from ..domains.data.sql import (
    ResourceSchema,
    RelationalUpdateCell,
    RelationalUpdateCommand,
    RelationalUpdateIntent,
    ValidatedRelationalUpdate,
    render_relational_update_statement,
    validate_relational_update_intent,
    validate_relational_write_scope,
)
from ..domains.data.sql.relational_upsert import (
    RelationalUpsertIntent,
    ValidatedRelationalUpsert,
    validate_relational_upsert_intent,
    validate_relational_upsert_scope,
)
from ..domains.data.sql.relational_update import _qualified_identity
from ..llm.models import ModelSensitivity
from .models import SourceRegistration
from ..errors import DaitaError
from ..security import SecretProvider, default_secret_provider
from ..storage.sqlite_records import RelationalWriteScope, SourcePermissionStateError
from .postgresql import (
    _DEFAULT_MAX_COLUMNS,
    _DEFAULT_MAX_INDEXES,
    _DEFAULT_MAX_RELATIONSHIPS,
    _DEFAULT_MAX_RESOURCES,
    PostgreSQLSourceError,
    PostgreSQLStructure,
    _close_postgresql_connection,
    _connect,
    _load_structure,
    _rollback_postgresql_transaction,
)
from .protocols import SourceStore

_PREVIEW_VALUE_BYTES = 64 * 1_024

_READINESS_ROLE_KEYS = (
    "superuser",
    "bypass_rls",
    "create_database",
    "create_role",
    "replication",
)
_READINESS_PRIVILEGE_KEYS = (
    "database_connect",
    "schema_usage",
    "table_select",
    "requested_columns_update",
)
_READINESS_RELATION_KEYS = (
    "catalog_admitted",
    "base_table",
    "partition",
    "inheritance",
    "row_level_security",
    "force_row_level_security",
    "user_triggers",
    "rewrite_rules",
)

_WRITE_GUARDRAILS_SQL = """
/* daita:postgresql.update_preview_guardrails */
SELECT
    relation.oid::pg_catalog.text AS relation_oid,
    relation.relkind::pg_catalog.text AS relation_kind,
    relation.relispartition AS is_partition,
    relation.relrowsecurity AS row_level_security,
    relation.relforcerowsecurity AS force_row_level_security,
    EXISTS (
        SELECT 1
        FROM pg_catalog.pg_inherits AS inheritance
        WHERE inheritance.inhrelid = relation.oid
           OR inheritance.inhparent = relation.oid
    ) AS has_inheritance,
    EXISTS (
        SELECT 1
        FROM pg_catalog.pg_trigger AS trigger
        WHERE trigger.tgrelid = relation.oid
          AND NOT trigger.tgisinternal
          AND trigger.tgenabled <> 'D'
    ) AS has_user_triggers,
    EXISTS (
        SELECT 1
        FROM pg_catalog.pg_rewrite AS rewrite
        WHERE rewrite.ev_class = relation.oid
          AND rewrite.rulename <> '_RETURN'
          AND rewrite.ev_enabled <> 'D'
    ) AS has_rewrite_rules,
    role.rolsuper AS role_superuser,
    role.rolbypassrls AS role_bypass_rls,
    role.rolcreatedb AS role_create_database,
    role.rolcreaterole AS role_create_role,
    role.rolreplication AS role_replication,
    pg_catalog.has_database_privilege(
        current_user,
        pg_catalog.current_database(),
        'CONNECT'
    ) AS can_connect,
    pg_catalog.has_schema_privilege(
        current_user,
        namespace.oid,
        'USAGE'
    ) AS can_use_schema,
    pg_catalog.has_table_privilege(
        current_user,
        relation.oid,
        'SELECT'
    ) AS can_select_table,
    NOT EXISTS (
        SELECT 1
        FROM pg_catalog.unnest($3::pg_catalog.text[]) AS requested(column_name)
        WHERE NOT pg_catalog.has_column_privilege(
            current_user,
            relation.oid,
            requested.column_name,
            'UPDATE'
        )
    ) AS can_update_columns
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
JOIN pg_catalog.pg_roles AS role
  ON role.rolname = current_user
WHERE namespace.nspname = $1
  AND relation.relname = $2
LIMIT 1
"""


class RelationalUpdatePreviewError(DaitaError):
    """Stable preview failure that excludes driver and server diagnostics."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("preview error code must be non-empty text")
        if not isinstance(message, str) or not message:
            raise ValueError("preview error message must be non-empty text")
        self.code = code
        super().__init__(message, error_code=code)


class RelationalUpdateExecutionError(DaitaError):
    """Stable write failure with bounded receipt/outcome details."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
        *,
        effect_observation: EffectObservation | None = None,
    ) -> None:
        self.effect_observation = effect_observation
        self.details = FrozenJsonObject.from_mapping(details or {})
        super().__init__(message, error_code=code)


class RelationalUpdateExecutionCancelled(asyncio.CancelledError):
    def __init__(self, observation: EffectObservation) -> None:
        self.effect_observation = observation
        super().__init__("native update cancelled after bounded transaction cleanup")


@dataclass(frozen=True, slots=True)
class RelationalUpdateReadiness:
    """Bounded, secret-free readiness facts for one exact update scope."""

    source_id: str
    resource_id: str
    assignment_columns: tuple[str, ...]
    daita_scope_ready: bool
    ready_for_preview: bool
    proves_execution: bool
    role_attributes: FrozenJsonObject
    privileges: FrozenJsonObject
    relation: FrozenJsonObject
    rejection_codes: tuple[str, ...]
    remediation_categories: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "readiness source_id"),
            (self.resource_id, "readiness resource_id"),
        ):
            if not isinstance(value, str) or not value or len(value) > 1_024:
                raise ValueError(f"{name} must be bounded non-empty text")
        assignment_columns = tuple(self.assignment_columns)
        if (
            not assignment_columns
            or len(assignment_columns) != len(set(assignment_columns))
            or any(
                not isinstance(column, str) or not column or len(column) > 256
                for column in assignment_columns
            )
        ):
            raise ValueError("readiness assignment columns are invalid")
        for boolean_value, name in (
            (self.daita_scope_ready, "daita_scope_ready"),
            (self.ready_for_preview, "ready_for_preview"),
            (self.proves_execution, "proves_execution"),
        ):
            if not isinstance(boolean_value, bool):
                raise TypeError(f"readiness {name} must be a boolean")
        if self.proves_execution:
            raise ValueError("readiness can never prove a future execution")
        for facts_value, keys, name in (
            (self.role_attributes, _READINESS_ROLE_KEYS, "role_attributes"),
            (self.privileges, _READINESS_PRIVILEGE_KEYS, "privileges"),
            (self.relation, _READINESS_RELATION_KEYS, "relation"),
        ):
            if not isinstance(facts_value, FrozenJsonObject) or set(facts_value) != set(
                keys
            ):
                raise ValueError(f"readiness {name} has invalid bounded facts")
            if any(
                facts_value[key] is not None and not isinstance(facts_value[key], bool)
                for key in keys
            ):
                raise TypeError(f"readiness {name} facts must be booleans or null")
        rejection_codes = _bounded_readiness_labels(
            self.rejection_codes,
            "rejection_codes",
        )
        remediation_categories = _bounded_readiness_labels(
            self.remediation_categories,
            "remediation_categories",
        )
        if self.ready_for_preview != (not rejection_codes):
            raise ValueError("readiness status must agree with rejection codes")
        object.__setattr__(self, "assignment_columns", assignment_columns)
        object.__setattr__(self, "rejection_codes", rejection_codes)
        object.__setattr__(
            self,
            "remediation_categories",
            remediation_categories,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the one safe representation shared by API, CLI, and TUI."""

        return {
            "source_id": self.source_id,
            "resource_id": self.resource_id,
            "assignment_columns": self.assignment_columns,
            "daita_scope_ready": self.daita_scope_ready,
            "ready_for_preview": self.ready_for_preview,
            "proves_execution": self.proves_execution,
            "role_attributes": self.role_attributes.to_dict(),
            "privileges": self.privileges.to_dict(),
            "relation": self.relation.to_dict(),
            "rejection_codes": self.rejection_codes,
            "remediation_categories": self.remediation_categories,
        }


class PostgreSQLWriteBackend:
    """Own native update/upsert readiness, previews and bounded transactions."""

    def __init__(
        self,
        sources: SourceStore,
        catalog: RelationalWriteCatalogReader,
        secret_provider: SecretProvider | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        statement_timeout_seconds: float = 5.0,
        lock_timeout_seconds: float = 1.0,
        cleanup_timeout_seconds: float = 1.0,
    ) -> None:
        if not isinstance(sources, SourceStore):
            raise TypeError("sources must implement SourceStore")
        for method_name in (
            "resource_schemas",
            "relational_write_scope_issue",
        ):
            if not callable(getattr(catalog, method_name, None)):
                raise TypeError(f"catalog must provide {method_name}")
        provider = default_secret_provider(secret_provider)
        if not isinstance(provider, SecretProvider):
            raise TypeError("secret_provider must implement SecretProvider")
        for value, name, maximum in (
            (statement_timeout_seconds, "statement_timeout_seconds", 60),
            (lock_timeout_seconds, "lock_timeout_seconds", 10),
            (cleanup_timeout_seconds, "cleanup_timeout_seconds", 10),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 < float(value) <= maximum
            ):
                raise ValueError(f"{name} must be positive and at most {maximum}")
        self._sources = sources
        self._catalog = catalog
        self._secret_provider = provider
        self._clock = clock or (lambda: datetime.now(UTC))
        self._statement_timeout_seconds = float(statement_timeout_seconds)
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)

    async def _require_update_scope(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
        execution: bool,
    ) -> RelationalWriteScope:
        try:
            issue = await self._catalog.relational_write_scope_issue(
                agent_id,
                source_id,
                resource_id,
                assignment_columns,
            )
        except SourcePermissionStateError:
            issue = (
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            )
        if issue is None:
            permission = await self._catalog.load_relational_write_scope(
                agent_id, source_id, resource_id
            )
            if permission is not None and "update" in permission.allowed_operations:
                return permission
            issue = (
                "resource_write_not_allowed",
                "The exact current update permission is unavailable.",
            )
        if execution:
            raise RelationalUpdateExecutionError(issue[0], issue[1])
        raise RelationalUpdatePreviewError(issue[0], issue[1])

    async def relational_update_readiness(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
    ) -> RelationalUpdateReadiness:
        """Inspect one exact resource/column scope without granting or mutating."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("readiness agent_id must be non-empty text")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be non-empty text")
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource_id must be non-empty text")
        if not isinstance(assignment_columns, tuple):
            raise TypeError("assignment_columns must be a tuple")
        if (
            not assignment_columns
            or len(assignment_columns) != len(set(assignment_columns))
            or any(
                not isinstance(column, str)
                or not column
                or len(column) > 256
                or "\x00" in column
                for column in assignment_columns
            )
        ):
            raise ValueError("assignment_columns must contain distinct bounded names")

        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
            or registration.adapter_id != "postgresql"
        ):
            return _readiness_result(
                source_id=source_id,
                resource_id=resource_id,
                assignment_columns=assignment_columns,
                daita_scope_ready=False,
                relation={"catalog_admitted": False},
                rejection_codes=("write_source_not_available",),
                remediation_categories=("attach_active_postgresql_source",),
            )
        try:
            scope_issue = await self._catalog.relational_write_scope_issue(
                agent_id,
                source_id,
                resource_id,
                assignment_columns,
            )
        except SourcePermissionStateError:
            scope_issue = (
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            )
        daita_scope_ready = scope_issue is None
        if scope_issue is not None:
            return _readiness_result(
                source_id=source_id,
                resource_id=resource_id,
                assignment_columns=assignment_columns,
                daita_scope_ready=False,
                relation={"catalog_admitted": False},
                rejection_codes=(scope_issue[0],),
                remediation_categories=(
                    (
                        "configure_source_permissions_again"
                        if scope_issue[0] == "resource_update_scope_stale"
                        else "configure_source_permissions"
                    ),
                ),
            )
        validation = validate_relational_write_scope(
            source_id,
            resource_id,
            assignment_columns,
            resources=await self._catalog.resource_schemas(agent_id, source_id),
        )
        if not validation.valid or validation.validated is None:
            return _readiness_result(
                source_id=source_id,
                resource_id=resource_id,
                assignment_columns=assignment_columns,
                daita_scope_ready=daita_scope_ready,
                relation={"catalog_admitted": False},
                rejection_codes=validation.issue_codes,
                remediation_categories=("refresh_or_select_supported_resource",),
            )
        validated = validation.validated

        connection = None
        transaction = None
        transaction_finished = False
        facts: dict[str, object] | None = None
        rejection_codes: tuple[str, ...] = ()
        remediation_categories: tuple[str, ...] = ()
        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read",
                readonly=True,
            )
            await transaction.start()
            await _configure_readiness_transaction(
                connection,
                statement_timeout_seconds=self._statement_timeout_seconds,
                lock_timeout_seconds=self._lock_timeout_seconds,
            )
            structure = await _load_structure(
                connection,
                registration,
                max_resources=_DEFAULT_MAX_RESOURCES,
                max_columns=_DEFAULT_MAX_COLUMNS,
                max_indexes=_DEFAULT_MAX_INDEXES,
                max_relationships=_DEFAULT_MAX_RELATIONSHIPS,
            )
            if structure.source_revision != validated.source_revision:
                rejection_codes = ("write_resource_not_writable",)
                remediation_categories = ("refresh_catalog",)
            else:
                table = next(
                    (
                        item
                        for item in structure.tables
                        if item.schema == validated.schema_name
                        and item.name == validated.relation_name
                    ),
                    None,
                )
                if table is None:
                    rejection_codes = ("write_resource_not_writable",)
                    remediation_categories = ("refresh_catalog",)
                else:
                    raw = await connection.fetchrow(
                        _WRITE_GUARDRAILS_SQL,
                        validated.schema_name,
                        validated.relation_name,
                        list(validated.assignment_columns),
                        timeout=self._statement_timeout_seconds,
                    )
                    facts = _guardrail_facts(raw)
                    (
                        rejection_codes,
                        remediation_categories,
                    ) = _readiness_rejections(facts)
            await transaction.commit()
            transaction_finished = True
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except Exception:
            rejection_codes = ("write_readiness_unavailable",)
            remediation_categories = ("check_connection_and_credentials",)
        finally:
            try:
                if transaction is not None and not transaction_finished:
                    await _rollback_postgresql_transaction(
                        transaction,
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
            finally:
                if connection is not None:
                    await _close_postgresql_connection(
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )

        return _readiness_result(
            source_id=source_id,
            resource_id=resource_id,
            assignment_columns=validated.assignment_columns,
            daita_scope_ready=daita_scope_ready,
            facts=facts,
            rejection_codes=_distinct_labels(rejection_codes),
            remediation_categories=_distinct_labels(remediation_categories),
        )

    async def preview_update(
        self,
        *,
        agent_id: str,
        intent: RelationalUpdateIntent,
    ) -> RelationalUpdatePreview:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("preview agent_id must be non-empty text")
        if not isinstance(intent, RelationalUpdateIntent):
            raise TypeError("intent must be RelationalUpdateIntent")
        registration = await self._sources.load_source(agent_id, intent.source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != intent.source_id
            or not registration.active
            or registration.adapter_id != "postgresql"
        ):
            raise RelationalUpdatePreviewError(
                "write_source_not_available",
                "The selected source is not an active PostgreSQL source owned by this agent.",
            )
        permission = await self._require_update_scope(
            agent_id=agent_id,
            source_id=intent.source_id,
            resource_id=intent.resource_id,
            assignment_columns=tuple(item.column for item in intent.assignments),
            execution=False,
        )
        validation = validate_relational_update_intent(
            intent,
            resources=await self._catalog.resource_schemas(
                agent_id,
                intent.source_id,
            ),
        )
        if not validation.valid or validation.validated is None:
            issue = validation.issues[0]
            raise RelationalUpdatePreviewError(issue.code, issue.message)
        validated = validation.validated
        statement = render_relational_update_statement(validated)
        bound_update_parameters = _bound_update_parameters(validated)

        connection = None
        transaction = None
        transaction_finished = False
        normalized_failure: tuple[str, str] | None = None
        stage = "connect"
        result: RelationalUpdatePreview | None = None
        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read",
                readonly=True,
            )
            await transaction.start()
            await _configure_write_transaction(
                connection,
                statement_timeout_seconds=self._statement_timeout_seconds,
                lock_timeout_seconds=self._lock_timeout_seconds,
            )
            stage = "structure"
            structure = await _load_structure(
                connection,
                registration,
                max_resources=_DEFAULT_MAX_RESOURCES,
                max_columns=_DEFAULT_MAX_COLUMNS,
                max_indexes=_DEFAULT_MAX_INDEXES,
                max_relationships=_DEFAULT_MAX_RELATIONSHIPS,
            )
            if structure.source_revision != validated.source_revision:
                raise RelationalUpdatePreviewError(
                    "write_resource_not_writable",
                    "The live PostgreSQL structure differs from the current catalog.",
                )
            table = _exact_live_table(structure, validated)
            live_structure_sha256 = _sha256_json(table.payload())
            stage = "guardrails"
            raw_guardrails = await connection.fetchrow(
                _WRITE_GUARDRAILS_SQL,
                validated.schema_name,
                validated.relation_name,
                [item.column for item in validated.assignments],
                timeout=self._statement_timeout_seconds,
            )
            guardrails = _admitted_guardrails(raw_guardrails)
            stage = "compile"
            await connection.fetch(
                "EXPLAIN (FORMAT JSON, VERBOSE FALSE, COSTS FALSE) " + statement.sql,
                *bound_update_parameters,
                timeout=self._statement_timeout_seconds,
            )
            stage = "preview"
            scan = await _scan_target_rows(
                connection,
                _target_select_sql(validated, statement.selection_where_sql),
                _bound_where_parameters(validated),
                validated,
            )
            if scan.matched_rows > permission.max_rows:
                raise RelationalUpdatePreviewError(
                    "write_row_limit",
                    "The exact update exceeds the admitted row ceiling.",
                )
            result = _build_preview(
                permission_fingerprint=permission.authorization_fingerprint,
                agent_id=agent_id,
                validated=validated,
                statement_sha256=statement.statement_sha256,
                live_structure_sha256=live_structure_sha256,
                guardrails=guardrails,
                scan=scan,
            )
            await transaction.commit()
            transaction_finished = True
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except RelationalUpdatePreviewError:
            raise
        except PostgreSQLSourceError:
            normalized_failure = (
                "write_source_not_available",
                "PostgreSQL update preview could not open the selected source.",
            )
        except Exception as error:
            normalized_failure = _normalized_failure(stage, error)
        finally:
            try:
                if transaction is not None and not transaction_finished:
                    await _rollback_postgresql_transaction(
                        transaction,
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
            finally:
                if connection is not None:
                    await _close_postgresql_connection(
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
        if normalized_failure is not None:
            raise RelationalUpdatePreviewError(*normalized_failure)
        if result is None:
            raise RelationalUpdatePreviewError(
                "write_preview_failed",
                "PostgreSQL could not complete the bounded read-only preview.",
            )
        return result

    async def execute_update(
        self,
        *,
        agent_id: str,
        execution: ToolExecution,
        command: RelationalUpdateCommand,
    ) -> RelationalUpdateResult:
        """Execute one receipt-backed update and classify commit certainty."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("update agent_id must be non-empty text")
        if not isinstance(execution, ToolExecution):
            raise TypeError("execution must be ToolExecution")
        if execution.capability_id != RELATIONAL_UPDATE_CAPABILITY_ID:
            raise ValueError("update execution capability identity is invalid")
        if not isinstance(command, RelationalUpdateCommand):
            raise TypeError("command must be RelationalUpdateCommand")
        if execution.effect_receipt_id is None:
            raise RelationalUpdateExecutionError(
                "write_receipt_unavailable",
                "Runtime receipt reservation is required before native execution.",
            )

        try:
            intent = command.intent
            registration = await self._sources.load_source(agent_id, intent.source_id)
            if (
                registration is None
                or registration.agent_id != agent_id
                or registration.id != intent.source_id
                or not registration.active
                or registration.adapter_id != "postgresql"
            ):
                raise RelationalUpdateExecutionError(
                    "write_source_not_available",
                    "The selected source is not an active PostgreSQL source owned by this agent.",
                )
            permission = await self._require_update_scope(
                agent_id=agent_id,
                source_id=intent.source_id,
                resource_id=intent.resource_id,
                assignment_columns=tuple(item.column for item in intent.assignments),
                execution=True,
            )
            validation = validate_relational_update_intent(
                intent,
                resources=await self._catalog.resource_schemas(
                    agent_id, intent.source_id
                ),
            )
            if not validation.valid or validation.validated is None:
                issue = validation.issues[0]
                raise RelationalUpdateExecutionError(issue.code, issue.message)
            validated = validation.validated
            statement = render_relational_update_statement(validated)
        except asyncio.CancelledError as error:
            raise RelationalUpdateExecutionCancelled(
                EffectObservation(
                    EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
                )
            ) from error
        except (RelationalUpdateExecutionError, RelationalUpdatePreviewError) as error:
            raise RelationalUpdateExecutionError(
                error.error_code,
                str(error),
                effect_observation=EffectObservation(
                    EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
                ),
            ) from None
        except Exception:
            raise RelationalUpdateExecutionError(
                "write_not_dispatched",
                "Native update admission failed before write dispatch.",
                effect_observation=EffectObservation(
                    EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
                ),
            ) from None

        connection = None
        transaction = None
        transaction_finished = False
        commit_attempted = False
        mutation_attempted = False
        target_set_sha256: str | None = None
        affected_rows: int | None = None
        terminal_outcome = EffectOutcome.NOT_APPLIED
        terminal_code: str | None = "write_not_committed"
        cancelled: asyncio.CancelledError | None = None
        try:
            await self._require_update_scope(
                agent_id=agent_id,
                source_id=intent.source_id,
                resource_id=intent.resource_id,
                assignment_columns=tuple(item.column for item in intent.assignments),
                execution=True,
            )
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(isolation="repeatable_read")
            await transaction.start()
            await _configure_write_transaction(
                connection,
                statement_timeout_seconds=self._statement_timeout_seconds,
                lock_timeout_seconds=self._lock_timeout_seconds,
            )
            structure = await _load_structure(
                connection,
                registration,
                max_resources=_DEFAULT_MAX_RESOURCES,
                max_columns=_DEFAULT_MAX_COLUMNS,
                max_indexes=_DEFAULT_MAX_INDEXES,
                max_relationships=_DEFAULT_MAX_RELATIONSHIPS,
            )
            if structure.source_revision != validated.source_revision:
                raise RelationalUpdateExecutionError(
                    "write_state_changed",
                    "The live PostgreSQL structure changed after approval.",
                )
            table = _exact_live_table(structure, validated)
            live_structure_sha256 = _sha256_json(table.payload())
            raw_guardrails = await connection.fetchrow(
                _WRITE_GUARDRAILS_SQL,
                validated.schema_name,
                validated.relation_name,
                [item.column for item in validated.assignments],
                timeout=self._statement_timeout_seconds,
            )
            guardrails = _admitted_guardrails(raw_guardrails)
            scan = await _scan_target_rows(
                connection,
                _target_select_sql(
                    validated,
                    statement.selection_where_sql,
                    for_update=True,
                ),
                _bound_where_parameters(validated),
                validated,
            )
            locked_preview = _build_preview(
                permission_fingerprint=permission.authorization_fingerprint,
                agent_id=agent_id,
                validated=validated,
                statement_sha256=statement.statement_sha256,
                live_structure_sha256=live_structure_sha256,
                guardrails=guardrails,
                scan=scan,
            )
            target_set_sha256 = locked_preview.fingerprint.target_set_sha256
            if (
                locked_preview.matched_rows != command.expected_affected_rows
                or locked_preview.fingerprint.preview_fingerprint
                != command.preview_fingerprint
            ):
                raise RelationalUpdateExecutionError(
                    "write_state_changed",
                    "The exact target set or write guardrails changed after approval.",
                )
            current_permission = await self._require_update_scope(
                agent_id=agent_id,
                source_id=intent.source_id,
                resource_id=intent.resource_id,
                assignment_columns=tuple(item.column for item in intent.assignments),
                execution=True,
            )
            if current_permission != permission:
                raise RelationalUpdateExecutionError(
                    "write_state_changed", "Native permission changed before mutation."
                )
            mutation_attempted = True
            status = await connection.execute(
                statement.sql,
                *_bound_update_parameters(validated),
                timeout=self._statement_timeout_seconds,
            )
            affected_rows = _affected_rows_from_status(status)
            if affected_rows != command.expected_affected_rows:
                raise RelationalUpdateExecutionError(
                    "write_affected_rows_mismatch",
                    "PostgreSQL changed a different number of rows than the approved plan.",
                )
            commit_attempted = True
            await transaction.commit()
            transaction_finished = True
            terminal_outcome = EffectOutcome.SUCCEEDED
            terminal_code = None
        except asyncio.CancelledError as error:
            cancelled = error
            if commit_attempted:
                terminal_outcome = EffectOutcome.UNCERTAIN
                terminal_code = "write_outcome_unknown"
            else:
                terminal_outcome = EffectOutcome.NOT_APPLIED
                terminal_code = "write_not_committed"
        except (RelationalUpdateExecutionError, RelationalUpdatePreviewError) as error:
            terminal_outcome = (
                EffectOutcome.UNCERTAIN
                if commit_attempted
                else EffectOutcome.NOT_APPLIED
            )
            terminal_code = (
                "write_outcome_unknown" if commit_attempted else error.error_code
            )
        except Exception as error:
            commit_rejected = commit_attempted and isinstance(
                getattr(error, "sqlstate", None), str
            )
            terminal_outcome = (
                EffectOutcome.UNCERTAIN
                if commit_attempted and not commit_rejected
                else EffectOutcome.NOT_APPLIED
            )
            terminal_code = (
                "write_outcome_unknown"
                if terminal_outcome is EffectOutcome.UNCERTAIN
                else _normalized_update_failure(error)
            )
        finally:
            try:
                if transaction is not None and not transaction_finished:
                    try:
                        rolled_back = await _rollback_postgresql_transaction(
                            transaction,
                            connection,
                            timeout_seconds=self._cleanup_timeout_seconds,
                        )
                    except asyncio.CancelledError as error:
                        cancelled = cancelled or error
                        rolled_back = False
                    if not rolled_back and mutation_attempted:
                        terminal_outcome, terminal_code = (
                            EffectOutcome.UNCERTAIN,
                            "write_outcome_unknown",
                        )
            finally:
                if connection is not None:
                    try:
                        await _close_postgresql_connection(
                            connection, timeout_seconds=self._cleanup_timeout_seconds
                        )
                    except asyncio.CancelledError as error:
                        cancelled = cancelled or error

        completed_at = self._clock()
        observation = EffectObservation(
            terminal_outcome,
            EffectEvidenceBasis.ADAPTER_VERIFIED,
            FrozenJsonObject.from_mapping(
                {
                    "source_id": validated.source_id,
                    "resource_id": validated.resource_id,
                    "intent_sha256": validated.intent_sha256,
                    "preview_fingerprint": command.preview_fingerprint,
                    "target_set_sha256": target_set_sha256,
                    "expected_affected_rows": command.expected_affected_rows,
                    "affected_rows": (
                        affected_rows
                        if terminal_outcome is EffectOutcome.SUCCEEDED
                        else (
                            0 if terminal_outcome is EffectOutcome.NOT_APPLIED else None
                        )
                    ),
                    "normalized_error_code": terminal_code,
                }
            ),
        )
        if cancelled is not None:
            raise RelationalUpdateExecutionCancelled(observation) from cancelled
        if terminal_outcome is not EffectOutcome.SUCCEEDED:
            assert terminal_code is not None
            raise RelationalUpdateExecutionError(
                terminal_code,
                (
                    "PostgreSQL commit certainty was lost; do not retry automatically."
                    if terminal_outcome is EffectOutcome.UNCERTAIN
                    else "PostgreSQL did not commit the approved update."
                ),
                {
                    "receipt_id": execution.effect_receipt_id,
                    "outcome": terminal_outcome.value,
                    "affected_rows": (
                        0 if terminal_outcome is EffectOutcome.NOT_APPLIED else None
                    ),
                },
                effect_observation=observation,
            )
        assert affected_rows is not None
        assert target_set_sha256 is not None
        return RelationalUpdateResult(
            receipt_id=execution.effect_receipt_id,
            source_id=validated.source_id,
            resource_id=validated.resource_id,
            source_revision=validated.source_revision,
            resource_revision=validated.resource_revision,
            preview_fingerprint=command.preview_fingerprint,
            intent_sha256=validated.intent_sha256,
            target_set_sha256=target_set_sha256,
            affected_rows=affected_rows,
            committed_at=completed_at.isoformat(),
            effect_observation=observation,
        )

    async def _admit_upsert(
        self,
        agent_id: str,
        intent: RelationalUpsertIntent,
        request_sensitivity: ModelSensitivity | None = None,
    ) -> tuple[SourceRegistration, ValidatedRelationalUpsert, RelationalWriteScope]:
        registration = await self._sources.load_source(agent_id, intent.source_id)
        if (
            registration is None
            or not registration.active
            or registration.agent_id != agent_id
            or registration.adapter_id != "postgresql"
        ):
            raise CapabilityInputError(
                "write_source_not_available", "The exact native source is unavailable."
            )
        scope = await self._catalog.load_relational_write_scope(
            agent_id, intent.source_id, intent.resource_id
        )
        if scope is None:
            raise CapabilityInputError(
                "resource_write_not_allowed",
                "The exact native write permission is unavailable or stale.",
            )
        issue = await self._catalog.relational_write_scope_issue(
            agent_id,
            intent.source_id,
            intent.resource_id,
            intent.update_columns,
            operation="upsert",
            insert_columns=intent.insert_columns,
            key_columns=intent.key_columns,
            row_count=len(intent.rows),
        )
        if issue is not None:
            raise CapabilityInputError(*issue)
        resource = next(
            (
                item
                for item in await self._catalog.resource_schemas(
                    agent_id, intent.source_id
                )
                if item.resource_id == intent.resource_id
            ),
            None,
        )
        if resource is None:
            raise CapabilityInputError(
                "write_resource_not_writable", "The exact table is no longer cataloged."
            )
        if request_sensitivity is not None:
            try:
                target_sensitivity = ModelSensitivity(resource.sensitivity_class)
            except ValueError:
                raise CapabilityInputError(
                    "write_sensitivity_denied", "Target classification is unknown."
                ) from None
            if request_sensitivity.routing_rank > target_sensitivity.routing_rank:
                raise CapabilityInputError(
                    "write_sensitivity_denied",
                    "The request exceeds the target's current admitted classification.",
                )
        validated = validate_relational_upsert_intent(
            intent,
            resource=resource,
            generated_identity_columns=scope.generated_identity_columns,
            max_rows=scope.max_rows,
        )
        return registration, validated, scope

    async def upsert_readiness(
        self, agent_id: str, constraints: FrozenJsonObject
    ) -> FrozenJsonObject:
        source_id, resource_id = str(constraints["source_id"]), str(
            constraints["resource_id"]
        )
        registration = await self._sources.load_source(agent_id, source_id)
        permission = await self._catalog.load_relational_write_scope(
            agent_id, source_id, resource_id
        )
        if (
            registration is None
            or permission is None
            or "upsert" not in permission.allowed_operations
        ):
            raise CapabilityInputError(
                "resource_write_not_allowed",
                "Explicit current upsert permission is required.",
            )
        resource = next(
            (
                item
                for item in await self._catalog.resource_schemas(agent_id, source_id)
                if item.resource_id == resource_id
            ),
            None,
        )
        if resource is None:
            raise CapabilityInputError(
                "write_resource_not_writable", "The exact table is no longer admitted."
            )
        inserts = cast(tuple[str, ...], constraints["allowed_insert_columns"])
        updates = cast(tuple[str, ...], constraints["allowed_update_columns"])
        validate_relational_upsert_scope(
            resource,
            key_columns=cast(tuple[str, ...], constraints["key_columns"]),
            insert_columns=inserts,
            update_columns=updates,
            generated_identity_columns=cast(
                tuple[str, ...], constraints["generated_identity_columns"]
            ),
        )
        connection = transaction = None
        finished = False
        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read", readonly=True
            )
            await transaction.start()
            await _configure_write_transaction(
                connection,
                statement_timeout_seconds=self._statement_timeout_seconds,
                lock_timeout_seconds=self._lock_timeout_seconds,
            )
            await self._inspect_upsert_target(
                connection, registration, resource, inserts, updates
            )
            if (
                await self._catalog.load_relational_write_scope(
                    agent_id, source_id, resource_id
                )
                != permission
            ):
                raise CapabilityInputError(
                    "write_state_changed",
                    "Upsert admission changed during readiness inspection.",
                )
            await transaction.commit()
            finished = True
            return FrozenJsonObject.from_mapping(
                {
                    "source_id": source_id,
                    "resource_id": resource_id,
                    "resource_revision": resource.revision,
                    "ready_for_preview": True,
                    "proves_execution": False,
                    "lock_mode": "EXCLUSIVE",
                    "statement_timeout_seconds": self._statement_timeout_seconds,
                    "lock_timeout_seconds": self._lock_timeout_seconds,
                    "identity_sequence_gaps_possible": bool(
                        permission.generated_identity_columns
                    ),
                }
            )
        except (CapabilityInputError, asyncio.CancelledError, ImportError):
            raise
        except Exception as error:
            raise CapabilityInputError(
                _normalized_failure("inspect", error)[0],
                "The bounded native upsert readiness inspection failed.",
            ) from None
        finally:
            try:
                if transaction is not None and not finished:
                    await _rollback_postgresql_transaction(
                        transaction,
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
            finally:
                if connection is not None:
                    await _close_postgresql_connection(
                        connection, timeout_seconds=self._cleanup_timeout_seconds
                    )

    async def _inspect_upsert_target(
        self,
        connection: Any,
        registration: SourceRegistration,
        resource: ResourceSchema,
        insert_columns: tuple[str, ...],
        update_columns: tuple[str, ...],
    ) -> tuple[Any, dict[str, object]]:
        structure = await _load_structure(
            connection,
            registration,
            max_resources=_DEFAULT_MAX_RESOURCES,
            max_columns=_DEFAULT_MAX_COLUMNS,
            max_indexes=_DEFAULT_MAX_INDEXES,
            max_relationships=_DEFAULT_MAX_RELATIONSHIPS,
        )
        if structure.source_revision != resource.source_revision:
            raise CapabilityInputError(
                "write_state_changed",
                "The live structure differs from the admitted preview structure.",
            )
        schema, table_name = _qualified_identity(resource)
        table = next(
            (
                table
                for table in structure.tables
                if table.schema == schema and table.name == table_name
            ),
            None,
        )
        if table is None:
            raise CapabilityInputError(
                "write_state_changed", "The exact target table disappeared."
            )
        raw = await connection.fetchrow(
            _UPSERT_GUARDRAILS_SQL,
            schema,
            table_name,
            list(update_columns),
            list(insert_columns),
            timeout=self._statement_timeout_seconds,
        )
        guardrails = _admitted_guardrails(raw)
        if (
            not isinstance(raw, Mapping)
            or raw.get("can_insert_columns") is not True
            or raw.get("can_lock_table") is not True
            or raw.get("unsupported_insert_features") is not False
        ):
            raise CapabilityInputError(
                "upsert_guardrail_rejected",
                "The insert branch or exclusive-lock permission has unsupported target features or privileges.",
            )
        guardrails.update(
            {
                name: raw[name]
                for name in (
                    "can_insert_columns",
                    "can_lock_table",
                    "unsupported_insert_features",
                )
            }
        )
        return table, guardrails

    async def _upsert_preview_on_connection(
        self,
        connection: Any,
        registration: SourceRegistration,
        validated: ValidatedRelationalUpsert,
        scope: RelationalWriteScope,
    ) -> tuple[FrozenJsonObject, tuple[str, ...]]:
        resource, intent = validated.resource, validated.intent
        table, guardrails = await self._inspect_upsert_target(
            connection,
            registration,
            resource,
            intent.insert_columns,
            intent.update_columns,
        )
        schema, table_name = _qualified_identity(resource)
        relation = _identifier(schema) + "." + _identifier(table_name)
        types = {column: name for column, _, name in resource.column_type_provenance}
        existing_rows: list[object] = []
        classifications: list[dict[str, object]] = []
        actions: list[str] = []
        for row in validated.rows:
            parameters = tuple(
                _upsert_bound_value(row[column], types[column])
                for column in intent.key_columns
            )
            where = " AND ".join(
                f"{_identifier(column)} = ${index}"
                for index, column in enumerate(intent.key_columns, 1)
            )
            compared = tuple(intent.update_columns)
            values = tuple(
                _upsert_bound_value(row[column], types[column]) for column in compared
            )
            changed = " OR ".join(
                f"{_identifier(column)} IS DISTINCT FROM ${index}"
                for index, column in enumerate(compared, len(parameters) + 1)
            )
            columns = tuple(dict.fromkeys((*intent.key_columns, *compared)))
            row_size = (
                "pg_catalog.pg_column_size(ROW("
                + ", ".join(_identifier(column) for column in columns)
                + "))"
            )
            selected = ", ".join(
                f"CASE WHEN {row_size} <= {_PREVIEW_VALUE_BYTES} THEN {_identifier(column)} ELSE NULL END AS {_identifier(column)}"
                for column in columns
            )
            records = await connection.fetch(
                f"/* daita:postgresql.upsert_target */ SELECT {selected}, xmin::pg_catalog.text AS __daita_xmin, "
                f"({changed}) AS __daita_changed, ({row_size} <= {_PREVIEW_VALUE_BYTES}) AS __daita_bounded "
                f"FROM ONLY {relation} WHERE {where} LIMIT 2",
                *parameters,
                *values,
                timeout=self._statement_timeout_seconds,
            )
            if len(records) > 1:
                raise CapabilityInputError(
                    "upsert_key_unsupported",
                    "The conflict key matched more than one row.",
                )
            before: dict[str, object] | None = None
            if records:
                record = records[0]
                if _record_value(record, "__daita_bounded") is not True:
                    raise CapabilityInputError(
                        "write_preview_too_large",
                        "Existing values exceed the bounded preview.",
                    )
                before = {
                    column: _preview_json_value_for_type(
                        _record_value(record, column), types[column]
                    )
                    for column in columns
                }
                before["__daita_xmin"] = _bounded_row_version_fact(
                    record, "__daita_xmin"
                )
                changed_value = _record_value(record, "__daita_changed")
                if type(changed_value) is not bool:
                    raise CapabilityInputError(
                        "write_preview_failed",
                        "Database change classification is unavailable.",
                    )
                action = "update" if changed_value else "unchanged"
            else:
                action = "insert"
            existing_rows.append(before)
            actions.append(action)
            classifications.append(
                {
                    "key": {column: row[column] for column in intent.key_columns},
                    "action": action,
                }
            )
        target_digest = _sha256_json(
            {"existing": existing_rows, "classifications": classifications}
        )
        fingerprint = _sha256_json(
            {
                "agent_id": registration.agent_id,
                "intent_sha256": validated.intent_sha256,
                "target_set_sha256": target_digest,
                "resource_revision": resource.revision,
                "permission_fingerprint": scope.authorization_fingerprint,
                "structure": table.payload(),
                "guardrails": guardrails,
            }
        )
        preview = FrozenJsonObject.from_mapping(
            {
                "source_id": intent.source_id,
                "resource_id": intent.resource_id,
                "resource_revision": resource.revision,
                "intent_sha256": validated.intent_sha256,
                "preview_fingerprint": fingerprint,
                "target_set_sha256": target_digest,
                "permission_fingerprint": scope.authorization_fingerprint,
                "input_count": len(validated.rows),
                "inserted_count": actions.count("insert"),
                "updated_count": actions.count("update"),
                "unchanged_count": actions.count("unchanged"),
                "classifications": classifications,
                "evidence_call_ids": intent.evidence_call_ids,
                "authorship": "model_derived",
                "identity_sequence_gaps_possible": bool(
                    scope.generated_identity_columns
                ),
            }
        )
        if len(canonical_json(preview).encode("utf-8")) > 256 * 1024:
            raise CapabilityInputError(
                "write_preview_too_large", "The exact preview exceeds its output bound."
            )
        return preview, tuple(actions)

    async def preview_upsert(
        self, *, agent_id: str, intent: RelationalUpsertIntent
    ) -> FrozenJsonObject:
        registration, validated, scope = await self._admit_upsert(agent_id, intent)
        connection = transaction = None
        finished = False
        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read", readonly=True
            )
            await transaction.start()
            await _configure_write_transaction(
                connection,
                statement_timeout_seconds=self._statement_timeout_seconds,
                lock_timeout_seconds=self._lock_timeout_seconds,
            )
            preview, _ = await self._upsert_preview_on_connection(
                connection, registration, validated, scope
            )
            # Revocation or a changed approval while waiting for I/O invalidates this preview.
            _, current, current_scope = await self._admit_upsert(agent_id, intent)
            if (
                current_scope != scope
                or current.resource.revision != validated.resource.revision
            ):
                raise CapabilityInputError(
                    "write_state_changed", "Native admission changed during preview."
                )
            await transaction.commit()
            finished = True
            return preview
        except (CapabilityInputError, asyncio.CancelledError, ImportError):
            raise
        except Exception as error:
            raise CapabilityInputError(
                _normalized_failure("inspect", error)[0],
                "The bounded native upsert preview failed.",
            ) from None
        finally:
            try:
                if transaction is not None and not finished:
                    await _rollback_postgresql_transaction(
                        transaction,
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
            finally:
                if connection is not None:
                    await _close_postgresql_connection(
                        connection, timeout_seconds=self._cleanup_timeout_seconds
                    )

    async def execute_upsert(
        self,
        *,
        agent_id: str,
        execution: ToolExecution,
        intent: RelationalUpsertIntent,
        preview_fingerprint: str,
    ) -> RelationalUpsertResult:
        if (
            execution.capability_id != "data.upsert_rows"
            or execution.effect_receipt_id is None
        ):
            raise ValueError(
                "upsert requires the exact runtime-reserved native execution"
            )
        connection = transaction = None
        committed = commit_attempted = mutation_attempted = False
        cancelled: asyncio.CancelledError | None = None
        code: str | None = "write_not_committed"
        outcome = EffectOutcome.NOT_APPLIED
        preview: FrozenJsonObject | None = None
        validated: ValidatedRelationalUpsert | None = None
        scope: RelationalWriteScope | None = None
        inserted = updated = unchanged = 0
        generated: list[dict[str, object]] = []
        try:
            registration, validated, scope = await self._admit_upsert(
                agent_id, intent, execution.request_sensitivity
            )
            connection = await _connect(registration, self._secret_provider)
            # READ COMMITTED ensures setup queries cannot establish a stale pre-lock snapshot.
            transaction = connection.transaction(isolation="read_committed")
            await transaction.start()
            await _configure_write_transaction(
                connection,
                statement_timeout_seconds=self._statement_timeout_seconds,
                lock_timeout_seconds=self._lock_timeout_seconds,
            )
            schema, table = _qualified_identity(validated.resource)
            relation = _identifier(schema) + "." + _identifier(table)
            try:
                await connection.execute(
                    f"LOCK TABLE ONLY {relation} IN EXCLUSIVE MODE",
                    timeout=self._lock_timeout_seconds,
                )
            except TimeoutError:
                raise CapabilityInputError(
                    "write_lock_timeout",
                    "The table lock deadline elapsed before mutation.",
                ) from None
            preview, actions = await self._upsert_preview_on_connection(
                connection, registration, validated, scope
            )
            if preview["preview_fingerprint"] != preview_fingerprint:
                raise CapabilityInputError(
                    "write_state_changed",
                    "Existence, values, structure or permission changed after the exact preview.",
                )
            _, current, current_scope = await self._admit_upsert(
                agent_id, intent, execution.request_sensitivity
            )
            if (
                current_scope != scope
                or current.resource.revision != validated.resource.revision
            ):
                raise CapabilityInputError(
                    "write_state_changed", "Native permission changed before mutation."
                )
            types = {
                column: name
                for column, _, name in validated.resource.column_type_provenance
            }
            for row, action in zip(validated.rows, actions, strict=True):
                if action == "unchanged":
                    unchanged += 1
                    continue
                mutation_attempted = True
                if action == "insert":
                    columns = intent.insert_columns
                    returning = (
                        ", ".join(
                            _identifier(column)
                            for column in scope.generated_identity_columns
                        )
                        or "1 AS __daita_inserted"
                    )
                    records = await connection.fetch(
                        f"INSERT INTO {relation} ("
                        + ", ".join(_identifier(column) for column in columns)
                        + ") VALUES ("
                        + ", ".join(f"${index}" for index in range(1, len(columns) + 1))
                        + f") RETURNING {returning}",
                        *(
                            _upsert_bound_value(row[column], types[column])
                            for column in columns
                        ),
                        timeout=self._statement_timeout_seconds,
                    )
                    if len(records) != 1:
                        raise CapabilityInputError(
                            "write_affected_rows_mismatch",
                            "The insertion count differs from the exact preview.",
                        )
                    inserted += 1
                    if scope.generated_identity_columns:
                        identity_values: dict[str, object] = {}
                        for column in scope.generated_identity_columns:
                            value = _record_value(records[0], column)
                            bits = {"int2": 16, "int4": 32, "int8": 64}[types[column]]
                            if (
                                not isinstance(value, int)
                                or isinstance(value, bool)
                                or not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1)
                            ):
                                raise CapabilityInputError(
                                    "upsert_identity_invalid",
                                    "The database returned an invalid generated identity.",
                                )
                            identity_values[column] = value
                        generated.append(
                            {
                                "key": {
                                    column: row[column] for column in intent.key_columns
                                },
                                "values": identity_values,
                            }
                        )
                else:
                    columns = (*intent.update_columns, *intent.key_columns)
                    assignments = ", ".join(
                        f"{_identifier(column)} = ${index}"
                        for index, column in enumerate(intent.update_columns, 1)
                    )
                    where = " AND ".join(
                        f"{_identifier(column)} = ${index}"
                        for index, column in enumerate(
                            intent.key_columns, len(intent.update_columns) + 1
                        )
                    )
                    status = await connection.execute(
                        f"UPDATE ONLY {relation} SET {assignments} WHERE {where}",
                        *(
                            _upsert_bound_value(row[column], types[column])
                            for column in columns
                        ),
                        timeout=self._statement_timeout_seconds,
                    )
                    if _affected_rows_from_status(status) != 1:
                        raise CapabilityInputError(
                            "write_affected_rows_mismatch",
                            "The update count differs from the exact preview.",
                        )
                    updated += 1
            if (inserted, updated, unchanged) != (
                preview["inserted_count"],
                preview["updated_count"],
                preview["unchanged_count"],
            ) or inserted + updated + unchanged != len(validated.rows):
                raise CapabilityInputError(
                    "write_affected_rows_mismatch",
                    "Batch counts do not sum to the exact input count.",
                )
            commit_attempted = True
            await transaction.commit()
            committed = True
            outcome, code = EffectOutcome.SUCCEEDED, None
        except asyncio.CancelledError as error:
            cancelled = error
            outcome = (
                EffectOutcome.UNCERTAIN
                if commit_attempted
                else EffectOutcome.NOT_APPLIED
            )
            code = (
                "write_outcome_unknown" if commit_attempted else "write_not_committed"
            )
        except Exception as error:
            rejected_commit = commit_attempted and isinstance(
                getattr(error, "sqlstate", None), str
            )
            outcome = (
                EffectOutcome.UNCERTAIN
                if commit_attempted and not rejected_commit
                else EffectOutcome.NOT_APPLIED
            )
            code = (
                "write_outcome_unknown"
                if outcome is EffectOutcome.UNCERTAIN
                else (
                    error.code
                    if isinstance(error, CapabilityInputError)
                    else _normalized_update_failure(error)
                )
            )
        finally:
            try:
                if transaction is not None and not committed:
                    try:
                        rolled_back = await _rollback_postgresql_transaction(
                            transaction,
                            connection,
                            timeout_seconds=self._cleanup_timeout_seconds,
                        )
                    except asyncio.CancelledError as error:
                        cancelled = cancelled or error
                        rolled_back = False
                    if not rolled_back and mutation_attempted:
                        outcome, code = EffectOutcome.UNCERTAIN, "write_outcome_unknown"
            finally:
                if connection is not None:
                    try:
                        await _close_postgresql_connection(
                            connection, timeout_seconds=self._cleanup_timeout_seconds
                        )
                    except asyncio.CancelledError as error:
                        cancelled = cancelled or error

        if validated is None or scope is None:
            observation = EffectObservation(
                EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
            )
        else:
            observation = EffectObservation(
                outcome,
                EffectEvidenceBasis.ADAPTER_VERIFIED,
                FrozenJsonObject.from_mapping(
                    {
                        "source_id": intent.source_id,
                        "resource_id": intent.resource_id,
                        "intent_sha256": validated.intent_sha256,
                        "preview_fingerprint": preview_fingerprint,
                        "target_set_sha256": (
                            None if preview is None else preview["target_set_sha256"]
                        ),
                        "input_count": len(intent.rows),
                        "inserted_count": (
                            inserted
                            if committed
                            else (0 if outcome is EffectOutcome.NOT_APPLIED else None)
                        ),
                        "updated_count": (
                            updated
                            if committed
                            else (0 if outcome is EffectOutcome.NOT_APPLIED else None)
                        ),
                        "unchanged_count": unchanged if committed else None,
                        "normalized_error_code": code,
                        "identity_sequence_gaps_possible": bool(
                            scope.generated_identity_columns
                        ),
                    }
                ),
            )
        if cancelled is not None:
            raise RelationalUpdateExecutionCancelled(observation) from cancelled
        if not committed:
            raise RelationalUpdateExecutionError(
                code or "write_not_committed",
                "The batch did not produce a verified commit; inspect its transaction evidence. Identity sequence allocations may leave gaps.",
                effect_observation=observation,
            )
        assert preview is not None
        return RelationalUpsertResult(
            FrozenJsonObject.from_mapping(
                {
                    **preview.to_dict(),
                    "receipt_id": execution.effect_receipt_id,
                    "committed_at": self._clock().isoformat(),
                    "generated_identities": generated,
                }
            ),
            observation,
        )


async def _configure_write_transaction(
    connection: object,
    *,
    statement_timeout_seconds: float,
    lock_timeout_seconds: float,
) -> None:
    execute = getattr(connection, "execute")
    statement_milliseconds = max(1, int(statement_timeout_seconds * 1_000))
    lock_milliseconds = max(1, int(lock_timeout_seconds * 1_000))
    await execute(
        "SELECT set_config('statement_timeout', $1, true)",
        f"{statement_milliseconds}ms",
    )
    await execute(
        "SELECT set_config('lock_timeout', $1, true)",
        f"{lock_milliseconds}ms",
    )
    await execute(
        "SELECT set_config('idle_in_transaction_session_timeout', $1, true)",
        f"{statement_milliseconds}ms",
    )
    await execute(
        "SELECT set_config('search_path', $1, true)",
        "pg_catalog",
    )


@dataclass(frozen=True, slots=True)
class _TargetScan:
    matched_rows: int
    target_set_sha256: str
    samples: tuple[RelationalUpdateSample, ...]


def _target_select_sql(
    validated: ValidatedRelationalUpdate,
    where_sql: str,
    *,
    for_update: bool = False,
) -> str:
    assignments = tuple(validated.assignments)
    row_expression = (
        "ROW(" + ", ".join(_identifier(item.column) for item in assignments) + ")"
    )
    size_check = (
        f"pg_catalog.pg_column_size({row_expression}) <= {_PREVIEW_VALUE_BYTES}"
    )
    selected = [
        f'{_identifier(column)} AS "__daita_primary_key_{index}"'
        for index, column in enumerate(validated.primary_key_columns)
    ]
    selected.extend(
        f"CASE WHEN {size_check} THEN {_identifier(cell.column)} ELSE NULL END "
        f'AS "__daita_before_{index}"'
        for index, cell in enumerate(assignments)
    )
    selected.extend(
        (
            f'{size_check} AS "__daita_within_preview_limit"',
            'xmin::pg_catalog.text AS "__daita_xmin"',
        )
    )
    order_by = ", ".join(
        _identifier(column) for column in validated.primary_key_columns
    )
    suffix = " FOR UPDATE" if for_update else ""
    return (
        "/* daita:postgresql.update_target_set */ SELECT "
        + ", ".join(selected)
        + " FROM ONLY "
        + _identifier(validated.schema_name)
        + "."
        + _identifier(validated.relation_name)
        + " WHERE "
        + where_sql
        + " ORDER BY "
        + order_by
        + suffix
    )


async def _scan_target_rows(
    connection: object,
    sql: str,
    parameters: tuple[object, ...],
    validated: ValidatedRelationalUpdate,
) -> _TargetScan:
    cursor_factory = getattr(connection, "cursor")(sql, *parameters)
    cursor = (
        await cursor_factory if inspect.isawaitable(cursor_factory) else cursor_factory
    )
    digest = sha256()
    matched_rows = 0
    samples: list[RelationalUpdateSample] = []

    async def accept(row: object) -> None:
        nonlocal matched_rows
        primary_key = tuple(
            RelationalUpdateCell(
                column,
                _preview_json_value_for_type(
                    _record_value(row, f"__daita_primary_key_{index}"),
                    validated.type_for(column)[1],
                ),
            )
            for index, column in enumerate(validated.primary_key_columns)
        )
        within_preview_limit = (
            _record_value(row, "__daita_within_preview_limit") is True
        )
        before = (
            tuple(
                RelationalUpdateCell(
                    cell.column,
                    _preview_json_value_for_type(
                        _record_value(row, f"__daita_before_{index}"),
                        validated.type_for(cell.column)[1],
                    ),
                )
                for index, cell in enumerate(validated.assignments)
            )
            if within_preview_limit
            else ()
        )
        assigned_state: object = (
            tuple(item.to_payload() for item in before)
            if within_preview_limit
            else {
                "oversized_row_version": _bounded_row_version_fact(row, "__daita_xmin")
            }
        )
        digest.update(
            canonical_json(
                {
                    "primary_key": tuple(item.to_payload() for item in primary_key),
                    "assigned_values": assigned_state,
                }
            ).encode("utf-8")
        )
        digest.update(b"\n")
        matched_rows += 1
        if len(samples) >= 5 or not within_preview_limit:
            return
        samples.append(
            RelationalUpdateSample(
                primary_key=primary_key,
                before=before,
                after=validated.assignments,
            )
        )

    iterator = getattr(cursor, "__aiter__", None)
    if callable(iterator):
        async for row in cursor:
            await accept(row)
    else:
        fetch = getattr(cursor, "fetch", None)
        if not callable(fetch):
            raise RelationalUpdatePreviewError(
                "write_preview_failed",
                "PostgreSQL did not provide a streaming target cursor.",
            )
        while True:
            batch = fetch(256)
            if not inspect.isawaitable(batch):
                raise RelationalUpdatePreviewError(
                    "write_preview_failed",
                    "PostgreSQL did not provide an asynchronous target cursor.",
                )
            rows = tuple(await batch)
            if not rows:
                break
            for row in rows:
                await accept(row)
            if len(rows) < 256:
                break

    return _TargetScan(
        matched_rows=matched_rows,
        target_set_sha256="sha256:" + digest.hexdigest(),
        samples=tuple(samples),
    )


def _affected_rows_from_status(value: object) -> int:
    if not isinstance(value, str):
        raise RelationalUpdateExecutionError(
            "write_affected_rows_mismatch",
            "PostgreSQL returned an invalid update status.",
        )
    prefix, separator, count = value.rpartition(" ")
    if separator != " " or prefix != "UPDATE" or not count.isdecimal():
        raise RelationalUpdateExecutionError(
            "write_affected_rows_mismatch",
            "PostgreSQL returned an invalid update status.",
        )
    return int(count)


def _normalized_update_failure(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "write_statement_timeout"
    sqlstate = getattr(error, "sqlstate", None)
    if isinstance(sqlstate, str) and sqlstate.startswith("23"):
        return "write_constraint_violation"
    if sqlstate == "42501":
        return "write_permission_denied"
    if sqlstate == "55P03":
        return "write_lock_timeout"
    if sqlstate == "57014":
        return "write_statement_timeout"
    return "write_not_committed"


def _exact_live_table(
    structure: PostgreSQLStructure,
    validated: ValidatedRelationalUpdate,
) -> Any:
    table = next(
        (
            item
            for item in structure.tables
            if item.schema == validated.schema_name
            and item.name == validated.relation_name
        ),
        None,
    )
    if table is None or table.kind.value != "table":
        raise RelationalUpdatePreviewError(
            "write_resource_not_writable",
            "The exact cataloged PostgreSQL base table is no longer current.",
        )
    return table


def _admitted_guardrails(value: object) -> dict[str, object]:
    facts = _guardrail_facts(value)
    rejection_codes, _remediation = _readiness_rejections(facts)
    if rejection_codes:
        raise RelationalUpdatePreviewError(
            "write_guardrail_rejected",
            "The PostgreSQL relation, role, or privileges do not satisfy preview guardrails.",
        )
    return facts


def _guardrail_facts(value: object) -> dict[str, object]:
    if value is None:
        raise RelationalUpdatePreviewError(
            "write_resource_not_writable",
            "The exact PostgreSQL relation is no longer available.",
        )
    facts = {
        "relation_oid": _record_value(value, "relation_oid"),
        "relation_kind": _record_value(value, "relation_kind"),
        "is_partition": _record_value(value, "is_partition"),
        "row_level_security": _record_value(value, "row_level_security"),
        "force_row_level_security": _record_value(value, "force_row_level_security"),
        "has_inheritance": _record_value(value, "has_inheritance"),
        "has_user_triggers": _record_value(value, "has_user_triggers"),
        "has_rewrite_rules": _record_value(value, "has_rewrite_rules"),
        "role_superuser": _record_value(value, "role_superuser"),
        "role_bypass_rls": _record_value(value, "role_bypass_rls"),
        "role_create_database": _record_value(value, "role_create_database"),
        "role_create_role": _record_value(value, "role_create_role"),
        "role_replication": _record_value(value, "role_replication"),
        "can_connect": _record_value(value, "can_connect"),
        "can_use_schema": _record_value(value, "can_use_schema"),
        "can_select_table": _record_value(value, "can_select_table"),
        "can_update_columns": _record_value(value, "can_update_columns"),
    }
    relation_oid = facts["relation_oid"]
    relation_kind = facts["relation_kind"]
    boolean_names = tuple(
        name for name in facts if name not in {"relation_oid", "relation_kind"}
    )
    if (
        not isinstance(relation_oid, str)
        or not relation_oid
        or len(relation_oid) > 32
        or not isinstance(relation_kind, str)
        or len(relation_kind) > 8
        or any(not isinstance(facts[name], bool) for name in boolean_names)
    ):
        raise RelationalUpdatePreviewError(
            "write_guardrail_rejected",
            "PostgreSQL returned invalid bounded write-readiness facts.",
        )
    return facts


def _readiness_rejections(
    facts: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    checks = (
        (
            facts["relation_kind"] != "r",
            "write_relation_not_base_table",
            "select_supported_base_table",
        ),
        (
            facts["is_partition"] is True,
            "write_relation_partitioned",
            "select_supported_base_table",
        ),
        (
            facts["has_inheritance"] is True,
            "write_relation_inherited",
            "select_supported_base_table",
        ),
        (
            facts["row_level_security"] is True,
            "write_relation_rls_enabled",
            "select_relation_without_rls",
        ),
        (
            facts["force_row_level_security"] is True,
            "write_relation_force_rls",
            "select_relation_without_rls",
        ),
        (
            facts["has_user_triggers"] is True,
            "write_relation_user_triggers",
            "select_relation_without_user_triggers",
        ),
        (
            facts["has_rewrite_rules"] is True,
            "write_relation_rewrite_rules",
            "select_relation_without_rewrite_rules",
        ),
        (
            facts["role_superuser"] is True,
            "write_role_superuser",
            "use_least_privileged_role",
        ),
        (
            facts["role_bypass_rls"] is True,
            "write_role_bypass_rls",
            "use_least_privileged_role",
        ),
        (
            facts["role_create_database"] is True,
            "write_role_create_database",
            "use_least_privileged_role",
        ),
        (
            facts["role_create_role"] is True,
            "write_role_create_role",
            "use_least_privileged_role",
        ),
        (
            facts["role_replication"] is True,
            "write_role_replication",
            "use_least_privileged_role",
        ),
        (
            facts["can_connect"] is not True,
            "write_privilege_connect_missing",
            "grant_connect_externally",
        ),
        (
            facts["can_use_schema"] is not True,
            "write_privilege_schema_usage_missing",
            "grant_schema_usage_externally",
        ),
        (
            facts["can_select_table"] is not True,
            "write_privilege_table_select_missing",
            "grant_table_select_externally",
        ),
        (
            facts["can_update_columns"] is not True,
            "write_privilege_column_update_missing",
            "grant_column_update_externally",
        ),
    )
    return (
        tuple(code for rejected, code, _category in checks if rejected),
        _distinct_labels(
            tuple(category for rejected, _code, category in checks if rejected)
        ),
    )


async def _configure_readiness_transaction(
    connection: object,
    *,
    statement_timeout_seconds: float,
    lock_timeout_seconds: float,
) -> None:
    execute = getattr(connection, "execute")
    statement_milliseconds = max(1, int(statement_timeout_seconds * 1_000))
    lock_milliseconds = max(1, int(lock_timeout_seconds * 1_000))
    await execute(
        "SELECT set_config('statement_timeout', $1, true)",
        f"{statement_milliseconds}ms",
    )
    await execute(
        "SELECT set_config('lock_timeout', $1, true)",
        f"{lock_milliseconds}ms",
    )
    await execute(
        "SELECT set_config('search_path', $1, true)",
        "pg_catalog",
    )


def _readiness_result(
    *,
    source_id: str,
    resource_id: str,
    assignment_columns: tuple[str, ...],
    daita_scope_ready: bool,
    facts: Mapping[str, object] | None = None,
    relation: Mapping[str, object] | None = None,
    rejection_codes: tuple[str, ...],
    remediation_categories: tuple[str, ...],
) -> RelationalUpdateReadiness:
    role_attributes: dict[str, object] = {}
    privileges: dict[str, object] = {}
    relation_facts: dict[str, object] = {}
    for key in _READINESS_ROLE_KEYS:
        role_attributes[key] = None
    for key in _READINESS_PRIVILEGE_KEYS:
        privileges[key] = None
    for key in _READINESS_RELATION_KEYS:
        relation_facts[key] = None
    relation_facts["catalog_admitted"] = True
    if facts is not None:
        role_attributes.update(
            {
                "superuser": facts["role_superuser"],
                "bypass_rls": facts["role_bypass_rls"],
                "create_database": facts["role_create_database"],
                "create_role": facts["role_create_role"],
                "replication": facts["role_replication"],
            }
        )
        privileges.update(
            {
                "database_connect": facts["can_connect"],
                "schema_usage": facts["can_use_schema"],
                "table_select": facts["can_select_table"],
                "requested_columns_update": facts["can_update_columns"],
            }
        )
        relation_facts.update(
            {
                "base_table": facts["relation_kind"] == "r",
                "partition": facts["is_partition"],
                "inheritance": facts["has_inheritance"],
                "row_level_security": facts["row_level_security"],
                "force_row_level_security": facts["force_row_level_security"],
                "user_triggers": facts["has_user_triggers"],
                "rewrite_rules": facts["has_rewrite_rules"],
            }
        )
    if relation is not None:
        unknown = set(relation) - set(_READINESS_RELATION_KEYS)
        if unknown:
            raise ValueError("readiness relation override is invalid")
        relation_facts.update(relation)
    return RelationalUpdateReadiness(
        source_id=source_id,
        resource_id=resource_id,
        assignment_columns=assignment_columns,
        daita_scope_ready=daita_scope_ready,
        ready_for_preview=not rejection_codes,
        proves_execution=False,
        role_attributes=FrozenJsonObject.from_mapping(role_attributes),
        privileges=FrozenJsonObject.from_mapping(privileges),
        relation=FrozenJsonObject.from_mapping(relation_facts),
        rejection_codes=rejection_codes,
        remediation_categories=remediation_categories,
    )


def _bounded_readiness_labels(
    values: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if (
        len(normalized) > 32
        or len(normalized) != len(set(normalized))
        or any(
            not isinstance(value, str)
            or not value
            or len(value) > 128
            or "\x00" in value
            for value in normalized
        )
    ):
        raise ValueError(f"readiness {name} must contain bounded distinct labels")
    return normalized


def _distinct_labels(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _build_preview(
    *,
    agent_id: str,
    permission_fingerprint: str,
    validated: ValidatedRelationalUpdate,
    statement_sha256: str,
    live_structure_sha256: str,
    guardrails: Mapping[str, object],
    scan: _TargetScan,
) -> RelationalUpdatePreview:
    fingerprint_payload = {
        "permission_fingerprint": permission_fingerprint,
        "agent_id": agent_id,
        "capability_id": RELATIONAL_UPDATE_PREVIEW_CAPABILITY_ID,
        "source_id": validated.source_id,
        "resource_id": validated.resource_id,
        "source_revision": validated.source_revision,
        "resource_revision": validated.resource_revision,
        "where": tuple(item.to_payload() for item in validated.where),
        "assignments": tuple(item.to_payload() for item in validated.assignments),
        "live_relation_identity": {
            "oid": guardrails["relation_oid"],
            "schema": validated.schema_name,
            "name": validated.relation_name,
        },
        "live_structure_sha256": live_structure_sha256,
        "target_set_sha256": scan.target_set_sha256,
        "matched_rows": scan.matched_rows,
        "guardrails": dict(guardrails),
        "statement_sha256": statement_sha256,
    }
    warnings: list[str] = []
    if scan.matched_rows == 0:
        warnings.append("target_not_found")
    if len(scan.samples) < min(scan.matched_rows, 5):
        warnings.append("oversized_sample_values_omitted")
    return RelationalUpdatePreview(
        source_id=validated.source_id,
        resource_id=validated.resource_id,
        resource_name=validated.resource_name,
        source_revision=validated.source_revision,
        resource_revision=validated.resource_revision,
        where=validated.where,
        assignments=validated.assignments,
        matched_rows=scan.matched_rows,
        samples=scan.samples,
        fingerprint=RelationalPreviewFingerprint(
            intent_sha256=validated.intent_sha256,
            target_set_sha256=scan.target_set_sha256,
            statement_sha256=statement_sha256,
            preview_fingerprint=_sha256_json(fingerprint_payload),
        ),
        checks=RelationalUpdatePreviewChecks(),
        warnings=tuple(warnings),
    )


def _bound_update_parameters(
    validated: ValidatedRelationalUpdate,
) -> tuple[object, ...]:
    return (
        *tuple(_bound_value(cell, validated) for cell in validated.assignments),
        *_bound_where_parameters(validated),
    )


def _bound_where_parameters(
    validated: ValidatedRelationalUpdate,
) -> tuple[object, ...]:
    parameters: list[object] = []
    for predicate in validated.where:
        if predicate.operator in {"is_null", "is_not_null"}:
            continue
        values = (
            predicate.value
            if predicate.operator in {"in", "not_in"}
            else (predicate.value,)
        )
        assert isinstance(values, tuple)
        parameters.extend(
            _bound_value(
                RelationalUpdateCell(predicate.column, value),
                validated,
            )
            for value in values
        )
    return tuple(parameters)


def _bound_value(
    cell: RelationalUpdateCell,
    validated: ValidatedRelationalUpdate,
) -> object:
    value = thaw_json(cell.value)
    if value is None:
        return None
    namespace, type_name = validated.type_for(cell.column)
    if namespace != "pg_catalog":
        raise RelationalUpdatePreviewError(
            "write_assignment_invalid",
            "The proposed value lacks admitted PostgreSQL type provenance.",
        )
    if type_name == "numeric":
        return Decimal(str(value))
    if type_name in {"float4", "float8"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RelationalUpdatePreviewError(
                "write_assignment_invalid",
                "The proposed value is incompatible with its PostgreSQL float type.",
            )
        return float(value)
    if type_name == "uuid":
        return UUID(str(value))
    if type_name == "date":
        return date.fromisoformat(str(value))
    if type_name in {"timestamp", "timestamptz"}:
        text = str(value)
        return datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    if type_name in {"json", "jsonb"}:
        return canonical_json(value)
    return value


def _preview_json_value(value: object) -> FrozenJsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise RelationalUpdatePreviewError(
            "write_preview_failed",
            "PostgreSQL returned an unsupported preview value.",
        )
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise RelationalUpdatePreviewError(
                "write_preview_failed",
                "PostgreSQL returned an unsupported preview value.",
            )
        return freeze_json({"type": "decimal", "value": str(value)})
    if isinstance(value, (datetime, date, UUID)):
        return value.isoformat() if isinstance(value, (datetime, date)) else str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RelationalUpdatePreviewError(
                "write_preview_failed",
                "PostgreSQL returned an unsupported preview value.",
            )
        return freeze_json(
            {
                key: _preview_json_value(item)
                for key, item in value.items()
                if isinstance(key, str)
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_preview_json_value(item) for item in value)
    raise RelationalUpdatePreviewError(
        "write_preview_failed",
        "PostgreSQL returned an unsupported preview value.",
    )


def _preview_json_value_for_type(
    value: object,
    type_name: str,
) -> FrozenJsonValue:
    if type_name in {"json", "jsonb"} and isinstance(value, str):
        try:
            return freeze_json(json.loads(value))
        except (TypeError, ValueError):
            raise RelationalUpdatePreviewError(
                "write_preview_failed",
                "PostgreSQL returned an invalid JSON preview value.",
            ) from None
    return _preview_json_value(value)


def _record_value(record: object, name: str) -> object:
    missing = object()
    if isinstance(record, Mapping):
        value = record.get(name, missing)
    else:
        getter = getattr(record, "get", None)
        value = getter(name, missing) if callable(getter) else missing
    if value is missing:
        raise RelationalUpdatePreviewError(
            "write_guardrail_rejected",
            "PostgreSQL returned incomplete bounded preview facts.",
        )
    return value


def _bounded_row_version_fact(record: object, name: str) -> str:
    value = _record_value(record, name)
    rendered = str(value)
    if not rendered or len(rendered) > 128:
        raise RelationalUpdatePreviewError(
            "write_guardrail_rejected",
            "PostgreSQL returned an invalid row-version fact.",
        )
    return rendered


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 256:
        raise RelationalUpdatePreviewError(
            "write_resource_not_writable",
            "The cataloged PostgreSQL identifier is invalid.",
        )
    return '"' + value.replace('"', '""') + '"'


def _sha256_json(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_failure(stage: str, error: BaseException) -> tuple[str, str]:
    sqlstate = getattr(error, "sqlstate", None)
    if sqlstate == "42501":
        return (
            "write_permission_denied",
            "PostgreSQL denied a privilege required for the read-only preview.",
        )
    if sqlstate == "55P03":
        return (
            "write_lock_timeout",
            "PostgreSQL update preview exceeded its fixed lock timeout.",
        )
    if sqlstate == "57014":
        return (
            "write_statement_timeout",
            "PostgreSQL update preview exceeded its fixed statement timeout.",
        )
    if stage == "compile":
        return (
            "write_compile_failed",
            "PostgreSQL could not compile the generated parameterized update shape.",
        )
    return (
        "write_preview_failed",
        "PostgreSQL could not complete the bounded read-only preview.",
    )


__all__ = [
    "RelationalUpdateExecutionError",
    "PostgreSQLWriteBackend",
    "RelationalUpdatePreviewError",
    "RelationalUpdateReadiness",
]


_UPSERT_GUARDRAILS_SQL = _WRITE_GUARDRAILS_SQL.replace(
    " AS can_update_columns",
    """ AS can_update_columns,
    pg_catalog.has_table_privilege(current_user, relation.oid, 'UPDATE') AS can_lock_table,
    NOT EXISTS (SELECT 1 FROM pg_catalog.unnest($4::pg_catalog.text[]) AS requested(column_name)
        WHERE NOT pg_catalog.has_column_privilege(current_user, relation.oid, requested.column_name, 'INSERT')) AS can_insert_columns,
    (EXISTS (SELECT 1 FROM pg_catalog.pg_constraint AS con WHERE con.conrelid = relation.oid
                 AND (con.contype IN ('c', 'x', 'f') OR con.condeferrable))
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_constraint AS con
          WHERE con.confrelid = relation.oid AND con.contype = 'f')
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_index AS idx
          JOIN pg_catalog.pg_class AS ic ON ic.oid = idx.indexrelid
          JOIN pg_catalog.pg_am AS am ON am.oid = ic.relam
          CROSS JOIN LATERAL pg_catalog.unnest(idx.indclass) AS cls(oid)
          JOIN pg_catalog.pg_opclass AS opc ON opc.oid = cls.oid
          JOIN pg_catalog.pg_namespace AS ns ON ns.oid = opc.opcnamespace
          WHERE idx.indrelid = relation.oid AND (idx.indexprs IS NOT NULL OR idx.indpred IS NOT NULL
            OR NOT idx.indisvalid OR NOT idx.indisready OR NOT idx.indimmediate
            OR am.amname <> 'btree' OR NOT opc.opcdefault OR ns.nspname <> 'pg_catalog')))
        AS unsupported_insert_features""",
)


def _upsert_bound_value(value: object, type_name: str) -> object:
    if value is None:
        return None
    if type_name == "numeric":
        return Decimal(str(value))
    if type_name in {"float4", "float8"}:
        return float(str(value))
    if type_name == "uuid":
        return UUID(str(value))
    if type_name == "date":
        return date.fromisoformat(str(value))
    if type_name in {"timestamp", "timestamptz"}:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return value
