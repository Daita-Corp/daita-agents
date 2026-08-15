"""Structured PostgreSQL update preview and transactional execution boundary."""

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
from typing import Any, Protocol
from uuid import UUID

from .._json import (
    FrozenJsonObject,
    FrozenJsonValue,
    canonical_json,
    freeze_json,
    thaw_json,
)
from ..capabilities import ToolExecution
from ..domains.data.capabilities import (
    PostgreSQLPreviewFingerprint,
    PostgreSQLUpdatePreview,
    PostgreSQLUpdatePreviewChecks,
    PostgreSQLUpdateResult,
    PostgreSQLUpdateSample,
)
from ..domains.data.controller import (
    POSTGRESQL_UPDATE_CAPABILITY_ID,
    POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
    PostgreSQLUpdateCatalogReader,
)
from ..domains.data.sql import (
    PostgreSQLUpdateCell,
    PostgreSQLUpdateCommand,
    PostgreSQLUpdateFilter,
    PostgreSQLUpdateIntent,
    ValidatedPostgreSQLUpdate,
    render_postgresql_update_statement,
    validate_postgresql_update_intent,
    validate_postgresql_update_scope,
)
from ..errors import PluginError
from ..security import SecretProvider, default_secret_provider
from ..storage.sqlite import (
    DatabaseWriteOutcome,
    DatabaseWriteReceipt,
    DatabaseWriteReceiptConflictError,
)
from ..storage.sqlite_records import SourcePermissionStateError
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


class PostgreSQLUpdatePreviewError(PluginError):
    """Stable preview failure that excludes driver and server diagnostics."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("preview error code must be non-empty text")
        if not isinstance(message, str) or not message:
            raise ValueError("preview error message must be non-empty text")
        self.code = code
        super().__init__(message, plugin_id="postgresql", error_code=code)


class PostgreSQLUpdateExecutionError(PluginError):
    """Stable write failure with bounded receipt/outcome details."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.details = FrozenJsonObject.from_mapping(details or {})
        super().__init__(message, plugin_id="postgresql", error_code=code)


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateReadiness:
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


class DatabaseWriteReceiptStore(Protocol):
    async def load_database_write_receipt_for_call(
        self, agent_id: str, run_id: str, call_id: str
    ) -> DatabaseWriteReceipt | None: ...

    async def start_database_write_receipt(
        self, receipt: DatabaseWriteReceipt
    ) -> DatabaseWriteReceipt: ...

    async def finish_database_write_receipt(
        self, receipt: DatabaseWriteReceipt
    ) -> DatabaseWriteReceipt: ...


class PostgreSQLUpdatePreviewBackend:
    """Validate, compile, and inspect one update plan without mutating."""

    def __init__(
        self,
        sources: SourceStore,
        catalog: PostgreSQLUpdateCatalogReader,
        secret_provider: SecretProvider | None = None,
        *,
        receipt_store: DatabaseWriteReceiptStore | None = None,
        clock: Callable[[], datetime] | None = None,
        statement_timeout_seconds: float = 5.0,
        lock_timeout_seconds: float = 1.0,
        cleanup_timeout_seconds: float = 1.0,
    ) -> None:
        if not isinstance(sources, SourceStore):
            raise TypeError("sources must implement SourceStore")
        for method_name in (
            "resource_schemas",
            "postgresql_update_scope_issue",
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
        if receipt_store is not None and not all(
            callable(getattr(receipt_store, name, None))
            for name in (
                "load_database_write_receipt_for_call",
                "start_database_write_receipt",
                "finish_database_write_receipt",
            )
        ):
            raise TypeError("receipt_store must provide the database receipt contract")
        self._receipt_store = receipt_store
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
    ) -> None:
        try:
            issue = await self._catalog.postgresql_update_scope_issue(
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
            return
        if execution:
            raise PostgreSQLUpdateExecutionError(issue[0], issue[1])
        raise PostgreSQLUpdatePreviewError(issue[0], issue[1])

    async def postgresql_update_readiness(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
    ) -> PostgreSQLUpdateReadiness:
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
            scope_issue = await self._catalog.postgresql_update_scope_issue(
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
        validation = validate_postgresql_update_scope(
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
        intent: PostgreSQLUpdateIntent,
    ) -> PostgreSQLUpdatePreview:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("preview agent_id must be non-empty text")
        if not isinstance(intent, PostgreSQLUpdateIntent):
            raise TypeError("intent must be PostgreSQLUpdateIntent")
        registration = await self._sources.load_source(agent_id, intent.source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != intent.source_id
            or not registration.active
            or registration.adapter_id != "postgresql"
        ):
            raise PostgreSQLUpdatePreviewError(
                "write_source_not_available",
                "The selected source is not an active PostgreSQL source owned by this agent.",
            )
        await self._require_update_scope(
            agent_id=agent_id,
            source_id=intent.source_id,
            resource_id=intent.resource_id,
            assignment_columns=tuple(item.column for item in intent.assignments),
            execution=False,
        )
        validation = validate_postgresql_update_intent(
            intent,
            resources=await self._catalog.resource_schemas(
                agent_id,
                intent.source_id,
            ),
        )
        if not validation.valid or validation.validated is None:
            issue = validation.issues[0]
            raise PostgreSQLUpdatePreviewError(issue.code, issue.message)
        validated = validation.validated
        statement = render_postgresql_update_statement(validated)
        bound_update_parameters = _bound_update_parameters(validated)

        connection = None
        transaction = None
        transaction_finished = False
        normalized_failure: tuple[str, str] | None = None
        stage = "connect"
        result: PostgreSQLUpdatePreview | None = None
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
                raise PostgreSQLUpdatePreviewError(
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
            result = _build_preview(
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
        except PostgreSQLUpdatePreviewError:
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
            raise PostgreSQLUpdatePreviewError(*normalized_failure)
        if result is None:
            raise PostgreSQLUpdatePreviewError(
                "write_preview_failed",
                "PostgreSQL could not complete the bounded read-only preview.",
            )
        return result

    async def execute_update(
        self,
        *,
        agent_id: str,
        execution: ToolExecution,
        command: PostgreSQLUpdateCommand,
    ) -> PostgreSQLUpdateResult:
        """Execute one receipt-backed update and classify commit certainty."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("update agent_id must be non-empty text")
        if not isinstance(execution, ToolExecution):
            raise TypeError("execution must be ToolExecution")
        if execution.capability_id != POSTGRESQL_UPDATE_CAPABILITY_ID:
            raise ValueError("update execution capability identity is invalid")
        if not isinstance(command, PostgreSQLUpdateCommand):
            raise TypeError("command must be PostgreSQLUpdateCommand")
        if self._receipt_store is None:
            raise PostgreSQLUpdateExecutionError(
                "write_receipt_unavailable",
                "Durable database write receipts are unavailable.",
            )

        intent = command.intent
        registration = await self._sources.load_source(agent_id, intent.source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != intent.source_id
            or not registration.active
            or registration.adapter_id != "postgresql"
        ):
            raise PostgreSQLUpdateExecutionError(
                "write_source_not_available",
                "The selected source is not an active PostgreSQL source owned by this agent.",
            )
        await self._require_update_scope(
            agent_id=agent_id,
            source_id=intent.source_id,
            resource_id=intent.resource_id,
            assignment_columns=tuple(item.column for item in intent.assignments),
            execution=True,
        )
        validation = validate_postgresql_update_intent(
            intent,
            resources=await self._catalog.resource_schemas(agent_id, intent.source_id),
        )
        if not validation.valid or validation.validated is None:
            issue = validation.issues[0]
            raise PostgreSQLUpdateExecutionError(issue.code, issue.message)
        validated = validation.validated
        statement = render_postgresql_update_statement(validated)
        receipt = DatabaseWriteReceipt.start(
            agent_id=agent_id,
            run_id=execution.run_id,
            call_id=execution.call_id,
            capability_id=execution.capability_id,
            source_id=validated.source_id,
            resource_id=validated.resource_id,
            intent_sha256=validated.intent_sha256,
            preview_fingerprint=command.preview_fingerprint,
            expected_affected_rows=command.expected_affected_rows,
            started_at=self._clock(),
        )
        try:
            existing = await self._receipt_store.load_database_write_receipt_for_call(
                agent_id,
                execution.run_id,
                execution.call_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise PostgreSQLUpdateExecutionError(
                "write_receipt_unavailable",
                "The durable receipt identity could not be checked.",
            ) from None
        if existing is not None:
            _raise_duplicate_receipt(existing, receipt)
        try:
            await self._receipt_store.start_database_write_receipt(receipt)
        except DatabaseWriteReceiptConflictError:
            current = await self._receipt_store.load_database_write_receipt_for_call(
                agent_id,
                execution.run_id,
                execution.call_id,
            )
            if current is not None:
                _raise_duplicate_receipt(current, receipt)
            raise PostgreSQLUpdateExecutionError(
                "write_receipt_unavailable",
                "The durable started receipt could not be established.",
            ) from None
        except asyncio.CancelledError:
            raise
        except Exception:
            raise PostgreSQLUpdateExecutionError(
                "write_receipt_unavailable",
                "The durable started receipt could not be established.",
            ) from None

        connection = None
        transaction = None
        transaction_finished = False
        commit_attempted = False
        target_set_sha256: str | None = None
        affected_rows: int | None = None
        terminal_outcome = DatabaseWriteOutcome.NOT_COMMITTED
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
                raise PostgreSQLUpdateExecutionError(
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
                raise PostgreSQLUpdateExecutionError(
                    "write_state_changed",
                    "The exact target set or write guardrails changed after approval.",
                )
            status = await connection.execute(
                statement.sql,
                *_bound_update_parameters(validated),
                timeout=self._statement_timeout_seconds,
            )
            affected_rows = _affected_rows_from_status(status)
            if affected_rows != command.expected_affected_rows:
                raise PostgreSQLUpdateExecutionError(
                    "write_affected_rows_mismatch",
                    "PostgreSQL changed a different number of rows than the approved plan.",
                )
            commit_attempted = True
            await transaction.commit()
            transaction_finished = True
            terminal_outcome = DatabaseWriteOutcome.COMMITTED
            terminal_code = None
        except asyncio.CancelledError as error:
            cancelled = error
            if commit_attempted:
                terminal_outcome = DatabaseWriteOutcome.OUTCOME_UNKNOWN
                terminal_code = "write_outcome_unknown"
            else:
                terminal_outcome = DatabaseWriteOutcome.NOT_COMMITTED
                terminal_code = "write_not_committed"
        except (PostgreSQLUpdateExecutionError, PostgreSQLUpdatePreviewError) as error:
            terminal_outcome = (
                DatabaseWriteOutcome.OUTCOME_UNKNOWN
                if commit_attempted
                else DatabaseWriteOutcome.NOT_COMMITTED
            )
            terminal_code = (
                "write_outcome_unknown" if commit_attempted else error.error_code
            )
        except Exception as error:
            commit_rejected = commit_attempted and isinstance(
                getattr(error, "sqlstate", None), str
            )
            terminal_outcome = (
                DatabaseWriteOutcome.OUTCOME_UNKNOWN
                if commit_attempted and not commit_rejected
                else DatabaseWriteOutcome.NOT_COMMITTED
            )
            terminal_code = (
                "write_outcome_unknown"
                if terminal_outcome is DatabaseWriteOutcome.OUTCOME_UNKNOWN
                else _normalized_update_failure(error)
            )
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

        completed_at = self._clock()
        terminal = receipt.finish(
            terminal_outcome,
            completed_at=completed_at,
            affected_rows=(
                affected_rows
                if terminal_outcome is DatabaseWriteOutcome.COMMITTED
                else (
                    0
                    if terminal_outcome is DatabaseWriteOutcome.NOT_COMMITTED
                    else None
                )
            ),
            normalized_error_code=terminal_code,
        )
        try:
            await self._receipt_store.finish_database_write_receipt(terminal)
        except asyncio.CancelledError:
            raise
        except Exception:
            code = (
                "write_outcome_unknown"
                if terminal_outcome
                in {
                    DatabaseWriteOutcome.COMMITTED,
                    DatabaseWriteOutcome.OUTCOME_UNKNOWN,
                }
                else "write_receipt_unavailable"
            )
            raise PostgreSQLUpdateExecutionError(
                code,
                "The terminal database write receipt could not be established.",
                {
                    "receipt_id": receipt.receipt_id,
                    "outcome": (
                        "outcome_unknown"
                        if code == "write_outcome_unknown"
                        else "not_committed"
                    ),
                    "affected_rows": None if code == "write_outcome_unknown" else 0,
                },
            ) from None
        if cancelled is not None:
            raise cancelled
        if terminal_outcome is not DatabaseWriteOutcome.COMMITTED:
            assert terminal_code is not None
            raise PostgreSQLUpdateExecutionError(
                terminal_code,
                (
                    "PostgreSQL commit certainty was lost; do not retry automatically."
                    if terminal_outcome is DatabaseWriteOutcome.OUTCOME_UNKNOWN
                    else "PostgreSQL did not commit the approved update."
                ),
                {
                    "receipt_id": receipt.receipt_id,
                    "outcome": terminal_outcome.value,
                    "affected_rows": (
                        0
                        if terminal_outcome is DatabaseWriteOutcome.NOT_COMMITTED
                        else None
                    ),
                },
            )
        assert affected_rows is not None
        assert target_set_sha256 is not None
        return PostgreSQLUpdateResult(
            receipt_id=receipt.receipt_id,
            source_id=validated.source_id,
            resource_id=validated.resource_id,
            source_revision=validated.source_revision,
            resource_revision=validated.resource_revision,
            preview_fingerprint=command.preview_fingerprint,
            intent_sha256=validated.intent_sha256,
            target_set_sha256=target_set_sha256,
            affected_rows=affected_rows,
            committed_at=completed_at.isoformat(),
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
    samples: tuple[PostgreSQLUpdateSample, ...]


def _target_select_sql(
    validated: ValidatedPostgreSQLUpdate,
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
    validated: ValidatedPostgreSQLUpdate,
) -> _TargetScan:
    cursor_factory = getattr(connection, "cursor")(sql, *parameters)
    cursor = (
        await cursor_factory if inspect.isawaitable(cursor_factory) else cursor_factory
    )
    digest = sha256()
    matched_rows = 0
    samples: list[PostgreSQLUpdateSample] = []

    async def accept(row: object) -> None:
        nonlocal matched_rows
        primary_key = tuple(
            PostgreSQLUpdateCell(
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
                PostgreSQLUpdateCell(
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
            PostgreSQLUpdateSample(
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
            raise PostgreSQLUpdatePreviewError(
                "write_preview_failed",
                "PostgreSQL did not provide a streaming target cursor.",
            )
        while True:
            batch = fetch(256)
            if not inspect.isawaitable(batch):
                raise PostgreSQLUpdatePreviewError(
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
        raise PostgreSQLUpdateExecutionError(
            "write_affected_rows_mismatch",
            "PostgreSQL returned an invalid update status.",
        )
    prefix, separator, count = value.rpartition(" ")
    if separator != " " or prefix != "UPDATE" or not count.isdecimal():
        raise PostgreSQLUpdateExecutionError(
            "write_affected_rows_mismatch",
            "PostgreSQL returned an invalid update status.",
        )
    return int(count)


def _raise_duplicate_receipt(
    existing: DatabaseWriteReceipt,
    proposed: DatabaseWriteReceipt,
) -> None:
    same_identity = (
        existing.agent_id == proposed.agent_id
        and existing.run_id == proposed.run_id
        and existing.call_id == proposed.call_id
        and existing.capability_id == proposed.capability_id
        and existing.source_id == proposed.source_id
        and existing.resource_id == proposed.resource_id
        and existing.intent_sha256 == proposed.intent_sha256
        and existing.preview_fingerprint == proposed.preview_fingerprint
        and existing.expected_affected_rows == proposed.expected_affected_rows
    )
    if not same_identity:
        raise PostgreSQLUpdateExecutionError(
            "write_receipt_integrity_error",
            "The run and call identity conflicts with a different database write intent.",
        )
    outcome = (
        "outcome_unknown"
        if existing.outcome is DatabaseWriteOutcome.STARTED
        else existing.outcome.value
    )
    raise PostgreSQLUpdateExecutionError(
        (
            "write_outcome_unknown"
            if outcome == "outcome_unknown"
            else "write_execution_duplicate"
        ),
        "This exact run and call identity already has a durable write receipt; it was not executed again.",
        {
            "receipt_id": existing.receipt_id,
            "outcome": outcome,
            "affected_rows": (
                existing.affected_rows
                if outcome == "committed"
                else 0 if outcome == "not_committed" else None
            ),
        },
    )


def _normalized_update_failure(error: BaseException) -> str:
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
    validated: ValidatedPostgreSQLUpdate,
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
        raise PostgreSQLUpdatePreviewError(
            "write_resource_not_writable",
            "The exact cataloged PostgreSQL base table is no longer current.",
        )
    return table


def _admitted_guardrails(value: object) -> dict[str, object]:
    facts = _guardrail_facts(value)
    rejection_codes, _remediation = _readiness_rejections(facts)
    if rejection_codes:
        raise PostgreSQLUpdatePreviewError(
            "write_guardrail_rejected",
            "The PostgreSQL relation, role, or privileges do not satisfy preview guardrails.",
        )
    return facts


def _guardrail_facts(value: object) -> dict[str, object]:
    if value is None:
        raise PostgreSQLUpdatePreviewError(
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
        raise PostgreSQLUpdatePreviewError(
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
) -> PostgreSQLUpdateReadiness:
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
    return PostgreSQLUpdateReadiness(
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
    validated: ValidatedPostgreSQLUpdate,
    statement_sha256: str,
    live_structure_sha256: str,
    guardrails: Mapping[str, object],
    scan: _TargetScan,
) -> PostgreSQLUpdatePreview:
    fingerprint_payload = {
        "agent_id": agent_id,
        "capability_id": POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
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
    return PostgreSQLUpdatePreview(
        source_id=validated.source_id,
        resource_id=validated.resource_id,
        resource_name=validated.resource_name,
        source_revision=validated.source_revision,
        resource_revision=validated.resource_revision,
        where=validated.where,
        assignments=validated.assignments,
        matched_rows=scan.matched_rows,
        samples=scan.samples,
        fingerprint=PostgreSQLPreviewFingerprint(
            intent_sha256=validated.intent_sha256,
            target_set_sha256=scan.target_set_sha256,
            statement_sha256=statement_sha256,
            preview_fingerprint=_sha256_json(fingerprint_payload),
        ),
        checks=PostgreSQLUpdatePreviewChecks(),
        warnings=tuple(warnings),
    )


def _bound_update_parameters(
    validated: ValidatedPostgreSQLUpdate,
) -> tuple[object, ...]:
    return (
        *tuple(_bound_value(cell, validated) for cell in validated.assignments),
        *_bound_where_parameters(validated),
    )


def _bound_where_parameters(
    validated: ValidatedPostgreSQLUpdate,
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
                PostgreSQLUpdateCell(predicate.column, value),
                validated,
            )
            for value in values
        )
    return tuple(parameters)


def _bound_value(
    cell: PostgreSQLUpdateCell,
    validated: ValidatedPostgreSQLUpdate,
) -> object:
    value = thaw_json(cell.value)
    if value is None:
        return None
    namespace, type_name = validated.type_for(cell.column)
    if namespace != "pg_catalog":
        raise PostgreSQLUpdatePreviewError(
            "write_assignment_invalid",
            "The proposed value lacks admitted PostgreSQL type provenance.",
        )
    if type_name == "numeric":
        return Decimal(str(value))
    if type_name in {"float4", "float8"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PostgreSQLUpdatePreviewError(
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
        raise PostgreSQLUpdatePreviewError(
            "write_preview_failed",
            "PostgreSQL returned an unsupported preview value.",
        )
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise PostgreSQLUpdatePreviewError(
                "write_preview_failed",
                "PostgreSQL returned an unsupported preview value.",
            )
        return freeze_json({"type": "decimal", "value": str(value)})
    if isinstance(value, (datetime, date, UUID)):
        return value.isoformat() if isinstance(value, (datetime, date)) else str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PostgreSQLUpdatePreviewError(
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
    raise PostgreSQLUpdatePreviewError(
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
            raise PostgreSQLUpdatePreviewError(
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
        raise PostgreSQLUpdatePreviewError(
            "write_guardrail_rejected",
            "PostgreSQL returned incomplete bounded preview facts.",
        )
    return value


def _bounded_row_version_fact(record: object, name: str) -> str:
    value = _record_value(record, name)
    rendered = str(value)
    if not rendered or len(rendered) > 128:
        raise PostgreSQLUpdatePreviewError(
            "write_guardrail_rejected",
            "PostgreSQL returned an invalid row-version fact.",
        )
    return rendered


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 256:
        raise PostgreSQLUpdatePreviewError(
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
    "DatabaseWriteReceiptStore",
    "PostgreSQLUpdateExecutionError",
    "PostgreSQLUpdatePreviewBackend",
    "PostgreSQLUpdatePreviewError",
    "PostgreSQLUpdateReadiness",
]
