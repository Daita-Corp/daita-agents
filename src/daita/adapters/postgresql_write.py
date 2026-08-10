"""Typed PostgreSQL update preview and transactional execution boundary."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
from typing import Any, Callable, Protocol
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
)
from ..domains.data.controller import (
    POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
    POSTGRESQL_UPDATE_CAPABILITY_ID,
    CatalogSchemaReader,
)
from ..domains.data.sql import (
    PostgreSQLUpdateCell,
    PostgreSQLUpdateCommand,
    PostgreSQLUpdateIntent,
    ValidatedPostgreSQLUpdate,
    render_postgresql_update_statement,
    validate_postgresql_update_intent,
)
from ..errors import PluginError
from ..security import SecretProvider, default_secret_provider
from ..storage.sqlite import (
    DatabaseWriteOutcome,
    DatabaseWriteReceipt,
    DatabaseWriteReceiptConflictError,
)
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
  ON role.rolname = pg_catalog.current_user
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
    """Validate, compile, and inspect one update without executing mutation."""

    def __init__(
        self,
        sources: SourceStore,
        catalog: CatalogSchemaReader,
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
        if not callable(getattr(catalog, "resource_schemas", None)):
            raise TypeError("catalog must provide resource_schemas")
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
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._statement_timeout_seconds = float(statement_timeout_seconds)
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)

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
        if registration.configuration.get("write_access") is not True:
            raise PostgreSQLUpdatePreviewError(
                "write_access_not_enabled",
                "PostgreSQL update preview requires user-owned write_access enablement.",
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
            timeout_milliseconds = max(
                1,
                int(self._statement_timeout_seconds * 1_000),
            )
            lock_timeout_milliseconds = max(
                1,
                int(self._lock_timeout_seconds * 1_000),
            )
            await connection.execute(
                "SELECT set_config('statement_timeout', $1, true)",
                f"{timeout_milliseconds}ms",
            )
            await connection.execute(
                "SELECT set_config('lock_timeout', $1, true)",
                f"{lock_timeout_milliseconds}ms",
            )
            await connection.execute(
                "SELECT set_config('search_path', $1, true)",
                "pg_catalog",
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
            explain_sql = (
                "EXPLAIN (FORMAT JSON, VERBOSE FALSE, COSTS FALSE) " + statement.sql
            )
            await connection.fetch(
                explain_sql,
                *bound_update_parameters,
                timeout=self._statement_timeout_seconds,
            )
            stage = "preview"
            preview_sql = _preview_select_sql(validated)
            rows = tuple(
                await connection.fetch(
                    preview_sql,
                    _bound_value(validated.match[0], validated),
                    timeout=self._statement_timeout_seconds,
                )
            )
            if len(rows) > 1:
                raise PostgreSQLUpdatePreviewError(
                    "write_guardrail_rejected",
                    "The primary-key preview exceeded the one-row bound.",
                )
            result = _build_preview(
                agent_id=agent_id,
                validated=validated,
                statement_sha256=statement.statement_sha256,
                live_structure_sha256=live_structure_sha256,
                guardrails=guardrails,
                row=(rows[0] if rows else None),
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
        """Execute one receipt-backed update and classify commit certainty exactly."""

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
        if registration.configuration.get("write_access") is not True:
            raise PostgreSQLUpdateExecutionError(
                "write_access_not_enabled",
                "PostgreSQL update requires user-owned write_access enablement.",
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
        returned_cells: tuple[PostgreSQLUpdateCell, ...] = ()
        terminal_outcome = DatabaseWriteOutcome.NOT_COMMITTED
        terminal_code: str | None = "write_not_committed"
        cancelled: asyncio.CancelledError | None = None
        try:
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
            locked_rows = tuple(
                await connection.fetch(
                    _locked_row_select_sql(validated),
                    _bound_value(validated.match[0], validated),
                    timeout=self._statement_timeout_seconds,
                )
            )
            if len(locked_rows) != 1:
                code = (
                    "write_target_not_found"
                    if not locked_rows
                    else "write_affected_rows_mismatch"
                )
                raise PostgreSQLUpdateExecutionError(
                    code,
                    "The locked primary-key target no longer matches the approved preview.",
                )
            locked_preview = _build_preview(
                agent_id=agent_id,
                validated=validated,
                statement_sha256=statement.statement_sha256,
                live_structure_sha256=live_structure_sha256,
                guardrails=guardrails,
                row=locked_rows[0],
            )
            if (
                locked_preview.would_affect != 1
                or locked_preview.fingerprint.preview_fingerprint
                != command.preview_fingerprint
            ):
                raise PostgreSQLUpdateExecutionError(
                    "write_state_changed",
                    "The target row or write guardrails changed after approval.",
                )
            returned_rows = await _fetch_update_rows(
                connection,
                statement.sql,
                _bound_update_parameters(validated),
            )
            returned_cells = _verified_returned_cells(
                returned_rows,
                validated,
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
                1
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
        return PostgreSQLUpdateResult(
            receipt_id=receipt.receipt_id,
            source_id=validated.source_id,
            resource_id=validated.resource_id,
            source_revision=validated.source_revision,
            resource_revision=validated.resource_revision,
            preview_fingerprint=command.preview_fingerprint,
            intent_sha256=validated.intent_sha256,
            returned=returned_cells,
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


def _locked_row_select_sql(validated: ValidatedPostgreSQLUpdate) -> str:
    preview = _preview_select_sql(validated)
    if not preview.endswith(" LIMIT 2"):
        raise RuntimeError("bounded PostgreSQL preview shape changed")
    return preview[: -len(" LIMIT 2")] + " LIMIT 2 FOR UPDATE"


async def _fetch_update_rows(
    connection: object,
    sql: str,
    parameters: tuple[object, ...],
) -> tuple[object, ...]:
    cursor_factory = getattr(connection, "cursor")(sql, *parameters)
    cursor = (
        await cursor_factory if inspect.isawaitable(cursor_factory) else cursor_factory
    )
    fetch = getattr(cursor, "fetch", None)
    if not callable(fetch):
        raise PostgreSQLUpdateExecutionError(
            "write_affected_rows_mismatch",
            "PostgreSQL did not provide a bounded RETURNING cursor.",
        )
    return tuple(await fetch(2))


def _verified_returned_cells(
    rows: tuple[object, ...],
    validated: ValidatedPostgreSQLUpdate,
) -> tuple[PostgreSQLUpdateCell, ...]:
    if len(rows) != 1:
        raise PostgreSQLUpdateExecutionError(
            "write_affected_rows_mismatch",
            "The generated update did not return exactly one row.",
        )
    row = rows[0]
    intended = (validated.match[0], *validated.assignments)
    returned: list[PostgreSQLUpdateCell] = []
    for cell in intended:
        type_name = validated.type_for(cell.column)[1]
        try:
            current = _preview_json_value_for_type(
                _record_value(row, cell.column),
                type_name,
            )
            expected = _preview_json_value_for_type(
                _bound_value(cell, validated),
                type_name,
            )
        except PostgreSQLUpdatePreviewError:
            raise PostgreSQLUpdateExecutionError(
                "write_affected_rows_mismatch",
                "The generated update returned invalid bounded values.",
            ) from None
        if current != expected:
            raise PostgreSQLUpdateExecutionError(
                "write_affected_rows_mismatch",
                "The generated update returned values different from the approved intent.",
            )
        returned.append(PostgreSQLUpdateCell(cell.column, current))
    return tuple(returned)


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
                1
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
    rejected = (
        relation_kind != "r"
        or facts["is_partition"] is True
        or facts["row_level_security"] is True
        or facts["force_row_level_security"] is True
        or facts["has_inheritance"] is True
        or facts["has_user_triggers"] is True
        or facts["has_rewrite_rules"] is True
        or facts["role_superuser"] is True
        or facts["role_bypass_rls"] is True
        or facts["role_create_database"] is True
        or facts["role_create_role"] is True
        or facts["role_replication"] is True
        or facts["can_connect"] is not True
        or facts["can_use_schema"] is not True
        or facts["can_select_table"] is not True
        or facts["can_update_columns"] is not True
    )
    if rejected:
        raise PostgreSQLUpdatePreviewError(
            "write_guardrail_rejected",
            "The PostgreSQL relation, role, or privileges do not satisfy preview guardrails.",
        )
    return facts


def _preview_select_sql(validated: ValidatedPostgreSQLUpdate) -> str:
    assignments = tuple(validated.assignments)
    row_expression = (
        "ROW(" + ", ".join(_identifier(item.column) for item in assignments) + ")"
    )
    size_check = (
        f"pg_catalog.pg_column_size({row_expression}) <= {_PREVIEW_VALUE_BYTES}"
    )
    selected = [f'{_identifier(validated.match[0].column)} AS "__daita_primary_key_0"']
    selected.extend(
        f"CASE WHEN {size_check} THEN {_identifier(cell.column)} ELSE NULL END "
        f'AS "__daita_before_{index}"'
        for index, cell in enumerate(assignments)
    )
    selected.extend(
        (
            f'{size_check} AS "__daita_within_preview_limit"',
            'tableoid::pg_catalog.text AS "__daita_tableoid"',
            'ctid::pg_catalog.text AS "__daita_ctid"',
            'xmin::pg_catalog.text AS "__daita_xmin"',
        )
    )
    return (
        "/* daita:postgresql.update_preview_row */ SELECT "
        + ", ".join(selected)
        + " FROM ONLY "
        + _identifier(validated.schema_name)
        + "."
        + _identifier(validated.relation_name)
        + " WHERE "
        + _identifier(validated.match[0].column)
        + " = $1 LIMIT 2"
    )


def _build_preview(
    *,
    agent_id: str,
    validated: ValidatedPostgreSQLUpdate,
    statement_sha256: str,
    live_structure_sha256: str,
    guardrails: Mapping[str, object],
    row: object | None,
) -> PostgreSQLUpdatePreview:
    would_affect = 0 if row is None else 1
    before: tuple[PostgreSQLUpdateCell, ...] = ()
    tuple_facts: dict[str, str] = {}
    live_primary_key: FrozenJsonValue | None = None
    if row is not None:
        if _record_value(row, "__daita_within_preview_limit") is not True:
            raise PostgreSQLUpdatePreviewError(
                "write_preview_failed",
                "The selected before-image exceeds the fixed preview bound.",
            )
        before = tuple(
            PostgreSQLUpdateCell(
                cell.column,
                _preview_json_value_for_type(
                    _record_value(row, f"__daita_before_{index}"),
                    validated.type_for(cell.column)[1],
                ),
            )
            for index, cell in enumerate(validated.assignments)
        )
        live_primary_key = _preview_json_value_for_type(
            _record_value(row, "__daita_primary_key_0"),
            validated.type_for(validated.match[0].column)[1],
        )
        tuple_facts = {
            "tableoid": _bounded_system_fact(row, "__daita_tableoid"),
            "ctid": _bounded_system_fact(row, "__daita_ctid"),
            "xmin": _bounded_system_fact(row, "__daita_xmin"),
        }
    row_version_sha256 = _sha256_json(
        {
            "relation": {
                "oid": guardrails["relation_oid"],
                "schema": validated.schema_name,
                "name": validated.relation_name,
            },
            "match": tuple(item.to_payload() for item in validated.match),
            "live_primary_key": live_primary_key,
            "before": tuple(item.to_payload() for item in before),
            "tuple": tuple_facts,
            "would_affect": would_affect,
        }
    )
    fingerprint_payload = {
        "agent_id": agent_id,
        "capability_id": POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
        "source_id": validated.source_id,
        "resource_id": validated.resource_id,
        "source_revision": validated.source_revision,
        "resource_revision": validated.resource_revision,
        "match": tuple(item.to_payload() for item in validated.match),
        "assignments": tuple(item.to_payload() for item in validated.assignments),
        "live_relation_identity": {
            "oid": guardrails["relation_oid"],
            "schema": validated.schema_name,
            "name": validated.relation_name,
        },
        "live_structure_sha256": live_structure_sha256,
        "row_version_sha256": row_version_sha256,
        "would_affect": would_affect,
        "guardrails": dict(guardrails),
        "statement_sha256": statement_sha256,
    }
    fingerprint = PostgreSQLPreviewFingerprint(
        intent_sha256=validated.intent_sha256,
        row_version_sha256=row_version_sha256,
        statement_sha256=statement_sha256,
        preview_fingerprint=_sha256_json(fingerprint_payload),
    )
    return PostgreSQLUpdatePreview(
        source_id=validated.source_id,
        resource_id=validated.resource_id,
        resource_name=validated.resource_name,
        source_revision=validated.source_revision,
        resource_revision=validated.resource_revision,
        match=validated.match,
        assignments=validated.assignments,
        would_affect=would_affect,
        before=before,
        after=(validated.assignments if row is not None else ()),
        fingerprint=fingerprint,
        checks=PostgreSQLUpdatePreviewChecks(),
        warnings=(() if row is not None else ("target_not_found",)),
    )


def _bound_update_parameters(
    validated: ValidatedPostgreSQLUpdate,
) -> tuple[object, ...]:
    return tuple(
        _bound_value(cell, validated)
        for cell in (*validated.assignments, *validated.match)
    )


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


def _bounded_system_fact(record: object, name: str) -> str:
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
]
