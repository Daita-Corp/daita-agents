from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from daita._json import FrozenJsonObject
from daita.adapters import postgresql_write as native
from daita.capabilities import CapabilityInputError, EffectOutcome, ToolExecution
from daita.catalog.models import ResourceKind, TabularColumn, TabularIndex
from daita.domains.data.sql import ResourceSchema
from daita.domains.data.sql.relational_upsert import (
    RelationalUpsertIntent,
    validate_relational_upsert_intent,
)
from daita.llm.models import ModelSensitivity
from daita.security import EmptySecretProvider
from daita.storage.sqlite_records import RelationalWriteScope
from test_relational_update_preview import (
    _registration,
    _SourceStore,
    _guardrails,
    NOW,
    SOURCE_ID,
    RESOURCE_ID,
    SOURCE_REVISION,
    RESOURCE_REVISION,
)


def resource() -> ResourceSchema:
    return ResourceSchema(
        source_id=SOURCE_ID,
        resource_id=RESOURCE_ID,
        name="companies",
        aliases=("public.companies",),
        columns=("id", "domain", "name", "evidence_url", "notes"),
        revision=RESOURCE_REVISION,
        source_revision=SOURCE_REVISION,
        resource_kind="table",
        sensitivity_class="restricted",
        writable=True,
        primary_key_columns=("id",),
        conflict_keys=(("id",), ("domain",)),
        column_nullability=(
            ("id", False),
            ("domain", False),
            ("name", False),
            ("evidence_url", False),
            ("notes", True),
        ),
        column_type_provenance=(
            ("id", "pg_catalog", "int8"),
            ("domain", "pg_catalog", "text"),
            ("name", "pg_catalog", "text"),
            ("evidence_url", "pg_catalog", "text"),
            ("notes", "pg_catalog", "text"),
        ),
        column_defaults=tuple(
            (name, None) for name in ("id", "domain", "name", "evidence_url", "notes")
        ),
        column_collations=(("domain", "C"),),
        identity_columns=("id",),
        updatable_columns=("name", "evidence_url", "notes"),
    )


def permission() -> RelationalWriteScope:
    return RelationalWriteScope(
        agent_id="agent-preview",
        source_id=SOURCE_ID,
        resource_id=RESOURCE_ID,
        resource_revision=RESOURCE_REVISION,
        allowed_operations=("upsert",),
        allowed_insert_columns=("domain", "name", "evidence_url", "notes"),
        allowed_update_columns=("name", "evidence_url", "notes"),
        key_columns=("domain",),
        generated_identity_columns=("id",),
        max_rows=10,
        authorization_fingerprint="sha256:" + "5" * 64,
    )


def intent(rows=None, **changes) -> RelationalUpsertIntent:
    values = {
        "source_id": SOURCE_ID,
        "resource_id": RESOURCE_ID,
        "key_columns": ("domain",),
        "insert_columns": ("domain", "name", "evidence_url"),
        "update_columns": ("name", "evidence_url"),
        "rows": (
            rows
            if rows is not None
            else (
                {
                    "domain": "new.test",
                    "name": "New",
                    "evidence_url": "https://source.test/new",
                },
            )
        ),
    }
    return RelationalUpsertIntent.from_mapping({**values, **changes})


class Catalog:
    def __init__(self):
        self.resource = resource()
        self.permission = permission()
        self.revoked = False

    async def resource_schemas(self, agent_id, source_id):
        return (self.resource,)

    async def load_relational_write_scope(self, agent_id, source_id, resource_id):
        return None if self.revoked else self.permission

    async def relational_write_scope_issue(
        self, agent_id, source_id, resource_id, assignment_columns, **kwargs
    ):
        if self.revoked:
            return ("resource_write_not_allowed", "Revoked")
        if kwargs.get("operation", "update") not in self.permission.allowed_operations:
            return ("resource_write_not_allowed", "Operation not granted")
        if not set(assignment_columns) <= set(self.permission.allowed_update_columns):
            return ("update_column_not_allowed", "Column not granted")
        if not set(kwargs.get("insert_columns", ())) <= set(
            self.permission.allowed_insert_columns
        ):
            return ("insert_column_not_allowed", "Column not granted")
        return None


class Database:
    def __init__(self):
        self.rows = {}
        self.next_id = 1
        self.log = []
        self.before_lock = None
        self.lock_error = None
        self.mutation_error = None
        self.commit_error = None
        self.rollback_error = None
        self.bad_count = False
        self.guardrails = {
            **_guardrails(),
            "can_insert_columns": True,
            "can_lock_table": True,
            "unsupported_insert_features": False,
        }
        self.structure_revision = SOURCE_REVISION

    def connect(self):
        return Connection(self)


class Connection:
    def __init__(self, db):
        self.db = db
        self.work = None
        self.readonly = True
        self.locked = False

    def transaction(self, **options):
        self.readonly = options.get("readonly", False)
        self.db.log.append(("transaction", options))
        return self

    async def start(self):
        self.work = deepcopy(self.db.rows)

    async def commit(self):
        self.db.log.append(("commit", self.readonly))
        if not self.readonly:
            self.db.rows = deepcopy(self.work)
            if self.db.commit_error:
                raise self.db.commit_error

    async def rollback(self):
        self.db.log.append(("rollback",))
        if self.db.rollback_error:
            raise self.db.rollback_error
        self.work = None

    async def close(self):
        self.db.log.append(("close",))

    def terminate(self):
        self.db.log.append(("terminate",))

    async def fetchrow(self, sql, *parameters, **kwargs):
        self.db.log.append(("guardrails", self.locked))
        return self.db.guardrails

    async def execute(self, sql, *parameters, **kwargs):
        self.db.log.append(("execute", sql, parameters))
        if sql.startswith("LOCK TABLE"):
            if self.db.lock_error:
                raise self.db.lock_error
            if self.db.before_lock:
                self.db.before_lock()
            self.work = deepcopy(self.db.rows)
            self.locked = True
        elif sql.startswith("UPDATE"):
            assert self.locked
            if self.db.mutation_error:
                raise self.db.mutation_error
            if self.db.bad_count:
                return "UPDATE 0"
            # Canonical columns are evidence_url, name, then conflict domain.
            evidence, name, domain = parameters
            assert self.work is not None
            self.work[domain].update(evidence_url=evidence, name=name, __daita_xmin="2")
            return "UPDATE 1"
        return "SELECT 1"

    async def fetch(self, sql, *parameters, **kwargs):
        self.db.log.append(("fetch", sql, parameters, self.locked))
        if "upsert_target" in sql:
            if not self.readonly:
                assert self.locked
            domain, evidence, name = parameters
            assert self.work is not None
            existing = self.work.get(domain)
            if existing is None:
                return ()
            return (
                {
                    **existing,
                    "__daita_bounded": True,
                    "__daita_changed": existing["name"] != name
                    or existing["evidence_url"] != evidence,
                },
            )
        assert sql.startswith("INSERT") and self.locked
        if self.db.mutation_error:
            raise self.db.mutation_error
        allocated = self.db.next_id
        self.db.next_id += 1  # Sequence allocations survive rollback.
        domain, evidence, name = parameters
        assert self.work is not None
        self.work[domain] = {
            "id": allocated,
            "domain": domain,
            "evidence_url": evidence,
            "name": name,
            "notes": None,
            "__daita_xmin": "1",
        }
        return () if self.db.bad_count else ({"id": allocated},)


@pytest.fixture
def setup(monkeypatch):
    db, catalog = Database(), Catalog()

    async def connect(*args, **kwargs):
        return db.connect()

    async def load_structure(*args, **kwargs):
        table = SimpleNamespace(
            schema="public",
            name="companies",
            kind=ResourceKind.TABLE,
            payload=lambda: {
                "schema": "public",
                "name": "companies",
                "revision": db.structure_revision,
            },
        )
        return SimpleNamespace(tables=(table,), source_revision=db.structure_revision)

    monkeypatch.setattr(native, "_connect", connect)
    monkeypatch.setattr(native, "_load_structure", load_structure)
    backend = native.PostgreSQLWriteBackend(
        _SourceStore(_registration()), catalog, EmptySecretProvider(), clock=lambda: NOW
    )
    return backend, db, catalog


async def execute(backend, batch, preview):
    execution = ToolExecution(
        run_id="run-upsert",
        call_id="write",
        capability_id="data.upsert_rows",
        request_sensitivity=ModelSensitivity.RESTRICTED,
        effect_receipt_id="effect-receipt:sha256:" + "9" * 64,
    )
    return await backend.execute_upsert(
        agent_id="agent-preview",
        execution=execution,
        intent=batch,
        preview_fingerprint=preview["preview_fingerprint"],
    )


def row(
    domain="existing.test", name="Existing", evidence="https://source.test/existing"
):
    return {
        "id": 42,
        "domain": domain,
        "name": name,
        "evidence_url": evidence,
        "notes": "keep omitted note",
        "__daita_xmin": "1",
    }


async def test_mixed_batch_locks_before_scan_and_commits_exact_counts(setup):
    backend, db, _ = setup
    db.rows["existing.test"] = row()
    db.rows["same.test"] = row("same.test", "Same", "https://source.test/same")
    batch = intent(
        (
            {
                "domain": "existing.test",
                "name": "Updated",
                "evidence_url": "https://source.test/update",
            },
            {
                "domain": "new.test",
                "name": "New",
                "evidence_url": "https://source.test/new",
            },
            {
                "domain": "same.test",
                "name": "Same",
                "evidence_url": "https://source.test/same",
            },
        )
    )
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)
    assert (
        preview["inserted_count"],
        preview["updated_count"],
        preview["unchanged_count"],
    ) == (1, 1, 1)
    assert db.next_id == 1
    result = await execute(backend, batch, preview)
    assert result.effect_observation.outcome is EffectOutcome.SUCCEEDED
    assert (
        result.data["input_count"],
        result.data["inserted_count"],
        result.data["updated_count"],
        result.data["unchanged_count"],
    ) == (3, 1, 1, 1)
    assert db.rows["existing.test"]["notes"] == "keep omitted note"
    assert db.rows["new.test"]["notes"] is None
    assert result.data["generated_identities"][0]["values"]["id"] == 1
    writes = [
        entry
        for entry in db.log
        if entry[0] in {"execute", "fetch"}
        and entry[1].startswith(("INSERT", "UPDATE"))
    ]
    assert len(writes) == 2
    assert any(
        entry[0] == "execute" and "IN EXCLUSIVE MODE" in entry[1] for entry in db.log
    )
    assert ("transaction", {"isolation": "read_committed"}) in db.log


async def test_verified_unchanged_batch_performs_no_row_mutation(setup):
    backend, db, _ = setup
    db.rows["existing.test"] = row()
    batch = intent(
        (
            {
                "domain": "existing.test",
                "name": "Existing",
                "evidence_url": "https://source.test/existing",
            },
        )
    )
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)
    result = await execute(backend, batch, preview)
    assert result.data["unchanged_count"] == 1
    assert result.data["inserted_count"] == result.data["updated_count"] == 0
    assert result.effect_observation.outcome is EffectOutcome.SUCCEEDED
    assert not any(
        entry[0] in {"execute", "fetch"} and entry[1].startswith(("INSERT", "UPDATE"))
        for entry in db.log
    )


@pytest.mark.parametrize("drift", ["insert", "delete", "value", "schema", "permission"])
async def test_drift_under_exclusive_lock_rejects_before_mutation(setup, drift):
    backend, db, catalog = setup
    existing = drift in {"delete", "value"}
    if existing:
        db.rows["new.test"] = row("new.test")
    batch = intent()
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)

    def concurrent_writer():
        if drift == "insert":
            db.rows["new.test"] = row("new.test")
        elif drift == "delete":
            del db.rows["new.test"]
        elif drift == "value":
            db.rows["new.test"]["name"] = "Concurrent value"
        elif drift == "schema":
            db.structure_revision = "catalog:sha256:" + "8" * 64
        else:
            catalog.revoked = True

    db.before_lock = concurrent_writer
    with pytest.raises(native.RelationalUpdateExecutionError) as error:
        await execute(backend, batch, preview)
    assert error.value.effect_observation is not None
    assert error.value.effect_observation.outcome is EffectOutcome.NOT_APPLIED
    assert not any(
        entry[0] in {"execute", "fetch"} and entry[1].startswith(("INSERT", "UPDATE"))
        for entry in db.log
    )
    assert ("rollback",) in db.log


class LockTimeout(Exception):
    sqlstate = "55P03"


@pytest.mark.parametrize(
    "failure", ["lock_timeout", "cancel", "count", "commit_loss", "rollback_loss"]
)
async def test_transaction_failures_preserve_row_and_sequence_certainty(setup, failure):
    backend, db, _ = setup
    batch = intent()
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)
    if failure == "lock_timeout":
        db.lock_error = LockTimeout()
    elif failure == "cancel":
        db.mutation_error = asyncio.CancelledError()
    elif failure in {"count", "rollback_loss"}:
        db.bad_count = True
        if failure == "rollback_loss":
            db.rollback_error = ConnectionError()
    else:
        db.commit_error = ConnectionError("commit response lost")
    with pytest.raises(
        (
            native.RelationalUpdateExecutionError,
            native.RelationalUpdateExecutionCancelled,
        )
    ) as error:
        await execute(backend, batch, preview)
    assert isinstance(
        error.value,
        (
            native.RelationalUpdateExecutionError,
            native.RelationalUpdateExecutionCancelled,
        ),
    )
    observation = error.value.effect_observation
    assert observation is not None and observation.payload is not None
    assert observation.outcome is (
        EffectOutcome.UNCERTAIN
        if failure in {"commit_loss", "rollback_loss"}
        else EffectOutcome.NOT_APPLIED
    )
    assert observation.payload["identity_sequence_gaps_possible"] is True
    if failure == "commit_loss":
        assert db.rows["new.test"]["name"] == "New"
        assert observation.payload["inserted_count"] is None
    else:
        assert not db.rows
    if failure in {"count", "rollback_loss"}:
        assert db.next_id == 2
    if failure == "count":
        assert observation.payload["inserted_count"] == 0
    if failure == "lock_timeout":
        assert isinstance(error.value, native.RelationalUpdateExecutionError)
        assert error.value.error_code == "write_lock_timeout"


@pytest.mark.parametrize(
    "change,code",
    [
        ({"conflict_keys": ()}, "upsert_key_unsupported"),
        ({"column_collations": (("domain", "en_US"),)}, "upsert_key_unsupported"),
        (
            {
                "column_nullability": (
                    ("id", False),
                    ("domain", True),
                    ("name", False),
                    ("evidence_url", False),
                    ("notes", True),
                )
            },
            "upsert_key_unsupported",
        ),
        (
            {
                "column_defaults": (
                    ("id", None),
                    ("domain", None),
                    ("name", None),
                    ("evidence_url", None),
                    ("notes", "now()"),
                )
            },
            "upsert_default_unsupported",
        ),
        ({"generated_columns": ("notes",)}, "upsert_generated_unsupported"),
    ],
)
def test_catalog_admission_rejects_unsupported_semantics(change, code):
    with pytest.raises(CapabilityInputError) as error:
        validate_relational_upsert_intent(
            intent(),
            resource=replace(resource(), **change),
            generated_identity_columns=("id",),
            max_rows=10,
        )
    assert error.value.code == code


def test_uniform_shape_null_omission_duplicates_and_identity_admission():
    with pytest.raises(ValueError, match="uniform"):
        intent(({"domain": "a", "name": "A"},))
    with pytest.raises(CapabilityInputError):
        validate_relational_upsert_intent(
            intent(), resource=resource(), generated_identity_columns=(), max_rows=10
        )
    with pytest.raises(CapabilityInputError) as error:
        validate_relational_upsert_intent(
            intent(({"domain": "a", "name": "A", "evidence_url": None},)),
            resource=resource(),
            generated_identity_columns=("id",),
            max_rows=10,
        )
    assert error.value.code == "upsert_value_invalid"
    with pytest.raises(CapabilityInputError) as error:
        validate_relational_upsert_intent(
            intent((intent().rows[0], intent().rows[0])),
            resource=resource(),
            generated_identity_columns=("id",),
            max_rows=10,
        )
    assert error.value.code == "upsert_duplicate_key"
    nullable = replace(
        resource(),
        column_nullability=tuple(
            (name, True if name == "evidence_url" else value)
            for name, value in resource().column_nullability
        ),
    )
    result = validate_relational_upsert_intent(
        intent(({"domain": "a", "name": "A", "evidence_url": None},)),
        resource=nullable,
        generated_identity_columns=("id",),
        max_rows=10,
    )
    assert result.rows[0]["evidence_url"] is None


def test_uuid_keys_use_database_equivalent_normalization():
    base = resource()
    uuid_resource = replace(
        base,
        column_type_provenance=tuple(
            (column, ns, "uuid" if column == "domain" else name)
            for column, ns, name in base.column_type_provenance
        ),
    )
    key = "a0000000-0000-0000-0000-000000000001"
    batch = intent(
        tuple(
            {"domain": value, "name": "A", "evidence_url": "https://source.test"}
            for value in (key, key.upper())
        )
    )
    with pytest.raises(CapabilityInputError) as error:
        validate_relational_upsert_intent(
            batch,
            resource=uuid_resource,
            generated_identity_columns=("id",),
            max_rows=10,
        )
    assert error.value.code == "upsert_duplicate_key"


async def test_sensitive_request_cannot_reach_lower_classified_target(setup):
    backend, db, catalog = setup
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=intent())
    catalog.resource = replace(catalog.resource, sensitivity_class="public")
    with pytest.raises(native.RelationalUpdateExecutionError) as error:
        await execute(backend, intent(), preview)
    assert error.value.effect_observation is not None
    assert error.value.effect_observation.outcome is EffectOutcome.NOT_APPLIED
    assert not db.rows


@pytest.mark.parametrize(
    "field,value",
    [
        ("can_insert_columns", False),
        ("can_lock_table", False),
        ("unsupported_insert_features", True),
    ],
)
async def test_insert_readiness_rejects_privileges_and_target_features_before_scan(
    setup, field, value
):
    backend, db, _ = setup
    db.guardrails[field] = value
    with pytest.raises(CapabilityInputError) as error:
        await backend.preview_upsert(agent_id="agent-preview", intent=intent())
    assert error.value.code == "upsert_guardrail_rejected"
    assert not any(
        entry[0] == "fetch" and "upsert_target" in entry[1] for entry in db.log
    )
    assert db.next_id == 1 and not db.rows


class GuardrailRecord:
    """asyncpg-style keyed record without Mapping inheritance."""

    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


async def test_upsert_accepts_driver_record_guardrails_through_preview_and_commit(
    setup,
):
    from collections.abc import Mapping

    backend, db, _ = setup
    db.guardrails = GuardrailRecord(db.guardrails)
    assert not isinstance(db.guardrails, Mapping)
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=intent())
    result = await execute(backend, intent(), preview)
    assert result.effect_observation.outcome is EffectOutcome.SUCCEEDED
    assert len(db.rows) == 1


@pytest.mark.parametrize(
    "field", ["can_insert_columns", "can_lock_table", "unsupported_insert_features"]
)
async def test_upsert_driver_record_requires_every_guardrail_fact(setup, field):
    backend, db, _ = setup
    db.guardrails.pop(field)
    db.guardrails = GuardrailRecord(db.guardrails)
    with pytest.raises(CapabilityInputError) as error:
        await backend.preview_upsert(agent_id="agent-preview", intent=intent())
    assert error.value.code == "write_preview_failed"
    assert not any(entry[0] == "fetch" for entry in db.log)
    assert not db.rows


def test_native_guardrails_include_foreign_keys_referencing_the_target():
    # An incoming ON UPDATE CASCADE must not mutate an unapproved second table.
    assert (
        "con.confrelid = relation.oid AND con.contype = 'f'"
        in native._UPSERT_GUARDRAILS_SQL
    )


@pytest.mark.parametrize("value", [None, True, "1", 2**63])
async def test_invalid_returned_identity_rolls_back_before_commit(
    setup, monkeypatch, value
):
    backend, db, _ = setup
    batch = intent()
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)
    original_fetch = Connection.fetch

    async def malformed_identity(self, sql, *args, **kwargs):
        result = await original_fetch(self, sql, *args, **kwargs)
        return ({"id": value},) if sql.startswith("INSERT") else result

    monkeypatch.setattr(Connection, "fetch", malformed_identity)
    with pytest.raises(native.RelationalUpdateExecutionError) as error:
        await execute(backend, batch, preview)
    assert error.value.error_code == "upsert_identity_invalid"
    assert error.value.effect_observation is not None
    assert error.value.effect_observation.outcome is EffectOutcome.NOT_APPLIED
    assert not db.rows and db.next_id == 2
    assert ("commit", False) not in db.log


@pytest.mark.parametrize("failure", ["count", "cancel"])
async def test_failure_after_second_insertion_rolls_back_entire_batch(
    setup, monkeypatch, failure
):
    backend, db, _ = setup
    batch = intent(
        tuple(
            {"domain": domain, "name": domain, "evidence_url": "https://source.test"}
            for domain in ("a.test", "b.test")
        )
    )
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)
    original_fetch = Connection.fetch
    insertions = 0

    async def fail_second(self, sql, *args, **kwargs):
        nonlocal insertions
        result = await original_fetch(self, sql, *args, **kwargs)
        if sql.startswith("INSERT"):
            insertions += 1
            if insertions == 2:
                if failure == "cancel":
                    raise asyncio.CancelledError()
                return ()
        return result

    monkeypatch.setattr(Connection, "fetch", fail_second)
    with pytest.raises(
        (
            native.RelationalUpdateExecutionError,
            native.RelationalUpdateExecutionCancelled,
        )
    ) as error:
        await execute(backend, batch, preview)
    assert isinstance(
        error.value,
        (
            native.RelationalUpdateExecutionError,
            native.RelationalUpdateExecutionCancelled,
        ),
    )
    observation = error.value.effect_observation
    assert observation is not None and observation.payload is not None
    assert observation.outcome is EffectOutcome.NOT_APPLIED
    assert observation.payload["inserted_count"] == 0
    assert observation.payload["identity_sequence_gaps_possible"] is True
    assert insertions == 2 and db.next_id == 3
    assert not db.rows
    assert ("rollback",) in db.log and ("commit", False) not in db.log


async def test_readiness_normalizes_database_failure_and_closes_transaction(
    setup, monkeypatch
):
    backend, db, catalog = setup

    async def failed_inspection(*args, **kwargs):
        raise ConnectionError("private connection details")

    monkeypatch.setattr(Connection, "fetchrow", failed_inspection)
    with pytest.raises(CapabilityInputError) as error:
        await backend.upsert_readiness(
            "agent-preview",
            FrozenJsonObject.from_mapping(
                {
                    **catalog.permission.constraints(),
                    "source_id": SOURCE_ID,
                    "resource_id": RESOURCE_ID,
                }
            ),
        )
    assert error.value.code == "write_preview_failed"
    assert "private connection details" not in str(error.value)
    assert ("rollback",) in db.log and ("close",) in db.log


async def test_explicit_nullable_value_clears_column_but_omission_does_not(setup):
    backend, db, catalog = setup
    catalog.resource = replace(
        catalog.resource,
        column_nullability=tuple(
            (name, True if name == "evidence_url" else nullable)
            for name, nullable in catalog.resource.column_nullability
        ),
    )
    db.rows["existing.test"] = row()
    batch = intent(
        ({"domain": "existing.test", "name": "Existing", "evidence_url": None},)
    )
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)
    result = await execute(backend, batch, preview)
    assert result.data["updated_count"] == 1
    assert db.rows["existing.test"]["evidence_url"] is None
    assert db.rows["existing.test"]["notes"] == "keep omitted note"


async def test_permission_revision_change_during_lock_wait_is_rejected(setup):
    backend, db, catalog = setup
    batch = intent()
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)

    def revise():
        catalog.permission = replace(
            catalog.permission, authorization_fingerprint="sha256:" + "6" * 64
        )

    db.before_lock = revise
    with pytest.raises(native.RelationalUpdateExecutionError) as error:
        await execute(backend, batch, preview)
    assert error.value.effect_observation is not None
    assert error.value.effect_observation.outcome is EffectOutcome.NOT_APPLIED
    assert not db.rows and db.next_id == 1


async def test_cancellation_during_commit_retains_uncertainty(setup):
    backend, db, _ = setup
    batch = intent()
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)
    db.commit_error = asyncio.CancelledError()
    with pytest.raises(native.RelationalUpdateExecutionCancelled) as error:
        await execute(backend, batch, preview)
    assert error.value.effect_observation.outcome is EffectOutcome.UNCERTAIN
    assert db.rows["new.test"]["name"] == "New"


async def test_cancellation_during_cleanup_retains_verified_commit(setup, monkeypatch):
    backend, db, _ = setup
    batch = intent()
    preview = await backend.preview_upsert(agent_id="agent-preview", intent=batch)

    async def cancelled_close(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(native, "_close_postgresql_connection", cancelled_close)
    with pytest.raises(native.RelationalUpdateExecutionCancelled) as error:
        await execute(backend, batch, preview)
    assert error.value.effect_observation.outcome is EffectOutcome.SUCCEEDED
    assert db.rows["new.test"]["name"] == "New"


def test_batch_order_has_one_canonical_operation_identity():
    rows = (
        {"domain": "a", "name": "A", "evidence_url": "https://source.test/a"},
        {"domain": "b", "name": "B", "evidence_url": "https://source.test/b"},
    )
    a = validate_relational_upsert_intent(
        intent(rows, evidence_call_ids=("research-a",)),
        resource=resource(),
        generated_identity_columns=("id",),
        max_rows=10,
    )
    b = validate_relational_upsert_intent(
        intent(tuple(reversed(rows)), evidence_call_ids=("research-b",)),
        resource=resource(),
        generated_identity_columns=("id",),
        max_rows=10,
    )
    assert a.intent_sha256 == b.intent_sha256


def test_omitted_required_values_and_oversized_batch_fail_before_io():
    with pytest.raises(CapabilityInputError) as error:
        validate_relational_upsert_intent(
            intent(
                ({"domain": "a", "evidence_url": "https://source.test/a"},),
                insert_columns=("domain", "evidence_url"),
                update_columns=("evidence_url",),
            ),
            resource=resource(),
            generated_identity_columns=("id",),
            max_rows=10,
        )
    assert error.value.code == "upsert_required_value_missing"
    with pytest.raises(CapabilityInputError) as error:
        validate_relational_upsert_intent(
            intent(
                tuple(
                    {
                        "domain": str(i),
                        "name": "A",
                        "evidence_url": "https://source.test",
                    }
                    for i in range(11)
                )
            ),
            resource=resource(),
            generated_identity_columns=("id",),
            max_rows=10,
        )
    assert error.value.code == "write_row_limit"
