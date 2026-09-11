"""Shared helpers extracted from ``test_upsert.py``."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from daita._json import FrozenJsonObject
from daita.adapters import postgresql_write as native
from daita.capabilities import CapabilityInputError, EffectOutcome, ToolExecution
from daita.catalog.models import ResourceKind
from daita.domains.data.sql import ResourceSchema
from daita.domains.data.sql.relational_upsert import (
    RelationalUpsertIntent,
    validate_relational_upsert_intent,
)
from daita.llm.models import ModelSensitivity
from daita.security import EmptySecretProvider
from daita.storage.sqlite_records import RelationalWriteScope
from tests.data.writes._preview_support import (
    NOW,
    RESOURCE_ID,
    RESOURCE_REVISION,
    SOURCE_ID,
    SOURCE_REVISION,
    _guardrails,
    _registration,
    _SourceStore,
)


class Database:
    def __init__(self):
        self.rows = {}
        self.next_id = 1
        self.log = []
        self.before_lock = None
        self.lock_error = None
        self.mutation_error = None
        self.commit_error: BaseException | None = None
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
        self.db.next_id += 1
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
