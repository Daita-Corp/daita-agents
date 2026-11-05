"""Reproducible baseline for Daita's PostgreSQL handling of a messy schema.

Opt-in: set ``DAITA_SCHEMA_BASELINE_POSTGRES_URL`` to a disposable database
named ``daita_schema_baseline`` (see README.md). Every run reseeds it from
``tests/fixtures/postgres-schema-baseline/seed.sql`` and writes JSON and
Markdown results. The offline run scripts an oracle model that issues each
question's known-correct SQL, so it measures the catalog gate and source cost,
not model reasoning. The live-model run is separately gated.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import sqlite3
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from daita import (
    Agent,
    AgentConfig,
    AgentEvent,
    LoopExitKind,
    LoopLimits,
    PostgreSQLReadLimits,
)
from daita.evaluation import BenchmarkJudgment, measure_observer_events
from daita.llm.models import FinishReason, ModelResponse, ToolCall
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference
from tests.support.live_comparison import (
    ANSWER_FORMAT,
    OUTCOMES,
    answer_record,
    claude_code_version,
    normalize,
    run_claude_code,
    system_summary,
)
from tests.support.paths import FIXTURES_ROOT, REPO_ROOT
from tests.support.workspace import workspace_for

_URL = "DAITA_SCHEMA_BASELINE_POSTGRES_URL"
_SOURCE_NAME = "Schema baseline"
_REPORT_DIR = "DAITA_SCHEMA_BASELINE_REPORT_DIR"
_LIVE_GATE = "DAITA_RUN_LIVE_SCHEMA_BASELINE_COMPARISON"
_SYSTEMS = "DAITA_SCHEMA_BASELINE_SYSTEMS"
_MODEL_ID = "DAITA_SCHEMA_BASELINE_MODEL_ID"
_MODEL_KEY = "DAITA_SCHEMA_BASELINE_LLM_API_KEY"
_MAX_COST = "DAITA_SCHEMA_BASELINE_MAX_COST_USD"
_CLAUDE = "DAITA_SCHEMA_BASELINE_CLAUDE_EXECUTABLE"
_REPEATS = "DAITA_SCHEMA_BASELINE_REPEATS"
_DAITA_MAX_TOKENS = "DAITA_SCHEMA_BASELINE_DAITA_MAX_TOKENS"
_READER = ("daita_baseline_reader", "daita_baseline_reader_password")
_DATABASE = "daita_schema_baseline"
_FIXTURE = FIXTURES_ROOT / "postgres-schema-baseline"
_CREDENTIAL = SecretReference.keychain("fixture:postgres-schema-baseline")
_ALL_SCHEMAS = ("crm", "events", "finance", "legacy", "ops", "sales")
# 987 base tables: under the 1,000-resource discovery limit.
_BOUNDED_SCHEMAS = ("crm", "events", "finance", "legacy", "sales")
_USER_QUERY_MARKERS = frozenset({"bounded_result", "exact_tabular"})
_MARKER = re.compile(r"/\* daita:postgresql\.([a-z_]+) \*/")
# The harness's own idea of ordinary column types, used only to explain why a
# table is missing; it is deliberately independent of Daita's allow-list.
_PLAIN_TYPES = frozenset(
    {
        "bool",
        "bpchar",
        "bytea",
        "date",
        "float4",
        "float8",
        "int2",
        "int4",
        "int8",
        "interval",
        "json",
        "jsonb",
        "numeric",
        "text",
        "time",
        "timestamp",
        "timestamptz",
        "uuid",
        "varchar",
    }
)
_STALE_STEPS = (
    (
        "add_partition",
        "CREATE TABLE events.page_views_20260501 PARTITION OF events.page_views "
        "FOR VALUES FROM ('2026-05-01 00:00:00+00') TO ('2026-05-02 00:00:00+00')",
    ),
    (
        "add_unrelated_table",
        "CREATE TABLE crm.unrelated_scratch (id integer PRIMARY KEY, note text)",
    ),
)

_INVENTORY_SQL = """
SELECT n.nspname AS schema_name,
       c.relname AS name,
       c.relkind::text AS relkind,
       c.relispartition AS is_partition,
       coalesce(json_agg(json_build_object(
           'column', a.attname,
           'type', format_type(a.atttypid, a.atttypmod),
           'type_schema', coalesce(etn.nspname, tn.nspname),
           'type_name', coalesce(et.typname, t.typname),
           'typtype', coalesce(et.typtype, t.typtype)::text,
           'array', et.oid IS NOT NULL,
           'extension', ext.extname
       ) ORDER BY a.attnum) FILTER (WHERE a.attnum IS NOT NULL), '[]') AS columns
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
LEFT JOIN pg_attribute AS a
  ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
LEFT JOIN pg_type AS t ON t.oid = a.atttypid
LEFT JOIN pg_namespace AS tn ON tn.oid = t.typnamespace
LEFT JOIN pg_type AS et ON et.oid = t.typelem AND t.typcategory = 'A'
LEFT JOIN pg_namespace AS etn ON etn.oid = et.typnamespace
LEFT JOIN pg_depend AS dep
  ON dep.classid = 'pg_type'::regclass
 AND dep.objid = coalesce(et.oid, t.oid)
 AND dep.deptype = 'e'
LEFT JOIN pg_extension AS ext ON ext.oid = dep.refobjid
WHERE n.nspname = ANY($1::text[]) AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
GROUP BY n.nspname, c.relname, c.relkind, c.relispartition
ORDER BY n.nspname, c.relname
"""

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        not os.environ.get(_URL),
        reason=(
            f"set {_URL} to a disposable {_DATABASE} database, for example after "
            "starting tests/fixtures/postgres-schema-baseline/compose.yaml"
        ),
    ),
]


@dataclass(frozen=True)
class _Target:
    host: str
    port: int
    user: str
    password: str
    ssl_mode: str


def _target() -> _Target:
    url = urlsplit(os.environ[_URL])
    if url.scheme not in {"postgres", "postgresql"} or url.path != f"/{_DATABASE}":
        pytest.fail(f"{_URL} must be a postgresql:// URL for database {_DATABASE}")
    return _Target(
        host=url.hostname or "127.0.0.1",
        port=url.port or 5432,
        user=unquote(url.username or "postgres"),
        password=unquote(url.password or ""),
        ssl_mode=parse_qs(url.query).get("sslmode", ["disable"])[0],
    )


class _Secrets:
    def __init__(self, password: str) -> None:
        self._password = password

    async def resolve(self, reference: SecretReference) -> str:
        assert reference == _CREDENTIAL
        return self._password


class _RoundTrips:
    """Count Daita's tagged statements on every connection except the admin's."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, admin: Any) -> None:
        import asyncpg  # type: ignore[import-untyped]

        self.counts: Counter[str] = Counter()
        # A bounded user read is one prepare plus cursor fetches; count the prepare.
        # Bulk discovery reads open `Connection.cursor`, which user reads never use.
        for name in ("execute", "fetch", "fetchrow", "fetchval", "prepare", "cursor"):
            original = getattr(asyncpg.Connection, name)

            def counted(
                connection: Any,
                query: object,
                *args: object,
                _original: Any = original,
                **kwargs: object,
            ) -> Any:
                if connection is not admin:
                    match = _MARKER.search(query) if isinstance(query, str) else None
                    self.counts[match.group(1) if match else "untagged"] += 1
                return _original(connection, query, *args, **kwargs)

            monkeypatch.setattr(asyncpg.Connection, name, counted)

    def snapshot(self) -> Counter[str]:
        return Counter(self.counts)

    def since(self, before: Counter[str]) -> dict[str, object]:
        delta = self.counts - before
        metadata = {
            key: value
            for key, value in sorted(delta.items())
            if key not in _USER_QUERY_MARKERS and key != "untagged"
        }
        return {
            "metadata": sum(metadata.values()),
            "metadata_by_statement": metadata,
            "user_query": sum(delta[key] for key in _USER_QUERY_MARKERS),
            "session_and_transaction": delta["untagged"],
        }


class _OracleModel(MockModelProvider):
    """Issue one question's known-correct SQL, then stop."""

    def __init__(self) -> None:
        super().__init__((), provider_id="mock:schema-baseline-oracle")

    def ask(self, arguments: Mapping[str, object]) -> None:
        self._script = (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(id="oracle", name="data_query", arguments=dict(arguments)),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="Answered."),
        )
        self._cursor = 0


def _questions() -> list[dict[str, Any]]:
    return json.loads((_FIXTURE / "questions.json").read_text())


async def _connect_admin(target: _Target) -> Any:
    import asyncpg

    return await asyncpg.connect(
        host=target.host,
        port=target.port,
        user=target.user,
        password=target.password,
        database=_DATABASE,
        ssl=False if target.ssl_mode == "disable" else target.ssl_mode,
        timeout=10,
        command_timeout=120,
        server_settings={"application_name": "daita-schema-baseline-admin"},
    )


async def _seed(admin: Any) -> dict[str, object]:
    started = time.perf_counter()
    # One implicit transaction per blank-line chunk keeps DDL under the lock table.
    for chunk in re.split(r"\n\s*\n", (_FIXTURE / "seed.sql").read_text()):
        if re.sub(r"--[^\n]*", "", chunk).strip():
            await admin.execute(chunk)
    return {
        "seed_seconds": round(time.perf_counter() - started, 3),
        "server_version": await admin.fetchval("SHOW server_version"),
        "citext_available": await admin.fetchval(
            "SELECT to_regtype('citext') IS NOT NULL"
        ),
        "base_tables_by_schema": {
            row["table_schema"]: row["count"]
            for row in await admin.fetch(
                "SELECT table_schema, count(*) FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE' AND table_schema = ANY($1::text[]) "
                "GROUP BY table_schema ORDER BY table_schema",
                list(_ALL_SCHEMAS),
            )
        },
    }


async def _ground_truth(
    admin: Any, questions: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    return {
        item["id"]: normalize(await admin.fetchval(item["sql"])) for item in questions
    }


def _missing_reason(relkind: str, columns: Sequence[Mapping[str, Any]]) -> list[str]:
    kinds = {"v": "view", "m": "materialized_view", "f": "foreign_table"}
    if relkind in kinds:
        return [kinds[relkind]]
    reasons = []
    for column in columns:
        prefix = "array_of_" if column["array"] else ""
        if column["extension"]:
            kind = f"extension_type:{column['extension']}"
        elif column["typtype"] == "e":
            kind = "enum"
        elif column["typtype"] == "d":
            kind = "domain"
        elif column["typtype"] == "c":
            kind = "composite"
        elif column["type_schema"] != "pg_catalog":
            kind = "user_type"
        elif column["type_name"] not in _PLAIN_TYPES:
            kind = f"builtin_type:{column['type_name']}"
        else:
            continue
        reasons.append(f"{prefix}{kind} ({column['column']} {column['type']})")
    return reasons or ["unexplained"]


async def _coverage(
    admin: Any, agent: Agent, source_id: str, schemas: Sequence[str]
) -> dict[str, object]:
    catalog = {
        resource.native_identity: resource
        for resource in await agent.list_catalog_resources(source_id=source_id)
    }
    missing: list[dict[str, object]] = []
    missing_columns: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    relations = 0
    for row in await admin.fetch(_INVENTORY_SQL, list(schemas)):
        relations += 1
        identity = f"{row['schema_name']}.{row['name']}"
        columns = json.loads(row["columns"])
        resource = catalog.get(identity)
        if resource is None:
            why = (
                ["partition_of_listed_parent"]
                if row["is_partition"]
                else _missing_reason(row["relkind"], columns)
            )
            reasons.update(item.split(" (")[0] for item in why)
            missing.append(
                {"table": identity, "relkind": row["relkind"], "reasons": why}
            )
            continue
        described: Any = await agent.inspect_catalog_resource(resource.id)
        cataloged = {
            column["name"]
            for facet in described["facets"]
            for column in facet["payload"].get("columns", ())
        }
        absent = [item["column"] for item in columns if item["column"] not in cataloged]
        if absent:
            missing_columns.append({"table": identity, "columns": absent})
    foreign_keys = await admin.fetchval(
        "SELECT count(*) FROM pg_constraint AS con "
        "JOIN pg_class AS src ON src.oid = con.conrelid "
        "JOIN pg_namespace AS sn ON sn.oid = src.relnamespace "
        "JOIN pg_class AS dst ON dst.oid = con.confrelid "
        "JOIN pg_namespace AS dn ON dn.oid = dst.relnamespace "
        "WHERE con.contype = 'f' AND sn.nspname = ANY($1::text[]) "
        "AND dn.nspname = ANY($1::text[])",
        list(schemas),
    )
    summary = await agent.catalog_summary()
    return {
        "foreign_keys_in_schemas": foreign_keys,
        "cataloged_relationships": summary.relationship_count,
        "relations_in_schemas": relations,
        "cataloged_resources": len(catalog),
        "partitions_cataloged_as_resources": sum(
            1 for name in catalog if re.search(r"page_views_\d{8}$", name)
        ),
        "missing_relations": len(missing),
        "missing_by_reason": dict(sorted(reasons.items())),
        "missing": missing,
        "cataloged_tables_with_missing_columns": missing_columns,
    }


async def _attach(
    agent: Agent,
    target: _Target,
    schemas: Sequence[str],
    trips: _RoundTrips,
    username: str | None = None,
) -> tuple[str | None, dict[str, object]]:
    before = trips.snapshot()
    started = time.perf_counter()
    source_id = None
    outcome: dict[str, object] = {"schemas": list(schemas)}
    try:
        source = await agent.attach_postgresql(
            host=target.host,
            port=target.port,
            database=_DATABASE,
            username=username or target.user,
            credential=_CREDENTIAL,
            schemas=tuple(schemas),
            ssl_mode=target.ssl_mode,
            name=_SOURCE_NAME,
        )
    except Exception as error:  # Every attach outcome is baseline evidence.
        outcome |= {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_code": getattr(error, "error_code", None)
            or getattr(error, "code", None),
            "message": str(error)[:300],
        }
    else:
        source_id = source.id
        summary = await agent.catalog_summary()
        outcome |= {
            "status": "attached",
            "resources": summary.resource_count,
            "relationships": summary.relationship_count,
        }
    outcome["seconds"] = round(time.perf_counter() - started, 3)
    outcome["round_trips"] = trips.since(before)
    return source_id, outcome


async def _resource_ids(agent: Agent, source_id: str) -> dict[str, str]:
    return {
        resource.native_identity: resource.id
        for resource in await agent.list_catalog_resources(source_id=source_id)
    }


def _query_outcome(evidence: Sequence[tuple[ToolCall, Any]]) -> dict[str, Any]:
    queries = [(call, block) for call, block in evidence if call.name == "data_query"]
    codes = []
    last_rows = None
    for _, block in queries:
        if block is None:
            codes.append("no_result")
        elif block.is_error:
            error = block.output.get("error")
            codes.append(
                str(error.get("code")) if isinstance(error, Mapping) else "error"
            )
        else:
            data = block.output.get("data")
            last_rows = data.get("rows") if isinstance(data, Mapping) else None
    return {
        "data_query_calls": len(queries),
        "error_codes": codes,
        "succeeded": any(
            block is not None and not block.is_error for _, block in queries
        ),
        "rows": last_rows,
        "resources_queried": sorted(
            {
                item
                for call, block in queries
                if block is not None and not block.is_error
                for item in _queried_resource_ids(block)
            }
        ),
    }


def _queried_resource_ids(block: Any) -> Sequence[str]:
    """The tables a successful read touched; the model names only its SQL."""

    data = block.output.get("data")
    resources = data.get("resource_ids") if isinstance(data, Mapping) else None
    return cast(Sequence[str], resources or ())


def _first_value(rows: object) -> str | None:
    if not isinstance(rows, Sequence) or not rows:
        return None
    row = rows[0]
    if isinstance(row, Mapping):
        return normalize(next(iter(row.values()))) if row else None
    if isinstance(row, Sequence) and not isinstance(row, str) and row:
        return normalize(row[0])
    return None


async def _measured_run(
    agent: Agent,
    message: str,
    source_id: str,
    events: list[AgentEvent],
    trips: _RoundTrips,
) -> tuple[Any, dict[str, object]]:
    diagnostics = agent.postgresql_read_diagnostics()
    before_trips = trips.snapshot()
    before_events = len(events)
    started = time.perf_counter()
    # Daita enforces its own 300 s wall limit; this only catches a broken one.
    async with asyncio.timeout(360):
        result = await agent.run(message, source_scope_ids=(source_id,))
    seconds = time.perf_counter() - started
    after = agent.postgresql_read_diagnostics()
    run_measurement = measure_observer_events(tuple(events[before_events:]))
    return result, {
        "run_seconds": round(seconds, 3),
        "query_seconds": round(after.query_seconds - diagnostics.query_seconds, 3),
        "round_trips": trips.since(before_trips),
        "turns": result.steps,
        "tool_calls": run_measurement.tool_calls,
        "failed_sql_calls": run_measurement.failed_sql_calls,
        "total_tokens": result.usage.total_tokens,
        "exit": result.reason,
    }


async def _ask_oracle(
    agent: Agent,
    model: _OracleModel,
    question: Mapping[str, Any],
    source_id: str,
    truth: str,
    events: list[AgentEvent],
    trips: _RoundTrips,
) -> dict[str, object]:
    ids = await _resource_ids(agent, source_id)
    found = [ids[name] for name in question["resources"] if name in ids]
    model.ask({"source": _SOURCE_NAME, "sql": question["sql"]})
    result, measured = await _measured_run(
        agent, question["question"], source_id, events, trips
    )
    query = _query_outcome(await agent.tool_evidence(result.run_id))
    answer = _first_value(query["rows"]) if query["succeeded"] else None
    if not query["succeeded"]:
        outcome = "refused"
    else:
        outcome = "correct" if answer == truth else "confident_wrong"
    judgment = BenchmarkJudgment(
        answer_correct=outcome == "correct",
        resource_selection_correct=len(found) == len(question["resources"]),
    )
    return {
        "id": question["id"],
        "category": question["category"],
        "outcome": outcome,
        "refusal_codes": query["error_codes"],
        "expected": truth,
        "answer": answer,
        "resources_missing_from_catalog": [
            name for name in question["resources"] if name not in ids
        ],
        "judgment": asdict(judgment),
        **measured,
    }


async def _stale_scenario(
    admin: Any,
    agent: Agent,
    model: _OracleModel,
    question: Mapping[str, Any],
    source_id: str,
    truth: str,
    events: list[AgentEvent],
    trips: _RoundTrips,
) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for name, ddl in _STALE_STEPS:
        await admin.execute(ddl)
        after_change = await _ask_oracle(
            agent, model, question, source_id, truth, events, trips
        )
        before = trips.snapshot()
        started = time.perf_counter()
        await agent.refresh_source(source_id)
        refresh = {
            "seconds": round(time.perf_counter() - started, 3),
            "round_trips": trips.since(before),
        }
        after_refresh = await _ask_oracle(
            agent, model, question, source_id, truth, events, trips
        )
        steps.append(
            {
                "change": name,
                "question": question["id"],
                "after_change": _brief(after_change),
                "refresh": refresh,
                "after_refresh": _brief(after_refresh),
            }
        )
    return steps


def _brief(run: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "outcome",
        "refusal_codes",
        "run_seconds",
        "query_seconds",
        "round_trips",
    )
    return {key: run[key] for key in keys}


def _report_directory() -> Path:
    configured = os.environ.get(_REPORT_DIR)
    path = (
        Path(configured).expanduser()
        if configured
        else REPO_ROOT / "live-evidence" / "schema-baseline"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _revision() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src")),
    }


def _summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    outcomes = Counter(str(run["outcome"]) for run in runs)
    codes = Counter(code for run in runs for code in run["refusal_codes"])
    metadata = [int(run["round_trips"]["metadata"]) for run in runs]
    return {
        "questions": len(runs),
        "outcomes": dict(sorted(outcomes.items())),
        "refusal_codes": dict(sorted(codes.items())),
        "mean_run_seconds": round(
            sum(float(run["run_seconds"]) for run in runs) / max(len(runs), 1), 3
        ),
        "total_query_seconds": round(
            sum(float(run["query_seconds"]) for run in runs), 3
        ),
        "mean_metadata_round_trips": round(sum(metadata) / max(len(metadata), 1), 1),
        "total_turns": sum(int(run["turns"]) for run in runs),
        "total_tokens": sum(int(run["total_tokens"]) for run in runs),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    meta = report["environment"]
    lines = [
        f"# PostgreSQL schema baseline — {report['mode']}",
        "",
        f"Commit `{meta['commit']}` ({meta['branch']}"
        f"{', src dirty' if meta['dirty'] else ''}) · PostgreSQL "
        f"{meta['server_version']} · citext {meta['citext_available']} · "
        f"{meta['generated_at']}",
        "",
        "## Attach / discovery",
        "",
        "| Variant | Base tables | Outcome | Seconds | Metadata round trips | "
        "Resources | Relationships |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, attach in report["attach"].items():
        status = attach["status"]
        if status == "failed":
            status = f"failed: `{attach['error_code'] or attach['error_type']}`"
        lines.append(
            f"| {name} | {attach['base_tables']} | {status} | {attach['seconds']} | "
            f"{attach['round_trips']['metadata']} | {attach.get('resources', '—')} | "
            f"{attach.get('relationships', '—')} |"
        )
    coverage = report.get("coverage")
    if coverage:
        lines += [
            "",
            "## Catalog coverage (bounded variant)",
            "",
            f"{coverage['cataloged_resources']} of "
            f"{coverage['relations_in_schemas']} relations cataloged; "
            f"{coverage['partitions_cataloged_as_resources']} partitions appear "
            "as separate resources; "
            f"{coverage['cataloged_relationships']} of "
            f"{coverage['foreign_keys_in_schemas']} foreign keys cataloged; "
            f"{len(coverage['cataloged_tables_with_missing_columns'])} cataloged "
            "tables are missing columns.",
            "",
            "| Missing because | Relations |",
            "| --- | ---: |",
            *(
                f"| {reason} | {count} |"
                for reason, count in coverage["missing_by_reason"].items()
            ),
        ]
    summary = report.get("summary")
    if summary:
        lines += [
            "",
            "## Questions",
            "",
            f"Outcomes: {summary['outcomes']} · refusal codes: "
            f"{summary['refusal_codes'] or 'none'}",
            f"Mean run {summary['mean_run_seconds']}s · query "
            f"{summary['total_query_seconds']}s in total · mean metadata round "
            f"trips per question {summary['mean_metadata_round_trips']} · turns "
            f"{summary['total_turns']} · tokens {summary['total_tokens']}",
            "",
            "| Question | Category | Outcome | Codes | Run s | "
            "Query s | Metadata trips | Loop steps | Tokens |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for run in report["questions"]:
            codes = ", ".join(run["refusal_codes"]) or "—"
            if run.get("resources_missing_from_catalog"):
                missing = ", ".join(run["resources_missing_from_catalog"])
                codes += f" (not cataloged: {missing})"
            lines.append(
                f"| {run['id']} | {run['category']} | {run['outcome']} | "
                f"{codes} | {run['run_seconds']} | {run['query_seconds']} | "
                f"{run['round_trips']['metadata']} | {run['turns']} | "
                f"{run['total_tokens']} |"
            )
    if report.get("stale_scenario"):
        lines += [
            "",
            "## Schema change between queries",
            "",
            "| Change | After change | Refresh s | After refresh |",
            "| --- | --- | ---: | --- |",
        ]
        for step in report["stale_scenario"]:
            before, after = step["after_change"], step["after_refresh"]
            lines.append(
                f"| {step['change']} | {before['outcome']} "
                f"{', '.join(before['refusal_codes'])} | {step['refresh']['seconds']} "
                f"| {after['outcome']} {', '.join(after['refusal_codes'])} |"
            )
    return "\n".join(lines) + "\n"


def _write(report: dict[str, Any], stem: str) -> Path:
    directory = _report_directory()
    name = f"{stem}-{report['environment']['commit']}"
    (directory / f"{name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    path = directory / f"{name}.md"
    path.write_text(_markdown(report))
    return path


async def _prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Target, Any, _RoundTrips, dict[str, Any], list[dict[str, Any]], dict]:
    target = _target()
    admin = await _connect_admin(target)
    trips = _RoundTrips(monkeypatch, admin)
    environment = {
        **_revision(),
        **await _seed(admin),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }
    questions = _questions()
    truth = await _ground_truth(admin, questions)
    return target, admin, trips, environment, questions, truth


def _base_tables(environment: Mapping[str, Any], schemas: Sequence[str]) -> int:
    counts = environment["base_tables_by_schema"]
    return sum(int(counts.get(schema, 0)) for schema in schemas)


async def test_postgresql_schema_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, admin, trips, environment, questions, truth = await _prepare(monkeypatch)
    model = _OracleModel()
    events: list[AgentEvent] = []
    report: dict[str, Any] = {
        "mode": "offline oracle SQL (catalog gate and source cost, not reasoning)",
        "environment": environment,
        "attach": {},
    }
    try:
        for variant, schemas in (
            ("full", _ALL_SCHEMAS),
            ("bounded", _BOUNDED_SCHEMAS),
        ):
            async with await Agent.create(
                f"schema-baseline-{variant}",
                root=tmp_path / variant,
                model=model,
                model_profile=model.model_profile,
                config=AgentConfig(
                    postgresql_reads=PostgreSQLReadLimits(pool_enabled=True)
                ),
                secret_provider=_Secrets(target.password),
                workspace=workspace_for(tmp_path / variant),
                observer=events.append,
            ) as agent:
                source_id, attach = await _attach(agent, target, schemas, trips)
                attach["base_tables"] = _base_tables(environment, schemas)
                report["attach"][variant] = attach
                if variant == "full" or source_id is None:
                    continue
                report["coverage"] = await _coverage(admin, agent, source_id, schemas)
                report["questions"] = [
                    await _ask_oracle(
                        agent,
                        model,
                        question,
                        source_id,
                        truth[question["id"]],
                        events,
                        trips,
                    )
                    for question in questions
                ]
                report["summary"] = _summary(report["questions"])
                report["stale_scenario"] = await _stale_scenario(
                    admin,
                    agent,
                    model,
                    questions[0],
                    source_id,
                    truth[questions[0]["id"]],
                    events,
                    trips,
                )
    finally:
        await admin.close()
    path = _write(report, "postgresql-schema-baseline")
    print(f"\nschema baseline written to {path}")
    assert report["attach"]["bounded"]["status"] == "attached", report["attach"]
    assert len(report["questions"]) == len(questions)


async def test_the_question_set_answers_from_the_source_with_an_empty_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, admin, trips, _environment, questions, truth = await _prepare(monkeypatch)
    model = _OracleModel()
    events: list[AgentEvent] = []
    try:
        async with await Agent.create(
            "schema-baseline-empty-catalog",
            root=tmp_path,
            model=model,
            model_profile=model.model_profile,
            secret_provider=_Secrets(target.password),
            workspace=workspace_for(tmp_path),
            observer=events.append,
        ) as agent:
            source_id, _attached = await _attach(agent, target, _BOUNDED_SCHEMAS, trips)
            assert source_id is not None
            with sqlite3.connect(next(tmp_path.rglob("state.db"))) as connection:
                connection.execute("DELETE FROM snapshots")
            runs = [
                await _ask_oracle(
                    agent,
                    model,
                    question,
                    source_id,
                    truth[question["id"]],
                    events,
                    trips,
                )
                for question in questions
            ]
    finally:
        await admin.close()
    assert {run["id"]: run["outcome"] for run in runs} == {
        question["id"]: "correct" for question in questions
    }
    for run in runs:
        round_trips = cast(Mapping[str, Any], run["round_trips"])
        # A read may check the functions it calls; it never rediscovers the source.
        assert round_trips["metadata_by_statement"] in (
            {},
            {"function_policy": 1},
        ), run["id"]


def _daita_limits() -> LoopLimits:
    from tests.support.live_billing import cost_cap

    model_id = os.environ.get(_MODEL_ID, "anthropic:claude-sonnet-5")
    tokens = os.environ.get(_DAITA_MAX_TOKENS)
    return LoopLimits(
        max_total_tokens=int(tokens) if tokens else LoopLimits().max_total_tokens,
        # Daita's Anthropic provider never claims a complete preflight price,
        # so a dollar cap refuses every run; keep the step/token/time limits.
        max_estimated_cost_usd=(
            None
            if model_id.startswith("anthropic:")
            else cost_cap(Decimal(os.environ.get(_MAX_COST, "0.50")))
        ),
    )


def _daita_calls(
    evidence: Sequence[tuple[ToolCall, Any]], names: Mapping[str, str]
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for call, block in evidence:
        error = None
        if block is None:
            error = "no_result"
        elif block.is_error:
            detail = block.output.get("error")
            error = str(detail.get("code")) if isinstance(detail, Mapping) else "error"
        arguments = call.arguments
        resources = (
            _queried_resource_ids(block)
            if block is not None and not block.is_error
            else ()
        )
        calls.append(
            {
                "tool": call.name,
                "sql": str(arguments.get("sql") or arguments.get("statements") or "")[
                    :1_000
                ],
                "resources": [
                    names.get(str(item), str(item))
                    for item in cast(Sequence[object], resources)
                ],
                "error": error,
            }
        )
    return calls


async def _run_daita(
    target: _Target,
    model_id: str,
    questions: Sequence[Mapping[str, Any]],
    truth: Mapping[str, str],
    trips: _RoundTrips,
    root: Path,
    entry: dict[str, Any],
    repeat: int,
) -> None:
    """Answer every question with a fresh agent; append results to ``entry``."""

    from tests.support.live_billing import live_provider

    def api_key() -> str:
        value = os.environ.get(_MODEL_KEY)
        if not value:
            pytest.fail(f"{_MODEL_KEY} must be set for API billing")
        return value

    profile, provider = live_provider(
        model_id, api_key=api_key, max_output_tokens=4_096
    )
    events: list[AgentEvent] = []
    try:
        # Default configuration: the comparison measures Daita as shipped.
        async with await Agent.create(
            "schema-baseline-comparison",
            root=root,
            model=provider,
            model_profile=profile,
            limits=_daita_limits(),
            secret_provider=_Secrets(_READER[1]),
            workspace=workspace_for(root),
            observer=events.append,
        ) as agent:
            source_id, attach = await _attach(
                agent, target, _BOUNDED_SCHEMAS, trips, username=_READER[0]
            )
            entry["attach"].append(attach)
            if source_id is None:
                pytest.fail(f"Daita could not attach: {attach}")
            names = {
                resource_id: name
                for name, resource_id in (await _resource_ids(agent, source_id)).items()
            }
            for question in questions:
                result, measured = await _measured_run(
                    agent,
                    question["question"] + ANSWER_FORMAT,
                    source_id,
                    events,
                    trips,
                )
                # A run refused before its first step (rejected key, unpriced cap)
                # is a setup failure; budget exhaustion later is a real outcome.
                if result.steps == 0 and result.kind is not LoopExitKind.COMPLETED:
                    pytest.fail(f"Daita could not run: {result.reason}")
                measured.pop("query_seconds")  # Recorded only with the pool enabled.
                estimate = result.usage.cost_estimate
                entry["questions"].append(
                    {
                        "id": question["id"],
                        "category": question["category"],
                        "repeat": repeat,
                        **answer_record(
                            truth[question["id"]],
                            result.final_text or "",
                            result.kind is LoopExitKind.COMPLETED,
                        ),
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "cache_read_tokens": result.usage.cache_read_tokens,
                        "cost_usd": (
                            None
                            if estimate.amount_usd is None
                            or estimate.status.value != "complete"
                            else str(estimate.amount_usd)
                        ),
                        "cost_status": estimate.status.value,
                        "calls": _daita_calls(
                            await agent.tool_evidence(result.run_id), names
                        ),
                        **measured,
                    }
                )
    finally:
        await provider.close()


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    meta = report["environment"]
    systems = report["systems"]
    limits = report["limits"]
    lines = [
        "# PostgreSQL schema baseline — Daita vs Claude Code",
        "",
        f"Commit `{meta['commit']}` ({meta['branch']}) · PostgreSQL "
        f"{meta['server_version']} · Claude Code {report.get('claude_code_version')} · "
        f"{meta['generated_at']} · 987 tables in scope for both, read-only role · "
        f"{report['repeats']} repeat(s)",
        "",
        f"Limits per question: Daita {limits['daita']['max_steps']} steps, "
        f"{limits['daita']['max_total_tokens']} tokens, "
        f"{limits['daita']['max_wall_time_seconds']} s; Claude Code "
        f"${limits['claude_code']['max_budget_usd']}.",
        "",
        "| System | Model | Runs | Correct | Confident-wrong | No answer | "
        "Incomplete | Mean s | Turns | Tool calls | Tokens | Input tokens/turn | "
        "Cost USD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for name, system in systems.items():
        summary = system["summary"]
        outcomes = summary["outcomes"]
        lines.append(
            f"| {name} | {system['model']} | {summary['runs']} | "
            + " | ".join(str(outcomes[item]) for item in OUTCOMES)
            + f" | {summary['mean_run_seconds']} | {summary['total_turns']} | "
            f"{summary['total_tool_calls']} | {summary['total_tokens']} | "
            f"{summary['mean_input_tokens_per_turn']} | "
            f"{summary['total_cost_usd'] or 'unknown'} |"
        )
    names = list(systems)
    lines += [
        "",
        "| Question | Category | "
        + " | ".join(f"{name} correct | {name} outcomes" for name in names)
        + " |",
        "| --- | --- | " + " | ".join("---: | ---" for _ in names) + " |",
    ]
    for question in report["question_ids"]:
        cells = []
        for name in names:
            runs = [run for run in systems[name]["questions"] if run["id"] == question]
            correct = sum(1 for run in runs if run["outcome"] == "correct")
            outcomes = ", ".join(
                sorted(run["outcome"] for run in runs if run["outcome"] != "correct")
            )
            cells.append(f"{correct}/{len(runs)} | {outcomes or '—'}")
        lines.append(
            f"| {question} | {report['categories'][question]} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "Scoring reads the requested last line: correct when it holds the known "
        "value; no answer when the run declines or asks for clarification; "
        "incomplete when a limit or error stopped the run; otherwise "
        "confidently wrong. It is a keyword rule, not a human label; "
        "`answer_elsewhere` in the JSON flags answers outside the last line.",
    ]
    return "\n".join(lines) + "\n"


@pytest.mark.requires_llm
@pytest.mark.requires_network
@pytest.mark.skipif(
    os.environ.get(_LIVE_GATE) != "1",
    reason=f"set {_LIVE_GATE}=1 to authorize live model cost for the comparison",
)
async def test_daita_and_claude_code_answer_the_question_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id = os.environ.get(_MODEL_ID, "anthropic:claude-sonnet-5")
    provider, _, model = model_id.partition(":")
    systems = [
        name
        for name in ("daita", "claude_code")
        if name
        in {
            item.strip()
            for item in os.environ.get(_SYSTEMS, "daita,claude_code").split(",")
        }
    ]
    if not systems:
        pytest.fail(f"{_SYSTEMS} must list daita and/or claude_code")
    if "claude_code" in systems and provider != "anthropic":
        pytest.fail(
            f"{_MODEL_ID} must be an anthropic: model to compare with Claude Code"
        )
    repeats = int(os.environ.get(_REPEATS, "1"))
    if not 1 <= repeats <= 5:
        pytest.fail(f"{_REPEATS} must be from 1 through 5")
    target, admin, trips, environment, questions, truth = await _prepare(monkeypatch)
    await admin.close()
    limits = _daita_limits()
    report: dict[str, Any] = {
        "environment": environment,
        "claude_code_version": (
            claude_code_version(_CLAUDE) if "claude_code" in systems else None
        ),
        "repeats": repeats,
        "limits": {
            "daita": {
                "max_steps": limits.max_steps,
                "max_total_tokens": limits.max_total_tokens,
                "max_wall_time_seconds": limits.max_wall_time_seconds,
            },
            "claude_code": {"max_budget_usd": os.environ.get(_MAX_COST, "0.50")},
        },
        "question_ids": [item["id"] for item in questions],
        "categories": {item["id"]: item["category"] for item in questions},
        "systems": {
            name: {
                "model": model_id if name == "daita" else model,
                "attach": [],
                "questions": [],
            }
            for name in systems
        },
    }
    directory = _report_directory()
    stem = f"postgresql-schema-comparison-{environment['commit']}-{model}"
    try:
        # Repeats outermost so a mid-run failure leaves both systems comparable.
        for repeat in range(repeats):
            for name in systems:
                entry = report["systems"][name]
                root = tmp_path / name / f"repeat-{repeat}"
                if name == "daita":
                    await _run_daita(
                        target, model_id, questions, truth, trips, root, entry, repeat
                    )
                else:
                    await run_claude_code(
                        model,
                        questions,
                        truth,
                        root,
                        entry,
                        repeat,
                        executable_env=_CLAUDE,
                        preamble=(
                            "A PostgreSQL database is reachable with psql; its "
                            "connection settings are already in the environment. "
                            "Answer from that database: "
                        ),
                        environment={
                            "PGHOST": target.host,
                            "PGPORT": str(target.port),
                            "PGDATABASE": _DATABASE,
                            "PGUSER": _READER[0],
                            "PGPASSWORD": _READER[1],
                            "PGSSLMODE": target.ssl_mode,
                        },
                        allowed_tools="Bash(psql:*)",
                        max_cost=os.environ.get(_MAX_COST, "0.50"),
                    )
    finally:
        # Paid results survive a later failure.
        for entry in report["systems"].values():
            entry["summary"] = system_summary(entry["questions"])
        (directory / f"{stem}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        (directory / f"{stem}.md").write_text(_comparison_markdown(report))
        print(f"\ncomparison written to {directory / stem}.md")
    for name, system in report["systems"].items():
        assert len(system["questions"]) == len(questions) * repeats, name
