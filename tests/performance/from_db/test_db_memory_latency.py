"""Local structured DB memory latency benchmarks.

Run:
    PYTHONDONTWRITEBYTECODE=1 pytest \
        tests/performance/from_db/test_db_memory_latency.py \
        -m performance -q -s

Optional:
    DAITA_DB_MEMORY_BENCH_OPS=100
    DAITA_DB_MEMORY_WRITE_P95_MS=100
    DAITA_DB_MEMORY_RECALL_P95_MS=75
    DAITA_DB_MEMORY_ROUND_TRIP_P95_MS=150
    DAITA_DB_MEMORY_RUNTIME_ROUND_TRIP_P95_MS=250
    DAITA_PERF_OUTPUT_DIR=.daita/benchmarks
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Awaitable, Callable

import pytest

from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.memory import (
    DBMemoryRecord,
    recall_db_memory_records,
    write_db_memory_record,
)
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.memory.memory_plugin import MemoryPlugin

pytestmark = pytest.mark.performance


async def test_structured_db_memory_write_and_recall_latency(tmp_path):
    operations = _env_int("DAITA_DB_MEMORY_BENCH_OPS", 50)
    warmup = min(5, max(1, operations // 10))
    source_identity = "sqlite:from_db:memory-latency-bench"
    memory = _memory_plugin(tmp_path, source_identity)

    for index in range(warmup):
        await _write_record(memory, source_identity, index=index, prefix="warmup")
        await recall_db_memory_records(
            memory,
            "latency warmup revenue orders.total",
            kinds=["metric_definition"],
            limit=3,
            score_threshold=0.0,
        )

    write_samples: list[float] = []
    recall_samples: list[float] = []
    round_trip_samples: list[float] = []
    failures: list[dict[str, Any]] = []

    for index in range(operations):
        marker = f"latency_metric_{index:04d}"
        round_trip_started = time.perf_counter()
        write_elapsed, write_result = await _timed(
            _write_record(
                memory,
                source_identity,
                index=index,
                prefix=marker,
            )
        )
        recall_elapsed, recall_result = await _timed(
            recall_db_memory_records(
                memory,
                f"{marker} revenue orders.total",
                kinds=["metric_definition"],
                limit=3,
                score_threshold=0.0,
            )
        )
        round_trip_samples.append(_elapsed_ms(round_trip_started))
        write_samples.append(write_elapsed)
        recall_samples.append(recall_elapsed)

        expected_key = f"metric:{marker}"
        keys = [
            item.get("metadata", {}).get("db_memory", {}).get("key")
            for item in recall_result
        ]
        if write_result.get("status") not in {"created", "updated"}:
            failures.append({"index": index, "reason": "write_failed", "keys": keys})
        elif expected_key not in keys:
            failures.append(
                {
                    "index": index,
                    "reason": "recall_missed_written_record",
                    "expected": expected_key,
                    "keys": keys,
                }
            )

    artifact = {
        "suite": "db-memory-latency",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "operations": operations,
            "warmup": warmup,
            "backend": "local_sqlite_fts5",
            "embedding": "disabled",
        },
        "summary": {
            "write_ms": _summary(write_samples),
            "recall_ms": _summary(recall_samples),
            "round_trip_ms": _summary(round_trip_samples),
            "failure_count": len(failures),
        },
        "failures": failures[:10],
    }
    _write_artifact(tmp_path, artifact)
    print("[benchmark]", json.dumps(artifact["summary"], sort_keys=True))

    assert not failures, failures[:3]
    assert artifact["summary"]["write_ms"]["p95"] < _env_float(
        "DAITA_DB_MEMORY_WRITE_P95_MS", 100.0
    )
    assert artifact["summary"]["recall_ms"]["p95"] < _env_float(
        "DAITA_DB_MEMORY_RECALL_P95_MS", 75.0
    )
    assert artifact["summary"]["round_trip_ms"]["p95"] < _env_float(
        "DAITA_DB_MEMORY_ROUND_TRIP_P95_MS", 150.0
    )


async def test_runtime_capability_db_memory_write_and_recall_latency(tmp_path):
    operations = _env_int("DAITA_DB_MEMORY_BENCH_OPS", 50)
    warmup = min(5, max(1, operations // 10))
    source_identity = "sqlite:from_db:memory-runtime-latency-bench"
    memory = _memory_plugin(tmp_path, source_identity)
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            plugins=(memory,),
            metadata={
                "from_db_options": {
                    "memory": {
                        "enabled": True,
                        "workspace_scope": "source",
                        "source_identity": source_identity,
                    }
                }
            },
        )
    )

    for index in range(warmup):
        marker = f"runtime_warmup_{index:04d}"
        await runtime.execute_capability(
            "memory.semantic.write",
            owner="memory",
            operation_type="memory.update",
            input={
                "db_memory_payload": _record_payload(
                    source_identity,
                    index=index,
                    prefix=marker,
                ),
                "db_memory_prompt": f"{marker} revenue",
            },
        )
        await runtime.execute_capability(
            "memory.semantic.recall",
            owner="memory",
            operation_type="memory.recall",
            input={
                "query": f"{marker} revenue orders.total",
                "category": "db_semantics",
                "source_identity": source_identity,
                "kinds": ["metric_definition"],
                "limit": 3,
                "score_threshold": 0.0,
                "retrieval_mode": "structured",
            },
        )

    write_samples: list[float] = []
    recall_samples: list[float] = []
    round_trip_samples: list[float] = []
    failures: list[dict[str, Any]] = []

    for index in range(operations):
        marker = f"runtime_latency_metric_{index:04d}"
        round_trip_started = time.perf_counter()
        write_elapsed, write_evidence = await _timed(
            runtime.execute_capability(
                "memory.semantic.write",
                owner="memory",
                operation_type="memory.update",
                input={
                    "db_memory_payload": _record_payload(
                        source_identity,
                        index=index,
                        prefix=marker,
                    ),
                    "db_memory_prompt": f"{marker} revenue",
                },
            )
        )
        recall_elapsed, recall_evidence = await _timed(
            runtime.execute_capability(
                "memory.semantic.recall",
                owner="memory",
                operation_type="memory.recall",
                input={
                    "query": f"{marker} revenue orders.total",
                    "category": "db_semantics",
                    "source_identity": source_identity,
                    "kinds": ["metric_definition"],
                    "limit": 3,
                    "score_threshold": 0.0,
                    "retrieval_mode": "structured",
                },
            )
        )
        round_trip_samples.append(_elapsed_ms(round_trip_started))
        write_samples.append(write_elapsed)
        recall_samples.append(recall_elapsed)

        write_payload = write_evidence[0].payload if write_evidence else {}
        recall_payload = recall_evidence[0].payload if recall_evidence else {}
        expected_key = f"metric:{marker}"
        keys = [
            item.get("metadata", {}).get("db_memory", {}).get("key")
            for item in recall_payload.get("results", [])
        ]
        if write_payload.get("success") is not True:
            failures.append({"index": index, "reason": "write_failed"})
        elif expected_key not in keys:
            failures.append(
                {
                    "index": index,
                    "reason": "recall_missed_written_record",
                    "expected": expected_key,
                    "keys": keys,
                }
            )

    artifact = {
        "suite": "db-memory-runtime-latency",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "operations": operations,
            "warmup": warmup,
            "backend": "local_sqlite_fts5",
            "embedding": "disabled",
            "path": "runtime.execute_capability",
        },
        "summary": {
            "write_ms": _summary(write_samples),
            "recall_ms": _summary(recall_samples),
            "round_trip_ms": _summary(round_trip_samples),
            "failure_count": len(failures),
        },
        "failures": failures[:10],
    }
    _write_artifact(tmp_path, artifact)
    print("[benchmark]", json.dumps(artifact["summary"], sort_keys=True))

    assert not failures, failures[:3]
    assert artifact["summary"]["round_trip_ms"]["p95"] < _env_float(
        "DAITA_DB_MEMORY_RUNTIME_ROUND_TRIP_P95_MS", 250.0
    )


def _memory_plugin(tmp_path: Path, source_identity: str) -> MemoryPlugin:
    memory = MemoryPlugin(
        auto_curate="manual",
        db_memory_mode=True,
        db_memory_retrieval_mode="structured",
    )
    memory.backend = LocalMemoryBackend(
        workspace=source_identity,
        agent_id="memory-latency-bench",
        scope="project",
        base_dir=tmp_path,
        embedder=None,
        default_source_identity=source_identity,
    )
    return memory


async def _write_record(
    memory: MemoryPlugin,
    source_identity: str,
    *,
    index: int,
    prefix: str,
) -> dict[str, Any]:
    return await write_db_memory_record(
        memory,
        DBMemoryRecord(**_record_payload(source_identity, index=index, prefix=prefix)),
    )


def _record_payload(source_identity: str, *, index: int, prefix: str) -> dict[str, Any]:
    return {
        "kind": "metric_definition",
        "key": f"metric:{prefix}",
        "text": (
            f"{prefix} revenue is SUM(orders.total) for complete orders. "
            f"Latency benchmark row {index}."
        ),
        "metadata": {
            "source_identity": source_identity,
            "workspace_scope": "source",
            "active": True,
            "confidence": 0.95,
            "aliases": [f"{prefix} revenue"],
            "schema_refs": ["orders.total", "orders.status"],
        },
        "importance": 0.8,
    }


async def _timed(awaitable: Awaitable[Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = await awaitable
    return _elapsed_ms(started), result


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "count": float(len(ordered)),
        "mean": round(statistics.fmean(ordered), 3) if ordered else 0.0,
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": round(max(ordered), 3) if ordered else 0.0,
    }


def _percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return round(samples[0], 3)
    rank = (len(samples) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(samples) - 1)
    fraction = rank - lower
    value = samples[lower] + (samples[upper] - samples[lower]) * fraction
    return round(value, 3)


def _write_artifact(tmp_path: Path, artifact: dict[str, Any]) -> None:
    root = Path(os.environ.get("DAITA_PERF_OUTPUT_DIR", tmp_path))
    output_dir = root / "db-memory-latency"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "structured-db-memory-latency.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), "utf-8")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default
