"""Soak for combined-query budgets: real agent, SQLite and folder inputs, one child per case.

Each case runs ``data_query`` with ``inputs`` through ``Agent.create`` and a scripted
model in its own subprocess, so peak RSS is its own. Results go to a JSON file under
``DAITA_SOAK_REPORT_DIR`` and a Markdown table on stdout (run with ``-s``).
``DAITA_SOAK_MATRIX=1`` adds the 25k/50k/100k budget comparison.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from daita import Agent, FolderSource, SQLiteSource
from daita._json import FrozenJsonObject, canonical_json, thaw_json
from daita.domains.data import capabilities as caps
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.slow

MIB = 1_024 * 1_024
SHIPPED = (caps.COMBINED_QUERY_TOTAL_MAX_ROWS, caps.COMBINED_QUERY_TOTAL_MAX_BYTES)
BUDGETS = ((25_000, 16 * MIB), (50_000, 32 * MIB), (100_000, 64 * MIB))
# wide: ~1 KiB rows; fill: share of the budget the inputs total; skew: one input
# holds 85% of the rows; folder: input 0 is a CSV folder file.
SHAPES: dict[str, dict[str, Any]] = {
    "narrow_2": dict(wide=False, sources=2, fill=0.9),
    "narrow_16": dict(wide=False, sources=16, fill=0.9),
    "wide_2": dict(wide=True, sources=2, fill=0.9),
    "wide_16": dict(wide=True, sources=16, fill=0.9),
    "skewed_16": dict(wide=False, sources=16, fill=0.9, skew=True),
    "folder_mixed": dict(wide=False, sources=4, fill=0.9, folder=True),
    "over_budget": dict(wide=False, sources=2, fill=1.1),
    "over_budget_wide": dict(wide=True, sources=2, fill=1.1),
}
RESULTS: list[dict[str, Any]] = []


def _columns(wide: bool) -> str:
    return "id, k, v, w, name, region" + (", pad" if wide else "")


def _row(i: int, wide: bool) -> tuple[Any, ...]:
    base = (
        i,
        i % 10,
        i % 7,
        i * 3 % 11,
        f"customer-{i % 9973:05d}",
        f"region-{i % 13:02d}",
    )
    return base + (("%08x" % i) * 112,) if wide else base  # pad: 896 chars


def _plan(shape: str, rows: int, size: int) -> dict[str, Any]:
    """Source layout, SQL and the expected answer, computed from the row formulas."""
    spec = SHAPES[shape]
    wide, count = spec["wide"], spec["sources"]
    row_bytes = (
        len(canonical_json(dict(zip(_columns(wide).split(", "), _row(50_001, wide)))))
        + 1
    )
    total = int(spec["fill"] * min(rows, size // row_bytes))
    if spec.get("skew"):
        big = int(total * 0.85)
        small = [(total - big) // (count - 1)] * (count - 1)
        counts = [big, *small]
    else:
        counts = [total // count + (i < total % count) for i in range(count)]
    join = count == 2
    if join:
        counts = [total // 2] * 2
    starts = [0] * count if join else [sum(counts[:i]) for i in range(count)]
    sources = [
        ("csv" if spec.get("folder") and i == 0 else "sqlite", starts[i], counts[i])
        for i in range(count)
    ]
    extra = ", SUM(length({p}pad)) AS p" if wide else ""
    if join:
        sql = (
            "SELECT a.k, COUNT(*) AS c, SUM(a.v + b.w) AS t"
            + extra.format(p="a.")
            + " FROM a JOIN b ON a.id = b.id GROUP BY a.k ORDER BY a.k"
        )
        aliases = ["a", "b"]
        ids = range(counts[0])
    else:
        aliases = [f"s{i}" for i in range(count)]
        union = " UNION ALL ".join(f"SELECT * FROM {alias}" for alias in aliases)
        sql = (
            "SELECT k, COUNT(*) AS c, SUM(v + w) AS t"
            + extra.format(p="")
            + f" FROM ({union}) GROUP BY k ORDER BY k"
        )
        ids = range(total)
    groups: dict[int, list[int]] = {}
    for i in ids:
        row = _row(i, wide)
        agg = groups.setdefault(row[1], [0, 0, 0])
        agg[0] += 1
        agg[1] += row[2] + row[3]
        agg[2] += len(row[6]) if wide else 0
    expected = [
        {"k": k, "c": a[0], "t": a[1], **({"p": a[2]} if wide else {})}
        for k, a in sorted(groups.items())
    ]
    return dict(
        wide=wide,
        sources=sources,
        aliases=aliases,
        sql=sql,
        total_rows=sum(counts),
        expected=None if spec["fill"] > 1 else expected,
    )


def _seed(cache: Path, kind: str, start: int, n: int, wide: bool, slot: int) -> str:
    """Create (once) the data file for ids start..start+n; returns its path."""
    name = f"{kind}_{start}_{n}_{int(wide)}_{slot}"
    path = (
        cache
        / (name + (".sqlite" if kind == "sqlite" else ""))
        / ("t.csv" if kind == "csv" else "")
    )
    if kind == "sqlite":
        if not path.exists():
            marks = ", ".join("?" * len(_columns(wide).split(", ")))
            with sqlite3.connect(path) as db:
                db.execute(f"CREATE TABLE t ({_columns(wide)})")
                db.executemany(
                    f"INSERT INTO t VALUES ({marks})",
                    (_row(i, wide) for i in range(start, start + n)),
                )
        return str(path)
    if not path.exists():
        path.parent.mkdir()
        with path.open("w", newline="") as handle:
            out = csv.writer(handle)
            out.writerow(_columns(wide).split(", "))
            out.writerows(_row(i, wide) for i in range(start, start + n))
    return str(path.parent)


def _call(arguments: dict[str, Any]) -> ModelResponse:
    call = ToolCall(id="q", name="data_query", arguments=arguments)
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,))


async def _child(spec: dict[str, Any]) -> dict[str, Any]:
    from daita.adapters import local_file_query as lfq

    caps.COMBINED_QUERY_TOTAL_MAX_ROWS, caps.COMBINED_QUERY_TOTAL_MAX_BYTES = (
        spec["rows"],
        spec["bytes"],
    )
    clock = time.perf_counter
    marks: dict[str, float] = {}
    phase = ["read"]
    workers: dict[int, list[Any]] = {}
    scratch = [0]
    real_rss, real_dir = lfq._process_rss_bytes, lfq._directory_bytes
    real_execute, real_relations = (
        caps.DataQueryCombinedExecutor.execute,
        lfq.LocalFileQueryBackend.query_relations,
    )

    def rss(pid: int | None) -> int | None:
        value = real_rss(pid)
        if value is not None and pid is not None:
            seen = workers.setdefault(pid, [phase[0], 0])
            seen[1] = max(seen[1], value)
        return value

    def directory(path: Path) -> int:
        scratch[0] = max(scratch[0], real_dir(path))
        return scratch[0]

    async def execute(self: Any, request: Any) -> Any:
        marks["exec0"] = clock()
        try:
            return await real_execute(self, request)
        finally:
            marks["exec1"] = clock()

    async def relations(self: Any, **kwargs: Any) -> Any:
        marks["engine0"] = clock()
        phase[0] = "engine"
        try:
            return await real_relations(self, **kwargs)
        finally:
            marks["engine1"] = clock()

    lfq._process_rss_bytes, lfq._directory_bytes = rss, directory
    caps.DataQueryCombinedExecutor.execute = execute  # type: ignore[method-assign]
    lfq.LocalFileQueryBackend.query_relations = relations  # type: ignore[method-assign]

    work = Path(tempfile.mkdtemp(prefix="soak-child-")).resolve()
    provider = ToolboxAwareMockModelProvider((), provider_id="mock:soak")
    agent = await Agent.create(
        "soak",
        root=work,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        workspace=workspace_for(work),
    )
    try:
        inputs = []
        for alias, (kind, path) in zip(spec["aliases"], spec["files"]):
            if kind == "csv":
                source = await agent.attach(FolderSource(path))
                inputs.append(
                    dict(
                        alias=alias,
                        source_id=source.id,
                        path_pattern="t.csv",
                        sql=f"SELECT {spec['cols']} FROM data",
                    )
                )
                continue
            source = await agent.attach(SQLiteSource(path))
            (resource_,) = await agent.list_catalog_resources(source_id=source.id)
            inputs.append(
                dict(
                    alias=alias,
                    source_id=source.id,
                    resource_ids=[resource_.id],
                    sql=f"SELECT {spec['cols']} FROM t",
                )
            )
        provider.replace_script(
            (
                _call({"inputs": inputs, "sql": spec["sql"]}),
                ModelResponse(finish_reason=FinishReason.STOP, text="done"),
            )
        )
        stall = [0.0, 0.0]
        started = clock()

        async def beat() -> None:
            last = clock()
            while True:
                await asyncio.sleep(0.01)
                now = clock()
                if now - last - 0.01 > stall[0]:
                    stall[:] = [now - last - 0.01, now - started]
                last = now

        heartbeat = asyncio.create_task(beat())
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = await agent.run("go")
        wall = clock() - started
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        heartbeat.cancel()
        transcript = await agent.transcript(result.run_id)
        block = next(
            b
            for m in transcript.messages
            if m.role is MessageRole.TOOL
            for b in m.content
            if isinstance(b, ToolResultBlock) and b.call_id == "q"
        )
    finally:
        await agent.close()
        shutil.rmtree(work, ignore_errors=True)
    unit = 1 if sys.platform == "darwin" else 1_024
    out_ = json.loads(
        canonical_json(thaw_json(FrozenJsonObject.from_mapping(block.output)))
    )
    reads_end = marks.get("engine0", marks.get("exec1", started))
    out: dict[str, Any] = dict(
        outcome=(
            "success"
            if not block.is_error
            else str((out_.get("error") or out_).get("code"))
        ),
        wall_s=wall,
        read_phase_s=reads_end - marks["exec0"],
        engine_s=marks["engine1"] - marks["engine0"] if "engine1" in marks else None,
        exec_s=marks["exec1"] - marks["exec0"],
        parent_rss_delta=(after - before) * unit,
        parent_rss_peak=after * unit,
        workers=[dict(phase=v[0], rss=v[1]) for v in workers.values()],
        scratch_peak=scratch[0],
        stall_ms=stall[0] * 1_000,
        stall_at_s=stall[1],
        read_end_at_s=reads_end - started,
    )
    if not block.is_error:
        data = out_["data"]
        reads = [item["read_ms"] / 1_000 for item in data["inputs"]]
        out.update(
            rows=data["rows"],
            read_sum_s=sum(reads),
            read_max_s=max(reads),
            input_rows=[item["rows"] for item in data["inputs"]],
        )
    return out


def main(argument: str) -> None:
    print("RESULT " + json.dumps(asyncio.run(_child(json.loads(argument)))))


@pytest.fixture(scope="module")
def cache() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="soak-seed-") as directory:
        yield Path(directory).resolve()


@pytest.fixture(scope="module", autouse=True)
def report() -> Iterator[None]:
    yield
    target = Path(
        os.environ.get("DAITA_SOAK_REPORT_DIR")
        or tempfile.mkdtemp(prefix="soak-report-")
    )
    target.mkdir(parents=True, exist_ok=True)
    cols = "budget shape outcome correct wall_s read_sum_s read_max_s read_phase_s engine_s parent_rss_delta_mib worker_rss_mib scratch_mib stall_ms input_margin_s total_margin_s worker_margin_s".split()
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in RESULTS:
        m = r["metrics"]
        w = max((x["rss"] for x in m["workers"]), default=0) / MIB

        def f(value: Any, spec: str = ".2f") -> str:
            return "-" if value is None else format(value, spec)

        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    (
                        f"{r['rows']}/{r['bytes'] // MIB}MiB",
                        r["shape"],
                        m["outcome"],
                        r["correct"],
                        f(m["wall_s"]),
                        f(m.get("read_sum_s")),
                        f(m.get("read_max_s")),
                        f(m["read_phase_s"]),
                        f(m["engine_s"]),
                        f(m["parent_rss_delta"] / MIB, ".0f"),
                        f"{w:.0f} (n={len(m['workers'])})",
                        f(m["scratch_peak"] / MIB, ".1f"),
                        f(m["stall_ms"], ".0f"),
                        f(None if "read_max_s" not in m else 5 - m["read_max_s"]),
                        f(15 - m["read_phase_s"]),
                        f(None if m["engine_s"] is None else 30 - m["engine_s"]),
                    ),
                )
            )
            + " |"
        )
    path = target / "combined_query_soak.json"
    path.write_text(json.dumps(RESULTS, indent=2, default=str))
    (target / "combined_query_soak.md").write_text("\n".join(lines) + "\n")
    print(f"\nsoak report: {path}\n" + "\n".join(lines))


def _run_case(cache: Path, shape: str, rows: int, size: int) -> None:
    plan = _plan(shape, rows, size)
    files = [
        (kind, _seed(cache, kind, start, n, plan["wide"], slot))
        for slot, (kind, start, n) in enumerate(plan["sources"])
    ]
    spec = dict(
        rows=rows,
        bytes=size,
        aliases=plan["aliases"],
        files=files,
        sql=plan["sql"],
        cols=_columns(plan["wide"]),
    )
    done = subprocess.run(
        [sys.executable, "-m", __name__, json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    line = next((x for x in done.stdout.splitlines() if x.startswith("RESULT ")), None)
    assert line, done.stderr[-2_000:]
    metrics = json.loads(line[7:])
    expected = plan["expected"]
    if expected is None:
        correct = (
            metrics["outcome"] == "combined_input_truncated" and "rows" not in metrics
        )
    else:
        got = [{k: int(v) for k, v in row.items()} for row in metrics.get("rows", ())]
        correct = metrics["outcome"] == "success" and got == expected
    RESULTS.append(
        dict(
            shape=shape,
            rows=rows,
            bytes=size,
            input_rows=plan["total_rows"],
            correct=correct,
            metrics=metrics,
        )
    )
    assert correct, (metrics["outcome"], done.stderr[-1_000:])


@pytest.mark.parametrize("shape", SHAPES)
def test_shipped_budget_answers_correctly(cache: Path, shape: str) -> None:
    _run_case(cache, shape, *SHIPPED)


@pytest.mark.skipif(
    not os.environ.get("DAITA_SOAK_MATRIX"), reason="set DAITA_SOAK_MATRIX=1"
)
@pytest.mark.parametrize("budget", BUDGETS, ids=lambda b: f"{b[0] // 1000}k")
@pytest.mark.parametrize("shape", SHAPES)
def test_budget_matrix(cache: Path, shape: str, budget: tuple[int, int]) -> None:
    _run_case(cache, shape, *budget)


if __name__ == "__main__":
    main(sys.argv[1])
