"""Production native containment and fault cleanup with real OS processes."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.adapters.analytical_workspace import NativePythonWorker
from daita.adapters.analytical_workspace.recovery import recover_generation
from tests.support.analysis import facts as evidence_facts, terminal
from tests.support.analysis_crash import HOST
from tests.support.workspace import workspace_for


async def test_installed_scientific_libraries_and_duckdb():
    worker = NativePythonWorker()
    try:
        await worker.start()
        result = await worker.execute(
            "import numpy, pandas, scipy, networkx, pyarrow, matplotlib, duckdb\n"
            "import matplotlib.pyplot as plt\n"
            "print(duckdb.sql('select sum(i) from range(10) t(i)').fetchone()[0])\n"
            "plt.plot([1,2],[3,4]); plt.savefig('plot.png')",
            deadline=asyncio.get_running_loop().time() + 30,
        )
        assert result["status"] == "success", result
        assert result["stdout"] == "45\n"
        assert (worker.scratch / "plot.png").is_file()
    finally:
        await worker.close()


async def test_native_filesystem_network_and_descendant_denials(tmp_path):
    private = tmp_path / "private.txt"
    private.write_text("private fixture")
    worker = NativePythonWorker()
    try:
        await worker.start()
        code = (
            "import os, socket\n"
            f"attempts = [lambda: open({str(private)!r}).read(), lambda: socket.create_connection(('127.0.0.1', 9)), os.fork]\n"
            "for attempt in attempts:\n"
            "    try:\n        attempt()\n    except OSError:\n        print('denied')\n"
            "    else:\n        raise AssertionError('native escape permitted')"
        )
        result = await worker.execute(
            code, deadline=asyncio.get_running_loop().time() + 5
        )
        assert result["status"] == "success", result
        assert result["stdout"] == "denied\ndenied\ndenied\n"
    finally:
        await worker.close()


async def test_real_timeout_retains_cpu_and_deletes():
    worker = NativePythonWorker()
    try:
        await worker.start()
        with pytest.raises(TimeoutError):
            await worker.execute(
                "while True: pass", deadline=asyncio.get_running_loop().time() + 0.2
            )
    finally:
        cleanup = await worker.close()
    assert worker.usage["cpu_complete"] is True
    assert worker.usage["user_cpu_seconds"] > 0
    assert worker.usage["wall_seconds"] >= 0.2
    assert cleanup["scratch_deleted"] is True
    assert cleanup["process_reaped"] is True


async def test_guardian_reaps_worker_and_deletes_when_host_is_killed():
    code = """
import asyncio, json
from daita.adapters.analytical_workspace import NativePythonWorker
async def main():
    w = NativePythonWorker()
    await w.start()
    print(json.dumps({'pid': w.worker_pid, **w.identity}), flush=True)
    await w.execute('while True: pass', deadline=asyncio.get_running_loop().time()+30)
asyncio.run(main())
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        assert process.stdout is not None
        facts = json.loads(await asyncio.to_thread(process.stdout.readline))
        assert facts["closure_path"] is None
        process.kill()
        await asyncio.to_thread(process.wait, 5)
        deadline = asyncio.get_running_loop().time() + 5
        while (
            Path(facts["scratch"]).exists()
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.02)
        assert not Path(facts["scratch"]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(facts["pid"], 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


@pytest.mark.parametrize(
    "boundary",
    [
        "allocation",
        "handoff",
        "guardian_inherited",
        "before_worker",
        "after_worker",
        "before_pid",
        "after_pid",
    ],
)
async def test_real_launch_guard_survives_host_death_at_every_handoff(
    boundary, tmp_path
):
    import fcntl

    process = subprocess.Popen(
        [sys.executable, "-c", HOST, str(tmp_path), boundary],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    facts = None
    assert process.stderr is not None
    try:
        deadline = asyncio.get_running_loop().time() + 20
        while not (tmp_path / "checkpoint").exists():
            assert process.poll() is None, process.stderr.read().decode()
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.02)
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        facts = metadata["facts"]
        process.kill()
        await asyncio.to_thread(process.wait, 5)
        if boundary in {"guardian_inherited", "before_worker", "after_worker"}:
            fd = os.open(facts["closure_path"], os.O_RDWR | os.O_NOFOLLOW)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        (tmp_path / "release").touch()
        for _ in range(2):
            agent = await Agent.open(
                "recovery", root=tmp_path, workspace=workspace_for(tmp_path)
            )
            try:
                transcript = await agent.transcript(metadata["run_id"])
                result = await terminal(agent, transcript.run.conversation_id)
                assert result.kind.value == "interrupted"
                generation = next(
                    r
                    for r in await agent.analysis_evidence(metadata["run_id"])
                    if r.kind == "generation"
                )
                recovered = evidence_facts(generation)["recovery"]
                assert (
                    __import__("json").loads(
                        __import__(
                            "daita._json", fromlist=["canonical_json"]
                        ).canonical_json(recovered)
                    )["usage"]["cpu_complete"]
                    is True
                )
                assert recovered["cleanup"]["process_reaped"] is True
                assert recovered["cleanup"]["scratch_deleted"] is True
                if boundary in {"allocation", "handoff", "guardian_inherited"}:
                    assert recovered["cleanup"]["process_spawned"] is False
                    assert recovered["usage"]["user_cpu_seconds"] == 0
                else:
                    assert recovered["cleanup"]["process_spawned"] is True
                    with pytest.raises(ProcessLookupError):
                        os.kill(recovered["pid"], 0)
            finally:
                await agent.close()
        assert not Path(facts["scratch"]).exists()
        assert not Path(facts["closure_path"]).exists()
    finally:
        (tmp_path / "release").touch()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stderr.close()
        if facts and Path(facts["closure_path"]).exists():
            await recover_generation(facts)
            Path(facts["closure_path"]).unlink()


@pytest.mark.parametrize(
    "tamper", ["nonce", "inode", "symlink", "phase", "empty", "unknown_cpu", "guard"]
)
async def test_native_recovery_requires_exact_identity_phase_and_ownership(
    tamper, tmp_path
):
    worker = NativePythonWorker(durable_closure=True)
    path = Path(str(worker.identity["closure_path"]))
    replacement = tmp_path / "original-proof"
    try:
        if tamper == "guard":
            with pytest.raises(RuntimeError, match="ownership proof unavailable"):
                await recover_generation(worker.identity)
            assert worker.scratch.exists()
            return
        await worker.start()
        await worker.close()
        raw = path.read_bytes()
        if tamper in {"inode", "symlink"}:
            path.rename(replacement)
            if tamper == "inode":
                path.write_bytes(raw)
            else:
                path.symlink_to(replacement)
        elif tamper == "empty":
            path.write_bytes(b"")
        else:
            facts = json.loads(raw)
            if tamper == "nonce":
                facts["closure_nonce"] = "0" * 64
            elif tamper == "phase":
                facts["kind"] = "launch_committed"
            else:
                facts["user_cpu_seconds"] = None
            path.write_text(json.dumps(facts))
        with pytest.raises((ValueError, RuntimeError, OSError)):
            await recover_generation(worker.identity)
        if replacement.exists():
            path.unlink()
            replacement.rename(path)
        else:
            path.write_bytes(raw)
        recovered = await recover_generation(worker.identity)
        assert json.loads(canonical_json(recovered))["usage"]["cpu_complete"] is True
    finally:
        await worker.close()
        worker.dispose_closure_evidence()


async def test_failed_native_allocation_disposes_its_exact_resources(monkeypatch):
    import daita.adapters.analytical_workspace.native as native

    original_directory = native.tempfile.mkdtemp
    allocated = []

    def directory(*args, **kwargs):
        path = original_directory(*args, **kwargs)
        allocated.append(Path(path))
        return path

    def fail(*args, **kwargs):
        raise OSError("closure allocation failed")

    monkeypatch.setattr(native.tempfile, "mkdtemp", directory)
    monkeypatch.setattr(native.tempfile, "mkstemp", fail)
    with pytest.raises(OSError, match="closure allocation"):
        NativePythonWorker(durable_closure=True)
    assert allocated and all(not path.exists() for path in allocated)


async def test_real_guardian_exec_failure_proves_nonspawn_before_releasing(monkeypatch):
    original = subprocess.Popen

    def exit_before_entry(args, *other, **kwargs):
        return original([sys.executable, "-c", "raise SystemExit(1)"], *other, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", exit_before_entry)
    worker = NativePythonWorker(durable_closure=True)
    try:
        with pytest.raises((RuntimeError, BrokenPipeError)):
            await worker.start()
        cleanup = await worker.close()
        assert cleanup["process_spawned"] is False
        assert cleanup["process_reaped"] is True
        assert cleanup["scratch_deleted"] is True
        assert worker.usage["cpu_complete"] is True
        assert worker.usage["user_cpu_seconds"] == 0
    finally:
        await worker.close()
        worker.dispose_closure_evidence()
