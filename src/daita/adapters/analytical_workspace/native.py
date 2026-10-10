"""Launch and measure an OS-contained CPython worker without host shell code."""

from __future__ import annotations

import asyncio
import ctypes
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from ...config import AnalysisLimits
from ...hosting.execution_governor import PermitLease
from ..local_file_query import _darwin_proc_pidinfo, _ProcTaskInfo
from .cleanup import remove_owned_scratch
from .runtime import runtime_status


class _MachTimebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


@lru_cache(maxsize=1)
def _cpu_tick_seconds() -> float | None:
    function = ctypes.CDLL("/usr/lib/libSystem.B.dylib").mach_timebase_info
    function.argtypes = [ctypes.POINTER(_MachTimebase)]
    function.restype = ctypes.c_int
    value = _MachTimebase()
    if function(ctypes.byref(value)) != 0 or value.denom == 0:
        return None
    return value.numer / value.denom / 1e9


def available() -> bool:
    return runtime_status()["available"] is True


def _profile(scratch: Path) -> str:
    quoted = lambda path: json.dumps(str(path))
    reads = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()}
    return "\n".join(
        [
            "(version 1)",
            "(deny default)",
            "(deny process-info*)",
            '(allow sysctl-read (sysctl-name "hw.ncpu") (sysctl-name "hw.activecpu") (sysctl-name "hw.physicalcpu") (sysctl-name "hw.logicalcpu") (sysctl-name "hw.memsize") (sysctl-name "hw.machine") (sysctl-name "hw.cputype") (sysctl-name "hw.cpufamily") (sysctl-name "hw.pagesize") (sysctl-name "kern.ostype") (sysctl-name "kern.hostname") (sysctl-name "kern.osrelease") (sysctl-name "kern.osversion") (sysctl-name "kern.version") (sysctl-name "kern.maxfilesperproc"))',
            '(allow file-read* (subpath "/usr/lib") (subpath "/System/Library"))',
            '(allow file-read* file-write-data (literal "/dev/null"))',
            '(allow file-read* (literal "/dev/urandom") (literal "/dev/random"))',
            "(allow file-read* "
            + " ".join(f"(subpath {quoted(p)})" for p in reads)
            + ")",
            f"(allow file-read* file-write* (subpath {quoted(scratch)}))",
        ]
    )


class NativePythonWorker:
    """One private worker; all reported resource facts come from the host."""

    def __init__(
        self,
        *,
        limits: AnalysisLimits = AnalysisLimits(),
        durable_closure: bool = False,
    ) -> None:
        if not available():
            raise RuntimeError("Native analysis requires macOS ARM64")
        import fcntl

        self.limits = limits
        self.runtime = runtime_status()
        self.scratch = Path(tempfile.mkdtemp(prefix="daita-analysis-")).resolve()
        self.scratch.chmod(0o700)
        self.process: subprocess.Popen[bytes] | None = None
        self.worker_pid: int | None = None
        self._keepalive: int | None = None
        self._evidence: int | None = None
        self._evidence_buffer = bytearray()
        self._buffer = bytearray()
        self.usage: dict[str, Any] = {
            "measurement_method": "macOS proc_pidinfo samples",
            "sample_seconds": limits.sample_seconds,
            "user_cpu_seconds": None,
            "system_cpu_seconds": None,
            "peak_rss_bytes": None,
            "wall_seconds": 0.0,
            "broker_wait_seconds": 0.0,
            "peak_scratch_bytes": 0,
            "peak_scratch_files": 0,
            "input_bytes": 0,
            "output_bytes": 0,
            "protocol_input_bytes": 0,
            "protocol_output_bytes": 0,
            "scratch_samples_incomplete": False,
            "observed_peak_scratch_bytes": 0,
            "observed_peak_scratch_files": 0,
        }
        self.cleanup: dict[str, object] | None = None
        self._closed_evidence: dict[str, Any] | None = None
        self._closing: asyncio.Task[dict[str, object]] | None = None
        self.revalidate: Callable[[], Awaitable[None]] | None = None
        self.observe: Callable[[Mapping[str, object]], None] | None = None
        self._last_check = 0.0
        self._last_observation = 0.0
        self._created_at = asyncio.get_running_loop().time()
        self._closure_path: Path | None = None
        self._closure_descriptor = -1
        self._closure_nonce = secrets.token_hex(32)
        scratch_identity = self.scratch.lstat()
        try:
            self._durable_closure = durable_closure
            self._closure_descriptor, path = tempfile.mkstemp(
                prefix="daita-analysis-closure-"
            )
            self._closure_path = Path(path).resolve()
            # flock belongs to the inherited open-file description. The host and
            # guardian overlap ownership across fork/exec; recovery must open a new
            # description and acquire this exact lock before trusting launch phase.
            fcntl.flock(self._closure_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.identity = self.identity_facts()
            allocation = {
                "kind": "allocated",
                "durable_closure": durable_closure,
                "closure_nonce": self._closure_nonce,
                "scratch_device": self.identity["scratch_device"],
                "scratch_inode": self.identity["scratch_inode"],
            }
            os.write(self._closure_descriptor, json.dumps(allocation).encode())
            os.fsync(self._closure_descriptor)
            if not durable_closure:
                # Standalone workers have no durable recovery owner. An
                # anonymous locked inode gives the guardian the same handoff
                # without leaving a named proof when its host dies.
                self._closure_path.unlink()
                self._closure_path = None
                self.identity.update(
                    closure_path=None, closure_device=None, closure_inode=None
                )
        except BaseException:
            if self._closure_descriptor >= 0:
                created = os.fstat(self._closure_descriptor)
                os.close(self._closure_descriptor)
                self._closure_descriptor = -1
                if self._closure_path is not None:
                    current = self._closure_path.lstat()
                    if (created.st_dev, created.st_ino) != (
                        current.st_dev,
                        current.st_ino,
                    ):
                        raise RuntimeError("Failed native allocation identity changed")
                    self._closure_path.unlink()
            remove_owned_scratch(
                self.scratch, scratch_identity.st_dev, scratch_identity.st_ino
            )
            raise

    def identity_facts(self) -> dict[str, object]:
        facts = self.scratch.lstat()
        closure = None if self._closure_path is None else self._closure_path.lstat()
        return {
            "runtime": self.runtime,
            "memory_reservation_bytes": self.limits.memory_bytes,
            "scratch_reservation_bytes": self.limits.scratch_bytes,
            "scratch": str(self.scratch),
            "scratch_device": facts.st_dev,
            "scratch_inode": facts.st_ino,
            "closure_path": (
                None if self._closure_path is None else str(self._closure_path)
            ),
            "closure_device": None if closure is None else closure.st_dev,
            "closure_inode": None if closure is None else closure.st_ino,
            "closure_nonce": self._closure_nonce,
        }

    def dispose_closure_evidence(self) -> None:
        if self._closure_path is not None:
            try:
                facts = self._closure_path.lstat()
            except FileNotFoundError:
                self._closure_path = None
                return
            if (
                not stat.S_ISREG(facts.st_mode)
                or facts.st_uid != os.getuid()
                or facts.st_nlink != 1
                or facts.st_dev != self.identity["closure_device"]
                or facts.st_ino != self.identity["closure_inode"]
            ):
                raise RuntimeError("Native closure evidence identity changed")
            self._closure_path.unlink()
            self._closure_path = None

    async def start(self) -> None:
        if self._closing is not None or self.process is not None:
            raise RuntimeError("Native launch ownership has already been consumed")
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.scratch),
            "TMPDIR": str(self.scratch),
            "MPLCONFIGDIR": str(self.scratch / "matplotlib"),
            "MPLBACKEND": "Agg",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NUMEXPR_MAX_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        entry = Path(__file__).with_name("guardian.py").resolve()
        keep_read, self._keepalive = os.pipe()
        self._evidence, evidence_write = os.pipe()
        os.set_blocking(self._evidence, False)
        try:
            with (self.scratch / "startup.log").open("wb") as stderr:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(entry),
                        str(keep_read),
                        str(evidence_write),
                        str(self.scratch),
                        str(self._closure_descriptor),
                        self._closure_nonce,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    env=env,
                    cwd=self.scratch,
                    start_new_session=True,
                    close_fds=True,
                    pass_fds=(
                        keep_read,
                        evidence_write,
                        *(
                            (self._closure_descriptor,)
                            if self._closure_descriptor >= 0
                            else ()
                        ),
                    ),
                )
            if self._closure_descriptor >= 0:
                os.close(self._closure_descriptor)
                self._closure_descriptor = -1
            os.write(self._keepalive, b"L")
            spawned = await self._guardian_event(asyncio.get_running_loop().time() + 10)
            if (
                set(spawned) != {"kind", "pid"}
                or spawned["kind"] != "spawned"
                or type(spawned["pid"]) is not int
            ):
                raise RuntimeError("Native guardian did not authenticate a worker")
            self.worker_pid = spawned["pid"]
            assert self.process.stdout is not None
            assert self.process.stdin is not None
            os.set_blocking(self.process.stdout.fileno(), False)
            os.set_blocking(self.process.stdin.fileno(), False)
            await self._send(
                {
                    "scratch": str(self.scratch),
                    "profile": _profile(self.scratch),
                    "cpu_seconds": math.ceil(self.limits.cpu_seconds),
                    "scratch_bytes": self.limits.scratch_bytes,
                    "log_bytes": self.limits.log_bytes,
                    "outputs_per_cell": self.limits.outputs_per_cell,
                },
                deadline=asyncio.get_running_loop().time() + 10,
            )
            ready = await self.receive(asyncio.get_running_loop().time() + 10)
            if ready != {"kind": "ready", "protocol": 1}:
                raise RuntimeError("Analysis worker did not establish containment")
            await self.pause()
            self.sample()
            if (
                self.usage["user_cpu_seconds"] is None
                or self.usage["system_cpu_seconds"] is None
                or self.usage["peak_rss_bytes"] is None
                or self.usage.get("peak_threads") is None
            ):
                raise RuntimeError(
                    "Required native resource measurements are unavailable"
                )
        except BaseException:
            await self.close()
            raise
        finally:
            os.close(keep_read)
            os.close(evidence_write)

    async def _send(self, value: dict[str, object], *, deadline: float) -> None:
        encoded = json.dumps(value, allow_nan=False).encode() + b"\n"
        if len(encoded) > 262_144:
            raise ValueError("Analysis protocol frame exceeds its byte allowance")
        assert self.process is not None and self.process.stdin is not None
        offset = 0
        while offset < len(encoded):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Analysis IPC delivery allowance exhausted")
            try:
                written = os.write(self.process.stdin.fileno(), encoded[offset:])
                offset += written
                self.usage["protocol_input_bytes"] += written
            except BlockingIOError:
                await asyncio.sleep(0.01)

    def sample(self) -> None:
        function = None if self.worker_pid is None else _darwin_proc_pidinfo()
        if function is not None:
            info = _ProcTaskInfo()
            size = ctypes.sizeof(info)
            if function(self.worker_pid, 4, 0, ctypes.byref(info), size) == size:
                scale = _cpu_tick_seconds()
                if scale is not None:
                    self.usage["user_cpu_seconds"] = info.total_user * scale
                    self.usage["system_cpu_seconds"] = info.total_system * scale
                self.usage["peak_rss_bytes"] = max(
                    int(self.usage["peak_rss_bytes"] or 0), info.resident_size
                )
                self.usage["peak_threads"] = max(
                    int(self.usage.get("peak_threads", 0)), info.threadnum
                )
                self.usage["thread_overshoot"] = max(
                    0, info.threadnum - self.limits.max_threads
                )
                if info.threadnum > self.limits.max_threads:
                    raise RuntimeError("Analysis thread allowance exhausted")
            else:
                self.usage["user_cpu_seconds"] = None
                self.usage["system_cpu_seconds"] = None
                raise RuntimeError("Fresh native CPU measurement is unavailable")
        size_bytes, files = 0, 0

        def observe_scratch() -> None:
            self.usage["observed_peak_scratch_bytes"] = max(
                self.usage["observed_peak_scratch_bytes"], size_bytes
            )
            self.usage["observed_peak_scratch_files"] = max(
                self.usage["observed_peak_scratch_files"], files
            )

        def incomplete() -> None:
            observe_scratch()
            self.usage["scratch_samples_incomplete"] = True
            self.usage["scratch_overshoot_bytes"] = None
            self.usage["scratch_overshoot_bytes_lower_bound"] = max(
                self.usage.get("scratch_overshoot_bytes_lower_bound", 0),
                size_bytes - self.limits.scratch_bytes,
            )
            self.usage["scratch_file_overshoot_lower_bound"] = max(
                self.usage.get("scratch_file_overshoot_lower_bound", 0),
                files - self.limits.scratch_files,
            )
            for key in (
                "current_scratch_bytes",
                "current_scratch_files",
                "peak_scratch_bytes",
                "peak_scratch_files",
            ):
                self.usage[key] = None

        directories = [self.scratch]
        try:
            while directories:
                with os.scandir(directories.pop()) as entries:
                    for entry in entries:
                        try:
                            facts = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        files += 1
                        if stat.S_ISREG(facts.st_mode):
                            size_bytes += facts.st_size
                        elif stat.S_ISDIR(facts.st_mode):
                            directories.append(Path(entry.path))
                        if (
                            files > self.limits.scratch_files
                            or size_bytes > self.limits.scratch_bytes
                        ):
                            incomplete()
                            raise RuntimeError(
                                "Analysis scratch allowance exhausted; observed usage is a lower bound"
                            )
        except OSError:
            incomplete()
            raise RuntimeError("Scratch measurement is unavailable") from None
        observe_scratch()
        if not self.usage["scratch_samples_incomplete"]:
            self.usage["peak_scratch_bytes"] = self.usage["observed_peak_scratch_bytes"]
            self.usage["peak_scratch_files"] = self.usage["observed_peak_scratch_files"]
        self.usage["current_scratch_bytes"] = size_bytes
        self.usage["current_scratch_files"] = files
        self.usage["cpu_overshoot_seconds"] = max(
            0.0,
            float(self.usage["user_cpu_seconds"] or 0)
            + float(self.usage["system_cpu_seconds"] or 0)
            - self.limits.cpu_seconds,
        )
        self.usage["memory_overshoot_bytes"] = max(
            0, int(self.usage["peak_rss_bytes"] or 0) - self.limits.memory_bytes
        )
        self.usage["scratch_overshoot_bytes"] = max(
            0, size_bytes - self.limits.scratch_bytes
        )
        cpu = float(self.usage["user_cpu_seconds"] or 0) + float(
            self.usage["system_cpu_seconds"] or 0
        )
        if (
            cpu > self.limits.cpu_seconds
            or int(self.usage["peak_rss_bytes"] or 0) > self.limits.memory_bytes
            or size_bytes > self.limits.scratch_bytes
            or files > self.limits.scratch_files
        ):
            raise RuntimeError("Analysis compute or scratch allowance exhausted")

    async def receive(self, deadline: float) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        while True:
            self.sample()
            now = asyncio.get_running_loop().time()
            if (
                self.revalidate is not None
                and now - self._last_check >= self.limits.revocation_seconds
            ):
                await self.revalidate()
                self._last_check = now
            if self.observe is not None and now - self._last_observation >= 1:
                self.observe(self.usage)
                self._last_observation = now
            try:
                raw = os.read(self.process.stdout.fileno(), 65_536)
                if not raw:
                    raise RuntimeError(
                        "Analysis worker exited before definite completion"
                    )
                self.usage["protocol_output_bytes"] += len(raw)
                self._buffer.extend(raw)
            except BlockingIOError:
                pass
            if len(self._buffer) > self.limits.result_bytes:
                raise RuntimeError("Analysis worker protocol byte allowance exhausted")
            if b"\n" in self._buffer:
                line, _, remaining = self._buffer.partition(b"\n")
                self._buffer = bytearray(remaining)
                result = json.loads(line)
                if not isinstance(result, dict):
                    raise RuntimeError("Invalid analysis worker frame")
                return result
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Analysis cell exceeded its wall-time allowance")
            await asyncio.sleep(self.limits.sample_seconds)

    async def execute(
        self,
        code: str,
        *,
        deadline: float,
        broker: (
            Callable[[str, Mapping[str, object]], Awaitable[dict[str, object]]] | None
        ) = None,
        compute: Callable[[], Awaitable[PermitLease]] | None = None,
        inputs: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        started = asyncio.get_running_loop().time()
        permit = None
        try:
            permit = None if compute is None else await compute()
            token = secrets.token_hex(32)
            assert self.process is not None
            assert self.worker_pid is not None
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Analysis compute resume deadline expired")
            self.resume()
            await self._send(
                {
                    "kind": "cell",
                    "code": code,
                    "token": token,
                    "inputs": dict(inputs or {}),
                },
                deadline=deadline,
            )
            sequence = 0
            rpc_bytes = 0
            while True:
                try:
                    result = await self.receive(deadline)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if broker is not None:
                        await broker(
                            "<invalid-analysis-frame>", {"reason": "unparseable"}
                        )
                    raise RuntimeError("Invalid analysis protocol encoding") from None
                if result.get("kind") != "child":
                    break
                sequence += 1
                if (
                    set(result) != {"kind", "token", "sequence", "name", "arguments"}
                    or result["token"] != token
                    or type(result["sequence"]) is not int
                    or result["sequence"] != sequence
                    or not isinstance(result["name"], str)
                    or not result["name"]
                    or len(result["name"]) > 256
                    or not isinstance(result["arguments"], dict)
                    or broker is None
                ):
                    if broker is not None:
                        await broker(
                            "<invalid-analysis-frame>", {"reason": "stale_or_malformed"}
                        )
                    raise RuntimeError("Invalid or stale analysis broker frame")
                wait_started = asyncio.get_running_loop().time()
                await self.pause()
                if permit is not None:
                    await permit.release()
                    permit = None
                try:
                    async with asyncio.timeout_at(deadline):
                        child_result = await broker(result["name"], result["arguments"])
                finally:
                    self.usage["broker_wait_seconds"] = (
                        float(self.usage["broker_wait_seconds"])
                        + asyncio.get_running_loop().time()
                        - wait_started
                    )
                from ..._json import canonical_json

                rpc_bytes += len(canonical_json(child_result).encode()) + len(
                    json.dumps(result).encode()
                )
                self.usage["input_bytes"] += len(canonical_json(child_result).encode())
                if (
                    rpc_bytes > self.limits.rpc_bytes
                    or sequence > self.limits.child_calls
                ):
                    raise RuntimeError("Analysis cell RPC allowance exhausted")
                permit = None if compute is None else await compute()
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("Analysis compute resume deadline expired")
                self.resume()
                await self._send(
                    {"result": json.loads(canonical_json(child_result))},
                    deadline=deadline,
                )
            if (
                set(result)
                != {
                    "kind",
                    "status",
                    "stdout",
                    "stderr",
                    "traceback",
                    "outputs",
                    "token",
                    "sequence",
                }
                or result.get("kind") != "cell"
                or result.get("status") not in {"success", "python_error"}
                or result.get("token") != token
                or type(result.get("sequence")) is not int
                or result["sequence"] != sequence
                or not isinstance(result.get("stdout"), str)
                or not isinstance(result.get("stderr"), str)
                or (
                    result.get("traceback") is not None
                    and not isinstance(result["traceback"], str)
                )
            ):
                raise RuntimeError("Invalid analysis cell completion")
            if any(
                len(result[key].encode()) > self.limits.log_bytes
                for key in ("stdout", "stderr")
            ):
                raise RuntimeError("Analysis log byte allowance exhausted")
            await self.pause()
            self.sample()
            result.pop("token")
            result.pop("sequence")
            self.usage["output_bytes"] = int(self.usage["output_bytes"]) + len(
                json.dumps(result).encode()
            )
            return result
        finally:
            if permit is not None:
                await permit.release()
            self.usage["wall_seconds"] = (
                float(self.usage["wall_seconds"])
                + asyncio.get_running_loop().time()
                - started
            )

    async def _guardian_event(self, deadline: float) -> dict[str, Any]:
        assert self._evidence is not None
        while True:
            try:
                raw = os.read(self._evidence, 8192)
                if not raw:
                    raise RuntimeError("Native guardian closure evidence unavailable")
                self._evidence_buffer.extend(raw)
            except BlockingIOError:
                pass
            if len(self._evidence_buffer) > 8192:
                raise RuntimeError("Invalid native guardian evidence")
            if b"\n" in self._evidence_buffer:
                line, _, remaining = self._evidence_buffer.partition(b"\n")
                self._evidence_buffer = bytearray(remaining)
                value = json.loads(line)
                if value.get("kind") == "closed":
                    self._closed_evidence = value
                return value
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("Native guardian cleanup deadline exceeded")
            await asyncio.sleep(0.02)

    async def pause(self) -> None:
        assert self._keepalive is not None
        os.write(self._keepalive, b"S")
        stopped = await self._guardian_event(
            asyncio.get_running_loop().time() + self.limits.cleanup_seconds
        )
        if stopped != {"kind": "stopped", "pid": self.worker_pid}:
            raise RuntimeError(
                "Worker suspension was not observed by its native parent"
            )

    def resume(self) -> None:
        # Recheck the newly admitted lifetime cap immediately before SIGCONT.
        self.sample()
        assert self._keepalive is not None
        os.write(self._keepalive, b"C")

    async def close(self) -> dict[str, object]:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        try:
            return await asyncio.shield(self._closing)
        except asyncio.CancelledError:
            await self._closing
            raise

    async def _close(self) -> dict[str, object]:
        if self.cleanup is not None:
            return self.cleanup
        started = asyncio.get_running_loop().time()
        exit_code = None
        reaped = self.process is None
        spawned = None if self.process is not None else False
        failure = None
        self.revalidate = None
        if self._keepalive is not None:
            os.close(self._keepalive)
            self._keepalive = None
        if self.process is None:
            self.usage.update(
                user_cpu_seconds=0.0,
                system_cpu_seconds=0.0,
                cpu_complete=True,
                measurement_method="host verified worker was not spawned",
                peak_rss_bytes=None,
            )
            if self._closure_descriptor >= 0:
                unspawned = {
                    "kind": "closed",
                    "closure_nonce": self._closure_nonce,
                    "process_spawned": False,
                    "pid": None,
                    "exit_code": 0,
                    "reason": "host_owned_nonspawn",
                    "user_cpu_seconds": 0.0,
                    "system_cpu_seconds": 0.0,
                    "peak_rss_bytes": None,
                    "cleanup_seconds": 0.0,
                }
                os.pwrite(self._closure_descriptor, json.dumps(unspawned).encode(), 0)
                os.ftruncate(
                    self._closure_descriptor, len(json.dumps(unspawned).encode())
                )
                os.fsync(self._closure_descriptor)
        try:
            self.sample()
        except RuntimeError:
            pass
        if self.process is not None:
            try:
                closed = self._closed_evidence or await self._guardian_event(
                    started + self.limits.cleanup_seconds
                )
                while closed == {"kind": "stopped", "pid": self.worker_pid}:
                    closed = await self._guardian_event(
                        started + self.limits.cleanup_seconds
                    )
                if (
                    closed.get("kind") != "closed"
                    or (
                        self.worker_pid is not None
                        and closed.get("pid") != self.worker_pid
                    )
                    or closed.get("closure_nonce") != self._closure_nonce
                    or type(closed.get("process_spawned")) is not bool
                    or type(closed.get("exit_code")) is not int
                ):
                    raise RuntimeError(
                        "Native guardian returned invalid closure evidence"
                    )
                for name in ("user_cpu_seconds", "system_cpu_seconds"):
                    value = closed.get(name)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise RuntimeError(
                            "Final native CPU measurement is unavailable"
                        )
                self.worker_pid = closed["pid"]
                spawned = closed["process_spawned"]
                exit_code = closed["exit_code"]
                await asyncio.to_thread(self.process.wait, self.limits.cleanup_seconds)
                reaped = True
                self.usage["user_cpu_seconds"] = closed["user_cpu_seconds"]
                self.usage["system_cpu_seconds"] = closed["system_cpu_seconds"]
                self.usage["peak_rss_bytes"] = (
                    max(
                        int(self.usage["peak_rss_bytes"] or 0), closed["peak_rss_bytes"]
                    )
                    if closed["process_spawned"]
                    else None
                )
                self.usage["measurement_method"] = (
                    "macOS proc_pidinfo samples; wait4 final CPU/RSS"
                )
                self.usage["cpu_complete"] = True
            except Exception as error:
                failure = type(error).__name__
                self.usage["cpu_complete"] = False
                if self._durable_closure:
                    try:
                        # A guardian can exit before emitting a frame. Its PID
                        # alone proves no worker fact; the exact guard and phase
                        # must independently establish non-spawn or closure.
                        await asyncio.to_thread(
                            self.process.wait, self.limits.cleanup_seconds
                        )
                        from .recovery import recover_generation

                        recovered = await recover_generation(
                            {
                                **self.identity,
                                "pid": self.worker_pid,
                                "usage": dict(self.usage),
                            }
                        )
                        recovered_usage = cast(Mapping[str, Any], recovered["usage"])
                        recovered_cleanup = cast(
                            Mapping[str, Any], recovered["cleanup"]
                        )
                        self.usage.update(recovered_usage)
                        self.worker_pid = cast(int | None, recovered["pid"])
                        spawned = recovered_cleanup["process_spawned"]
                        exit_code = recovered_cleanup["exit_code"]
                        reaped = recovered_cleanup["process_reaped"] is True
                    except Exception:
                        pass  # Retain the original failure and unknown usage.
            for channel in (self.process.stdin, self.process.stdout):
                if channel is not None:
                    channel.close()
        if self._evidence is not None:
            os.close(self._evidence)
            self._evidence = None
        if self.usage.get("cpu_complete") is True:
            self.usage["cpu_overshoot_seconds"] = max(
                0.0,
                self.usage["user_cpu_seconds"]
                + self.usage["system_cpu_seconds"]
                - self.limits.cpu_seconds,
            )
        if reaped and self.scratch.exists():
            try:
                facts = self.scratch.lstat()
                if (
                    not stat.S_ISDIR(facts.st_mode)
                    or facts.st_uid != os.getuid()
                    or facts.st_dev != self.identity["scratch_device"]
                    or facts.st_ino != self.identity["scratch_inode"]
                ):
                    raise RuntimeError("Scratch identity changed")
                remove_owned_scratch(
                    self.scratch,
                    cast(int, self.identity["scratch_device"]),
                    cast(int, self.identity["scratch_inode"]),
                )
            except Exception as error:
                failure = type(error).__name__
        if self._closure_descriptor >= 0:
            os.close(self._closure_descriptor)
            self._closure_descriptor = -1
        self.usage["lifetime_wall_seconds"] = (
            asyncio.get_running_loop().time() - self._created_at
        )
        deleted = not self.scratch.exists() and not self.scratch.is_symlink()
        self.cleanup = {
            "close_requested": True,
            "broker_admission_stopped": True,
            "child_io_settled": True,
            "process_reaped": reaped,
            "process_spawned": spawned,
            "descriptors_closed": self._keepalive is None
            and self._evidence is None
            and self._closure_descriptor < 0
            and (
                self.process is None
                or all(
                    channel is None or channel.closed
                    for channel in (self.process.stdin, self.process.stdout)
                )
            ),
            "failure": failure,
            "exit_code": exit_code,
            "scratch_deleted": deleted,
            "exit_signal": (
                -exit_code if isinstance(exit_code, int) and exit_code < 0 else None
            ),
            "remaining_bytes": 0 if deleted else None,
            "remaining_files": 0 if deleted else None,
            "cleanup_seconds": asyncio.get_running_loop().time() - started,
        }
        if not self._durable_closure and reaped and deleted:
            self.dispose_closure_evidence()
        return self.cleanup
