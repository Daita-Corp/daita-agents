"""Trusted parent of the contained interpreter; reap and delete on host death."""

from __future__ import annotations

import json
import os
import runpy
import select
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    keepalive, evidence = int(sys.argv[1]), int(sys.argv[2])
    scratch = Path(sys.argv[3])
    original = scratch.lstat()
    closure, nonce = int(sys.argv[4]), sys.argv[5]
    entry = Path(__file__).with_name("worker.py")

    def persist(value: dict[str, object]) -> None:
        if closure >= 0:
            raw = json.dumps({**value, "closure_nonce": nonce}).encode()
            os.pwrite(closure, raw, 0)
            os.ftruncate(closure, len(raw))
            os.fsync(closure)

    def emit(value: dict[str, object]) -> None:
        try:
            os.write(evidence, json.dumps(value).encode() + b"\n")
        except BrokenPipeError:
            pass

    if closure < 0:
        raise RuntimeError("Native launch requires inherited ownership proof")
    identity = os.fstat(closure)
    allocated = json.loads(os.pread(closure, 8192, 0))
    if (
        not stat.S_ISREG(identity.st_mode)
        or identity.st_uid != os.getuid()
        or type(allocated.get("durable_closure")) is not bool
        or identity.st_nlink != (1 if allocated.get("durable_closure") else 0)
        or allocated.get("kind") != "allocated"
        or allocated.get("closure_nonce") != nonce
        or allocated.get("scratch_device") != original.st_dev
        or allocated.get("scratch_inode") != original.st_ino
    ):
        raise RuntimeError("Native launch ownership authentication failed")

    def finish(facts: dict[str, object]) -> None:
        started = time.monotonic()
        deletion_error = None
        try:
            remove = runpy.run_path(str(Path(__file__).with_name("cleanup.py")))[
                "remove_owned_scratch"
            ]
            remove(scratch, original.st_dev, original.st_ino)
        except FileNotFoundError:
            pass
        except OSError:
            deletion_error = "scratch_deletion_failed"
        facts.update(
            scratch_deleted=not scratch.exists() and not scratch.is_symlink(),
            deletion_error=deletion_error,
            cleanup_seconds=time.monotonic() - started,
            closure_nonce=nonce,
        )
        try:
            persist(facts)
        finally:
            emit(facts)
            os.close(closure)

    # No worker can be created before the explicit host handoff. EOF closes an
    # abandoned allocation. A death racing the subsequent spawn is handled by
    # the ordinary reap loop, while the inherited guard excludes recovery.
    launch = os.read(keepalive, 1)
    if launch == b"L":
        os.set_blocking(keepalive, False)
        try:
            if os.read(keepalive, 1) == b"":
                launch = b""  # Handoff queued, but the host is already gone.
        except BlockingIOError:
            pass
        finally:
            os.set_blocking(keepalive, True)
    if launch != b"L":
        finish(
            {
                "kind": "closed",
                "process_spawned": False,
                "pid": None,
                "exit_code": 0,
                "reason": "abandoned_handoff",
                "user_cpu_seconds": 0.0,
                "system_cpu_seconds": 0.0,
                "peak_rss_bytes": None,
                "cleanup_seconds": 0.0,
            }
        )
        return
    try:
        persist({"kind": "launch_committed"})
        worker = subprocess.Popen(
            [sys.executable, "-I", "-B", str(entry)],
            stdin=sys.stdin.buffer,
            stdout=sys.stdout.buffer,
            stderr=sys.stderr.buffer,
            close_fds=True,
            cwd=scratch,
        )
    except Exception:
        facts = {
            "kind": "closed",
            "process_spawned": False,
            "pid": None,
            "exit_code": 0,
            "reason": "native_spawn_failed",
            "user_cpu_seconds": 0.0,
            "system_cpu_seconds": 0.0,
            "peak_rss_bytes": None,
            "cleanup_seconds": 0.0,
            "closure_nonce": nonce,
        }
        finish(facts)
        return
    reason = "process_exit"
    try:
        persist({"kind": "worker_spawned", "pid": worker.pid})
        # Only the worker owns the data channel. The proof descriptor is excluded
        # from its close_fds launch and remains owned by this trusted parent.
        sys.stdin.close()
        sys.stdout.close()
        emit({"kind": "spawned", "pid": worker.pid})
        while True:
            pid, status, usage = os.wait4(worker.pid, os.WNOHANG)
            if pid:
                break
            readable, _, _ = select.select([keepalive], [], [], 0.02)
            if readable:
                command = os.read(keepalive, 1)
                if not command:
                    reason = "host_channel_closed"
                    try:
                        os.kill(worker.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    pid, status, usage = os.wait4(worker.pid, 0)
                    break
                if command == b"S":
                    try:
                        os.kill(worker.pid, signal.SIGSTOP)
                    except ProcessLookupError:
                        pid, status, usage = os.wait4(worker.pid, 0)
                        break
                    pid, status, usage = os.wait4(worker.pid, os.WUNTRACED)
                    if not os.WIFSTOPPED(status):
                        break
                    emit({"kind": "stopped", "pid": worker.pid})
                elif command == b"C":
                    try:
                        os.kill(worker.pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                else:
                    reason = "invalid_host_channel"
                    os.kill(worker.pid, signal.SIGKILL)
                    pid, status, usage = os.wait4(worker.pid, 0)
                    break
    except Exception:
        # Launch/phase/channel failure must never abandon a real child.
        reason = "guardian_launch_or_channel_failed"
        try:
            os.kill(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pid, status, usage = os.wait4(worker.pid, 0)
    worker.returncode = os.waitstatus_to_exitcode(status)
    facts = {
        "kind": "closed",
        "process_spawned": True,
        "pid": pid,
        "exit_code": worker.returncode,
        "reason": reason,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "peak_rss_bytes": usage.ru_maxrss,
        "closure_nonce": nonce,
    }
    finish(facts)


if __name__ == "__main__":
    main()
