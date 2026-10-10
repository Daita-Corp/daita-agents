"""Join authenticated native-parent closure files to durable run evidence."""

from __future__ import annotations

import asyncio
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from .cleanup import remove_owned_scratch


async def recover_generation(facts: Mapping[str, object]) -> dict[str, object]:
    import fcntl

    path = Path(str(facts["closure_path"]))
    scratch = Path(str(facts["scratch"]))
    if (
        not path.is_absolute()
        or not scratch.is_absolute()
        or not path.name.startswith("daita-analysis-closure-")
        or not scratch.name.startswith("daita-analysis-")
    ):
        raise ValueError("Invalid authenticated native remnant identity")
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_uid != os.getuid()
            or identity.st_nlink != 1
            or identity.st_dev != facts["closure_device"]
            or identity.st_ino != facts["closure_inode"]
            or identity.st_size > 8192
        ):
            raise ValueError("Native closure remnant identity changed")
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        "Native ownership proof unavailable; analysis admission is blocked"
                    ) from None
                await asyncio.sleep(0.02)
        raw = os.pread(descriptor, 8193, 0)
        if not raw or len(raw) > 8192:
            raise RuntimeError(
                "Native launch or closure remains unobserved; analysis admission is blocked"
            )
        closure = json.loads(raw)
        if (
            not isinstance(closure, dict)
            or closure.get("closure_nonce") != facts["closure_nonce"]
        ):
            raise ValueError("Native launch evidence failed authentication")
        if closure.get("kind") == "allocated":
            if (
                closure.get("scratch_device") != facts["scratch_device"]
                or closure.get("scratch_inode") != facts["scratch_inode"]
                or closure.get("durable_closure") is not True
            ):
                raise ValueError("Native allocation identity changed")
            # Exclusive ownership excludes a living host, forked pre-exec
            # guardian, or delayed launch. The guardian must durably commit
            # launch before creating a worker, so allocated proves non-spawn.
            closure = {
                "kind": "closed",
                "closure_nonce": facts["closure_nonce"],
                "process_spawned": False,
                "pid": None,
                "exit_code": 0,
                "reason": "ownership_guard_verified_nonspawn",
                "user_cpu_seconds": 0.0,
                "system_cpu_seconds": 0.0,
                "peak_rss_bytes": None,
                "cleanup_seconds": 0.0,
            }
            encoded = json.dumps(closure).encode()
            os.pwrite(descriptor, encoded, 0)
            os.ftruncate(descriptor, len(encoded))
            os.fsync(descriptor)
        if (
            closure.get("kind") != "closed"
            or closure.get("closure_nonce") != facts["closure_nonce"]
            or (facts.get("pid") is not None and closure.get("pid") != facts["pid"])
            or type(closure.get("exit_code")) is not int
            or type(closure.get("process_spawned")) is not bool
            or (
                closure["process_spawned"]
                and (type(closure.get("pid")) is not int or closure["pid"] <= 0)
            )
            or (not closure["process_spawned"] and closure.get("pid") is not None)
        ):
            raise ValueError("Native parent closure evidence failed authentication")
        for name in ("user_cpu_seconds", "system_cpu_seconds", "cleanup_seconds"):
            value = closure.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Native closure measurement is unavailable")
        if not closure["process_spawned"] and (
            closure["user_cpu_seconds"] != 0 or closure["system_cpu_seconds"] != 0
        ):
            raise ValueError("Non-spawn closure has inconsistent consumption")
        if scratch.exists() or scratch.is_symlink():
            original = scratch.lstat()
            if (
                not stat.S_ISDIR(original.st_mode)
                or scratch.is_symlink()
                or original.st_uid != os.getuid()
                or original.st_dev != facts["scratch_device"]
                or original.st_ino != facts["scratch_inode"]
            ):
                raise ValueError("Native scratch remnant identity changed")
            remove_owned_scratch(
                scratch, int(facts["scratch_device"]), int(facts["scratch_inode"])
            )
        result = dict(facts)
        prior_usage = facts.get("usage")
        usage = dict(prior_usage) if isinstance(prior_usage, Mapping) else {}
        usage.update(
            {
                name: closure[name]
                for name in ("user_cpu_seconds", "system_cpu_seconds", "peak_rss_bytes")
            }
        )
        usage.update(
            {
                "measurement_method": (
                    "native guardian wait4 after host crash"
                    if closure["process_spawned"]
                    else "exclusive native ownership proof: never spawned"
                ),
                "cpu_complete": True,
                "wall_seconds": usage.get("wall_seconds"),
                "broker_wait_seconds": usage.get("broker_wait_seconds"),
                "unavailable": [
                    "final_cell_wall_seconds",
                    "final_broker_wait_seconds",
                    "final_scratch_peak",
                ],
            }
        )
        result.update(
            {
                "status": "recovered_closed",
                "pid": closure["pid"],
                "usage": usage,
                "cleanup": {
                    "close_requested": True,
                    "broker_admission_stopped": True,
                    "process_reaped": True,
                    "process_spawned": closure["process_spawned"],
                    "descriptors_closed": True,
                    "child_io_settled": True,
                    "exit_code": closure["exit_code"],
                    "scratch_deleted": not scratch.exists(),
                    "remaining_bytes": 0 if not scratch.exists() else None,
                    "remaining_files": 0 if not scratch.exists() else None,
                    "cleanup_seconds": closure["cleanup_seconds"],
                    "reason": closure["reason"],
                },
            }
        )
        return result
    finally:
        os.close(descriptor)


def dispose_generation_proof(facts: Mapping[str, object]) -> None:
    """Idempotent exact proof disposal, only after durable closure persistence."""
    import fcntl

    path = Path(str(facts["closure_path"]))
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return
    try:
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_uid != os.getuid()
            or identity.st_nlink != 1
            or identity.st_dev != facts["closure_device"]
            or identity.st_ino != facts["closure_inode"]
        ):
            raise ValueError("Native proof disposal identity changed")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise ValueError("Native proof disposal path changed")
        path.unlink()
    finally:
        os.close(descriptor)
