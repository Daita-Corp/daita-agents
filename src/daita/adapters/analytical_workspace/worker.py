"""Private worker entry point. Establish OS containment before receiving cells."""

from __future__ import annotations

import contextlib
import ctypes
import io
import json
import os
import resource
import sys
import threading
import traceback
from types import SimpleNamespace


def _sandbox(profile: str) -> None:
    library = ctypes.CDLL("/usr/lib/libsandbox.dylib", use_errno=True)
    apply = library.sandbox_init
    apply.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
    apply.restype = ctypes.c_int
    error = ctypes.c_char_p()
    if apply(profile.encode(), 0, ctypes.byref(error)) != 0:
        raise RuntimeError("Native analysis containment could not be established")


class _BoundedText(io.StringIO):
    def __init__(self, max_bytes: int) -> None:
        super().__init__()
        self.byte_count = 0
        self.max_bytes = max_bytes

    def write(self, value: str) -> int:
        size = len(value.encode("utf-8"))
        if self.byte_count + size > self.max_bytes:
            raise RuntimeError("Cell log byte allowance exhausted")
        self.byte_count += size
        return super().write(value)


def main() -> None:
    channel = sys.stdout
    request_stream = sys.stdin
    bootstrap = json.loads(request_stream.readline(262_145))
    resource.setrlimit(
        resource.RLIMIT_CPU, (bootstrap["cpu_seconds"], bootstrap["cpu_seconds"] + 1)
    )
    resource.setrlimit(resource.RLIMIT_FSIZE, (bootstrap["scratch_bytes"],) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    os.chdir(bootstrap["scratch"])
    _sandbox(bootstrap["profile"])
    namespace: dict[str, object] = {"__name__": "__analysis__"}
    admitted_inputs: dict[str, dict[str, object]] = {}
    channel.write(json.dumps({"kind": "ready", "protocol": 1}) + "\n")
    channel.flush()
    while raw := request_stream.readline(262_145):
        if len(raw.encode()) > 262_144:
            return
        request = json.loads(raw)
        if request.get("kind") == "close":
            return
        stdout, stderr = _BoundedText(bootstrap["log_bytes"]), _BoundedText(
            bootstrap["log_bytes"]
        )
        sequence = 0
        token = request["token"]

        broker_lock = threading.Lock()

        def call(name: str, arguments: dict[str, object]) -> object:
            with broker_lock:
                return invoke(name, arguments)

        def invoke(name: str, arguments: dict[str, object]) -> object:
            nonlocal sequence
            sequence += 1
            frame = json.dumps(
                {
                    "kind": "child",
                    "token": token,
                    "sequence": sequence,
                    "name": name,
                    "arguments": arguments,
                },
                allow_nan=False,
            )
            if len(frame.encode()) > 65_536:
                raise ValueError("Broker request byte allowance exhausted")
            channel.write(frame + "\n")
            channel.flush()
            return json.loads(request_stream.readline(262_145))["result"]

        namespace["tools"] = SimpleNamespace(call=call)
        outputs: dict[str, str] = {}

        def add_output(name: str, path: str) -> None:
            if (
                len(outputs) >= bootstrap["outputs_per_cell"]
                or not isinstance(name, str)
                or not isinstance(path, str)
            ):
                raise ValueError("Output candidate allowance exhausted")
            outputs[name] = path

        namespace["outputs"] = SimpleNamespace(add=add_output)
        admitted_inputs.update(request.get("inputs", {}))

        def input_path(name: str) -> str:
            return str(admitted_inputs[name]["path"])

        def input_value(name: str) -> object:
            binding = admitted_inputs[name]
            if binding["media_type"] == "application/json":
                with open(input_path(name), encoding="utf-8") as source:
                    return json.load(source)
            if (
                str(binding["media_type"]).startswith("text/")
                or binding["media_type"] == "application/sql"
            ):
                with open(input_path(name), encoding="utf-8") as source:
                    return source.read()
            raise ValueError("Use inputs.path(name) with an installed format reader")

        namespace["inputs"] = SimpleNamespace(get=input_value, path=input_path)
        status, trace = "success", None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(request["code"], "<analysis-cell>", "exec"), namespace)
        except BaseException:
            status = "python_error"
            trace = traceback.format_exc(limit=8)[-8192:]
        channel.write(
            json.dumps(
                {
                    "kind": "cell",
                    "status": status,
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "traceback": trace,
                    "outputs": [
                        {"name": name, "path": path} for name, path in outputs.items()
                    ],
                    "token": token,
                    "sequence": sequence,
                }
            )
            + "\n"
        )
        channel.flush()


if __name__ == "__main__":
    main()
