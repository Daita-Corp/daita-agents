"""Private worker entry point. Establish OS containment before receiving cells."""

from __future__ import annotations

import contextlib
import ctypes
import io
import json
import os
import re
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

        def batch(calls: list[dict[str, object]]) -> object:
            with broker_lock:
                response = invoke("<analysis-batch>", {"calls": calls})
                if not isinstance(response, dict) or not isinstance(
                    response.get("results"), list
                ):
                    raise ValueError("Invalid batch response")
                return response["results"]

        def capture(
            tool_name: str, arguments: dict[str, object], *, name: str
        ) -> object:
            logical_name = name
            with broker_lock:
                response = invoke(
                    "<analysis-capture>",
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "input_name": logical_name,
                    },
                )
                if not isinstance(response, dict):
                    raise ValueError("Invalid capture response")
                manifest = response.pop("input_manifest", {})
                if not isinstance(manifest, dict) or set(manifest) - {logical_name}:
                    raise ValueError("Invalid capture input manifest")
                for logical, binding in manifest.items():
                    if (
                        not isinstance(binding, dict)
                        or re.fullmatch(
                            r"input-[0-9a-f]{32}\.arrow", str(binding.get("path"))
                        )
                        is None
                    ):
                        raise ValueError("Invalid capture input binding")
                    admitted_inputs[logical] = binding
                return response

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
            raw_response = request_stream.readline(262_145)
            if len(raw_response.encode()) > 262_144:
                raise ValueError("Broker response byte allowance exhausted")
            response = json.loads(raw_response)
            if (
                set(response) != {"kind", "token", "sequence", "result"}
                or response["kind"] != "child_result"
                or response["token"] != token
                or type(response["sequence"]) is not int
                or response["sequence"] != sequence
            ):
                raise ValueError("Invalid or stale broker response identity")
            return response["result"]

        namespace["tools"] = SimpleNamespace(call=call, batch=batch, capture=capture)
        outputs: dict[str, str] = {}

        def add_output(name: str, path: str) -> None:
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) is None
            ):
                raise ValueError(
                    "Output names must start with a letter and contain only letters, digits, '_' or '-'"
                )
            if (
                not isinstance(path, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", path) is None
            ):
                raise ValueError(
                    "Output paths must be relative scratch filenames without directories"
                )
            if len(outputs) >= bootstrap["outputs_per_cell"]:
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
