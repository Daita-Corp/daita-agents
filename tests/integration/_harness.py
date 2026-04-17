"""
Shared harness for live integration tests.

Three utilities:

* ``build_live_agent`` — constructs an ``Agent`` using ``OPENAI_API_KEY``,
  skipping the test when the key is missing.
* ``timed`` — async context manager that logs wall-clock duration to stderr.
  Per-spec decision: integration tests use *soft* timing — we print, we
  don't assert. Review logs after the run.
* ``DockerContainer`` — thin wrapper around the docker CLI for standing up
  throwaway Postgres / MySQL / MongoDB instances. Idempotent teardown via
  ``docker rm -f``.

Also: accuracy helpers. ``assert_tool_called`` and
``assert_answer_mentions`` are the two coarse but reliable checks we apply
to every live-LLM test — did the agent pick the right tool, and does the
final answer contain the known-correct tokens?
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_live_agent(
    *,
    name: str,
    tools: Sequence[Any],
    model: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
):
    """Construct a real OpenAI-backed ``Agent`` for live integration tests.

    Skips the test when ``OPENAI_API_KEY`` is unset. Uses ``gpt-4o-mini`` by
    default — accurate enough for tool-picking assertions, ~10× cheaper than
    ``gpt-4o`` on a full integration run.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping live-LLM test")

    from daita.agents.agent import Agent

    kwargs = dict(
        name=name,
        llm_provider="openai",
        model=model,
        api_key=api_key,
        tools=list(tools),
    )
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return Agent(**kwargs)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class TimingRecord:
    """One ``timed`` observation. Published to stderr and captured in tests
    that want to inspect the number (e.g. relative comparisons)."""

    label: str
    elapsed_ms: float


@contextlib.asynccontextmanager
async def timed(label: str, record: Optional[List[TimingRecord]] = None):
    """Async context manager that logs wall-clock elapsed to stderr.

    Per-spec: soft timing only. No thresholds, no assertions. Review the
    captured output after the run. Pass ``record`` when a test wants to
    compare timings across calls.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        # stderr so pytest captures it under `-s` / on failure
        print(f"[TIMING] {label}: {elapsed_ms:.1f} ms", file=sys.stderr)
        if record is not None:
            record.append(TimingRecord(label=label, elapsed_ms=elapsed_ms))


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------


def assert_tool_called(
    result: Dict[str, Any],
    tool_name: str,
    *,
    message: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Verify the agent called ``tool_name`` at least once.

    Returns every matching call so tests can inspect arguments.
    """
    calls = result.get("tool_calls") or []
    hits = [c for c in calls if c.get("tool") == tool_name]
    if not hits:
        tool_names = [c.get("tool") for c in calls]
        msg = message or (
            f"Agent did not call '{tool_name}'. Tools actually called: {tool_names}"
        )
        raise AssertionError(msg)
    return hits


def assert_answer_mentions(
    result: Dict[str, Any],
    tokens: Iterable[str],
    *,
    any_of: bool = False,
) -> None:
    """Verify the agent's natural-language answer mentions each token.

    ``any_of=False`` (default) requires every token to appear. ``any_of=True``
    passes when at least one appears. Comparison is case-insensitive and
    substring — LLMs paraphrase, so exact match is usually too strict.
    """
    text = (result.get("result") or "").lower()
    tokens = [t for t in tokens if t]
    if not tokens:
        return
    matches = [t for t in tokens if t.lower() in text]
    if any_of:
        if not matches:
            raise AssertionError(
                f"Answer mentioned none of {list(tokens)}. Answer: {text[:400]!r}"
            )
    else:
        missing = [t for t in tokens if t.lower() not in text]
        if missing:
            raise AssertionError(
                f"Answer missing tokens {missing}. Answer: {text[:400]!r}"
            )


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    """Return True iff ``docker`` is on PATH and the daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def require_docker() -> None:
    """Skip the test if docker is unavailable."""
    if not _docker_available():
        pytest.skip("docker not available — skipping containerized integration test")


def _free_port() -> int:
    """Return a currently-free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_tcp(host: str, port: int, timeout: float = 60.0) -> None:
    """Block until ``host:port`` accepts a TCP connection or ``timeout`` elapses."""
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(0.5)
    raise TimeoutError(
        f"{host}:{port} did not open within {timeout}s (last error: {last_err})"
    )


@dataclass
class DockerContainer:
    """Handle for a running container. ``remove()`` is always safe to call."""

    name: str
    image: str
    host_port: int
    container_port: int

    @property
    def host(self) -> str:
        return "127.0.0.1"

    def remove(self) -> None:
        """Force-remove the container. Idempotent; swallows errors."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.name],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:
            logger.warning("docker rm -f %s failed: %s", self.name, exc)


def start_container(
    image: str,
    *,
    container_port: int,
    env: Optional[Dict[str, str]] = None,
    cmd: Optional[List[str]] = None,
    tag_prefix: str = "daita-it",
    readiness_timeout: float = 90.0,
) -> DockerContainer:
    """Pull + run a throwaway container; return once its port is listening.

    The caller is responsible for calling ``container.remove()`` on teardown —
    use ``with contextlib.closing(...)`` or a ``finally``. We intentionally
    don't rely on ``--rm`` because it hides crash logs.
    """
    require_docker()

    host_port = _free_port()
    name = f"{tag_prefix}-{uuid.uuid4().hex[:8]}"
    args = ["docker", "run", "-d", "--name", name, "-p", f"{host_port}:{container_port}"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    args.append(image)
    if cmd:
        args += cmd

    try:
        proc = subprocess.run(args, capture_output=True, timeout=180, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"docker run failed ({' '.join(args)}): "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc

    container = DockerContainer(
        name=name,
        image=image,
        host_port=host_port,
        container_port=container_port,
    )
    try:
        _wait_for_tcp(container.host, host_port, timeout=readiness_timeout)
    except TimeoutError:
        # Dump container logs so the test failure message is actionable.
        logs = subprocess.run(
            ["docker", "logs", name], capture_output=True, timeout=10
        ).stdout.decode("utf-8", errors="replace")
        container.remove()
        raise RuntimeError(
            f"Container {image} didn't open {container_port} in "
            f"{readiness_timeout}s. Logs:\n{logs}"
        )
    return container
