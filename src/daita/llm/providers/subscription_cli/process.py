"""Own bounded official-client subprocess execution and retirement."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import stat
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Protocol

from ..._lifecycle import (
    AttemptLifecycle,
    CloseCoordinator,
    NativeOwner,
    await_cleanup,
    execute_generate_attempt,
    join_until,
    shutdown_deadline,
)
from ...errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from ...models import ModelRequest, ModelResponse

_MAX_REQUEST_BYTES = 16 * 1_024 * 1_024


_MAX_STDOUT_BYTES = 4 * 1_024 * 1_024


_MAX_STDERR_BYTES = 256 * 1_024


_MAX_COMMAND_ARGUMENT_BYTES = 1 * 1_024 * 1_024


_PROCESS_STOP_GRACE_SECONDS = 1.0


_SAFE_SUBSCRIPTION_ENVIRONMENT = frozenset(
    {
        "ALL_PROXY",
        "APPDATA",
        "CLAUDE_CODE_GIT_BASH_PATH",
        "CLAUDE_CONFIG_DIR",
        "COMSPEC",
        "DBUS_SESSION_BUS_ADDRESS",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOCALAPPDATA",
        "LOGNAME",
        "NODE_EXTRA_CA_CERTS",
        "NO_PROXY",
        "PATH",
        "PATHEXT",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USER",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


@dataclass(frozen=True, slots=True)
class _Command:
    arguments: tuple[str, ...]
    stdin: bytes
    cwd: Path
    environment: Mapping[str, str]
    deadline: float
    cleanup_timeout_seconds: float = 5.0
    native_owner: NativeOwner = dataclass_field(default_factory=NativeOwner)
    cleanup_deadline: Callable[[], float] | None = None

    def shutdown_limit(self) -> float:
        return (
            self.cleanup_deadline()
            if self.cleanup_deadline is not None
            else shutdown_deadline(seconds=self.cleanup_timeout_seconds)
        )


@dataclass(frozen=True, slots=True)
class _CompletedCommand:
    returncode: int
    stdout: bytes
    stderr: bytes


class _CommandRunner(Protocol):
    def __call__(self, command: _Command) -> Awaitable[_CompletedCommand]: ...


class _ExecutableUnavailable(Exception):
    pass


class _CommandOutputLimit(Exception):
    pass


async def _read_bounded(
    stream: asyncio.StreamReader,
    limit: int,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await stream.read(min(64 * 1_024, limit - size + 1))
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise _CommandOutputLimit
        chunks.append(chunk)


async def _stop_process(
    process: asyncio.subprocess.Process, *, deadline: float, owner: NativeOwner
) -> None:
    loop = asyncio.get_running_loop()
    # Leave part of the same grace for forceful termination and confirmed reap.
    term_deadline = loop.time() + min(
        _PROCESS_STOP_GRACE_SECONDS, max(0.0, deadline - loop.time()) / 2
    )
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            while _process_group_exists(process.pid) and loop.time() < term_deadline:
                await asyncio.sleep(0.01)
            if _process_group_exists(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    else:
        tree_stopper: asyncio.subprocess.Process | None = None
        if os.name == "nt" and shutil.which("taskkill") is not None:
            try:
                tree_stopper = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    creationflags=0x08000000,  # CREATE_NO_WINDOW
                )
                async with asyncio.timeout_at(term_deadline):
                    await tree_stopper.wait()
            except (OSError, ProcessLookupError, TimeoutError):
                if tree_stopper is not None and tree_stopper.returncode is None:
                    tree_stopper.kill()
        if process.returncode is None:
            process.terminate()
            try:
                async with asyncio.timeout_at(term_deadline):
                    await process.wait()
                    return
            except (ProcessLookupError, TimeoutError):
                pass
            process.kill()
    try:
        reap = owner.retain(asyncio.create_task(process.wait()))
        await join_until(reap, deadline)
        if os.name == "posix":
            while (
                _process_group_exists(process.pid)
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.01)
            if _process_group_exists(process.pid):
                raise TimeoutError
    except TimeoutError:
        raise ModelProviderError(ProviderErrorCode.CLEANUP_TIMEOUT) from None
    except ProcessLookupError:
        pass


def _process_group_exists(process_id: int) -> bool:
    try:
        os.killpg(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _run_command(command: _Command) -> _CompletedCommand:
    if len(command.stdin) > _MAX_REQUEST_BYTES:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "subscription client request exceeds its byte bound",
        )
    if (
        sum(len(argument.encode("utf-8")) for argument in command.arguments)
        > _MAX_COMMAND_ARGUMENT_BYTES
    ):
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "subscription client arguments exceed their byte bound",
        )
    directory = command.cwd.stat()
    if not stat.S_ISDIR(directory.st_mode) or directory.st_mode & 0o077:
        raise ModelProviderError(
            ProviderErrorCode.LOCAL_ACCESS_ERROR,
            "subscription client working directory is not owner-only",
        )
    executable = shutil.which(command.arguments[0])
    if executable is None:
        raise _ExecutableUnavailable(command.arguments[0])
    if asyncio.get_running_loop().time() >= command.deadline:
        raise ModelProviderError(ProviderErrorCode.TIMEOUT)
    try:
        spawn = command.native_owner.start(
            asyncio.create_subprocess_exec(
                executable,
                *command.arguments[1:],
                cwd=command.cwd,
                env=dict(command.environment),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
                creationflags=(
                    0x00000200 if os.name == "nt" else 0
                ),  # CREATE_NEW_PROCESS_GROUP
            )
        )
        process = await join_until(spawn, command.deadline)
    except (asyncio.CancelledError, TimeoutError) as original:
        limit = command.shutdown_limit()

        async def retire_spawn():
            # Only native startup and process release remain owned here. A late
            # process cannot return command output to the retired attempt.
            late_process = await asyncio.shield(spawn)
            await _stop_process(
                late_process, deadline=limit, owner=command.native_owner
            )

        cleanup = command.native_owner.retain(asyncio.create_task(retire_spawn()))
        try:
            await await_cleanup(cleanup, deadline=limit, owner=command.native_owner)
        except ModelProviderError:
            command.native_owner.poisoned = True
        if isinstance(original, asyncio.CancelledError):
            raise
        raise ModelProviderError(
            ProviderErrorCode.TIMEOUT, cleanup_unresolved=command.native_owner.poisoned
        ) from None
    except OSError as error:
        raise _ExecutableUnavailable(command.arguments[0]) from error
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(_read_bounded(process.stdout, _MAX_STDOUT_BYTES))
    stderr_task = asyncio.create_task(_read_bounded(process.stderr, _MAX_STDERR_BYTES))
    cleanup_owner = command.native_owner
    cleanup_task = None
    cleanup_limit = None
    failed = False

    async def stop():
        nonlocal cleanup_task, cleanup_limit
        if cleanup_task is None:
            cleanup_limit = command.shutdown_limit()
            cleanup_task = asyncio.create_task(
                _stop_process(
                    process, deadline=cleanup_limit, owner=command.native_owner
                )
            )
        try:
            await await_cleanup(
                cleanup_task, deadline=cleanup_limit, owner=cleanup_owner
            )
        except ModelProviderError:
            # Called only after command failure/cancellation. The native owner
            # retains cleanup failure and blocks admission; preserve the cause.
            pass

    try:
        async with asyncio.timeout_at(command.deadline):
            try:
                process.stdin.write(command.stdin)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()
            stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
            returncode = await process.wait()
    except asyncio.CancelledError:
        failed = True
        await stop()
        raise
    except TimeoutError as error:
        failed = True
        await stop()
        raise ModelProviderError(
            ProviderErrorCode.TIMEOUT,
            "subscription client did not respond before the timeout",
        ) from error
    except _CommandOutputLimit as error:
        failed = True
        await stop()
        raise ModelProviderError(
            ProviderErrorCode.OUTPUT_LIMIT,
            "subscription client output exceeded its bound",
        ) from error
    finally:
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        limit = cleanup_limit or command.shutdown_limit()
        drain = asyncio.ensure_future(
            asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        )
        try:
            await await_cleanup(drain, deadline=limit, owner=cleanup_owner)
        except ModelProviderError:
            if not failed:
                raise
    return _CompletedCommand(returncode, stdout, stderr)


def _subscription_environment() -> dict[str, str]:
    """Expose only OS, login-location, TLS, and proxy process context."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_SUBSCRIPTION_ENVIRONMENT
    }
    environment["NO_COLOR"] = "1"
    environment["TERM"] = "dumb"
    return environment


def _owner_only_directory(path: Path) -> None:
    path.chmod(0o700)
    mode = path.stat().st_mode
    if not stat.S_ISDIR(mode) or mode & 0o077:
        raise ModelProviderError(
            ProviderErrorCode.LOCAL_ACCESS_ERROR,
            "subscription client working directory could not be isolated",
        )


def _write_owner_only(path: Path, value: bytes) -> None:
    if len(value) > _MAX_REQUEST_BYTES:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "subscription client request exceeds its byte bound",
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(value)
            stream.flush()
    finally:
        os.close(descriptor)


class _SubscriptionExecution:
    """Own the common bounded lifecycle for official-client subprocess calls."""

    def __init__(self) -> None:
        self.owner = NativeOwner()
        self._close = CloseCoordinator(self.owner)
        self._closed = False

    async def close(self, *, deadline: float | None = None) -> None:
        self._closed = True
        await self._close.drain(deadline=deadline)

    async def generate(
        self,
        request: ModelRequest,
        *,
        provider_id: str,
        display_name: str,
        operation: Callable[[ModelRequest, AttemptLifecycle], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        async def normalized(
            admitted: ModelRequest, attempt: AttemptLifecycle
        ) -> ModelResponse:
            try:
                if self._closed:
                    raise ModelProviderError(ProviderErrorCode.OWNER_UNAVAILABLE)
                return await operation(admitted, attempt)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                raise ModelProviderError(
                    ProviderErrorCode.TIMEOUT,
                    f"{display_name} subscription request exceeded its attempt deadline",
                    provider_id=provider_id,
                ) from None
            except ModelProviderError:
                raise
            except (
                KeyError,
                TypeError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                raise ModelProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    f"{display_name} subscription client returned a malformed response",
                    provider_id=provider_id,
                    diagnostic=ProviderFailureDiagnostic(
                        phase=ProviderFailurePhase.SUBSCRIPTION_OUTPUT,
                        code="response_decode_failed",
                    ),
                ) from None
            except Exception:
                raise ModelProviderError(
                    ProviderErrorCode.PROVIDER_UNAVAILABLE,
                    f"{display_name} subscription provider boundary failed",
                    provider_id=provider_id,
                ) from None

        return await execute_generate_attempt(
            self.owner,
            request,
            provider_id=provider_id,
            boundary_name=display_name,
            operation=normalized,
            headers_supported=False,
        )
