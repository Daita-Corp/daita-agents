"""Bounded local Unix-socket framing and protocol records.

This module owns transport validation only.  It does not dispatch host methods
and creates no sockets or background work until an explicit async call is made.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
from types import TracebackType
from typing import Self, TypeAlias

from .._json import (
    FrozenJsonObject,
    FrozenJsonValue,
    canonical_json,
    freeze_json,
    thaw_json,
)

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
MAX_REQUEST_ID_BYTES = 128
MAX_METHOD_BYTES = 128
MAX_IDEMPOTENCY_KEY_BYTES = 256
MAX_ERROR_CODE_BYTES = 64
MAX_ERROR_MESSAGE_BYTES = 512
SOCKET_FILENAME = "agent.sock"

_METHOD_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_REQUEST_KEYS = frozenset(
    {
        "version",
        "request_id",
        "method",
        "idempotency_key",
        "params",
    }
)
_RESPONSE_COMMON_KEYS = frozenset({"version", "request_id", "ok"})
_SUCCESS_RESPONSE_KEYS = _RESPONSE_COMMON_KEYS | {"result"}
_ERROR_RESPONSE_KEYS = _RESPONSE_COMMON_KEYS | {"error"}
_ERROR_KEYS = frozenset({"code", "message", "retryable", "details"})


class LocalProtocolError(ValueError):
    """A bounded, peer-safe protocol failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LocalSocketSecurityError(RuntimeError):
    """The local endpoint does not satisfy its ownership or mode contract."""


def _bounded_text(
    value: object,
    *,
    field_name: str,
    maximum_bytes: int,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8 text") from exc
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{field_name} exceeds its byte limit")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValueError(f"{field_name} has an invalid format")
    return value


def _protocol_text(
    value: object,
    *,
    field_name: str,
    maximum_bytes: int,
    pattern: re.Pattern[str] | None = None,
) -> str:
    try:
        return _bounded_text(
            value,
            field_name=field_name,
            maximum_bytes=maximum_bytes,
            pattern=pattern,
        )
    except (TypeError, ValueError) as exc:
        raise LocalProtocolError("invalid_envelope", str(exc)) from exc


def _frozen_object(value: object, *, field_name: str) -> FrozenJsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    return FrozenJsonObject.from_mapping(value)


@dataclass(frozen=True, slots=True)
class LocalRequest:
    """One versioned local-host request envelope."""

    version: int
    request_id: str
    method: str
    idempotency_key: str | None
    params: FrozenJsonObject

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != PROTOCOL_VERSION:
            raise ValueError(f"version must equal {PROTOCOL_VERSION}")
        _bounded_text(
            self.request_id,
            field_name="request_id",
            maximum_bytes=MAX_REQUEST_ID_BYTES,
        )
        _bounded_text(
            self.method,
            field_name="method",
            maximum_bytes=MAX_METHOD_BYTES,
            pattern=_METHOD_PATTERN,
        )
        if self.idempotency_key is not None:
            _bounded_text(
                self.idempotency_key,
                field_name="idempotency_key",
                maximum_bytes=MAX_IDEMPOTENCY_KEY_BYTES,
            )
        if not isinstance(self.params, FrozenJsonObject):
            raise TypeError("params must be a FrozenJsonObject")

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        method: str,
        params: Mapping[str, object] | None = None,
        idempotency_key: str | None = None,
    ) -> Self:
        return cls(
            version=PROTOCOL_VERSION,
            request_id=request_id,
            method=method,
            idempotency_key=idempotency_key,
            params=FrozenJsonObject.from_mapping({} if params is None else params),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "method": self.method,
            "idempotency_key": self.idempotency_key,
            "params": self.params.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LocalError:
    """Stable structured error details returned by a local host."""

    code: str
    message: str
    retryable: bool = False
    details: FrozenJsonObject = field(default_factory=lambda: FrozenJsonObject(()))

    def __post_init__(self) -> None:
        _bounded_text(
            self.code,
            field_name="error code",
            maximum_bytes=MAX_ERROR_CODE_BYTES,
            pattern=_ERROR_CODE_PATTERN,
        )
        _bounded_text(
            self.message,
            field_name="error message",
            maximum_bytes=MAX_ERROR_MESSAGE_BYTES,
        )
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a bool")
        if not isinstance(self.details, FrozenJsonObject):
            raise TypeError("details must be a FrozenJsonObject")

    @classmethod
    def create(
        cls,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> Self:
        return cls(
            code=code,
            message=message,
            retryable=retryable,
            details=FrozenJsonObject.from_mapping({} if details is None else details),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details.to_dict(),
        }


def _validate_response_common(version: int, request_id: str) -> None:
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise ValueError(f"version must equal {PROTOCOL_VERSION}")
    _bounded_text(
        request_id,
        field_name="request_id",
        maximum_bytes=MAX_REQUEST_ID_BYTES,
    )


@dataclass(frozen=True, slots=True)
class LocalSuccessResponse:
    version: int
    request_id: str
    result: FrozenJsonValue

    def __post_init__(self) -> None:
        _validate_response_common(self.version, self.request_id)
        object.__setattr__(self, "result", freeze_json(self.result))

    @classmethod
    def create(cls, *, request_id: str, result: object = None) -> Self:
        return cls(
            version=PROTOCOL_VERSION,
            request_id=request_id,
            result=freeze_json(result),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "ok": True,
            "result": thaw_json(self.result),
        }


@dataclass(frozen=True, slots=True)
class LocalErrorResponse:
    version: int
    request_id: str
    error: LocalError

    def __post_init__(self) -> None:
        _validate_response_common(self.version, self.request_id)
        if not isinstance(self.error, LocalError):
            raise TypeError("error must be a LocalError")

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        code: str,
        message: str,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> Self:
        return cls(
            version=PROTOCOL_VERSION,
            request_id=request_id,
            error=LocalError.create(
                code=code,
                message=message,
                retryable=retryable,
                details=details,
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "ok": False,
            "error": self.error.to_wire(),
        }


LocalResponse: TypeAlias = LocalSuccessResponse | LocalErrorResponse


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_decoded_utf8(value: object) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            item.encode("utf-8", errors="strict")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _decode_json_object(payload: bytes) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise LocalProtocolError("invalid_frame", "JSON payload length is invalid")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LocalProtocolError("invalid_utf8", "frame is not strict UTF-8") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (RecursionError, ValueError, json.JSONDecodeError) as exc:
        raise LocalProtocolError("invalid_json", "frame is not strict JSON") from exc
    try:
        _validate_decoded_utf8(decoded)
    except UnicodeEncodeError as exc:
        raise LocalProtocolError(
            "invalid_json", "frame contains invalid Unicode"
        ) from exc
    if not isinstance(decoded, dict):
        raise LocalProtocolError("invalid_envelope", "envelope must be a JSON object")
    try:
        canonical_json(decoded).encode("utf-8", errors="strict")
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as exc:
        raise LocalProtocolError("invalid_json", "frame is not strict JSON") from exc
    return decoded


def _encode_json_object(value: Mapping[str, object]) -> bytes:
    try:
        payload = canonical_json(value).encode("utf-8")
    except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise LocalProtocolError("invalid_json", "value is not strict JSON") from exc
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise LocalProtocolError("frame_too_large", "encoded frame exceeds 1 MiB")
    return payload


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
) -> None:
    if frozenset(value) != expected:
        raise LocalProtocolError("invalid_envelope", "envelope fields do not match")


def encode_request(request: LocalRequest) -> bytes:
    """Encode a request JSON body without its four-byte frame header."""

    if not isinstance(request, LocalRequest):
        raise TypeError("request must be a LocalRequest")
    return _encode_json_object(request.to_wire())


def decode_request(payload: bytes) -> LocalRequest:
    value = _decode_json_object(payload)
    _require_exact_keys(value, _REQUEST_KEYS)
    version = value["version"]
    request_id = value["request_id"]
    method = value["method"]
    idempotency_key = value["idempotency_key"]
    params = value["params"]
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise LocalProtocolError(
            "unsupported_version", "protocol version is unsupported"
        )
    request_text = _protocol_text(
        request_id,
        field_name="request_id",
        maximum_bytes=MAX_REQUEST_ID_BYTES,
    )
    method_text = _protocol_text(
        method,
        field_name="method",
        maximum_bytes=MAX_METHOD_BYTES,
        pattern=_METHOD_PATTERN,
    )
    if idempotency_key is not None:
        idempotency_key = _protocol_text(
            idempotency_key,
            field_name="idempotency_key",
            maximum_bytes=MAX_IDEMPOTENCY_KEY_BYTES,
        )
    try:
        frozen_params = _frozen_object(params, field_name="params")
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as exc:
        raise LocalProtocolError(
            "invalid_envelope", "params must be strict JSON"
        ) from exc
    return LocalRequest(
        version=version,
        request_id=request_text,
        method=method_text,
        idempotency_key=idempotency_key,
        params=frozen_params,
    )


def encode_response(response: LocalResponse) -> bytes:
    """Encode a response JSON body without its four-byte frame header."""

    if not isinstance(response, (LocalSuccessResponse, LocalErrorResponse)):
        raise TypeError("response must be a local response record")
    return _encode_json_object(response.to_wire())


def decode_response(payload: bytes) -> LocalResponse:
    value = _decode_json_object(payload)
    if type(value.get("ok")) is not bool:
        raise LocalProtocolError("invalid_envelope", "ok must be a bool")
    expected = _SUCCESS_RESPONSE_KEYS if value["ok"] else _ERROR_RESPONSE_KEYS
    _require_exact_keys(value, expected)
    version = value["version"]
    request_id = value["request_id"]
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise LocalProtocolError(
            "unsupported_version", "protocol version is unsupported"
        )
    request_text = _protocol_text(
        request_id,
        field_name="request_id",
        maximum_bytes=MAX_REQUEST_ID_BYTES,
    )
    if value["ok"]:
        try:
            result = freeze_json(value["result"])
        except (RecursionError, TypeError, ValueError) as exc:
            raise LocalProtocolError(
                "invalid_envelope", "result must be strict JSON"
            ) from exc
        return LocalSuccessResponse(
            version=version,
            request_id=request_text,
            result=result,
        )

    error_value = value["error"]
    if not isinstance(error_value, dict):
        raise LocalProtocolError("invalid_envelope", "error must be a JSON object")
    _require_exact_keys(error_value, _ERROR_KEYS)
    code = _protocol_text(
        error_value["code"],
        field_name="error code",
        maximum_bytes=MAX_ERROR_CODE_BYTES,
        pattern=_ERROR_CODE_PATTERN,
    )
    message = _protocol_text(
        error_value["message"],
        field_name="error message",
        maximum_bytes=MAX_ERROR_MESSAGE_BYTES,
    )
    retryable = error_value["retryable"]
    if type(retryable) is not bool:
        raise LocalProtocolError("invalid_envelope", "retryable must be a bool")
    try:
        details = _frozen_object(error_value["details"], field_name="details")
    except (RecursionError, TypeError, ValueError) as exc:
        raise LocalProtocolError(
            "invalid_envelope", "details must be strict JSON"
        ) from exc
    return LocalErrorResponse(
        version=version,
        request_id=request_text,
        error=LocalError(
            code=code,
            message=message,
            retryable=retryable,
            details=details,
        ),
    )


def encode_frame(payload: bytes) -> bytes:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise LocalProtocolError("invalid_frame", "frame length must be 1..1048576")
    return len(payload).to_bytes(4, "big") + payload


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as exc:
        raise LocalProtocolError(
            "truncated_frame", "frame header is incomplete"
        ) from exc
    length = int.from_bytes(header, "big")
    if not 0 < length <= MAX_FRAME_BYTES:
        raise LocalProtocolError("invalid_frame", "frame length must be 1..1048576")
    try:
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise LocalProtocolError("truncated_frame", "frame body is incomplete") from exc


async def write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(encode_frame(payload))
    await writer.drain()


async def read_request(reader: asyncio.StreamReader) -> LocalRequest:
    return decode_request(await read_frame(reader))


async def write_request(
    writer: asyncio.StreamWriter,
    request: LocalRequest,
) -> None:
    await write_frame(writer, encode_request(request))


async def read_response(reader: asyncio.StreamReader) -> LocalResponse:
    return decode_response(await read_frame(reader))


async def write_response(
    writer: asyncio.StreamWriter,
    response: LocalResponse,
) -> None:
    await write_frame(writer, encode_response(response))


def local_socket_path(agent_home: str | Path) -> Path:
    home = Path(agent_home)
    if not home.is_absolute():
        raise ValueError("agent_home must be an absolute path")
    return home / "run" / SOCKET_FILENAME


def prepare_local_socket_path(agent_home: str | Path) -> Path:
    """Create and secure the endpoint directory, without touching a socket."""

    path = local_socket_path(agent_home)
    run_directory = path.parent
    try:
        run_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        metadata = os.lstat(run_directory)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise LocalSocketSecurityError("run path is not a real directory")
    except OSError as exc:
        raise LocalSocketSecurityError("could not create local run directory") from exc
    metadata = os.lstat(run_directory)
    if metadata.st_uid != os.geteuid():
        raise LocalSocketSecurityError("run directory is not owned by this user")
    os.chmod(run_directory, 0o700)
    return path


def secure_bound_local_socket(path: str | Path) -> Path:
    """Set and verify mode 0600 on an already bound Unix socket."""

    socket_path = Path(path)
    try:
        metadata = os.lstat(socket_path)
    except OSError as exc:
        raise LocalSocketSecurityError("local socket is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISSOCK(metadata.st_mode):
        raise LocalSocketSecurityError("local endpoint is not a Unix socket")
    if metadata.st_uid != os.geteuid():
        raise LocalSocketSecurityError("local socket is not owned by this user")
    os.chmod(socket_path, 0o600)
    validate_local_socket(socket_path)
    return socket_path


def validate_local_socket(path: str | Path) -> None:
    socket_path = Path(path)
    try:
        run_metadata = os.lstat(socket_path.parent)
        socket_metadata = os.lstat(socket_path)
    except OSError as exc:
        raise LocalSocketSecurityError("local socket is unavailable") from exc
    if stat.S_ISLNK(run_metadata.st_mode) or not stat.S_ISDIR(run_metadata.st_mode):
        raise LocalSocketSecurityError("run path is not a real directory")
    if stat.S_IMODE(run_metadata.st_mode) != 0o700:
        raise LocalSocketSecurityError("run directory mode must be 0700")
    if run_metadata.st_uid != os.geteuid():
        raise LocalSocketSecurityError("run directory is not owned by this user")
    if stat.S_ISLNK(socket_metadata.st_mode) or not stat.S_ISSOCK(
        socket_metadata.st_mode
    ):
        raise LocalSocketSecurityError("local endpoint is not a Unix socket")
    if stat.S_IMODE(socket_metadata.st_mode) != 0o600:
        raise LocalSocketSecurityError("local socket mode must be 0600")
    if socket_metadata.st_uid != os.geteuid():
        raise LocalSocketSecurityError("local socket is not owned by this user")


class LocalAgentClient:
    """A single-request client for one explicit local socket connection."""

    def __init__(self, agent_home: str | Path) -> None:
        self._socket_path = local_socket_path(agent_home)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._sent_request_id: str | None = None
        self._received = False
        self._closed = False

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> Self:
        if self._closed:
            raise RuntimeError("client is closed")
        if self._writer is not None:
            raise RuntimeError("client is already connected")
        validate_local_socket(self._socket_path)
        self._reader, self._writer = await asyncio.open_unix_connection(
            path=str(self._socket_path)
        )
        return self

    async def send(self, request: LocalRequest) -> None:
        if self._writer is None or self._writer.is_closing():
            raise RuntimeError("client is not connected")
        if self._sent_request_id is not None:
            raise RuntimeError("one request is allowed per connection")
        await write_request(self._writer, request)
        self._sent_request_id = request.request_id

    async def read(self) -> LocalResponse:
        if self._reader is None or self._sent_request_id is None:
            raise RuntimeError("a request must be sent before reading")
        if self._received:
            raise RuntimeError("one response is allowed per connection")
        response = await read_response(self._reader)
        self._received = True
        if response.request_id != self._sent_request_id:
            raise LocalProtocolError(
                "request_mismatch",
                "response request_id does not match the request",
            )
        return response

    async def request(self, request: LocalRequest) -> LocalResponse:
        await self.connect()
        try:
            await self.send(request)
            return await self.read()
        finally:
            await self.close()

    async def close(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        self._closed = True
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def __aenter__(self) -> Self:
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
