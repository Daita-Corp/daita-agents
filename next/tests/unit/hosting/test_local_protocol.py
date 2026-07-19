from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import daita.hosting.local_protocol as local_protocol
from daita.hosting.local_protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    LocalAgentClient,
    LocalErrorResponse,
    LocalProtocolError,
    LocalRequest,
    LocalSocketSecurityError,
    LocalSuccessResponse,
    decode_request,
    decode_response,
    encode_frame,
    encode_request,
    encode_response,
    local_socket_path,
    prepare_local_socket_path,
    read_frame,
    secure_bound_local_socket,
    validate_local_socket,
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _reader_for(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


class _FakeWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


def _mock_connection(
    monkeypatch: pytest.MonkeyPatch,
    response: LocalSuccessResponse,
) -> tuple[_FakeWriter, list[str]]:
    reader = _reader_for(encode_frame(encode_response(response)))
    writer = _FakeWriter()
    opened: list[str] = []

    async def open_unix_connection(*, path: str) -> tuple[object, object]:
        opened.append(path)
        return reader, writer

    monkeypatch.setattr(local_protocol, "validate_local_socket", lambda path: None)
    monkeypatch.setattr(asyncio, "open_unix_connection", open_unix_connection)
    return writer, opened


def test_request_and_response_records_round_trip_as_strict_json() -> None:
    request = LocalRequest.create(
        request_id="request-1",
        method="chat.submit",
        idempotency_key="chat-1",
        params={"message": "hello", "options": [1, True, None]},
    )

    assert decode_request(encode_request(request)) == request

    success = LocalSuccessResponse.create(
        request_id=request.request_id,
        result={"operation_id": "operation-1", "accepted": True},
    )
    assert decode_response(encode_response(success)) == success

    failure = LocalErrorResponse.create(
        request_id=request.request_id,
        code="operation_conflict",
        message="operation state changed",
        retryable=True,
        details={"operation_id": "operation-1"},
    )
    assert decode_response(encode_response(failure)) == failure


@pytest.mark.parametrize(
    "body,code",
    [
        (b"[]", "invalid_envelope"),
        (b'{"version":1', "invalid_json"),
        (b'{"value":NaN}', "invalid_json"),
        (b'{"value":1,"value":2}', "invalid_json"),
        (b'{"value":"\\ud800"}', "invalid_json"),
        (b"\xff", "invalid_utf8"),
    ],
)
def test_request_decoder_rejects_non_strict_json(body: bytes, code: str) -> None:
    with pytest.raises(LocalProtocolError) as caught:
        decode_request(body)

    assert caught.value.code == code


@pytest.mark.parametrize(
    "change,code",
    [
        ({"version": 2}, "unsupported_version"),
        ({"version": True}, "unsupported_version"),
        ({"request_id": 9}, "invalid_envelope"),
        ({"method": "Chat Submit"}, "invalid_envelope"),
        ({"idempotency_key": 9}, "invalid_envelope"),
        ({"params": []}, "invalid_envelope"),
        ({"extra": True}, "invalid_envelope"),
    ],
)
def test_request_decoder_rejects_invalid_envelopes(
    change: dict[str, object],
    code: str,
) -> None:
    value: dict[str, object] = {
        "version": PROTOCOL_VERSION,
        "request_id": "request-1",
        "method": "status",
        "idempotency_key": None,
        "params": {},
    }
    value.update(change)

    with pytest.raises(LocalProtocolError) as caught:
        decode_request(_json_bytes(value))

    assert caught.value.code == code


def test_request_decoder_rejects_missing_and_oversized_fields() -> None:
    missing = {
        "version": PROTOCOL_VERSION,
        "request_id": "request-1",
        "method": "status",
        "idempotency_key": None,
    }
    oversized = {
        **missing,
        "request_id": "r" * 129,
        "params": {},
    }

    with pytest.raises(LocalProtocolError, match="fields do not match"):
        decode_request(_json_bytes(missing))
    with pytest.raises(LocalProtocolError, match="byte limit"):
        decode_request(_json_bytes(oversized))


def test_protocol_rejects_surrogate_text_inside_valid_envelopes() -> None:
    request = (
        b'{"version":1,"request_id":"request-1","method":"chat.submit",'
        b'"idempotency_key":null,"params":{"message":"\\ud800"}}'
    )
    response = (
        b'{"version":1,"request_id":"request-1","ok":true,'
        b'"result":{"message":"\\ud800"}}'
    )

    with pytest.raises(LocalProtocolError) as request_error:
        decode_request(request)
    with pytest.raises(LocalProtocolError) as response_error:
        decode_response(response)

    assert request_error.value.code == "invalid_json"
    assert response_error.value.code == "invalid_json"


@pytest.mark.parametrize(
    "value",
    [
        {"version": 1, "request_id": "r", "ok": 1, "result": {}},
        {"version": 1, "request_id": "r", "ok": True, "error": {}},
        {
            "version": 1,
            "request_id": "r",
            "ok": False,
            "error": {
                "code": "bad",
                "message": "bad request",
                "retryable": "no",
                "details": {},
            },
        },
    ],
)
def test_response_decoder_rejects_wrong_shapes(value: object) -> None:
    with pytest.raises(LocalProtocolError, match="must|fields"):
        decode_response(_json_bytes(value))


def test_frame_codec_is_big_endian_and_bounded() -> None:
    payload = b'{"ok":true}'

    framed = encode_frame(payload)

    assert framed[:4] == len(payload).to_bytes(4, "big")
    assert framed[4:] == payload
    with pytest.raises(LocalProtocolError, match="1..1048576"):
        encode_frame(b"")
    with pytest.raises(LocalProtocolError, match="1..1048576"):
        encode_frame(b"x" * (MAX_FRAME_BYTES + 1))


@pytest.mark.parametrize(
    "data,code",
    [
        (b"\x00\x00", "truncated_frame"),
        ((0).to_bytes(4, "big"), "invalid_frame"),
        ((MAX_FRAME_BYTES + 1).to_bytes(4, "big"), "invalid_frame"),
        ((5).to_bytes(4, "big") + b"no", "truncated_frame"),
    ],
)
async def test_frame_reader_fails_closed(data: bytes, code: str) -> None:
    with pytest.raises(LocalProtocolError) as caught:
        await read_frame(_reader_for(data))

    assert caught.value.code == code


def test_socket_path_and_permissions_are_exact(tmp_path: Path) -> None:
    agent_home = tmp_path / "agent"
    agent_home.mkdir()

    path = prepare_local_socket_path(agent_home)

    assert path == agent_home / "run" / "agent.sock"
    assert stat.S_IMODE(os.lstat(path.parent).st_mode) == 0o700


def test_bound_socket_is_chmodded_to_0600_and_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "run" / "agent.sock"
    socket_mode = stat.S_IFSOCK | 0o777
    chmod_calls: list[tuple[Path, int]] = []

    def fake_lstat(candidate: str | Path) -> SimpleNamespace:
        candidate_path = Path(candidate)
        if candidate_path == path.parent:
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=os.geteuid())
        if candidate_path == path:
            return SimpleNamespace(st_mode=socket_mode, st_uid=os.geteuid())
        raise FileNotFoundError(candidate_path)

    def fake_chmod(candidate: str | Path, mode: int) -> None:
        nonlocal socket_mode
        chmod_calls.append((Path(candidate), mode))
        socket_mode = stat.S_IFSOCK | mode

    monkeypatch.setattr(os, "lstat", fake_lstat)
    monkeypatch.setattr(os, "chmod", fake_chmod)

    assert secure_bound_local_socket(path) == path
    validate_local_socket(path)
    assert chmod_calls == [(path, 0o600)]


def test_prepare_socket_path_rejects_symlinked_run_directory(tmp_path: Path) -> None:
    agent_home = tmp_path / "agent"
    elsewhere = tmp_path / "elsewhere"
    agent_home.mkdir()
    elsewhere.mkdir()
    (agent_home / "run").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(LocalSocketSecurityError, match="real directory"):
        prepare_local_socket_path(agent_home)


def test_local_socket_path_requires_an_absolute_agent_home() -> None:
    with pytest.raises(ValueError, match="absolute"):
        local_socket_path(Path("relative-agent"))


async def test_local_client_performs_one_request_and_creates_no_hidden_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_home = tmp_path / "agent"
    writer, opened = _mock_connection(
        monkeypatch,
        LocalSuccessResponse.create(
            request_id="request-1",
            result={"state": "running"},
        ),
    )
    before = asyncio.all_tasks()
    client = LocalAgentClient(agent_home)
    assert asyncio.all_tasks() == before

    response = await client.request(
        LocalRequest.create(request_id="request-1", method="host.status")
    )

    assert isinstance(response, LocalSuccessResponse)
    assert (
        response.result
        == LocalSuccessResponse.create(
            request_id="request-1", result={"state": "running"}
        ).result
    )
    assert opened == [str(agent_home / "run" / "agent.sock")]
    sent_length = int.from_bytes(writer.data[:4], "big")
    sent = decode_request(bytes(writer.data[4 : 4 + sent_length]))
    assert sent.method == "host.status"
    assert writer.closed
    assert not client.connected


async def test_client_enforces_one_request_and_response_per_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_home = tmp_path / "agent"
    _mock_connection(
        monkeypatch,
        LocalSuccessResponse.create(request_id="request-1", result={}),
    )
    client = LocalAgentClient(agent_home)
    request = LocalRequest.create(request_id="request-1", method="host.status")
    try:
        await client.connect()
        await client.send(request)
        with pytest.raises(RuntimeError, match="one request"):
            await client.send(request)
        assert isinstance(await client.read(), LocalSuccessResponse)
        with pytest.raises(RuntimeError, match="one response"):
            await client.read()
    finally:
        await client.close()


async def test_client_rejects_mismatched_response_and_insecure_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_home = tmp_path / "agent"
    _mock_connection(
        monkeypatch,
        LocalSuccessResponse.create(request_id="different-request"),
    )
    request = LocalRequest.create(request_id="request-1", method="host.status")
    with pytest.raises(LocalProtocolError, match="does not match"):
        await LocalAgentClient(agent_home).request(request)

    def reject_insecure(path: Path) -> None:
        raise LocalSocketSecurityError("local socket mode must be 0600")

    monkeypatch.setattr(local_protocol, "validate_local_socket", reject_insecure)
    with pytest.raises(LocalSocketSecurityError, match="0600"):
        await LocalAgentClient(agent_home).connect()
