"""Bounded subscription-authentication flows owned by Daita.

The Codex device flow is intentionally independent of the Codex executable.
Daita stores the resulting OAuth bundle through its existing secret-reference
boundary and refreshes it without reading or mutating another application's
login state.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .errors import ModelProviderError, ProviderErrorCode

_AUTH_BASE_URL = "https://auth.openai.com"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_DEVICE_CALLBACK_URL = f"{_AUTH_BASE_URL}/deviceauth/callback"
_DEVICE_VERIFICATION_URL = f"{_AUTH_BASE_URL}/codex/device"
_REQUEST_TIMEOUT_SECONDS = 30.0
_LOGIN_TIMEOUT_SECONDS = 15.0 * 60.0
_DEFAULT_POLL_INTERVAL_SECONDS = 5.0
_MIN_POLL_INTERVAL_SECONDS = 1.0
_MAX_RESPONSE_BYTES = 256 * 1_024
_MAX_ERROR_BYTES = 8 * 1_024
_MAX_SECRET_BYTES = 64 * 1_024
_MAX_ACCESS_TOKEN_CHARACTERS = 32 * 1_024
_MAX_REFRESH_TOKEN_CHARACTERS = 32 * 1_024
_MAX_ACCOUNT_ID_CHARACTERS = 256
_MAX_DEVICE_FIELD_CHARACTERS = 4 * 1_024
_MAX_USER_CODE_CHARACTERS = 256
_MAX_AUTHORIZATION_FIELD_CHARACTERS = 16 * 1_024
_REFRESH_SKEW_SECONDS = 120.0
_AUTH_CLAIM = "https://api.openai.com/auth"


@dataclass(frozen=True, slots=True)
class CodexDevicePrompt:
    """User-visible instructions for completing one device authorization."""

    verification_url: str
    user_code: str
    expires_in_seconds: int

    def __post_init__(self) -> None:
        if self.verification_url != _DEVICE_VERIFICATION_URL:
            raise ValueError("Codex verification URL is invalid")
        _bounded_ascii_graphic(
            self.user_code,
            "user code",
            maximum=_MAX_USER_CODE_CHARACTERS,
        )
        if (
            not isinstance(self.expires_in_seconds, int)
            or isinstance(self.expires_in_seconds, bool)
            or not 1 <= self.expires_in_seconds <= 24 * 60 * 60
        ):
            raise ValueError("Codex device-code expiry is invalid")


@dataclass(frozen=True, slots=True)
class CodexOAuthCredential:
    """One Daita-owned Codex OAuth credential bundle."""

    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str

    def __post_init__(self) -> None:
        _bounded_ascii_graphic(
            self.access_token,
            "access token",
            maximum=_MAX_ACCESS_TOKEN_CHARACTERS,
        )
        _bounded_ascii_graphic(
            self.refresh_token,
            "refresh token",
            maximum=_MAX_REFRESH_TOKEN_CHARACTERS,
        )
        _bounded_ascii_graphic(
            self.account_id,
            "account id",
            maximum=_MAX_ACCOUNT_ID_CHARACTERS,
        )
        if (
            not isinstance(self.expires_at, (int, float))
            or isinstance(self.expires_at, bool)
            or not math.isfinite(float(self.expires_at))
            or self.expires_at <= 0
        ):
            raise ValueError("Codex OAuth expiry must be a finite positive timestamp")
        if len(self.to_secret().encode("utf-8")) > _MAX_SECRET_BYTES:
            raise ValueError("Codex OAuth credential exceeds its 64 KiB bound")

    @property
    def needs_refresh(self) -> bool:
        return self.expires_at <= time.time() + _REFRESH_SKEW_SECONDS

    def to_secret(self) -> str:
        return json.dumps(
            {
                "access_token": self.access_token,
                "account_id": self.account_id,
                "expires_at": self.expires_at,
                "refresh_token": self.refresh_token,
                "type": "codex_oauth",
                "version": 1,
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_secret(cls, value: str) -> CodexOAuthCredential:
        if not isinstance(value, str) or not value:
            raise ValueError("Codex OAuth credential must be non-empty text")
        if len(value.encode("utf-8")) > _MAX_SECRET_BYTES:
            raise ValueError("Codex OAuth credential exceeds its 64 KiB bound")
        try:
            decoded = json.loads(
                value,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Codex OAuth credential is malformed") from error
        if not isinstance(decoded, dict) or set(decoded) != {
            "access_token",
            "account_id",
            "expires_at",
            "refresh_token",
            "type",
            "version",
        }:
            raise ValueError("Codex OAuth credential fields are invalid")
        if decoded["type"] != "codex_oauth" or decoded["version"] != 1:
            raise ValueError("Codex OAuth credential version is unsupported")
        return cls(
            access_token=cast(str, decoded["access_token"]),
            refresh_token=cast(str, decoded["refresh_token"]),
            expires_at=cast(float, decoded["expires_at"]),
            account_id=cast(str, decoded["account_id"]),
        )


@dataclass(frozen=True, slots=True)
class _HttpResult:
    status: int
    body: bytes

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is invalid: {value}")


def _bounded_ascii_graphic(value: object, label: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or not value.isascii()
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise ValueError(
            f"Codex OAuth {label} must be bounded printable ASCII without whitespace"
        )
    return value


def _json_object(result: _HttpResult, operation: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            result.body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ModelProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            f"OpenAI returned a malformed {operation} response",
        ) from error
    if not isinstance(value, dict):
        raise ModelProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            f"OpenAI returned a malformed {operation} response",
        )
    return value


def _required_text(value: object, label: str, *, maximum: int) -> str:
    try:
        return _bounded_ascii_graphic(value, label, maximum=maximum)
    except ValueError:
        raise ModelProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            f"OpenAI {label} response was incomplete",
        ) from None


def _bounded_read(response: object, limit: int) -> bytes:
    read = getattr(response, "read", None)
    if not callable(read):
        raise ValueError("HTTP response has no readable body")
    body = cast(bytes, read(limit + 1))
    if not isinstance(body, bytes) or len(body) > limit:
        raise ValueError("HTTP response body exceeded its bound")
    return body


def _post_sync(
    url: str,
    *,
    body: bytes,
    content_type: str,
) -> _HttpResult:
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": content_type,
            "User-Agent": "daita",
            "originator": "daita",
        },
    )
    try:
        opener = build_opener(_NoRedirect())
        with opener.open(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            return _HttpResult(
                status=cast(int, response.status),
                body=_bounded_read(response, _MAX_RESPONSE_BYTES),
            )
    except HTTPError as error:
        try:
            body_value = _bounded_read(error, _MAX_ERROR_BYTES)
        finally:
            error.close()
        return _HttpResult(status=error.code, body=body_value)
    except (TimeoutError, URLError, OSError) as error:
        raise ModelProviderError(
            ProviderErrorCode.PROVIDER_UNAVAILABLE,
            "Daita could not reach OpenAI's subscription login service",
        ) from error


async def _post_json(url: str, body: Mapping[str, object]) -> _HttpResult:
    encoded = json.dumps(
        dict(body), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return await asyncio.to_thread(
        _post_sync,
        url,
        body=encoded,
        content_type="application/json",
    )


async def _post_form(url: str, body: Mapping[str, str]) -> _HttpResult:
    encoded = urlencode(dict(body)).encode("ascii")
    return await asyncio.to_thread(
        _post_sync,
        url,
        body=encoded,
        content_type="application/x-www-form-urlencoded",
    )


def _positive_seconds(value: object, default: float) -> float:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    ):
        return min(60.0, max(float(value), _MIN_POLL_INTERVAL_SECONDS))
    return default


def _jwt_payload(token: str) -> Mapping[str, object] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padding = "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode((parts[1] + padding).encode("ascii"))
        value = json.loads(
            decoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _account_id(token: str) -> str:
    payload = _jwt_payload(token)
    auth = None if payload is None else payload.get(_AUTH_CLAIM)
    account = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    if not isinstance(account, str) or not account.strip():
        raise ModelProviderError(
            ProviderErrorCode.AUTHENTICATION_ERROR,
            "OpenAI login did not identify a ChatGPT account",
        )
    try:
        return _bounded_ascii_graphic(
            account,
            "account id",
            maximum=_MAX_ACCOUNT_ID_CHARACTERS,
        )
    except ValueError:
        raise ModelProviderError(
            ProviderErrorCode.AUTHENTICATION_ERROR,
            "OpenAI login did not identify a valid ChatGPT account",
        ) from None


def _token_expiry(token: str) -> float | None:
    payload = _jwt_payload(token)
    value = None if payload is None else payload.get("exp")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    ):
        return float(value)
    return None


def _oauth_credential(value: Mapping[str, object]) -> CodexOAuthCredential:
    access = _required_text(
        value.get("access_token"),
        "token exchange",
        maximum=_MAX_ACCESS_TOKEN_CHARACTERS,
    )
    refresh = _required_text(
        value.get("refresh_token"),
        "token exchange",
        maximum=_MAX_REFRESH_TOKEN_CHARACTERS,
    )
    expires_in = value.get("expires_in")
    expiry = (
        time.time() + float(expires_in)
        if isinstance(expires_in, (int, float))
        and not isinstance(expires_in, bool)
        and math.isfinite(float(expires_in))
        and expires_in > 0
        else _token_expiry(access)
    )
    if expiry is None:
        raise ModelProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            "OpenAI token exchange response was incomplete",
        )
    return CodexOAuthCredential(
        access_token=access,
        refresh_token=refresh,
        expires_at=expiry,
        account_id=_account_id(access),
    )


async def login_codex_subscription(
    *,
    on_verification: Callable[[CodexDevicePrompt], None],
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Authorize Daita through OpenAI's Codex device flow.

    Returns one opaque, persistable secret bundle. No Codex installation or
    pre-existing Codex login is consulted.
    """

    if not callable(on_verification):
        raise TypeError("on_verification must be callable")
    if on_progress is not None and not callable(on_progress):
        raise TypeError("on_progress must be callable")
    if on_progress is not None:
        on_progress("Requesting a ChatGPT device code")
    requested = await _post_json(
        f"{_AUTH_BASE_URL}/api/accounts/deviceauth/usercode",
        {"client_id": _CLIENT_ID},
    )
    if not requested.ok:
        raise ModelProviderError(
            ProviderErrorCode.AUTHENTICATION_ERROR,
            "OpenAI could not start Codex device login",
        )
    requested_body = _json_object(requested, "device login")
    device_auth_id = _required_text(
        requested_body.get("device_auth_id"),
        "device login",
        maximum=_MAX_DEVICE_FIELD_CHARACTERS,
    )
    user_code = _required_text(
        requested_body.get("user_code", requested_body.get("usercode")),
        "device login",
        maximum=_MAX_USER_CODE_CHARACTERS,
    )
    interval = _positive_seconds(
        requested_body.get("interval"), _DEFAULT_POLL_INTERVAL_SECONDS
    )
    on_verification(
        CodexDevicePrompt(
            verification_url=_DEVICE_VERIFICATION_URL,
            user_code=user_code,
            expires_in_seconds=int(_LOGIN_TIMEOUT_SECONDS),
        )
    )
    if on_progress is not None:
        on_progress("Waiting for ChatGPT authorization")

    deadline = time.monotonic() + _LOGIN_TIMEOUT_SECONDS
    authorization_code: str | None = None
    code_verifier: str | None = None
    while time.monotonic() < deadline:
        result = await _post_json(
            f"{_AUTH_BASE_URL}/api/accounts/deviceauth/token",
            {"device_auth_id": device_auth_id, "user_code": user_code},
        )
        if result.ok:
            body = _json_object(result, "device authorization")
            authorization_code = _required_text(
                body.get("authorization_code"),
                "device authorization",
                maximum=_MAX_AUTHORIZATION_FIELD_CHARACTERS,
            )
            code_verifier = _required_text(
                body.get("code_verifier"),
                "device authorization",
                maximum=_MAX_AUTHORIZATION_FIELD_CHARACTERS,
            )
            break
        if result.status not in {403, 404}:
            raise ModelProviderError(
                ProviderErrorCode.AUTHENTICATION_ERROR,
                "OpenAI rejected Codex device authorization",
            )
        await asyncio.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    if authorization_code is None or code_verifier is None:
        raise ModelProviderError(
            ProviderErrorCode.TIMEOUT,
            "OpenAI Codex device authorization timed out",
        )

    if on_progress is not None:
        on_progress("Completing ChatGPT authorization")
    exchanged = await _post_form(
        f"{_AUTH_BASE_URL}/oauth/token",
        {
            "client_id": _CLIENT_ID,
            "code": authorization_code,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "redirect_uri": _DEVICE_CALLBACK_URL,
        },
    )
    if not exchanged.ok:
        raise ModelProviderError(
            ProviderErrorCode.AUTHENTICATION_ERROR,
            "OpenAI rejected the Codex token exchange",
        )
    return _oauth_credential(_json_object(exchanged, "token exchange")).to_secret()


async def refresh_codex_subscription(
    credential: CodexOAuthCredential,
) -> CodexOAuthCredential:
    """Refresh one Daita-owned Codex credential, including token rotation."""

    if not isinstance(credential, CodexOAuthCredential):
        raise TypeError("credential must be a CodexOAuthCredential")
    result = await _post_form(
        f"{_AUTH_BASE_URL}/oauth/token",
        {
            "client_id": _CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": credential.refresh_token,
        },
    )
    if not result.ok:
        raise ModelProviderError(
            ProviderErrorCode.AUTHENTICATION_ERROR,
            "OpenAI Codex subscription login expired; sign in again",
        )
    return _oauth_credential(_json_object(result, "token refresh"))


__all__ = [
    "CodexDevicePrompt",
    "CodexOAuthCredential",
    "login_codex_subscription",
    "refresh_codex_subscription",
]
