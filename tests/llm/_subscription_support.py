"""Shared helpers extracted from ``test_subscription_providers.py``."""

from __future__ import annotations

import base64
import json
import time

from daita.llm.subscription_auth import (
    CodexOAuthCredential,
)


def _jwt(account_id: str = "account-1", *, expires_at: int | None = None) -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return ".".join(
        (
            encode({"alg": "none"}),
            encode(
                {
                    "exp": expires_at or int(time.time()) + 3_600,
                    "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
                }
            ),
            encode("signature"),
        )
    )


def _credential(*, expired: bool = False) -> CodexOAuthCredential:
    expiry = time.time() - 10 if expired else time.time() + 3_600
    return CodexOAuthCredential(
        access_token=_jwt(expires_at=int(expiry)),
        refresh_token="refresh-token",
        expires_at=expiry,
        account_id="account-1",
    )
