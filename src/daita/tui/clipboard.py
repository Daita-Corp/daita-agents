"""Copy bounded terminal content through pbcopy or OSC 52 and report the mechanism."""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_CLIPBOARD_UTF8_BYTES = 64 * 1_024
_CLIPBOARD_TIMEOUT_SECONDS = 1.0
FORCE_SELECTION_GUIDANCE = (
    "Hold your terminal's selection bypass modifier (often Shift) "
    "to use terminal-owned selection."
)


@dataclass(frozen=True, slots=True)
class ClipboardResult:
    """Truthful outcome of one bounded terminal clipboard delivery attempt."""

    status: str
    mechanism: str
    message: str


def clipboard_mechanism(
    *,
    platform: str,
    environ: Mapping[str, str],
) -> str:
    """Choose one reviewed clipboard path without probing arbitrary commands."""

    if environ.get("SSH_TTY") or environ.get("SSH_CONNECTION"):
        return "osc52"
    return "pbcopy" if platform == "darwin" else "osc52"


def osc52_sequence(payload: bytes, *, tmux: bool) -> str:
    """Encode text so selected control-shaped data cannot become terminal syntax."""

    if len(payload) > MAX_CLIPBOARD_UTF8_BYTES:
        raise ValueError("clipboard payload exceeds the 64 KiB UTF-8 limit")
    encoded = base64.b64encode(payload).decode("ascii")
    request = f"\x1b]52;c;{encoded}\x07"
    return f"\x1bPtmux;\x1b{request}\x1b\\" if tmux else request


def copy_with_pbcopy(payload: bytes) -> ClipboardResult:
    """Invoke the exact local macOS clipboard utility with a fixed timeout."""

    try:
        import subprocess

        completed = subprocess.run(
            ("/usr/bin/pbcopy",),
            input=payload,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_CLIPBOARD_TIMEOUT_SECONDS,
        )
    except Exception:
        return ClipboardResult(
            "failure",
            "pbcopy",
            f"Copy failed. {FORCE_SELECTION_GUIDANCE}",
        )
    if completed.returncode != 0:
        return ClipboardResult(
            "failure",
            "pbcopy",
            f"Copy failed. {FORCE_SELECTION_GUIDANCE}",
        )
    return ClipboardResult("copied", "pbcopy", "Copied")


def send_osc52_request(
    output: Any,
    payload: bytes,
    *,
    tmux: bool,
) -> ClipboardResult:
    """Send one bounded, unacknowledged terminal clipboard request."""

    try:
        sequence = osc52_sequence(payload, tmux=tmux)
        writer = getattr(output, "write_raw", None)
        if callable(writer):
            writer(sequence)
        elif hasattr(output, "write"):
            output.write(sequence)
        else:
            sys.stdout.write(sequence)
        flusher = getattr(output, "flush", None)
        if callable(flusher):
            flusher()
        elif hasattr(sys.stdout, "flush"):
            sys.stdout.flush()
    except Exception:
        return ClipboardResult(
            "failure",
            "osc52-tmux" if tmux else "osc52",
            f"Copy failed. {FORCE_SELECTION_GUIDANCE}",
        )
    return ClipboardResult(
        "requested",
        "osc52-tmux" if tmux else "osc52",
        "Copy request sent to terminal",
    )


async def deliver_clipboard(
    text: str,
    *,
    output: Any = None,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ClipboardResult:
    """Deliver only one bounded visible selection through the chosen path."""

    if not isinstance(text, str) or not text:
        return ClipboardResult("failure", "none", "Copy failed: selection is empty.")
    payload = text.encode("utf-8")
    if len(payload) > MAX_CLIPBOARD_UTF8_BYTES:
        return ClipboardResult(
            "failure",
            "none",
            "Copy failed: selection exceeds the 64 KiB UTF-8 limit.",
        )
    environment = os.environ if environ is None else environ
    mechanism = clipboard_mechanism(
        platform=sys.platform if platform is None else platform,
        environ=environment,
    )
    if mechanism == "pbcopy":
        try:
            return await asyncio.to_thread(copy_with_pbcopy, payload)
        except Exception:
            return ClipboardResult(
                "failure",
                "pbcopy",
                f"Copy failed. {FORCE_SELECTION_GUIDANCE}",
            )
    return send_osc52_request(
        sys.stdout if output is None else output,
        payload,
        tmux=bool(environment.get("TMUX")),
    )
