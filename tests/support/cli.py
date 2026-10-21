"""Run the daita CLI in-process with captured streams."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from daita import cli


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def invoke(
    argv: list[str], *, stdin: str = "", tty: bool = False
) -> tuple[int, str, str]:
    """Run ``daita argv`` and return its exit code, stdout and stderr."""

    input_stream = TTYBuffer(stdin) if tty else io.StringIO(stdin)
    stdout = TTYBuffer() if tty else io.StringIO()
    stderr = TTYBuffer() if tty else io.StringIO()
    with (
        patch.object(sys, "stdin", input_stream),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            code = cli.main(argv)
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def json_lines(text: str) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in text.splitlines() if line.strip())


__all__ = ["TTYBuffer", "invoke", "json_lines"]
