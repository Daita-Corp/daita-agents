"""Host paths live under the state root and the socket path fits the OS limit."""

from __future__ import annotations

from pathlib import Path

import pytest

from daita.host.paths import MAX_SOCKET_PATH_BYTES, host_paths

pytestmark = pytest.mark.unit

# Resolution touches no files, and pytest's tmp_path can already exceed the limit.
BASE = Path("/tmp").resolve() / "daita-host-paths"


def test_host_files_live_under_the_resolved_state_root() -> None:
    paths = host_paths(BASE)

    assert paths.socket == BASE / "run" / "host.sock"
    assert paths.lock == BASE / "run" / "host.lock"
    assert paths.status == BASE / "run" / "host.json"
    assert paths.log == BASE / "logs" / "host.log"


def test_a_socket_path_over_the_limit_names_it_and_asks_for_a_shorter_root() -> None:
    suffix = len("/r/run/host.sock")
    fits = BASE / ("r" + "x" * (MAX_SOCKET_PATH_BYTES - len(str(BASE)) - suffix))
    too_long = Path(str(fits) + "x")

    assert len(str(host_paths(fits).socket)) == MAX_SOCKET_PATH_BYTES
    with pytest.raises(ValueError, match="Use a shorter --root") as raised:
        host_paths(too_long)
    assert str(too_long / "run" / "host.sock") in str(raised.value)
