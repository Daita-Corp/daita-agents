"""Agent fixtures: a real in-process Daita host for terminal surfaces."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from tests.support.host import StartHost, hosts


@pytest.fixture
async def start_host() -> AsyncIterator[StartHost]:
    """Start real hosts on short sockets; each is stopped and removed afterwards."""

    async with hosts() as start:
        yield start
