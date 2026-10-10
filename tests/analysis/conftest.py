"""Native slice tests apply to the explicitly supported first-delivery platform."""

import platform
import sys

import pytest


@pytest.fixture(autouse=True)
def supported_native_platform():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip("Native analysis first delivery supports macOS ARM64 only")
