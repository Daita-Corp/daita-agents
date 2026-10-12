"""Inspect local scientific runtime identity without allocating an interpreter."""

from __future__ import annotations

import platform
import sys
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..._json import canonical_json

PACKAGE_VERSIONS = {
    "duckdb": "1.5.5",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.17.1",
    "pyarrow": "25.0.1",
    "matplotlib": "3.11.2",
    "networkx": "3.6.1",
}


def runtime_status() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in PACKAGE_VERSIONS:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    supported = sys.platform == "darwin" and platform.machine() == "arm64"
    helpers = {
        name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "worker.py",
            "guardian.py",
            "native.py",
            "outputs.py",
            "runtime.py",
            "cleanup.py",
        )
    }
    material = {
        "python": platform.python_version(),
        "executable_sha256": sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "packages": packages,
        "helpers": helpers,
        "platform": sys.platform,
        "architecture": platform.machine(),
        "protocol": 1,
    }
    return {
        "available": supported
        and all(
            packages[name] == expected for name, expected in PACKAGE_VERSIONS.items()
        ),
        "supported_platform": supported,
        "packages": packages,
        "required_packages": PACKAGE_VERSIONS,
        "runtime_identity": "sha256:"
        + sha256(canonical_json(material).encode()).hexdigest(),
        "identity_material": material,
    }
