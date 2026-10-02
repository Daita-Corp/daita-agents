"""Private JSON Schema assertion worker; no Daita import or credential access."""

from __future__ import annotations

import json
import sys
from typing import Never

from jsonschema import (  # type: ignore[import-untyped]
    Draft7Validator,
    Draft202012Validator,
)
from referencing import Registry


def _deny_reference(uri: str) -> Never:
    del uri
    raise ValueError("External schema retrieval is forbidden")


def _validator(schema):
    dialect = schema.get("$schema", "https://json-schema.org/draft/2020-12/schema")
    if dialect == "http://json-schema.org/draft-07/schema#":
        selected = Draft7Validator
    elif dialect == "https://json-schema.org/draft/2020-12/schema":
        selected = Draft202012Validator
    else:
        raise ValueError("Unsupported schema dialect")
    selected.check_schema(schema)
    return selected(schema, registry=Registry(retrieve=_deny_reference))  # type: ignore[call-arg]


def _main() -> None:
    # Bound CPU as well as the parent's cancellable wall deadline. Linux also
    # supports an address-space limit; Darwin rejects RLIMIT_AS changes.
    import resource

    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    request = json.loads(sys.stdin.buffer.read(1024 * 1024 + 1))
    result: object
    if request["operation"] == "schemas":
        checks: list[bool] = []
        for schema in request["schemas"]:
            try:
                _validator(schema)
            except Exception:
                checks.append(False)
            else:
                checks.append(True)
        result = checks
    elif request["operation"] == "value":
        try:
            result = _validator(request["schema"]).is_valid(request["value"])
        except Exception:
            result = False
    else:
        raise ValueError("Unsupported validation operation")
    sys.stdout.write(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    _main()
