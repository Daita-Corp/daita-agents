"""Bounded admission and isolated validation of external JSON Schema contracts.

Projection changes annotations and expands local references; jsonschema remains
the only assertion validator. Untrusted regex/composition work runs in a private
process so cancellation and deadlines can actually stop it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast

from ._json import FrozenJsonObject, canonical_json

MAX_SCHEMA_BYTES = 64 * 1024
MAX_SCHEMA_DEPTH = 12
MAX_SCHEMA_NODES = 1024
MAX_SCHEMA_REFERENCES = 32
MAX_VALUE_BYTES = 256 * 1024
MAX_VALUE_DEPTH = 20
MAX_VALUE_NODES = 4096
VALIDATION_TIMEOUT_SECONDS = 5.0
_MAX_WORKER_BYTES = 1024 * 1024
_DRAFT_7 = "http://json-schema.org/draft-07/schema#"
_DRAFT_2020 = "https://json-schema.org/draft/2020-12/schema"
_ANNOTATIONS = frozenset(
    {
        "description",
        "title",
        "examples",
        "default",
        "$comment",
        "deprecated",
        "readOnly",
        "writeOnly",
        "format",
        "x-mcp-header",
    }
)
PROJECTED_SCHEMA_KEYWORDS = frozenset(
    {
        "type",
        "enum",
        "const",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "properties",
        "required",
        "additionalProperties",
        "minProperties",
        "maxProperties",
        "propertyNames",
        "anyOf",
        "oneOf",
        "allOf",
    }
)
_ASSERTIONS = PROJECTED_SCHEMA_KEYWORDS
_DECLARATIONS = frozenset({"$schema", "$ref", "$defs", "definitions"})
_MAPPINGS = frozenset({"properties", "$defs", "definitions"})
_COMPOSITIONS = frozenset({"anyOf", "oneOf", "allOf"})


def _bound_tree(value: object, *, depth: int, nodes: int) -> None:
    pending = [(value, 1)]
    count = 0
    while pending:
        item, level = pending.pop()
        count += 1
        if level > depth or count > nodes:
            raise ValueError("JSON Schema contract exceeds its depth or node bound")
        if isinstance(item, Mapping):
            pending.extend((child, level + 1) for child in item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend((child, level + 1) for child in item)


def project_json_schema(schema: Mapping[str, object]) -> tuple[FrozenJsonObject, str]:
    """Bound a supported contract and derive an assertion-preserving projection.

    The caller checks the complete schema with check_json_schemas before remote
    admission. This function performs finite structural/projection work only.
    """

    _bound_tree(schema, depth=MAX_SCHEMA_DEPTH * 3, nodes=MAX_SCHEMA_NODES * 4)
    raw = FrozenJsonObject.from_mapping(schema)
    encoded = canonical_json(raw).encode("utf-8")
    if len(encoded) > MAX_SCHEMA_BYTES:
        raise ValueError("schema exceeds the fixed byte bound")
    dialect = raw.get("$schema", _DRAFT_2020)
    if dialect not in {_DRAFT_7, _DRAFT_2020}:
        raise ValueError("schema dialect is unsupported")
    if raw.get("type") != "object":
        raise ValueError("schema root must have type object")
    count = 0
    references = 0

    def project(node: object, depth: int, path: tuple[str, ...]) -> object:
        nonlocal count, references
        count += 1
        if depth > MAX_SCHEMA_DEPTH or count > MAX_SCHEMA_NODES:
            raise ValueError("schema expansion exceeds the fixed depth or node bound")
        if isinstance(node, bool):
            return node
        if not isinstance(node, Mapping):
            raise ValueError("schema rule must be an object or boolean")
        unsupported = set(node) - _ASSERTIONS - _ANNOTATIONS - _DECLARATIONS
        if unsupported:
            raise ValueError(f"unsupported schema keyword: {sorted(unsupported)[0]}")
        if "$schema" in node and node["$schema"] != dialect:
            raise ValueError("nested schema dialect change is unsupported")
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/"):
                raise ValueError(
                    "only local JSON Pointer schema references are supported"
                )
            if ref in path:
                raise ValueError("recursive schema references are unsupported")
            references += 1
            if references > MAX_SCHEMA_REFERENCES:
                raise ValueError("schema exceeds the fixed reference bound")
            target: object = raw
            for token in ref[2:].split("/"):
                token = token.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, Mapping) or token not in target:
                    raise ValueError(
                        "schema reference does not resolve to a local rule"
                    )
                target = target[token]
            expanded = project(target, depth + 1, (*path, ref))
            siblings = {key: item for key, item in node.items() if key in _ASSERTIONS}
            if dialect == _DRAFT_7 or not siblings:
                return expanded
            return {"allOf": [expanded, project(siblings, depth + 1, path)]}
        result: dict[str, object] = {}
        for key, item in node.items():
            if key in _ANNOTATIONS or key in {"$schema", "$defs", "definitions"}:
                continue
            if key == "properties":
                if not isinstance(item, Mapping) or len(item) > 128:
                    raise ValueError("schema properties must be a bounded object")
                result[key] = {
                    name: project(rule, depth + 1, path) for name, rule in item.items()
                }
            elif key == "propertyNames":
                rule = project(item, depth + 1, path)
                # Every JSON object key is already a string. Remove only this
                # provably redundant assertion; retain all other name rules.
                if rule is not True and rule != {"type": "string"}:
                    result[key] = rule
            elif key in {"items", "additionalProperties"}:
                result[key] = project(item, depth + 1, path)
            elif key in _COMPOSITIONS:
                if not isinstance(item, (list, tuple)) or not 1 <= len(item) <= 8:
                    raise ValueError(
                        "schema composition requires one to eight alternatives"
                    )
                result[key] = [project(rule, depth + 1, path) for rule in item]
            else:
                result[key] = item
        return result

    # Check all rules, including unused definitions, so no network reference or
    # unsupported assertion is retained in an admitted raw contract.
    pending: list[tuple[object, int]] = [(raw, 1)]
    while pending:
        node, depth = pending.pop()
        if not isinstance(node, Mapping):
            continue
        project(node, depth, ())
        for key, item in node.items():
            if key in _MAPPINGS and isinstance(item, Mapping):
                pending.extend((child, depth + 1) for child in item.values())
            elif key in _COMPOSITIONS and isinstance(item, (list, tuple)):
                pending.extend((child, depth + 1) for child in item)
            elif key in {"items", "additionalProperties", "propertyNames"}:
                pending.append((item, depth + 1))
    count = 0
    references = 0
    projected = cast(Mapping[str, object], project(raw, 1, ()))
    frozen = FrozenJsonObject.from_mapping(projected)
    if len(canonical_json(frozen).encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise ValueError("schema projection exceeds the fixed byte bound")
    return frozen, "sha256:" + sha256(encoded).hexdigest()


async def check_json_schemas(schemas: Sequence[FrozenJsonObject]) -> tuple[bool, ...]:
    """Check bounded complete contracts with their declared jsonschema dialect."""

    if not schemas:
        return ()
    if len(schemas) > 512:
        raise ValueError("JSON Schema inspection exceeds its schema count bound")
    batches: list[list[FrozenJsonObject]] = [[]]
    size = 0
    for schema in schemas:
        amount = len(canonical_json(schema).encode("utf-8"))
        if amount > MAX_SCHEMA_BYTES:
            raise ValueError("JSON Schema exceeds its fixed byte bound")
        if size + amount > _MAX_WORKER_BYTES // 2:
            batches.append([])
            size = 0
        batches[-1].append(schema)
        size += amount
    results: list[bool] = []
    try:
        async with asyncio.timeout(VALIDATION_TIMEOUT_SECONDS):
            for batch in batches:
                result = await _worker({"operation": "schemas", "schemas": batch})
                if not isinstance(result, list) or len(result) != len(batch):
                    raise ValueError(
                        "JSON Schema validation worker returned invalid evidence"
                    )
                if any(not isinstance(item, bool) for item in result):
                    raise ValueError(
                        "JSON Schema validation worker returned invalid evidence"
                    )
                results.extend(result)
    except TimeoutError:
        raise ValueError(
            "JSON Schema validation exceeded its finite deadline"
        ) from None
    return tuple(results)


async def validate_json_schema_value(
    schema: Mapping[str, object], value: Mapping[str, object]
) -> None:
    _bound_tree(value, depth=MAX_VALUE_DEPTH, nodes=MAX_VALUE_NODES)
    frozen = FrozenJsonObject.from_mapping(value)
    if len(canonical_json(frozen).encode("utf-8")) > MAX_VALUE_BYTES:
        raise ValueError("JSON Schema value exceeds the fixed byte bound")
    if (
        await _worker({"operation": "value", "schema": schema, "value": frozen})
        is not True
    ):
        raise ValueError("JSON Schema value is invalid")


async def _worker(request: Mapping[str, object]) -> object:
    payload = canonical_json(request).encode("utf-8")
    if len(payload) > _MAX_WORKER_BYTES:
        raise ValueError("JSON Schema validation request exceeds its byte bound")
    process: asyncio.subprocess.Process | None = None
    start = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(Path(__file__).with_name("_json_schema_worker.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={},
        )
    )
    try:
        async with asyncio.timeout(VALIDATION_TIMEOUT_SECONDS):
            process = await asyncio.shield(start)
            stdout, _ = await process.communicate(payload)
            if process.returncode != 0 or len(stdout) > 2048:
                raise ValueError("JSON Schema validation worker failed")
            return json.loads(stdout)
    except TimeoutError:
        raise ValueError(
            "JSON Schema validation exceeded its finite deadline"
        ) from None
    finally:
        # Even cancellation during spawn must acquire, kill and reap the child.
        if process is None:
            process = await asyncio.shield(start)
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await asyncio.shield(process.wait())


__all__ = ["check_json_schemas", "project_json_schema", "validate_json_schema_value"]
