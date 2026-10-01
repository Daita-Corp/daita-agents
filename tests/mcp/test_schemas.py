"""External schema semantics and cancellation use the real assertion worker."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import monotonic

import pytest

from daita import _json_schema
from daita._json import FrozenJsonObject
from daita._json_schema import (
    check_json_schemas,
    project_json_schema,
    validate_json_schema_value,
)


@pytest.mark.parametrize(
    "dialect",
    [
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2020-12/schema",
    ],
)
@pytest.mark.parametrize("composition", ["anyOf", "oneOf"])
async def test_projection_preserves_nullable_union_and_local_reference_semantics(
    dialect, composition
):
    raw = FrozenJsonObject.from_mapping(
        {
            "$schema": dialect,
            "type": "object",
            "definitions": {
                "value": {
                    composition: [
                        {"type": "string", "pattern": "^[a-z]+$"},
                        {"type": "null"},
                    ]
                }
            },
            "properties": {
                "value": {"$ref": "#/definitions/value"},
                "optional": {"type": "integer", "default": 7},
            },
            "required": ["value"],
            "additionalProperties": False,
            "propertyNames": {"type": "string"},
        }
    )
    projected, _ = project_json_schema(raw)
    assert "$schema" not in projected and "definitions" not in projected
    assert "propertyNames" not in projected
    assert await check_json_schemas((raw,)) == (True,)
    for value in ("lowercase", None):
        arguments = {"value": value}
        await validate_json_schema_value(raw, arguments)
        await validate_json_schema_value(projected, arguments)
        assert "optional" not in arguments
    for invalid_arguments in (
        {},
        {"value": "UPPER"},
        {"value": False},
        {"value": None, "invented": 1},
    ):
        for schema in (raw, projected):
            with pytest.raises(ValueError, match="invalid"):
                await validate_json_schema_value(schema, invalid_arguments)


@pytest.mark.parametrize("composition,valid", [("anyOf", True), ("oneOf", False)])
async def test_overlapping_union_branches_keep_anyof_and_oneof_distinct(
    composition, valid
):
    schema = {
        "type": "object",
        "properties": {
            "value": {composition: [{"type": "integer"}, {"type": "number"}]}
        },
    }
    projected, _ = project_json_schema(schema)
    if valid:
        await validate_json_schema_value(projected, {"value": 2})
    else:
        with pytest.raises(ValueError, match="invalid"):
            await validate_json_schema_value(projected, {"value": 2})


@pytest.mark.parametrize(
    "dialect,accepted",
    [
        ("http://json-schema.org/draft-07/schema#", True),
        ("https://json-schema.org/draft/2020-12/schema", False),
    ],
)
async def test_reference_sibling_assertions_follow_the_declared_dialect(
    dialect, accepted
):
    raw = {
        "$schema": dialect,
        "type": "object",
        "definitions": {"text": {"type": "string"}},
        "properties": {"value": {"$ref": "#/definitions/text", "maxLength": 2}},
    }
    projected, _ = project_json_schema(raw)
    for schema in (raw, projected):
        if accepted:
            await validate_json_schema_value(schema, {"value": "long"})
        else:
            with pytest.raises(ValueError, match="invalid"):
                await validate_json_schema_value(schema, {"value": "long"})


@pytest.mark.parametrize(
    "reference",
    ["https://invalid.test/schema", "file:///private/secret", "#/definitions/loop"],
)
def test_network_file_and_recursive_references_cannot_enter_a_contract(reference):
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": reference}},
        "definitions": {"loop": {"$ref": "#/definitions/loop"}},
    }
    with pytest.raises(ValueError):
        project_json_schema(schema)


async def test_schema_meta_validation_is_per_contract_and_does_not_echo_values():
    valid = FrozenJsonObject.from_mapping({"type": "object", "properties": {}})
    invalid = FrozenJsonObject.from_mapping(
        {"type": "object", "required": "not-an-array"}
    )
    assert await check_json_schemas((valid, invalid)) == (True, False)
    bounded = {"type": "object", "properties": {"value": {"type": "integer"}}}
    with pytest.raises(ValueError) as failure:
        await validate_json_schema_value(bounded, {"value": "sensitive"})
    assert "sensitive" not in str(failure.value)


@pytest.mark.parametrize("cancel", [False, True])
async def test_expensive_regex_validation_stops_and_reaps_its_process(
    monkeypatch, cancel
):
    launched: list[asyncio.subprocess.Process] = []
    started = asyncio.Event()
    original = asyncio.create_subprocess_exec

    async def launch(*args, **kwargs):
        process = await original(*args, **kwargs)
        launched.append(process)
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(_json_schema, "VALIDATION_TIMEOUT_SECONDS", 1.0)
    schema: Mapping[str, object] = {
        "type": "object",
        "properties": {"value": {"type": "string", "pattern": "^(a+)+$"}},
    }
    beginning = monotonic()
    task = asyncio.create_task(
        validate_json_schema_value(schema, {"value": "a" * 30000 + "!"})
    )
    await asyncio.wait_for(started.wait(), 2)
    # The parent loop stays responsive while the isolated validator works.
    await asyncio.sleep(0.05)
    assert not task.done()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(ValueError, match="finite deadline"):
            await task
    assert monotonic() - beginning < 3
    assert len(launched) == 1 and launched[0].returncode is not None
