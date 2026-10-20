"""Deterministic snapshot of everything the model is shown.

Scripted models replay old behaviour, so a change to a tool name, description,
input schema, toolbox presentation, guidance text or result shape passes the
deterministic suite by construction. This module runs one canonical offline
agent (local SQLite source, bounded workspace, scripted model, fixed clock, IDs
and timezone) and records the model-visible surface: every registry and
control tool definition, the instruction paragraphs the loop sends, the tools
offered at each step and the key structure of each rendered tool result.
``tests/architecture/test_model_surface.py`` keeps the committed snapshot
current and gates it on the last live smoke record.

The snapshot writes one compact line per tool, paragraph and step, so a review
diff names exactly what changed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from daita import Agent, LocalWorkspace
from daita._json import FrozenJsonObject
from daita.capabilities import (
    RESERVED_TOOL_NAMES,
    TOOLBOX_DEFINITIONS,
    CallTargetLane,
    CapabilityRegistry,
    ToolDefinition,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind
from tests.support.paths import REPO_ROOT

SNAPSHOT_DIRECTORY = REPO_ROOT / "tests" / "fixtures" / "model-surface"
SNAPSHOT_PATH = SNAPSHOT_DIRECTORY / "surface.json"
DIGEST_PATH = SNAPSHOT_DIRECTORY / "surface.sha256"
UPDATE_COMMAND = ".venv/bin/python scripts/model_surface.py update"

NOW = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)
TIMEZONE = "America/Chicago"
USER_MESSAGE = "How many orders are in the shop database? Count them exactly."
# Every on-demand tool applicable to this fixture's foreground run; loading
# them renders their code-owned procedure guidance into the next request.
ON_DEMAND_LOAD = (
    "artifact_create",
    "artifact_edit_text",
    "artifact_save_local",
    "artifact_set_export_location",
    "delivery_inspect",
    "memory_set",
    "python_execute",
    "routine_save",
    "skill_save",
)
# Lanes the model can be shown: foreground user runs, graph task workers, and
# machine origins that carry no per-call target ceiling.
LANES: tuple[tuple[str, CallTargetLane | None], ...] = (
    ("foreground", CallTargetLane.FOREGROUND),
    ("graph_task", CallTargetLane.GRAPH_TASK),
    ("machine", None),
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _script() -> tuple[ModelResponse, ...]:
    def calls(name: str, arguments: Mapping[str, object]) -> ModelResponse:
        call = ToolCall(id=f"call-{name}", name=name, arguments=dict(arguments))
        return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(call,))

    return (
        calls("catalog_search", {"query": "orders"}),
        calls(
            "data_query",
            {
                "source_id": "s1",
                "resource_ids": ["t1"],
                "sql": "SELECT COUNT(*) AS order_count FROM orders",
            },
        ),
        calls("toolbox_load", {"tool_names": list(ON_DEMAND_LOAD)}),
        ModelResponse(finish_reason=FinishReason.STOP, text="There are 3 orders."),
    )


def _seed(root: Path) -> tuple[Path, Path]:
    database = root / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE orders(id INTEGER PRIMARY KEY, placed_at TEXT NOT NULL, "
            "status TEXT NOT NULL, total_minor INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO orders(placed_at, status, total_minor) VALUES (?, ?, ?)",
            (
                ("2026-01-02", "paid", 1250),
                ("2026-01-03", "refunded", 400),
                ("2026-01-05", "paid", 980),
            ),
        )
    connection.close()
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("Quarterly notes.\n", encoding="utf-8")
    return database, workspace


def _ids() -> Any:
    counts: dict[str, int] = {}

    def create(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
        return f"{prefix}-{counts[prefix]}"

    return create


@contextmanager
def _fixed_host() -> Iterator[None]:
    """Pin the timezone and report the native analysis runtime as available.

    Analysis availability is a host fact (macOS ARM64 with pinned packages)
    that changes the registry and guidance; pinning it keeps the snapshot
    identical on CI and developer machines. The fixture never executes Python.
    """

    from daita.adapters.analytical_workspace import native, runtime
    from daita.domains.analysis import domain

    real = runtime.runtime_status

    def pinned() -> dict[str, object]:
        status = dict(real())
        status.update(
            available=True,
            supported_platform=True,
            runtime_identity="sha256:" + "0" * 64,
        )
        status.pop("identity_material", None)
        return status

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"TZ": TIMEZONE}))
        for module in (runtime, native, domain):
            stack.enter_context(patch.object(module, "runtime_status", pinned))
        stack.enter_context(patch.object(domain, "available", lambda: True))
        yield


def _plain(value: object) -> Any:
    if isinstance(value, FrozenJsonObject):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _shape(value: object) -> Any:
    """Key structure of a rendered result; values become their JSON type."""

    if isinstance(value, Mapping):
        return {str(key): _shape(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape(value[0])] if value else []
    return {bool: "bool", int: "int", float: "float", str: "str"}.get(
        type(value), "null"
    )


def _definition(definition: ToolDefinition) -> dict[str, object]:
    return {
        "description": definition.description,
        "input_schema": _plain(definition.input_schema),
    }


def _registry_tools(registry: CapabilityRegistry) -> dict[str, object]:
    tools: dict[str, object] = {}
    for name in sorted(registry.tool_names):
        view, capability = registry.resolve_tool(name)
        presentation = view.presentation
        rendered = {
            lane_name: _definition(registry.tool_definition(name, lane=lane))
            for lane_name, lane in LANES
        }
        foreground = rendered.pop("foreground")
        record: dict[str, object] = {
            "toolbox_id": presentation.toolbox_id.value,
            "load_mode": presentation.load_mode.value,
            "text_trust": presentation.text_trust.value,
            "summary": presentation.summary,
            "when_to_use": presentation.when_to_use,
            "keywords": list(presentation.keywords),
            "access_mode": capability.access_mode.value,
            "operational_effect": capability.operational_effect.value,
            **foreground,
        }
        if view.mode_capability_ids:
            record["mode_argument"] = view.mode_argument
            record["default_mode"] = view.default_mode
            record["modes"] = [mode for mode, _ in view.mode_capability_ids]
        differences = {
            lane: {
                key: value for key, value in item.items() if value != foreground[key]
            }
            for lane, item in rendered.items()
            if item != foreground
        }
        if differences:
            record["lane_differences"] = differences
        tools[name] = record
    return tools


def _requests(
    requests: Sequence[ModelRequest], registry: CapabilityRegistry
) -> dict[str, object]:
    """Instruction paragraphs, control tools and what each step offers and returns."""

    paragraphs: dict[str, None] = {}
    control: dict[str, object] = {}
    steps: list[object] = []
    seen_results: set[str] = set()
    for request in requests:
        overrides: dict[str, object] = {}
        for definition in request.tools:
            projected = _definition(definition)
            if definition.name in RESERVED_TOOL_NAMES:
                if control.setdefault(definition.name, projected) != projected:
                    overrides[definition.name] = projected
            elif projected != _definition(
                registry.tool_definition(
                    definition.name, lane=CallTargetLane.FOREGROUND
                )
            ):
                overrides[definition.name] = projected
        system = request.messages[0] if request.messages else None
        if system is not None and system.role is MessageRole.SYSTEM:
            for block in system.content:
                if isinstance(block, TextBlock):
                    paragraphs.update(dict.fromkeys(block.text.split("\n\n")))
        # Each tool result appears in every later request; record it once.
        results = {}
        for message in request.messages:
            for block in message.content:
                if (
                    isinstance(block, ToolResultBlock)
                    and block.call_id not in seen_results
                ):
                    seen_results.add(block.call_id)
                    results[block.call_id] = _shape(
                        model_tool_result(
                            block,
                            request.tool_result_refs.get(block.call_id),
                            request.scope_refs,
                        )
                    )
        step: dict[str, object] = {
            "offered_tools": sorted(item.name for item in request.tools),
            "allow_parallel_tool_calls": request.allow_parallel_tool_calls,
            "sensitivity": request.sensitivity.value,
        }
        if results:
            step["new_result_shapes"] = results
        if overrides:
            step["definitions_differing_from_registry"] = overrides
        steps.append(step)
    return {
        "instructions": list(paragraphs),
        "control_tools": control,
        "steps": steps,
    }


async def _capture(root: Path) -> dict[str, object]:
    database, workspace = _seed(root)
    downloads = root / "downloads"
    downloads.mkdir()
    provider = MockModelProvider(_script(), provider_id="mock:model-surface")
    agent = await Agent.create(
        "model-surface",
        root=root / "homes",
        workspace=LocalWorkspace(workspace),
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=200_000,
            max_output_tokens=8_000,
            supports_tools=True,
            supports_parallel_tools=True,
        ),
        clock=lambda: NOW,
        id_factory=_ids(),
        # A fixed existing folder: the machine's own Downloads would make the
        # destination's availability depend on where the snapshot runs.
        downloads_directory=downloads,
    )
    try:
        await agent.attach_sqlite(database, name="shop")
        result = await agent.run(USER_MESSAGE)
        if result.kind is not LoopExitKind.COMPLETED:
            raise RuntimeError(f"surface fixture run did not complete: {result}")
        transcript = await agent.transcript(result.run_id)
        failed = [
            block for _, block in transcript.tool_pairs if block and block.is_error
        ]
        if failed:
            raise RuntimeError(
                f"surface fixture tool call failed; the script is stale: {failed}"
            )
        registry = agent._embedded._capabilities
        return {
            "toolboxes": {
                item.id.value: {"label": item.label, "summary": item.summary}
                for item in TOOLBOX_DEFINITIONS
            },
            "registry_tools": _registry_tools(registry),
            **_requests(provider.requests, registry),
        }
    finally:
        await agent.close()


def _normalise(text: str, root: Path) -> str:
    """Remove the only run-varying material: temporary paths and digests."""

    text = text.replace(str(root), "<fixture-root>")
    numbers: dict[str, str] = {}

    def digest(match: re.Match[str]) -> str:
        return numbers.setdefault(match.group(0), f"sha256:<digest-{len(numbers) + 1}>")

    return _DIGEST.sub(digest, text)


def canonical_text(surface: Mapping[str, object]) -> str:
    """Valid JSON with one compact line per section entry, for reviewable diffs."""

    def compact(value: object) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    lines = ["{"]
    sections = sorted(surface.items())
    for index, (section, value) in enumerate(sections):
        if isinstance(value, Mapping):
            entries = [
                f"    {compact(key)}: {compact(item)}"
                for key, item in sorted(value.items())
            ]
            opening, closing = "{", "}"
        else:
            assert isinstance(value, list)
            entries = [f"    {compact(item)}" for item in value]
            opening, closing = "[", "]"
        lines.append(f"  {compact(section)}: {opening}")
        lines.append(",\n".join(entries))
        lines.append(f"  {closing}" + ("," if index < len(sections) - 1 else ""))
    lines.append("}")
    return "\n".join(line for line in lines if line) + "\n"


def compute_surface_text() -> str:
    """Compose the fixture offline and return the canonical snapshot text."""

    with tempfile.TemporaryDirectory(prefix="daita-model-surface-") as directory:
        # Agent homes refuse aliased paths, and macOS temp lives under a symlink.
        root = Path(directory).resolve()
        with _fixed_host():
            surface = asyncio.run(_capture(root))
        text = _normalise(json.dumps(surface, ensure_ascii=False), root)
    return canonical_text(json.loads(text))


def digest_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def committed_digest() -> str | None:
    try:
        return DIGEST_PATH.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def write_snapshot(text: str) -> str:
    SNAPSHOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(text, encoding="utf-8")
    digest = digest_text(text)
    DIGEST_PATH.write_text(digest + "\n", encoding="utf-8")
    return digest


def describe_changes(old_text: str, new_text: str) -> list[str]:
    """Name the sections and entries whose model-facing content changed."""

    try:
        old = json.loads(old_text)
    except json.JSONDecodeError:
        return ["committed snapshot is not valid JSON"]
    new = json.loads(new_text)
    changes: list[str] = []
    for section in sorted(set(old) | set(new)):
        before, after = old.get(section), new.get(section)
        if before == after:
            continue
        if isinstance(before, dict) and isinstance(after, dict):
            for name in sorted(set(before) | set(after)):
                if before.get(name) != after.get(name):
                    state = (
                        "added"
                        if name not in before
                        else "removed" if name not in after else "changed"
                    )
                    changes.append(f"{section}: {name} {state}")
        else:
            changes.append(f"{section} changed")
    return changes or ["formatting differs"]


__all__ = [
    "DIGEST_PATH",
    "SNAPSHOT_PATH",
    "UPDATE_COMMAND",
    "committed_digest",
    "compute_surface_text",
    "describe_changes",
    "digest_text",
    "write_snapshot",
]
