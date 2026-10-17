"""Deterministic snapshot of everything the model is shown.

Scripted models replay old behaviour, so a change to a tool name, description,
input schema, toolbox presentation, static guidance or rendered tool result
passes the deterministic suite by construction. This module composes one
canonical offline agent (local SQLite source, bounded workspace, scripted
model, fixed clock, fixed IDs and timezone) and extracts the model-visible
surface as canonical JSON. ``tests/architecture/test_model_surface.py`` keeps
the committed snapshot current and gates it against the last live smoke record.

The extraction only observes production projections: registry tool
definitions per call-target lane and the exact canonical requests the loop
submits. Results are rendered with ``model_tool_result`` exactly as provider
adapters render them.
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
    CanonicalMessage,
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

SURFACE_FORMAT = 1
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
    "file_query",
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
    def call(call_id: str, name: str, arguments: Mapping[str, object]) -> ToolCall:
        return ToolCall(id=call_id, name=name, arguments=dict(arguments))

    def calls(*items: ToolCall) -> ModelResponse:
        return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=items)

    return (
        calls(call("call-search", "catalog_search", {"query": "orders"})),
        calls(
            call(
                "call-query",
                "data_query",
                {
                    "source_id": "s1",
                    "resource_ids": ["t1"],
                    "sql": "SELECT COUNT(*) AS order_count FROM orders",
                },
            )
        ),
        calls(call("call-load", "toolbox_load", {"tool_names": list(ON_DEMAND_LOAD)})),
        ModelResponse(finish_reason=FinishReason.STOP, text="There are 3 orders."),
    )


def _seed(root: Path) -> tuple[Path, Path]:
    database = root / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE orders("
            "id INTEGER PRIMARY KEY, placed_at TEXT NOT NULL, "
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
def _fixed_timezone() -> Iterator[None]:
    prior = os.environ.get("TZ")
    os.environ["TZ"] = TIMEZONE
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = prior


@contextmanager
def _pinned_analysis_runtime() -> Iterator[None]:
    """Report the native analysis runtime as available on every platform.

    Its availability is a host fact (macOS ARM64 with pinned packages) that
    changes the registry and guidance. Pinning it makes the snapshot identical
    on CI and developer machines and covers the larger, analysis-enabled
    surface. Only availability metadata is pinned; the fixture never executes
    Python.
    """

    from daita.adapters.analytical_workspace import native, runtime
    from daita.domains.analysis import domain

    real = runtime.runtime_status

    def pinned() -> dict[str, object]:
        status = dict(real())
        status["available"] = True
        status["supported_platform"] = True
        status["runtime_identity"] = "sha256:" + "0" * 64
        status.pop("identity_material", None)
        return status

    def available() -> bool:
        return True

    with ExitStack() as stack:
        stack.enter_context(patch.object(runtime, "runtime_status", pinned))
        stack.enter_context(patch.object(native, "runtime_status", pinned))
        stack.enter_context(patch.object(domain, "runtime_status", pinned))
        stack.enter_context(patch.object(domain, "available", available))
        yield


def _plain(value: object) -> Any:
    if isinstance(value, FrozenJsonObject):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _without_timings(value: Any) -> Any:
    """Replace measured durations in results; their presence stays visible."""

    if isinstance(value, Mapping):
        return {
            key: (
                "<measured>"
                if isinstance(key, str)
                and ("elapsed" in key or "duration" in key)
                and isinstance(item, (int, float))
                else _without_timings(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_without_timings(item) for item in value]
    return value


def _definition(definition: ToolDefinition) -> dict[str, object]:
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": _plain(definition.input_schema),
    }


def _registry_tools(registry: CapabilityRegistry) -> dict[str, object]:
    tools: dict[str, object] = {}
    for name in sorted(registry.tool_names):
        view, capability = registry.resolve_tool(name)
        rendered = {
            lane_name: _definition(registry.tool_definition(name, lane=lane))
            for lane_name, lane in LANES
        }
        foreground = rendered["foreground"]
        presentation = view.presentation
        record: dict[str, object] = {
            "toolbox_id": presentation.toolbox_id.value,
            "load_mode": presentation.load_mode.value,
            "text_trust": presentation.text_trust.value,
            "summary": presentation.summary,
            "when_to_use": presentation.when_to_use,
            "keywords": list(presentation.keywords),
            "access_mode": capability.access_mode.value,
            "operational_effect": capability.operational_effect.value,
            "description": foreground["description"],
            "input_schema": foreground["input_schema"],
        }
        if view.mode_capability_ids:
            record["mode_argument"] = view.mode_argument
            record["default_mode"] = view.default_mode
            record["modes"] = [mode for mode, _ in view.mode_capability_ids]
        lane_differences = {
            lane_name: {
                key: value
                for key, value in definition.items()
                if key != "name" and value != foreground[key]
            }
            for lane_name, definition in rendered.items()
            if lane_name != "foreground" and definition != foreground
        }
        if lane_differences:
            record["lane_differences"] = lane_differences
        tools[name] = record
    return tools


def _message(message: CanonicalMessage, request: ModelRequest) -> dict[str, object]:
    record: dict[str, object] = {"role": message.role.value}
    texts = [block.text for block in message.content if isinstance(block, TextBlock)]
    if texts:
        record["text"] = texts if len(texts) > 1 else texts[0]
    if message.tool_calls:
        record["tool_calls"] = [
            {"name": call.name, "arguments": _plain(call.arguments)}
            for call in message.tool_calls
        ]
    results = [
        _without_timings(
            model_tool_result(
                block, request.tool_result_refs.get(block.call_id), request.scope_refs
            )
        )
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    if results:
        record["tool_results"] = results
    return record


def _requests(
    requests: Sequence[ModelRequest], registry: CapabilityRegistry
) -> tuple[list[object], dict[str, object]]:
    rendered: list[object] = []
    control: dict[str, object] = {}
    previous: list[dict[str, object]] = []
    previous_instructions: object = None
    for step, request in enumerate(requests):
        messages = [_message(message, request) for message in request.messages]
        offered: list[str] = []
        overrides: dict[str, object] = {}
        for definition in request.tools:
            offered.append(definition.name)
            projected = _definition(definition)
            if definition.name in RESERVED_TOOL_NAMES:
                prior = control.setdefault(definition.name, projected)
                if prior != projected:
                    overrides[definition.name] = projected
                continue
            expected = _definition(
                registry.tool_definition(
                    definition.name, lane=CallTargetLane.FOREGROUND
                )
            )
            if projected != expected:
                overrides[definition.name] = projected
        # The leading system message holds the prepared instructions and the
        # guidance for the callable set; it is recorded once per change.
        instructions: object = None
        if messages and messages[0]["role"] == MessageRole.SYSTEM.value:
            text = messages[0].get("text")
            # Paragraphs keep snapshot diffs reviewable line by line.
            instructions = text.split("\n\n") if isinstance(text, str) else text
            messages = messages[1:]
        # Requests repeat the transcript; record only what each step adds.
        shared = 0
        while (
            shared < min(len(previous), len(messages))
            and previous[shared] == messages[shared]
        ):
            shared += 1
        record: dict[str, object] = {
            "step": step,
            "instructions": (
                "<unchanged from previous step>"
                if instructions == previous_instructions and step
                else instructions
            ),
            "offered_tools": sorted(offered),
            "unchanged_leading_messages": shared,
            "messages": messages[shared:],
            "allow_parallel_tool_calls": request.allow_parallel_tool_calls,
            "max_total_tokens": request.max_total_tokens,
            "sensitivity": request.sensitivity.value,
        }
        if overrides:
            record["definitions_differing_from_registry"] = overrides
        rendered.append(record)
        previous = messages
        previous_instructions = instructions
    return rendered, control


async def _capture(root: Path) -> dict[str, object]:
    database, workspace = _seed(root)
    downloads = root / "downloads"
    downloads.mkdir()
    provider = MockModelProvider(_script(), provider_id="mock:model-surface")
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=200_000,
        max_output_tokens=8_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )
    agent = await Agent.create(
        "model-surface",
        root=root / "homes",
        workspace=LocalWorkspace(workspace),
        model=provider,
        model_profile=profile,
        clock=lambda: NOW,
        id_factory=_ids(),
        # A fixed existing folder: the machine's own Downloads would make the
        # destination's availability depend on where the snapshot runs.
        downloads_directory=downloads,
    )
    try:
        await agent.attach_sqlite(database, name="shop")
        result = await agent.run(USER_MESSAGE)
        registry = agent._embedded._capabilities
        if result.kind is not LoopExitKind.COMPLETED:
            raise RuntimeError(f"surface fixture run did not complete: {result}")
        requests, control = _requests(provider.requests, registry)
        for step in requests:
            assert isinstance(step, dict)
            for message in step["messages"]:
                for item in message.get("tool_results", ()):
                    if item["is_error"]:
                        raise RuntimeError(
                            "surface fixture tool call failed; the fixture script "
                            f"is stale: {item['output']}"
                        )
        return {
            "format": SURFACE_FORMAT,
            "fixture": {
                "user_message": USER_MESSAGE,
                "timezone": TIMEZONE,
                "clock": NOW.isoformat(),
                "on_demand_load": list(ON_DEMAND_LOAD),
            },
            "toolboxes": [
                {"id": item.id.value, "label": item.label, "summary": item.summary}
                for item in TOOLBOX_DEFINITIONS
            ],
            "registry_tools": _registry_tools(registry),
            "control_tools": control,
            "foreground_requests": requests,
        }
    finally:
        await agent.close()


def _normalise(text: str, root: Path) -> str:
    """Remove the only run-varying material: temporary paths and digests.

    Source and resource identities derive from fixture paths, so every digest
    is renumbered by first appearance. The content they cover is captured
    directly elsewhere in the surface.
    """

    for spelling in {str(root), str(root.resolve())}:
        text = text.replace(spelling, "<fixture-root>")
    numbers: dict[str, str] = {}

    def digest(match: re.Match[str]) -> str:
        return numbers.setdefault(match.group(0), f"sha256:<digest-{len(numbers) + 1}>")

    return _DIGEST.sub(digest, text)


def canonical_text(surface: Mapping[str, object]) -> str:
    return json.dumps(surface, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def compute_surface_text() -> str:
    """Compose the fixture offline and return the canonical snapshot text."""

    with tempfile.TemporaryDirectory(prefix="daita-model-surface-") as directory:
        root = Path(directory)
        with _fixed_timezone(), _pinned_analysis_runtime():
            surface = asyncio.run(_capture(root))
        text = _normalise(canonical_text(surface), root)
    # Reparse so the committed form is canonical after normalisation.
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
    """Name the tools and sections whose model-facing content changed."""

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
        if (
            isinstance(before, dict)
            and isinstance(after, dict)
            and section
            in {
                "registry_tools",
                "control_tools",
            }
        ):
            for name in sorted(set(before) | set(after)):
                if name not in before:
                    changes.append(f"{section}: added {name}")
                elif name not in after:
                    changes.append(f"{section}: removed {name}")
                elif before[name] != after[name]:
                    fields = sorted(
                        key
                        for key in set(before[name]) | set(after[name])
                        if before[name].get(key) != after[name].get(key)
                    )
                    changes.append(f"{section}: {name} changed ({', '.join(fields)})")
        elif isinstance(before, list) and isinstance(after, list):
            if len(before) != len(after):
                changes.append(f"{section}: {len(before)} -> {len(after)} entries")
            for index, (left, right) in enumerate(zip(before, after)):
                if left != right:
                    if isinstance(left, dict) and isinstance(right, dict):
                        fields = sorted(
                            key
                            for key in set(left) | set(right)
                            if left.get(key) != right.get(key)
                        )
                        changes.append(
                            f"{section}[{index}] changed ({', '.join(fields)})"
                        )
                    else:
                        changes.append(f"{section}[{index}] changed")
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
