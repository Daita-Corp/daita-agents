"""Small, example-only helpers shared by the retained v2 walkthroughs."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from daita import Agent
from daita.llm import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
)

OFFLINE_PROFILE = ModelProfile(
    id="example:offline",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
)


class ScriptedModel:
    """A deterministic provider used only to keep examples runnable offline."""

    provider_id = OFFLINE_PROFILE.id

    def __init__(self) -> None:
        self._responses: list[ModelResponse] = []
        self.requests: list[ModelRequest] = []

    def extend(self, *responses: ModelResponse) -> None:
        self._responses.extend(responses)

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return request.allow_parallel_tool_calls in {None, False}

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("offline example model exhausted its explicit script")
        return self._responses.pop(0)


def tool_response(
    call_id: str,
    name: str,
    arguments: Mapping[str, object],
) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def final_response(text: str, evidence_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        text=f"{text} [evidence:{evidence_id}]",
    )


def sequential_ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def parser(description: str, *, include_root: bool = True) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    if include_root:
        result.add_argument(
            "--root",
            type=Path,
            help=(
                "Explicit fresh v2 state root. If omitted, the example creates "
                "and removes a fresh temporary root."
            ),
        )
    return result


@contextmanager
def example_root(value: Path | None, label: str) -> Iterator[Path]:
    """Yield an explicit state root; never fall back to the user's home."""

    if value is not None:
        resolved = value.expanduser().resolve(strict=False)
        resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
        yield resolved
        return
    with TemporaryDirectory(prefix=f"daita-v2-{label}-") as directory:
        # macOS exposes /var as a symlink to /private/var; normalize the
        # temporary directory before the agent's strict anti-alias admission.
        root = Path(directory).resolve() / "state"
        root.mkdir(mode=0o700)
        yield root


def seed_sales_database(path: Path) -> Path:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL
            );
            CREATE TABLE orders (
                id TEXT PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                status TEXT NOT NULL,
                amount REAL NOT NULL
            );
            INSERT INTO customers (id, name, region) VALUES
                (1, 'Ada Lovelace', 'North America'),
                (2, 'Grace Hopper', 'North America'),
                (3, 'Katherine Johnson', 'Europe');
            INSERT INTO orders (id, customer_id, status, amount) VALUES
                ('order-1', 1, 'pending', 120.0),
                ('order-2', 2, 'complete', 450.0),
                ('order-3', 3, 'complete', 300.0);
            """)
    return path


async def create_offline_agent(
    name: str,
    root: Path,
    model: ScriptedModel,
) -> Agent:
    return await Agent.create(
        name,
        root=root,
        model=model,
        model_profile=OFFLINE_PROFILE,
        id_factory=sequential_ids(),
    )


def print_snapshot(snapshot: object) -> None:
    operation = getattr(snapshot, "operation")
    tasks: Sequence[object] = getattr(snapshot, "tasks")
    evidence: Sequence[object] = getattr(snapshot, "evidence")
    events: Sequence[object] = getattr(snapshot, "events")
    print(f"operation: {operation.id}")
    print(f"status: {operation.status.value}")
    print("tasks:")
    for task in tasks:
        print(f"  - {task.capability_id}: {task.status.value}")
    print("evidence:")
    for item in evidence:
        print(f"  - {item.id}: {item.kind}")
    print(f"committed events: {len(events)}")
