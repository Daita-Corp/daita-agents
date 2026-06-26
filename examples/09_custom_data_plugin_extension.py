"""Custom data plugin extension declared through runtime contracts.

Run:
    python examples/09_custom_data_plugin_extension.py
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from daita.agents.agent import Agent
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    RiskLevel,
    Task,
    ToolView,
)

from local_sqlite_fixtures import temporary_sales_sqlite

CAPABILITY_ID = "example_dataset.summarize"


@dataclass(frozen=True)
class DatasetSummaryExecutor:
    id: str = "example_dataset.summarize_executor"
    capability_ids: frozenset[str] = frozenset({CAPABILITY_ID})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        payload = {
            "dataset": str(task.input.get("dataset") or "sales"),
            "tables": ["customers", "orders", "order_items", "products"],
            "grain": "one row per order item for revenue analysis",
            "owner_hint": context.get("capability_owner"),
        }
        return [
            Evidence(
                kind="example_dataset.summary",
                owner="example_dataset",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
                schema_version="1.0.0",
            )
        ]


class ExampleDatasetPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="example_dataset",
        display_name="Example Dataset",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
        provides=frozenset({"dataset_summary"}),
    )

    def declare_capabilities(self) -> tuple[Capability, ...]:
        return (
            Capability(
                id=CAPABILITY_ID,
                owner="example_dataset",
                description="Return a typed business summary for the example dataset.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"dataset.summarize"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={
                    "type": "object",
                    "properties": {"dataset": {"type": "string"}},
                    "additionalProperties": False,
                },
                output_evidence=frozenset({"example_dataset.summary"}),
                executor="example_dataset.summarize_executor",
                model_visible=True,
                side_effecting=False,
            ),
        )

    def get_executors(self) -> tuple[DatasetSummaryExecutor, ...]:
        return (DatasetSummaryExecutor(),)

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        return (
            EvidenceSchema(
                kind="example_dataset.summary",
                owner="example_dataset",
                json_schema={
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "tables": {"type": "array", "items": {"type": "string"}},
                        "grain": {"type": "string"},
                    },
                    "required": ["dataset", "tables", "grain"],
                    "additionalProperties": True,
                },
                description="Typed summary for the example dataset.",
            ),
        )

    def get_tool_views(self) -> tuple[ToolView, ...]:
        return (
            ToolView(
                name="example_dataset_summary",
                capability_id=CAPABILITY_ID,
                description="Summarize the example dataset contract.",
                parameters={
                    "type": "object",
                    "properties": {"dataset": {"type": "string"}},
                    "additionalProperties": False,
                },
            ),
        )


def task_capability_sequence(snapshot) -> list[str]:
    return [task.capability_id for task in snapshot.tasks]


def evidence_kinds(evidence) -> list[str]:
    return [item.kind for item in evidence]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Initialize Agent.from_db() and print plugin declarations only.",
    )
    args = parser.parse_args()

    async with temporary_sales_sqlite() as db_path:
        plugin = ExampleDatasetPlugin()
        agent = await Agent.from_db(
            str(db_path),
            name="CustomDataPluginExtension",
            plugins=[plugin],
            cache_ttl=0,
            memory=False,
        )
        try:
            inspection = await agent.describe()
            print(f"SQLite fixture: {db_path}")
            print(f"Plugin id: {plugin.manifest.id}")
            print(f"Registered plugins: {', '.join(inspection.plugin_ids)}")
            print(f"Capability id: example_dataset:{CAPABILITY_ID}")
            print()

            if args.setup_only:
                return

            evidence = await agent.runtime.execute_capability(
                CAPABILITY_ID,
                owner="example_dataset",
                operation_type="dataset.summarize",
                input={"dataset": "sales"},
            )
            operation_id = evidence[0].operation_id if evidence else None
            snapshot = (
                await agent.runtime.inspect_operation(operation_id)
                if operation_id is not None
                else None
            )

            print("Capability operation")
            print(f"  id: {operation_id}")
            if snapshot is not None:
                print(f"  status: {snapshot.operation.status.value}")
                print("  task capability sequence:")
                for capability_id in task_capability_sequence(snapshot):
                    print(f"    - {capability_id}")
            print("  evidence kinds:")
            for kind in evidence_kinds(evidence):
                print(f"    - {kind}")
            print()

            payload = evidence[0].payload if evidence else {}
            print("Typed evidence payload")
            print(f"  dataset: {payload.get('dataset')}")
            print(f"  tables: {', '.join(payload.get('tables') or [])}")
            print(f"  grain: {payload.get('grain')}")
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
