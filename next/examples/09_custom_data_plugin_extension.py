"""Declare and explicitly load one narrow v2 data capability extension."""

from __future__ import annotations

from collections.abc import Mapping

from daita.extensions import (
    ConfiguredExtension,
    ExtensionKind,
    ExtensionManifest,
    ExtensionRegistration,
    ExtensionRegistry,
    LocalCapability,
    tool,
)

from _shared import parser

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"dataset": {"type": "string"}},
    "required": ["dataset"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "dataset": {"type": "string"},
        "grain": {"type": "string"},
    },
    "required": ["dataset", "grain"],
    "additionalProperties": False,
}


def registration() -> ExtensionRegistration:
    @tool(
        id="example.dataset.summary",
        owner="example.dataset",
        name="summarize_example_dataset",
        description="Return the declared grain for an example dataset.",
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
    )
    async def summarize(arguments: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "dataset": arguments["dataset"],
            "grain": "one row per order",
        }

    assert isinstance(summarize, LocalCapability)
    manifest = ExtensionManifest(
        id="example.dataset",
        version="1.0.0",
        kind=ExtensionKind.CAPABILITY_PROVIDER,
        declarations=summarize.declarations(),
    )
    return ExtensionRegistration(manifest=manifest, executors=(summarize.executor,))


def run() -> None:
    parser(__doc__, include_root=False).parse_args()
    registry = ExtensionRegistry.load(
        (
            ConfiguredExtension(
                id="example.dataset",
                factory=registration,
            ),
        )
    )
    manifest = registry.manifest("example.dataset")
    capability = manifest.declarations.capabilities[0]
    tool_view = manifest.declarations.tool_views[0]
    print(f"loaded extension: {manifest.id}@{manifest.version}")
    print(f"capability: {capability.id}")
    print(f"model-visible tool: {tool_view.name}")
    print("The handler is invoked only through the governed runtime boundary.")


if __name__ == "__main__":
    run()
