from collections.abc import Awaitable, Callable

import pytest

from daita.plugins.memory.memory_plugin import MemoryPlugin
from daita.plugins.memory.memory_tools import (
    handle_list_by_category,
    handle_list_memories,
    handle_query_facts,
    handle_read_memory,
    handle_recall,
    handle_reinforce,
    handle_remember,
    handle_scratch,
    handle_think,
    handle_traverse_memory,
    handle_update_memory,
)
from daita.plugins.memory.metadata import MemoryMetadata


@pytest.mark.parametrize(
    "invoke",
    [
        lambda plugin: handle_remember(plugin, "remember this"),
        lambda plugin: handle_recall(plugin, "remember"),
        lambda plugin: handle_list_by_category(plugin, "rules"),
        lambda plugin: handle_update_memory(plugin, "old", "new"),
        lambda plugin: handle_read_memory(plugin),
        lambda plugin: handle_list_memories(plugin),
        lambda plugin: handle_query_facts(plugin, entity="orders"),
        lambda plugin: handle_reinforce(plugin, "memory-1", "positive"),
    ],
)
async def test_memory_storage_handlers_require_setup(
    invoke: Callable[[MemoryPlugin], Awaitable[object]],
):
    with pytest.raises(RuntimeError, match="MemoryPlugin must be set up"):
        await invoke(MemoryPlugin())


@pytest.mark.parametrize(
    ("invoke", "message"),
    [
        (lambda plugin: handle_scratch(plugin, "scratch"), "Working memory"),
        (lambda plugin: handle_think(plugin, "scratch"), "Working memory"),
        (lambda plugin: handle_traverse_memory(plugin, "orders"), "Memory graph"),
    ],
)
async def test_optional_memory_feature_handlers_require_enablement(
    invoke: Callable[[MemoryPlugin], Awaitable[object]],
    message: str,
):
    with pytest.raises(RuntimeError, match=message):
        await invoke(MemoryPlugin())


async def test_memory_plugin_operations_require_setup():
    plugin = MemoryPlugin()

    with pytest.raises(RuntimeError, match="before recall"):
        await plugin._execute_db_semantic_recall({})
    with pytest.raises(RuntimeError, match="before updating memory"):
        await plugin.mark_important("orders", 0.8)
    with pytest.raises(RuntimeError, match="before deleting memory"):
        await plugin.forget("orders")


def test_memory_metadata_requires_initialized_created_at():
    metadata = MemoryMetadata(content="orders use customer_id")
    metadata.created_at = None

    with pytest.raises(RuntimeError, match="created_at was not initialized"):
        metadata.should_prune()
