"""
Tests for session-scoped working memory (scratchpad).
"""

from daita.plugins.memory.working_memory import WorkingMemory


class TestWorkingMemoryScratch:
    def test_scratch_auto_key(self):
        wm = WorkingMemory()
        key = wm.scratch("checked table A")
        assert key == "scratch_1"
        assert len(wm) == 1

    def test_scratch_custom_key(self):
        wm = WorkingMemory()
        key = wm.scratch("progress note", key="my_key")
        assert key == "my_key"

    def test_scratch_overwrite_key(self):
        wm = WorkingMemory()
        wm.scratch("first", key="k")
        wm.scratch("second", key="k")
        assert len(wm) == 1
        assert wm.get("k")["content"] == "second"

    def test_get_existing(self):
        wm = WorkingMemory()
        wm.scratch("hello", key="k1")
        item = wm.get("k1")
        assert item is not None
        assert item["content"] == "hello"
        assert item["promoted"] is False

    def test_get_missing(self):
        wm = WorkingMemory()
        assert wm.get("nonexistent") is None


class TestWorkingMemoryThink:
    def test_substring_match(self):
        wm = WorkingMemory()
        wm.scratch("checked table users")
        wm.scratch("checked table orders")
        wm.scratch("unrelated item")

        results = wm.think("table")
        assert len(results) == 2

    def test_keyword_match(self):
        wm = WorkingMemory()
        wm.scratch("PostgreSQL connection pool limit is 100")
        wm.scratch("Redis cache TTL is 3600 seconds")

        results = wm.think("PostgreSQL limit")
        assert len(results) >= 1
        assert "PostgreSQL" in results[0]["content"]

    def test_no_match(self):
        wm = WorkingMemory()
        wm.scratch("hello world")
        results = wm.think("nonexistent query")
        assert results == []

    def test_limit(self):
        wm = WorkingMemory()
        for i in range(10):
            wm.scratch(f"item {i} with common words")
        results = wm.think("common", limit=3)
        assert len(results) == 3

    def test_case_insensitive(self):
        wm = WorkingMemory()
        wm.scratch("PostgreSQL is great")
        results = wm.think("postgresql")
        assert len(results) == 1


class TestWorkingMemoryPromote:
    def test_promote_marks_item(self):
        wm = WorkingMemory()
        wm.scratch("important finding", key="k1")
        result = wm.promote("k1")
        assert result is not None
        assert result["promoted"] is True

        # Verify the item is marked in store
        item = wm.get("k1")
        assert item["promoted"] is True

    def test_promote_missing_key(self):
        wm = WorkingMemory()
        assert wm.promote("nonexistent") is None


class TestWorkingMemoryClear:
    def test_clear_empties_store(self):
        wm = WorkingMemory()
        wm.scratch("a")
        wm.scratch("b")
        wm.scratch("c")
        assert len(wm) == 3

        wm.clear()
        assert len(wm) == 0
        assert wm.dump() == []

    def test_clear_resets_counter(self):
        wm = WorkingMemory()
        wm.scratch("a")
        wm.scratch("b")
        wm.clear()
        key = wm.scratch("c")
        assert key == "scratch_1"


class TestWorkingMemoryDump:
    def test_dump_returns_all(self):
        wm = WorkingMemory()
        wm.scratch("a", key="k1")
        wm.scratch("b", key="k2")
        items = wm.dump()
        assert len(items) == 2
        keys = {i["key"] for i in items}
        assert keys == {"k1", "k2"}
