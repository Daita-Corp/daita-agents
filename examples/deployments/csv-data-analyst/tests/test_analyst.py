"""
Tests for the CSV Analyst agent.

Covers tool functions directly (fast, no LLM) and agent instantiation.
LLM integration tests are marked requires_llm and skipped by default.
"""

import os
import sys
import pytest

# Add project root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_sales.csv")


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

class TestAgentCreation:
    def test_agent_can_be_created(self):
        from agents.csv_analyst import create_agent
        agent = create_agent(SAMPLE_CSV)
        assert agent is not None
        assert agent.name == "CSV Analyst"

    def test_agent_has_all_tools(self):
        from agents.csv_analyst import create_agent
        agent = create_agent(SAMPLE_CSV)
        tool_names = {t.name for t in agent.tools}
        assert "load_csv" in tool_names
        assert "get_summary_stats" in tool_names
        assert "aggregate" in tool_names
        assert "top_n" in tool_names
        assert "count_values" in tool_names
        assert "filter_and_summarise" in tool_names


# ---------------------------------------------------------------------------
# Tool unit tests (no LLM required)
# ---------------------------------------------------------------------------

class TestLoadCsv:
    @pytest.mark.asyncio
    async def test_loads_sample_file(self):
        from agents.csv_analyst import load_csv
        result = await load_csv(SAMPLE_CSV)
        assert "error" not in result
        assert result["shape"]["rows"] > 0
        assert "product" in result["columns"]
        assert "revenue" in result["columns"]
        assert "region" in result["columns"]

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self):
        from agents.csv_analyst import load_csv
        result = await load_csv("data/does_not_exist.csv")
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_preview_has_five_rows(self):
        from agents.csv_analyst import load_csv
        result = await load_csv(SAMPLE_CSV)
        assert len(result["preview"]) == 5


class TestSummaryStats:
    @pytest.mark.asyncio
    async def test_all_numeric_columns(self):
        from agents.csv_analyst import get_summary_stats
        result = await get_summary_stats(SAMPLE_CSV)
        assert "error" not in result
        assert "statistics" in result
        assert "revenue" in result["statistics"]

    @pytest.mark.asyncio
    async def test_specific_columns(self):
        from agents.csv_analyst import get_summary_stats
        result = await get_summary_stats(SAMPLE_CSV, columns=["revenue", "units_sold"])
        assert "error" not in result
        assert "revenue" in result["statistics"]
        assert "units_sold" in result["statistics"]


class TestAggregate:
    @pytest.mark.asyncio
    async def test_sum_revenue_by_region(self):
        from agents.csv_analyst import aggregate
        result = await aggregate(SAMPLE_CSV, "region", "revenue", "sum")
        assert "error" not in result
        assert len(result["results"]) == 4  # North, South, East, West
        # All regions should have revenue > 0
        for row in result["results"]:
            assert row["sum_revenue"] > 0

    @pytest.mark.asyncio
    async def test_mean_units_by_category(self):
        from agents.csv_analyst import aggregate
        result = await aggregate(SAMPLE_CSV, "category", "units_sold", "mean")
        assert "error" not in result
        assert len(result["results"]) == 3  # Electronics, Peripherals, Furniture

    @pytest.mark.asyncio
    async def test_invalid_function_returns_error(self):
        from agents.csv_analyst import aggregate
        result = await aggregate(SAMPLE_CSV, "region", "revenue", "median")
        assert "error" in result


class TestTopN:
    @pytest.mark.asyncio
    async def test_top_5_by_revenue(self):
        from agents.csv_analyst import top_n
        result = await top_n(SAMPLE_CSV, "revenue", n=5)
        assert "error" not in result
        assert len(result["results"]) == 5
        # Results should be sorted descending
        revenues = [r["revenue"] for r in result["results"]]
        assert revenues == sorted(revenues, reverse=True)

    @pytest.mark.asyncio
    async def test_bottom_3_by_units_sold(self):
        from agents.csv_analyst import top_n
        result = await top_n(SAMPLE_CSV, "units_sold", n=3, ascending=True)
        assert "error" not in result
        assert len(result["results"]) == 3
        units = [r["units_sold"] for r in result["results"]]
        assert units == sorted(units)

    @pytest.mark.asyncio
    async def test_columns_to_show(self):
        from agents.csv_analyst import top_n
        result = await top_n(SAMPLE_CSV, "revenue", n=3, columns_to_show=["product", "revenue"])
        assert "error" not in result
        for row in result["results"]:
            assert set(row.keys()) == {"product", "revenue"}


class TestCountValues:
    @pytest.mark.asyncio
    async def test_count_by_region(self):
        from agents.csv_analyst import count_values
        result = await count_values(SAMPLE_CSV, "region")
        assert "error" not in result
        assert result["total_unique_values"] == 4
        assert len(result["results"]) == 4

    @pytest.mark.asyncio
    async def test_count_by_product(self):
        from agents.csv_analyst import count_values
        result = await count_values(SAMPLE_CSV, "product")
        assert "error" not in result
        # Each row should have a count > 0
        for row in result["results"]:
            assert row["count"] > 0


class TestFilterAndSummarise:
    @pytest.mark.asyncio
    async def test_filter_by_region(self):
        from agents.csv_analyst import filter_and_summarise
        result = await filter_and_summarise(SAMPLE_CSV, "region == 'North'")
        assert "error" not in result
        assert result["matching_rows"] > 0
        for row in result["results"]:
            assert row["region"] == "North"

    @pytest.mark.asyncio
    async def test_filter_with_summary(self):
        from agents.csv_analyst import filter_and_summarise
        result = await filter_and_summarise(
            SAMPLE_CSV,
            "category == 'Electronics'",
            summarise_column="revenue",
        )
        assert "error" not in result
        assert "summary" in result
        assert result["summary"]["sum"] > 0

    @pytest.mark.asyncio
    async def test_compound_filter(self):
        from agents.csv_analyst import filter_and_summarise
        result = await filter_and_summarise(
            SAMPLE_CSV,
            "region == 'West' and units_sold > 10",
        )
        assert "error" not in result
        for row in result["results"]:
            assert row["region"] == "West"
            assert row["units_sold"] > 10


# ---------------------------------------------------------------------------
# LLM integration tests (require OPENAI_API_KEY)
# ---------------------------------------------------------------------------

class TestAgentIntegration:
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    async def test_top_products_question(self):
        from agents.csv_analyst import create_agent
        agent = create_agent(SAMPLE_CSV)
        await agent.start()
        try:
            result = await agent.run(
                f"Load {SAMPLE_CSV} and tell me the top 3 products by total revenue."
            )
            assert isinstance(result, str)
            assert len(result) > 20
            # Should mention Laptop since it has highest unit price
            assert "laptop" in result.lower() or "Laptop" in result
        finally:
            await agent.stop()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    async def test_regional_breakdown_question(self):
        from agents.csv_analyst import create_agent
        agent = create_agent(SAMPLE_CSV)
        await agent.start()
        try:
            result = await agent.run(
                f"Using {SAMPLE_CSV}, what is the total revenue for each region?"
            )
            assert isinstance(result, str)
            # All four regions should be mentioned
            for region in ["North", "South", "East", "West"]:
                assert region in result
        finally:
            await agent.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
