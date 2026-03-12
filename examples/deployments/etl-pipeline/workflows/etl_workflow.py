"""
ETL Pipeline Workflow

Three-agent linear pipeline:
  Extractor → (raw_data) → Transformer → (transformed_data) → Loader

The Extractor pulls unprocessed events from the source database and relays
them to the Transformer. The Transformer cleans and normalises the records
and relays them to the Loader. The Loader writes the results to the
destination reporting database.
"""

from daita import Workflow


def create_workflow() -> Workflow:
    """Build the ETL pipeline workflow."""
    from agents.extractor import create_agent as create_extractor
    from agents.transformer import create_agent as create_transformer
    from agents.loader import create_agent as create_loader

    workflow = Workflow("ETL Pipeline")

    workflow.add_agent("extractor", create_extractor())
    workflow.add_agent("transformer", create_transformer())
    workflow.add_agent("loader", create_loader())

    # Linear pipeline: extractor → transformer → loader
    workflow.connect("extractor", "raw_data", "transformer")
    workflow.connect("transformer", "transformed_data", "loader")

    return workflow


async def run_workflow(batch_size: int = 1000) -> dict:
    """
    Trigger one ETL run.

    Args:
        batch_size: Number of records to extract per run (default 1000).
    """
    workflow = create_workflow()
    try:
        await workflow.start()
        await workflow.inject_data(
            "extractor",
            f"Extract up to {batch_size} unprocessed events from the source database "
            "and pass them to the Transformer.",
        )
        return {"status": "success", "message": "ETL pipeline triggered"}
    finally:
        await workflow.stop()


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await run_workflow()
        print(f"Result: {result}")

    asyncio.run(main())
