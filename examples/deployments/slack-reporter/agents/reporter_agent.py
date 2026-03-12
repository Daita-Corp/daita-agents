"""
Slack Reporter Agent

Queries a PostgreSQL database for daily sales metrics and posts a formatted
digest to a Slack channel.

Use case: Daily sales report posted to #analytics every morning at 9 AM,
configured via the cron schedule in daita-project.yaml.
"""

import os
from daita import Agent
from daita.plugins import postgresql, slack


# Expected database schema (adjust to your schema):
#
#   orders (id, customer_id, status, total_amount, created_at)
#   order_items (id, order_id, product_id, quantity, unit_price)
#   products (id, name, category)
#   customers (id, name, email)


def create_agent() -> Agent:
    """Create the Slack Reporter agent with PostgreSQL and Slack plugins."""
    db = postgresql(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/mydb")
    )
    slack_plugin = slack(
        token=os.getenv("SLACK_BOT_TOKEN", ""),
        default_channel=os.getenv("SLACK_CHANNEL_ID", "#analytics"),
    )

    return Agent(
        name="Slack Reporter",
        model="gpt-4o-mini",
        prompt="""You are a sales reporting agent. Each time you are invoked, you:

1. QUERY the database for today's key sales metrics using SQL tools:
   - Total revenue today vs yesterday (% change)
   - Number of orders today vs yesterday
   - Top 5 products by revenue today
   - Top 3 customers by order value today
   - Any orders with status = 'failed' or 'refunded'

   Assume this schema:
     orders(id, customer_id, status, total_amount, created_at)
     order_items(id, order_id, product_id, quantity, unit_price)
     products(id, name, category)
     customers(id, name, email)

   Use DATE(created_at) = CURRENT_DATE for today's data.

2. FORMAT a Slack digest using this structure:
   ```
   *Daily Sales Digest — {DATE}*

   *Revenue*
   Today: $X,XXX.XX  |  Yesterday: $X,XXX.XX  |  Change: +X.X%

   *Orders*
   Today: N orders  |  Yesterday: N orders

   *Top Products Today*
   1. Product Name — $X,XXX.XX
   2. ...

   *Top Customers Today*
   1. Customer Name — $X,XXX.XX
   2. ...

   {If failed/refunded orders exist:}
   ⚠️ *Attention: N orders need review*
   ```

3. POST the digest to the default Slack channel using the send_message tool.

Always use real numbers from the queries. If a query fails, note it in the report
and continue with the remaining queries.""",
        plugins=[db, slack_plugin],
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_agent()
        await agent.start()
        try:
            result = await agent.run(
                "Run the daily sales digest for today and post it to Slack."
            )
            print(result)
        finally:
            await agent.stop()

    asyncio.run(main())
