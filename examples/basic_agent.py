"""
Basic agent example — tool-calling with OpenAI.

Prerequisites:
    pip install daita-agents
    export OPENAI_API_KEY=sk-...

Run:
    python examples/basic_agent.py
"""
import asyncio
from daita import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real agent you'd call a weather API here
    return f"Sunny, 22°C in {city}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def main():
    agent = Agent(
        name="assistant",
        llm_provider="openai",
        model="gpt-4o-mini",
        tools=[get_weather, calculate],
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
    )

    result = await agent.run("What's the weather in Tokyo, and what is 42 * 7?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
