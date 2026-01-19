#!/usr/bin/env python3
"""Minimal example: Simplest possible agent.

Run with:
    python minimal.py

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - pip install langchain-anthropic langgraph
"""

import asyncio
import os
import sys

sys.path.insert(0, str(__file__).rsplit("/examples", 1)[0])

from deepagents import create_skill_agent


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first")
        return

    # Minimal agent - no skills, just the base agent
    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Hello! What can you do?"}]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
