#!/usr/bin/env python3
"""Simple example with custom skills.

Run with:
    python run.py
    python run.py "Explain how async/await works in Python"

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - pip install langchain-anthropic langgraph
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(__file__).rsplit("/examples", 1)[0])

from deepagents import create_skill_agent


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='sk-...'")
        sys.exit(1)

    skills_dir = Path(__file__).parent / "skills"

    # Create agent with skills from directory
    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant with research and coding skills.",
        skills_directory=skills_dir,
    )

    # Get query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What skills do you have available?"

    print(f"Query: {query}\n")

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": query}]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
