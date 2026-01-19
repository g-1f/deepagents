#!/usr/bin/env python3
"""One-shot example: Run a single query and exit.

Good for testing that everything works.

Run with:
    python run_oneshot.py
    python run_oneshot.py "What's your analysis of the current market?"

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - pip install langchain-anthropic langgraph
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(__file__).rsplit("/examples", 1)[0])

from deepagents import create_skill_agent


async def main():
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Give me a brief market analysis framework I should use for evaluating stocks."

    print(f"Query: {query}\n")
    print("-" * 50)

    # Skills directory
    skills_dir = Path(__file__).parent / "skills"

    # Create agent
    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
        system_prompt="You are a helpful trading assistant.",
        skills_directory=skills_dir,
    )

    # Run query
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": query}]
    })

    # Print response
    messages = result.get("messages", [])
    if messages:
        print(f"\n{messages[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
