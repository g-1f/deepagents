#!/usr/bin/env python3
"""Plugin example: Load skills dynamically from directory.

This demonstrates progressive disclosure - skills are discovered
automatically from .py files in the skills/ directory.

Run with:
    python run_with_plugins.py

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - pip install langchain-anthropic langgraph
"""

import asyncio
import os
from pathlib import Path

# Ensure the package is importable
import sys
sys.path.insert(0, str(__file__).rsplit("/examples", 1)[0])

from deepagents import create_skill_agent, SkillRegistry


def list_available_skills(skills_dir: Path) -> list[str]:
    """List skills that will be loaded from directory."""
    registry = SkillRegistry()
    registry.load_from_directory(skills_dir)
    return [s["name"] for s in registry.list_skills()]


async def main():
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Export it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Skills directory (relative to this script)
    skills_dir = Path(__file__).parent / "skills"

    # Show what skills will be loaded
    print("Discovering skills from:", skills_dir)
    available_skills = list_available_skills(skills_dir)
    print(f"Found {len(available_skills)} skills: {', '.join(available_skills)}\n")

    # Create agent with plugin-based skills
    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
        system_prompt="""You are Alpha Cortex, an advanced AI trading assistant.
Your role is to help with market analysis, research, and trade planning.
Use your specialized skills when the task matches their expertise.""",
        skills_directory=skills_dir,
    )

    print("=" * 50)
    print("Alpha Cortex Trading Agent Ready!")
    print("=" * 50)
    print("\nTo add new skills, create a .py file in the skills/ directory")
    print("with a register(registry) function.\n")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            print("\nThinking...\n")

            # Run the agent
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": user_input}]
            })

            # Extract and print response
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(f"Alpha Cortex: {last_msg.content}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
