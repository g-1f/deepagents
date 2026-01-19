#!/usr/bin/env python3
"""Basic example: Create an agent with inline skills.

Run with:
    python run_basic.py

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - pip install langchain-anthropic langgraph
"""

import asyncio
import os

# Ensure the package is importable
import sys
sys.path.insert(0, str(__file__).rsplit("/examples", 1)[0])

from deepagents import create_skill_agent


async def main():
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Export it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Create agent with inline skills
    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
        system_prompt="You are a helpful trading assistant. Use your skills when appropriate.",
        skills=[
            {
                "name": "market-analysis",
                "description": "Analyze market conditions and provide insights",
                "system_prompt": """You are a market analyst. Provide clear, concise analysis of:
- Current market sentiment
- Key price levels
- Potential catalysts
Always be factual and mention when you're uncertain.""",
                "tools": [],
            },
            {
                "name": "risk-assessment",
                "description": "Assess trading risks and suggest position sizing",
                "system_prompt": """You are a risk manager. For any trade idea:
- Identify key risks
- Suggest appropriate position size (never more than 2% account risk)
- Recommend stop-loss levels
Be conservative and prioritize capital preservation.""",
                "tools": [],
            },
        ],
    )

    print("Trading Agent Ready!")
    print("=" * 50)
    print("Available skills: market-analysis, risk-assessment")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            # Run the agent
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": user_input}]
            })

            # Extract and print response
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(f"\nAssistant: {last_msg.content}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
