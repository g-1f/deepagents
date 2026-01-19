#!/usr/bin/env python3
"""Runnable demo script for the DeepAgents Skills Architecture.

This script demonstrates:
1. Creating fake tools that simulate async API calls
2. Defining skills that use multiple tools in parallel
3. Creating a supervisor agent with skills
4. Invoking skills directly and through the agent
5. Parallel skill execution

Usage:
    # With real API (requires ANTHROPIC_API_KEY):
    python run_skills_demo.py

    # With mock model (no API key needed):
    python run_skills_demo.py --mock

Requirements:
    pip install langchain-core langgraph langchain-anthropic pyyaml
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Any
from unittest.mock import MagicMock

# Add the library to path if running from examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../libs/deepagents"))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool


# =============================================================================
# Fake Tools - Simulate real API calls with latency
# =============================================================================


def create_market_data_tool() -> BaseTool:
    """Fake market data API tool."""

    async def afetch(symbol: str) -> str:
        """Fetch market data (async)."""
        await asyncio.sleep(0.1)  # Simulate API latency
        data = {
            "AAPL": {"price": 185.50, "change": 2.3, "volume": "52M", "pe": 28.5},
            "GOOGL": {"price": 141.20, "change": -0.8, "volume": "28M", "pe": 24.2},
            "MSFT": {"price": 378.90, "change": 1.5, "volume": "31M", "pe": 35.1},
            "TSLA": {"price": 248.30, "change": -3.2, "volume": "89M", "pe": 62.4},
            "NVDA": {"price": 875.40, "change": 4.1, "volume": "45M", "pe": 58.9},
        }.get(symbol.upper(), {"price": 100.0, "change": 0.0, "volume": "1M", "pe": 20.0})

        return f"""Market Data for {symbol.upper()}:
  Price: ${data['price']:.2f}
  Change: {data['change']:+.1f}%
  Volume: {data['volume']}
  P/E Ratio: {data['pe']:.1f}"""

    def fetch(symbol: str) -> str:
        return asyncio.get_event_loop().run_until_complete(afetch(symbol))

    return StructuredTool.from_function(
        func=fetch,
        coroutine=afetch,
        name="fetch_market_data",
        description="Fetch real-time market data for a stock symbol.",
    )


def create_news_tool() -> BaseTool:
    """Fake news API tool."""

    async def afetch(query: str, limit: int = 3) -> str:
        """Fetch news (async)."""
        await asyncio.sleep(0.15)  # Simulate API latency

        fake_headlines = {
            "AAPL": [
                "Apple announces new AI features for iPhone",
                "Apple's services revenue hits record high",
                "Analysts upgrade Apple stock on strong iPhone demand",
            ],
            "TSLA": [
                "Tesla delivers record number of vehicles in Q4",
                "Elon Musk hints at new Tesla product line",
                "Tesla faces increased competition in EV market",
            ],
            "NVDA": [
                "NVIDIA reports explosive growth in AI chip demand",
                "New NVIDIA GPU breaks performance records",
                "NVIDIA partners with major cloud providers",
            ],
        }

        headlines = fake_headlines.get(
            query.upper(),
            [f"{query} sees market activity", f"Investors watch {query} closely", f"Analysts review {query} outlook"],
        )[:limit]

        return f"News for {query}:\n" + "\n".join(f"  - {h}" for h in headlines)

    def fetch(query: str, limit: int = 3) -> str:
        return asyncio.get_event_loop().run_until_complete(afetch(query, limit))

    return StructuredTool.from_function(
        func=fetch,
        coroutine=afetch,
        name="fetch_news",
        description="Fetch recent news articles about a topic or stock.",
    )


def create_sentiment_tool() -> BaseTool:
    """Fake sentiment analysis tool."""

    async def aanalyze(text: str) -> str:
        """Analyze sentiment (async)."""
        await asyncio.sleep(0.05)

        text_lower = text.lower()
        if any(w in text_lower for w in ["record", "growth", "upgrade", "strong", "explosive"]):
            return "Sentiment: BULLISH (score: +0.72)"
        elif any(w in text_lower for w in ["competition", "decline", "downgrade", "weak"]):
            return "Sentiment: BEARISH (score: -0.58)"
        return "Sentiment: NEUTRAL (score: +0.05)"

    def analyze(text: str) -> str:
        return asyncio.get_event_loop().run_until_complete(aanalyze(text))

    return StructuredTool.from_function(
        func=analyze,
        coroutine=aanalyze,
        name="analyze_sentiment",
        description="Analyze sentiment of text. Returns BULLISH, BEARISH, or NEUTRAL.",
    )


def create_technical_analysis_tool() -> BaseTool:
    """Fake technical analysis tool."""

    async def aanalyze(symbol: str, indicator: str = "all") -> str:
        """Run technical analysis (async)."""
        await asyncio.sleep(0.2)  # Simulate computation

        indicators = {
            "AAPL": {"rsi": 58, "macd": "bullish crossover", "sma_trend": "above 50-day"},
            "TSLA": {"rsi": 72, "macd": "bearish divergence", "sma_trend": "below 50-day"},
            "NVDA": {"rsi": 65, "macd": "bullish momentum", "sma_trend": "above 200-day"},
        }.get(symbol.upper(), {"rsi": 50, "macd": "neutral", "sma_trend": "sideways"})

        if indicator == "rsi":
            return f"RSI(14) for {symbol}: {indicators['rsi']} ({'overbought' if indicators['rsi'] > 70 else 'neutral' if indicators['rsi'] > 30 else 'oversold'})"
        elif indicator == "macd":
            return f"MACD for {symbol}: {indicators['macd']}"
        else:
            return f"""Technical Analysis for {symbol.upper()}:
  RSI(14): {indicators['rsi']} ({'overbought' if indicators['rsi'] > 70 else 'neutral'})
  MACD: {indicators['macd']}
  Trend: {indicators['sma_trend']}"""

    def analyze(symbol: str, indicator: str = "all") -> str:
        return asyncio.get_event_loop().run_until_complete(aanalyze(symbol, indicator))

    return StructuredTool.from_function(
        func=analyze,
        coroutine=aanalyze,
        name="technical_analysis",
        description="Run technical analysis (RSI, MACD, SMA) for a stock symbol.",
    )


# =============================================================================
# Skill Definitions
# =============================================================================


def get_market_research_skill() -> dict[str, Any]:
    """Define the market research skill."""
    return {
        "name": "market-research",
        "description": "Research market conditions by fetching price data and news in parallel, then analyzing sentiment",
        "system_prompt": """You are a market researcher. When asked to research a stock:

1. FIRST, call these tools IN PARALLEL (in the same response):
   - fetch_market_data(symbol) - get current price and metrics
   - fetch_news(query=symbol) - get recent news

2. THEN, analyze the news with analyze_sentiment

3. Return a structured summary:
   ## Market Research: {SYMBOL}
   ### Price Data
   [market data results]
   ### News Summary
   [key headlines]
   ### Sentiment
   [sentiment analysis]
   ### Recommendation
   [your assessment]

IMPORTANT: Always call fetch_market_data and fetch_news in parallel to save time.""",
        "tools": [
            create_market_data_tool(),
            create_news_tool(),
            create_sentiment_tool(),
        ],
        "max_iterations": 10,
    }


def get_technical_analysis_skill() -> dict[str, Any]:
    """Define the technical analysis skill."""
    return {
        "name": "technical-analysis",
        "description": "Perform technical analysis with indicators like RSI, MACD, and moving averages",
        "system_prompt": """You are a technical analyst. When asked to analyze a stock:

1. Run technical_analysis(symbol) to get all indicators
2. Interpret the results
3. Provide a technical outlook

Format your response as:
## Technical Analysis: {SYMBOL}
### Indicators
[indicator values]
### Interpretation
[what the indicators suggest]
### Technical Outlook
[bullish/bearish/neutral with reasoning]""",
        "tools": [
            create_technical_analysis_tool(),
            create_market_data_tool(),
        ],
        "max_iterations": 8,
    }


def get_comparison_skill() -> dict[str, Any]:
    """Define the multi-asset comparison skill."""
    return {
        "name": "asset-comparison",
        "description": "Compare multiple stocks by fetching their data in parallel",
        "system_prompt": """You are a comparative analyst. When asked to compare stocks:

1. Fetch data for ALL symbols IN PARALLEL:
   - Call fetch_market_data for each symbol in the SAME response
   - Call fetch_news for each symbol in the SAME response

2. Create a comparison table

3. Provide a ranking with reasoning

IMPORTANT: Fetch all data in parallel for efficiency. Do not call tools one at a time.""",
        "tools": [
            create_market_data_tool(),
            create_news_tool(),
        ],
        "max_iterations": 15,
    }


# =============================================================================
# Demo Functions
# =============================================================================


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n--- {text} ---")


async def demo_parallel_tools() -> None:
    """Demonstrate parallel tool execution."""
    print_header("Demo 1: Parallel Tool Execution")

    market_tool = create_market_data_tool()
    news_tool = create_news_tool()

    symbols = ["AAPL", "NVDA", "TSLA"]

    # Sequential execution
    print_section("Sequential Execution")
    start = time.time()
    for symbol in symbols:
        await market_tool.ainvoke({"symbol": symbol})
        await news_tool.ainvoke({"query": symbol})
    seq_time = time.time() - start
    print(f"  Time: {seq_time:.2f}s")

    # Parallel execution
    print_section("Parallel Execution")
    start = time.time()
    tasks = []
    for symbol in symbols:
        tasks.append(market_tool.ainvoke({"symbol": symbol}))
        tasks.append(news_tool.ainvoke({"query": symbol}))
    results = await asyncio.gather(*tasks)
    par_time = time.time() - start
    print(f"  Time: {par_time:.2f}s")

    print_section("Result")
    print(f"  Speedup: {seq_time/par_time:.1f}x faster with parallel execution")


async def demo_skill_registry() -> None:
    """Demonstrate skill registry and listing."""
    print_header("Demo 2: Skill Registry")

    from deepagents.skills import SkillRegistry

    registry = SkillRegistry()
    registry.register(get_market_research_skill())
    registry.register(get_technical_analysis_skill())
    registry.register(get_comparison_skill())

    print_section("Registered Skills")
    for skill in registry.list_skills():
        print(f"  - {skill['name']}: {skill['description'][:50]}...")

    print_section("Skill Details")
    market_skill = registry.get("market-research")
    if market_skill:
        print(f"  Name: {market_skill.name}")
        print(f"  Tools: {len(market_skill.config.tools)}")
        print(f"  Max Iterations: {market_skill.config.max_iterations}")


async def demo_skill_executor(use_mock: bool = True) -> None:
    """Demonstrate skill executor with parallel skills."""
    print_header("Demo 3: Skill Executor (Parallel Skills)")

    from deepagents.skills import SimpleSkill, SkillConfig, create_executor

    # Create skills
    skill1 = SimpleSkill(SkillConfig.from_dict(get_market_research_skill()))
    skill2 = SimpleSkill(SkillConfig.from_dict(get_technical_analysis_skill()))

    if use_mock:
        # Create mock model
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.invoke.return_value = AIMessage(
            content="Analysis complete: AAPL shows bullish momentum with strong fundamentals."
        )
        mock_model.bind_tools.return_value = mock_model
        model = mock_model
        print("  (Using mock model)")
    else:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model_name="claude-sonnet-4-20250514", max_tokens=4096)
        print("  (Using real API)")

    executor = create_executor(model, timeout=30.0, max_parallel=3)

    print_section("Executing Skills in Parallel")
    start = time.time()
    results = await executor.aexecute_parallel([
        (skill1, "Research AAPL stock"),
        (skill2, "Analyze AAPL technicals"),
    ])
    elapsed = time.time() - start

    print_section("Results")
    for result in results:
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  [{status}] {result.skill_name}: {result.execution_time:.2f}s")
        if result.result:
            preview = result.result[:100] + "..." if len(result.result) > 100 else result.result
            print(f"    Output: {preview}")

    print(f"\n  Total time: {elapsed:.2f}s (ran in parallel)")


async def demo_full_agent(use_mock: bool = True) -> None:
    """Demonstrate full agent with skills."""
    print_header("Demo 4: Full Agent with Skills")

    from deepagents import create_skill_agent

    skills = [
        get_market_research_skill(),
        get_technical_analysis_skill(),
        get_comparison_skill(),
    ]

    if use_mock:
        print("  (Using mock - agent created but not invoked)")

        # Just create the agent to show it works
        agent = create_skill_agent(
            system_prompt="""You are Alpha Cortex, a trading AI assistant.
Use your specialized skills to analyze markets and provide insights.""",
            skills=skills,
            include_filesystem_tools=False,
        )

        print_section("Agent Created Successfully")
        print(f"  Type: {type(agent).__name__}")
        print(f"  Skills: {len(skills)}")
        print("  Available skills:")
        for skill in skills:
            print(f"    - {skill['name']}")

    else:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model_name="claude-sonnet-4-20250514", max_tokens=4096)

        agent = create_skill_agent(
            model=model,
            system_prompt="""You are Alpha Cortex, a trading AI assistant.
Use your specialized skills to analyze markets and provide insights.
When asked about stocks, use the appropriate skill.""",
            skills=skills,
            include_filesystem_tools=False,
        )

        print_section("Invoking Agent")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="Research AAPL stock for me")]
        })

        print_section("Agent Response")
        last_message = result["messages"][-1]
        print(last_message.content)


async def demo_yaml_config() -> None:
    """Demonstrate loading skills from YAML."""
    print_header("Demo 5: Loading Skills from YAML")

    from deepagents.skills import SkillRegistry

    # Use the example config file
    config_path = os.path.join(
        os.path.dirname(__file__),
        "../../libs/deepagents/deepagents/config/skills.yaml"
    )

    if os.path.exists(config_path):
        registry = SkillRegistry()
        registry.load_from_config(config_path)

        print_section(f"Loaded from: {os.path.basename(config_path)}")
        print(f"  Skills loaded: {len(registry.skills)}")
        print_section("Available Skills")
        for skill in registry.list_skills():
            print(f"  - {skill['name']}")
            print(f"    {skill['description'][:60]}...")
    else:
        print(f"  Config file not found: {config_path}")
        print("  Skipping YAML demo...")


async def main() -> None:
    """Run all demos."""
    parser = argparse.ArgumentParser(description="DeepAgents Skills Demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model instead of real API",
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific demo (1-5)",
    )
    args = parser.parse_args()

    use_mock = args.mock or not os.environ.get("ANTHROPIC_API_KEY")

    if use_mock:
        print("\n[Running with mock model - no API calls will be made]")
    else:
        print("\n[Running with real API - ANTHROPIC_API_KEY detected]")

    demos = [
        (1, demo_parallel_tools),
        (2, demo_skill_registry),
        (3, lambda: demo_skill_executor(use_mock)),
        (4, lambda: demo_full_agent(use_mock)),
        (5, demo_yaml_config),
    ]

    if args.demo:
        # Run specific demo
        for num, demo_fn in demos:
            if num == args.demo:
                await demo_fn()
                break
    else:
        # Run all demos
        for _, demo_fn in demos:
            await demo_fn()

    print_header("Demo Complete!")
    print("""
Key Takeaways:
1. Tools can run in parallel for significant speedup
2. Skills are isolated subagents with their own context
3. SkillExecutor can run multiple skills in parallel
4. Main agent can invoke skills via invoke_skill tool
5. Skills can be defined in code or loaded from YAML
""")


if __name__ == "__main__":
    asyncio.run(main())
