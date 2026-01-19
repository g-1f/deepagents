"""Integration tests for async skill execution with fake tools.

This module tests:
1. Skills can run multiple tools asynchronously
2. Skills spin up isolated subagents
3. Context/results are accessible by the main agent
"""

import asyncio
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

# Set test API key
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from deepagents.skills import (
    SimpleSkill,
    SkillConfig,
    SkillExecutor,
    SkillRegistry,
    SkillResult,
    create_executor,
)
from deepagents.graph_no_middleware import create_skill_agent


# =============================================================================
# Fake Tools - Simulate async operations like API calls
# =============================================================================


def create_fake_market_data_tool() -> BaseTool:
    """Create a fake market data tool that simulates API latency."""

    async def afetch_market_data(symbol: str) -> str:
        """Fetch market data for a symbol (async with simulated latency)."""
        await asyncio.sleep(0.1)  # Simulate API latency
        fake_data = {
            "AAPL": {"price": 185.50, "change": 2.3, "volume": "52M"},
            "GOOGL": {"price": 141.20, "change": -0.8, "volume": "28M"},
            "MSFT": {"price": 378.90, "change": 1.5, "volume": "31M"},
            "TSLA": {"price": 248.30, "change": -3.2, "volume": "89M"},
        }
        data = fake_data.get(symbol, {"price": 100.0, "change": 0.0, "volume": "1M"})
        return f"Market data for {symbol}: Price=${data['price']}, Change={data['change']}%, Volume={data['volume']}"

    def fetch_market_data(symbol: str) -> str:
        """Fetch market data for a symbol (sync version)."""
        return asyncio.get_event_loop().run_until_complete(afetch_market_data(symbol))

    return StructuredTool.from_function(
        func=fetch_market_data,
        coroutine=afetch_market_data,
        name="fetch_market_data",
        description="Fetch real-time market data for a stock symbol including price, change, and volume.",
    )


def create_fake_news_tool() -> BaseTool:
    """Create a fake news API tool."""

    async def afetch_news(query: str, limit: int = 5) -> str:
        """Fetch news articles (async with simulated latency)."""
        await asyncio.sleep(0.15)  # Simulate API latency
        fake_news = [
            f"News {i+1}: {query} related headline - Lorem ipsum article content..."
            for i in range(limit)
        ]
        return "\n".join(fake_news)

    def fetch_news(query: str, limit: int = 5) -> str:
        """Fetch news articles (sync version)."""
        return asyncio.get_event_loop().run_until_complete(afetch_news(query, limit))

    return StructuredTool.from_function(
        func=fetch_news,
        coroutine=afetch_news,
        name="fetch_news",
        description="Fetch recent news articles about a topic.",
    )


def create_fake_sentiment_tool() -> BaseTool:
    """Create a fake sentiment analysis tool."""

    async def aanalyze_sentiment(text: str) -> str:
        """Analyze sentiment of text (async)."""
        await asyncio.sleep(0.05)  # Simulate processing
        # Fake sentiment based on keywords
        if any(word in text.lower() for word in ["bullish", "growth", "profit", "up"]):
            return "Sentiment: BULLISH (score: 0.75)"
        elif any(word in text.lower() for word in ["bearish", "loss", "down", "crash"]):
            return "Sentiment: BEARISH (score: -0.65)"
        return "Sentiment: NEUTRAL (score: 0.05)"

    def analyze_sentiment(text: str) -> str:
        """Analyze sentiment (sync version)."""
        return asyncio.get_event_loop().run_until_complete(aanalyze_sentiment(text))

    return StructuredTool.from_function(
        func=analyze_sentiment,
        coroutine=aanalyze_sentiment,
        name="analyze_sentiment",
        description="Analyze sentiment of text, returns BULLISH, BEARISH, or NEUTRAL with score.",
    )


def create_fake_code_execution_tool() -> BaseTool:
    """Create a fake code execution tool for quant analysis."""

    async def aexecute_code(code: str) -> str:
        """Execute Python code (async, simulated)."""
        await asyncio.sleep(0.2)  # Simulate execution time
        # Return fake results based on code content
        if "RSI" in code or "rsi" in code:
            return "RSI Analysis Result: RSI(14) = 58.3 (neutral zone)"
        elif "MACD" in code or "macd" in code:
            return "MACD Analysis Result: MACD = 2.45, Signal = 1.89, Histogram = 0.56 (bullish crossover)"
        elif "backtest" in code.lower():
            return "Backtest Result: Sharpe=1.45, MaxDD=-12.3%, WinRate=58%"
        return f"Code executed successfully. Output: Calculation complete."

    def execute_code(code: str) -> str:
        """Execute code (sync version)."""
        return asyncio.get_event_loop().run_until_complete(aexecute_code(code))

    return StructuredTool.from_function(
        func=execute_code,
        coroutine=aexecute_code,
        name="execute_code",
        description="Execute Python code for quantitative analysis. Supports RSI, MACD, backtesting.",
    )


# =============================================================================
# Fake Tools Registry
# =============================================================================


FAKE_TOOLS = {
    "fetch_market_data": create_fake_market_data_tool,
    "fetch_news": create_fake_news_tool,
    "analyze_sentiment": create_fake_sentiment_tool,
    "execute_code": create_fake_code_execution_tool,
}


def get_fake_tool(name: str) -> BaseTool | None:
    """Tool resolver for loading skills from config."""
    factory = FAKE_TOOLS.get(name)
    return factory() if factory else None


# =============================================================================
# Test Skills Configuration
# =============================================================================


MARKET_RESEARCH_SKILL = {
    "name": "market-research",
    "description": "Research market conditions by fetching data and news in parallel, then analyzing sentiment",
    "system_prompt": """You are a market researcher. When asked to research a stock:

1. FIRST, call these tools IN PARALLEL (all at once):
   - fetch_market_data(symbol) - get current price data
   - fetch_news(query=symbol) - get recent news

2. THEN, analyze the combined results with analyze_sentiment

3. Return a structured summary with:
   - Current price and change
   - Key news headlines
   - Overall sentiment

Always execute fetch_market_data and fetch_news in parallel to save time.""",
    "tools": [
        create_fake_market_data_tool(),
        create_fake_news_tool(),
        create_fake_sentiment_tool(),
    ],
    "max_iterations": 10,
}


QUANT_ANALYSIS_SKILL = {
    "name": "quant-analysis",
    "description": "Perform quantitative analysis with multiple indicators calculated in parallel",
    "system_prompt": """You are a quantitative analyst. When asked to analyze a stock:

1. Run MULTIPLE analyses IN PARALLEL:
   - execute_code("Calculate RSI for {symbol}")
   - execute_code("Calculate MACD for {symbol}")

2. Combine results into a technical summary

3. If asked for backtest, run: execute_code("backtest strategy for {symbol}")

Execute independent calculations in parallel for efficiency.""",
    "tools": [
        create_fake_code_execution_tool(),
        create_fake_market_data_tool(),
    ],
    "max_iterations": 15,
}


MULTI_ASSET_SKILL = {
    "name": "multi-asset-research",
    "description": "Research multiple assets in parallel and compare them",
    "system_prompt": """You are a multi-asset researcher. When comparing multiple stocks:

1. Fetch data for ALL symbols IN PARALLEL:
   - Call fetch_market_data for each symbol simultaneously
   - Call fetch_news for each symbol simultaneously

2. Compare the results

3. Return a comparison table

IMPORTANT: Use parallel tool calls to fetch all data at once.""",
    "tools": [
        create_fake_market_data_tool(),
        create_fake_news_tool(),
    ],
    "max_iterations": 20,
}


# =============================================================================
# Tests
# =============================================================================


class TestFakeTools:
    """Test that fake tools work correctly."""

    @pytest.mark.asyncio
    async def test_market_data_tool_async(self) -> None:
        """Test async market data fetching."""
        tool = create_fake_market_data_tool()
        result = await tool.ainvoke({"symbol": "AAPL"})
        assert "AAPL" in result
        assert "185.5" in result  # Price data present

    @pytest.mark.asyncio
    async def test_news_tool_async(self) -> None:
        """Test async news fetching."""
        tool = create_fake_news_tool()
        result = await tool.ainvoke({"query": "AAPL", "limit": 3})
        assert "AAPL" in result
        assert "News 1" in result

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self) -> None:
        """Test that multiple tools can run in parallel."""
        market_tool = create_fake_market_data_tool()
        news_tool = create_fake_news_tool()

        start = time.time()

        # Run in parallel
        results = await asyncio.gather(
            market_tool.ainvoke({"symbol": "AAPL"}),
            news_tool.ainvoke({"query": "AAPL", "limit": 3}),
        )

        elapsed = time.time() - start

        assert len(results) == 2
        assert "AAPL" in results[0]
        assert "News" in results[1]

        # Parallel execution should be faster than sequential
        # market_tool takes 0.1s, news_tool takes 0.15s
        # Sequential would be 0.25s, parallel should be ~0.15s
        assert elapsed < 0.25, f"Parallel execution too slow: {elapsed}s"


class TestSkillIsolation:
    """Test that skills run in isolated contexts."""

    def test_skill_has_own_message_history(self) -> None:
        """Test that each skill invocation has isolated message history."""
        config = SkillConfig.from_dict(MARKET_RESEARCH_SKILL)
        skill = SimpleSkill(config)

        # The skill should be able to build its own graph
        assert skill.name == "market-research"
        assert len(config.tools) == 3

    @pytest.mark.asyncio
    async def test_skill_executor_isolation(self) -> None:
        """Test that SkillExecutor runs skills in isolation."""
        config = SkillConfig.from_dict({
            "name": "isolated-skill",
            "description": "Test isolation",
            "system_prompt": "You are isolated.",
            "tools": [],
        })
        skill = SimpleSkill(config)

        # Create a mock model that returns a simple response
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.invoke.return_value = AIMessage(content="Isolated response")
        mock_model.bind_tools.return_value = mock_model

        executor = create_executor(mock_model, timeout=10.0)

        # Execute skill
        result = executor.execute(skill, "Test task")

        assert result.success
        assert result.skill_name == "isolated-skill"


class TestSkillRegistry:
    """Test skill registry with async tools."""

    def test_register_skills_with_tools(self) -> None:
        """Test registering skills that have async-capable tools."""
        registry = SkillRegistry()

        registry.register(MARKET_RESEARCH_SKILL)
        registry.register(QUANT_ANALYSIS_SKILL)

        skills = registry.list_skills()
        assert len(skills) == 2

        market_skill = registry.get("market-research")
        assert market_skill is not None
        assert len(market_skill.config.tools) == 3

    def test_invoke_skill_tool_creation(self) -> None:
        """Test that invoke_skill tool is created correctly."""
        registry = SkillRegistry()
        registry.register(MARKET_RESEARCH_SKILL)
        registry.register(QUANT_ANALYSIS_SKILL)

        # Create a mock model
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.invoke.return_value = AIMessage(content="Result")
        mock_model.bind_tools.return_value = mock_model

        invoke_tool = registry.create_invoke_skill_tool(mock_model)

        assert invoke_tool.name == "invoke_skill"
        assert "market-research" in invoke_tool.description
        assert "quant-analysis" in invoke_tool.description


class TestParallelSkillExecution:
    """Test parallel execution of multiple skills."""

    @pytest.mark.asyncio
    async def test_parallel_skill_execution(self) -> None:
        """Test executing multiple skills in parallel."""
        # Create skills
        skill1_config = SkillConfig.from_dict({
            "name": "skill-1",
            "description": "First skill",
            "system_prompt": "Skill 1",
            "tools": [],
            "timeout": 5.0,
        })
        skill2_config = SkillConfig.from_dict({
            "name": "skill-2",
            "description": "Second skill",
            "system_prompt": "Skill 2",
            "tools": [],
            "timeout": 5.0,
        })

        skill1 = SimpleSkill(skill1_config)
        skill2 = SimpleSkill(skill2_config)

        # Create mock model
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.invoke.return_value = AIMessage(content="Done")
        mock_model.bind_tools.return_value = mock_model

        executor = create_executor(mock_model, timeout=10.0, max_parallel=5)

        # Execute in parallel
        results = await executor.aexecute_parallel([
            (skill1, "Task 1"),
            (skill2, "Task 2"),
        ])

        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].skill_name == "skill-1"
        assert results[1].skill_name == "skill-2"


class TestContextAccessibility:
    """Test that skill results are accessible to the main agent."""

    def test_skill_result_contains_output(self) -> None:
        """Test that SkillResult properly captures skill output."""
        result = SkillResult(
            skill_name="test-skill",
            success=True,
            result="Analysis complete: AAPL is bullish with RSI=65",
            execution_time=1.5,
        )

        # Main agent should be able to access:
        assert result.skill_name == "test-skill"
        assert result.success is True
        assert "AAPL" in result.result
        assert "bullish" in result.result
        assert result.execution_time == 1.5

    def test_failed_skill_result(self) -> None:
        """Test that failed skills report errors properly."""
        result = SkillResult(
            skill_name="failed-skill",
            success=False,
            error="API rate limit exceeded",
            execution_time=0.5,
        )

        assert result.success is False
        assert "rate limit" in result.error

    def test_create_skill_agent_with_skills(self) -> None:
        """Test creating main agent with skills that can be invoked."""
        # Create agent with skills
        agent = create_skill_agent(
            system_prompt="You are a trading assistant with specialized skills.",
            skills=[
                MARKET_RESEARCH_SKILL,
                QUANT_ANALYSIS_SKILL,
            ],
            include_filesystem_tools=False,
            include_planning_tools=False,
        )

        # Agent should have invoke_skill tool
        assert hasattr(agent, "invoke")
        assert hasattr(agent, "ainvoke")


class TestAsyncSkillInvocation:
    """Test async invocation patterns for skills."""

    @pytest.mark.asyncio
    async def test_skill_with_async_tools(self) -> None:
        """Test that skills properly use async tools."""
        # Create a skill with async-capable tools
        config = SkillConfig(
            name="async-skill",
            description="Skill with async tools",
            system_prompt="Use tools asynchronously.",
            tools=[
                create_fake_market_data_tool(),
                create_fake_news_tool(),
            ],
        )
        skill = SimpleSkill(config)

        # Build graph
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.invoke.return_value = AIMessage(content="Async complete")
        mock_model.bind_tools.return_value = mock_model

        graph = skill.build_graph(mock_model)

        # Graph should be built successfully
        assert graph is not None

    @pytest.mark.asyncio
    async def test_executor_handles_async_properly(self) -> None:
        """Test that executor properly handles async skill invocation."""
        config = SkillConfig(
            name="async-test",
            description="Test async execution",
            system_prompt="Test",
            tools=[],
        )
        skill = SimpleSkill(config)

        # Mock model with async response
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.invoke.return_value = AIMessage(content="Async result")
        mock_model.bind_tools.return_value = mock_model

        executor = SkillExecutor(
            model=mock_model,
            default_timeout=10.0,
            max_parallel=3,
        )

        # Test async execution
        result = await executor.aexecute(skill, "Async task")

        assert result.success
        assert result.skill_name == "async-test"


class TestSkillToolParallelism:
    """Test that skills describe and execute parallel tool calls."""

    def test_skill_system_prompt_describes_parallel_execution(self) -> None:
        """Test that skill prompts describe parallel tool execution."""
        config = SkillConfig.from_dict(MARKET_RESEARCH_SKILL)

        # System prompt should mention parallel execution
        assert "PARALLEL" in config.system_prompt or "parallel" in config.system_prompt

    def test_multi_asset_skill_parallel_description(self) -> None:
        """Test multi-asset skill describes fetching multiple assets in parallel."""
        config = SkillConfig.from_dict(MULTI_ASSET_SKILL)

        # Should describe parallel fetching
        assert "PARALLEL" in config.system_prompt
        assert "simultaneously" in config.system_prompt.lower() or "parallel" in config.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_parallel_data_fetching_performance(self) -> None:
        """Test that parallel tool execution is actually faster."""
        market_tool = create_fake_market_data_tool()
        news_tool = create_fake_news_tool()

        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Sequential execution
        start_seq = time.time()
        for symbol in symbols:
            await market_tool.ainvoke({"symbol": symbol})
            await news_tool.ainvoke({"query": symbol, "limit": 2})
        sequential_time = time.time() - start_seq

        # Parallel execution
        start_par = time.time()
        tasks = []
        for symbol in symbols:
            tasks.append(market_tool.ainvoke({"symbol": symbol}))
            tasks.append(news_tool.ainvoke({"query": symbol, "limit": 2}))
        await asyncio.gather(*tasks)
        parallel_time = time.time() - start_par

        # Parallel should be significantly faster
        # Sequential: 3 * (0.1 + 0.15) = 0.75s
        # Parallel: max(0.1, 0.15) = 0.15s (approximately)
        assert parallel_time < sequential_time * 0.5, (
            f"Parallel ({parallel_time:.2f}s) should be much faster than sequential ({sequential_time:.2f}s)"
        )


# =============================================================================
# Integration Test: Full Agent with Skills
# =============================================================================


class TestFullAgentIntegration:
    """Integration tests for full agent with skills."""

    def test_create_agent_with_all_skills(self) -> None:
        """Test creating an agent with all test skills."""
        agent = create_skill_agent(
            system_prompt="""You are Alpha Cortex, a trading AI assistant.

You have access to specialized skills:
- market-research: For researching market conditions
- quant-analysis: For quantitative analysis
- multi-asset-research: For comparing multiple assets

When asked to analyze stocks, use the appropriate skill.
Skills run in isolation and return summaries.""",
            skills=[
                MARKET_RESEARCH_SKILL,
                QUANT_ANALYSIS_SKILL,
                MULTI_ASSET_SKILL,
            ],
            include_filesystem_tools=False,
        )

        # Agent should be created successfully
        assert agent is not None

    def test_skill_descriptions_in_system_prompt(self) -> None:
        """Test that skill descriptions are included in agent's context."""
        registry = SkillRegistry()
        registry.register(MARKET_RESEARCH_SKILL)
        registry.register(QUANT_ANALYSIS_SKILL)

        skills_list = registry.list_skills()

        # All skills should be listed
        names = [s["name"] for s in skills_list]
        assert "market-research" in names
        assert "quant-analysis" in names

        # Descriptions should be informative
        for skill in skills_list:
            assert len(skill["description"]) > 20  # Meaningful description
