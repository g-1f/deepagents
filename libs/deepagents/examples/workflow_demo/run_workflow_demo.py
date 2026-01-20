#!/usr/bin/env python3
"""Comprehensive demo of WorkflowSkill with subagent orchestration.

This demo showcases:
1. WorkflowSkill with dynamic subagent orchestration
2. Single skill activation (progressive disclosure)
3. Session management with checkpointing
4. Comprehensive logging of intermediate tool calls

Run with:
    python run_workflow_demo.py
    python run_workflow_demo.py "Analyze AAPL stock"
    python run_workflow_demo.py --session my-session "Continue analysis"

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - pip install langchain-anthropic langgraph
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_skill_agent
from deepagents.skills import (
    SkillRegistry,
    WorkflowSkill,
    create_workflow_skill,
)


# =============================================================================
# Logging Configuration
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better readability."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # Format message with color
        formatted = (
            f"{color}[{timestamp}] [{record.levelname:8}] "
            f"[{record.name}]{reset} {record.getMessage()}"
        )

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(debug: bool = False) -> None:
    """Configure logging with colors and appropriate levels."""
    level = logging.DEBUG if debug else logging.INFO

    # Create handler with colored formatter
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]

    # Set specific loggers
    logging.getLogger("deepagents").setLevel(level)
    logging.getLogger("workflow_demo").setLevel(level)

    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger("workflow_demo")


# =============================================================================
# Tool Call Logging Callback
# =============================================================================

class ToolCallLoggingCallback(BaseCallbackHandler):
    """Callback handler that logs all tool calls with details."""

    def __init__(self) -> None:
        self.call_stack: list[dict[str, Any]] = []
        self.depth = 0

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Log when a tool starts execution."""
        tool_name = serialized.get("name", "unknown")
        run_id = kwargs.get("run_id", "unknown")

        indent = "  " * self.depth
        logger.info(f"{indent}┌─ TOOL START: {tool_name}")
        logger.debug(f"{indent}│  Run ID: {run_id}")
        logger.debug(f"{indent}│  Input: {input_str[:200]}...")

        self.call_stack.append({
            "tool": tool_name,
            "start_time": time.time(),
            "depth": self.depth,
        })
        self.depth += 1

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log when a tool completes."""
        self.depth = max(0, self.depth - 1)

        if self.call_stack:
            call_info = self.call_stack.pop()
            duration = time.time() - call_info["start_time"]
            tool_name = call_info["tool"]

            indent = "  " * self.depth
            output_preview = str(output)[:300].replace("\n", " ")
            logger.info(f"{indent}└─ TOOL END: {tool_name} ({duration:.2f}s)")
            logger.debug(f"{indent}   Output: {output_preview}...")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Log when a tool fails."""
        self.depth = max(0, self.depth - 1)

        if self.call_stack:
            call_info = self.call_stack.pop()
            tool_name = call_info["tool"]

            indent = "  " * self.depth
            logger.error(f"{indent}└─ TOOL ERROR: {tool_name}")
            logger.error(f"{indent}   Error: {error}")

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Log when LLM is invoked."""
        model = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        indent = "  " * self.depth
        logger.debug(f"{indent}├─ LLM START: {model}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log when LLM completes."""
        indent = "  " * self.depth
        logger.debug(f"{indent}├─ LLM END")


# =============================================================================
# Mock Tools for Demonstration
# =============================================================================

def create_mock_tools() -> dict[str, StructuredTool]:
    """Create mock tools that simulate real functionality."""

    @tool
    def market_data_fetcher(symbol: str) -> str:
        """Fetch market data for a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., AAPL, GOOGL)

        Returns:
            Market data including price, volume, and basic metrics.
        """
        logger.info(f"[market_data_fetcher] Fetching data for {symbol}")
        time.sleep(0.5)  # Simulate API call

        # Mock data
        data = {
            "AAPL": {"price": 178.50, "change": 2.3, "volume": "52.3M", "pe_ratio": 28.5, "market_cap": "2.8T"},
            "GOOGL": {"price": 141.20, "change": -0.8, "volume": "21.1M", "pe_ratio": 24.2, "market_cap": "1.8T"},
            "MSFT": {"price": 378.90, "change": 1.5, "volume": "18.7M", "pe_ratio": 35.1, "market_cap": "2.9T"},
        }.get(symbol.upper(), {"price": 100.0, "change": 0.0, "volume": "10M", "pe_ratio": 20.0, "market_cap": "100B"})

        return f"""Market Data for {symbol.upper()}:
- Current Price: ${data['price']:.2f}
- Daily Change: {data['change']:+.1f}%
- Volume: {data['volume']}
- P/E Ratio: {data['pe_ratio']}
- Market Cap: ${data['market_cap']}
"""

    @tool
    def technical_analyzer(market_data: str) -> str:
        """Perform technical analysis on market data.

        Args:
            market_data: Raw market data to analyze

        Returns:
            Technical analysis including trends and signals.
        """
        logger.info("[technical_analyzer] Analyzing technical indicators")
        time.sleep(0.7)  # Simulate computation

        return """Technical Analysis:
- Trend: BULLISH (14-day moving average above 50-day)
- RSI: 62 (neutral to slightly overbought)
- MACD: Positive crossover 3 days ago
- Support Level: $172.00
- Resistance Level: $185.00
- Bollinger Bands: Price near upper band (potential pullback)
- Volume Analysis: Above average volume supports bullish trend
- Key Signal: BUY with caution near resistance
"""

    @tool
    def sentiment_analyzer(symbol: str) -> str:
        """Analyze market sentiment from news and social media.

        Args:
            symbol: Stock ticker to analyze sentiment for

        Returns:
            Sentiment analysis summary.
        """
        logger.info(f"[sentiment_analyzer] Analyzing sentiment for {symbol}")
        time.sleep(0.6)  # Simulate API calls

        return f"""Sentiment Analysis for {symbol.upper()}:
- Overall Sentiment: POSITIVE (72% bullish)
- News Sentiment: 8/10 (Recent product announcements well received)
- Social Media: Trending positive, +15% mention increase
- Analyst Ratings: 18 Buy, 5 Hold, 2 Sell
- Insider Activity: Net buying in last 30 days
- Institutional Interest: Increased holdings by 3.2%
- Key Topics: AI integration, Services growth, iPhone demand
"""

    @tool
    def report_generator(analysis_data: str) -> str:
        """Generate a comprehensive investment report.

        Args:
            analysis_data: Combined analysis data to synthesize

        Returns:
            Formatted investment report.
        """
        logger.info("[report_generator] Generating investment report")
        time.sleep(0.5)  # Simulate generation

        return """
═══════════════════════════════════════════════════════════════
                    INVESTMENT ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────
Based on comprehensive technical and sentiment analysis, the stock
shows FAVORABLE conditions for investment with moderate risk.

KEY FINDINGS
─────────────────────────────────────────────────────────────────
✓ Strong bullish trend supported by volume
✓ Positive market sentiment (72% bullish)
✓ Institutional accumulation ongoing
⚠ RSI approaching overbought territory
⚠ Price near resistance level

RECOMMENDATION
─────────────────────────────────────────────────────────────────
Rating: BUY
Entry Zone: $175-178
Target Price: $195 (12-month)
Stop Loss: $168 (-5.5%)
Risk Level: MODERATE

RISK FACTORS
─────────────────────────────────────────────────────────────────
• Market volatility from macroeconomic conditions
• Potential pullback from overbought RSI
• Competition in core product segments

═══════════════════════════════════════════════════════════════
              Report generated by AI Analysis System
═══════════════════════════════════════════════════════════════
"""

    @tool
    def web_search(query: str) -> str:
        """Search the web for information.

        Args:
            query: Search query string

        Returns:
            Search results summary.
        """
        logger.info(f"[web_search] Searching for: {query}")
        time.sleep(0.4)

        return f"""Web Search Results for "{query}":

1. [News] Latest developments show strong momentum...
2. [Analysis] Expert opinions indicate positive outlook...
3. [Report] Industry trends support growth trajectory...
4. [Forum] Community discussion highlights key opportunities...

Summary: Multiple sources indicate positive market conditions.
"""

    @tool
    def calculator(expression: str) -> str:
        """Perform mathematical calculations.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Calculation result.
        """
        logger.info(f"[calculator] Evaluating: {expression}")
        try:
            # Safe eval for basic math
            allowed = set("0123456789+-*/().% ")
            if all(c in allowed for c in expression):
                result = eval(expression)
                return f"Result: {result}"
            return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {e}"

    return {
        "market_data_fetcher": market_data_fetcher,
        "technical_analyzer": technical_analyzer,
        "sentiment_analyzer": sentiment_analyzer,
        "report_generator": report_generator,
        "web_search": web_search,
        "calculator": calculator,
    }


# =============================================================================
# Session Manager
# =============================================================================

class SessionManager:
    """Manages conversation sessions with persistence."""

    def __init__(self, storage_dir: str | Path = ".sessions") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.checkpointer = MemorySaver()
        self._session_metadata: dict[str, dict[str, Any]] = {}

    def create_session(self, session_id: str | None = None) -> str:
        """Create a new session."""
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        self._session_metadata[session_id] = {
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
            "last_active": datetime.now().isoformat(),
        }

        logger.info(f"Created session: {session_id}")
        return session_id

    def get_config(self, session_id: str) -> dict[str, Any]:
        """Get LangGraph config for a session."""
        return {
            "configurable": {
                "thread_id": session_id,
            }
        }

    def update_session(self, session_id: str) -> None:
        """Update session metadata."""
        if session_id in self._session_metadata:
            self._session_metadata[session_id]["message_count"] += 1
            self._session_metadata[session_id]["last_active"] = datetime.now().isoformat()

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata."""
        return self._session_metadata.get(session_id)

    def list_sessions(self) -> list[str]:
        """List all active sessions."""
        return list(self._session_metadata.keys())


# =============================================================================
# Demo Agent Factory
# =============================================================================

def create_demo_agent(
    session_manager: SessionManager,
    use_workflow_skills: bool = True,
    use_skill_discovery: bool = True,
    debug: bool = False,
):
    """Create the demo agent with skills and session management.

    Args:
        session_manager: Session manager for checkpointing
        use_workflow_skills: Whether to use workflow-based skills
        use_skill_discovery: Whether to use single skill activation
        debug: Enable debug mode

    Returns:
        Configured agent graph
    """
    # Create mock tools
    tool_registry = create_mock_tools()

    if use_workflow_skills:
        # Define workflow skills
        skills = [
            # Stock analysis workflow skill
            {
                "name": "stock-analyzer",
                "description": "Comprehensive stock analysis with market data, technical analysis, sentiment analysis, and investment recommendations",
                "system_prompt": """You are an expert financial analyst.
Analyze stocks thoroughly using all available tools and provide actionable insights.
Be specific with numbers and cite your data sources.""",
                "workflow": [
                    {
                        "tools": "market_data_fetcher",
                        "task_template": "Fetch market data for the stock mentioned in: {{input}}",
                        "name": "fetch_data",
                    },
                    {
                        "tools": ["technical_analyzer", "sentiment_analyzer"],
                        "task_template": "Analyze this market data:\n{{previous_output}}",
                        "name": "parallel_analysis",
                    },
                    {
                        "tools": "report_generator",
                        "task_template": "Generate investment report based on:\n{{previous_output}}",
                        "name": "generate_report",
                    },
                ],
            },
            # Research workflow skill
            {
                "name": "researcher",
                "description": "Research any topic by searching the web and synthesizing information",
                "system_prompt": "You are a thorough researcher. Search for information and provide comprehensive summaries.",
                "workflow": [
                    {
                        "tools": "web_search",
                        "task_template": "Search for information about: {{input}}",
                        "name": "search",
                    },
                ],
            },
        ]
    else:
        # Simple skills (non-workflow)
        skills = [
            {
                "name": "stock-analyzer",
                "description": "Analyze stocks using available market tools",
                "system_prompt": "You are a financial analyst. Use the available tools to analyze stocks.",
                "tools": list(tool_registry.values())[:4],  # First 4 tools
            },
            {
                "name": "researcher",
                "description": "Research topics using web search",
                "system_prompt": "You are a researcher. Use web search to find information.",
                "tools": [tool_registry["web_search"]],
            },
        ]

    # Create the agent
    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
        system_prompt="""You are a helpful AI assistant with specialized skills.

When users ask about stocks, financial analysis, or investments, delegate to the appropriate skill.
When users ask for research on any topic, use the researcher skill.

For simple questions or calculations, handle them directly without using skills.""",
        skills=skills,
        tool_registry=tool_registry if use_workflow_skills else None,
        tools=[tool_registry["calculator"]],  # Direct tool access
        checkpointer=session_manager.checkpointer,
        use_skill_discovery=use_skill_discovery,
        include_planning_tools=True,
        include_filesystem_tools=False,  # Disable for demo
        debug=debug,
    )

    return agent


# =============================================================================
# Main Demo Runner
# =============================================================================

async def run_demo(
    query: str,
    session_id: str | None = None,
    debug: bool = False,
    interactive: bool = False,
) -> None:
    """Run the demo with the given query."""

    setup_logging(debug)

    logger.info("=" * 60)
    logger.info("WorkflowSkill Demo - Subagent Orchestration")
    logger.info("=" * 60)

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set!")
        logger.info("Set it with: export ANTHROPIC_API_KEY='sk-...'")
        sys.exit(1)

    # Initialize session manager
    session_manager = SessionManager()

    # Create or resume session
    if session_id:
        if session_id not in session_manager.list_sessions():
            session_id = session_manager.create_session(session_id)
        logger.info(f"Using session: {session_id}")
    else:
        session_id = session_manager.create_session()

    # Create callback for logging
    callback = ToolCallLoggingCallback()

    # Create agent
    logger.info("Creating agent with workflow skills...")
    agent = create_demo_agent(
        session_manager,
        use_workflow_skills=True,
        use_skill_discovery=True,
        debug=debug,
    )

    # Get session config
    config = session_manager.get_config(session_id)
    config["callbacks"] = [callback]

    async def process_query(q: str) -> str:
        """Process a single query."""
        logger.info(f"\n{'─' * 60}")
        logger.info(f"USER QUERY: {q}")
        logger.info(f"{'─' * 60}")

        start_time = time.time()

        # Invoke agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=q)]},
            config=config,
        )

        duration = time.time() - start_time
        session_manager.update_session(session_id)

        # Extract response
        messages = result.get("messages", [])
        response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content
                break

        logger.info(f"\n{'─' * 60}")
        logger.info(f"RESPONSE ({duration:.2f}s):")
        logger.info(f"{'─' * 60}")

        return response

    if interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive Mode - Type 'quit' to exit")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                response = await process_query(user_input)
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.exception(f"Error: {e}")
    else:
        # Single query mode
        response = await process_query(query)
        print(f"\n{response}")

    # Print session info
    info = session_manager.get_session_info(session_id)
    if info:
        logger.info(f"\nSession Info: {info}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WorkflowSkill Demo with Subagent Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_workflow_demo.py "Analyze AAPL stock"
  python run_workflow_demo.py --debug "Research AI trends"
  python run_workflow_demo.py --session my-session "Continue our discussion"
  python run_workflow_demo.py --interactive
        """,
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="Analyze AAPL stock and give me an investment recommendation",
        help="Query to process (default: stock analysis example)",
    )
    parser.add_argument(
        "--session", "-s",
        help="Session ID for conversation continuity",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )

    args = parser.parse_args()

    asyncio.run(run_demo(
        query=args.query,
        session_id=args.session,
        debug=args.debug,
        interactive=args.interactive,
    ))


if __name__ == "__main__":
    main()
