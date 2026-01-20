"""
Skill Definitions for Deep Agent
================================
Skills define:
- When they should activate (trigger patterns)
- What tools they need
- How to execute (DAG of steps)
"""

from deep_agent import Skill, SkillStep, ExecutionMode, DeepAgent


# ============================================================================
# MARKET ANALYSIS SKILL
# ============================================================================

MARKET_ANALYSIS_SKILL = Skill(
    name="market_analysis",
    description="""Comprehensive market analysis combining news sentiment, 
    technical indicators, and fundamental data. Use for queries about market 
    conditions, stock analysis, or investment research.""",
    trigger_patterns=[
        "analyze", "market", "stock", "sentiment", "technical analysis",
        "fundamental", "research", "outlook", "forecast", "trend",
        "what do you think about", "should I buy", "price target"
    ],
    priority=10,
    system_prompt="""You are a quantitative market analyst. Provide data-driven 
    insights while clearly distinguishing between facts and opinions. Always 
    cite sources and confidence levels.""",
    steps=[
        SkillStep(
            step_id="news_sentiment",
            description="Fetch and analyze recent news for sentiment signals",
            tools=["fetch_news", "analyze_sentiment"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=[],
            prompt_template="""Analyze news sentiment for the target asset.
            Focus on:
            1. Key headlines and their implications
            2. Overall sentiment score (-1 to +1)
            3. Notable events or catalysts"""
        ),
        SkillStep(
            step_id="technical_analysis",
            description="Compute technical indicators and identify patterns",
            tools=["fetch_ohlcv", "compute_indicators"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=[],
            prompt_template="""Perform technical analysis:
            1. Trend direction (MA crossovers, ADX)
            2. Momentum (RSI, MACD)
            3. Volatility (ATR, Bollinger Bands)
            4. Support/resistance levels
            5. Pattern recognition"""
        ),
        SkillStep(
            step_id="fundamentals",
            description="Gather fundamental data and valuation metrics",
            tools=["fetch_fundamentals"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=[],
            prompt_template="""Analyze fundamental data:
            1. Key financial ratios (P/E, P/B, ROE)
            2. Revenue/earnings trends
            3. Competitive positioning
            4. Analyst estimates"""
        ),
        SkillStep(
            step_id="synthesis",
            description="Synthesize all analyses into actionable insights",
            tools=[],  # No tools - pure synthesis
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["news_sentiment", "technical_analysis", "fundamentals"],
            prompt_template="""Synthesize the analyses into a coherent view:
            1. Bull case with evidence
            2. Bear case with evidence
            3. Key risks and catalysts
            4. Suggested action (if appropriate)
            
            Be balanced and highlight uncertainty."""
        )
    ]
)


# ============================================================================
# OPTIONS STRATEGY SKILL
# ============================================================================

OPTIONS_STRATEGY_SKILL = Skill(
    name="options_strategy",
    description="""Design and analyze options strategies including position 
    sizing, Greeks analysis, and risk/reward profiling. Use for options-related 
    queries about strategies, pricing, or hedging.""",
    trigger_patterns=[
        "options", "calls", "puts", "spread", "straddle", "strangle",
        "iron condor", "butterfly", "greeks", "delta", "gamma", "theta",
        "vega", "implied volatility", "IV", "premium", "strike", "expiry",
        "hedge", "collar", "protective put"
    ],
    priority=15,
    system_prompt="""You are a derivatives specialist with expertise in options 
    strategies. Focus on risk management and clear explanation of trade mechanics. 
    Always include position sizing based on Kelly criterion or similar frameworks.""",
    steps=[
        SkillStep(
            step_id="market_context",
            description="Get current market conditions and IV environment",
            tools=["fetch_ohlcv", "fetch_options_chain"],
            mode=ExecutionMode.PARALLEL,
            depends_on=[],
            prompt_template="""Gather market context:
            1. Current price and recent movement
            2. IV rank and IV percentile
            3. Term structure of volatility
            4. Skew analysis"""
        ),
        SkillStep(
            step_id="strategy_design",
            description="Design optimal options strategy based on outlook",
            tools=["options_pricer", "greeks_calculator"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["market_context"],
            prompt_template="""Design the options strategy:
            1. Select appropriate structure based on outlook
            2. Choose strikes and expirations
            3. Calculate Greeks for the position
            4. Model P&L scenarios
            5. Determine max loss and breakevens"""
        ),
        SkillStep(
            step_id="position_sizing",
            description="Calculate optimal position size using Kelly criterion",
            tools=["kelly_calculator"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["strategy_design"],
            prompt_template="""Calculate position sizing:
            1. Estimate probability of profit
            2. Calculate expected value
            3. Apply Kelly criterion (use fractional Kelly for safety)
            4. Consider portfolio-level risk limits
            5. Account for margin requirements"""
        )
    ]
)


# ============================================================================
# PORTFOLIO RISK SKILL
# ============================================================================

PORTFOLIO_RISK_SKILL = Skill(
    name="portfolio_risk",
    description="""Analyze portfolio risk metrics including VaR, correlation 
    analysis, factor exposures, and stress testing. Use for risk management 
    queries or portfolio health checks.""",
    trigger_patterns=[
        "portfolio", "risk", "VaR", "value at risk", "correlation",
        "diversification", "exposure", "beta", "factor", "stress test",
        "drawdown", "sharpe", "sortino", "risk metrics", "risk report"
    ],
    priority=12,
    system_prompt="""You are a portfolio risk manager. Prioritize accurate 
    risk quantification and clear communication of tail risks. Use multiple 
    risk measures to provide a comprehensive view.""",
    steps=[
        SkillStep(
            step_id="data_collection",
            description="Fetch portfolio holdings and historical data",
            tools=["fetch_portfolio", "fetch_ohlcv"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=[],
            prompt_template="""Collect portfolio data:
            1. Current holdings and weights
            2. Historical prices (minimum 1 year)
            3. Benchmark data for comparison"""
        ),
        SkillStep(
            step_id="risk_metrics",
            description="Compute standard risk metrics",
            tools=["risk_calculator"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["data_collection"],
            prompt_template="""Calculate risk metrics:
            1. VaR (95% and 99%, parametric and historical)
            2. CVaR / Expected Shortfall
            3. Volatility (realized and EWMA)
            4. Beta to benchmark
            5. Max drawdown analysis"""
        ),
        SkillStep(
            step_id="correlation_analysis",
            description="Analyze correlations and factor exposures",
            tools=["correlation_analyzer", "factor_model"],
            mode=ExecutionMode.PARALLEL,
            depends_on=["data_collection"],
            prompt_template="""Analyze dependencies:
            1. Correlation matrix of holdings
            2. Identify correlated clusters
            3. Factor exposures (market, size, value, momentum)
            4. Hidden concentration risks"""
        ),
        SkillStep(
            step_id="stress_testing",
            description="Run stress test scenarios",
            tools=["stress_tester"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["risk_metrics", "correlation_analysis"],
            prompt_template="""Run stress tests:
            1. Historical scenarios (2008, 2020, etc.)
            2. Hypothetical scenarios (rate shock, credit event)
            3. Sensitivity analysis
            4. Liquidity stress"""
        )
    ]
)


# ============================================================================
# DATA RETRIEVAL SKILL
# ============================================================================

DATA_RETRIEVAL_SKILL = Skill(
    name="data_retrieval",
    description="""Simple data retrieval and display. Use for straightforward 
    queries asking for specific data points like prices, quotes, or basic 
    financial data without complex analysis.""",
    trigger_patterns=[
        "price of", "quote", "what is", "show me", "get", "fetch",
        "current", "latest", "today's", "closing price", "open",
        "high", "low", "volume"
    ],
    priority=5,  # Lower priority - fallback for simple queries
    system_prompt="""You are a data retrieval agent. Provide accurate, 
    current data with minimal commentary. Include timestamps for all data.""",
    steps=[
        SkillStep(
            step_id="fetch_data",
            description="Retrieve the requested data",
            tools=["fetch_ohlcv", "fetch_quote"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=[],
            prompt_template="""Retrieve and format the requested data:
            1. Get the most current available data
            2. Include relevant context (change, volume, etc.)
            3. Format clearly for display"""
        )
    ]
)


# ============================================================================
# NEWS DIGEST SKILL
# ============================================================================

NEWS_DIGEST_SKILL = Skill(
    name="news_digest",
    description="""Aggregate and summarize relevant news from multiple sources. 
    Use for queries about news, events, or 'what's happening' type questions.""",
    trigger_patterns=[
        "news", "headlines", "what's happening", "recent events",
        "latest developments", "update on", "any news about",
        "what happened", "breaking"
    ],
    priority=8,
    system_prompt="""You are a financial news analyst. Prioritize accuracy 
    and relevance. Distinguish between facts and speculation. Highlight 
    market-moving information.""",
    steps=[
        SkillStep(
            step_id="aggregate_news",
            description="Fetch news from multiple sources",
            tools=["fetch_news", "fetch_sec_filings"],
            mode=ExecutionMode.PARALLEL,
            depends_on=[],
            prompt_template="""Aggregate news:
            1. General market news
            2. Company-specific news (if applicable)
            3. SEC filings and regulatory news
            4. Earnings announcements"""
        ),
        SkillStep(
            step_id="analyze_and_summarize",
            description="Analyze relevance and create summary",
            tools=["analyze_sentiment"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["aggregate_news"],
            prompt_template="""Summarize the news:
            1. Key headlines with implications
            2. Overall market sentiment
            3. Actionable insights
            4. Upcoming events to watch"""
        )
    ]
)


# ============================================================================
# CODE GENERATION SKILL
# ============================================================================

CODE_GENERATION_SKILL = Skill(
    name="code_generation",
    description="""Generate Python code for quantitative finance tasks including 
    data analysis, backtesting, and visualization. Use for coding-related queries.""",
    trigger_patterns=[
        "code", "python", "script", "backtest", "implement",
        "write a function", "create a class", "algorithm",
        "visualization", "plot", "chart", "pandas", "numpy"
    ],
    priority=7,
    system_prompt="""You are a quantitative developer. Write clean, efficient, 
    well-documented Python code. Follow best practices and include error handling. 
    Prefer vectorized operations over loops.""",
    steps=[
        SkillStep(
            step_id="understand_requirements",
            description="Clarify and validate requirements",
            tools=[],  # No tools - analysis only
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=[],
            prompt_template="""Analyze the coding request:
            1. What is the core functionality needed?
            2. What inputs and outputs are expected?
            3. What libraries are appropriate?
            4. Are there edge cases to handle?"""
        ),
        SkillStep(
            step_id="generate_code",
            description="Generate the requested code",
            tools=["code_executor"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["understand_requirements"],
            prompt_template="""Generate the code:
            1. Include docstrings and type hints
            2. Add inline comments for complex logic
            3. Handle common errors
            4. Provide usage examples"""
        ),
        SkillStep(
            step_id="validate_code",
            description="Test and validate the generated code",
            tools=["code_executor"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["generate_code"],
            prompt_template="""Validate the code:
            1. Run with sample inputs
            2. Check edge cases
            3. Verify output correctness
            4. Note any limitations"""
        )
    ]
)


# ============================================================================
# REGISTRATION FUNCTION
# ============================================================================

def register_default_skills(agent: DeepAgent):
    """Register all default skills with the agent."""
    
    skills = [
        MARKET_ANALYSIS_SKILL,
        OPTIONS_STRATEGY_SKILL,
        PORTFOLIO_RISK_SKILL,
        DATA_RETRIEVAL_SKILL,
        NEWS_DIGEST_SKILL,
        CODE_GENERATION_SKILL,
    ]
    
    for skill in skills:
        agent.register_skill(skill)
    
    print(f"Registered {len(skills)} skills:")
    for skill in skills:
        print(f"  - {skill.name} (priority: {skill.priority})")


# ============================================================================
# SKILL BUILDER HELPERS
# ============================================================================

def create_simple_skill(
    name: str,
    description: str,
    triggers: list[str],
    tools: list[str],
    prompt: str = "",
    priority: int = 5
) -> Skill:
    """Helper to create a simple single-step skill."""
    return Skill(
        name=name,
        description=description,
        trigger_patterns=triggers,
        priority=priority,
        system_prompt=prompt,
        steps=[
            SkillStep(
                step_id="execute",
                description=description,
                tools=tools,
                mode=ExecutionMode.SEQUENTIAL,
                depends_on=[],
                prompt_template=prompt
            )
        ]
    )


def create_sequential_skill(
    name: str,
    description: str,
    triggers: list[str],
    steps: list[tuple[str, str, list[str]]],  # (step_id, description, tools)
    priority: int = 5
) -> Skill:
    """Helper to create a sequential multi-step skill."""
    skill_steps = []
    prev_step_id = None
    
    for step_id, step_desc, tools in steps:
        skill_steps.append(SkillStep(
            step_id=step_id,
            description=step_desc,
            tools=tools,
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=[prev_step_id] if prev_step_id else []
        ))
        prev_step_id = step_id
    
    return Skill(
        name=name,
        description=description,
        trigger_patterns=triggers,
        priority=priority,
        steps=skill_steps
    )
