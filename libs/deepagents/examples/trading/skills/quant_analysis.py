"""Quantitative analysis skill plugin.

This skill provides quantitative analysis capabilities for trading agents.
Drop this file into your skills directory to enable quant analysis.
"""


def register(registry):
    """Register the quant-analysis skill with the registry."""
    registry.register({
        "name": "quant-analysis",
        "description": "Perform quantitative analysis including technical indicators, statistical analysis, and backtesting",
        "system_prompt": """You are a quantitative analyst with expertise in financial markets.

Your capabilities:
1. Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)
2. Perform statistical analysis (correlation, regression, volatility)
3. Analyze price patterns and chart formations
4. Run backtests on trading strategies
5. Build and validate quantitative models

Guidelines:
- Show your calculations and methodology
- Include confidence intervals where applicable
- Note any assumptions or limitations
- Provide visual representations when helpful
- Compare results to benchmarks

Output format:
- Analysis Summary
- Key Metrics & Indicators
- Statistical Findings
- Visualizations (described or generated)
- Recommendations with confidence levels
""",
        "tools": [],  # Add tools like execute_code, market_data when available
        "max_iterations": 20,
        "timeout": 300,
        "metadata": {
            "category": "analysis",
            "requires_data": True,
        },
    })
