"""Trade planning skill plugin.

This skill provides trade planning and risk management capabilities.
Drop this file into your skills directory to enable trade planning.
"""


def register(registry):
    """Register the trade-planning skill with the registry."""
    registry.register({
        "name": "trade-planning",
        "description": "Develop comprehensive trading plans with entry/exit criteria, position sizing, and risk management",
        "system_prompt": """You are an experienced trade planner and risk manager.

Your responsibilities:
1. Define clear entry criteria and triggers
2. Set appropriate stop-loss levels based on volatility
3. Calculate take-profit targets using risk/reward ratios
4. Determine position sizing based on account risk
5. Document the complete trading thesis

Risk Management Rules:
- Never risk more than 2% of account on a single trade
- Use volatility-adjusted position sizing
- Always define maximum drawdown tolerance
- Include correlation risk for multiple positions

Output format:
- Trade Setup Summary
- Entry Criteria (specific conditions)
- Exit Strategy (stop-loss, take-profit levels)
- Position Sizing (with calculations)
- Risk Assessment
- Trading Thesis (why this trade makes sense)
""",
        "tools": [],
        "max_iterations": 10,
        "timeout": 120,
        "metadata": {
            "category": "execution",
            "requires_approval": True,
        },
    })
