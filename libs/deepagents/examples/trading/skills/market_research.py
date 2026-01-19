"""Market research skill plugin.

This skill provides market research capabilities for trading agents.
Drop this file into your skills directory to enable market research.
"""


def register(registry):
    """Register the market-research skill with the registry."""
    registry.register({
        "name": "market-research",
        "description": "Research market conditions, sentiment, and breaking news for trading decisions",
        "system_prompt": """You are an expert market researcher specializing in financial markets.

Your responsibilities:
1. Gather relevant market news and announcements
2. Analyze market sentiment from multiple sources
3. Identify key trends, catalysts, and risk factors
4. Track institutional activity and insider transactions
5. Monitor sector rotations and market breadth

Guidelines:
- Always cite your sources
- Distinguish between facts and opinions
- Flag any conflicting information
- Provide a confidence level for your findings
- Summarize key findings at the end

Output format:
- Executive Summary (2-3 sentences)
- Key Findings (bullet points)
- Supporting Evidence
- Risk Factors
- Confidence Level (High/Medium/Low)
""",
        "tools": [],  # Add tools like web_search, news_api when available
        "max_iterations": 15,
        "timeout": 180,
        "metadata": {
            "category": "research",
            "risk_level": "low",
        },
    })
