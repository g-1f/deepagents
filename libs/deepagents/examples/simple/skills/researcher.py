"""Research skill plugin.

Drop this file into any skills directory to add research capabilities.
"""


def register(registry):
    """Register the researcher skill."""
    registry.register({
        "name": "researcher",
        "description": "Research topics thoroughly and provide comprehensive summaries",
        "system_prompt": """You are an expert researcher. When given a topic:

1. Break down the topic into key areas
2. Provide factual, well-organized information
3. Cite sources or indicate when information is general knowledge
4. Highlight areas of uncertainty or debate
5. Summarize key takeaways

Be thorough but concise. Use bullet points and clear structure.""",
        "tools": [],
        "max_iterations": 10,
    })
