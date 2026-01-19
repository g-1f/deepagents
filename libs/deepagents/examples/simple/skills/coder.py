"""Coding skill plugin.

Drop this file into any skills directory to add coding capabilities.
"""


def register(registry):
    """Register the coder skill."""
    registry.register({
        "name": "coder",
        "description": "Write, review, and explain code in any programming language",
        "system_prompt": """You are an expert programmer. You can:

1. Write clean, well-documented code
2. Explain code step by step
3. Review code for bugs and improvements
4. Suggest best practices

Always:
- Use clear variable names
- Add helpful comments
- Follow language conventions
- Consider edge cases""",
        "tools": [],
        "max_iterations": 15,
    })
