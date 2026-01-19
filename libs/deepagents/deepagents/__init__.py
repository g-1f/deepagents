"""DeepAgents package.

This package provides two agent architectures:

1. **Middleware-based (LangChain 1.0+)**: Uses `create_deep_agent` with middleware
   for composable agent capabilities. Requires `langchain>=1.0.0`.

2. **Middleware-free (Skills-based)**: Uses `create_skill_agent` for a simpler
   architecture with config-driven skills. Works with `langchain>=0.2.0`.

Example (Middleware-free):
    ```python
    from deepagents import create_skill_agent

    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
        system_prompt="You are Alpha Cortex, a trading AI.",
        skills=[
            {
                "name": "market-research",
                "description": "Research market conditions",
                "system_prompt": "You are a market researcher...",
                "tools": [web_search, news_api]
            }
        ]
    )
    ```
"""

__all__: list[str] = []

# Middleware-based architecture (LangChain 1.0+)
# These imports are optional - they require langchain>=1.0.0 with middleware support
try:
    from deepagents.graph import create_deep_agent
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.memory import MemoryMiddleware
    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

    __all__.extend([
        "CompiledSubAgent",
        "FilesystemMiddleware",
        "MemoryMiddleware",
        "SubAgent",
        "SubAgentMiddleware",
        "create_deep_agent",
    ])
except ImportError:
    # Middleware-based features not available without langchain 1.0+
    pass

# Middleware-free skills architecture (LangChain 0.2+ / langgraph 0.1+)
# This is the primary architecture for backwards compatibility
from deepagents.graph_no_middleware import create_skill_agent, create_trading_agent
from deepagents.skills import (
    Skill,
    SkillConfig,
    SkillExecutor,
    SkillRegistry,
    SkillResult,
    SimpleSkill,
    create_invoke_skill_tool,
    load_skills_from_yaml,
)

__all__.extend([
    # Middleware-free (primary)
    "create_skill_agent",
    "create_trading_agent",
    "Skill",
    "SkillConfig",
    "SkillExecutor",
    "SkillRegistry",
    "SkillResult",
    "SimpleSkill",
    "create_invoke_skill_tool",
    "load_skills_from_yaml",
])
