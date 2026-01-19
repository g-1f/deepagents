"""Skills module for middleware-free agent architecture.

This module provides a skills-based agent architecture that works with
langchain>=0.2.0 and langgraph>=0.1.0 (pre-middleware versions).

Skills are isolated subagents with custom tools and system prompts that can
be invoked by a supervisor agent via the `invoke_skill` tool.

Example:
    ```python
    from deepagents.skills import SkillRegistry, SkillConfig, create_invoke_skill_tool

    # Create a skill registry
    registry = SkillRegistry()

    # Register skills from dicts
    registry.register({
        "name": "market-research",
        "description": "Research market conditions and sentiment",
        "system_prompt": "You are an expert market researcher...",
        "tools": [web_search_tool, news_api]
    })

    # Or load from YAML config
    registry.load_from_config("config/skills.yaml")

    # Create the invoke_skill tool for the supervisor
    invoke_skill = registry.create_invoke_skill_tool(model)
    ```
"""

from deepagents.skills.base import (
    Skill,
    SkillConfig,
    SkillState,
    SimpleSkill,
    create_skill_from_config,
)
from deepagents.skills.executor import (
    SkillExecutor,
    SkillResult,
    create_executor,
)
from deepagents.skills.registry import (
    SkillRegistry,
    create_invoke_skill_tool,
    load_skills_from_yaml,
)

__all__ = [
    # Base classes
    "Skill",
    "SkillConfig",
    "SkillState",
    "SimpleSkill",
    "create_skill_from_config",
    # Registry
    "SkillRegistry",
    "create_invoke_skill_tool",
    "load_skills_from_yaml",
    # Executor
    "SkillExecutor",
    "SkillResult",
    "create_executor",
]
