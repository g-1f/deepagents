"""DeepAgents package.

A skills-based agent architecture with progressive disclosure.

Example:
    ```python
    from deepagents import create_skill_agent

    # Using skill plugins from directory (progressive disclosure)
    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
        skills_directory="./skills/",  # Drop .py files here to add skills
    )

    # Or using inline skill definitions
    agent = create_skill_agent(
        model="anthropic:claude-sonnet-4-20250514",
        skills=[
            {
                "name": "research",
                "description": "Research a topic",
                "system_prompt": "You are a researcher...",
                "tools": [web_search]
            }
        ]
    )
    ```
"""

from deepagents.graph_no_middleware import create_skill_agent
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

__all__ = [
    "create_skill_agent",
    "Skill",
    "SkillConfig",
    "SkillExecutor",
    "SkillRegistry",
    "SkillResult",
    "SimpleSkill",
    "create_invoke_skill_tool",
    "load_skills_from_yaml",
]
