"""Skills module for middleware-free agent architecture.

This module provides a skills-based agent architecture that works with
langchain>=0.2.0 and langgraph>=0.1.0 (pre-middleware versions).

Skills are isolated subagents with custom tools and system prompts that can
be invoked by a supervisor agent via the `invoke_skill` or `activate_skill` tools.

Key Concepts:
- **SimpleSkill**: A basic ReAct-style skill with direct tool access
- **WorkflowSkill**: A skill that orchestrates multiple tools via subagents,
  with sequential steps and parallel tool execution within steps

Skill Discovery (Progressive Disclosure):
The `activate_skill` tool enables progressive disclosure - the main agent
doesn't need to know specific skill names, just describe the task and the
system automatically discovers and activates the appropriate skill.

Example:
    ```python
    from deepagents.skills import SkillRegistry, create_workflow_skill

    # Create a skill registry
    registry = SkillRegistry()

    # Register a workflow skill
    registry.register_workflow_skill({
        "name": "stock-analyzer",
        "description": "Analyze stocks comprehensively",
        "workflow": [
            {"tools": "market_data", "task_template": "Get data for {{input}}"},
            {"tools": ["technical_analysis", "sentiment_analysis"]},  # parallel
            {"tools": "report_generator"},
        ],
    }, tool_registry={"market_data": market_data_tool, ...})

    # Create skill discovery tool (single activation point)
    activate_skill = registry.create_skill_discovery_tool(model)

    # Main agent just describes task - skill is auto-discovered
    result = activate_skill("Analyze AAPL stock performance")
    ```
"""

from deepagents.skills.base import (
    Skill,
    SkillConfig,
    SkillState,
    SimpleSkill,
    WorkflowStep,
    WorkflowConfig,
    WorkflowSkill,
    create_skill_from_config,
    create_workflow_skill,
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
    # Workflow skills
    "WorkflowStep",
    "WorkflowConfig",
    "WorkflowSkill",
    "create_workflow_skill",
    # Registry
    "SkillRegistry",
    "create_invoke_skill_tool",
    "load_skills_from_yaml",
    # Executor
    "SkillExecutor",
    "SkillResult",
    "create_executor",
]
