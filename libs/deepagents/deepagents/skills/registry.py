"""Skill registry for managing and loading skills.

This module provides a registry for skill management, supporting:
- Registration of skills by name
- Loading skills from YAML/JSON configuration files
- Creation of the `invoke_skill` tool for supervisor agents
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool

from deepagents.skills.base import Skill, SkillConfig, SimpleSkill, create_skill_from_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Registry for managing skills.

    The registry provides:
    - Skill registration and lookup by name
    - Loading skills from configuration files
    - Generation of the invoke_skill tool for supervisor agents

    Example:
        ```python
        registry = SkillRegistry()

        # Register a skill from dict
        registry.register({
            "name": "market-research",
            "description": "Research market conditions",
            "system_prompt": "You are a market researcher...",
            "tools": [web_search_tool, news_api_tool]
        })

        # Or load from YAML
        registry.load_from_config("config/skills.yaml")

        # Create the invoke_skill tool
        tool = registry.create_invoke_skill_tool(model)
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty skill registry."""
        self._skills: dict[str, Skill] = {}
        self._tool_registry: dict[str, list[BaseTool]] = {}

    @property
    def skills(self) -> dict[str, Skill]:
        """Get all registered skills."""
        return self._skills.copy()

    def register(
        self,
        skill: Skill | SkillConfig | dict[str, Any],
        *,
        tools: list[BaseTool] | None = None,
    ) -> None:
        """Register a skill.

        Args:
            skill: Skill instance, SkillConfig, or dictionary config.
            tools: Optional tools to associate with this skill (for tool name resolution).

        Raises:
            ValueError: If skill name is already registered.
        """
        if isinstance(skill, dict):
            skill_instance = create_skill_from_config(skill)
        elif isinstance(skill, SkillConfig):
            skill_instance = SimpleSkill(skill)
        else:
            skill_instance = skill

        name = skill_instance.name

        if name in self._skills:
            logger.warning("Overwriting existing skill: %s", name)

        self._skills[name] = skill_instance

        if tools:
            self._tool_registry[name] = tools

        logger.debug("Registered skill: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a skill from the registry.

        Args:
            name: Name of the skill to remove.
        """
        self._skills.pop(name, None)
        self._tool_registry.pop(name, None)

    def get(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: Name of the skill.

        Returns:
            Skill instance or None if not found.
        """
        return self._skills.get(name)

    def get_tools(self, name: str) -> list[BaseTool]:
        """Get tools associated with a skill.

        Args:
            name: Name of the skill.

        Returns:
            List of tools for the skill.
        """
        return self._tool_registry.get(name, [])

    def list_skills(self) -> list[dict[str, str]]:
        """List all registered skills with their descriptions.

        Returns:
            List of dicts with 'name' and 'description' keys.
        """
        return [
            {"name": skill.name, "description": skill.description}
            for skill in self._skills.values()
        ]

    def load_from_config(
        self,
        config_path: str | Path,
        *,
        tool_resolver: Callable[[str], BaseTool | None] | None = None,
    ) -> None:
        """Load skills from a YAML or JSON configuration file.

        The config file should have a 'skills' key containing a list of skill configs:

        ```yaml
        skills:
          - name: market-research
            description: Research market conditions
            system_prompt: You are a market researcher...
            tools: [web_search, news_api]  # Tool names to resolve
          - name: quant-analysis
            description: Quantitative analysis
            system_prompt: You are a quant analyst...
            tools: [execute_code, market_data]
        ```

        Args:
            config_path: Path to the YAML or JSON configuration file.
            tool_resolver: Optional function to resolve tool names to BaseTool instances.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config format is invalid.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            if config_path.suffix in (".yaml", ".yml"):
                config = yaml.safe_load(f)
            else:
                import json

                config = json.load(f)

        if not isinstance(config, dict) or "skills" not in config:
            raise ValueError("Config must have a 'skills' key with list of skill definitions")

        skills_list = config["skills"]
        if not isinstance(skills_list, list):
            raise ValueError("'skills' must be a list")

        for skill_data in skills_list:
            # Resolve tool names if tool_resolver provided
            if tool_resolver and "tools" in skill_data:
                resolved_tools = []
                for tool_name in skill_data.get("tools", []):
                    if isinstance(tool_name, str):
                        tool = tool_resolver(tool_name)
                        if tool:
                            resolved_tools.append(tool)
                        else:
                            logger.warning("Could not resolve tool: %s", tool_name)
                    else:
                        # Already a tool instance
                        resolved_tools.append(tool_name)
                skill_data = {**skill_data, "tools": resolved_tools}

            self.register(skill_data)

        logger.info("Loaded %d skills from %s", len(skills_list), config_path)

    def load_from_dict(
        self,
        config: dict[str, Any],
        *,
        tool_resolver: Callable[[str], BaseTool | None] | None = None,
    ) -> None:
        """Load skills from a dictionary configuration.

        Args:
            config: Dictionary with 'skills' key.
            tool_resolver: Optional function to resolve tool names.
        """
        if "skills" not in config:
            raise ValueError("Config must have a 'skills' key")

        for skill_data in config["skills"]:
            if tool_resolver and "tools" in skill_data:
                resolved_tools = []
                for tool_name in skill_data.get("tools", []):
                    if isinstance(tool_name, str):
                        tool = tool_resolver(tool_name)
                        if tool:
                            resolved_tools.append(tool)
                    else:
                        resolved_tools.append(tool_name)
                skill_data = {**skill_data, "tools": resolved_tools}

            self.register(skill_data)

    def create_invoke_skill_tool(
        self,
        model: BaseChatModel,
        *,
        additional_context: dict[str, Any] | None = None,
    ) -> BaseTool:
        """Create the invoke_skill tool for supervisor agents.

        This tool allows a supervisor agent to invoke registered skills by name.

        Args:
            model: The language model to use for skill execution.
            additional_context: Optional context to pass to all skills.

        Returns:
            A StructuredTool that can invoke skills by name.
        """
        registry = self

        def invoke_skill(
            skill_name: str,
            task: str,
        ) -> str:
            """Invoke a skill to handle a specific task.

            Args:
                skill_name: Name of the skill to invoke.
                task: Detailed task description for the skill.

            Returns:
                Summary of the skill's work.
            """
            skill = registry.get(skill_name)
            if not skill:
                available = ", ".join(registry._skills.keys())
                return f"Error: Unknown skill '{skill_name}'. Available skills: {available}"

            try:
                result = skill.invoke(
                    task,
                    model,
                    context=additional_context,
                )
                return result
            except Exception as e:
                logger.exception("Error invoking skill %s", skill_name)
                return f"Error executing skill '{skill_name}': {e}"

        async def ainvoke_skill(
            skill_name: str,
            task: str,
        ) -> str:
            """Invoke a skill asynchronously.

            Args:
                skill_name: Name of the skill to invoke.
                task: Detailed task description for the skill.

            Returns:
                Summary of the skill's work.
            """
            skill = registry.get(skill_name)
            if not skill:
                available = ", ".join(registry._skills.keys())
                return f"Error: Unknown skill '{skill_name}'. Available skills: {available}"

            try:
                result = await skill.ainvoke(
                    task,
                    model,
                    context=additional_context,
                )
                return result
            except Exception as e:
                logger.exception("Error invoking skill %s", skill_name)
                return f"Error executing skill '{skill_name}': {e}"

        # Build description with available skills
        skills_desc = "\n".join(
            f"- **{s.name}**: {s.description}" for s in self._skills.values()
        )
        if not skills_desc:
            skills_desc = "(No skills registered)"

        description = f"""Invoke a specialized skill to handle a task.

Available skills:
{skills_desc}

The skill will execute in an isolated context and return a summary of its work.
Use this to delegate complex, specialized tasks to the appropriate skill.
"""

        return StructuredTool.from_function(
            func=invoke_skill,
            coroutine=ainvoke_skill,
            name="invoke_skill",
            description=description,
        )


def load_skills_from_yaml(
    path: str | Path,
    *,
    tool_resolver: Callable[[str], BaseTool | None] | None = None,
) -> SkillRegistry:
    """Convenience function to load skills from YAML and return a registry.

    Args:
        path: Path to the YAML configuration file.
        tool_resolver: Optional function to resolve tool names.

    Returns:
        SkillRegistry with loaded skills.
    """
    registry = SkillRegistry()
    registry.load_from_config(path, tool_resolver=tool_resolver)
    return registry


def create_invoke_skill_tool(
    skills: Sequence[Skill | SkillConfig | dict[str, Any]],
    model: BaseChatModel,
    *,
    additional_context: dict[str, Any] | None = None,
) -> BaseTool:
    """Create an invoke_skill tool from a list of skills.

    Convenience function for creating the tool without manually creating a registry.

    Args:
        skills: List of skills (instances, configs, or dicts).
        model: The language model to use.
        additional_context: Optional context to pass to skills.

    Returns:
        The invoke_skill tool.
    """
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry.create_invoke_skill_tool(model, additional_context=additional_context)


__all__ = [
    "SkillRegistry",
    "create_invoke_skill_tool",
    "load_skills_from_yaml",
]
