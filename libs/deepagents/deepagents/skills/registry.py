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

from deepagents.skills.base import (
    Skill,
    SkillConfig,
    SimpleSkill,
    WorkflowSkill,
    WorkflowConfig,
    create_skill_from_config,
)

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

    def load_from_directory(
        self,
        directory: str | Path,
        *,
        tool_resolver: Callable[[str], BaseTool | None] | None = None,
    ) -> None:
        """Load skills from Python modules in a directory (plugin pattern).

        Each Python module in the directory can define a `register(registry)` function
        that will be called with this registry instance. This enables progressive
        disclosure - users can add skills by simply dropping a `.py` file into the
        skills directory.

        Example skill module:
            ```python
            # my_skill.py
            def register(registry):
                registry.register({
                    "name": "my-skill",
                    "description": "Does something useful",
                    "system_prompt": "You are a helpful assistant...",
                    "tools": []
                })
            ```

        Args:
            directory: Path to the directory containing skill modules.
            tool_resolver: Optional function to resolve tool names to BaseTool instances.

        Raises:
            FileNotFoundError: If directory doesn't exist.
        """
        import importlib.util
        import sys

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Skills directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Store tool_resolver for use by register() calls
        self._current_tool_resolver = tool_resolver

        loaded_count = 0
        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue  # Skip __init__.py, __pycache__, etc.

            module_name = f"_skill_plugin_{py_file.stem}"

            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    logger.warning("Could not load spec for: %s", py_file)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Call register() if it exists
                if hasattr(module, "register") and callable(module.register):
                    module.register(self)
                    loaded_count += 1
                    logger.debug("Loaded skill plugin: %s", py_file.name)
                else:
                    logger.warning("No register() function in: %s", py_file.name)

            except Exception as e:
                logger.exception("Error loading skill plugin %s: %s", py_file.name, e)
            finally:
                # Clean up module from sys.modules to avoid conflicts
                sys.modules.pop(module_name, None)

        self._current_tool_resolver = None
        logger.info("Loaded %d skill plugins from %s", loaded_count, directory)

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

        description = f"""Invoke a single specialized skill to handle a task.

Available skills:
{skills_desc}

The skill will execute in an isolated context and return a summary of its work.
Use this for single-skill tasks. For multi-skill tasks, use invoke_skills_parallel or invoke_skills_sequential instead.
"""

        return StructuredTool.from_function(
            func=invoke_skill,
            coroutine=ainvoke_skill,
            name="invoke_skill",
            description=description,
        )

    def create_orchestration_tools(
        self,
        model: BaseChatModel,
        *,
        additional_context: dict[str, Any] | None = None,
        max_parallel: int = 5,
    ) -> list[BaseTool]:
        """Create all skill invocation tools including orchestration.

        This creates three tools:
        - invoke_skill: Single skill invocation
        - invoke_skills_parallel: Run multiple skills concurrently
        - invoke_skills_sequential: Chain skills with output passing

        Args:
            model: The language model to use for skill execution.
            additional_context: Optional context to pass to all skills.
            max_parallel: Maximum concurrent skill executions.

        Returns:
            List of skill invocation tools.
        """
        import asyncio

        registry = self

        # Single skill invocation (reuse existing)
        invoke_skill_tool = self.create_invoke_skill_tool(
            model, additional_context=additional_context
        )

        # Build skills description for tool docstrings
        skills_desc = "\n".join(
            f"- **{s.name}**: {s.description}" for s in self._skills.values()
        )
        if not skills_desc:
            skills_desc = "(No skills registered)"

        # =====================================================================
        # Parallel Execution Tool
        # =====================================================================

        def invoke_skills_parallel(
            invocations: list[dict[str, str]],
        ) -> str:
            """Invoke multiple skills in parallel.

            Args:
                invocations: List of skill invocations, each with:
                    - skill_name: Name of the skill to invoke
                    - task: Task description for that skill

            Returns:
                Combined results from all skills.
            """
            if not invocations:
                return "Error: No skill invocations provided"

            # Validate all skills exist first
            errors = []
            for inv in invocations:
                skill_name = inv.get("skill_name", "")
                if not registry.get(skill_name):
                    errors.append(f"Unknown skill: '{skill_name}'")
            if errors:
                available = ", ".join(registry._skills.keys())
                return f"Error: {'; '.join(errors)}. Available: {available}"

            # Execute in parallel using threads
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=min(len(invocations), max_parallel)) as executor:
                future_to_name = {}
                for inv in invocations:
                    skill_name = inv["skill_name"]
                    task = inv.get("task", "")
                    skill = registry.get(skill_name)
                    future = executor.submit(
                        skill.invoke, task, model, context=additional_context
                    )
                    future_to_name[future] = skill_name

                for future in as_completed(future_to_name):
                    skill_name = future_to_name[future]
                    try:
                        results[skill_name] = future.result()
                    except Exception as e:
                        logger.exception("Error in parallel skill %s", skill_name)
                        results[skill_name] = f"Error: {e}"

            # Format results
            output_parts = []
            for inv in invocations:  # Preserve original order
                skill_name = inv["skill_name"]
                result = results.get(skill_name, "No result")
                output_parts.append(f"## {skill_name}\n{result}")

            return "\n\n".join(output_parts)

        async def ainvoke_skills_parallel(
            invocations: list[dict[str, str]],
        ) -> str:
            """Invoke multiple skills in parallel (async).

            Args:
                invocations: List of skill invocations.

            Returns:
                Combined results from all skills.
            """
            if not invocations:
                return "Error: No skill invocations provided"

            # Validate all skills exist first
            errors = []
            for inv in invocations:
                skill_name = inv.get("skill_name", "")
                if not registry.get(skill_name):
                    errors.append(f"Unknown skill: '{skill_name}'")
            if errors:
                available = ", ".join(registry._skills.keys())
                return f"Error: {'; '.join(errors)}. Available: {available}"

            # Execute in parallel using asyncio
            semaphore = asyncio.Semaphore(max_parallel)

            async def run_skill(skill_name: str, task: str) -> tuple[str, str]:
                async with semaphore:
                    skill = registry.get(skill_name)
                    try:
                        result = await skill.ainvoke(
                            task, model, context=additional_context
                        )
                        return skill_name, result
                    except Exception as e:
                        logger.exception("Error in parallel skill %s", skill_name)
                        return skill_name, f"Error: {e}"

            tasks = [
                run_skill(inv["skill_name"], inv.get("task", ""))
                for inv in invocations
            ]
            results_list = await asyncio.gather(*tasks)
            results = dict(results_list)

            # Format results
            output_parts = []
            for inv in invocations:  # Preserve original order
                skill_name = inv["skill_name"]
                result = results.get(skill_name, "No result")
                output_parts.append(f"## {skill_name}\n{result}")

            return "\n\n".join(output_parts)

        parallel_description = f"""Invoke multiple skills in PARALLEL (concurrently).

Use this when:
- Tasks are INDEPENDENT and don't depend on each other's outputs
- You want to gather information from multiple sources simultaneously
- Speed is important and skills can run at the same time

Example: Researching a company from multiple angles simultaneously:
```json
[
  {{"skill_name": "market-research", "task": "Research AAPL market position"}},
  {{"skill_name": "financial-analysis", "task": "Analyze AAPL financials"}},
  {{"skill_name": "news-analysis", "task": "Find recent AAPL news"}}
]
```

Available skills:
{skills_desc}
"""

        parallel_tool = StructuredTool.from_function(
            func=invoke_skills_parallel,
            coroutine=ainvoke_skills_parallel,
            name="invoke_skills_parallel",
            description=parallel_description,
        )

        # =====================================================================
        # Sequential Execution Tool
        # =====================================================================

        def invoke_skills_sequential(
            invocations: list[dict[str, str]],
        ) -> str:
            """Invoke skills sequentially, passing outputs forward.

            Args:
                invocations: List of skill invocations in order. Each has:
                    - skill_name: Name of the skill to invoke
                    - task: Task description (can reference {{previous_output}})

            Returns:
                Final result after all skills complete.
            """
            if not invocations:
                return "Error: No skill invocations provided"

            # Validate all skills exist first
            errors = []
            for inv in invocations:
                skill_name = inv.get("skill_name", "")
                if not registry.get(skill_name):
                    errors.append(f"Unknown skill: '{skill_name}'")
            if errors:
                available = ", ".join(registry._skills.keys())
                return f"Error: {'; '.join(errors)}. Available: {available}"

            # Execute sequentially
            previous_output = ""
            all_outputs: list[tuple[str, str]] = []

            for i, inv in enumerate(invocations):
                skill_name = inv["skill_name"]
                task = inv.get("task", "")

                # Inject previous output if referenced
                if "{{previous_output}}" in task:
                    task = task.replace("{{previous_output}}", previous_output)
                elif i > 0 and previous_output:
                    # Auto-append context from previous skill
                    task = f"{task}\n\nContext from previous step ({invocations[i-1]['skill_name']}):\n{previous_output}"

                skill = registry.get(skill_name)
                try:
                    result = skill.invoke(task, model, context=additional_context)
                    previous_output = result
                    all_outputs.append((skill_name, result))
                except Exception as e:
                    logger.exception("Error in sequential skill %s", skill_name)
                    error_msg = f"Error: {e}"
                    all_outputs.append((skill_name, error_msg))
                    # Continue with error as context
                    previous_output = error_msg

            # Format results showing the chain
            output_parts = []
            for i, (skill_name, result) in enumerate(all_outputs):
                step_num = i + 1
                output_parts.append(f"## Step {step_num}: {skill_name}\n{result}")

            return "\n\n".join(output_parts)

        async def ainvoke_skills_sequential(
            invocations: list[dict[str, str]],
        ) -> str:
            """Invoke skills sequentially (async).

            Args:
                invocations: List of skill invocations in order.

            Returns:
                Final result after all skills complete.
            """
            if not invocations:
                return "Error: No skill invocations provided"

            # Validate all skills exist first
            errors = []
            for inv in invocations:
                skill_name = inv.get("skill_name", "")
                if not registry.get(skill_name):
                    errors.append(f"Unknown skill: '{skill_name}'")
            if errors:
                available = ", ".join(registry._skills.keys())
                return f"Error: {'; '.join(errors)}. Available: {available}"

            # Execute sequentially
            previous_output = ""
            all_outputs: list[tuple[str, str]] = []

            for i, inv in enumerate(invocations):
                skill_name = inv["skill_name"]
                task = inv.get("task", "")

                # Inject previous output if referenced
                if "{{previous_output}}" in task:
                    task = task.replace("{{previous_output}}", previous_output)
                elif i > 0 and previous_output:
                    # Auto-append context from previous skill
                    task = f"{task}\n\nContext from previous step ({invocations[i-1]['skill_name']}):\n{previous_output}"

                skill = registry.get(skill_name)
                try:
                    result = await skill.ainvoke(
                        task, model, context=additional_context
                    )
                    previous_output = result
                    all_outputs.append((skill_name, result))
                except Exception as e:
                    logger.exception("Error in sequential skill %s", skill_name)
                    error_msg = f"Error: {e}"
                    all_outputs.append((skill_name, error_msg))
                    previous_output = error_msg

            # Format results showing the chain
            output_parts = []
            for i, (skill_name, result) in enumerate(all_outputs):
                step_num = i + 1
                output_parts.append(f"## Step {step_num}: {skill_name}\n{result}")

            return "\n\n".join(output_parts)

        sequential_description = f"""Invoke skills SEQUENTIALLY, passing each output to the next.

Use this when:
- Tasks DEPEND on each other's outputs
- You need a pipeline/chain of processing
- Later steps need results from earlier steps

The output from each skill is automatically passed to the next. You can explicitly
reference it with {{{{previous_output}}}} in the task description.

Example: Research then analyze then summarize:
```json
[
  {{"skill_name": "market-research", "task": "Research AAPL competitors"}},
  {{"skill_name": "financial-analysis", "task": "Compare AAPL financials against competitors found"}},
  {{"skill_name": "report-writer", "task": "Write executive summary of the analysis"}}
]
```

Available skills:
{skills_desc}
"""

        sequential_tool = StructuredTool.from_function(
            func=invoke_skills_sequential,
            coroutine=ainvoke_skills_sequential,
            name="invoke_skills_sequential",
            description=sequential_description,
        )

        return [invoke_skill_tool, parallel_tool, sequential_tool]

    def create_skill_discovery_tool(
        self,
        model: BaseChatModel,
        *,
        additional_context: dict[str, Any] | None = None,
    ) -> BaseTool:
        """Create a single skill discovery and activation tool.

        This tool implements the progressive disclosure pattern:
        1. Main agent describes what it needs to accomplish
        2. The tool discovers the best matching skill
        3. The skill is activated and executed
        4. Only the final result is returned to the main agent

        This is the preferred pattern for skill invocation - the main agent
        doesn't need to know about specific skills, just describe the task.

        Args:
            model: The language model to use for skill execution.
            additional_context: Optional context to pass to skills.

        Returns:
            A StructuredTool that discovers and activates the appropriate skill.
        """
        registry = self

        def activate_skill(task: str) -> str:
            """Activate the most appropriate skill to handle a task.

            This tool automatically discovers which skill is best suited for
            the given task based on skill descriptions and activates it.
            You don't need to specify the skill name - just describe what
            you need accomplished.

            Args:
                task: Detailed description of what you want to accomplish.

            Returns:
                Result from the activated skill.
            """
            # Find the best matching skill
            skill, confidence = registry._discover_skill(task, model)

            if not skill:
                available = ", ".join(registry._skills.keys())
                return f"No suitable skill found for task. Available skills: {available}"

            logger.info(
                "Discovered skill '%s' (confidence: %.2f) for task: %s",
                skill.name, confidence, task[:100]
            )

            try:
                result = skill.invoke(task, model, context=additional_context)
                return result
            except Exception as e:
                logger.exception("Error invoking skill %s", skill.name)
                return f"Error executing skill '{skill.name}': {e}"

        async def aactivate_skill(task: str) -> str:
            """Activate the most appropriate skill asynchronously.

            Args:
                task: Detailed description of what you want to accomplish.

            Returns:
                Result from the activated skill.
            """
            # Find the best matching skill
            skill, confidence = registry._discover_skill(task, model)

            if not skill:
                available = ", ".join(registry._skills.keys())
                return f"No suitable skill found for task. Available skills: {available}"

            logger.info(
                "Discovered skill '%s' (confidence: %.2f) for task: %s",
                skill.name, confidence, task[:100]
            )

            try:
                result = await skill.ainvoke(task, model, context=additional_context)
                return result
            except Exception as e:
                logger.exception("Error invoking skill %s", skill.name)
                return f"Error executing skill '{skill.name}': {e}"

        # Build skill catalog for description
        skills_catalog = "\n".join(
            f"- **{s.name}**: {s.description}" for s in self._skills.values()
        )
        if not skills_catalog:
            skills_catalog = "(No skills registered)"

        description = f"""Activate the most appropriate skill to handle a task.

This tool automatically discovers and activates the best skill for your task.
Just describe what you need - the system handles skill selection.

Available skill capabilities:
{skills_catalog}

Example: "Research and analyze AAPL stock performance" will automatically
activate a research or analysis skill if available.
"""

        return StructuredTool.from_function(
            func=activate_skill,
            coroutine=aactivate_skill,
            name="activate_skill",
            description=description,
        )

    def _discover_skill(
        self,
        task: str,
        model: BaseChatModel,
    ) -> tuple[Skill | None, float]:
        """Discover the best matching skill for a task.

        Uses semantic matching between the task description and skill
        descriptions to find the most appropriate skill.

        Args:
            task: The task description.
            model: Language model for semantic matching.

        Returns:
            Tuple of (best_skill, confidence_score). Returns (None, 0.0) if
            no suitable skill is found.
        """
        if not self._skills:
            return None, 0.0

        # For a single skill, just return it
        if len(self._skills) == 1:
            skill = list(self._skills.values())[0]
            return skill, 1.0

        # Use the model to select the best skill
        skills_info = "\n".join(
            f"{i+1}. {s.name}: {s.description}"
            for i, s in enumerate(self._skills.values())
        )

        selection_prompt = f"""Given the following task, select the most appropriate skill to handle it.

Task: {task}

Available skills:
{skills_info}

Respond with ONLY the skill name (e.g., "market-research") that best matches the task.
If no skill is a good match, respond with "NONE".
"""

        try:
            from langchain_core.messages import HumanMessage

            response = model.invoke([HumanMessage(content=selection_prompt)])
            selected_name = str(response.content).strip().lower()

            # Clean up the response
            selected_name = selected_name.replace('"', '').replace("'", "")

            if selected_name == "none":
                return None, 0.0

            # Find the skill by name (case-insensitive)
            for name, skill in self._skills.items():
                if name.lower() == selected_name:
                    return skill, 0.9  # High confidence from model selection

            # Fuzzy match - check if response contains a skill name
            for name, skill in self._skills.items():
                if name.lower() in selected_name:
                    return skill, 0.7

            # Fall back to first skill if nothing matched
            logger.warning(
                "Model returned '%s' but no matching skill found. "
                "Falling back to first skill.",
                selected_name
            )
            return list(self._skills.values())[0], 0.3

        except Exception as e:
            logger.exception("Error in skill discovery")
            # Fall back to first skill
            return list(self._skills.values())[0], 0.2

    def register_workflow_skill(
        self,
        config: dict[str, Any],
        tool_registry: dict[str, BaseTool] | None = None,
    ) -> None:
        """Register a workflow-based skill from configuration.

        Args:
            config: Skill configuration with 'workflow' key.
            tool_registry: Tool registry for resolving tool names.

        Example config:
            ```yaml
            name: stock-analyzer
            description: Analyze stocks comprehensively
            workflow:
              - tools: market_data
                task_template: "Get data for {{input}}"
              - tools: [technical_analysis, sentiment_analysis]
              - tools: report_generator
            ```
        """
        if "workflow" not in config:
            raise ValueError("Workflow skill config must have 'workflow' key")

        skill = WorkflowSkill.from_config(config, tool_registry)
        self.register(skill)


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
