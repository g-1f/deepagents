"""Base classes for skills architecture.

This module provides the foundational classes for a middleware-free skills system
that works with langchain>=0.2.0 and langgraph>=0.1.0 (pre-middleware versions).

Skills are isolated subagents with custom tools and system prompts that can be
invoked by a supervisor agent.

Key concepts:
- **Skill**: An isolated agent with its own tools and system prompt
- **WorkflowSkill**: A skill that orchestrates multiple tools via subagents
- **WorkflowStep**: A step in a workflow (serial or parallel tool execution)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)


@dataclass
class SkillConfig:
    """Configuration for a skill.

    Skills are isolated subagents with custom tools and system prompts.
    They can be defined via Python dicts, YAML configuration, or directly
    with this dataclass.

    Attributes:
        name: Unique identifier for the skill (lowercase alphanumeric and hyphens).
        description: What the skill does - used by supervisor to decide when to invoke.
        system_prompt: Instructions for the skill agent.
        tools: List of tools the skill can use.
        model: Optional model override for this skill.
        max_iterations: Maximum number of agent iterations (default 25).
        timeout: Timeout in seconds for skill execution (default 300).
        metadata: Arbitrary key-value pairs for additional configuration.
        allowed_tools: List of pre-approved tools for this skill.
    """

    name: str
    description: str
    system_prompt: str
    tools: list[BaseTool | Callable | dict[str, Any]] = field(default_factory=list)
    model: str | BaseChatModel | None = None
    max_iterations: int = 25
    timeout: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillConfig:
        """Create a SkillConfig from a dictionary.

        This is the primary method for creating skills from YAML or JSON config files.

        Args:
            data: Dictionary containing skill configuration.

        Returns:
            SkillConfig instance.

        Raises:
            ValueError: If required fields (name, description) are missing.
        """
        if "name" not in data:
            raise ValueError("Skill config must have 'name' field")
        if "description" not in data:
            raise ValueError("Skill config must have 'description' field")

        return cls(
            name=data["name"],
            description=data["description"],
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            tools=data.get("tools", []),
            model=data.get("model"),
            max_iterations=data.get("max_iterations", 25),
            timeout=data.get("timeout", 300.0),
            metadata=data.get("metadata", {}),
            allowed_tools=data.get("allowed_tools", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "model": self.model if isinstance(self.model, str) else None,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "metadata": self.metadata,
            "allowed_tools": self.allowed_tools,
        }


@dataclass
class WorkflowStep:
    """A step in a workflow.

    Each step can execute one or more tools. If multiple tools are specified,
    they run in parallel. Steps are executed sequentially, with each step
    receiving the output from previous steps.

    Attributes:
        tools: Tool name(s) for this step. String for single tool, list for parallel.
        task_template: Task description template. Can use {{input}}, {{previous_output}}.
        parallel: Whether tools in this step run in parallel (auto-detected from tools).
        name: Optional name for this step (for debugging/logging).
        pass_output: Whether to pass this step's output to the next step (default True).
    """

    tools: str | list[str]
    task_template: str = "{{input}}"
    name: str | None = None
    pass_output: bool = True

    @property
    def is_parallel(self) -> bool:
        """Check if this step runs multiple tools in parallel."""
        return isinstance(self.tools, list) and len(self.tools) > 1

    @property
    def tool_list(self) -> list[str]:
        """Get tools as a list."""
        if isinstance(self.tools, str):
            return [self.tools]
        return self.tools

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowStep:
        """Create a WorkflowStep from a dictionary."""
        return cls(
            tools=data["tools"],
            task_template=data.get("task_template", data.get("task", "{{input}}")),
            name=data.get("name"),
            pass_output=data.get("pass_output", True),
        )


@dataclass
class WorkflowConfig:
    """Configuration for a workflow-based skill.

    A workflow defines a series of steps, where each step spins up subagents
    to execute tools. This enables dynamic orchestration based on the skill's
    needs.

    Example:
        ```yaml
        workflow:
          - tools: web_search
            task_template: "Search for: {{input}}"
          - tools: [sentiment_analyzer, fact_checker]  # parallel
            task_template: "Analyze: {{previous_output}}"
          - tools: summarizer
            task_template: "Summarize findings: {{previous_output}}"
        ```

    Attributes:
        steps: List of workflow steps to execute.
        synthesize_prompt: Prompt for final synthesis (optional).
    """

    steps: list[WorkflowStep] = field(default_factory=list)
    synthesize_prompt: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[dict[str, Any]]) -> WorkflowConfig:
        """Create a WorkflowConfig from a dictionary or list.

        Args:
            data: Either a dict with 'steps' key, or a list of step dicts.
        """
        if isinstance(data, list):
            steps = [WorkflowStep.from_dict(s) for s in data]
            return cls(steps=steps)

        steps = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            steps=steps,
            synthesize_prompt=data.get("synthesize_prompt"),
        )


class Skill(ABC):
    """Abstract base class for skills.

    Skills are isolated agents that can be invoked by a supervisor.
    Each skill has its own context window and tools.
    """

    def __init__(self, config: SkillConfig) -> None:
        """Initialize the skill.

        Args:
            config: Configuration for this skill.
        """
        self.config = config
        self._graph: CompiledStateGraph | None = None

    @property
    def name(self) -> str:
        """Get the skill name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get the skill description."""
        return self.config.description

    @abstractmethod
    def build_graph(
        self,
        model: BaseChatModel,
        *,
        checkpointer: Checkpointer | None = None,
        store: BaseStore | None = None,
    ) -> CompiledStateGraph:
        """Build the LangGraph for this skill.

        Args:
            model: The language model to use.
            checkpointer: Optional checkpointer for state persistence.
            store: Optional store for persistent storage.

        Returns:
            Compiled state graph ready for execution.
        """
        ...

    @abstractmethod
    def invoke(
        self,
        task: str,
        model: BaseChatModel,
        *,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Invoke the skill with a task.

        Args:
            task: The task description for this skill to complete.
            model: The language model to use.
            context: Optional context to pass to the skill.
            config: Optional runtime configuration.

        Returns:
            Summary of the skill's work as a string.
        """
        ...

    @abstractmethod
    async def ainvoke(
        self,
        task: str,
        model: BaseChatModel,
        *,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Invoke the skill asynchronously.

        Args:
            task: The task description for this skill to complete.
            model: The language model to use.
            context: Optional context to pass to the skill.
            config: Optional runtime configuration.

        Returns:
            Summary of the skill's work as a string.
        """
        ...


# Type for skill state
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class SkillState(TypedDict):
    """State for skill execution."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation messages."""

    iteration_count: int
    """Number of iterations completed."""


class SimpleSkill(Skill):
    """A simple ReAct-style skill implementation.

    This skill uses a basic ReAct loop: the model calls tools,
    observes results, and continues until it produces a final answer.
    """

    def build_graph(
        self,
        model: BaseChatModel,
        *,
        checkpointer: Checkpointer | None = None,
        store: BaseStore | None = None,
    ) -> CompiledStateGraph:
        """Build a ReAct-style graph for this skill.

        Args:
            model: The language model to use.
            checkpointer: Optional checkpointer for state persistence.
            store: Optional store for persistent storage.

        Returns:
            Compiled ReAct graph.
        """
        from langgraph.prebuilt import ToolNode

        # Resolve tools - if they're strings, look them up
        resolved_tools: list[BaseTool] = []
        for tool in self.config.tools:
            if isinstance(tool, BaseTool):
                resolved_tools.append(tool)
            elif callable(tool):
                # Convert callable to tool
                from langchain_core.tools import StructuredTool

                resolved_tools.append(StructuredTool.from_function(tool))
            # Skip dict tools - they need to be resolved externally

        # Bind tools to model if any
        if resolved_tools:
            model_with_tools = model.bind_tools(resolved_tools)
        else:
            model_with_tools = model

        # Create the graph
        builder = StateGraph(SkillState)

        # Agent node - calls the model
        def agent_node(state: SkillState) -> dict[str, Any]:
            """Call the model and return response."""
            messages = state["messages"]

            # Add system prompt if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.config.system_prompt), *messages]

            response = model_with_tools.invoke(messages)
            return {
                "messages": [response],
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        # Routing function
        def should_continue(state: SkillState) -> Literal["tools", "end"]:
            """Determine if we should continue or end."""
            messages = state["messages"]
            last_message = messages[-1]

            # Check iteration limit
            if state.get("iteration_count", 0) >= self.config.max_iterations:
                return "end"

            # Check if the model wants to call tools
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"

            return "end"

        # Add nodes
        builder.add_node("agent", agent_node)
        if resolved_tools:
            builder.add_node("tools", ToolNode(resolved_tools))

        # Add edges
        builder.set_entry_point("agent")
        if resolved_tools:
            builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": "__end__"})
            builder.add_edge("tools", "agent")
        else:
            builder.add_edge("agent", "__end__")

        return builder.compile(checkpointer=checkpointer, store=store)

    def invoke(
        self,
        task: str,
        model: BaseChatModel,
        *,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Invoke the skill synchronously.

        Args:
            task: The task description.
            model: The language model to use.
            context: Optional context (ignored in simple implementation).
            config: Optional runtime configuration.

        Returns:
            The final message content from the skill.
        """
        graph = self.build_graph(model)

        # Create initial state
        initial_state: SkillState = {
            "messages": [HumanMessage(content=task)],
            "iteration_count": 0,
        }

        # Run the graph
        result = graph.invoke(initial_state, config=config)

        # Extract final message
        messages = result.get("messages", [])
        if messages:
            final_msg = messages[-1]
            if isinstance(final_msg, AIMessage):
                return str(final_msg.content)

        return "Skill completed without response."

    async def ainvoke(
        self,
        task: str,
        model: BaseChatModel,
        *,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Invoke the skill asynchronously.

        Args:
            task: The task description.
            model: The language model to use.
            context: Optional context (ignored in simple implementation).
            config: Optional runtime configuration.

        Returns:
            The final message content from the skill.
        """
        graph = self.build_graph(model)

        # Create initial state
        initial_state: SkillState = {
            "messages": [HumanMessage(content=task)],
            "iteration_count": 0,
        }

        # Run the graph asynchronously
        result = await graph.ainvoke(initial_state, config=config)

        # Extract final message
        messages = result.get("messages", [])
        if messages:
            final_msg = messages[-1]
            if isinstance(final_msg, AIMessage):
                return str(final_msg.content)

        return "Skill completed without response."


def create_skill_from_config(config: SkillConfig | dict[str, Any]) -> SimpleSkill:
    """Create a skill from configuration.

    Args:
        config: SkillConfig instance or dictionary.

    Returns:
        SimpleSkill instance.
    """
    if isinstance(config, dict):
        config = SkillConfig.from_dict(config)
    return SimpleSkill(config)


class WorkflowSkill(Skill):
    """A skill that orchestrates multiple tools via subagents.

    This skill type enables dynamic tool orchestration based on a workflow
    definition. Each step in the workflow spins up one or more subagents
    to execute tools, with results flowing from step to step.

    Key features:
    - **Sequential steps**: Steps execute in order, each receiving previous output
    - **Parallel tools within a step**: Multiple tools in a step run concurrently
    - **Subagent isolation**: Each tool execution runs in an isolated subagent
    - **Dynamic binding**: Tools are bound at runtime from a tool registry
    - **Comprehensive logging**: All subagent activity is logged

    Example workflow:
        ```yaml
        name: stock-analysis
        description: Analyze a stock with research and recommendations
        workflow:
          - tools: market_data_fetcher
            task_template: "Fetch market data for {{input}}"
          - tools: [technical_analyzer, news_analyzer]  # parallel
            task_template: "Analyze: {{previous_output}}"
          - tools: recommendation_generator
            task_template: "Generate recommendation based on: {{previous_output}}"
        ```

    The main agent only sees the final output from the last step.
    """

    def __init__(
        self,
        config: SkillConfig,
        workflow: WorkflowConfig,
        tool_registry: dict[str, BaseTool] | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the workflow skill.

        Args:
            config: Basic skill configuration.
            workflow: Workflow definition with steps.
            tool_registry: Registry mapping tool names to BaseTool instances.
            verbose: Enable detailed logging of workflow execution.
        """
        super().__init__(config)
        self.workflow = workflow
        self.tool_registry = tool_registry or {}
        self._max_parallel = 5
        self._verbose = verbose

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Register a tool for use in the workflow.

        Args:
            name: Tool name (must match workflow step tool references).
            tool: The tool instance.
        """
        self.tool_registry[name] = tool

    def register_tools(self, tools: dict[str, BaseTool]) -> None:
        """Register multiple tools.

        Args:
            tools: Mapping of tool names to tool instances.
        """
        self.tool_registry.update(tools)

    def _resolve_tool(self, tool_name: str) -> BaseTool | None:
        """Resolve a tool name to a BaseTool instance."""
        return self.tool_registry.get(tool_name)

    def _create_subagent_for_tool(
        self,
        tool: BaseTool,
        model: BaseChatModel,
    ) -> CompiledStateGraph:
        """Create a minimal subagent graph for a single tool.

        The subagent is a simple ReAct-style agent with just the one tool.
        This provides isolation and allows the tool to be called multiple
        times if needed to complete its task.

        Args:
            tool: The tool to bind to the subagent.
            model: The language model to use.

        Returns:
            Compiled graph for the subagent.
        """
        from langgraph.prebuilt import ToolNode

        model_with_tool = model.bind_tools([tool])

        builder = StateGraph(SkillState)

        def agent_node(state: SkillState) -> dict[str, Any]:
            messages = state["messages"]
            # Add a focused system prompt for tool execution
            system_msg = SystemMessage(
                content=f"""You are a focused agent with access to the '{tool.name}' tool.
Your task is to use this tool to complete the given request.
Execute the tool and return a clear, structured result.
Be concise but comprehensive in your output."""
            )
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [system_msg, *messages]

            response = model_with_tool.invoke(messages)
            return {
                "messages": [response],
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        def should_continue(state: SkillState) -> Literal["tools", "end"]:
            messages = state["messages"]
            last_message = messages[-1]

            # Limit iterations for subagent
            if state.get("iteration_count", 0) >= 10:
                return "end"

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"

            return "end"

        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode([tool]))

        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": "__end__"})
        builder.add_edge("tools", "agent")

        return builder.compile()

    def _log_subagent_start(self, tool_name: str, task: str, step_name: str | None = None) -> None:
        """Log subagent start with formatting."""
        if self._verbose:
            step_info = f" (step: {step_name})" if step_name else ""
            logger.info(
                "┌─── SUBAGENT START: %s%s ───",
                tool_name, step_info
            )
            logger.debug("│ Task: %s", task[:200] + "..." if len(task) > 200 else task)

    def _log_subagent_end(self, tool_name: str, output: str, duration: float) -> None:
        """Log subagent completion with formatting."""
        if self._verbose:
            output_preview = output[:300].replace("\n", " ")
            logger.info(
                "└─── SUBAGENT END: %s (%.2fs) ───",
                tool_name, duration
            )
            logger.debug("│ Output: %s...", output_preview)

    def _log_step_start(self, step_idx: int, step: WorkflowStep) -> None:
        """Log workflow step start."""
        if self._verbose:
            step_name = step.name or f"step_{step_idx}"
            is_parallel = "PARALLEL" if step.is_parallel else "SERIAL"
            tools_str = ", ".join(step.tool_list)
            logger.info(
                "╔═══ WORKFLOW STEP %d: %s (%s) ═══",
                step_idx + 1, step_name, is_parallel
            )
            logger.info("║ Tools: %s", tools_str)

    def _log_step_end(self, step_idx: int, step: WorkflowStep, duration: float) -> None:
        """Log workflow step completion."""
        if self._verbose:
            step_name = step.name or f"step_{step_idx}"
            logger.info(
                "╚═══ STEP %d COMPLETE: %s (%.2fs) ═══",
                step_idx + 1, step_name, duration
            )

    def _run_subagent(
        self,
        tool_name: str,
        task: str,
        model: BaseChatModel,
        step_name: str | None = None,
    ) -> str:
        """Run a subagent for a tool synchronously.

        Args:
            tool_name: Name of the tool.
            task: Task description for the subagent.
            model: Language model to use.
            step_name: Optional step name for logging.

        Returns:
            The subagent's output as a string.
        """
        import time
        start_time = time.time()

        tool = self._resolve_tool(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found in registry"

        self._log_subagent_start(tool_name, task, step_name)

        try:
            graph = self._create_subagent_for_tool(tool, model)
            result = graph.invoke({
                "messages": [HumanMessage(content=task)],
                "iteration_count": 0,
            })

            messages = result.get("messages", [])
            output = f"Subagent for '{tool_name}' completed without output"
            if messages:
                final_msg = messages[-1]
                if isinstance(final_msg, AIMessage):
                    output = str(final_msg.content)

            duration = time.time() - start_time
            self._log_subagent_end(tool_name, output, duration)
            return output

        except Exception as e:
            duration = time.time() - start_time
            logger.exception("Error running subagent for %s (%.2fs)", tool_name, duration)
            return f"Error executing '{tool_name}': {e}"

    async def _run_subagent_async(
        self,
        tool_name: str,
        task: str,
        model: BaseChatModel,
        step_name: str | None = None,
    ) -> str:
        """Run a subagent for a tool asynchronously.

        Args:
            tool_name: Name of the tool.
            task: Task description for the subagent.
            model: Language model to use.
            step_name: Optional step name for logging.

        Returns:
            The subagent's output as a string.
        """
        import time
        start_time = time.time()

        tool = self._resolve_tool(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found in registry"

        self._log_subagent_start(tool_name, task, step_name)

        try:
            graph = self._create_subagent_for_tool(tool, model)
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=task)],
                "iteration_count": 0,
            })

            messages = result.get("messages", [])
            output = f"Subagent for '{tool_name}' completed without output"
            if messages:
                final_msg = messages[-1]
                if isinstance(final_msg, AIMessage):
                    output = str(final_msg.content)

            duration = time.time() - start_time
            self._log_subagent_end(tool_name, output, duration)
            return output

        except Exception as e:
            duration = time.time() - start_time
            logger.exception("Error running subagent for %s (%.2fs)", tool_name, duration)
            return f"Error executing '{tool_name}': {e}"

    def _render_task(
        self,
        template: str,
        input_text: str,
        previous_output: str,
        step_outputs: dict[str, str],
    ) -> str:
        """Render a task template with variable substitution.

        Supported variables:
        - {{input}}: The original input to the skill
        - {{previous_output}}: Output from the previous step
        - {{step_N}}: Output from step N (0-indexed)
        - {{tool_name}}: Output from a specific tool in previous parallel step
        """
        result = template
        result = result.replace("{{input}}", input_text)
        result = result.replace("{{previous_output}}", previous_output)

        # Replace step references
        for key, value in step_outputs.items():
            result = result.replace(f"{{{{{key}}}}}", value)

        return result

    def _execute_step(
        self,
        step: WorkflowStep,
        task: str,
        model: BaseChatModel,
        step_idx: int = 0,
    ) -> dict[str, str]:
        """Execute a workflow step (sync).

        If the step has multiple tools, they run in parallel.

        Args:
            step: The workflow step to execute.
            task: The rendered task for this step.
            model: Language model to use.
            step_idx: Step index for logging.

        Returns:
            Dict mapping tool names to their outputs.
        """
        import time
        start_time = time.time()
        step_name = step.name or f"step_{step_idx}"

        self._log_step_start(step_idx, step)

        tool_names = step.tool_list

        if len(tool_names) == 1:
            # Single tool - run directly
            output = self._run_subagent(tool_names[0], task, model, step_name)
            duration = time.time() - start_time
            self._log_step_end(step_idx, step, duration)
            return {tool_names[0]: output}

        # Multiple tools - run in parallel
        results: dict[str, str] = {}

        if self._verbose:
            logger.info("║ Running %d tools in parallel...", len(tool_names))

        with ThreadPoolExecutor(max_workers=min(len(tool_names), self._max_parallel)) as executor:
            future_to_tool = {
                executor.submit(self._run_subagent, tool_name, task, model, step_name): tool_name
                for tool_name in tool_names
            }

            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    results[tool_name] = future.result()
                except Exception as e:
                    logger.exception("Error in parallel subagent %s", tool_name)
                    results[tool_name] = f"Error: {e}"

        duration = time.time() - start_time
        self._log_step_end(step_idx, step, duration)
        return results

    async def _execute_step_async(
        self,
        step: WorkflowStep,
        task: str,
        model: BaseChatModel,
        step_idx: int = 0,
    ) -> dict[str, str]:
        """Execute a workflow step asynchronously.

        If the step has multiple tools, they run in parallel.

        Args:
            step: The workflow step to execute.
            task: The rendered task for this step.
            model: Language model to use.
            step_idx: Step index for logging.

        Returns:
            Dict mapping tool names to their outputs.
        """
        import time
        start_time = time.time()
        step_name = step.name or f"step_{step_idx}"

        self._log_step_start(step_idx, step)

        tool_names = step.tool_list

        if len(tool_names) == 1:
            # Single tool - run directly
            output = await self._run_subagent_async(tool_names[0], task, model, step_name)
            duration = time.time() - start_time
            self._log_step_end(step_idx, step, duration)
            return {tool_names[0]: output}

        # Multiple tools - run in parallel with semaphore
        if self._verbose:
            logger.info("║ Running %d tools in parallel...", len(tool_names))

        semaphore = asyncio.Semaphore(self._max_parallel)

        async def run_with_semaphore(tool_name: str) -> tuple[str, str]:
            async with semaphore:
                output = await self._run_subagent_async(tool_name, task, model, step_name)
                return tool_name, output

        tasks = [run_with_semaphore(tool_name) for tool_name in tool_names]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results: dict[str, str] = {}
        for i, result in enumerate(results_list):
            tool_name = tool_names[i]
            if isinstance(result, Exception):
                logger.exception("Error in parallel subagent %s", tool_name)
                results[tool_name] = f"Error: {result}"
            else:
                results[tool_name] = result[1]

        duration = time.time() - start_time
        self._log_step_end(step_idx, step, duration)
        return results

    def _combine_step_outputs(self, outputs: dict[str, str]) -> str:
        """Combine outputs from parallel tool executions.

        Args:
            outputs: Dict mapping tool names to their outputs.

        Returns:
            Combined output string.
        """
        if len(outputs) == 1:
            return list(outputs.values())[0]

        parts = []
        for tool_name, output in outputs.items():
            parts.append(f"## {tool_name}\n{output}")

        return "\n\n".join(parts)

    def build_graph(
        self,
        model: BaseChatModel,
        *,
        checkpointer: Checkpointer | None = None,
        store: BaseStore | None = None,
    ) -> CompiledStateGraph:
        """Build a graph for this skill.

        Note: WorkflowSkill doesn't use a traditional graph - it orchestrates
        subagents directly. This method returns a minimal wrapper graph.
        """
        # For WorkflowSkill, we don't use a pre-built graph
        # Instead, invoke() orchestrates subagents dynamically
        raise NotImplementedError(
            "WorkflowSkill uses dynamic orchestration. Use invoke() directly."
        )

    def invoke(
        self,
        task: str,
        model: BaseChatModel,
        *,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Execute the workflow synchronously.

        Processes each step in sequence, with parallel tool execution
        within steps as defined by the workflow.

        Args:
            task: The input task for the skill.
            model: Language model to use.
            context: Optional additional context.
            config: Optional runtime configuration.

        Returns:
            Final output from the workflow.
        """
        import time
        workflow_start = time.time()

        input_text = task
        previous_output = ""
        step_outputs: dict[str, str] = {}
        all_outputs: list[tuple[str, dict[str, str]]] = []

        if self._verbose:
            logger.info("╔" + "═" * 58 + "╗")
            logger.info("║ WORKFLOW START: %-40s ║", self.name)
            logger.info("║ Steps: %-51d ║", len(self.workflow.steps))
            logger.info("╚" + "═" * 58 + "╝")
            logger.debug("Task: %s", task[:200] + "..." if len(task) > 200 else task)

        for i, step in enumerate(self.workflow.steps):
            step_name = step.name or f"step_{i}"

            # Render the task template
            rendered_task = self._render_task(
                step.task_template,
                input_text,
                previous_output,
                step_outputs,
            )

            # Execute the step
            outputs = self._execute_step(step, rendered_task, model, i)
            all_outputs.append((step_name, outputs))

            # Update state for next step
            if step.pass_output:
                combined = self._combine_step_outputs(outputs)
                previous_output = combined
                step_outputs[f"step_{i}"] = combined
                for tool_name, output in outputs.items():
                    step_outputs[tool_name] = output

        # Return the final output
        result = previous_output
        if self.workflow.synthesize_prompt:
            if self._verbose:
                logger.info("╔═══ SYNTHESIS STEP ═══")

            # Optional synthesis step
            synthesis_task = self._render_task(
                self.workflow.synthesize_prompt,
                input_text,
                previous_output,
                step_outputs,
            )
            # Use the model directly for synthesis (no tool needed)
            response = model.invoke([
                SystemMessage(content=self.config.system_prompt),
                HumanMessage(content=synthesis_task),
            ])
            result = str(response.content)

            if self._verbose:
                logger.info("╚═══ SYNTHESIS COMPLETE ═══")

        workflow_duration = time.time() - workflow_start
        if self._verbose:
            logger.info("╔" + "═" * 58 + "╗")
            logger.info("║ WORKFLOW COMPLETE: %-38s ║", self.name)
            logger.info("║ Duration: %-47.2fs ║", workflow_duration)
            logger.info("╚" + "═" * 58 + "╝")

        return result

    async def ainvoke(
        self,
        task: str,
        model: BaseChatModel,
        *,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Execute the workflow asynchronously.

        Processes each step in sequence, with parallel tool execution
        within steps as defined by the workflow.

        Args:
            task: The input task for the skill.
            model: Language model to use.
            context: Optional additional context.
            config: Optional runtime configuration.

        Returns:
            Final output from the workflow.
        """
        import time
        workflow_start = time.time()

        input_text = task
        previous_output = ""
        step_outputs: dict[str, str] = {}
        all_outputs: list[tuple[str, dict[str, str]]] = []

        if self._verbose:
            logger.info("╔" + "═" * 58 + "╗")
            logger.info("║ WORKFLOW START: %-40s ║", self.name)
            logger.info("║ Steps: %-51d ║", len(self.workflow.steps))
            logger.info("╚" + "═" * 58 + "╝")
            logger.debug("Task: %s", task[:200] + "..." if len(task) > 200 else task)

        for i, step in enumerate(self.workflow.steps):
            step_name = step.name or f"step_{i}"

            # Render the task template
            rendered_task = self._render_task(
                step.task_template,
                input_text,
                previous_output,
                step_outputs,
            )

            # Execute the step
            outputs = await self._execute_step_async(step, rendered_task, model, i)
            all_outputs.append((step_name, outputs))

            # Update state for next step
            if step.pass_output:
                combined = self._combine_step_outputs(outputs)
                previous_output = combined
                step_outputs[f"step_{i}"] = combined
                for tool_name, output in outputs.items():
                    step_outputs[tool_name] = output

        # Return the final output
        result = previous_output
        if self.workflow.synthesize_prompt:
            if self._verbose:
                logger.info("╔═══ SYNTHESIS STEP ═══")

            # Optional synthesis step
            synthesis_task = self._render_task(
                self.workflow.synthesize_prompt,
                input_text,
                previous_output,
                step_outputs,
            )
            # Use the model directly for synthesis (no tool needed)
            response = await model.ainvoke([
                SystemMessage(content=self.config.system_prompt),
                HumanMessage(content=synthesis_task),
            ])
            result = str(response.content)

            if self._verbose:
                logger.info("╚═══ SYNTHESIS COMPLETE ═══")

        workflow_duration = time.time() - workflow_start
        if self._verbose:
            logger.info("╔" + "═" * 58 + "╗")
            logger.info("║ WORKFLOW COMPLETE: %-38s ║", self.name)
            logger.info("║ Duration: %-47.2fs ║", workflow_duration)
            logger.info("╚" + "═" * 58 + "╝")

        return result

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        tool_registry: dict[str, BaseTool] | None = None,
    ) -> WorkflowSkill:
        """Create a WorkflowSkill from a configuration dictionary.

        Args:
            config: Dictionary with skill config and 'workflow' key.
            tool_registry: Optional tool registry for resolving tool names.

        Returns:
            WorkflowSkill instance.

        Example config:
            ```yaml
            name: stock-analyzer
            description: Analyze stocks with multiple data sources
            system_prompt: You are a stock analysis expert.
            workflow:
              - tools: market_data
                task_template: "Get data for {{input}}"
              - tools: [technical_analysis, sentiment_analysis]
                task_template: "Analyze: {{previous_output}}"
              - tools: report_generator
                task_template: "Generate report: {{previous_output}}"
            ```
        """
        skill_config = SkillConfig.from_dict(config)

        workflow_data = config.get("workflow", [])
        workflow = WorkflowConfig.from_dict(workflow_data)

        return cls(skill_config, workflow, tool_registry)


def create_workflow_skill(
    name: str,
    description: str,
    workflow: list[dict[str, Any]],
    tool_registry: dict[str, BaseTool] | None = None,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs: Any,
) -> WorkflowSkill:
    """Convenience function to create a WorkflowSkill.

    Args:
        name: Skill name.
        description: Skill description.
        workflow: List of workflow step definitions.
        tool_registry: Tool registry for resolving tool names.
        system_prompt: System prompt for the skill.
        **kwargs: Additional SkillConfig parameters.

    Returns:
        WorkflowSkill instance.

    Example:
        ```python
        skill = create_workflow_skill(
            name="research-and-analyze",
            description="Research a topic and analyze findings",
            workflow=[
                {"tools": "web_search", "task_template": "Search: {{input}}"},
                {"tools": ["analyzer", "fact_checker"]},  # parallel
                {"tools": "summarizer", "task_template": "Summarize: {{previous_output}}"},
            ],
            tool_registry={"web_search": web_search_tool, ...},
        )
        ```
    """
    config = SkillConfig(
        name=name,
        description=description,
        system_prompt=system_prompt,
        **kwargs,
    )
    workflow_config = WorkflowConfig.from_dict(workflow)
    return WorkflowSkill(config, workflow_config, tool_registry)


__all__ = [
    "Skill",
    "SkillConfig",
    "SkillState",
    "SimpleSkill",
    "WorkflowStep",
    "WorkflowConfig",
    "WorkflowSkill",
    "create_skill_from_config",
    "create_workflow_skill",
]
