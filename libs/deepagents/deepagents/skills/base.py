"""Base classes for skills architecture.

This module provides the foundational classes for a middleware-free skills system
that works with langchain>=0.2.0 and langgraph>=0.1.0 (pre-middleware versions).

Skills are isolated subagents with custom tools and system prompts that can be
invoked by a supervisor agent.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
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


__all__ = [
    "Skill",
    "SkillConfig",
    "SkillState",
    "SimpleSkill",
    "create_skill_from_config",
]
